from __future__ import annotations

from typing import Sequence

import numpy as np

from .env_state import ObservationSnapshot, PostExplanationState, RuntimeCache, WorldFrame
from .human import HumanMode

DIST_EPS = 1e-8
LOCAL_CROWDING_RADIUS_METERS = 1.0


def empty_observation_snapshot(n_humans: int) -> ObservationSnapshot:
    return ObservationSnapshot(
        nearest_human_distance=np.full((n_humans,), np.nan, dtype=np.float32),
        local_crowding_count_1m=np.zeros((n_humans,), dtype=np.int32),
        human_robot_distance=np.zeros((n_humans,), dtype=np.float32),
        nearest_human_distance_mean_1s=np.full((n_humans,), np.nan, dtype=np.float32),
        human_robot_distance_mean_1s=np.zeros((n_humans,), dtype=np.float32),
    )


def collect_robot_pose(data, robot_body_id: int) -> tuple[float, float, float]:
    x = float(data.xpos[robot_body_id, 0])
    y = float(data.xpos[robot_body_id, 1])
    yaw = float(data.qpos[2])
    return x, y, yaw


def collect_human_poses(data, humans: Sequence, human_body_ids: Sequence[int]) -> np.ndarray:
    n_humans = len(humans)
    human_xyz = np.empty((n_humans, 3), dtype=np.float32)
    for idx, (human, human_body_id) in enumerate(zip(humans, human_body_ids)):
        human_xyz[idx, 0] = float(data.xpos[human_body_id, 0])
        human_xyz[idx, 1] = float(data.xpos[human_body_id, 1])
        human_xyz[idx, 2] = float(data.qpos[human.qpos_idx + 2])
    return human_xyz


def compute_human_pairwise_distances(human_xy) -> np.ndarray:
    human_xy = np.asarray(human_xy, dtype=np.float32)
    n_humans = int(human_xy.shape[0])
    if n_humans == 0:
        return np.zeros((0, 0), dtype=np.float32)
    pairwise_diff = human_xy[:, None, :] - human_xy[None, :, :]
    pairwise_dist = np.linalg.norm(pairwise_diff, axis=2).astype(np.float32)
    np.fill_diagonal(pairwise_dist, np.inf)
    return pairwise_dist


def compute_nearest_human_distances_from_pairwise(pairwise_dist) -> np.ndarray:
    pairwise_dist = np.asarray(pairwise_dist, dtype=np.float32)
    if pairwise_dist.shape == (0, 0):
        return np.zeros((0,), dtype=np.float32)
    nearest = np.min(pairwise_dist, axis=1).astype(np.float32)
    nearest[~np.isfinite(nearest)] = np.nan
    return nearest


def compute_local_crowding_count_1m_from_pairwise(pairwise_dist) -> np.ndarray:
    # Get the minimum distance to each human, then count
    pairwise_dist = np.asarray(pairwise_dist, dtype=np.float32)
    if pairwise_dist.shape == (0, 0):
        return np.zeros((0,), dtype=np.int32)
    return np.count_nonzero(pairwise_dist < LOCAL_CROWDING_RADIUS_METERS, axis=1).astype(
        np.int32
    )


def compute_human_robot_distances(human_xy, robot_xy) -> np.ndarray:
    human_xy = np.asarray(human_xy, dtype=np.float32)
    if human_xy.size == 0:
        return np.zeros((0,), dtype=np.float32)
    robot_xy = np.asarray(robot_xy, dtype=np.float32)
    return np.linalg.norm(human_xy - robot_xy[None, :], axis=1).astype(np.float32)


def compute_social_repulsion(human_xy, social_distance: float, repulsion_gain: float) -> np.ndarray:
    human_xy = np.asarray(human_xy, dtype=np.float32)
    n_humans = int(human_xy.shape[0])
    if n_humans == 0 or social_distance <= 1e-6:
        return np.zeros((n_humans, 2), dtype=np.float32)

    pairwise_dist = compute_human_pairwise_distances(human_xy)
    mask = (pairwise_dist > 1e-6) & (pairwise_dist < social_distance)
    repulsion_vectors = np.zeros((n_humans, 2), dtype=np.float32)
    for idx in range(n_humans):
        neighbors = mask[idx]
        if not np.any(neighbors):
            continue

        diff = human_xy[idx] - human_xy[neighbors]
        dist = pairwise_dist[idx, neighbors]
        directions = diff / dist[:, None]
        # Distance-based repulsion strength
        strengths = (social_distance - dist) / social_distance
        repulsion_vectors[idx] = repulsion_gain * np.sum(
            directions * strengths[:, None],
            axis=0,
        )

    return repulsion_vectors


def tick_observation_age(cache: RuntimeCache) -> None:
    cache.sample_age_steps += 1


def refresh_observation_snapshot(
    *,
    cache: RuntimeCache,
    hh_distance_metric,
    hr_distance_metric,
    human_xy,
    robot_xy,
    observation_update_period_steps: int,
    force: bool = False,
) -> ObservationSnapshot:
    n_humans = int(np.asarray(human_xy, dtype=np.float32).shape[0])
    should_refresh = force or cache.observations is None
    if (not should_refresh) and cache.sample_age_steps >= int(observation_update_period_steps):
        should_refresh = True

    if not should_refresh:
        return cache.observations

    # Compute all derived observations and update the cache if timeout
    pairwise_dist = compute_human_pairwise_distances(human_xy)
    nearest_human_distance = compute_nearest_human_distances_from_pairwise(pairwise_dist)
    local_crowding_count_1m = compute_local_crowding_count_1m_from_pairwise(pairwise_dist)
    human_robot_distance = compute_human_robot_distances(human_xy, robot_xy)
    nearest_human_distance_mean_1s = hh_distance_metric.update(nearest_human_distance)
    human_robot_distance_mean_1s = hr_distance_metric.update(human_robot_distance)

    cache.observations = ObservationSnapshot(
        nearest_human_distance=np.array(nearest_human_distance, dtype=np.float32),
        local_crowding_count_1m=np.array(local_crowding_count_1m, dtype=np.int32),
        human_robot_distance=np.array(human_robot_distance, dtype=np.float32),
        nearest_human_distance_mean_1s=np.array(
            nearest_human_distance_mean_1s,
            dtype=np.float32,
        ),
        human_robot_distance_mean_1s=np.array(
            human_robot_distance_mean_1s,
            dtype=np.float32,
        ),
    )
    cache.refresh_counter += 1
    cache.sample_age_steps = 0
    return cache.observations


def build_world_frame(
    *,
    data,
    robot_body_id: int,
    humans: Sequence,
    human_body_ids: Sequence[int],
    cache: RuntimeCache,
    hh_distance_metric,
    hr_distance_metric,
    observation_update_period_steps: int,
    social_distance: float,
    repulsion_gain: float,
    force_observations: bool = False,
    tick_age_before_refresh: bool = False,
) -> WorldFrame:
    robot_pose = collect_robot_pose(data, robot_body_id)
    robot_xy = np.array(robot_pose[:2], dtype=np.float32)
    human_xyz = collect_human_poses(data, humans, human_body_ids)
    human_xy = human_xyz[:, :2]
    human_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

    if tick_age_before_refresh:
        tick_observation_age(cache)

    # Refresh observations if needed
    observations = refresh_observation_snapshot(
        cache=cache,
        hh_distance_metric=hh_distance_metric,
        hr_distance_metric=hr_distance_metric,
        human_xy=human_xy,
        robot_xy=robot_xy,
        observation_update_period_steps=observation_update_period_steps,
        force=force_observations,
    )
    pairwise_distances = compute_human_pairwise_distances(human_xy)
    repulsion_vectors = compute_social_repulsion(human_xy, social_distance, repulsion_gain)
    return WorldFrame(
        robot_pose=robot_pose,
        robot_xy=robot_xy,
        human_xyz=human_xyz,
        human_xy=human_xy,
        human_yaw=human_yaw,
        pairwise_distances=pairwise_distances,
        repulsion_vectors=repulsion_vectors,
        observations=observations,
    )


def build_human_goals(humans: Sequence, robot_xy, post_explanation_state: PostExplanationState) -> np.ndarray:
    n_humans = len(humans)
    if n_humans == 0:
        return np.zeros((0, 2), dtype=np.float32)

    goals = np.empty((n_humans, 2), dtype=np.float32)
    robot_xy = np.asarray(robot_xy, dtype=np.float32)
    for idx, human in enumerate(humans):
        if post_explanation_state.active and idx < len(post_explanation_state.targets):
            goals[idx] = np.asarray(post_explanation_state.targets[idx], dtype=np.float32)
        elif human.mode == HumanMode.LISTENING:
            goals[idx] = robot_xy
        else:
            goals[idx] = np.asarray(human.current_waypoint, dtype=np.float32)
    return goals


def compute_reached_goal_indices(
    *,
    humans: Sequence,
    human_xy,
    human_goals,
    post_explanation_state: PostExplanationState,
    robot_xy,
    robot_yaw: float,
    listen_reached_min_distance: float,
    human_goal_threshold: float,
    listening_sector_half_angle: float,
) -> list[int]:
    reached_goal_indices: list[int] = []
    robot_xy = np.asarray(robot_xy, dtype=np.float32)
    human_xy = np.asarray(human_xy, dtype=np.float32)
    human_goals = np.asarray(human_goals, dtype=np.float32)

    for idx, (human, pos_xy, goal_xy) in enumerate(zip(humans, human_xy, human_goals)):
        if post_explanation_state.active and idx < len(post_explanation_state.targets):
            reached = float(np.linalg.norm(pos_xy - goal_xy)) < float(human_goal_threshold)
        elif human.mode == HumanMode.LISTENING:
            dist_to_robot = float(np.linalg.norm(pos_xy - robot_xy))
            in_sector = human.is_within_listening_front_sector(
                point_xy=pos_xy,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                sector_half_angle=listening_sector_half_angle,
            )
            reached = dist_to_robot > float(listen_reached_min_distance) and in_sector
        else:
            reached = float(np.linalg.norm(pos_xy - goal_xy)) < float(human_goal_threshold)

        if reached:
            reached_goal_indices.append(idx)

    return reached_goal_indices


def resolve_fuzzy_metric_input(rolling_mean_value: float, current_value: float) -> float:
    if np.isfinite(rolling_mean_value):
        return float(rolling_mean_value)
    if np.isfinite(current_value):
        return float(current_value)
    return 0.0
