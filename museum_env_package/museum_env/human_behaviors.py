"""Mode-specific human behavior implementations.

This module keeps the high-churn step logic out of ``human.py`` so the
``Human`` class can stay focused on state, transitions, and shared motion
utilities.
"""

import numpy as np

from .human import (
    DISTRACTED_EXHIBIT_LOOK_RADIUS,
    DISTRACTED_FALLBACK_DISTANCE,
    DISTRACTED_HUMAN_LOOK_RADIUS,
    DISTRACTED_SOURCE_LISTENING,
    DISTRACTED_SPEED_SCALE,
    DISTRACTED_YAW_DEVIATION_MAX_DEG,
    DISTRACTED_YAW_DEVIATION_MIN_DEG,
    HUMAN_ROTATION_STOP_DEG,
    HUMAN_YAW_RATE_GAIN,
    LISTENING_IMPATIENT_SWAY_SPEED_METERS_PER_SEC,
    LISTENING_IMPATIENT_TARGET_REACHED_DEG,
    LISTENING_RING_GAIN,
    LISTENING_SECTOR_PROJECTION_EPS,
    NORM_EPS,
    OVERWHELMED_STAGE_SWITCH_DIST,
    HumanMode,
    logger,
)


def step_behavior(human, ctx, pose):
    """Dispatch behavior based on explicit variants first, then on mode."""
    behavior_kind = ctx.get("behavior_kind")

    # These variants intentionally override the default mode behavior. They are
    # used for post-explanation coordination where mode alone is not specific
    # enough to describe the intended motion policy.
    if behavior_kind == "post_explanation_yield":
        return _step_post_explanation_yield(human, ctx, pose)
    if behavior_kind == "post_explanation_listening_anchor":
        return _step_post_explanation_listening_anchor(human, ctx, pose)
    if behavior_kind is not None:
        raise ValueError(f"Unknown human behavior kind {behavior_kind!r}")

    if human.mode == HumanMode.WANDERING:
        return _step_wandering(human, ctx, pose)
    if human.mode == HumanMode.FOLLOWING:
        human.assign_target_from_context(ctx)
        return _step_following(human, ctx, pose)
    if human.mode == HumanMode.LISTENING:
        return _step_listening(human, ctx, pose)
    if human.mode == HumanMode.DISTRACTED:
        return _step_distracted(human, ctx, pose)
    if human.mode == HumanMode.OVERWHELMED:
        return _step_overwhelmed(human, ctx, pose)
    if human.mode == HumanMode.IMPATIENT:
        return _step_impatient(human, ctx, pose)
    raise ValueError(f"Unknown human mode {human.mode}")


def _step_wandering(human, ctx, pose):
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    yaw = pose[2]
    to_waypoint = human.current_waypoint - current_xy
    if np.linalg.norm(to_waypoint) < human.waypoint_threshold:
        human.current_waypoint = human._random_waypoint()
        to_waypoint = human.current_waypoint - current_xy
    return human._move(to_waypoint, yaw, ctx, current_xy)


def _step_following(human, ctx, pose):
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    return human._move(human.current_waypoint - current_xy, pose[2], ctx, current_xy)


def _step_listening(human, ctx, pose):
    yaw = pose[2]
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
    robot_yaw = ctx["robot_yaw"]
    to_robot = robot_xy - current_xy
    dist_to_robot = np.linalg.norm(to_robot)
    desired_yaw = np.arctan2(to_robot[1], to_robot[0]) if dist_to_robot > NORM_EPS else robot_yaw
    yaw_err = human._wrap_to_pi(desired_yaw - yaw)

    target_xy = _compute_listening_sector_target_point(
        human,
        current_xy=current_xy,
        robot_xy=robot_xy,
        robot_yaw=robot_yaw,
        listen_radius=ctx["listen_radius"],
        sector_half_angle=ctx["listening_sector_half_angle"],
    )
    v_goal = LISTENING_RING_GAIN * (target_xy - current_xy)
    v_total = v_goal + np.asarray(ctx["repulsion"], dtype=np.float32)
    v_total += human._compute_hr_spacing_force(
        current_xy=current_xy,
        robot_xy=robot_xy,
        distance_min=human.hr_distance_min,
        distance_max=None,
    )
    speed = np.linalg.norm(v_total)
    if speed > human.max_speed and speed > NORM_EPS:
        v_total = v_total / speed * human.max_speed

    action = human._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
    return human._apply_wall_constraint_to_action(action, current_xy)


def _step_post_explanation_listening_anchor(human, ctx, pose):
    """Listen around a frozen anchor pose while still reacting to live repulsion."""
    yaw = pose[2]
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    anchor_robot_xy = np.asarray(ctx["anchor_robot_xy"], dtype=np.float32)
    live_robot_xy = np.asarray(ctx["live_robot_xy"], dtype=np.float32)
    anchor_robot_yaw = float(ctx["anchor_robot_yaw"])

    to_anchor_robot = anchor_robot_xy - current_xy
    dist_to_anchor_robot = np.linalg.norm(to_anchor_robot)
    desired_yaw = (
        np.arctan2(to_anchor_robot[1], to_anchor_robot[0])
        if dist_to_anchor_robot > NORM_EPS
        else anchor_robot_yaw
    )
    yaw_err = human._wrap_to_pi(desired_yaw - yaw)

    target_xy = _compute_listening_sector_target_point(
        human,
        current_xy=current_xy,
        robot_xy=anchor_robot_xy,
        robot_yaw=anchor_robot_yaw,
        listen_radius=ctx["listen_radius"],
        sector_half_angle=ctx["listening_sector_half_angle"],
    )
    v_goal = LISTENING_RING_GAIN * (target_xy - current_xy)
    v_total = v_goal + np.asarray(ctx["repulsion"], dtype=np.float32)
    v_total += human._compute_hr_spacing_force(
        current_xy=current_xy,
        robot_xy=live_robot_xy,
        distance_min=human.hr_distance_min,
        distance_max=None,
    )
    speed = np.linalg.norm(v_total)
    if speed > human.max_speed and speed > NORM_EPS:
        v_total = v_total / speed * human.max_speed

    action = human._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
    return human._apply_wall_constraint_to_action(action, current_xy)


def _step_distracted(human, ctx, pose):
    if human.distracted_source == DISTRACTED_SOURCE_LISTENING:
        return _step_listening_distracted(human, ctx, pose)

    yaw = pose[2]
    human.distracted_timer += 1
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    if human.distracted_target_xy is None:
        _initialize_following_distracted_target(human, ctx, current_xy=current_xy, current_yaw=yaw)

    # A following-distracted human still aims for the normal follow slot, but
    # moves more slowly so they naturally fall behind the rest of the group.
    human.assign_target_from_context(ctx, mode=HumanMode.FOLLOWING)
    follow_target_xy = np.asarray(human.current_waypoint, dtype=np.float32)
    to_follow_target = follow_target_xy - current_xy
    dist_to_follow_target = np.linalg.norm(to_follow_target)

    move_speed_limit = DISTRACTED_SPEED_SCALE * human.max_speed
    if dist_to_follow_target > NORM_EPS:
        v_goal = move_speed_limit * (to_follow_target / dist_to_follow_target)
    else:
        v_goal = np.zeros(2, dtype=np.float32)

    robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
    v_total = v_goal + np.asarray(ctx["repulsion"], dtype=np.float32)
    v_total += human._compute_hr_spacing_force(
        current_xy=current_xy,
        robot_xy=robot_xy,
        distance_min=human.hr_distance_min,
        distance_max=human.hr_distance_max,
    )
    speed = np.linalg.norm(v_total)
    if speed > move_speed_limit and speed > NORM_EPS:
        v_total = v_total / speed * move_speed_limit

    # Translation keeps serving the group-following objective, while yaw keeps
    # tracking the distraction target as a proxy for both gaze and body turn.
    desired_yaw = _resolve_focus_yaw(human, current_xy, fallback_yaw=yaw)
    yaw_err = human._wrap_to_pi(desired_yaw - yaw)
    action = human._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)

    if human.distracted_timer >= human.distracted_duration:
        human.set_mode(human.distracted_recovery_mode)
        if human.enable_event_logs:
            logger.info(f">>> {human.name} recovered -> {human.mode.upper()}")

    return human._apply_wall_constraint_to_action(action, current_xy)


def _step_listening_distracted(human, ctx, pose):
    human.distracted_timer += 1
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    desired_yaw = human.distracted_target_yaw
    if desired_yaw is None:
        _initialize_listening_distracted_target(
            human,
            current_xy=current_xy,
            robot_xy=np.asarray(ctx["robot_xy"], dtype=np.float32),
        )
        desired_yaw = human.distracted_target_yaw
    yaw_err = human._wrap_to_pi(desired_yaw - pose[2])

    if abs(yaw_err) >= np.deg2rad(HUMAN_ROTATION_STOP_DEG):
        action = human._compose_action(np.zeros(2, dtype=np.float32), HUMAN_YAW_RATE_GAIN * yaw_err)
        return human._apply_wall_constraint_to_action(action, current_xy)
    return np.zeros(3, dtype=np.float32)


def _step_overwhelmed(human, ctx, pose):
    pos_xy = np.asarray(pose[:2], dtype=np.float32)
    leave_dir = np.asarray(human.overwhelmed_leave_dir, dtype=np.float32)
    leave_dir = leave_dir / np.linalg.norm(leave_dir)
    desired_yaw = np.arctan2(leave_dir[1], leave_dir[0])

    if human.overwhelmed_stage == "backoff":
        backoff_target = human.overwhelmed_backoff_start_xy + human.overwhelmed_backoff_dist * leave_dir
        to_target = backoff_target - pos_xy
        dist_to_target = np.linalg.norm(to_target)
        if dist_to_target < OVERWHELMED_STAGE_SWITCH_DIST:
            human.overwhelmed_stage = "leave"
            to_target = np.zeros(2, dtype=np.float32)
            dist_to_target = 0.0

        if dist_to_target > NORM_EPS:
            backoff_speed = min(human.overwhelmed_leave_speed, human.max_speed)
            v_xy = backoff_speed * (to_target / dist_to_target)
        else:
            v_xy = np.zeros(2, dtype=np.float32)

        action = human._compose_action(v_xy, HUMAN_YAW_RATE_GAIN * human._wrap_to_pi(desired_yaw - pose[2]))
        return human._apply_wall_constraint_to_action(action, pos_xy)

    human.overwhelmed_leave_timer += 1
    v_xy = min(human.overwhelmed_leave_speed, human.max_speed) * leave_dir
    if human.overwhelmed_leave_timer >= human.overwhelmed_leave_duration:
        human.set_mode(human.overwhelmed_recovery_mode)
        if human.enable_event_logs:
            logger.info(f">>> {human.name} recovered from OVERWHELMED -> {human.mode.upper()}")

    action = human._compose_action(v_xy, HUMAN_YAW_RATE_GAIN * human._wrap_to_pi(desired_yaw - pose[2]))
    return human._apply_wall_constraint_to_action(action, pos_xy)


def _step_impatient(human, ctx, pose):
    human.impatient_timer += 1
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    yaw = pose[2]
    if human.impatient_recovery_mode == HumanMode.LISTENING:
        robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
        to_robot = robot_xy - current_xy
        base_yaw = np.arctan2(to_robot[1], to_robot[0]) if np.linalg.norm(to_robot) > NORM_EPS else yaw
        desired_yaw = human._wrap_to_pi(
            base_yaw + human.listening_impatient_turn_sign * human.listening_impatient_yaw_deviation
        )
        yaw_err = human._wrap_to_pi(desired_yaw - yaw)
        if abs(yaw_err) <= np.deg2rad(LISTENING_IMPATIENT_TARGET_REACHED_DEG):
            human.listening_impatient_turn_sign *= -1.0
            desired_yaw = human._wrap_to_pi(
                base_yaw + human.listening_impatient_turn_sign * human.listening_impatient_yaw_deviation
            )
            yaw_err = human._wrap_to_pi(desired_yaw - yaw)

        perp = np.array([-np.sin(base_yaw), np.cos(base_yaw)], dtype=np.float32)
        v_total = (
            human.listening_impatient_turn_sign
            * LISTENING_IMPATIENT_SWAY_SPEED_METERS_PER_SEC
            * perp
        )
        v_total += np.asarray(ctx["repulsion"], dtype=np.float32)
        speed = np.linalg.norm(v_total)
        if speed > human.max_speed and speed > NORM_EPS:
            v_total = v_total / speed * human.max_speed
        action = human._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
        action = human._apply_wall_constraint_to_action(action, current_xy)
    else:
        human.assign_target_from_context(ctx)
        action = _step_following(human, ctx, pose)

    if human.impatient_timer >= human.impatient_duration:
        human.set_mode(human.impatient_recovery_mode)
        if human.enable_event_logs:
            logger.info(f">>> {human.name} recovered from IMPATIENT -> {human.mode.upper()}")
    return action


def _step_post_explanation_yield(human, ctx, pose):
    """Move toward a temporary yield target without exposing a custom public step."""
    target_xy = np.asarray(ctx["target_xy"], dtype=np.float32)
    human.current_waypoint = target_xy.copy()
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    return human._move(target_xy - current_xy, pose[2], ctx, current_xy)


def _resolve_focus_yaw(human, current_xy, *, fallback_yaw: float) -> float:
    focus_target_xy = None
    if human.distracted_target_xy is not None:
        focus_target_xy = np.asarray(human.distracted_target_xy, dtype=np.float32)

    if focus_target_xy is not None:
        focus_delta = focus_target_xy - current_xy
        if np.linalg.norm(focus_delta) > NORM_EPS:
            return float(np.arctan2(focus_delta[1], focus_delta[0]))
    if human.distracted_target_yaw is not None:
        return float(human.distracted_target_yaw)
    return float(fallback_yaw)


def _select_nearest_candidate(current_xy, candidates, max_distance: float, *, exclude_index=None):
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.size == 0:
        return None
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)

    deltas = candidates[:, :2] - np.asarray(current_xy, dtype=np.float32)[None, :]
    distances = np.linalg.norm(deltas, axis=1)
    if exclude_index is not None and 0 <= int(exclude_index) < len(distances):
        distances[int(exclude_index)] = np.inf

    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    if (not np.isfinite(best_distance)) or best_distance > float(max_distance):
        return None
    return np.asarray(candidates[best_idx, :2], dtype=np.float32)


def _get_distractor_exhibit_points(human):
    exhibit_points = human.map_layout.metadata.get("distractor_exhibit_points", ())
    points = np.asarray(exhibit_points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return points.reshape(-1, 2)


def _initialize_following_distracted_target(human, ctx, current_xy, current_yaw: float):
    # The focus target is chosen once per distracted episode so the human keeps
    # attending to one thing instead of jittering between multiple candidates.
    current_xy = np.asarray(current_xy, dtype=np.float32)

    exhibit_target_xy = _select_nearest_candidate(
        current_xy,
        _get_distractor_exhibit_points(human),
        DISTRACTED_EXHIBIT_LOOK_RADIUS,
    )
    # If there's a nearby exhibit, look at it.
    if exhibit_target_xy is not None:
        target_yaw = np.arctan2(
            exhibit_target_xy[1] - current_xy[1],
            exhibit_target_xy[0] - current_xy[0],
        )
        human._set_distracted_target_state(target_yaw=target_yaw, target_xy=exhibit_target_xy)
        return
    
    human_target_xy = _select_nearest_candidate(
        current_xy,
        ctx.get("human_xy", np.zeros((0, 2), dtype=np.float32)),
        DISTRACTED_HUMAN_LOOK_RADIUS,
        exclude_index=ctx.get("index"),
    )
    # If there's a nearby human, look at them
    if human_target_xy is not None:
        target_yaw = np.arctan2(
            human_target_xy[1] - current_xy[1],
            human_target_xy[0] - current_xy[0],
        )
        human._set_distracted_target_state(target_yaw=target_yaw, target_xy=human_target_xy)
        return
    # Otherwise, pick a random direction
    target_yaw, sampled_target_xy = _sample_distracted_target_candidate(current_xy, current_yaw)
    human._set_distracted_target_state(target_yaw=target_yaw, target_xy=sampled_target_xy)


def _sample_distracted_target_candidate(current_xy, current_yaw: float):
    deviation_deg = np.random.uniform(
        DISTRACTED_YAW_DEVIATION_MIN_DEG,
        DISTRACTED_YAW_DEVIATION_MAX_DEG,
    )
    deviation_sign = -1.0 if np.random.rand() < 0.5 else 1.0
    deviation_rad = np.deg2rad(deviation_deg) * deviation_sign
    target_yaw = _wrap_to_pi(current_yaw + deviation_rad)
    direction_xy = np.array([np.cos(target_yaw), np.sin(target_yaw)], dtype=np.float32)
    return float(target_yaw), np.asarray(
        current_xy + DISTRACTED_FALLBACK_DISTANCE * direction_xy,
        dtype=np.float32,
    )


def _initialize_listening_distracted_target(human, current_xy, robot_xy):
    current_xy = np.asarray(current_xy, dtype=np.float32)
    robot_xy = np.asarray(robot_xy, dtype=np.float32)
    deviation_deg = np.random.uniform(45.0, 90.0)
    deviation_sign = -1.0 if np.random.rand() < 0.5 else 1.0
    robot_facing_yaw = np.arctan2(robot_xy[1] - current_xy[1], robot_xy[0] - current_xy[0])
    target_yaw = _wrap_to_pi(robot_facing_yaw + deviation_sign * np.deg2rad(deviation_deg))
    human._set_distracted_target_state(target_yaw=target_yaw, target_xy=current_xy)


def _compute_listening_sector_target_point(
    human,
    *,
    current_xy,
    robot_xy,
    robot_yaw: float,
    listen_radius: float,
    sector_half_angle: float,
):
    current_xy = np.asarray(current_xy, dtype=np.float32)
    robot_xy = np.asarray(robot_xy, dtype=np.float32)
    rel_xy = current_xy - robot_xy
    if np.dot(rel_xy, rel_xy) <= (NORM_EPS * NORM_EPS):
        absolute_angle = robot_yaw
    else:
        absolute_angle = np.arctan2(rel_xy[1], rel_xy[0])
    relative_angle = human._wrap_to_pi(absolute_angle - robot_yaw)
    half_angle = max(0.0, sector_half_angle - LISTENING_SECTOR_PROJECTION_EPS)
    clamped_angle = human._wrap_to_pi(robot_yaw + np.clip(relative_angle, -half_angle, half_angle))
    return np.array(
        [
            robot_xy[0] + listen_radius * np.cos(clamped_angle),
            robot_xy[1] + listen_radius * np.sin(clamped_angle),
        ],
        dtype=np.float32,
    )


def _wrap_to_pi(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi
