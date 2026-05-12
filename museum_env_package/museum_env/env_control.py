"""Environment-side control helpers for robot and crowd orchestration."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .env_constants import (
    FOLLOWING_CALLBACK_DISTANCE_THRESHOLD_METERS,
    FOLLOWING_SLOWDOWN_DISTANCE_THRESHOLD_METERS,
    FOLLOWING_SLOWDOWN_SPEED_SCALE,
    FOLLOW_RADIUS_DEFAULT,
    HUMAN_WALL_FOOTPRINT_RADIUS,
    LISTENING_REPULSION_SCALE,
    ROBOT_FOOTPRINT_RADIUS_METERS,
    ROBOT_PERSONAL_SPACE_BACKOFF_SPEED_METERS,
    ROBOT_PERSONAL_SPACE_MARGIN_METERS,
    ROBOT_PERSONAL_SPACE_PROBE_DISTANCE_METERS,
    ROBOT_PERSONAL_SPACE_TRIGGER_METERS,
)
from .env_runtime import resolve_fuzzy_metric_input
from .env_state import (
    FOLLOW_PHASE_TRANSIT,
    LISTEN_PHASE_INTRO,
    LISTEN_PHASE_PAUSED,
    LISTEN_PHASE_WAIT,
    POST_EXPLANATION_ROLE_WAIT,
    POST_EXPLANATION_ROLE_YIELD,
)
from .human import (
    DISTRACTED_SOURCE_FOLLOWING,
    DISTRACTED_SOURCE_LISTENING,
    HumanMode,
)
from .robot import RobotMode
from .spatial_utils import raycast_hit_distance, wrap_to_pi


def reset_following_wait_episode(env) -> None:
    """Clear the temporary waiting/callback state used during follow transit."""
    env._following_wait_elapsed_steps = 0
    env._following_wait_callback_triggered = False
    env._following_callback_override_target_idx = None


def update_human_listening_session_progress(env) -> None:
    """Advance listening-session timers only while the listening controller is active."""
    if not env.listening_state.fuzzy_active:
        return
    for human in env.humans:
        human.update_listening_session_progress(active=(human.mode == HumanMode.LISTENING))


def _current_human_modes(env) -> list[str]:
    """Return the current mode snapshot for all humans."""
    return [human.mode for human in env.humans]


def _repulsion_vec(world_frame, idx: int) -> np.ndarray:
    """Return one human's cached repulsion vector as a float32 array."""
    return np.asarray(world_frame.repulsion_vectors[idx], dtype=np.float32)


def should_evaluate_fuzzy(env, idx: int, context: str) -> bool:
    """Decide whether fuzzy state should be recomputed for this human and phase."""
    # Fuzzy transitions are only meaningful in the phase that owns the behavior:
    # follow-time stress is evaluated during transit, and listening stress only
    # while the listening controller is actively running.
    if context == "following" and env.follow_phase != FOLLOW_PHASE_TRANSIT:
        return False
    if context == "listening" and not env.listening_state.fuzzy_active:
        return False
    debug_state = env.fuzzy_debug[idx]
    # Recompute immediately when the context changes or there is no cached
    # dominant state yet; otherwise wait until the observation refresh ticks.
    if debug_state.dominant_state is None or debug_state.context != context:
        return True
    return env.runtime_cache.refresh_counter > debug_state.refresh_counter


def compute_human_fuzzy_debug(env, idx: int, context: str, session_steps: int, observations):
    """Build fuzzy inputs and evaluate the human state classifier."""
    human = env.humans[idx]
    inputs = env.following_fuzzy_engine.clip_inputs(
        following_time=float(session_steps) * float(env.dt),
        hhd=resolve_fuzzy_metric_input(
            rolling_mean_value=float(observations.nearest_human_distance_mean_1s[idx]),
            current_value=float(observations.nearest_human_distance[idx]),
        ),
        hrd=resolve_fuzzy_metric_input(
            rolling_mean_value=float(observations.human_robot_distance_mean_1s[idx]),
            current_value=float(observations.human_robot_distance[idx]),
        ),
        density=float(observations.local_crowding_count_1m[idx]),
    )
    result = env.following_fuzzy_engine.compute(
        **inputs,
        context=context,
        profile=human.profile,
    )
    return {"inputs": inputs, "result": result}


def record_fuzzy_debug(env, idx: int, context: str, fuzzy_debug: dict) -> None:
    """Persist the latest fuzzy inputs, scores, and dominant state for debugging."""
    debug_state = env.fuzzy_debug[idx]
    debug_state.context = str(context)
    debug_state.inputs = dict(fuzzy_debug["inputs"])
    debug_state.scores = {
        state: float(fuzzy_debug["result"][state])
        for state in ("overwhelmed", "distracted", "impatient", "engaged")
    }
    debug_state.dominant_state = str(fuzzy_debug["result"]["dominant_state"])
    debug_state.refresh_counter = int(env.runtime_cache.refresh_counter)


def apply_fuzzy_transition(
    env,
    human,
    idx: int,
    context: str,
    fuzzy_result: dict,
    fuzzy_inputs: dict,
    world_frame,
) -> None:
    """Apply the fuzzy engine's dominant-state transition to one human."""
    dominant_state = fuzzy_result["dominant_state"]
    if dominant_state == "engaged":
        return

    if context == "listening":
        recovery_mode = HumanMode.LISTENING
        distracted_source = DISTRACTED_SOURCE_LISTENING
    else:
        recovery_mode = HumanMode.FOLLOWING
        distracted_source = DISTRACTED_SOURCE_FOLLOWING
    fuzzy_inputs_log = (
        f"\n    >>> fuzzy_inputs: "
        f"following_time={float(fuzzy_inputs['following_time']):.1f}, "
        f"hhd={float(fuzzy_inputs['hhd']):.2f}, "
        f"hrd={float(fuzzy_inputs['hrd']):.2f}, "
        f"density={float(fuzzy_inputs['density']):.1f}"
    )

    if dominant_state == "distracted":
        human.distracted_source = distracted_source
        human.distracted_recovery_mode = recovery_mode
        human.set_mode(HumanMode.DISTRACTED)
        msg = f">>> {human.name} became DISTRACTED!"
        if context == "listening":
            msg = f">>> {human.name} became DISTRACTED while listening!"
        env._log_event(f"{msg}{fuzzy_inputs_log}")
        return

    if dominant_state == "impatient":
        human.start_impatient(recovery_mode=recovery_mode)
        msg = f">>> {human.name} became IMPATIENT!"
        if recovery_mode == HumanMode.LISTENING:
            msg = f">>> {human.name} became IMPATIENT while listening!"
        env._log_event(f"{msg}{fuzzy_inputs_log}")
        return

    if dominant_state == "overwhelmed":
        current_xy = np.array(human.get_pose(env.data)[:2], dtype=np.float32)
        human.start_overwhelmed(
            robot_xy=np.array(world_frame.robot_xy, dtype=np.float32),
            current_xy=current_xy,
            recovery_mode=recovery_mode,
        )
        msg = f">>> {human.name} became OVERWHELMED!"
        if context == "listening":
            msg = f">>> {human.name} became OVERWHELMED while listening!"
        env._log_event(f"{msg}{fuzzy_inputs_log}")


def _maybe_apply_fuzzy(env, human, idx: int, context: str, session_steps: int, world_frame) -> None:
    """Run fuzzy evaluation only when the human is in the expected base mode."""
    if human.mode != (HumanMode.FOLLOWING if context == "following" else HumanMode.LISTENING):
        return
    if not should_evaluate_fuzzy(env, idx, context=context):
        return
    fuzzy_debug = compute_human_fuzzy_debug(
        env,
        idx=idx,
        context=context,
        session_steps=int(session_steps),
        observations=world_frame.observations,
    )
    record_fuzzy_debug(env, idx, context=context, fuzzy_debug=fuzzy_debug)
    apply_fuzzy_transition(
        env,
        human,
        idx=idx,
        context=context,
        fuzzy_result=fuzzy_debug["result"],
        fuzzy_inputs=fuzzy_debug["inputs"],
        world_frame=world_frame,
    )


def _build_move_ctx(env, human, idx: int, world_frame, repulsion_vec, human_modes, impatient_recovery_mode):
    """Assemble the shared movement context passed into one human controller."""
    return {
        "index": idx,
        "n_humans": len(env.humans),
        "robot_pose": world_frame.robot_pose,
        "robot_xy": world_frame.robot_xy,
        "human_xy": world_frame.human_xy,
        "human_modes": human_modes,
        "repulsion": repulsion_vec,
        "fan_half_angle": env.follow_fan_half_angle,
        "impatient_fan_half_angle": env.impatient_fan_half_angle,
        "impatient_front_offset": human.impatient_front_offset,
        "impatient_recovery_mode": impatient_recovery_mode,
    }


def apply_general_phase_strategy(env, human, idx: int, world_frame) -> np.ndarray:
    """Compute one human action during the generic following or wandering phase."""
    repulsion_vec = _repulsion_vec(world_frame, idx)
    if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
        human.set_mode(HumanMode.FOLLOWING if env.follow_phase is not None else HumanMode.WANDERING)

    human.update_following_duration(
        eligible_following=env.follow_phase is not None and human.mode == HumanMode.FOLLOWING
    )
    _maybe_apply_fuzzy(
        env,
        human,
        idx=idx,
        context="following",
        session_steps=human.following_steps,
        world_frame=world_frame,
    )
    # The movement context is rebuilt from the live frame so each phase can pass
    # the right recovery mode and steering parameters without cross-phase state.
    return human.step(
        env.model,
        env.data,
        {
            **_build_move_ctx(
                env,
                human,
                idx,
                world_frame,
                repulsion_vec,
                _current_human_modes(env),
                HumanMode.FOLLOWING if env.follow_phase is not None else HumanMode.WANDERING,
            ),
            "follow_radius": FOLLOW_RADIUS_DEFAULT,
        },
    )


def apply_listening_phase_strategy(env, human, idx: int, world_frame) -> np.ndarray:
    """Compute one human action while the robot is in the listening controller."""
    repulsion_vec = LISTENING_REPULSION_SCALE * _repulsion_vec(world_frame, idx)
    if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
        human.set_mode(HumanMode.LISTENING)

    _maybe_apply_fuzzy(
        env,
        human,
        idx=idx,
        context="listening",
        session_steps=human.listening_steps,
        world_frame=world_frame,
    )
    effective_robot_yaw = float(world_frame.robot_pose[2])
    # During question handling the robot may temporarily rotate toward a speaker,
    # but listeners should keep anchoring to the yaw the explanation returns to.
    if env.listening_state.question_active and env.listening_state.question_return_yaw is not None:
        effective_robot_yaw = float(env.listening_state.question_return_yaw)
    return human.step(
        env.model,
        env.data,
        {
            **_build_move_ctx(
                env,
                human,
                idx,
                world_frame,
                repulsion_vec,
                _current_human_modes(env),
                HumanMode.LISTENING,
            ),
            "robot_yaw": effective_robot_yaw,
            "listen_radius": env.listen_fan_radius,
            "listening_sector_half_angle": env.listen_front_sector_half_angle,
        },
    )


def apply_post_explanation_phase_strategy(env, human, idx: int, world_frame) -> np.ndarray:
    """Compute one human action during the post-explanation settle-out phase."""
    repulsion_vec = _repulsion_vec(world_frame, idx)
    role = env.post_explanation_state.roles[idx]
    target_xy = np.array(env.post_explanation_state.targets[idx], dtype=np.float32)
    anchor_robot_xy = (
        np.array(env.post_explanation_state.anchor_robot_xy, dtype=np.float32)
        if env.post_explanation_state.anchor_robot_xy is not None
        else np.array(world_frame.robot_xy, dtype=np.float32)
    )
    move_ctx = _build_move_ctx(
        env,
        human,
        idx,
        world_frame,
        repulsion_vec,
        _current_human_modes(env),
        HumanMode.FOLLOWING if role == POST_EXPLANATION_ROLE_YIELD else HumanMode.LISTENING,
    )
    if human.mode in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
        return human.step(env.model, env.data, move_ctx)

    # Yield-role humans peel away and resume following-like motion, while wait-role
    # humans stay anchored to the explanation pose until the settle phase ends.
    if role == POST_EXPLANATION_ROLE_YIELD:
        human.set_mode(HumanMode.FOLLOWING)
        return human.step(
            env.model,
            env.data,
            {
                **move_ctx,
                "behavior_kind": "post_explanation_yield",
                "target_xy": target_xy.copy(),
            },
        )

    human.set_mode(HumanMode.LISTENING)
    return human.step(
        env.model,
        env.data,
        {
            "behavior_kind": "post_explanation_listening_anchor",
            "robot_xy": world_frame.robot_xy,
            "robot_yaw": world_frame.robot_pose[2],
            "repulsion": LISTENING_REPULSION_SCALE * repulsion_vec,
            "listen_radius": env.post_explanation_state.listen_radii[idx],
            "listening_sector_half_angle": env.listen_front_sector_half_angle,
            "anchor_robot_xy": anchor_robot_xy,
            "anchor_robot_yaw": float(env.post_explanation_state.anchor_robot_yaw),
            "live_robot_xy": world_frame.robot_xy,
        },
    )


def apply_human_controls(env, world_frame) -> None:
    """Write the current per-human controller outputs into the MuJoCo control buffer."""
    for idx, human in enumerate(env.humans):
        if env.post_explanation_state.active:
            human_action = apply_post_explanation_phase_strategy(env, human, idx, world_frame)
        elif env.listening_state.controller_active:
            human_action = apply_listening_phase_strategy(env, human, idx, world_frame)
        else:
            human_action = apply_general_phase_strategy(env, human, idx, world_frame)
        ctrl_idx = 3 + idx * 3
        env.data.ctrl[ctrl_idx : ctrl_idx + 3] = human_action


def get_nearest_close_human_idx(world_frame) -> Optional[int]:
    """Return the nearest human inside the robot personal-space trigger radius."""
    distances = np.asarray(world_frame.observations.human_robot_distance, dtype=np.float32)
    if distances.size == 0:
        return None
    nearest_idx = int(np.argmin(distances))
    if float(distances[nearest_idx]) >= float(ROBOT_PERSONAL_SPACE_TRIGGER_METERS):
        return None
    return nearest_idx


def is_robot_backoff_direction_safe(
    env,
    world_frame,
    target_idx: int,
    direction_xy,
) -> tuple[bool, float, float, float]:
    """Check whether a candidate robot backoff direction increases clearance safely."""
    direction_xy = np.asarray(direction_xy, dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction_xy))
    if direction_norm <= 1e-6:
        return False, float("-inf"), float("-inf"), float("-inf")

    direction_unit = direction_xy / direction_norm
    robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
    human_xy = np.asarray(world_frame.human_xy, dtype=np.float32)
    if not (0 <= int(target_idx) < human_xy.shape[0]):
        return False, float("-inf"), float("-inf"), float("-inf")

    probe_xy = robot_xy + (float(ROBOT_PERSONAL_SPACE_PROBE_DISTANCE_METERS) * direction_unit)
    current_target_distance = float(world_frame.observations.human_robot_distance[int(target_idx)])
    candidate_target_distance = float(np.linalg.norm(human_xy[int(target_idx)] - probe_xy))
    if candidate_target_distance <= current_target_distance + 1e-6:
        return False, float("-inf"), float("-inf"), float("-inf")

    required_wall_clearance = float(
        ROBOT_FOOTPRINT_RADIUS_METERS + ROBOT_PERSONAL_SPACE_PROBE_DISTANCE_METERS
    )
    hit_distance = raycast_hit_distance(
        env.model,
        env.data,
        env.robot_body_id,
        direction_unit,
    )
    raycast_clearance = float("inf") if hit_distance is None else float(hit_distance)
    if raycast_clearance < required_wall_clearance:
        return False, float("-inf"), float("-inf"), raycast_clearance

    required_human_clearance = float(
        ROBOT_FOOTPRINT_RADIUS_METERS
        + HUMAN_WALL_FOOTPRINT_RADIUS
        + ROBOT_PERSONAL_SPACE_MARGIN_METERS
    )
    human_clearances = np.linalg.norm(human_xy - probe_xy[None, :], axis=1).astype(np.float32)
    min_human_clearance = float(np.min(human_clearances)) if human_clearances.size else float("inf")
    if np.any(human_clearances < required_human_clearance):
        return False, float("-inf"), min_human_clearance, raycast_clearance

    target_distance_gain = candidate_target_distance - current_target_distance
    return True, float(target_distance_gain), min_human_clearance, raycast_clearance


def apply_robot_personal_space_backoff_if_needed(env, robot_action, world_frame) -> np.ndarray:
    """Override the robot action when it must immediately back off from a nearby human."""
    adjusted_action = np.array(robot_action, dtype=np.float32, copy=True)
    target_idx = get_nearest_close_human_idx(world_frame)
    if target_idx is None:
        return adjusted_action

    listening_override_allowed = env.listening_state.phase in (
        LISTEN_PHASE_INTRO,
        LISTEN_PHASE_WAIT,
        LISTEN_PHASE_PAUSED,
    )
    if env.robot.callback_active or env.robot.mode == RobotMode.CALLBACK:
        return adjusted_action
    if (not listening_override_allowed) and env.robot.mode == RobotMode.STOP:
        return adjusted_action

    robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
    target_xy = np.asarray(world_frame.human_xy[int(target_idx)], dtype=np.float32)
    away_dir = np.asarray(robot_xy - target_xy, dtype=np.float32)
    away_norm = float(np.linalg.norm(away_dir))
    if away_norm <= 1e-6:
        planar_action = np.asarray(adjusted_action[:2], dtype=np.float32)
        planar_norm = float(np.linalg.norm(planar_action))
        if planar_norm > 1e-6:
            away_dir = -planar_action
        else:
            robot_yaw = float(world_frame.robot_pose[2])
            away_dir = np.array([-np.cos(robot_yaw), -np.sin(robot_yaw)], dtype=np.float32)
            if float(np.linalg.norm(away_dir)) <= 1e-6:
                away_dir = np.array([-1.0, 0.0], dtype=np.float32)
    away_norm = float(np.linalg.norm(away_dir))
    if away_norm <= 1e-6:
        return adjusted_action
    away_dir = away_dir / away_norm

    chosen_label = "blocked"
    chosen_direction = None
    away_safe, _, _, _ = is_robot_backoff_direction_safe(
        env,
        world_frame,
        target_idx=int(target_idx),
        direction_xy=away_dir,
    )
    if away_safe:
        chosen_label = "away"
        chosen_direction = np.asarray(away_dir, dtype=np.float32)
    else:
        left_perp = np.array([-away_dir[1], away_dir[0]], dtype=np.float32)
        right_perp = -left_perp
        best_candidate = None
        # When backing straight away is blocked, prefer the lateral option that
        # opens the most distance to the target while preserving human and wall
        # clearance for the probe position.
        for label, candidate_dir in (("left", left_perp), ("right", right_perp)):
            is_safe, target_distance_gain, min_human_clearance, raycast_clearance = (
                is_robot_backoff_direction_safe(
                    env,
                    world_frame,
                    target_idx=int(target_idx),
                    direction_xy=candidate_dir,
                )
            )
            if not is_safe:
                continue
            score = (
                float(target_distance_gain),
                float(min_human_clearance),
                float(raycast_clearance),
            )
            if best_candidate is None or score > best_candidate[1]:
                best_candidate = (label, score, np.asarray(candidate_dir, dtype=np.float32))
        if best_candidate is not None:
            chosen_label = str(best_candidate[0])
            chosen_direction = np.asarray(best_candidate[2], dtype=np.float32)

    if chosen_direction is None:
        adjusted_action[:2] = 0.0
    else:
        adjusted_action[:2] = (
            float(ROBOT_PERSONAL_SPACE_BACKOFF_SPEED_METERS) * np.asarray(chosen_direction, dtype=np.float32)
        )
        env.robot.mode = RobotMode.MOVE

    person_label = f"person{int(target_idx) + 1}"
    distance_now = float(world_frame.observations.human_robot_distance[int(target_idx)])
    env._log_event(
        f">>> Robot personal-space backoff near {person_label} "
        f"(distance={distance_now:.2f}m) direction={chosen_label}."
    )
    return adjusted_action


def get_following_front_sector_target_idx(env, world_frame) -> Optional[int]:
    """Pick the nearest eligible human currently inside the callback front sector."""
    human_xy = np.asarray(world_frame.human_xy, dtype=np.float32)
    if human_xy.shape[0] == 0:
        return None

    robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
    robot_yaw = float(world_frame.robot_pose[2])
    relative_xy = human_xy[:, :2] - robot_xy[None, :]
    distances = np.linalg.norm(relative_xy, axis=1).astype(np.float32)

    candidate_indices: list[int] = []
    candidate_distances: list[float] = []
    for idx, diff_xy in enumerate(relative_xy):
        # Front-sector callbacks are intentionally stricter than the generic
        # "farthest human fell behind" rule: only people already ahead of the
        # robot's attention cone are considered here.
        rel_angle = (
            0.0
            if float(np.dot(diff_xy, diff_xy)) <= 1e-12
            else wrap_to_pi(float(np.arctan2(diff_xy[1], diff_xy[0])) - robot_yaw)
        )
        if abs(rel_angle) > float(env.following_callback_front_sector_half_angle):
            continue
        candidate_indices.append(int(idx))
        candidate_distances.append(float(distances[idx]))

    if not candidate_indices:
        return None
    nearest_idx = int(np.argmin(np.asarray(candidate_distances, dtype=np.float32)))
    return int(candidate_indices[nearest_idx])


def apply_callback_response_if_needed(env, events) -> None:
    """Resolve the targeted distracted human's response when a callback cue ends."""
    target_idx = env.robot.callback_target_idx
    if target_idx is None or not (0 <= int(target_idx) < len(env.humans)):
        return

    target_human = env.humans[int(target_idx)]
    if target_human.mode != HumanMode.DISTRACTED:
        return

    profile_probs = env.callback_response_profile_probs.get(target_human.profile, {})
    rejoin_weight = max(0.0, float(profile_probs.get("rejoin", 0.0)))
    ignore_weight = max(0.0, float(profile_probs.get("ignore", 0.0)))
    total_weight = rejoin_weight + ignore_weight
    rejoin_prob = 0.0 if total_weight <= 0.0 else (rejoin_weight / total_weight)

    if float(env.np_random.random()) < rejoin_prob:
        restored_mode = HumanMode.LISTENING if env.listening_state.controller_active else HumanMode.FOLLOWING
        target_human.set_mode(restored_mode)
        events.callback_success = True
        env._log_event(f">>> {target_human.name} responded to callback and rejoined ({restored_mode}).")
        return

    events.callback_ignored = True
    env._log_event(f">>> {target_human.name} ignored callback and stayed distracted.")


def start_following_callback(env, world_frame) -> bool:
    """Start a following callback cue toward the selected or fallback target human."""
    distances = np.asarray(world_frame.observations.human_robot_distance, dtype=np.float32)
    human_xy = np.asarray(world_frame.human_xy, dtype=np.float32)
    if distances.size == 0 or human_xy.shape[0] == 0:
        return False
    target_idx = env._following_callback_override_target_idx
    # Use the front-sector override when available; otherwise fall back to the
    # more general "farthest person from the robot" callback target.
    if target_idx is None or not (0 <= int(target_idx) < human_xy.shape[0]):
        target_idx = int(np.argmax(distances))
    if not (0 <= target_idx < human_xy.shape[0]):
        return False
    env.robot.start_callback(
        target_idx=target_idx,
        target_xy=np.asarray(human_xy[target_idx, :2], dtype=np.float32),
        cue_steps=int(env.following_callback_cue_steps),
    )
    env._following_wait_callback_triggered = True
    return True


def apply_following_crowd_regulation_if_needed(
    env,
    robot_action,
    robot_mode: str,
    world_frame,
) -> tuple[np.ndarray, bool]:
    """Regulate robot motion during follow transit and trigger callbacks when needed."""
    adjusted_action = np.array(robot_action, dtype=np.float32, copy=True)
    distances = np.asarray(world_frame.observations.human_robot_distance, dtype=np.float32)
    should_start_callback = False
    if env.follow_phase != FOLLOW_PHASE_TRANSIT:
        reset_following_wait_episode(env)
        return adjusted_action, should_start_callback
    if robot_mode != RobotMode.MOVE:
        return adjusted_action, should_start_callback

    front_sector_target_idx = get_following_front_sector_target_idx(env, world_frame)
    if front_sector_target_idx is not None:
        front_sector_target_distance = float(distances[int(front_sector_target_idx)])
        # A front-sector target gets immediate priority: the robot pauses and
        # cues that person right away instead of waiting through the generic
        # follow-gap timer used for lagging crowd regulation.
        if front_sector_target_distance > float(ROBOT_PERSONAL_SPACE_TRIGGER_METERS) + 1e-6:
            env._following_callback_override_target_idx = int(front_sector_target_idx)
            if not env._following_wait_callback_triggered:
                env.robot.mode = RobotMode.STOP
                should_start_callback = True
                return np.zeros(3, dtype=np.float32), should_start_callback
            return adjusted_action, should_start_callback
        env._following_callback_override_target_idx = None

    env._following_callback_override_target_idx = None
    if distances.size == 0:
        reset_following_wait_episode(env)
        return adjusted_action, should_start_callback
    max_hr_distance = float(np.max(distances))
    if max_hr_distance > float(FOLLOWING_CALLBACK_DISTANCE_THRESHOLD_METERS):
        env.robot.mode = RobotMode.STOP
        env._following_wait_elapsed_steps += 1
        if (
            (not env._following_wait_callback_triggered)
            and env._following_wait_elapsed_steps >= int(env.following_callback_wait_steps)
        ):
            should_start_callback = True
        return np.zeros(3, dtype=np.float32), should_start_callback
    reset_following_wait_episode(env)
    if max_hr_distance <= float(FOLLOWING_SLOWDOWN_DISTANCE_THRESHOLD_METERS):
        return adjusted_action, should_start_callback

    # Above the slowdown threshold we keep moving, but deliberately soften the
    # planar command so the crowd has a chance to compress before a callback.
    adjusted_action[:2] *= float(FOLLOWING_SLOWDOWN_SPEED_SCALE)
    return adjusted_action, should_start_callback
