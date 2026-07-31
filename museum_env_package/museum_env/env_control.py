"""Environment-side control helpers for robot and crowd orchestration."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .env_constants import (
    FOLLOW_RADIUS_DEFAULT,
    LISTENING_REPULSION_SCALE,
    ROBOT_FRONT_BLOCKING_TRIGGER_METERS,
)
from .env_runtime import resolve_fuzzy_metric_input
from .env_state import (
    FOLLOW_PHASE_TRANSIT,
    POST_EXPLANATION_ROLE_WAIT,
    POST_EXPLANATION_ROLE_YIELD,
)
from .fuzzy.human_states import in_ahead_region
from .human import (
    DISTRACTED_SOURCE_FOLLOWING,
    DISTRACTED_SOURCE_LISTENING,
    HumanMode,
)
from .robot import ROBOT_TURN_DONE_YAW_ERR, ROBOT_TURN_GAIN, ROBOT_YAW_RATE_LIMIT, RobotMode
from .spatial_utils import raycast_hit_distance, wrap_to_pi

_FRONT_BLOCKING_BYPASS_ARC_RADIANS = np.deg2rad(90.0)
_FRONT_BLOCKING_BYPASS_LOOKAHEAD_RADIANS = np.deg2rad(10.0)
_FRONT_BLOCKING_SIDE_RAY_TIE_TOLERANCE_METERS = 1e-3
_FRONT_BLOCKING_HUMAN_AVOIDANCE_RADIUS_METERS = 0.7
_FRONT_BLOCKING_HUMAN_AVOIDANCE_GAIN = 0.5


def reset_following_wait_episode(env) -> None:
    """Clear the temporary waiting/callback state used during follow transit."""
    env._following_wait_elapsed_steps = 0
    env._following_wait_callback_triggered = False
    env._following_callback_override_target_idx = None


def update_human_listening_session_progress(env) -> None:
    """Advance listening-session timers only while the robot explanation is active."""
    if not env.listening_state.fuzzy_active:
        return
    for human in env.humans:
        if human.mode == HumanMode.LISTENING:
            human.listening_steps += 1


def _current_human_modes(env) -> list[str]:
    """Return the current mode snapshot for all humans."""
    return [human.mode for human in env.humans]


def _repulsion_vec(world_frame, idx: int) -> np.ndarray:
    """Return one human's cached repulsion vector as a float32 array."""
    return np.asarray(world_frame.repulsion_vectors[idx], dtype=np.float32)


def _compute_robot_relative_angle_deg(world_frame, idx: int) -> float:
    """Return one human's robot-relative bearing in degrees."""
    robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
    human_xy = np.asarray(world_frame.human_xy[idx], dtype=np.float32)
    robot_yaw = world_frame.robot_pose[2]
    diff_xy = human_xy - robot_xy
    return np.rad2deg(wrap_to_pi(float(np.arctan2(diff_xy[1], diff_xy[0])) - robot_yaw))


def _is_front_blocking_pass_request_candidate(env, idx: int) -> bool:
    """Return whether a nearby human is eligible for pass-request blocking logic."""
    human = env.humans[idx]
    if human.mode != HumanMode.FOLLOWING:
        return True
    if not (0 <= int(idx) < len(env.fuzzy_debug)):
        return True
    return str(env.fuzzy_debug[idx].dominant_state) != "engaged"


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


def compute_human_fuzzy_debug(env, idx: int, context: str, session_steps: int, world_frame):
    """Build fuzzy inputs and evaluate the human state classifier."""
    human = env.humans[idx]
    observations = world_frame.observations
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
        angle=_compute_robot_relative_angle_deg(world_frame, idx),
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
        for state in ("overwhelmed", "distracted", "impatient", "engaged", "curiosity")
    }
    debug_state.dominant_state = str(fuzzy_debug["result"]["dominant_state"])
    debug_state.ahead_active = bool(fuzzy_debug.get("ahead_active", False))
    debug_state.refresh_counter = int(env.runtime_cache.refresh_counter)


# def maybe_log_following_ahead_entry(env, idx: int, fuzzy_debug: dict) -> None:
#     """Log when a following human newly enters the fuzzy `ahead` region."""
#     debug_state = env.fuzzy_debug[idx]
#     ahead_active = bool(fuzzy_debug.get("ahead_active", False))
#     if debug_state.context != "following":
#         previous_ahead_active = False
#     else:
#         previous_ahead_active = bool(debug_state.ahead_active)
#     if (not previous_ahead_active) and ahead_active:
#         human = env.humans[idx]
#         angle_deg = float(fuzzy_debug["inputs"]["angle"])
#         # env._log_event(
#         #     f">>> {human.name} entered FOLLOWING ahead region (angle={angle_deg:.1f} deg)."
#         # )


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
        f"density={float(fuzzy_inputs['density']):.1f}, "
        f"angle={float(fuzzy_inputs['angle']):.1f}"
    )

    if dominant_state == "curiosity":
        if context != "following":
            return
        if int(human.curiosity_retrigger_cooldown_steps_remaining) > 0:
            return
        human.start_curiosity(recovery_mode=HumanMode.FOLLOWING)
        env._log_event(f">>> {human.name} became CURIOUS!{fuzzy_inputs_log}")
        return

    if dominant_state == "distracted":
        human.distracted_source = distracted_source
        human.distracted_recovery_mode = recovery_mode
        human.set_mode(HumanMode.DISTRACTED)
        env._record_episode_trigger("distracted")
        msg = f">>> {human.name} became DISTRACTED!"
        if context == "listening":
            msg = f">>> {human.name} became DISTRACTED while listening!"
        env._log_event(f"{msg}{fuzzy_inputs_log}")
        return

    if dominant_state == "impatient":
        human.start_impatient(recovery_mode=recovery_mode)
        env._record_episode_trigger("impatient")
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
        env._record_episode_trigger("overwhelmed")
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
        world_frame=world_frame,
    )
    fuzzy_debug["ahead_active"] = bool(
        context == "following"
        and in_ahead_region(
            float(fuzzy_debug["inputs"]["angle"]),
            context=context,
            profile=human.profile,
        )
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
        "step_id": int(env.step_count),
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
    if human.mode not in (
        HumanMode.CURIOSITY,
        HumanMode.DISTRACTED,
        HumanMode.OVERWHELMED,
        HumanMode.IMPATIENT,
    ):
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
            "step_id": int(env.step_count),
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
        if int(human.curiosity_retrigger_cooldown_steps_remaining) > 0:
            human.curiosity_retrigger_cooldown_steps_remaining = int(human.curiosity_retrigger_cooldown_steps_remaining) - 1
        if int(human.callback_retrigger_cooldown_steps_remaining) > 0:
            human.callback_retrigger_cooldown_steps_remaining = (
                int(human.callback_retrigger_cooldown_steps_remaining) - 1
            )
        if env.post_explanation_state.active:
            human_action = apply_post_explanation_phase_strategy(env, human, idx, world_frame)
        elif env.listening_state.controller_active:
            human_action = apply_listening_phase_strategy(env, human, idx, world_frame)
        else:
            human_action = apply_general_phase_strategy(env, human, idx, world_frame)
        ctrl_idx = 3 + idx * 3
        env.data.ctrl[ctrl_idx : ctrl_idx + 3] = human_action


def get_nearest_front_blocking_human_idx(env, world_frame) -> Optional[int]:
    """Return the nearest human who blocks the robot inside the ahead sector."""
    distances = np.asarray(world_frame.observations.human_robot_distance, dtype=np.float32)
    if distances.size == 0:
        return None

    best_idx = None
    best_distance = float("inf")
    # traverse all humans
    for idx, distance in enumerate(distances):
        if float(distance) >= float(ROBOT_FRONT_BLOCKING_TRIGGER_METERS):
            continue
        if not in_ahead_region(
            _compute_robot_relative_angle_deg(world_frame, idx),
            context="following",
            profile=env.humans[idx].profile,
        ):
            continue
        if not _is_front_blocking_pass_request_candidate(env, idx):
            continue
        if float(distance) < best_distance:
            best_idx = int(idx)
            best_distance = float(distance)
    return best_idx


def _start_front_blocking_pass_request(env, state, *, blocker_idx: int, reset_state: bool = False) -> None:
    """Enter the spoken pass-request wait state for the current blocker."""
    if reset_state:
        state.reset()
    state.blocker_idx = int(blocker_idx)
    state.speech_steps_remaining = int(env.robot_pass_request_steps)
    env._log_event(
        f">>> Robot requested passage from person{int(blocker_idx) + 1} and is waiting "
        f"{env.robot_pass_request_steps * env.dt:.1f}s."
    )


def apply_pass_request_response_if_needed(env) -> bool:
    """Resolve the current front-blocking human's response after the wait ends."""
    state = env.robot_front_blocking_state
    blocker_idx = state.blocker_idx
    if blocker_idx is None or not (0 <= int(blocker_idx) < len(env.humans)):
        return False

    blocker = env.humans[int(blocker_idx)]
    response_prob = min(
        1.0,
        float(env.pass_request_response_profile_probs.get(blocker.profile, 0.0)),
    )
    if float(env.np_random.random()) >= response_prob:
        env._log_event(f">>> {blocker.name} ignored the pass request and stayed {blocker.mode}.")
        return False

    restored_mode = HumanMode.LISTENING if env.listening_state.controller_active else HumanMode.FOLLOWING
    blocker.set_mode(restored_mode)
    env._log_event(
        f">>> {blocker.name} responded to the pass request and rejoined ({restored_mode})."
    )
    state.reset()
    return True


def apply_robot_front_blocking_stop_if_needed(env, robot_action, world_frame) -> np.ndarray:
    """Stop the robot and keep requesting passage while someone blocks the front."""
    adjusted_action = np.array(robot_action, dtype=np.float32, copy=True)
    state = env.robot_front_blocking_state

    if env.robot.mode != RobotMode.MOVE:
        state.reset()
        return adjusted_action

    # Clear any legacy bypass runtime so this controller stays in request-only mode.
    if state.bypass_active or state.bypass_turn_target_yaw is not None:
        state.reset()

    if state.blocker_idx is None:
        blocker_idx = get_nearest_front_blocking_human_idx(env, world_frame)
        if blocker_idx is None:
            return adjusted_action

        _start_front_blocking_pass_request(env, state, blocker_idx=int(blocker_idx))

    if int(state.speech_steps_remaining) > 0:
        adjusted_action[:] = 0.0
        env.robot.mode = RobotMode.STOP
        return adjusted_action

    if apply_pass_request_response_if_needed(env):
        return adjusted_action

    blocker_idx = get_nearest_front_blocking_human_idx(env, world_frame)
    if blocker_idx is None:
        state.reset()
        return adjusted_action

    _start_front_blocking_pass_request(
        env,
        state,
        blocker_idx=int(blocker_idx),
        reset_state=True,
    )
    adjusted_action[:] = 0.0
    env.robot.mode = RobotMode.STOP
    return adjusted_action


def _compute_robot_seek_action(env, *, robot_pose, target_xy) -> np.ndarray:
    """Track one temporary XY target using the robot's existing move controller gains."""
    dx = float(target_xy[0]) - float(robot_pose[0])
    dy = float(target_xy[1]) - float(robot_pose[1])
    dist = float(np.hypot(dx, dy) + 1e-8)
    desired_yaw = float(np.arctan2(dy, dx))
    yaw_err = wrap_to_pi(desired_yaw - float(robot_pose[2]))

    v = float(np.clip(env.robot.k_v * dist, 0.0, env.robot.v_max))
    vx = v * float(np.cos(float(robot_pose[2])))
    vy = v * float(np.sin(float(robot_pose[2])))
    yaw_rate = float(np.clip(env.robot.k_yaw * yaw_err, -ROBOT_YAW_RATE_LIMIT, ROBOT_YAW_RATE_LIMIT))
    return np.array([vx, vy, yaw_rate], dtype=np.float32)


def _compute_robot_turn_action(*, current_yaw: float, target_yaw: float) -> np.ndarray:
    """Rotate in place until the robot faces the desired bypass tangent."""
    yaw_err = wrap_to_pi(float(target_yaw) - float(current_yaw))
    action = np.zeros(3, dtype=np.float32)
    if abs(yaw_err) < ROBOT_TURN_DONE_YAW_ERR:
        return action
    action[2] = float(np.clip(ROBOT_TURN_GAIN * yaw_err, -ROBOT_YAW_RATE_LIMIT, ROBOT_YAW_RATE_LIMIT))
    return action


def _fallback_front_blocking_bypass_direction_sign(world_frame, blocker_idx: int) -> float:
    """Keep the previous deterministic side choice when ray distances are inconclusive."""
    relative_angle_deg = float(_compute_robot_relative_angle_deg(world_frame, int(blocker_idx)))
    return -1.0 if relative_angle_deg >= 0.0 else 1.0


def _select_front_blocking_bypass_direction_sign(env, *, blocker_idx: int, world_frame, start_angle: float) -> float:
    """Choose bypass side by comparing left/right ray free distance from the robot pose."""
    robot_yaw = float(world_frame.robot_pose[2])
    left_dir = np.array([-np.sin(robot_yaw), np.cos(robot_yaw)], dtype=np.float32)
    right_dir = -left_dir
    left_hit_distance = raycast_hit_distance(env.model, env.data, env.robot_body_id, left_dir)
    right_hit_distance = raycast_hit_distance(env.model, env.data, env.robot_body_id, right_dir)

    if left_hit_distance is None and right_hit_distance is None:
        return _fallback_front_blocking_bypass_direction_sign(world_frame, blocker_idx)
    if left_hit_distance is None:
        chosen_side_dir = left_dir
    elif right_hit_distance is None:
        chosen_side_dir = right_dir
    elif float(left_hit_distance) > float(right_hit_distance) + _FRONT_BLOCKING_SIDE_RAY_TIE_TOLERANCE_METERS:
        chosen_side_dir = left_dir
    elif float(right_hit_distance) > float(left_hit_distance) + _FRONT_BLOCKING_SIDE_RAY_TIE_TOLERANCE_METERS:
        chosen_side_dir = right_dir
    else:
        return _fallback_front_blocking_bypass_direction_sign(world_frame, blocker_idx)

    positive_sign_dir = np.array([np.cos(float(start_angle) + (0.5 * np.pi)), np.sin(float(start_angle) + (0.5 * np.pi))],dtype=np.float32,)
    negative_sign_dir = np.array([np.cos(float(start_angle) - (0.5 * np.pi)), np.sin(float(start_angle) - (0.5 * np.pi))],dtype=np.float32,)
    
    return 1.0 if float(np.dot(positive_sign_dir, chosen_side_dir)) >= float(np.dot(negative_sign_dir, chosen_side_dir)) else -1.0


def _start_front_blocking_bypass(env, state, *, blocker_idx: int, world_frame) -> None:
    """Initialize a 60-degree arc around the current blocker on the opposite side."""
    blocker_xy = np.asarray(world_frame.human_xy[int(blocker_idx)], dtype=np.float32)
    robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
    offset_xy = robot_xy - blocker_xy

    state.blocker_idx = int(blocker_idx)
    state.bypass_active = True
    state.bypass_center_xy = blocker_xy.copy()
    state.bypass_radius = float(max(np.linalg.norm(offset_xy), 1e-6))
    state.bypass_start_angle = float(np.arctan2(offset_xy[1], offset_xy[0]))
    state.bypass_direction_sign = _select_front_blocking_bypass_direction_sign(
        env=env,
        blocker_idx=int(blocker_idx),
        world_frame=world_frame,
        start_angle=float(state.bypass_start_angle),
    )
    state.bypass_turn_target_yaw = wrap_to_pi(
        float(state.bypass_start_angle) + (float(state.bypass_direction_sign) * (0.5 * np.pi))
    )


def _front_blocking_bypass_progress(state, *, world_frame) -> float:
    """Return the signed angular progress made along the active bypass arc."""
    center_xy = np.asarray(state.bypass_center_xy, dtype=np.float32)
    robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
    current_angle = float(np.arctan2(robot_xy[1] - center_xy[1], robot_xy[0] - center_xy[0]))
    progress = float(state.bypass_direction_sign) * wrap_to_pi(
        current_angle - float(state.bypass_start_angle)
    )
    return max(0.0, progress)


def _compute_front_blocking_human_avoidance_offset(state, *, world_frame) -> np.ndarray:
    """Return a bounded XY repulsion offset from nearby humans during bypass."""
    robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
    human_xy = np.asarray(world_frame.human_xy, dtype=np.float32)
    if human_xy.size == 0:
        return np.zeros(2, dtype=np.float32)

    avoidance_xy = np.zeros(2, dtype=np.float32)
    fallback_dir = np.array(
        [np.cos(float(world_frame.robot_pose[2])), np.sin(float(world_frame.robot_pose[2]))],
        dtype=np.float32,
    )
    blocker_idx = state.blocker_idx
    for idx, person_xy in enumerate(human_xy):
        if blocker_idx is not None and int(idx) == int(blocker_idx):
            continue
        diff_xy = robot_xy - np.asarray(person_xy, dtype=np.float32)
        dist = float(np.linalg.norm(diff_xy))
        if dist >= _FRONT_BLOCKING_HUMAN_AVOIDANCE_RADIUS_METERS:
            continue
        if dist <= 1e-6:
            away_dir = fallback_dir
        else:
            away_dir = diff_xy / dist
        strength = float(
            _FRONT_BLOCKING_HUMAN_AVOIDANCE_GAIN
            * (_FRONT_BLOCKING_HUMAN_AVOIDANCE_RADIUS_METERS - dist)
            / _FRONT_BLOCKING_HUMAN_AVOIDANCE_RADIUS_METERS
        )
        avoidance_xy += np.asarray(strength * away_dir, dtype=np.float32)

    avoidance_norm = float(np.linalg.norm(avoidance_xy))
    if avoidance_norm <= 1e-6:
        return np.zeros(2, dtype=np.float32)
    return np.asarray(avoidance_xy, dtype=np.float32)


def _compute_robot_bypass_action(env, *, state, world_frame) -> np.ndarray:
    """Follow the current bypass arc and lightly deflect away from nearby humans."""
    center_xy = np.asarray(state.bypass_center_xy, dtype=np.float32)
    robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
    current_angle = float(np.arctan2(robot_xy[1] - center_xy[1], robot_xy[0] - center_xy[0]))
    progress = _front_blocking_bypass_progress(state, world_frame=world_frame)
    remaining_angle = max(0.0, float(_FRONT_BLOCKING_BYPASS_ARC_RADIANS) - progress)
    lookahead_angle = min(float(_FRONT_BLOCKING_BYPASS_LOOKAHEAD_RADIANS), remaining_angle)
    target_angle = current_angle + float(state.bypass_direction_sign) * lookahead_angle
    target_xy = center_xy + float(state.bypass_radius) * np.array(
        [np.cos(target_angle), np.sin(target_angle)],
        dtype=np.float32,
    )
    target_xy = np.asarray(
        target_xy + _compute_front_blocking_human_avoidance_offset(state, world_frame=world_frame),
        dtype=np.float32,
    )
    return _compute_robot_seek_action(
        env,
        robot_pose=world_frame.robot_pose,
        target_xy=target_xy,
    )


def advance_robot_front_blocking_runtime(env) -> None:
    """Advance the one-shot pass-request timer after the current step is reported."""
    state = env.robot_front_blocking_state
    if int(state.speech_steps_remaining) > 0:
        state.speech_steps_remaining = max(0, int(state.speech_steps_remaining) - 1)


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


def get_farthest_lagging_callback_target_idx(env, distances: np.ndarray) -> Optional[int]:
    """Pick the farthest callback target that is not in same-person cooldown."""
    farthest_idx = None
    farthest_distance = -np.inf
    for idx, distance in enumerate(np.asarray(distances, dtype=np.float32)):
        if int(env.humans[idx].callback_retrigger_cooldown_steps_remaining) > 0:
            continue
        if float(distance) > farthest_distance:
            farthest_distance = float(distance)
            farthest_idx = int(idx)
    return farthest_idx


def apply_callback_response_if_needed(env, events) -> None:
    """Resolve the targeted stress-state human's response when a callback cue ends."""
    target_idx = env.robot.callback_target_idx
    if target_idx is None or not (0 <= int(target_idx) < len(env.humans)):
        return

    target_human = env.humans[int(target_idx)]
    target_mode = str(target_human.mode)
    if target_mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
        return

    mode_probs = env.callback_response_profile_probs_by_mode.get(target_mode, {})
    profile_probs = mode_probs.get(target_human.profile, {})
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
    env._log_event(f">>> {target_human.name} ignored callback and stayed {target_mode}.")


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
        target_idx = get_farthest_lagging_callback_target_idx(env, distances)
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
    grace_active = int(getattr(env, "_following_callback_resume_grace_steps_remaining", 0)) > 0
    if grace_active:
        env._following_callback_resume_grace_steps_remaining = max(
            0,
            int(env._following_callback_resume_grace_steps_remaining) - 1,
        )
        reset_following_wait_episode(env)

    front_sector_target_idx = None
    if not grace_active:
        front_sector_target_idx = get_following_front_sector_target_idx(env, world_frame)
    if front_sector_target_idx is not None:
        front_sector_target_distance = float(distances[int(front_sector_target_idx)])
        # A front-sector target gets immediate priority: the robot pauses and
        # cues that person right away once they are beyond the configured
        # callback distance threshold, instead of waiting through the generic
        # follow-gap timer used for lagging crowd regulation.
        if front_sector_target_distance > float(env.callback_trigger_distance_meters):
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
    lagging_target_idx = get_farthest_lagging_callback_target_idx(env, distances)
    lagging_target_distance = (
        float(distances[int(lagging_target_idx)])
        if lagging_target_idx is not None
        else None
    )
    if (
        (not grace_active)
        and lagging_target_distance is not None
        and lagging_target_distance > float(env.guide_config.callback_distance_m)
    ):
        env.robot.mode = RobotMode.STOP
        env._following_wait_elapsed_steps += 1
        if (
            (not env._following_wait_callback_triggered)
            and env._following_wait_elapsed_steps >= int(env.following_callback_wait_steps)
        ):
            should_start_callback = True
        return np.zeros(3, dtype=np.float32), should_start_callback
    reset_following_wait_episode(env)
    if max_hr_distance <= float(env.guide_config.slow_down_distance_m):
        return adjusted_action, should_start_callback

    # Above the slowdown threshold we keep moving, but deliberately soften the
    # planar command so the crowd has a chance to compress before a callback.
    adjusted_action[:2] *= float(env.guide_config.slowdown_speed_scale)
    return adjusted_action, should_start_callback
