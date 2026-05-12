from __future__ import annotations

import numpy as np

from . import env_control
from .env_constants import (
    HUMAN_FOLLOW_DISTANCE_DEFAULT,
    LISTEN_DISTANCE_SHORTEN_THRESHOLD_METERS,
    LISTEN_QUESTION_TURN_DONE_YAW_ERR,
    LISTEN_QUESTION_TURN_YAW_RATE,
    POST_EXPLANATION_HOLD_RESUME_DISTANCE,
    POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD,
    POST_EXPLANATION_YIELD_CLOSE_DISTANCE,
    POST_EXPLANATION_YIELD_CORRIDOR_WIDTH,
    POST_EXPLANATION_YIELD_DISTANCE,
)
from .env_state import (
    FOLLOW_PHASE_PRE_LISTEN_ENGAGE,
    FOLLOW_PHASE_TRANSIT,
    LISTEN_PHASE_INTRO,
    LISTEN_PHASE_PAUSED,
    LISTEN_PHASE_WAIT,
    LISTEN_QUESTION_COMPLETION_FINISH_WAIT,
    LISTEN_QUESTION_COMPLETION_RESUME_WAIT,
    LISTEN_QUESTION_PHASE_ANSWER,
    LISTEN_QUESTION_PHASE_NONE,
    LISTEN_QUESTION_PHASE_TURN_BACK,
    LISTEN_QUESTION_PHASE_TURN_TO_HUMAN,
    LISTEN_QUESTION_TIMING_MID_RANDOM,
    LISTEN_QUESTION_TIMING_POST_WAIT,
    POST_EXPLANATION_ROLE_WAIT,
    POST_EXPLANATION_ROLE_YIELD,
)
from .human import HumanMode
from .robot import RobotMode, RobotSpeechMode


def sync_robot_speaker_state(env) -> None:
    if env.listening_state.phase == LISTEN_PHASE_WAIT:
        env.robot.set_speech_mode(RobotSpeechMode.EXPLANATION)
        return
    if (
        env.listening_state.phase == LISTEN_PHASE_PAUSED
        and env.listening_state.question_phase == LISTEN_QUESTION_PHASE_ANSWER
    ):
        env.robot.set_speech_mode(RobotSpeechMode.ANSWER)
        return
    env.robot.set_speech_mode(RobotSpeechMode.NONE)


def build_question_turn_action(env, current_yaw: float, target_yaw: float) -> np.ndarray:
    yaw_err = env.robot._wrap_to_pi(float(target_yaw) - float(current_yaw))
    action = np.zeros(3, dtype=np.float32)
    if abs(yaw_err) < float(LISTEN_QUESTION_TURN_DONE_YAW_ERR):
        return action

    max_yaw_delta = float(LISTEN_QUESTION_TURN_YAW_RATE) * float(env.dt)
    if abs(yaw_err) <= max_yaw_delta:
        action[2] = float(yaw_err / float(env.dt))
    else:
        action[2] = float(np.sign(yaw_err) * float(LISTEN_QUESTION_TURN_YAW_RATE))
    return action


def set_listening_question_human_speaking(env, active: bool) -> None:
    idx = env.listening_state.question_human_idx
    if idx is None:
        return
    if 0 <= int(idx) < len(env.humans):
        env.humans[int(idx)].speaking_active = bool(active)


def clear_listening_question_humans(env) -> None:
    set_listening_question_human_speaking(env, False)
    env.listening_state.clear_active_question()


def get_listening_hold_robot_action(env, world_frame) -> np.ndarray:
    action = np.zeros(3, dtype=np.float32)
    env.robot.mode = RobotMode.STOP
    if env.listening_state.phase != LISTEN_PHASE_PAUSED:
        return action

    question_phase = env.listening_state.question_phase
    if question_phase == LISTEN_QUESTION_PHASE_TURN_TO_HUMAN:
        idx = env.listening_state.question_human_idx
        if idx is None or not (0 <= int(idx) < len(env.humans)):
            return action
        target_xy = np.asarray(world_frame.human_xy[int(idx)], dtype=np.float32)
        robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
        desired_yaw = float(np.arctan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0]))
        return build_question_turn_action(env, world_frame.robot_pose[2], desired_yaw)

    if question_phase == LISTEN_QUESTION_PHASE_TURN_BACK:
        target_yaw = env.listening_state.question_return_yaw
        if target_yaw is None:
            return action
        return build_question_turn_action(env, world_frame.robot_pose[2], float(target_yaw))

    return action


def scalar_cross_2d(a_xy, b_xy) -> float:
    return float(a_xy[0] * b_xy[1] - a_xy[1] * b_xy[0])


def build_post_explanation_yield_target(env, current_xy, robot_xy, outbound_dir):
    current_xy = np.array(current_xy, dtype=np.float32)
    robot_xy = np.array(robot_xy, dtype=np.float32)
    outbound_dir = np.array(outbound_dir, dtype=np.float32)
    diff = current_xy - robot_xy
    dist_to_robot = float(np.linalg.norm(diff))
    if dist_to_robot > 1e-6:
        away_dir = diff / dist_to_robot
    else:
        away_dir = -outbound_dir
        away_norm = float(np.linalg.norm(away_dir))
        away_dir = np.array([1.0, 0.0], dtype=np.float32) if away_norm <= 1e-6 else away_dir / away_norm

    left_perp = np.array([-outbound_dir[1], outbound_dir[0]], dtype=np.float32)
    left_norm = float(np.linalg.norm(left_perp))
    left_perp = np.array([0.0, 1.0], dtype=np.float32) if left_norm <= 1e-6 else left_perp / left_norm
    right_perp = -left_perp

    side_sign = scalar_cross_2d(diff, outbound_dir)
    preferred_lateral = left_perp if side_sign >= 0.0 else right_perp
    fallback_lateral = right_perp if side_sign >= 0.0 else left_perp
    candidate_dirs = (away_dir, preferred_lateral, fallback_lateral)

    current_clearance = abs(scalar_cross_2d(diff, outbound_dir))
    best_target = current_xy.copy()
    best_score = 0.0
    for direction in candidate_dirs:
        candidate_xy = current_xy + float(POST_EXPLANATION_YIELD_DISTANCE) * np.asarray(direction, dtype=np.float32)
        move_vec = candidate_xy - current_xy
        move_dist = float(np.linalg.norm(move_vec))
        if move_dist <= 0.02:
            continue

        new_diff = candidate_xy - robot_xy
        new_dist = float(np.linalg.norm(new_diff))
        new_clearance = abs(scalar_cross_2d(new_diff, outbound_dir))
        score = (new_dist - dist_to_robot) + 0.5 * (new_clearance - current_clearance)
        if score > best_score:
            best_score = score
            best_target = np.array(candidate_xy, dtype=np.float32)

    return best_target


def start_post_explanation_hold(env, robot_xy, robot_yaw: float, human_xy) -> None:
    robot_xy = np.array(robot_xy, dtype=np.float32)
    goal_xy = np.array(env.robot.get_current_waypoint(), dtype=np.float32)
    outbound_vec = goal_xy - robot_xy
    outbound_norm = float(np.linalg.norm(outbound_vec))
    if outbound_norm <= 1e-6:
        outbound_dir = np.array([np.cos(robot_yaw), np.sin(robot_yaw)], dtype=np.float32)
        fallback_norm = float(np.linalg.norm(outbound_dir))
        outbound_dir = (
            np.array([1.0, 0.0], dtype=np.float32)
            if fallback_norm <= 1e-6
            else outbound_dir / fallback_norm
        )
    else:
        outbound_dir = outbound_vec / outbound_norm

    env.post_explanation_state.active = True
    env.post_explanation_state.robot_start_xy = robot_xy.copy()
    env.post_explanation_state.anchor_robot_xy = robot_xy.copy()
    env.post_explanation_state.anchor_robot_yaw = float(robot_yaw)

    human_xy = np.asarray(human_xy, dtype=np.float32)
    n_humans = len(env.humans)
    targets = np.zeros((n_humans, 2), dtype=np.float32)
    listen_radii = np.zeros((n_humans,), dtype=np.float32)
    roles = [POST_EXPLANATION_ROLE_WAIT] * n_humans
    half_width = 0.5 * float(POST_EXPLANATION_YIELD_CORRIDOR_WIDTH)
    for idx in range(n_humans):
        current_xy = np.array(human_xy[idx], dtype=np.float32)
        diff = current_xy - robot_xy
        dist_to_robot = float(np.linalg.norm(diff))
        forward = float(np.dot(diff, outbound_dir))
        lateral = abs(scalar_cross_2d(diff, outbound_dir))
        should_yield = bool(
            dist_to_robot <= float(POST_EXPLANATION_YIELD_CLOSE_DISTANCE)
            or (forward >= 0.0 and lateral <= half_width)
        )

        listen_radii[idx] = max(float(np.linalg.norm(diff)), env.listen_stand_threshold)
        if should_yield:
            roles[idx] = POST_EXPLANATION_ROLE_YIELD
            targets[idx] = build_post_explanation_yield_target(
                env,
                current_xy=current_xy,
                robot_xy=robot_xy,
                outbound_dir=outbound_dir,
            )
        else:
            targets[idx] = current_xy

    env.post_explanation_state.roles = roles
    env.post_explanation_state.targets = targets
    env.post_explanation_state.listen_radii = listen_radii


def maybe_finish_post_explanation_hold(env, robot_xy, robot_speed: float) -> None:
    if (not env.post_explanation_state.active) or env.post_explanation_state.robot_start_xy is None:
        return
    moved_dist = float(
        np.linalg.norm(np.asarray(robot_xy, dtype=np.float32) - env.post_explanation_state.robot_start_xy)
    )
    if (
        robot_speed >= float(POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD)
        and moved_dist >= float(POST_EXPLANATION_HOLD_RESUME_DISTANCE)
    ):
        env.post_explanation_state.reset()
        env.follow_phase = FOLLOW_PHASE_TRANSIT


def maybe_activate_follow_phase_from_robot_progress(env, robot_xy) -> None:
    if (
        env.post_explanation_state.active
        or env.robot.listen_mode
        or env.follow_phase is not None
        or env.listening_state.interrupted
    ):
        return
    if env.robot_start_xy is None:
        return

    moved_dist = float(
        np.linalg.norm(np.asarray(robot_xy, dtype=np.float32) - np.asarray(env.robot_start_xy, dtype=np.float32))
    )
    if moved_dist < float(getattr(env, "human_follow_distance", HUMAN_FOLLOW_DISTANCE_DEFAULT)):
        return
    env.follow_phase = FOLLOW_PHASE_TRANSIT if env.robot.listen_done else FOLLOW_PHASE_PRE_LISTEN_ENGAGE


def maybe_shorten_listening_wait(env, world_frame) -> None:
    if env.listening_state.phase != LISTEN_PHASE_WAIT:
        return

    target_steps = env.listening_state.ensure_wait_target_steps(env.listen_wait_steps)
    human_xy = getattr(world_frame, "human_xy", None)
    robot_xy = getattr(world_frame, "robot_xy", None)
    if human_xy is not None and robot_xy is not None:
        human_xy = np.asarray(human_xy, dtype=np.float32)
        if human_xy.size == 0:
            return
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        distances = np.linalg.norm(human_xy - robot_xy[None, :], axis=1).astype(np.float32)
    else:
        distances = np.asarray(world_frame.observations.human_robot_distance, dtype=np.float32)
    if distances.size == 0:
        return

    state = env.listening_state
    newly_triggered_indices = [
        idx
        for idx, distance in enumerate(distances)
        if float(distance) > float(LISTEN_DISTANCE_SHORTEN_THRESHOLD_METERS)
        and idx not in state.distance_shorten_triggered_indices
    ]
    if not newly_triggered_indices:
        return

    state.distance_shorten_triggered_indices.update(int(idx) for idx in newly_triggered_indices)
    new_target_steps = max(
        1,
        target_steps - (len(newly_triggered_indices) * int(env.listen_distance_shorten_steps)),
    )
    applied_reduction_steps = target_steps - new_target_steps
    state.wait_target_steps = int(new_target_steps)

    people = ", ".join(f"person{int(idx) + 1}" for idx in newly_triggered_indices)
    env._log_event(
        f">>> Listening wait shortened by {applied_reduction_steps * env.dt:.1f}s "
        f"after {people} exceeded {LISTEN_DISTANCE_SHORTEN_THRESHOLD_METERS:.1f}m; "
        f"target now {state.wait_target_steps * env.dt:.1f}s."
    )


def prepare_listening_question_plan(env) -> None:
    if env.listening_state.phase != LISTEN_PHASE_WAIT:
        return

    state = env.listening_state
    if state.session_has_question or state.question_timing_mode is not None or state.question_fired:
        return

    state.session_has_question = float(env.np_random.random()) < float(env.listen_question_probability)
    if not state.session_has_question:
        state.question_fired = True
        return

    if float(env.np_random.random()) < float(env.listen_question_after_explanation_probability):
        state.question_timing_mode = LISTEN_QUESTION_TIMING_POST_WAIT
        return

    start_step = max(1, int(env.listen_wait_steps) // 2)
    end_step = int(env.listen_wait_steps) - 1
    if start_step > end_step:
        state.session_has_question = False
        return

    state.question_timing_mode = LISTEN_QUESTION_TIMING_MID_RANDOM
    state.question_trigger_step = int(env.np_random.integers(start_step, end_step + 1))


def maybe_start_listening_question(env, events, timing_mode: str, world_frame) -> bool:
    if env.listening_state.phase != LISTEN_PHASE_WAIT:
        return False
    if not env.listening_state.session_has_question or env.listening_state.question_fired:
        return False
    if env.listening_state.question_timing_mode != timing_mode:
        return False

    if timing_mode == LISTEN_QUESTION_TIMING_MID_RANDOM:
        trigger_step = env.listening_state.question_trigger_step
        if trigger_step is None or env.listening_state.counter < int(trigger_step):
            return False

    candidate_indices = [
        idx for idx, human in enumerate(env.humans) if human.mode == HumanMode.LISTENING
    ]
    if not candidate_indices:
        env.listening_state.question_fired = True
        env._log_event(">>> Listening question skipped: no LISTENING human available.")
        return False

    question_human_idx = int(candidate_indices[int(env.np_random.integers(0, len(candidate_indices)))])
    env.listening_state.pause()
    env.listening_state.question_human_idx = question_human_idx
    env.listening_state.question_phase = LISTEN_QUESTION_PHASE_TURN_TO_HUMAN
    env.listening_state.question_ask_steps_remaining = int(env.listen_question_pause_steps)
    env.listening_state.question_return_yaw = float(world_frame.robot_pose[2])
    env.listening_state.question_completion_mode = (
        LISTEN_QUESTION_COMPLETION_FINISH_WAIT
        if timing_mode == LISTEN_QUESTION_TIMING_POST_WAIT
        else LISTEN_QUESTION_COMPLETION_RESUME_WAIT
    )
    env.listening_state.question_fired = True
    set_listening_question_human_speaking(env, True)
    events.question_started = True
    env._log_event(f">>> Listening question started ({timing_mode}) by person{question_human_idx + 1}.")
    return True


def _reset_human_listening_sessions(env) -> None:
    for human in env.humans:
        human.reset_listening_session_state()


def finish_listening_wait(env, events, world_frame) -> None:
    is_final = bool(env.listening_state.is_final)
    events.completed_listen_wait = True
    clear_listening_question_humans(env)
    env.listening_state.enter_idle()
    _reset_human_listening_sessions(env)

    if is_final:
        events.final_listen_ready = True
        env._log_event(">>> Listening wait complete at final display.")
        return

    env.robot.on_listening_complete()
    env.follow_phase = None
    env.robot_start_xy = np.array(world_frame.robot_xy, dtype=np.float32)
    start_post_explanation_hold(
        env,
        robot_xy=world_frame.robot_xy,
        robot_yaw=world_frame.robot_pose[2],
        human_xy=world_frame.human_xy,
    )
    env._log_event(">>> Listening wait complete. Resume MOVE to Room B.")


def _complete_listening_question(
    env,
    events,
    world_frame,
    *,
    finish_wait_now: bool = False,
    finish_if_wait_elapsed: bool = False,
) -> None:
    clear_listening_question_humans(env)
    env.listening_state.resume()
    events.question_completed = True
    env._log_event(">>> Listening question completed.")
    if finish_wait_now:
        finish_listening_wait(env, events, world_frame)
        return
    if finish_if_wait_elapsed and (
        env.listening_state.phase == LISTEN_PHASE_WAIT
        and env.listening_state.counter >= env.listening_state.ensure_wait_target_steps(env.listen_wait_steps)
    ):
        finish_listening_wait(env, events, world_frame)


def progress_listening_question_pause(env, events, world_frame) -> bool:
    if env.listening_state.phase != LISTEN_PHASE_PAUSED or not env.listening_state.question_active:
        return False

    question_phase = env.listening_state.question_phase
    if question_phase == LISTEN_QUESTION_PHASE_TURN_TO_HUMAN:
        idx = env.listening_state.question_human_idx
        if idx is None or not (0 <= int(idx) < len(env.humans)):
            clear_listening_question_humans(env)
            env.listening_state.resume()
            return True

        env.listening_state.question_ask_steps_remaining = max(
            0,
            int(env.listening_state.question_ask_steps_remaining) - 1,
        )
        target_xy = np.asarray(world_frame.human_xy[int(idx)], dtype=np.float32)
        robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
        desired_yaw = float(np.arctan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0]))
        yaw_err = env.robot._wrap_to_pi(desired_yaw - float(world_frame.robot_pose[2]))
        if (
            env.listening_state.question_ask_steps_remaining > 0
            or abs(yaw_err) >= float(LISTEN_QUESTION_TURN_DONE_YAW_ERR)
        ):
            return True

        set_listening_question_human_speaking(env, False)
        env.listening_state.question_phase = LISTEN_QUESTION_PHASE_ANSWER
        env.listening_state.question_answer_steps_remaining = int(env.listen_question_pause_steps)
        return True

    if question_phase == LISTEN_QUESTION_PHASE_ANSWER:
        env.listening_state.question_answer_steps_remaining = max(
            0,
            int(env.listening_state.question_answer_steps_remaining) - 1,
        )
        if env.listening_state.question_answer_steps_remaining > 0:
            return True
        if env.listening_state.question_completion_mode == LISTEN_QUESTION_COMPLETION_FINISH_WAIT:
            _complete_listening_question(env, events, world_frame, finish_wait_now=True)
            return True
        env.listening_state.question_phase = LISTEN_QUESTION_PHASE_TURN_BACK
        return True

    if question_phase == LISTEN_QUESTION_PHASE_TURN_BACK:
        target_yaw = env.listening_state.question_return_yaw
        if target_yaw is None:
            clear_listening_question_humans(env)
            env.listening_state.resume()
            events.question_completed = True
            env._log_event(">>> Listening question completed.")
            return True

        yaw_err = env.robot._wrap_to_pi(float(target_yaw) - float(world_frame.robot_pose[2]))
        if abs(yaw_err) >= float(LISTEN_QUESTION_TURN_DONE_YAW_ERR):
            return True

        _complete_listening_question(
            env,
            events,
            world_frame,
            finish_if_wait_elapsed=True,
        )
        return True

    return question_phase != LISTEN_QUESTION_PHASE_NONE


def progress_listening_phase(env, events, world_frame) -> None:
    if env.listening_state.phase == LISTEN_PHASE_INTRO:
        env.listening_state.counter += 1
        if env.listening_state.counter >= env.listen_intro_delay_steps:
            is_final = env.listening_state.is_final
            env.listening_state.enter_wait(is_final=is_final)
            env.listening_state.initialize_wait_runtime(env.listen_wait_steps)
            prepare_listening_question_plan(env)
            events.started_listen_wait = True
            env._log_event(
                f">>> Listening explanation started after {env.listen_intro_delay_seconds:.1f}s delay."
            )
        return

    if progress_listening_question_pause(env, events, world_frame):
        return
    if env.listening_state.phase != LISTEN_PHASE_WAIT:
        return

    env.listening_state.counter += 1
    maybe_shorten_listening_wait(env, world_frame)
    if maybe_start_listening_question(env, events, LISTEN_QUESTION_TIMING_MID_RANDOM, world_frame):
        return
    if env.listening_state.counter < env.listening_state.ensure_wait_target_steps(env.listen_wait_steps):
        return
    if env.listening_state.counter >= int(env.listen_wait_steps):
        if maybe_start_listening_question(env, events, LISTEN_QUESTION_TIMING_POST_WAIT, world_frame):
            return
    finish_listening_wait(env, events, world_frame)


def _begin_listening_intro(env, events, pre_frame) -> None:
    events.entered_listen = True
    env.follow_phase = None
    clear_listening_question_humans(env)
    env.listening_state.enter_intro(is_final=env.robot.is_final_reached(pre_frame.robot_pose))
    _reset_human_listening_sessions(env)
    env.post_explanation_state.reset()
    rx, ry, ryaw = pre_frame.robot_pose
    env._log_event(
        f">>> Robot entering LISTEN mode. robot=({rx:.2f}, {ry:.2f}, yaw={ryaw:.2f}); "
        f"silent 3s preparation started while humans regulate to a "
        f"{env.listen_fan_radius:.2f}m ring inside the front 160 deg sector."
    )


def compute_robot_action(env, pre_frame, events) -> tuple[np.ndarray, bool]:
    waiting_listen_hold = env.listening_state.phase in (LISTEN_PHASE_WAIT, LISTEN_PHASE_PAUSED)
    robot_action = np.zeros(3, dtype=np.float32)
    callback_mode_this_step = False
    if waiting_listen_hold:
        return get_listening_hold_robot_action(env, pre_frame), callback_mode_this_step

    was_listening = bool(env.robot.listen_mode)
    robot_action = np.array(
        env.robot.step(
            robot_pose=pre_frame.robot_pose,
            human_xyz=pre_frame.human_xyz,
        ),
        dtype=np.float32,
    )
    callback_mode_this_step = env.robot.mode == RobotMode.CALLBACK
    if env.robot.mode == RobotMode.CALLBACK:
        if bool(env.robot.callback_cue_completed_this_step):
            events.callback_completed = True
            target_idx = env.robot.callback_target_idx
            person_label = "none" if target_idx is None else f"person{int(target_idx) + 1}"
            env._log_event(f">>> Robot callback completed for {person_label}.")
            env.robot.finish_callback()
    else:
        robot_action, should_start_callback = env_control.apply_following_crowd_regulation_if_needed(
            env,
            robot_action,
            robot_mode=env.robot.mode,
            world_frame=pre_frame,
        )
        if should_start_callback and env_control.start_following_callback(env, pre_frame):
            events.callback_triggered = True
            target_idx = env.robot.callback_target_idx
            person_label = "none" if target_idx is None else f"person{int(target_idx) + 1}"
            env._log_event(
                f">>> Robot callback triggered for {person_label} after "
                f"{env.following_callback_wait_steps * env.dt:.1f}s wait."
            )
            robot_action = np.array(
                env.robot.step(
                    robot_pose=pre_frame.robot_pose,
                    human_xyz=pre_frame.human_xyz,
                ),
                dtype=np.float32,
            )
            callback_mode_this_step = True

    if (not was_listening) and bool(env.robot.listen_mode):
        _begin_listening_intro(env, events, pre_frame)
    return robot_action, callback_mode_this_step
