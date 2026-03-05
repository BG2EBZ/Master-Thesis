import numpy as np

from .human import HumanMode
from .human_fsm import decide_mode as decide_human_fsm_mode
from .robot import RobotMode


def build_active_human_fsm_ctx(env, callback_stay_steps: int):
    return {
        "follow_enabled": bool(env.follow_humans),
        "robot_listen_mode": bool(env.robot.listen_mode),
        "variant_trigger": None,
        "distracted_timeout": False,
        "callback_response": None,
        "callback_stay_steps": int(callback_stay_steps),
        "fear_response": None,
        "fear_completed": False,
        "fear_stay_active": False,
        "attack_hit": False,
        "impatient_timeout": False,
    }


def apply_human_mode_transition(env, human, human_fsm: dict, reason: str, callback_stay_steps: int) -> bool:
    mode_before = str(human.mode)
    mode_after = str(human_fsm["next_mode"])
    effects = dict(human_fsm.get("effects", {}))
    changed = False

    callback_submode = effects.get("set_callback_submode", None)
    if callback_submode == "stay":
        stay_steps = int(effects.get("callback_stay_steps", callback_stay_steps))
        changed = bool(human.apply_callback_response("stay", stay_steps=stay_steps)) or changed
    elif callback_submode == "ignore":
        changed = bool(human.apply_callback_response("ignore", stay_steps=0)) or changed

    if effects.get("restore_listen_anchor", False):
        anchor = human.attack_origin_listen_waypoint
        if anchor is not None:
            human.current_waypoint = np.array(anchor, dtype=np.float32)
        changed = True

    if mode_after != mode_before:
        human.transition_to(mode_after, reason=reason)
        changed = True

    return bool(changed)


def is_robot_in_move_stage(env, robot_pose, waypoint_reached_dist: float, dist_eps: float):
    if env.robot.listen_mode or env.listen_wait_active or env.robot.callback_active:
        return False
    rx, ry, _ = robot_pose
    wx, wy = env.robot.get_current_waypoint()
    dist = float(np.hypot(wx - rx, wy - ry) + dist_eps)
    return dist >= waypoint_reached_dist


def refresh_callback_rearm_flags(env):
    for idx, human in enumerate(env.humans):
        if idx < len(env.callback_triggered_for_current_distracted) and human.mode != HumanMode.DISTRACTED:
            env.callback_triggered_for_current_distracted[idx] = False


def build_callback_request(
    env,
    human_xy,
    robot_pose,
    waypoint_reached_dist: float,
    dist_eps: float,
    callback_hold_seconds: float,
    callback_distracted_trigger_steps: int,
):
    if not is_robot_in_move_stage(env, robot_pose, waypoint_reached_dist=waypoint_reached_dist, dist_eps=dist_eps):
        return None
    hold_steps = max(1, int(round(float(callback_hold_seconds) / float(env.timestep))))
    for idx in sorted(env.callback_target_whitelist):
        if idx < 0 or idx >= len(env.humans):
            continue
        human = env.humans[idx]
        if human.mode != HumanMode.DISTRACTED:
            continue
        if human.distracted_timer < int(callback_distracted_trigger_steps):
            continue
        if idx < len(env.callback_triggered_for_current_distracted):
            if env.callback_triggered_for_current_distracted[idx]:
                continue
        if idx >= human_xy.shape[0]:
            continue
        return {
            "target_idx": int(idx),
            "target_xy": np.array(human_xy[idx], dtype=np.float32),
            "hold_steps": int(hold_steps),
        }
    return None


def get_nearest_attack_threat(env, robot_xy, human_xy):
    if human_xy.size == 0:
        return None

    nearest_idx = None
    nearest_dist = None
    for idx, human in enumerate(env.humans):
        if human.mode != HumanMode.ATTACK:
            continue
        if idx >= human_xy.shape[0]:
            continue
        dist = float(np.linalg.norm(human_xy[idx] - robot_xy))
        if nearest_idx is None or dist < nearest_dist:
            nearest_idx = idx
            nearest_dist = dist

    if nearest_idx is None:
        return None

    return {
        "idx": int(nearest_idx),
        "dist": float(nearest_dist),
        "xy": np.array(human_xy[nearest_idx], dtype=np.float32),
    }


def compute_move_back_action(robot_xy, threat_xy, move_back_speed: float, dist_eps: float):
    diff = np.array(robot_xy - threat_xy, dtype=np.float32)
    norm = float(np.linalg.norm(diff))
    if norm < dist_eps:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = diff / norm
    v_xy = float(move_back_speed) * direction
    return np.array([v_xy[0], v_xy[1], 0.0], dtype=np.float32)


def sample_callback_response(rejoin_prob: float, stay_prob: float, ignore_prob: float):
    u = float(np.random.rand())
    rejoin_threshold = float(rejoin_prob)
    stay_threshold = rejoin_threshold + float(stay_prob)
    ignore_threshold = stay_threshold + float(ignore_prob)
    if u < rejoin_threshold:
        return "rejoin"
    if u < stay_threshold:
        return "stay"
    if u < ignore_threshold:
        return "ignore"
    return "ignore"


def sample_fear_response(move_back_prob: float, stay_prob: float, continue_hit_prob: float):
    u = float(np.random.rand())
    move_back_threshold = float(move_back_prob)
    stay_threshold = move_back_threshold + float(stay_prob)
    continue_hit_threshold = stay_threshold + float(continue_hit_prob)
    if u < move_back_threshold:
        return "move_back"
    if u < stay_threshold:
        return "stay"
    if u < continue_hit_threshold:
        return "continue_hit"
    return "continue_hit"


def build_wait_robot_fsm_ctx(env, threat_exists: bool, threat_dist, move_back_safe_distance: float):
    return {
        "callback_request_exists": False,
        "callback_done": False,
        "reached_display": False,
        "turn_done": False,
        "listen_mode": bool(env.robot.listen_mode),
        "callback_active": bool(env.robot.callback_active),
        "threat_exists": bool(threat_exists),
        "threat_dist": threat_dist,
        "move_back_safe_distance": float(move_back_safe_distance),
        "listen_wait_active": bool(env.listen_wait_active),
    }


def apply_wait_robot_fsm_effects(
    env,
    wait_effects: dict,
    threat,
    move_back_was_active: bool,
    robot_xy: np.ndarray,
    events: dict,
    move_back_speed: float,
    dist_eps: float,
):
    rb_action = np.zeros(3, dtype=np.float32)

    if wait_effects.get("set_move_back", False) and threat is not None:
        env.move_back_active = True
        env.move_back_attacker_idx = int(threat["idx"])
        env.robot.mode = RobotMode.MOVE_BACK
        rb_action = compute_move_back_action(
            robot_xy=robot_xy,
            threat_xy=threat["xy"],
            move_back_speed=move_back_speed,
            dist_eps=dist_eps,
        )
        if not move_back_was_active:
            events["move_back_triggered"] = True
            env._log_event(
                f">>> Robot MOVE_BACK triggered by person{env.move_back_attacker_idx + 1} "
                f"(dist={threat['dist']:.3f}m)."
            )
        return rb_action

    if wait_effects.get("hold_move_back", False) and threat is not None:
        env.move_back_active = True
        env.move_back_attacker_idx = int(threat["idx"])
        env.robot.mode = RobotMode.MOVE_BACK
        return rb_action

    env.move_back_active = False
    env.move_back_attacker_idx = None
    env.robot.mode = RobotMode.STOP
    if move_back_was_active and threat is None:
        events["move_back_completed"] = True
        env._log_event(">>> Robot MOVE_BACK completed (attack ended).")
    return rb_action


def apply_callback_response_via_human_fsm(
    env,
    recover_idx: int,
    callback_response: str,
    callback_stay_steps: int,
    happy_hold_seconds: float,
    events: dict,
):
    recover_human = env.humans[recover_idx]
    if recover_human.mode != HumanMode.DISTRACTED:
        return None, False

    callback_ctx = build_active_human_fsm_ctx(env, callback_stay_steps=callback_stay_steps)
    callback_ctx["callback_response"] = str(callback_response)
    callback_ctx["callback_stay_steps"] = int(callback_stay_steps)
    callback_fsm = decide_human_fsm_mode(
        current_mode=str(recover_human.mode),
        ctx=callback_ctx,
        table=env.human_fsm_table,
    )
    callback_reason = (
        "callback_rejoin"
        if str(callback_response) == "rejoin"
        else f"callback_response_{callback_response}"
    )
    recovered = apply_human_mode_transition(
        env=env,
        human=recover_human,
        human_fsm=callback_fsm,
        reason=callback_reason,
        callback_stay_steps=callback_stay_steps,
    )

    if recovered:
        env.callback_last_response = str(callback_response)
        env.callback_last_response_target_idx = int(recover_idx)
        events[f"callback_response_{callback_response}"] = True
        env._log_event(f">>> person{recover_idx + 1} callback response: {callback_response}.")

    if recovered and callback_response == "rejoin":
        events["callback_forced_recovery"] = True
        hold_steps = max(1, int(round(float(happy_hold_seconds) / float(env.timestep))))
        env.robot.trigger_happy(hold_steps)
        events["happy_triggered"] = True
        env._log_event(f">>> person{recover_idx + 1} forced recovery by CALLBACK -> FOLLOWING.")

    return callback_response, bool(recovered)


def apply_fear_response_via_human_fsm(env, response: str, callback_stay_steps: int, events: dict):
    if not events.get("fear_triggered", False):
        return

    idx = env.fear_attacker_idx
    if idx is None or idx < 0 or idx >= len(env.humans):
        return

    human = env.humans[idx]
    if human.mode != HumanMode.ATTACK:
        return

    fear_ctx = build_active_human_fsm_ctx(env, callback_stay_steps=callback_stay_steps)
    fear_ctx["fear_response"] = str(response)
    fear_fsm = decide_human_fsm_mode(
        current_mode=str(human.mode),
        ctx=fear_ctx,
        table=env.human_fsm_table,
    )
    freeze_attack = bool(fear_fsm["effects"].get("freeze_attack", False))
    apply_human_mode_transition(
        env=env,
        human=human,
        human_fsm=fear_fsm,
        reason=f"fear_response_{response}",
        callback_stay_steps=callback_stay_steps,
    )

    events[f"fear_response_{response}"] = True
    env.fear_current_response_mode = str(response)
    env.fear_current_response_target_idx = int(idx)
    env.fear_current_freeze_attack = bool(freeze_attack)
    env.fear_last_response = str(response)
    env.fear_last_response_target_idx = int(idx)
    env._log_event(f">>> person{idx + 1} fear response: {response}.")


def resolve_fear_on_complete_via_human_fsm(env, events: dict):
    if not events.get("fear_completed", False):
        return

    if env.fear_current_response_mode == "stay":
        idx = env.fear_current_response_target_idx
        if idx is not None and 0 <= idx < len(env.humans):
            human = env.humans[idx]
            if human.mode == HumanMode.ATTACK:
                resolve_ctx = build_active_human_fsm_ctx(env, callback_stay_steps=0)
                resolve_ctx["fear_completed"] = True
                resolve_ctx["fear_stay_active"] = True
                resolve_fsm = decide_human_fsm_mode(
                    current_mode=str(human.mode),
                    ctx=resolve_ctx,
                    table=env.human_fsm_table,
                )
                apply_human_mode_transition(
                    env=env,
                    human=human,
                    human_fsm=resolve_fsm,
                    reason="fear_stay_resolve_to_listening",
                    callback_stay_steps=0,
                )

    env.fear_current_response_mode = None
    env.fear_current_response_target_idx = None
    env.fear_current_freeze_attack = False
