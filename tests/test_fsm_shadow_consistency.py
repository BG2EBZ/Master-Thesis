from museum_env.human_fsm import decide_mode as decide_human_mode
from museum_env.robot_fsm import ROBOT_STATE_WAIT, decide_mode as decide_robot_mode


def _legacy_wait_robot_decision(move_back_was_active: bool, threat_exists: bool, threat_too_close: bool):
    if not threat_exists:
        return "stop", "clear_move_back"
    if threat_too_close:
        return "move_back", "set_move_back"
    if move_back_was_active:
        return "move_back", "hold_move_back"
    return "stop", "clear_move_back"


def _legacy_active_human_target(mode: str, robot_listen_mode: bool, follow_enabled: bool):
    if robot_listen_mode:
        if mode != "overwhelmed":
            return "listening"
        return mode

    if mode not in ("distracted", "overwhelmed", "impatient"):
        return "following" if follow_enabled else "wandering"
    return mode


def _robot_ctx(threat_exists: bool, threat_dist: float | None, safe_dist: float = 0.8):
    return {
        "callback_request_exists": False,
        "callback_done": False,
        "reached_display": False,
        "turn_done": False,
        "listen_mode": False,
        "callback_active": False,
        "threat_exists": bool(threat_exists),
        "threat_dist": threat_dist,
        "move_back_safe_distance": safe_dist,
        "listen_wait_active": True,
    }


def _human_ctx(robot_listen_mode: bool, follow_enabled: bool):
    return {
        "follow_enabled": bool(follow_enabled),
        "robot_listen_mode": bool(robot_listen_mode),
        "variant_trigger": None,
        "distracted_timeout": False,
        "callback_response": None,
        "fear_response": None,
        "fear_completed": False,
        "fear_stay_active": False,
        "attack_hit": False,
        "impatient_timeout": False,
    }


def _single_effect_key(decision):
    effects = decision["effects"]
    if not effects:
        return None
    return next(iter(effects.keys()))


def test_wait_robot_fsm_matches_legacy_rule_table():
    for move_back_was_active in (False, True):
        for threat_exists in (False, True):
            for threat_too_close in (False, True):
                if (not threat_exists) and threat_too_close:
                    continue

                threat_dist = 0.2 if threat_too_close else (1.2 if threat_exists else None)
                current_mode = "move_back" if move_back_was_active else ROBOT_STATE_WAIT
                decision = decide_robot_mode(current_mode=current_mode, ctx=_robot_ctx(threat_exists, threat_dist))

                legacy_mode, legacy_effect = _legacy_wait_robot_decision(
                    move_back_was_active=move_back_was_active,
                    threat_exists=threat_exists,
                    threat_too_close=threat_too_close,
                )

                assert decision["next_mode"] == legacy_mode
                assert _single_effect_key(decision) == legacy_effect


def test_active_human_fsm_matches_legacy_env_mode_switch_rule():
    modes = [
        "wandering",
        "following",
        "listening",
        "distracted",
        "overwhelmed",
        "impatient",
        "attack",
    ]

    for mode in modes:
        for robot_listen_mode in (False, True):
            for follow_enabled in (False, True):
                decision = decide_human_mode(current_mode=mode, ctx=_human_ctx(robot_listen_mode, follow_enabled))
                legacy_mode = _legacy_active_human_target(
                    mode=mode,
                    robot_listen_mode=robot_listen_mode,
                    follow_enabled=follow_enabled,
                )
                assert decision["next_mode"] == legacy_mode
