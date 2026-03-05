from museum_env.robot_fsm import ROBOT_STATE_WAIT, build_transition_table, decide_mode


def _ctx(**kwargs):
    base = {
        "callback_request_exists": False,
        "callback_done": False,
        "reached_display": False,
        "turn_done": False,
        "listen_mode": False,
        "callback_active": False,
        "threat_exists": False,
        "threat_dist": None,
        "move_back_safe_distance": 0.8,
        "listen_wait_active": False,
    }
    base.update(kwargs)
    return base


def _single_effect_key(decision):
    effects = decision["effects"]
    if not effects:
        return None
    return next(iter(effects.keys()))


def test_move_callback_request_has_priority_over_reached_display():
    decision = decide_mode(
        current_mode="move",
        ctx=_ctx(callback_request_exists=True, reached_display=True),
        table=build_transition_table(),
    )
    assert decision["next_mode"] == "callback"
    assert decision["effects"].get("start_callback", False) is True


def test_callback_done_returns_to_move():
    decision = decide_mode(
        current_mode="callback",
        ctx=_ctx(callback_done=True),
        table=build_transition_table(),
    )
    assert decision["next_mode"] == "move"
    assert decision["effects"].get("finish_callback", False) is True


def test_move_reached_display_switches_to_stop():
    decision = decide_mode(
        current_mode="move",
        ctx=_ctx(reached_display=True),
        table=build_transition_table(),
    )
    assert decision["next_mode"] == "stop"


def test_stop_turn_done_requests_enter_listen_effect():
    decision = decide_mode(
        current_mode="stop",
        ctx=_ctx(turn_done=True, listen_mode=False),
        table=build_transition_table(),
    )
    assert decision["next_mode"] == "stop"
    assert decision["effects"].get("enter_listen", False) is True


def test_wait_threat_too_close_switches_to_move_back():
    decision = decide_mode(
        current_mode=ROBOT_STATE_WAIT,
        ctx=_ctx(threat_exists=True, threat_dist=0.3, move_back_safe_distance=0.8),
        table=build_transition_table(),
    )
    assert decision["next_mode"] == "move_back"
    assert _single_effect_key(decision) == "set_move_back"


def test_move_back_with_safe_distance_holds_move_back():
    decision = decide_mode(
        current_mode="move_back",
        ctx=_ctx(threat_exists=True, threat_dist=1.1, move_back_safe_distance=0.8),
        table=build_transition_table(),
    )
    assert decision["next_mode"] == "move_back"
    assert _single_effect_key(decision) == "hold_move_back"


def test_move_back_threat_cleared_returns_to_stop():
    decision = decide_mode(
        current_mode="move_back",
        ctx=_ctx(threat_exists=False),
        table=build_transition_table(),
    )
    assert decision["next_mode"] == "stop"
    assert _single_effect_key(decision) == "clear_move_back"
