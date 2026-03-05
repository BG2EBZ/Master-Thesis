from museum_env.human_fsm import (
    HUMAN_MODE_ATTACK,
    HUMAN_MODE_DISTRACTED,
    HUMAN_MODE_FOLLOWING,
    HUMAN_MODE_LISTENING,
    HUMAN_MODE_WANDERING,
    build_transition_table,
    decide_mode,
)


def _ctx(**kwargs):
    base = {
        "follow_enabled": False,
        "robot_listen_mode": False,
        "variant_trigger": None,
        "distracted_timeout": False,
        "callback_response": None,
        "fear_response": None,
        "fear_completed": False,
        "fear_stay_active": False,
        "attack_hit": False,
        "impatient_timeout": False,
        "callback_stay_steps": 400,
    }
    base.update(kwargs)
    return base


def test_wandering_to_following_when_follow_enabled():
    decision = decide_mode(HUMAN_MODE_WANDERING, _ctx(follow_enabled=True), build_transition_table())
    assert decision["next_mode"] == HUMAN_MODE_FOLLOWING


def test_following_variant_to_distracted():
    decision = decide_mode(HUMAN_MODE_FOLLOWING, _ctx(variant_trigger="distracted"), build_transition_table())
    assert decision["next_mode"] == HUMAN_MODE_DISTRACTED


def test_following_variant_to_impatient():
    decision = decide_mode(HUMAN_MODE_FOLLOWING, _ctx(variant_trigger="impatient"), build_transition_table())
    assert decision["next_mode"] == "impatient"


def test_robot_listen_mode_forces_listening():
    decision = decide_mode(HUMAN_MODE_FOLLOWING, _ctx(robot_listen_mode=True), build_transition_table())
    assert decision["next_mode"] == HUMAN_MODE_LISTENING


def test_distracted_callback_rejoin_to_following():
    decision = decide_mode(HUMAN_MODE_DISTRACTED, _ctx(callback_response="rejoin"), build_transition_table())
    assert decision["next_mode"] == HUMAN_MODE_FOLLOWING


def test_distracted_callback_stay_sets_submode_effect():
    decision = decide_mode(
        HUMAN_MODE_DISTRACTED,
        _ctx(callback_response="stay", callback_stay_steps=400),
        build_transition_table(),
    )
    assert decision["next_mode"] == HUMAN_MODE_DISTRACTED
    assert decision["effects"].get("set_callback_submode") == "stay"
    assert decision["effects"].get("callback_stay_steps") == 400


def test_distracted_callback_ignore_sets_submode_effect():
    decision = decide_mode(HUMAN_MODE_DISTRACTED, _ctx(callback_response="ignore"), build_transition_table())
    assert decision["next_mode"] == HUMAN_MODE_DISTRACTED
    assert decision["effects"].get("set_callback_submode") == "ignore"


def test_attack_fear_move_back_to_listening_with_anchor_restore_effect():
    decision = decide_mode(HUMAN_MODE_ATTACK, _ctx(fear_response="move_back"), build_transition_table())
    assert decision["next_mode"] == HUMAN_MODE_LISTENING
    assert decision["effects"].get("restore_listen_anchor", False) is True


def test_attack_fear_stay_keeps_attack_and_freezes():
    decision = decide_mode(HUMAN_MODE_ATTACK, _ctx(fear_response="stay"), build_transition_table())
    assert decision["next_mode"] == HUMAN_MODE_ATTACK
    assert decision["effects"].get("freeze_attack", False) is True


def test_attack_fear_completed_after_stay_to_listening():
    decision = decide_mode(
        HUMAN_MODE_ATTACK,
        _ctx(fear_completed=True, fear_stay_active=True),
        build_transition_table(),
    )
    assert decision["next_mode"] == HUMAN_MODE_LISTENING


def test_attack_hit_has_higher_priority_than_fear_continue_hit():
    decision = decide_mode(
        HUMAN_MODE_ATTACK,
        _ctx(attack_hit=True, fear_response="continue_hit"),
        build_transition_table(),
    )
    assert decision["next_mode"] == HUMAN_MODE_LISTENING
