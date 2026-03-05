import numpy as np
import pytest

rules = pytest.importorskip("museum_env.rules")


def test_resolve_text_label_priority():
    assert rules.resolve_text_label(fear_active=True, any_distracted=True, speaker_active=True) == "I_need_more_space"
    assert rules.resolve_text_label(fear_active=False, any_distracted=True, speaker_active=True) == "Please_follow_me"
    assert rules.resolve_text_label(fear_active=False, any_distracted=False, speaker_active=True) == "explanation"
    assert rules.resolve_text_label(fear_active=False, any_distracted=False, speaker_active=False) == "none"


def test_resolve_text_label_visibility_priority():
    assert rules.resolve_text_label_visibility(fear_active=True, any_distracted=True, speaker_active=True) == (
        True,
        False,
        False,
    )
    assert rules.resolve_text_label_visibility(fear_active=False, any_distracted=True, speaker_active=True) == (
        False,
        True,
        False,
    )
    assert rules.resolve_text_label_visibility(fear_active=False, any_distracted=False, speaker_active=True) == (
        False,
        False,
        True,
    )


def test_build_callback_request_only_in_move_stage():
    human_xy = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    human_modes = ["distracted", "following"]
    distracted_timers = [450, 0]
    whitelist = (0,)
    triggered = [False, False]

    request = rules.build_callback_request(
        human_xy=human_xy,
        human_modes=human_modes,
        human_distracted_timers=distracted_timers,
        robot_pose=(0.0, 0.0, 0.0),
        current_waypoint=(5.0, 0.0),
        callback_target_whitelist=whitelist,
        callback_triggered_for_current_distracted=triggered,
        timestep=0.002,
        callback_hold_seconds=1.0,
        callback_distracted_trigger_steps=400,
        listen_mode=False,
        listen_wait_active=False,
        callback_active=False,
        waypoint_reached_dist=0.2,
        dist_eps=1e-8,
        distracted_mode="distracted",
    )

    assert request is not None
    assert request["target_idx"] == 0
    assert request["hold_steps"] == 500
    assert np.allclose(request["target_xy"], np.array([2.0, 0.0], dtype=np.float32))

    blocked = rules.build_callback_request(
        human_xy=human_xy,
        human_modes=human_modes,
        human_distracted_timers=distracted_timers,
        robot_pose=(0.0, 0.0, 0.0),
        current_waypoint=(5.0, 0.0),
        callback_target_whitelist=whitelist,
        callback_triggered_for_current_distracted=triggered,
        timestep=0.002,
        callback_hold_seconds=1.0,
        callback_distracted_trigger_steps=400,
        listen_mode=True,
        listen_wait_active=False,
        callback_active=False,
        waypoint_reached_dist=0.2,
        dist_eps=1e-8,
        distracted_mode="distracted",
    )
    assert blocked is None


def test_find_nearest_attack_threat_and_move_back_action():
    robot_xy = np.array([0.0, 0.0], dtype=np.float32)
    human_xy = np.array(
        [
            [0.3, 0.0],
            [1.0, 0.0],
            [0.2, 0.2],
        ],
        dtype=np.float32,
    )
    human_modes = ["attack", "following", "attack"]

    threat = rules.find_nearest_attack_threat(
        robot_xy=robot_xy,
        human_xy=human_xy,
        human_modes=human_modes,
        attack_mode="attack",
    )

    assert threat is not None
    assert threat["idx"] == 2
    assert threat["dist"] < 0.3

    action = rules.compute_move_back_action(
        robot_xy=robot_xy,
        threat_xy=threat["xy"],
        move_back_speed=0.6,
        dist_eps=1e-8,
    )

    assert action.shape == (3,)
    assert np.isclose(np.linalg.norm(action[:2]), 0.6, atol=1e-6)
    assert action[2] == 0.0
