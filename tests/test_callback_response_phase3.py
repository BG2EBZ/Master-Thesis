import numpy as np
import pytest

from museum_env.env import CALLBACK_DISTRACTED_TRIGGER_STEPS, MuseumEnv
from museum_env.human import HumanMode


def _make_env():
    env = MuseumEnv(render_mode=None, enable_event_logs=False)
    env.reset(seed=0)
    env.follow_humans = True
    for human in env.humans:
        human.configure_following_variant(None, 0.0)
    return env


def _prime_immediate_callback_completion(
    env: MuseumEnv,
    *,
    target_idx: int = 0,
    distracted_timer: int = 500,
    distracted_duration: int = 1200,
):
    human = env.humans[target_idx]
    human.transition_to(HumanMode.DISTRACTED, reason="test_prime_distracted", force=True)
    human.distracted_timer = int(distracted_timer)
    human.distracted_duration = int(distracted_duration)
    human.callback_response_mode = None
    human.callback_stay_steps_remaining = 0

    env.callback_active_target_idx = int(target_idx)
    env.robot.callback_active = True
    env.robot.callback_target_idx = int(target_idx)
    env.robot.callback_target_xy = None
    env.robot.callback_hold_steps_remaining = 1
    env.robot.callback_turn_done = True


@pytest.mark.parametrize(
    ("rand_value", "expected_response"),
    [
        (0.0, "rejoin"),
        (0.5, "stay"),
        (0.9, "ignore"),
    ],
)
def test_callback_response_sampling_and_event_flags(monkeypatch, rand_value, expected_response):
    env = _make_env()
    try:
        _prime_immediate_callback_completion(env, target_idx=0)
        monkeypatch.setattr(np.random, "rand", lambda: rand_value)
        _, _, _, _, info = env.step(None)

        events = info["events"]
        assert events["callback_completed"] is True
        assert events[f"callback_response_{expected_response}"] is True

        other_responses = {"rejoin", "stay", "ignore"} - {expected_response}
        for response in other_responses:
            assert events[f"callback_response_{response}"] is False

        assert info["status"]["callback_last_response"] == expected_response
        assert info["status"]["callback_last_response_target_idx"] == 0
    finally:
        env.close()


def test_rejoin_triggers_forced_recovery_and_happy(monkeypatch):
    env = _make_env()
    try:
        _prime_immediate_callback_completion(env, target_idx=0)
        monkeypatch.setattr(np.random, "rand", lambda: 0.0)
        _, _, _, _, info = env.step(None)

        assert info["events"]["callback_response_rejoin"] is True
        assert info["events"]["callback_forced_recovery"] is True
        assert info["events"]["happy_triggered"] is True
        assert env.humans[0].mode == HumanMode.FOLLOWING
    finally:
        env.close()


def test_stay_freezes_for_400_steps_and_pauses_distracted_timer(monkeypatch):
    env = _make_env()
    try:
        _prime_immediate_callback_completion(
            env,
            target_idx=0,
            distracted_timer=420,
            distracted_duration=1200,
        )
        monkeypatch.setattr(np.random, "rand", lambda: 0.5)

        _, _, _, _, info = env.step(None)
        assert info["events"]["callback_response_stay"] is True
        assert env.humans[0].distracted_timer == 420
        assert env.humans[0].callback_response_mode == "stay"
        assert env.humans[0].callback_stay_steps_remaining == 399

        for _ in range(398):
            env.step(None)

        assert env.humans[0].callback_response_mode == "stay"
        assert env.humans[0].callback_stay_steps_remaining == 1
        assert env.humans[0].distracted_timer == 420

        env.step(None)
        assert env.humans[0].callback_response_mode is None
        assert env.humans[0].distracted_timer == 420

        env.step(None)
        assert env.humans[0].distracted_timer == 421
    finally:
        env.close()


def test_ignore_keeps_distracted_until_original_timeout(monkeypatch):
    env = _make_env()
    try:
        _prime_immediate_callback_completion(
            env,
            target_idx=0,
            distracted_timer=5,
            distracted_duration=8,
        )
        monkeypatch.setattr(np.random, "rand", lambda: 0.9)

        env.step(None)
        assert env.humans[0].mode == HumanMode.DISTRACTED
        assert env.humans[0].callback_response_mode == "ignore"
        assert env.humans[0].distracted_timer == 6

        env.step(None)
        assert env.humans[0].mode == HumanMode.DISTRACTED
        assert env.humans[0].distracted_timer == 7

        env.step(None)
        assert env.humans[0].mode == HumanMode.DISTRACTED
        assert env.humans[0].distracted_timer == 8

        env.step(None)
        assert env.humans[0].mode == HumanMode.FOLLOWING
    finally:
        env.close()


def test_no_retry_within_same_distracted_episode(monkeypatch):
    env = _make_env()
    try:
        _prime_immediate_callback_completion(
            env,
            target_idx=0,
            distracted_timer=450,
            distracted_duration=1200,
        )
        monkeypatch.setattr(np.random, "rand", lambda: 0.5)

        _, _, _, _, info = env.step(None)
        assert info["events"]["callback_completed"] is True
        assert info["events"]["callback_response_stay"] is True
        assert env.callback_triggered_for_current_distracted[0] is True

        for _ in range(5):
            _, _, _, _, step_info = env.step(None)
            assert step_info["events"]["callback_triggered"] is False

        env.humans[0].transition_to(HumanMode.FOLLOWING, reason="test_exit_distracted", force=True)
        env._refresh_callback_rearm_flags()
        assert env.callback_triggered_for_current_distracted[0] is False

        env.humans[0].transition_to(HumanMode.DISTRACTED, reason="test_reenter_distracted", force=True)
        env.humans[0].distracted_timer = CALLBACK_DISTRACTED_TRIGGER_STEPS
        human_xy = env._get_human_poses()[:, :2]
        robot_pose = env._get_robot_pose()

        original_stage_check = env._is_robot_in_move_stage
        env._is_robot_in_move_stage = lambda *_: True
        try:
            callback_request = env._build_callback_request(human_xy=human_xy, robot_pose=robot_pose)
        finally:
            env._is_robot_in_move_stage = original_stage_check
        assert callback_request is not None
    finally:
        env.close()
