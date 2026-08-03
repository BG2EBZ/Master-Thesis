import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env import env_control
from museum_env.human import Human, HumanMode


def _make_human() -> Human:
    return Human("person1", "person1", 0, max_speed=1.0)


def _fuzzy_inputs() -> dict:
    return {
        "following_time": 3.0,
        "listening_time": 4.0,
        "total_duration_time": 7.0,
        "pre_duration_time": 5.0,
        "hhd": 1.0,
        "hrd": 1.0,
        "density": 2.0,
        "angle": 0.0,
    }


class _FakeEnv:
    def __init__(self):
        self.data = SimpleNamespace(qpos=np.array([1.0, 2.0, 0.0], dtype=np.float32))
        self.recorded_triggers = []
        self.logged_events = []

    def _record_episode_trigger(self, trigger_name: str) -> None:
        self.recorded_triggers.append(str(trigger_name))

    def _log_event(self, message: str) -> None:
        self.logged_events.append(str(message))


class HumanActiveContextTimeResetTest(unittest.TestCase):
    def test_following_reset_deducts_only_following_streak(self):
        human = _make_human()
        human.following_steps = 7
        human.cumulative_following_steps = 20
        human.listening_steps = 5
        human.cumulative_listening_steps = 30
        human.first_listening_steps = 11
        human.pre_duration_steps = 13

        human.reset_active_context_time("following")

        self.assertEqual(human.following_steps, 0)
        self.assertEqual(human.cumulative_following_steps, 13)
        self.assertEqual(human.listening_steps, 5)
        self.assertEqual(human.cumulative_listening_steps, 30)
        self.assertEqual(human.first_listening_steps, 11)
        self.assertEqual(human.pre_duration_steps, 13)

    def test_listening_reset_deducts_only_listening_streak(self):
        human = _make_human()
        human.following_steps = 7
        human.cumulative_following_steps = 20
        human.listening_steps = 5
        human.cumulative_listening_steps = 30
        human.first_listening_steps = 11
        human.pre_duration_steps = 13

        human.reset_active_context_time("listening")

        self.assertEqual(human.following_steps, 7)
        self.assertEqual(human.cumulative_following_steps, 20)
        self.assertEqual(human.listening_steps, 0)
        self.assertEqual(human.cumulative_listening_steps, 25)
        self.assertEqual(human.first_listening_steps, 11)
        self.assertEqual(human.pre_duration_steps, 13)

    def test_reset_clamps_cumulative_at_zero(self):
        human = _make_human()
        human.following_steps = 9
        human.cumulative_following_steps = 3
        human.listening_steps = 8
        human.cumulative_listening_steps = 2

        human.reset_active_context_time("following")
        human.reset_active_context_time("listening")

        self.assertEqual(human.cumulative_following_steps, 0)
        self.assertEqual(human.cumulative_listening_steps, 0)


class FuzzyTransitionContextTimeResetTest(unittest.TestCase):
    def test_following_distracted_resets_before_mode_transition(self):
        human = _make_human()
        human.set_mode(HumanMode.FOLLOWING)
        human.following_steps = 6
        human.cumulative_following_steps = 20
        env = _FakeEnv()
        world_frame = SimpleNamespace(robot_xy=np.array([0.0, 0.0], dtype=np.float32))

        env_control.apply_fuzzy_transition(
            env,
            human,
            idx=0,
            context="following",
            fuzzy_result={"dominant_state": "distracted"},
            fuzzy_inputs=_fuzzy_inputs(),
            world_frame=world_frame,
        )

        self.assertEqual(human.mode, HumanMode.DISTRACTED)
        self.assertEqual(human.following_steps, 0)
        self.assertEqual(human.cumulative_following_steps, 14)
        self.assertEqual(env.recorded_triggers, ["distracted"])

    def test_listening_impatient_resets_listening_but_keeps_pre_duration(self):
        human = _make_human()
        human.set_mode(HumanMode.LISTENING)
        human.listening_steps = 4
        human.cumulative_listening_steps = 10
        human.first_listening_steps = 10
        human.pre_duration_steps = 99
        env = _FakeEnv()
        world_frame = SimpleNamespace(robot_xy=np.array([0.0, 0.0], dtype=np.float32))

        env_control.apply_fuzzy_transition(
            env,
            human,
            idx=0,
            context="listening",
            fuzzy_result={"dominant_state": "impatient"},
            fuzzy_inputs=_fuzzy_inputs(),
            world_frame=world_frame,
        )

        self.assertEqual(human.mode, HumanMode.IMPATIENT)
        self.assertEqual(human.listening_steps, 0)
        self.assertEqual(human.cumulative_listening_steps, 6)
        self.assertEqual(human.first_listening_steps, 10)
        self.assertEqual(human.pre_duration_steps, 99)
        self.assertEqual(env.recorded_triggers, ["impatient"])

    def test_following_overwhelmed_resets_before_mode_transition(self):
        human = _make_human()
        human.set_mode(HumanMode.FOLLOWING)
        human.following_steps = 5
        human.cumulative_following_steps = 12
        env = _FakeEnv()
        world_frame = SimpleNamespace(robot_xy=np.array([0.0, 0.0], dtype=np.float32))

        env_control.apply_fuzzy_transition(
            env,
            human,
            idx=0,
            context="following",
            fuzzy_result={"dominant_state": "overwhelmed"},
            fuzzy_inputs=_fuzzy_inputs(),
            world_frame=world_frame,
        )

        self.assertEqual(human.mode, HumanMode.OVERWHELMED)
        self.assertEqual(human.following_steps, 0)
        self.assertEqual(human.cumulative_following_steps, 7)
        self.assertEqual(env.recorded_triggers, ["overwhelmed"])

    def test_curiosity_does_not_deduct_cumulative_following(self):
        human = _make_human()
        human.set_mode(HumanMode.FOLLOWING)
        human.following_steps = 5
        human.cumulative_following_steps = 12
        env = _FakeEnv()
        world_frame = SimpleNamespace(robot_xy=np.array([0.0, 0.0], dtype=np.float32))

        env_control.apply_fuzzy_transition(
            env,
            human,
            idx=0,
            context="following",
            fuzzy_result={"dominant_state": "curiosity"},
            fuzzy_inputs=_fuzzy_inputs(),
            world_frame=world_frame,
        )

        self.assertEqual(human.mode, HumanMode.CURIOSITY)
        self.assertEqual(human.following_steps, 0)
        self.assertEqual(human.cumulative_following_steps, 12)
        self.assertEqual(env.recorded_triggers, [])


if __name__ == "__main__":
    unittest.main()
