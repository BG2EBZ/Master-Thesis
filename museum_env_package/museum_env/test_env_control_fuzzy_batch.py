import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import museum_env.env_control as env_control
from museum_env.env_state import FuzzyDebugState
from museum_env.fuzzy import FollowingFuzzyEngine
from museum_env.human import HumanMode, HumanProfile


class _FakeHuman:
    def __init__(
        self,
        name: str,
        *,
        mode: str,
        profile: str,
        following_steps: int = 0,
        listening_steps: int = 0,
    ):
        self.name = name
        self.mode = mode
        self.profile = profile
        self.following_steps = int(following_steps)
        self.listening_steps = int(listening_steps)
        self.curiosity_retrigger_cooldown_steps_remaining = 0
        self.impatient_front_offset = 1.2
        self.distracted_source = None
        self.distracted_recovery_mode = HumanMode.FOLLOWING
        self.seen_human_modes: list[tuple[str, ...]] = []

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def update_following_duration(self, eligible_following: bool) -> None:
        if eligible_following:
            self.following_steps += 1
        else:
            self.following_steps = 0

    def step(self, _model, _data, ctx):
        self.seen_human_modes.append(tuple(ctx["human_modes"]))
        return np.zeros(3, dtype=np.float32)

    def get_pose(self, _data):
        return (0.0, 0.0, 0.0)

    def start_curiosity(self, recovery_mode: str = HumanMode.FOLLOWING) -> None:
        self.mode = HumanMode.CURIOSITY
        self.distracted_recovery_mode = recovery_mode

    def start_impatient(self, recovery_mode: str = HumanMode.FOLLOWING) -> None:
        self.mode = HumanMode.IMPATIENT
        self.distracted_recovery_mode = recovery_mode

    def start_overwhelmed(self, robot_xy, current_xy, recovery_mode: str = HumanMode.FOLLOWING) -> None:
        del robot_xy, current_xy
        self.mode = HumanMode.OVERWHELMED
        self.distracted_recovery_mode = recovery_mode


class FuzzyBatchEnvControlTests(unittest.TestCase):
    def _build_world_frame(self, n_humans: int):
        return SimpleNamespace(
            robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            robot_pose=(0.0, 0.0, 0.0),
            human_xy=np.array(
                [[float(idx + 1), 0.25 * float(idx)] for idx in range(n_humans)],
                dtype=np.float32,
            ),
            repulsion_vectors=np.zeros((n_humans, 2), dtype=np.float32),
            observations=SimpleNamespace(
                nearest_human_distance_mean_1s=np.array([0.6, 1.1, 1.4][:n_humans], dtype=np.float32),
                nearest_human_distance=np.array([0.5, 1.0, 1.3][:n_humans], dtype=np.float32),
                human_robot_distance_mean_1s=np.array([1.2, 1.4, 1.8][:n_humans], dtype=np.float32),
                human_robot_distance=np.array([1.0, 1.3, 1.6][:n_humans], dtype=np.float32),
                local_crowding_count_1m=np.array([2, 3, 1][:n_humans], dtype=np.int32),
            ),
        )

    def _build_front_blocking_world_frame(self, distances):
        distances = np.asarray(distances, dtype=np.float32)
        return SimpleNamespace(
            robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            robot_pose=(0.0, 0.0, 0.0),
            human_xy=np.array(
                [[float(idx + 1), 0.0] for idx in range(len(distances))],
                dtype=np.float32,
            ),
            observations=SimpleNamespace(
                human_robot_distance=distances,
            ),
        )

    def _build_env(self, humans, *, follow_phase="transit", listening_controller_active=False, listening_fuzzy_active=False):
        n_humans = len(humans)
        return SimpleNamespace(
            humans=humans,
            dt=0.05,
            follow_phase=follow_phase,
            follow_fan_half_angle=1.0,
            impatient_fan_half_angle=0.5,
            listen_front_sector_half_angle=1.0,
            listen_fan_radius=1.0,
            following_fuzzy_engine=FollowingFuzzyEngine(),
            runtime_cache=SimpleNamespace(refresh_counter=5),
            fuzzy_debug=[FuzzyDebugState() for _ in humans],
            listening_state=SimpleNamespace(
                controller_active=bool(listening_controller_active),
                fuzzy_active=bool(listening_fuzzy_active),
                question_active=False,
                question_return_yaw=None,
            ),
            post_explanation_state=SimpleNamespace(
                active=False,
                roles=[],
                targets=np.zeros((n_humans, 2), dtype=np.float32),
                anchor_robot_xy=None,
                anchor_robot_yaw=0.0,
                listen_radii=np.ones((n_humans,), dtype=np.float32),
            ),
            step_count=7,
            model=object(),
            data=SimpleNamespace(ctrl=np.zeros(3 + (3 * n_humans), dtype=np.float32)),
            _log_event=lambda _msg: None,
            _record_episode_trigger=lambda _kind: None,
        )

    def test_should_evaluate_fuzzy_matches_current_phase_and_cache_rules(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL, following_steps=4),
            _FakeHuman("person2", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL, following_steps=6),
            _FakeHuman("person3", mode=HumanMode.FOLLOWING, profile=HumanProfile.NEURODIVERGENT, following_steps=9),
        ]
        env = self._build_env(humans, follow_phase=env_control.FOLLOW_PHASE_TRANSIT, listening_fuzzy_active=True)
        env.fuzzy_debug[1].context = "following"
        env.fuzzy_debug[1].dominant_state = "engaged"
        env.fuzzy_debug[1].refresh_counter = env.runtime_cache.refresh_counter
        env.fuzzy_debug[2].context = "listening"
        env.fuzzy_debug[2].dominant_state = "engaged"
        env.fuzzy_debug[2].refresh_counter = env.runtime_cache.refresh_counter

        self.assertTrue(env_control.should_evaluate_fuzzy(env, 0, context="following"))
        self.assertFalse(env_control.should_evaluate_fuzzy(env, 1, context="following"))
        self.assertTrue(env_control.should_evaluate_fuzzy(env, 2, context="following"))

        env.follow_phase = None
        self.assertFalse(env_control.should_evaluate_fuzzy(env, 0, context="following"))

    def test_compute_human_fuzzy_debug_uses_clipped_inputs_and_profile(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NEURODIVERGENT, following_steps=4),
        ]
        env = self._build_env(humans, follow_phase=env_control.FOLLOW_PHASE_TRANSIT)
        world_frame = self._build_world_frame(len(humans))
        clipped_inputs = {
            "following_time": 0.2,
            "hhd": 0.6,
            "hrd": 1.2,
            "density": 2.0,
            "angle": 0.0,
        }
        fuzzy_result = {
            "dominant_state": "engaged",
            "overwhelmed": 0.0,
            "distracted": 0.0,
            "impatient": 0.0,
            "engaged": 1.0,
            "curiosity": 0.0,
        }
        env.following_fuzzy_engine.clip_inputs = Mock(return_value=clipped_inputs)
        env.following_fuzzy_engine.compute = Mock(return_value=fuzzy_result)

        fuzzy_debug = env_control.compute_human_fuzzy_debug(
            env,
            idx=0,
            context="following",
            session_steps=humans[0].following_steps,
            world_frame=world_frame,
        )

        self.assertEqual(fuzzy_debug["inputs"], clipped_inputs)
        self.assertEqual(fuzzy_debug["result"], fuzzy_result)
        compute_kwargs = env.following_fuzzy_engine.compute.call_args.kwargs
        self.assertEqual(compute_kwargs["context"], "following")
        self.assertEqual(compute_kwargs["profile"], HumanProfile.NEURODIVERGENT)
        self.assertAlmostEqual(float(compute_kwargs["following_time"]), 0.2, places=7)

    def test_maybe_apply_fuzzy_only_sets_ahead_active_for_following(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.LISTENING, profile=HumanProfile.NORMAL),
        ]
        env = self._build_env(humans, listening_fuzzy_active=True)
        world_frame = self._build_world_frame(len(humans))
        fuzzy_debug = {
            "inputs": {"following_time": 1.0, "hhd": 2.0, "hrd": 3.0, "density": 4.0, "angle": 5.0},
            "result": {
                "dominant_state": "engaged",
                "overwhelmed": 0.0,
                "distracted": 0.0,
                "impatient": 0.0,
                "engaged": 1.0,
                "curiosity": 0.0,
            },
        }

        with patch("museum_env.env_control.compute_human_fuzzy_debug", return_value=dict(fuzzy_debug)):
            with patch("museum_env.env_control.in_ahead_region", return_value=True):
                with patch("museum_env.env_control.record_fuzzy_debug") as record_mock:
                    with patch("museum_env.env_control.apply_fuzzy_transition"):
                        env_control._maybe_apply_fuzzy(
                            env,
                            humans[0],
                            idx=0,
                            context="listening",
                            session_steps=1,
                            world_frame=world_frame,
                        )

        self.assertFalse(bool(record_mock.call_args.kwargs["fuzzy_debug"]["ahead_active"]))

    def test_maybe_apply_fuzzy_calls_record_and_transition_once_per_result(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL),
        ]
        env = self._build_env(humans, follow_phase=env_control.FOLLOW_PHASE_TRANSIT)
        world_frame = self._build_world_frame(len(humans))
        fuzzy_debug = {
            "inputs": {"following_time": 0.2, "hhd": 0.6, "hrd": 1.2, "density": 2.0, "angle": 0.0},
            "result": {
                "dominant_state": "distracted",
                "overwhelmed": 0.0,
                "distracted": 0.7,
                "impatient": 0.0,
                "engaged": 0.2,
                "curiosity": 0.0,
            },
        }

        with patch("museum_env.env_control.compute_human_fuzzy_debug", return_value=dict(fuzzy_debug)):
            with patch("museum_env.env_control.in_ahead_region", return_value=False):
                with patch("museum_env.env_control.record_fuzzy_debug") as record_mock:
                    with patch("museum_env.env_control.apply_fuzzy_transition") as transition_mock:
                        env_control._maybe_apply_fuzzy(
                            env,
                            humans[0],
                            idx=0,
                            context="following",
                            session_steps=4,
                            world_frame=world_frame,
                        )

        self.assertEqual(record_mock.call_count, 1)
        self.assertEqual(transition_mock.call_count, 1)
        self.assertEqual(record_mock.call_args.args[1], 0)
        self.assertEqual(transition_mock.call_args.kwargs["idx"], 0)

    def test_apply_human_controls_uses_live_mode_snapshot_after_each_transition(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL, following_steps=4),
            _FakeHuman("person2", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL, following_steps=5),
        ]
        env = self._build_env(humans, follow_phase=env_control.FOLLOW_PHASE_TRANSIT)
        world_frame = self._build_world_frame(len(humans))
        distracted_result = {
            "dominant_state": "distracted",
            "overwhelmed": 0.0,
            "distracted": 0.8,
            "impatient": 0.0,
            "engaged": 0.1,
            "curiosity": 0.0,
        }
        fuzzy_debug_sequence = [
            {
                "inputs": {"following_time": 0.2, "hhd": 0.6, "hrd": 1.2, "density": 2.0, "angle": 0.0},
                "result": dict(distracted_result),
            },
            {
                "inputs": {"following_time": 0.25, "hhd": 1.1, "hrd": 1.4, "density": 3.0, "angle": 5.0},
                "result": dict(distracted_result),
            },
        ]

        with patch("museum_env.env_control.compute_human_fuzzy_debug", side_effect=fuzzy_debug_sequence):
            with patch("museum_env.env_control.in_ahead_region", return_value=True):
                env_control.apply_human_controls(env, world_frame)

        self.assertEqual(humans[0].seen_human_modes, [(HumanMode.DISTRACTED, HumanMode.FOLLOWING)])
        self.assertEqual(humans[1].seen_human_modes, [(HumanMode.DISTRACTED, HumanMode.DISTRACTED)])

    def test_front_blocking_skips_following_engaged_human(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL),
        ]
        env = self._build_env(humans)
        env.fuzzy_debug[0].dominant_state = "engaged"
        world_frame = self._build_front_blocking_world_frame([0.3])

        with patch("museum_env.env_control.in_ahead_region", return_value=True):
            blocker_idx = env_control.get_nearest_front_blocking_human_idx(env, world_frame)

        self.assertIsNone(blocker_idx)

    def test_front_blocking_keeps_following_non_engaged_human_eligible(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL),
        ]
        env = self._build_env(humans)
        env.fuzzy_debug[0].dominant_state = "distracted"
        world_frame = self._build_front_blocking_world_frame([0.3])

        with patch("museum_env.env_control.in_ahead_region", return_value=True):
            blocker_idx = env_control.get_nearest_front_blocking_human_idx(env, world_frame)

        self.assertEqual(blocker_idx, 0)

    def test_front_blocking_keeps_non_following_human_eligible(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.IMPATIENT, profile=HumanProfile.NORMAL),
        ]
        env = self._build_env(humans)
        env.fuzzy_debug[0].dominant_state = "engaged"
        world_frame = self._build_front_blocking_world_frame([0.3])

        with patch("museum_env.env_control.in_ahead_region", return_value=True):
            blocker_idx = env_control.get_nearest_front_blocking_human_idx(env, world_frame)

        self.assertEqual(blocker_idx, 0)

    def test_front_blocking_treats_missing_dominant_state_as_eligible(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL),
        ]
        env = self._build_env(humans)
        env.fuzzy_debug[0].dominant_state = None
        world_frame = self._build_front_blocking_world_frame([0.3])

        with patch("museum_env.env_control.in_ahead_region", return_value=True):
            blocker_idx = env_control.get_nearest_front_blocking_human_idx(env, world_frame)

        self.assertEqual(blocker_idx, 0)

    def test_front_blocking_skips_engaged_follower_and_selects_next_eligible_human(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL),
            _FakeHuman("person2", mode=HumanMode.DISTRACTED, profile=HumanProfile.NORMAL),
        ]
        env = self._build_env(humans)
        env.fuzzy_debug[0].dominant_state = "engaged"
        env.fuzzy_debug[1].dominant_state = "engaged"
        world_frame = self._build_front_blocking_world_frame([0.2, 0.35])

        with patch("museum_env.env_control.in_ahead_region", return_value=True):
            blocker_idx = env_control.get_nearest_front_blocking_human_idx(env, world_frame)

        self.assertEqual(blocker_idx, 1)


if __name__ == "__main__":
    unittest.main()
