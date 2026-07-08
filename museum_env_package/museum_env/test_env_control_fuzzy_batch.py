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

    def test_collect_fuzzy_candidates_matches_current_eligibility_rules(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL, following_steps=4),
            _FakeHuman("person2", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL, following_steps=6),
            _FakeHuman("person3", mode=HumanMode.DISTRACTED, profile=HumanProfile.NEURODIVERGENT, following_steps=9),
        ]
        env = self._build_env(humans, follow_phase=env_control.FOLLOW_PHASE_TRANSIT)
        env.fuzzy_debug[1].context = "following"
        env.fuzzy_debug[1].dominant_state = "engaged"
        env.fuzzy_debug[1].refresh_counter = env.runtime_cache.refresh_counter
        world_frame = self._build_world_frame(len(humans))

        candidates = env_control._collect_fuzzy_candidates(env, context="following", world_frame=world_frame)

        self.assertEqual([candidate["idx"] for candidate in candidates], [0])
        self.assertEqual(candidates[0]["profile"], HumanProfile.NORMAL)
        self.assertAlmostEqual(float(candidates[0]["inputs"]["following_time"]), 0.2, places=7)
        self.assertAlmostEqual(float(candidates[0]["inputs"]["hhd"]), 0.6, places=7)
        self.assertAlmostEqual(float(candidates[0]["inputs"]["hrd"]), 1.2, places=7)
        self.assertAlmostEqual(float(candidates[0]["inputs"]["density"]), 2.0, places=7)
        self.assertAlmostEqual(float(candidates[0]["inputs"]["angle"]), 0.0, places=7)

    def test_compute_fuzzy_batch_groups_by_profile_and_preserves_input_order(self):
        candidates = [
            {"idx": 0, "profile": HumanProfile.NORMAL, "inputs": {"following_time": 1.0, "hhd": 2.0, "hrd": 3.0, "density": 4.0, "angle": 5.0}},
            {"idx": 1, "profile": HumanProfile.NEURODIVERGENT, "inputs": {"following_time": 6.0, "hhd": 7.0, "hrd": 8.0, "density": 9.0, "angle": 10.0}},
            {"idx": 2, "profile": HumanProfile.NORMAL, "inputs": {"following_time": 11.0, "hhd": 12.0, "hrd": 13.0, "density": 14.0, "angle": 15.0}},
        ]
        compute_batch_mock = Mock(
            side_effect=[
                [
                    {"dominant_state": "engaged", "overwhelmed": 0.0, "distracted": 0.0, "impatient": 0.0, "engaged": 1.0, "curiosity": 0.0},
                    {"dominant_state": "curiosity", "overwhelmed": 0.0, "distracted": 0.0, "impatient": 0.0, "engaged": 0.3, "curiosity": 0.9},
                ],
                [
                    {"dominant_state": "distracted", "overwhelmed": 0.0, "distracted": 0.8, "impatient": 0.1, "engaged": 0.2, "curiosity": 0.0},
                ],
            ]
        )
        env = SimpleNamespace(
            following_fuzzy_engine=SimpleNamespace(compute_batch=compute_batch_mock),
        )

        with patch("museum_env.env_control.in_ahead_region", return_value=True):
            fuzzy_batch = env_control._compute_fuzzy_batch(env, candidates, context="following")

        self.assertEqual(compute_batch_mock.call_count, 2)
        normal_args, normal_kwargs = compute_batch_mock.call_args_list[0]
        nd_args, nd_kwargs = compute_batch_mock.call_args_list[1]
        np.testing.assert_allclose(
            normal_args[0],
            np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [11.0, 12.0, 13.0, 14.0, 15.0]], dtype=np.float32),
        )
        self.assertEqual(normal_kwargs["context"], "following")
        self.assertEqual(normal_kwargs["profile"], HumanProfile.NORMAL)
        np.testing.assert_allclose(
            nd_args[0],
            np.array([[6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
        )
        self.assertEqual(nd_kwargs["profile"], HumanProfile.NEURODIVERGENT)
        self.assertEqual([entry["idx"] for entry in fuzzy_batch], [0, 1, 2])
        self.assertTrue(all(bool(entry["ahead_active"]) for entry in fuzzy_batch))

    def test_compute_fuzzy_batch_only_sets_ahead_active_for_following(self):
        env = SimpleNamespace(
            following_fuzzy_engine=SimpleNamespace(
                compute_batch=Mock(
                    return_value=[
                        {
                            "dominant_state": "engaged",
                            "overwhelmed": 0.0,
                            "distracted": 0.0,
                            "impatient": 0.0,
                            "engaged": 1.0,
                            "curiosity": 0.0,
                        }
                    ]
                )
            )
        )
        candidates = [
            {
                "idx": 0,
                "profile": HumanProfile.NORMAL,
                "inputs": {"following_time": 1.0, "hhd": 2.0, "hrd": 3.0, "density": 4.0, "angle": 5.0},
            }
        ]

        with patch("museum_env.env_control.in_ahead_region", return_value=True):
            fuzzy_batch = env_control._compute_fuzzy_batch(env, candidates, context="listening")

        self.assertFalse(bool(fuzzy_batch[0]["ahead_active"]))

    def test_apply_fuzzy_batch_calls_record_and_transition_once_per_result(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL),
            _FakeHuman("person2", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL),
        ]
        env = self._build_env(humans)
        world_frame = self._build_world_frame(len(humans))
        fuzzy_batch = [
            {
                "idx": 0,
                "inputs": {"following_time": 0.2, "hhd": 0.6, "hrd": 1.2, "density": 2.0, "angle": 0.0},
                "result": {"dominant_state": "engaged", "overwhelmed": 0.0, "distracted": 0.0, "impatient": 0.0, "engaged": 1.0, "curiosity": 0.0},
                "ahead_active": True,
            },
            {
                "idx": 1,
                "inputs": {"following_time": 0.3, "hhd": 1.0, "hrd": 1.3, "density": 3.0, "angle": 10.0},
                "result": {"dominant_state": "distracted", "overwhelmed": 0.0, "distracted": 0.7, "impatient": 0.0, "engaged": 0.2, "curiosity": 0.0},
                "ahead_active": False,
            },
        ]

        with patch("museum_env.env_control.record_fuzzy_debug") as record_mock:
            with patch("museum_env.env_control.apply_fuzzy_transition") as transition_mock:
                env_control._apply_fuzzy_batch(env, fuzzy_batch, context="following", world_frame=world_frame)

        self.assertEqual(record_mock.call_count, 2)
        self.assertEqual(transition_mock.call_count, 2)
        self.assertEqual(record_mock.call_args_list[0].args[1], 0)
        self.assertEqual(record_mock.call_args_list[1].args[1], 1)
        self.assertEqual(transition_mock.call_args_list[0].kwargs["idx"], 0)
        self.assertEqual(transition_mock.call_args_list[1].kwargs["idx"], 1)

    def test_apply_human_controls_max_cleanup_uses_post_transition_global_mode_snapshot(self):
        humans = [
            _FakeHuman("person1", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL, following_steps=4),
            _FakeHuman("person2", mode=HumanMode.FOLLOWING, profile=HumanProfile.NORMAL, following_steps=5),
        ]
        env = self._build_env(humans, follow_phase=env_control.FOLLOW_PHASE_TRANSIT)
        world_frame = self._build_world_frame(len(humans))
        fuzzy_batch = [
            {
                "idx": 0,
                "inputs": {"following_time": 0.2, "hhd": 0.6, "hrd": 1.2, "density": 2.0, "angle": 0.0},
                "result": {"dominant_state": "distracted", "overwhelmed": 0.0, "distracted": 0.8, "impatient": 0.0, "engaged": 0.1, "curiosity": 0.0},
                "ahead_active": True,
            },
            {
                "idx": 1,
                "inputs": {"following_time": 0.25, "hhd": 1.1, "hrd": 1.4, "density": 3.0, "angle": 5.0},
                "result": {"dominant_state": "distracted", "overwhelmed": 0.0, "distracted": 0.9, "impatient": 0.0, "engaged": 0.1, "curiosity": 0.0},
                "ahead_active": True,
            },
        ]

        def set_distracted_mode(_env, human, **_kwargs):
            human.set_mode(HumanMode.DISTRACTED)

        with patch("museum_env.env_control._collect_fuzzy_candidates", return_value=[{"idx": 0}, {"idx": 1}]):
            with patch("museum_env.env_control._compute_fuzzy_batch", return_value=fuzzy_batch):
                with patch("museum_env.env_control.record_fuzzy_debug"):
                    with patch("museum_env.env_control.apply_fuzzy_transition", side_effect=set_distracted_mode):
                        env_control.apply_human_controls(env, world_frame)

        self.assertEqual(humans[0].seen_human_modes, [(HumanMode.DISTRACTED, HumanMode.DISTRACTED)])
        self.assertEqual(humans[1].seen_human_modes, [(HumanMode.DISTRACTED, HumanMode.DISTRACTED)])


if __name__ == "__main__":
    unittest.main()
