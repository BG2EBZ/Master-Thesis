from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env import MuseumEnv
from museum_env import env_flow
from museum_env.guide_config import GuideBehaviorConfig
from museum_env.env_state import StepEvents


class MuseumEnvTimingTests(unittest.TestCase):
    def setUp(self):
        self.env = MuseumEnv(render_mode=None, enable_event_logs=False, n_humans=1)

    def tearDown(self):
        self.env.close()

    def test_exposes_split_physics_and_decision_timing(self):
        self.assertAlmostEqual(float(self.env.model.opt.timestep), 0.01, places=7)
        self.assertAlmostEqual(self.env.physics_dt, 0.01, places=7)
        self.assertAlmostEqual(self.env.decision_dt, 0.05, places=7)
        self.assertAlmostEqual(self.env.dt, 0.05, places=7)
        self.assertEqual(self.env.physics_steps_per_decision, 5)
        self.assertEqual(self.env._steps(0.1), 2)

    def test_one_env_step_advances_five_physics_steps_but_one_decision_step(self):
        self.env.max_steps = 1
        self.env.reset(seed=123)
        start_time = float(self.env.data.time)

        _obs, _reward, _terminated, truncated, info = self.env.step(None)

        self.assertTrue(truncated)
        self.assertEqual(self.env.step_count, 1)
        self.assertAlmostEqual(float(self.env.data.time) - start_time, 0.05, places=7)
        self.assertAlmostEqual(float(info["episode"]["duration_seconds"]), 0.05, places=7)

    def test_policy_parameters_scale_listen_wait_runtime(self):
        self.assertAlmostEqual(self.env.listen_wait_base_seconds, 30.0, places=7)
        self.assertAlmostEqual(self.env.listen_wait_seconds, 30.0, places=7)
        self.assertEqual(self.env.listen_wait_steps, 600)

        self.env.set_guide_behavior_config(
            GuideBehaviorConfig(
                slow_down_distance_m=2.5,
                callback_distance_m=3.5,
                callback_wait_seconds=2.0,
                slowdown_speed_scale=0.7,
                explanation_time_scale=0.7,
                callback_same_person_cooldown_seconds=20.0,
            )
        )

        self.assertAlmostEqual(self.env.listen_wait_seconds, 21.0, places=7)
        self.assertEqual(self.env.listen_wait_steps, 420)

    def test_listening_question_plan_uses_scaled_wait_steps(self):
        self.env.reset(seed=123)
        self.env.set_guide_behavior_config(
            GuideBehaviorConfig(
                slow_down_distance_m=2.5,
                callback_distance_m=3.5,
                callback_wait_seconds=2.0,
                slowdown_speed_scale=0.7,
                explanation_time_scale=0.7,
                callback_same_person_cooldown_seconds=20.0,
            )
        )
        self.env.listen_question_probability = 1.0
        self.env.listen_question_after_explanation_probability = 0.0
        self.env.listening_state.enter_wait(is_final=False)
        self.env.listening_state.initialize_wait_runtime(self.env.listen_wait_steps)

        env_flow.prepare_listening_question_plan(self.env)

        self.assertEqual(self.env.listening_state.question_timing_mode, "mid_random")
        self.assertIsNotNone(self.env.listening_state.question_trigger_step)
        self.assertGreaterEqual(
            int(self.env.listening_state.question_trigger_step),
            max(1, int(self.env.listen_wait_steps) // 2),
        )
        self.assertLessEqual(
            int(self.env.listening_state.question_trigger_step),
            int(self.env.listen_wait_steps) - 1,
        )

    def test_progress_listening_phase_does_not_shorten_wait_for_far_human(self):
        self.env.reset(seed=123)
        self.env.listen_question_probability = 0.0
        self.env.listening_state.enter_wait(is_final=False)
        self.env.listening_state.initialize_wait_runtime(self.env.listen_wait_steps)
        original_target_steps = int(self.env.listening_state.wait_target_steps)
        events = StepEvents()
        world_frame = SimpleNamespace(
            robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            robot_pose=(0.0, 0.0, 0.0),
            human_xy=np.array([[3.0, 0.0]], dtype=np.float32),
            observations=SimpleNamespace(
                human_robot_distance=np.array([3.0], dtype=np.float32),
            ),
        )

        env_flow.progress_listening_phase(self.env, events, world_frame)

        self.assertEqual(self.env.listening_state.counter, 1)
        self.assertEqual(self.env.listening_state.wait_target_steps, original_target_steps)
        self.assertEqual(len(self.env.listening_state.distance_shorten_triggered_indices), 0)


if __name__ == "__main__":
    unittest.main()
