import sys
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from museum_env import MuseumEnv


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


if __name__ == "__main__":
    unittest.main()
