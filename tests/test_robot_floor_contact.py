from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env import MuseumEnv


class RobotFloorContactTests(unittest.TestCase):
    def setUp(self):
        self.env = MuseumEnv(render_mode=None, enable_event_logs=False, n_humans=0)

    def tearDown(self):
        self.env.close()

    def test_floor_is_visual_only(self):
        floor_id = self.env.model.geom("floor").id

        self.assertEqual(int(self.env.model.geom_contype[floor_id]), 0)
        self.assertEqual(int(self.env.model.geom_conaffinity[floor_id]), 0)

    def test_robot_move_command_produces_decision_scale_displacement(self):
        self.env.reset(seed=7)
        previous_xy = None
        displacements = []

        for _ in range(60):
            _obs, _reward, terminated, truncated, info = self.env.step(None)
            self.assertFalse(terminated)
            self.assertFalse(truncated)

            robot_info = info["robot"]
            robot_xy = np.asarray(robot_info["pose_xy"], dtype=np.float32)
            planar_command_speed = float(
                np.hypot(
                    robot_info["action"]["vx"],
                    robot_info["action"]["vy"],
                )
            )

            if previous_xy is not None and planar_command_speed >= 0.9:
                displacements.append(float(np.linalg.norm(robot_xy - previous_xy)))
            previous_xy = robot_xy.copy()

        self.assertGreaterEqual(len(displacements), 20)
        self.assertGreater(float(np.mean(displacements)), 0.03)


if __name__ == "__main__":
    unittest.main()
