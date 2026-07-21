from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.human import Human
from museum_env import human_behaviors


class DistractedRobotSpacingTests(unittest.TestCase):
    def setUp(self):
        self.human = Human(
            name="person1",
            body_name="person1",
            qpos_idx=0,
            max_speed=1.5,
        )

    def test_focus_distracted_keeps_only_robot_near_push(self):
        self.human.distracted_target_xy = np.array([2.0, 0.0], dtype=np.float32)
        self.human.distracted_stop_reached = False
        captured = {}

        def fake_compose_move_velocity(**kwargs):
            captured.update(kwargs)
            return np.zeros(2, dtype=np.float32)

        with patch.object(self.human, "_compose_move_velocity", side_effect=fake_compose_move_velocity):
            action = human_behaviors._step_following_distracted_focus(
                self.human,
                {
                    "repulsion": np.zeros(2, dtype=np.float32),
                    "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                },
                current_xy=np.array([0.0, 0.0], dtype=np.float32),
                yaw=0.0,
            )

        self.assertIsNone(captured["hr_distance_max"])
        self.assertEqual(float(action[0]), 0.0)
        self.assertEqual(float(action[1]), 0.0)

    def test_conversation_distracted_keeps_only_robot_near_push(self):
        self.human.distracted_target_xy = np.array([2.0, 0.0], dtype=np.float32)
        self.human.distracted_target_yaw = 0.0
        captured = {}

        def fake_compose_move_velocity(**kwargs):
            captured.update(kwargs)
            return np.zeros(2, dtype=np.float32)

        with patch.object(self.human, "_compose_move_velocity", side_effect=fake_compose_move_velocity):
            action = human_behaviors._step_distracted_conversation(
                self.human,
                {
                    "repulsion": np.zeros(2, dtype=np.float32),
                    "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                },
                current_xy=np.array([0.0, 0.0], dtype=np.float32),
                current_yaw=0.0,
                partner_xy=np.array([1.5, 0.0], dtype=np.float32),
            )

        self.assertIsNone(captured["hr_distance_max"])
        self.assertEqual(float(action[0]), 0.0)
        self.assertEqual(float(action[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
