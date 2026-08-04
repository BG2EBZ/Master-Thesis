import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env import human_behaviors
from museum_env.human import (
    Human,
    HumanMode,
    anchor_velocity_field,
    robot_sector_velocity_field,
)


def _make_human(max_speed: float = 1.0) -> Human:
    human = Human("person1", "person1", 0, max_speed=max_speed)
    human._raycast_hit_distance = lambda _direction: None
    return human


class RobotSectorVelocityFieldTest(unittest.TestCase):
    def test_following_field_is_zero_inside_rear_sector_at_radius(self):
        velocity = robot_sector_velocity_field(
            current_xy=np.array([-1.0, 0.0], dtype=np.float32),
            robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            center_yaw=np.pi,
            desired_radius=1.0,
            sector_half_angle=np.deg2rad(80.0),
            radius_gain=3.0,
            sector_gain=1.5,
            speed_limit=2.0,
            deadband=0.05,
        )

        np.testing.assert_allclose(velocity, np.zeros(2, dtype=np.float32), atol=1e-6)

    def test_following_field_pulls_far_human_and_pushes_close_human(self):
        far_velocity = robot_sector_velocity_field(
            current_xy=np.array([-2.0, 0.0], dtype=np.float32),
            robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            center_yaw=np.pi,
            desired_radius=1.0,
            sector_half_angle=np.deg2rad(80.0),
            radius_gain=3.0,
            sector_gain=1.5,
            speed_limit=3.0,
            deadband=0.05,
        )
        close_velocity = robot_sector_velocity_field(
            current_xy=np.array([-0.5, 0.0], dtype=np.float32),
            robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            center_yaw=np.pi,
            desired_radius=1.0,
            sector_half_angle=np.deg2rad(80.0),
            radius_gain=3.0,
            sector_gain=1.5,
            speed_limit=3.0,
            deadband=0.05,
        )

        self.assertGreater(float(far_velocity[0]), 0.0)
        self.assertLess(float(close_velocity[0]), 0.0)
        self.assertAlmostEqual(float(far_velocity[1]), 0.0, places=6)
        self.assertAlmostEqual(float(close_velocity[1]), 0.0, places=6)

    def test_listening_field_outside_front_sector_adds_tangent_correction(self):
        velocity = robot_sector_velocity_field(
            current_xy=np.array([0.0, -1.0], dtype=np.float32),
            robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            center_yaw=0.0,
            desired_radius=1.0,
            sector_half_angle=np.deg2rad(70.0),
            radius_gain=3.0,
            sector_gain=1.5,
            speed_limit=2.0,
            deadband=0.05,
        )

        self.assertGreater(float(velocity[0]), 0.0)
        self.assertAlmostEqual(float(velocity[1]), 0.0, places=6)


class HumanBehaviorVelocityFieldTest(unittest.TestCase):
    def test_anchor_field_returns_zero_for_nonfinite_anchor(self):
        velocity = anchor_velocity_field(
            current_xy=np.array([0.0, 0.0], dtype=np.float32),
            anchor_xy=np.array([np.nan, 1.0], dtype=np.float32),
            desired_radius=0.8,
            radius_gain=3.0,
            speed_limit=1.0,
            deadband=0.05,
        )

        np.testing.assert_allclose(velocity, np.zeros(2, dtype=np.float32), atol=1e-6)

    def test_conversation_with_nonfinite_partner_returns_finite_action(self):
        human = _make_human(max_speed=1.0)
        human.distracted_target_yaw = np.nan
        ctx = {
            "repulsion": np.zeros(2, dtype=np.float32),
            "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
        }

        action = human_behaviors._step_distracted_conversation(
            human,
            ctx,
            current_xy=np.array([1.0, 0.0], dtype=np.float32),
            current_yaw=0.0,
            partner_xy=np.array([np.nan, np.nan], dtype=np.float32),
        )

        self.assertTrue(np.all(np.isfinite(action)))

    def test_listening_action_faces_robot_and_respects_speed_limit(self):
        human = _make_human(max_speed=1.0)
        human.set_mode(HumanMode.LISTENING)
        ctx = {
            "repulsion": np.zeros(2, dtype=np.float32),
            "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(70.0),
        }

        action = human_behaviors._step_listening_like(
            human,
            ctx,
            pose=(0.0, -1.0, 0.0),
            anchor_robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            anchor_robot_yaw=0.0,
            live_robot_xy=np.array([0.0, 0.0], dtype=np.float32),
        )

        self.assertLessEqual(float(np.linalg.norm(action[:2])), 1.0 + 1e-6)
        self.assertGreater(float(action[2]), 0.0)

    def test_overwhelmed_backoff_uses_projection_to_switch_to_leave(self):
        human = _make_human(max_speed=1.0)
        human.start_overwhelmed(
            robot_xy=np.array([0.0, 0.0], dtype=np.float32),
            current_xy=np.array([1.0, 0.0], dtype=np.float32),
        )

        action = human_behaviors._step_overwhelmed(human, {}, pose=(1.0, 0.0, 0.0))
        self.assertGreater(float(action[0]), 0.0)
        self.assertEqual(human.overwhelmed_stage, "backoff")

        human_behaviors._step_overwhelmed(human, {}, pose=(1.31, 0.0, 0.0))
        self.assertEqual(human.overwhelmed_stage, "leave")


if __name__ == "__main__":
    unittest.main()
