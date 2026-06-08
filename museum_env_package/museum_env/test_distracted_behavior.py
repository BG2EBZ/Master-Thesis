import unittest
from unittest.mock import patch

import numpy as np

from museum_env import human_behaviors
from museum_env.human import Human


class DistractedBehaviorTests(unittest.TestCase):
    def _make_human(self):
        human = Human("person1", "person1", qpos_idx=3, max_speed=1.0)
        human.enable_event_logs = False
        return human

    def _make_ctx(self):
        return {
            "repulsion": np.zeros(2, dtype=np.float32),
            "robot_xy": np.array([0.5, 0.0], dtype=np.float32),
            "human_xy": np.array([[0.0, 0.0]], dtype=np.float32),
            "human_modes": [],
            "index": 0,
        }

    def test_following_distracted_focus_keeps_lateral_force_aware_velocity(self):
        human = self._make_human()
        human.distracted_target_xy = np.array([1.0, 0.0], dtype=np.float32)
        ctx = self._make_ctx()
        lateral_velocity = np.array([0.0, 0.2], dtype=np.float32)

        with patch.object(human, "_compose_move_velocity", return_value=lateral_velocity):
            action = human_behaviors._step_following_distracted_focus(
                human,
                ctx,
                current_xy=np.array([0.0, 0.0], dtype=np.float32),
                yaw=0.0,
            )

        self.assertTrue(np.allclose(action[:2], lateral_velocity))

    def test_following_distracted_focus_keeps_zero_force_aware_velocity(self):
        human = self._make_human()
        human.distracted_target_xy = np.array([1.0, 0.0], dtype=np.float32)
        ctx = self._make_ctx()
        zero_velocity = np.zeros(2, dtype=np.float32)

        with patch.object(human, "_compose_move_velocity", return_value=zero_velocity):
            action = human_behaviors._step_following_distracted_focus(
                human,
                ctx,
                current_xy=np.array([0.0, 0.0], dtype=np.float32),
                yaw=0.0,
            )

        self.assertTrue(np.allclose(action[:2], zero_velocity))

    def test_distracted_conversation_still_uses_force_aware_velocity(self):
        human = self._make_human()
        human.distracted_target_xy = np.array([1.0, 0.0], dtype=np.float32)
        human.distracted_target_yaw = 0.0
        ctx = self._make_ctx()
        force_aware_velocity = np.array([0.1, 0.15], dtype=np.float32)

        with patch.object(human, "_compose_move_velocity", return_value=force_aware_velocity):
            action = human_behaviors._step_distracted_conversation(
                human,
                ctx,
                current_xy=np.array([0.0, 0.0], dtype=np.float32),
                current_yaw=0.0,
                partner_xy=np.array([1.5, 0.0], dtype=np.float32),
            )

        self.assertTrue(np.allclose(action[:2], force_aware_velocity))


if __name__ == "__main__":
    unittest.main()
