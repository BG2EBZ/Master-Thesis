import unittest
from types import SimpleNamespace

import numpy as np

from museum_env import env_control
from museum_env.env_state import RobotFrontBlockingState
from museum_env.human import HumanProfile
from museum_env.robot import RobotMode


def _make_world_frame(robot_pose, human_xy):
    human_xy = np.asarray(human_xy, dtype=np.float32)
    robot_xy = np.asarray(robot_pose[:2], dtype=np.float32)
    if human_xy.size == 0:
        distances = np.zeros((0,), dtype=np.float32)
        human_xy = np.zeros((0, 2), dtype=np.float32)
    else:
        distances = np.linalg.norm(human_xy - robot_xy[None, :], axis=1).astype(np.float32)
    return SimpleNamespace(
        robot_pose=tuple(float(value) for value in robot_pose),
        robot_xy=robot_xy,
        human_xy=human_xy,
        observations=SimpleNamespace(human_robot_distance=distances),
    )


class FrontBlockingControlTests(unittest.TestCase):
    def _make_env(self, n_humans=1):
        robot = SimpleNamespace(
            mode=RobotMode.MOVE,
            k_v=1.0,
            v_max=1.0,
            k_yaw=2.0,
        )
        return SimpleNamespace(
            robot=robot,
            robot_front_blocking_state=RobotFrontBlockingState(),
            robot_pass_request_steps=20,
            dt=0.1,
            humans=[SimpleNamespace(profile=HumanProfile.NORMAL) for _ in range(n_humans)],
            _log_event=lambda _msg: None,
        )

    def test_front_blocking_starts_with_pass_request_wait(self):
        env = self._make_env()
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        action = env_control.apply_robot_front_blocking_stop_if_needed(
            env,
            np.array([0.3, 0.0, 0.2], dtype=np.float32),
            world_frame,
        )

        self.assertTrue(np.allclose(action, np.zeros(3, dtype=np.float32)))
        self.assertEqual(env.robot.mode, RobotMode.STOP)
        self.assertEqual(env.robot_front_blocking_state.blocker_idx, 0)
        self.assertEqual(env.robot_front_blocking_state.speech_steps_remaining, 20)
        self.assertFalse(env.robot_front_blocking_state.bypass_active)

    def test_front_blocking_resets_when_blocker_disappears_after_wait(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[1.2, 0.0]])
        base_action = np.array([0.3, 0.0, 0.1], dtype=np.float32)

        action = env_control.apply_robot_front_blocking_stop_if_needed(env, base_action, world_frame)

        self.assertTrue(np.allclose(action, base_action))
        self.assertIsNone(env.robot_front_blocking_state.blocker_idx)
        self.assertEqual(env.robot_front_blocking_state.speech_steps_remaining, 0)
        self.assertFalse(env.robot_front_blocking_state.bypass_active)

    def test_front_blocking_starts_with_in_place_turn_when_blocker_persists(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        action = env_control.apply_robot_front_blocking_stop_if_needed(
            env,
            np.array([0.3, 0.0, 0.1], dtype=np.float32),
            world_frame,
        )

        self.assertTrue(env.robot_front_blocking_state.bypass_active)
        self.assertLess(env.robot_front_blocking_state.bypass_direction_sign, 0.0)
        self.assertGreater(env.robot_front_blocking_state.bypass_radius, 0.0)
        self.assertIsNotNone(env.robot_front_blocking_state.bypass_turn_target_yaw)
        self.assertTrue(np.allclose(action[:2], np.zeros(2, dtype=np.float32)))
        self.assertNotEqual(float(action[2]), 0.0)

    def test_front_blocking_starts_arc_after_turn_alignment(self):
        env = self._make_env()
        state = env.robot_front_blocking_state
        state.blocker_idx = 0
        state.bypass_active = True
        state.bypass_center_xy = np.array([0.4, 0.1], dtype=np.float32)
        state.bypass_radius = float(np.hypot(0.4, 0.1))
        state.bypass_start_angle = float(np.arctan2(-0.1, -0.4))
        state.bypass_direction_sign = -1.0
        state.bypass_turn_target_yaw = float(
            state.bypass_start_angle + (state.bypass_direction_sign * (0.5 * np.pi))
        )
        world_frame = _make_world_frame((0.0, 0.0, state.bypass_turn_target_yaw), [[0.4, 0.1]])

        action = env_control.apply_robot_front_blocking_stop_if_needed(
            env,
            np.array([0.3, 0.0, 0.1], dtype=np.float32),
            world_frame,
        )

        self.assertIsNone(env.robot_front_blocking_state.bypass_turn_target_yaw)
        self.assertFalse(np.allclose(action[:2], np.zeros(2, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
