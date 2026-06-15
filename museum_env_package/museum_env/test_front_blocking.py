import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from museum_env import env_control
from museum_env.env_state import ListeningState, RobotFrontBlockingState
from museum_env.human import HumanMode, HumanProfile
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


class _DummyHuman:
    def __init__(self, *, profile=HumanProfile.NORMAL, mode=HumanMode.FOLLOWING, name="person1"):
        self.profile = profile
        self.mode = mode
        self.name = name

    def set_mode(self, mode: str) -> None:
        self.mode = mode


class FrontBlockingControlTests(unittest.TestCase):
    def _make_env(self, n_humans=1):
        robot = SimpleNamespace(
            mode=RobotMode.MOVE,
            k_v=1.0,
            v_max=1.0,
            k_yaw=2.0,
        )
        listening_state = ListeningState()
        return SimpleNamespace(
            robot=robot,
            robot_front_blocking_state=RobotFrontBlockingState(),
            robot_pass_request_steps=20,
            pass_request_response_profile_probs={
                HumanProfile.NORMAL: 0.5,
                HumanProfile.NEURODIVERGENT: 0.3,
            },
            listening_state=listening_state,
            np_random=np.random.default_rng(0),
            dt=0.1,
            model=None,
            data=None,
            robot_body_id=0,
            humans=[
                _DummyHuman(
                    profile=HumanProfile.NORMAL,
                    mode=HumanMode.FOLLOWING,
                    name=f"person{idx + 1}",
                )
                for idx in range(n_humans)
            ],
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

    def test_pass_request_response_wait_in_progress_does_not_sample(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 1
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 1.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        action = env_control.apply_robot_front_blocking_stop_if_needed(
            env,
            np.array([0.3, 0.0, 0.1], dtype=np.float32),
            world_frame,
        )

        self.assertTrue(np.allclose(action, np.zeros(3, dtype=np.float32)))
        self.assertEqual(env.humans[0].mode, HumanMode.FOLLOWING)
        self.assertEqual(env.robot_front_blocking_state.blocker_idx, 0)
        self.assertEqual(env.robot_front_blocking_state.speech_steps_remaining, 1)

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

    def test_pass_request_success_restores_following_and_skips_bypass(self):
        env = self._make_env()
        env.humans[0].mode = HumanMode.WANDERING
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 1.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])
        base_action = np.array([0.3, 0.0, 0.1], dtype=np.float32)

        action = env_control.apply_robot_front_blocking_stop_if_needed(env, base_action, world_frame)

        self.assertTrue(np.allclose(action, base_action))
        self.assertEqual(env.humans[0].mode, HumanMode.FOLLOWING)
        self.assertIsNone(env.robot_front_blocking_state.blocker_idx)
        self.assertEqual(env.robot_front_blocking_state.speech_steps_remaining, 0)
        self.assertFalse(env.robot_front_blocking_state.bypass_active)

    def test_pass_request_success_restores_listening_when_controller_active(self):
        env = self._make_env()
        env.humans[0].mode = HumanMode.DISTRACTED
        env.listening_state.enter_wait(is_final=False)
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 1.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        action = env_control.apply_robot_front_blocking_stop_if_needed(
            env,
            np.array([0.3, 0.0, 0.1], dtype=np.float32),
            world_frame,
        )

        self.assertTrue(np.allclose(action, np.array([0.3, 0.0, 0.1], dtype=np.float32)))
        self.assertEqual(env.humans[0].mode, HumanMode.LISTENING)
        self.assertIsNone(env.robot_front_blocking_state.blocker_idx)
        self.assertFalse(env.robot_front_blocking_state.bypass_active)

    def test_front_blocking_starts_with_in_place_turn_when_blocker_persists(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 0.0
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

    def test_pass_request_nd_failure_falls_through_to_bypass(self):
        env = self._make_env()
        env.humans[0].profile = HumanProfile.NEURODIVERGENT
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NEURODIVERGENT] = 0.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        action = env_control.apply_robot_front_blocking_stop_if_needed(
            env,
            np.array([0.3, 0.0, 0.1], dtype=np.float32),
            world_frame,
        )

        self.assertTrue(env.robot_front_blocking_state.bypass_active)
        self.assertEqual(env.humans[0].mode, HumanMode.FOLLOWING)
        self.assertTrue(np.allclose(action[:2], np.zeros(2, dtype=np.float32)))
        self.assertNotEqual(float(action[2]), 0.0)

    def test_pass_request_response_helper_is_noop_for_invalid_blocker(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 99

        responded = env_control.apply_pass_request_response_if_needed(env)

        self.assertFalse(responded)
        self.assertEqual(env.humans[0].mode, HumanMode.FOLLOWING)

    def test_front_blocking_prefers_left_side_when_left_ray_is_farther(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 0.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        with patch("museum_env.env_control.raycast_hit_distance", side_effect=[2.0, 0.5]):
            env_control.apply_robot_front_blocking_stop_if_needed(
                env,
                np.array([0.3, 0.0, 0.1], dtype=np.float32),
                world_frame,
            )

        self.assertLess(env.robot_front_blocking_state.bypass_direction_sign, 0.0)

    def test_front_blocking_prefers_right_side_when_right_ray_is_farther(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 0.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        with patch("museum_env.env_control.raycast_hit_distance", side_effect=[0.5, 2.0]):
            env_control.apply_robot_front_blocking_stop_if_needed(
                env,
                np.array([0.3, 0.0, 0.1], dtype=np.float32),
                world_frame,
            )

        self.assertGreater(env.robot_front_blocking_state.bypass_direction_sign, 0.0)

    def test_front_blocking_uses_angle_fallback_when_side_rays_tie(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 0.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        with patch("museum_env.env_control.raycast_hit_distance", side_effect=[1.0, 1.0]):
            env_control.apply_robot_front_blocking_stop_if_needed(
                env,
                np.array([0.3, 0.0, 0.1], dtype=np.float32),
                world_frame,
            )

        self.assertLess(env.robot_front_blocking_state.bypass_direction_sign, 0.0)

    def test_front_blocking_prefers_open_side_when_one_side_has_no_hit(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 0.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        with patch("museum_env.env_control.raycast_hit_distance", side_effect=[None, 0.5]):
            env_control.apply_robot_front_blocking_stop_if_needed(
                env,
                np.array([0.3, 0.0, 0.1], dtype=np.float32),
                world_frame,
            )

        self.assertLess(env.robot_front_blocking_state.bypass_direction_sign, 0.0)

    def test_front_blocking_uses_angle_fallback_when_both_sides_are_open(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 0.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.4, 0.1]])

        with patch("museum_env.env_control.raycast_hit_distance", side_effect=[None, None]):
            env_control.apply_robot_front_blocking_stop_if_needed(
                env,
                np.array([0.3, 0.0, 0.1], dtype=np.float32),
                world_frame,
            )

        self.assertLess(env.robot_front_blocking_state.bypass_direction_sign, 0.0)

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

    def test_front_blocking_rechecks_after_turn_alignment_and_restarts_wait_if_still_blocked(self):
        env = self._make_env(n_humans=2)
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
        aligned_yaw = float(state.bypass_turn_target_yaw)
        ahead_xy = np.array(
            [0.3 * np.cos(aligned_yaw), 0.3 * np.sin(aligned_yaw)],
            dtype=np.float32,
        )
        world_frame = _make_world_frame((0.0, 0.0, aligned_yaw), [[0.4, 0.1], ahead_xy])

        action = env_control.apply_robot_front_blocking_stop_if_needed(
            env,
            np.array([0.3, 0.0, 0.1], dtype=np.float32),
            world_frame,
        )

        self.assertTrue(np.allclose(action, np.zeros(3, dtype=np.float32)))
        self.assertEqual(env.robot.mode, RobotMode.STOP)
        self.assertEqual(env.robot_front_blocking_state.blocker_idx, 1)
        self.assertEqual(env.robot_front_blocking_state.speech_steps_remaining, 20)
        self.assertFalse(env.robot_front_blocking_state.bypass_active)
        self.assertIsNone(env.robot_front_blocking_state.bypass_turn_target_yaw)

    def test_bypass_human_avoidance_offset_is_zero_when_all_humans_are_far(self):
        state = RobotFrontBlockingState(
            bypass_active=True,
            bypass_center_xy=np.array([0.4, 0.1], dtype=np.float32),
            bypass_radius=float(np.hypot(0.4, 0.1)),
            bypass_start_angle=float(np.arctan2(-0.1, -0.4)),
            bypass_direction_sign=-1.0,
        )
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[3.0, 3.0], [-3.0, -3.0]])

        offset = env_control._compute_front_blocking_human_avoidance_offset(
            state,
            world_frame=world_frame,
        )

        self.assertTrue(np.allclose(offset, np.zeros(2, dtype=np.float32)))

    def test_bypass_human_avoidance_offset_deflects_away_from_nearby_human(self):
        state = RobotFrontBlockingState(
            bypass_active=True,
            bypass_center_xy=np.array([0.4, 0.1], dtype=np.float32),
            bypass_radius=float(np.hypot(0.4, 0.1)),
            bypass_start_angle=float(np.arctan2(-0.1, -0.4)),
            bypass_direction_sign=-1.0,
        )
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.25, 0.0]])

        offset = env_control._compute_front_blocking_human_avoidance_offset(
            state,
            world_frame=world_frame,
        )

        self.assertLess(float(offset[0]), 0.0)
        self.assertAlmostEqual(float(offset[1]), 0.0, places=5)

    def test_bypass_human_avoidance_offset_ignores_blocker_when_nearby(self):
        state = RobotFrontBlockingState(
            blocker_idx=0,
            bypass_active=True,
            bypass_center_xy=np.array([0.4, 0.1], dtype=np.float32),
            bypass_radius=float(np.hypot(0.4, 0.1)),
            bypass_start_angle=float(np.arctan2(-0.1, -0.4)),
            bypass_direction_sign=-1.0,
        )
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.25, 0.0]])

        offset = env_control._compute_front_blocking_human_avoidance_offset(
            state,
            world_frame=world_frame,
        )

        self.assertTrue(np.allclose(offset, np.zeros(2, dtype=np.float32)))

    def test_bypass_action_is_deflected_when_nearby_human_blocks_arc_path(self):
        env = self._make_env(n_humans=2)
        state = env.robot_front_blocking_state
        state.bypass_active = True
        state.bypass_center_xy = np.array([0.4, 0.1], dtype=np.float32)
        state.bypass_radius = float(np.hypot(0.4, 0.1))
        state.bypass_start_angle = float(np.arctan2(-0.1, -0.4))
        state.bypass_direction_sign = -1.0
        aligned_yaw = float(state.bypass_start_angle + (state.bypass_direction_sign * (0.5 * np.pi)))
        baseline_frame = _make_world_frame((0.0, 0.0, aligned_yaw), [[3.0, 3.0], [-3.0, -3.0]])
        crowded_frame = _make_world_frame((0.0, 0.0, aligned_yaw), [[0.25, 0.0], [3.0, 3.0]])

        baseline_action = env_control._compute_robot_bypass_action(
            env,
            state=state,
            world_frame=baseline_frame,
        )
        crowded_action = env_control._compute_robot_bypass_action(
            env,
            state=state,
            world_frame=crowded_frame,
        )

        self.assertNotAlmostEqual(float(baseline_action[2]), float(crowded_action[2]), places=5)


if __name__ == "__main__":
    unittest.main()
