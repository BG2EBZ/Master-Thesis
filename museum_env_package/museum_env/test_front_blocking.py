import unittest
from types import SimpleNamespace

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

    def test_front_blocking_restarts_pass_request_when_blocker_persists(self):
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

        self.assertTrue(np.allclose(action, np.zeros(3, dtype=np.float32)))
        self.assertEqual(env.robot.mode, RobotMode.STOP)
        self.assertEqual(env.robot_front_blocking_state.blocker_idx, 0)
        self.assertEqual(env.robot_front_blocking_state.speech_steps_remaining, 20)
        self.assertFalse(env.robot_front_blocking_state.bypass_active)

    def test_pass_request_nd_failure_restarts_pass_request(self):
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

        self.assertEqual(env.robot.mode, RobotMode.STOP)
        self.assertEqual(env.humans[0].mode, HumanMode.FOLLOWING)
        self.assertTrue(np.allclose(action, np.zeros(3, dtype=np.float32)))
        self.assertEqual(env.robot_front_blocking_state.blocker_idx, 0)
        self.assertEqual(env.robot_front_blocking_state.speech_steps_remaining, 20)
        self.assertFalse(env.robot_front_blocking_state.bypass_active)

    def test_pass_request_response_helper_is_noop_for_invalid_blocker(self):
        env = self._make_env()
        env.robot_front_blocking_state.blocker_idx = 99

        responded = env_control.apply_pass_request_response_if_needed(env)

        self.assertFalse(responded)
        self.assertEqual(env.humans[0].mode, HumanMode.FOLLOWING)

    def test_front_blocking_retargets_new_nearest_blocker_after_ignored_request(self):
        env = self._make_env(n_humans=2)
        env.robot_front_blocking_state.blocker_idx = 0
        env.robot_front_blocking_state.speech_steps_remaining = 0
        env.pass_request_response_profile_probs[HumanProfile.NORMAL] = 0.0
        world_frame = _make_world_frame((0.0, 0.0, 0.0), [[0.7, 0.0], [0.35, 0.05]])

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


if __name__ == "__main__":
    unittest.main()
