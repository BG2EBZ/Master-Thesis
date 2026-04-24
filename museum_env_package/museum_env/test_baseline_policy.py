import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from museum_env.env import MuseumEnv
from museum_env.human import (
    DISTRACTED_SOURCE_FOLLOWING,
    DISTRACTED_SOURCE_LISTENING,
    DISTRACTED_SPEED_SCALE,
    HUMAN_YAW_RATE_GAIN,
    Human,
    HumanMode,
    HumanProfile,
)
from museum_env.map_layouts import AxisAlignedRect, MapLayout
from museum_env.robot import RobotCallbackPhase, RobotMode


class MuseumEnvRefactorTests(unittest.TestCase):
    def _make_env(self, **kwargs):
        defaults = {
            "render_mode": None,
            "enable_event_logs": False,
        }
        defaults.update(kwargs)
        return MuseumEnv(**defaults)

    def _run_until(self, env, predicate, max_steps):
        last_info = None
        for step in range(max_steps):
            _, _, terminated, truncated, info = env.step(None)
            last_info = info
            if predicate(info):
                return step, info
            if terminated or truncated:
                return step, info
        self.fail(f"condition not met within {max_steps} steps; last_info={last_info}")

    def _make_pose_data(self, pose):
        return SimpleNamespace(qpos=np.array(pose, dtype=np.float32))

    def _arm_callback(
        self,
        env,
        *,
        target_idx: int,
        phase=RobotCallbackPhase.CUE,
        attempt_index: int = 1,
        cue_total_steps: int = 1,
        cue_elapsed_steps: int = 0,
        response_sampled: bool = True,
    ):
        target_xy = np.array(env.humans[target_idx].get_pose(env.data)[:2], dtype=np.float32)
        env.callback_state.active_target_idx = int(target_idx)
        env.robot.callback_active = True
        env.robot.callback_target_idx = int(target_idx)
        env.robot.callback_target_xy = target_xy
        env.robot.callback_attempt_index = int(attempt_index)
        env.robot.callback_phase = str(phase)
        env.robot.callback_cue_total_steps = int(cue_total_steps)
        env.robot.callback_cue_elapsed_steps = int(cue_elapsed_steps)
        env.robot.callback_response_sampled = bool(response_sampled)
        env.robot.callback_cue_completed_this_step = False
        env.robot.callback_turn_done = phase != RobotCallbackPhase.TURN
        env.robot.mode = RobotMode.CALLBACK

    def test_reset_contract_and_default_profile(self):
        env = self._make_env(n_humans=5)
        try:
            obs, info = env.reset(seed=10)
            self.assertEqual(obs.shape, (4,))
            self.assertEqual(info, {})
            self.assertEqual(len(env.humans), 5)
            self.assertEqual(env.humans[0].profile, HumanProfile.NEURODIVERGENT)
            for human in env.humans[1:]:
                self.assertEqual(human.profile, HumanProfile.NORMAL)
            self.assertEqual(env.listening_state.phase, "idle")
        finally:
            env.close()

    def test_step_info_uses_compact_contract(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=11)
            _, _, _, _, info = env.step(None)
            self.assertEqual(sorted(info.keys()), ["events", "humans", "robot", "state"])
            self.assertNotIn("metrics", info)
            self.assertNotIn("status", info)
            self.assertEqual(
                sorted(info["state"].keys()),
                [
                    "callback_phase",
                    "follow_phase",
                    "listen_phase",
                    "robot_emotion",
                    "robot_mode",
                    "speaker_active",
                    "step_count",
                    "terminated_reason",
                ],
            )
        finally:
            env.close()

    def test_first_listen_flow_reaches_intro_and_wait(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=12)
            _, info = self._run_until(env, lambda info: info["events"]["entered_listen"], 6000)
            self.assertEqual(info["state"]["listen_phase"], "intro")

            _, info = self._run_until(env, lambda info: info["events"]["started_listen_wait"], 2500)
            self.assertEqual(info["state"]["listen_phase"], "wait")
            self.assertTrue(info["state"]["speaker_active"])
        finally:
            env.close()

    def test_post_explanation_resumes_transit_follow_after_wait_completion(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=13)
            env.listen_intro_delay_steps = 2
            env.listen_wait_steps = 3
            env.robot.v_max = 3.0

            _, info = self._run_until(
                env,
                lambda info: info["events"]["completed_listen_wait"],
                22000,
            )
            self.assertEqual(info["state"]["listen_phase"], "idle")

            _, info = self._run_until(
                env,
                lambda info: info["state"]["follow_phase"] == "transit_follow",
                8000,
            )
            self.assertEqual(info["state"]["listen_phase"], "idle")
        finally:
            env.close()

    def test_callback_starts_from_paused_listening(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=14)
            human = env.humans[0]
            env.listening_state.enter_wait(False)
            env.listening_state.pause()
            env.robot.listen_mode = False
            human.set_mode(HumanMode.DISTRACTED)
            human.distracted_source = DISTRACTED_SOURCE_LISTENING
            human.distracted_recovery_mode = HumanMode.LISTENING

            _, _, _, _, info = env.step(None)

            self.assertTrue(info["events"]["callback_triggered"])
            self.assertTrue(env.robot.callback_active)
            self.assertIsNotNone(info["state"]["callback_phase"])
        finally:
            env.close()

    def test_callback_success_resumes_wait_phase(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=15)
            human = env.humans[0]
            human.set_mode(HumanMode.LISTENING)
            env.listening_state.enter_wait(False)
            env.listening_state.counter = 7
            env.listening_state.pause()
            env.robot.listen_mode = False
            env.callback_state.success_mode = HumanMode.LISTENING
            self._arm_callback(env, target_idx=0)

            _, _, _, _, info = env.step(None)

            self.assertTrue(info["events"]["callback_completed"])
            self.assertTrue(info["events"]["callback_success"])
            self.assertEqual(env.listening_state.phase, "wait")
            self.assertEqual(env.listening_state.counter, 8)
            self.assertTrue(env.robot.listen_mode)
        finally:
            env.close()

    def test_callback_retries_once_when_target_stays_distracted(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=16)
            human = env.humans[0]
            human.set_mode(HumanMode.DISTRACTED)
            human.distracted_recovery_mode = HumanMode.FOLLOWING
            env.callback_state.success_mode = HumanMode.FOLLOWING
            self._arm_callback(env, target_idx=0)

            _, _, _, _, info = env.step(None)

            self.assertFalse(info["events"]["callback_completed"])
            self.assertTrue(env.robot.callback_active)
            self.assertEqual(env.robot.callback_attempt_index, 2)
        finally:
            env.close()

    def test_fuzzy_evaluates_once_per_observation_refresh_cycle(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=17)
            env.follow_phase = "transit_follow"
            env.observation_update_period_steps = 2
            engaged_result = {
                "overwhelmed": 0.1,
                "distracted": 0.1,
                "impatient": 0.1,
                "engaged": 0.9,
                "dominant_state": "engaged",
                "dominant_value": 0.9,
            }

            with patch.object(env.following_fuzzy_engine, "compute", return_value=engaged_result) as mock_compute:
                env.step(None)
                self.assertEqual(mock_compute.call_count, 1)

                env.step(None)
                self.assertEqual(mock_compute.call_count, 1)

                env.step(None)
                self.assertEqual(mock_compute.call_count, 2)
                self.assertEqual(env.fuzzy_debug[0].context, "following")
        finally:
            env.close()

    def test_following_distracted_moves_toward_follow_slot_with_speed_limit(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
        human.distracted_target_xy = np.array([6.0, 6.0], dtype=np.float32)
        human.distracted_target_yaw = float(np.pi / 2.0)

        pose = (6.0, 5.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[6.0, 5.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        with patch.object(
            human,
            "_apply_wall_constraint_to_action",
            side_effect=lambda action, current_xy: np.array(action, dtype=np.float32),
        ):
            action = human.step(None, data, ctx)

        expected_velocity = np.array(
            [-DISTRACTED_SPEED_SCALE * human.max_speed, 0.0],
            dtype=np.float32,
        )
        np.testing.assert_allclose(action[:2], expected_velocity, atol=1e-6)
        self.assertLessEqual(
            float(np.linalg.norm(action[:2])),
            DISTRACTED_SPEED_SCALE * human.max_speed + 1e-6,
        )

    def test_following_distracted_orients_toward_nearest_exhibit_while_moving(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (6.2, 6.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 6.0, 0.0),
            "robot_xy": np.array([5.0, 6.0], dtype=np.float32),
            "human_xy": np.array([[6.2, 6.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        with patch.object(
            human,
            "_apply_wall_constraint_to_action",
            side_effect=lambda action, current_xy: np.array(action, dtype=np.float32),
        ):
            action = human.step(None, data, ctx)

        expected_focus = np.array([9.786, 6.65], dtype=np.float32)
        expected_yaw = float(np.arctan2(expected_focus[1] - pose[1], expected_focus[0] - pose[0]))
        expected_yaw_rate = HUMAN_YAW_RATE_GAIN * expected_yaw

        np.testing.assert_allclose(human.distracted_target_xy, expected_focus, atol=1e-6)
        np.testing.assert_allclose(action[2], expected_yaw_rate, atol=1e-5)
        self.assertLess(float(action[0]), 0.0)

    def test_following_distracted_falls_back_to_synthetic_focus_target(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (6.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 0.0, 0.0),
            "robot_xy": np.array([5.0, 0.0], dtype=np.float32),
            "human_xy": np.array([[6.0, 0.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        with patch.object(
            human,
            "_apply_wall_constraint_to_action",
            side_effect=lambda action, current_xy: np.array(action, dtype=np.float32),
        ):
            action = human.step(None, data, ctx)

        focus_distance = float(
            np.linalg.norm(human.distracted_target_xy - np.array(pose[:2], dtype=np.float32))
        )
        self.assertAlmostEqual(focus_distance, 1.0, places=6)
        self.assertTrue(np.isfinite(action).all())

    def test_post_explanation_yield_uses_behavior_kind_through_step(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.FOLLOWING)

        pose = (2.0, 2.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "behavior_kind": "post_explanation_yield",
            "target_xy": np.array([4.0, 2.0], dtype=np.float32),
            "robot_pose": (1.0, 2.0, 0.0),
            "robot_xy": np.array([1.0, 2.0], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
        }

        with patch.object(
            human,
            "_apply_wall_constraint_to_action",
            side_effect=lambda action, current_xy: np.array(action, dtype=np.float32),
        ):
            action = human.step(None, data, ctx)

        self.assertGreater(float(action[0]), 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, places=6)
        np.testing.assert_allclose(human.current_waypoint, ctx["target_xy"], atol=1e-6)

    def test_post_explanation_listening_anchor_uses_behavior_kind_through_step(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.LISTENING)

        pose = (1.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "behavior_kind": "post_explanation_listening_anchor",
            "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "repulsion": np.zeros(2, dtype=np.float32),
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(70.0),
            "anchor_robot_xy": np.array([0.0, 1.0], dtype=np.float32),
            "anchor_robot_yaw": 0.0,
            "live_robot_xy": np.array([0.0, 0.0], dtype=np.float32),
        }

        with patch.object(
            human,
            "_apply_wall_constraint_to_action",
            side_effect=lambda action, current_xy: np.array(action, dtype=np.float32),
        ):
            action = human.step(None, data, ctx)

        expected_yaw = float(np.arctan2(1.0, -1.0))
        expected_yaw_rate = HUMAN_YAW_RATE_GAIN * expected_yaw
        np.testing.assert_allclose(action[2], expected_yaw_rate, atol=1e-5)

    def test_full_flow_reaches_final_listen_ready(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=18)
            env.listen_intro_delay_steps = 1
            env.listen_wait_steps = 2
            env.robot.v_max = 3.0
            env.max_steps = 60000

            _, info = self._run_until(
                env,
                lambda info: info["events"]["final_listen_ready"],
                30000,
            )

            self.assertEqual(info["state"]["terminated_reason"], "final_listen_ready")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
