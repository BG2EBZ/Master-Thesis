import unittest
from unittest.mock import patch

import numpy as np

from museum_env.env import MOVE_BACK_SPEED, MuseumEnv
from museum_env.human import HumanMode, HumanProfile


class _FixedRandom:
    def __init__(self, value):
        self.value = float(value)

    def random(self):
        return self.value


class TestSimplifiedTriggerProbabilities(unittest.TestCase):
    def _make_env(self, **kwargs):
        defaults = {
            "distracted_lambda_max_nd_per_sec": 0.0,
            "distracted_lambda_max_normal_per_sec": 0.0,
        }
        defaults.update(kwargs)
        return MuseumEnv(
            render_mode=None,
            enable_event_logs=False,
            strict_action_validation=True,
            **defaults,
        )

    def test_robot_default_speed_and_move_back_speed(self):
        env = self._make_env()
        try:
            self.assertEqual(env.robot.v_max, 1.0)
            self.assertEqual(MOVE_BACK_SPEED, 0.6)
        finally:
            env.close()

    def _assert_in_walkable(self, human, xy, margin=0.20, tol=1e-5):
        xy_arr = np.array(xy, dtype=np.float32)
        projected = human._project_point_to_walkable(xy_arr, margin)
        self.assertLessEqual(float(np.linalg.norm(xy_arr - projected)), float(tol))

    def test_person1_defaults_to_neurodivergent_profile(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=1)
            self.assertEqual(env.humans[0].profile, HumanProfile.NEURODIVERGENT)
            for human in env.humans[1:]:
                self.assertEqual(human.profile, HumanProfile.NORMAL)
        finally:
            env.close()

    def test_distracted_duration_defaults_to_ten_seconds_for_all_humans(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            expected_steps = max(1, int(round(10.0 / float(env.timestep))))
            for human in env.humans:
                self.assertAlmostEqual(human.max_distracted_duration_seconds, 10.0)
                self.assertEqual(human.distracted_duration, expected_steps)
        finally:
            env.close()

    def test_distracted_duration_stays_fixed_across_resets(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            expected_steps = max(1, int(round(10.0 / float(env.timestep))))
            observed = []
            for seed in (101, 102, 103):
                env.reset(seed=seed)
                durations = tuple(int(human.distracted_duration) for human in env.humans)
                observed.append(durations)
                self.assertEqual(durations, (expected_steps,) * len(env.humans))
            self.assertEqual(len(set(observed)), 1)
        finally:
            env.close()

    def test_distracted_timeout_recovers_exactly_at_configured_max_steps(self):
        env = self._make_env(
            max_distracted_duration_seconds=0.006,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=888)
            human = env.humans[1]
            expected_steps = max(1, int(round(0.006 / float(env.timestep))))
            self.assertEqual(human.distracted_duration, expected_steps)

            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            ctx = {
                "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "stand_threshold": env.listen_stand_threshold,
                "dt": float(env.timestep),
            }

            with patch.object(human, "_step_wandering", return_value=np.zeros(3, dtype=np.float32)), \
                 patch("numpy.random.uniform", return_value=0.0):
                for _ in range(max(0, expected_steps - 1)):
                    human.step(env.model, env.data, ctx)
                    self.assertEqual(human.mode, HumanMode.DISTRACTED)
                human.step(env.model, env.data, ctx)

            self.assertEqual(human.mode, HumanMode.FOLLOWING)
        finally:
            env.close()

    def test_callback_response_defaults_are_profile_specific(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            self.assertAlmostEqual(env.callback_rejoin_prob_normal, 0.60)
            self.assertAlmostEqual(env.callback_stay_prob_normal, 0.25)
            self.assertAlmostEqual(env.callback_ignore_prob_normal, 0.15)
            self.assertAlmostEqual(env.callback_rejoin_prob_nd, 0.35)
            self.assertAlmostEqual(env.callback_stay_prob_nd, 0.40)
            self.assertAlmostEqual(env.callback_ignore_prob_nd, 0.25)

            env.np_random = _FixedRandom(0.59)
            self.assertEqual(env._sample_callback_response(HumanProfile.NORMAL), "rejoin")
            env.np_random = _FixedRandom(0.60)
            self.assertEqual(env._sample_callback_response(HumanProfile.NORMAL), "stay")
            env.np_random = _FixedRandom(0.85)
            self.assertEqual(env._sample_callback_response(HumanProfile.NORMAL), "ignore")

            env.np_random = _FixedRandom(0.34)
            self.assertEqual(env._sample_callback_response(HumanProfile.NEURODIVERGENT), "rejoin")
            env.np_random = _FixedRandom(0.35)
            self.assertEqual(env._sample_callback_response(HumanProfile.NEURODIVERGENT), "stay")
            env.np_random = _FixedRandom(0.75)
            self.assertEqual(env._sample_callback_response(HumanProfile.NEURODIVERGENT), "ignore")
        finally:
            env.close()

    def test_callback_response_can_be_overridden_via_constructor(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
            callback_rejoin_prob_normal=0.10,
            callback_stay_prob_normal=0.20,
            callback_ignore_prob_normal=0.70,
            callback_rejoin_prob_nd=0.80,
            callback_stay_prob_nd=0.10,
            callback_ignore_prob_nd=0.10,
        )
        try:
            self.assertAlmostEqual(env.callback_rejoin_prob_normal, 0.10)
            self.assertAlmostEqual(env.callback_stay_prob_normal, 0.20)
            self.assertAlmostEqual(env.callback_ignore_prob_normal, 0.70)
            self.assertAlmostEqual(env.callback_rejoin_prob_nd, 0.80)
            self.assertAlmostEqual(env.callback_stay_prob_nd, 0.10)
            self.assertAlmostEqual(env.callback_ignore_prob_nd, 0.10)

            env.np_random = _FixedRandom(0.15)
            self.assertEqual(env._sample_callback_response(HumanProfile.NORMAL), "stay")
            env.np_random = _FixedRandom(0.95)
            self.assertEqual(env._sample_callback_response(HumanProfile.NORMAL), "ignore")
            env.np_random = _FixedRandom(0.05)
            self.assertEqual(env._sample_callback_response(HumanProfile.NEURODIVERGENT), "rejoin")
        finally:
            env.close()

    def test_callback_request_not_triggered_when_distance_is_at_or_below_threshold(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=120)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[target_idx].distracted_timer = 0
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.0, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNone(request)
        finally:
            env.close()

    def test_callback_request_triggers_when_distance_exceeds_threshold_even_at_timer_zero(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=121)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[target_idx].distracted_timer = 0
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([2.01, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNotNone(request)
            self.assertEqual(request["target_idx"], target_idx)
            self.assertGreaterEqual(request["hold_steps"], 1)
        finally:
            env.close()

    def test_callback_request_keeps_longest_distracted_priority_under_distance_gate(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=122)
            idx_short = 1
            idx_long = 2
            env.humans[idx_short].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[idx_long].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.humans[idx_short].distracted_timer = 5
            env.humans[idx_long].distracted_timer = 9
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[idx_short] = np.array([2.1, 0.0], dtype=np.float32)
            human_xy[idx_long] = np.array([2.2, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNotNone(request)
            self.assertEqual(request["target_idx"], idx_long)
        finally:
            env.close()

    def test_callback_request_distance_threshold_can_be_overridden_and_is_strict(self):
        env = self._make_env(
            callback_trigger_distance_meters=1.5,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=123)
            self.assertAlmostEqual(env.callback_trigger_distance_meters, 1.5)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)

            human_xy[target_idx] = np.array([1.5, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                boundary_request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNone(boundary_request)

            human_xy[target_idx] = np.array([1.5001, 0.0], dtype=np.float32)
            with patch.object(env, "_is_robot_in_move_stage", return_value=True):
                above_request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNotNone(above_request)
            self.assertEqual(above_request["target_idx"], target_idx)
        finally:
            env.close()

    def test_callback_trigger_distance_constructor_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            self._make_env(
                callback_trigger_distance_meters=0.0,
                impatient_prob=0.0,
                overwhelmed_wait_trigger_prob=0.0,
                attack_wait_trigger_prob=0.0,
            )

    def test_callback_request_is_blocked_when_robot_not_in_move_stage(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=124)
            target_idx = 1
            env.humans[target_idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            human_xy = np.zeros((len(env.humans), 2), dtype=np.float32)
            human_xy[target_idx] = np.array([3.0, 0.0], dtype=np.float32)

            env.robot.listen_mode = True
            request = env._build_callback_request(human_xy=human_xy, robot_pose=(0.0, 0.0, 0.0))
            self.assertIsNone(request)
        finally:
            env.close()

    def test_callback_trigger_distance_exposed_in_info_status(self):
        env = self._make_env(
            callback_trigger_distance_meters=1.75,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=125)
            _, _, _, _, info = env.step(None)
            self.assertAlmostEqual(info["status"]["callback_trigger_distance_meters"], 1.75)
        finally:
            env.close()

    def test_callback_completion_samples_by_target_human_profile(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            for idx in (0, 1):
                with self.subTest(target_index=idx):
                    env.reset(seed=200 + idx)
                    env.callback_active_target_idx = idx
                    env.robot.callback_active = True
                    env.humans[idx].transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")

                    def _robot_step_stub(*_args, **_kwargs):
                        env.robot.callback_active = False
                        return {
                            "action": np.zeros(3, dtype=np.float32),
                            "dist": 1.0,
                            "desired_yaw": 0.0,
                            "actual_yaw": 0.0,
                            "mode": "move",
                            "enter_listen": False,
                            "emotion": str(env.robot.emotion),
                            "speaker_active": bool(env.robot.speaker_active),
                        }

                    with patch.object(env.robot, "step", side_effect=_robot_step_stub), \
                         patch.object(env, "_build_callback_request", return_value=None), \
                         patch.object(env, "_compute_social_repulsion", return_value=[
                             np.zeros(2, dtype=np.float32) for _ in env.humans
                         ]), \
                         patch.object(env, "_update_humans_and_apply_ctrl", return_value=np.zeros((len(env.humans), 3), dtype=np.float32)), \
                         patch.object(env, "_sample_callback_response", return_value="ignore") as callback_sampler:
                        env._step_active_branch(external_action_received=False)
                        callback_sampler.assert_called_once_with(profile=env.humans[idx].profile)
        finally:
            env.close()

    def test_callback_stay_steps_uses_one_sim_second(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=300)
            target_idx = 1
            target_human = env.humans[target_idx]
            target_human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            env.callback_active_target_idx = target_idx
            env.robot.callback_active = True

            def _robot_step_stub(*_args, **_kwargs):
                env.robot.callback_active = False
                return {
                    "action": np.zeros(3, dtype=np.float32),
                    "dist": 1.0,
                    "desired_yaw": 0.0,
                    "actual_yaw": 0.0,
                    "mode": "move",
                    "enter_listen": False,
                    "emotion": str(env.robot.emotion),
                    "speaker_active": bool(env.robot.speaker_active),
                }

            with patch.object(env.robot, "step", side_effect=_robot_step_stub), \
                 patch.object(env, "_build_callback_request", return_value=None), \
                 patch.object(env, "_compute_social_repulsion", return_value=[
                     np.zeros(2, dtype=np.float32) for _ in env.humans
                 ]), \
                 patch.object(env, "_update_humans_and_apply_ctrl", return_value=np.zeros((len(env.humans), 3), dtype=np.float32)), \
                 patch.object(env, "_sample_callback_response", return_value="stay"), \
                 patch.object(target_human, "apply_callback_response", return_value=True) as apply_response:
                env._step_active_branch(external_action_received=False)
                expected_stay_steps = max(1, int(round(1.0 / float(env.timestep))))
                apply_response.assert_called_once_with(response="stay", stay_steps=expected_stay_steps)
        finally:
            env.close()

    def test_callback_stay_rejoin_probability_is_profile_specific(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            test_cases = [
                (1, 0.74, True),   # Normal rejoin
                (1, 0.75, False),  # Normal continue distracted (strict < threshold)
                (0, 0.39, True),   # ND rejoin
                (0, 0.40, False),  # ND continue distracted
            ]
            for i, (idx, rand_value, expect_rejoin) in enumerate(test_cases):
                with self.subTest(target_index=idx, rand_value=rand_value):
                    env.reset(seed=320 + i)
                    human = env.humans[idx]
                    human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
                    human.callback_response_mode = "stay"
                    human.callback_stay_steps_remaining = 1
                    ctx = {
                        "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                        "robot_yaw": 0.0,
                        "repulsion": np.zeros(2, dtype=np.float32),
                        "stand_threshold": env.listen_stand_threshold,
                        "dt": float(env.timestep),
                    }
                    with patch("numpy.random.rand", return_value=rand_value):
                        action = human.step(env.model, env.data, ctx)
                    self.assertTrue(np.allclose(action, np.zeros(3, dtype=np.float32)))
                    if expect_rejoin:
                        self.assertEqual(human.mode, HumanMode.FOLLOWING)
                        self.assertTrue(human.callback_stay_rejoin_this_step)
                    else:
                        self.assertEqual(human.mode, HumanMode.DISTRACTED)
                        self.assertIsNone(human.callback_response_mode)
                        self.assertFalse(human.callback_stay_rejoin_this_step)
        finally:
            env.close()

    def test_callback_stay_continue_returns_to_regular_distracted_branch(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=340)
            human = env.humans[1]
            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            human.callback_response_mode = "stay"
            human.callback_stay_steps_remaining = 1
            ctx = {
                "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
                "robot_yaw": 0.0,
                "repulsion": np.zeros(2, dtype=np.float32),
                "stand_threshold": env.listen_stand_threshold,
                "dt": float(env.timestep),
            }

            with patch("numpy.random.rand", return_value=0.99):
                first_action = human.step(env.model, env.data, ctx)
            self.assertTrue(np.allclose(first_action, np.zeros(3, dtype=np.float32)))
            self.assertEqual(human.mode, HumanMode.DISTRACTED)
            self.assertIsNone(human.callback_response_mode)
            self.assertFalse(human.callback_stay_rejoin_this_step)

            with patch.object(human, "_step_wandering", return_value=np.array([0.12, -0.03, 0.5], dtype=np.float32)) as wander_mock, \
                 patch("numpy.random.uniform", return_value=0.0):
                second_action = human.step(env.model, env.data, ctx)
            wander_mock.assert_called_once()
            self.assertAlmostEqual(float(second_action[0]), 0.06, places=6)
            self.assertAlmostEqual(float(second_action[1]), -0.015, places=6)
            self.assertEqual(human.mode, HumanMode.DISTRACTED)
        finally:
            env.close()

    def test_delayed_callback_rejoin_triggers_happy_once(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=360)
            human = env.humans[1]
            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            human.callback_response_mode = "stay"
            human.callback_stay_steps_remaining = 1
            human.callback_stay_rejoin_probability = 1.0

            _, _, _, _, info_first = env.step(None)
            self.assertTrue(info_first["events"]["callback_forced_recovery"])
            self.assertTrue(info_first["events"]["happy_triggered"])

            _, _, _, _, info_second = env.step(None)
            self.assertFalse(info_second["events"]["callback_forced_recovery"])
            self.assertFalse(info_second["events"]["happy_triggered"])
        finally:
            env.close()

    def test_distracted_lambda_zero_never_triggers_after_threshold(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=2)
            dt = float(env.timestep)
            for human in env.humans:
                human.set_following_distracted_window_active(True)
                human.following_steps = int(np.ceil((human.following_distracted_ramp_start_seconds + 120.0) / dt))
                for _ in range(100):
                    self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
        finally:
            env.close()

    def test_distracted_threshold_is_strict_for_nd_and_normal(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=0.15,
            distracted_lambda_max_normal_per_sec=0.08,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=3)
            dt = float(env.timestep)
            for human in (env.humans[0], env.humans[1]):
                human.set_following_distracted_window_active(True)
                human.following_steps = human._get_distracted_follow_threshold_steps(dt=dt)
                self.assertEqual(human._compute_distracted_follow_step_probability(dt=dt), 0.0)
                with patch("numpy.random.rand", return_value=0.0):
                    self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
        finally:
            env.close()

    def test_distracted_probability_ramps_up_after_threshold(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=0.15,
            distracted_lambda_max_normal_per_sec=0.08,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=4)
            dt = float(env.timestep)
            human = env.humans[1]
            self.assertEqual(human.profile, HumanProfile.NORMAL)
            human.set_following_distracted_window_active(True)
            early_seconds = human.following_distracted_ramp_start_seconds + 0.2 * human.following_distracted_rise_seconds
            late_seconds = human.following_distracted_ramp_start_seconds + 0.8 * human.following_distracted_rise_seconds
            human.following_steps = int(np.ceil(early_seconds / dt))
            p_early = human._compute_distracted_follow_step_probability(dt=dt)
            human.following_steps = int(np.ceil(late_seconds / dt))
            p_late = human._compute_distracted_follow_step_probability(dt=dt)
            self.assertGreater(p_late, p_early)
            self.assertGreater(p_early, 0.0)
            probe = 0.5 * (p_early + p_late)
            with patch("numpy.random.rand", return_value=probe):
                human.following_steps = int(np.ceil(early_seconds / dt))
                self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
                human.following_steps = int(np.ceil(late_seconds / dt))
                self.assertEqual(human._maybe_trigger_following_variant(dt=dt), HumanMode.DISTRACTED)
        finally:
            env.close()

    def test_distracted_step_probability_matches_lambda_max_asymptote(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=0.15,
            distracted_lambda_max_normal_per_sec=0.08,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=44)
            dt = float(env.timestep)
            human = env.humans[0]
            human.set_following_distracted_window_active(True)
            long_follow_seconds = human.following_distracted_ramp_start_seconds + 10.0 * human.following_distracted_rise_seconds
            human.following_steps = int(np.ceil(long_follow_seconds / dt))
            p_step = human._compute_distracted_follow_step_probability(dt=dt)
            expected = float(1.0 - np.exp(-human.following_distracted_lambda_max_per_sec * dt))
            self.assertAlmostEqual(p_step, expected, places=12)
        finally:
            env.close()

    def test_impatient_prob_zero_never_triggers(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=5)
            dt = float(env.timestep)
            for human in env.humans:
                for _ in range(100):
                    self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
        finally:
            env.close()

    def test_impatient_prob_one_always_triggers(self):
        env = self._make_env(
            impatient_prob=1.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=6)
            dt = float(env.timestep)
            for human in env.humans:
                self.assertEqual(human._maybe_trigger_following_variant(dt=dt), HumanMode.IMPATIENT)
        finally:
            env.close()

    def test_distracted_window_inactive_blocks_trigger_before_first_listen_complete(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=100.0,
            distracted_lambda_max_normal_per_sec=100.0,
            distracted_ramp_start_nd_seconds=0.0,
            distracted_ramp_start_normal_seconds=0.0,
            distracted_rise_nd_seconds=1.0,
            distracted_rise_normal_seconds=1.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=7)
            dt = float(env.timestep)
            human = env.humans[0]
            human.following_steps = int(np.ceil(5.0 / dt))
            human.set_following_distracted_window_active(env._is_distracted_follow_window_active())
            self.assertFalse(env._is_distracted_follow_window_active())
            with patch("numpy.random.rand", return_value=0.0):
                self.assertIsNone(human._maybe_trigger_following_variant(dt=dt))
        finally:
            env.close()

    def test_distracted_window_active_during_travel_between_explanations(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=100.0,
            distracted_lambda_max_normal_per_sec=100.0,
            distracted_ramp_start_nd_seconds=0.0,
            distracted_ramp_start_normal_seconds=0.0,
            distracted_rise_nd_seconds=1.0,
            distracted_rise_normal_seconds=1.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=8)
            env.robot.listen_done = True
            env.robot.listen_mode = False
            env.listen_wait_active = False
            dt = float(env.timestep)
            human = env.humans[0]
            human.following_steps = int(np.ceil(5.0 / dt))
            human.set_following_distracted_window_active(env._is_distracted_follow_window_active())
            self.assertTrue(env._is_distracted_follow_window_active())
            with patch("numpy.random.rand", return_value=0.0):
                self.assertEqual(human._maybe_trigger_following_variant(dt=dt), HumanMode.DISTRACTED)
        finally:
            env.close()

    def test_distracted_window_closes_once_final_listening_starts_or_waits(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=9)
            env.robot.listen_done = True
            env.robot.listen_mode = True
            self.assertFalse(env._is_distracted_follow_window_active())
            env.robot.listen_mode = False
            env.listen_wait_active = True
            self.assertFalse(env._is_distracted_follow_window_active())
        finally:
            env.close()

    def test_following_duration_resets_on_env_reset(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=10)
            env.humans[0].following_steps = 123
            env.reset(seed=11)
            self.assertEqual(env.humans[0].following_steps, 0)
        finally:
            env.close()

    def test_following_duration_resets_when_human_leaves_following(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=12)
            human = env.humans[0]
            human.transition_to(HumanMode.FOLLOWING, reason="test_force_following")
            human.following_steps = 123
            human.transition_to(HumanMode.DISTRACTED, reason="test_force_distracted")
            self.assertEqual(human.following_steps, 0)
        finally:
            env.close()

    def test_overwhelmed_only_triggers_in_wait_window(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=1.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=13)
            _, _, _, _, info = env.step(None)
            self.assertFalse(info["events"]["overwhelmed_triggered"])
            self.assertEqual(info["status"]["last_overwhelmed_trigger_indices"], [])

            env.listen_wait_active = True
            _, _, _, _, info = env.step(None)
            self.assertTrue(info["events"]["overwhelmed_triggered"])
            self.assertGreater(len(info["status"]["last_overwhelmed_trigger_indices"]), 0)
        finally:
            env.close()

    def test_attack_only_triggers_in_wait_window(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=1.0,
        )
        try:
            env.reset(seed=14)
            _, _, _, _, info = env.step(None)
            self.assertFalse(info["events"]["attack_triggered"])
            self.assertEqual(info["status"]["last_attack_trigger_indices"], [])

            env.listen_wait_active = True
            _, _, _, _, info = env.step(None)
            self.assertTrue(info["events"]["attack_triggered"])
            self.assertGreater(len(info["status"]["last_attack_trigger_indices"]), 0)
        finally:
            env.close()

    def test_fixed_cap_five_for_overwhelmed(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=1.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=15)
            env.listen_wait_active = True
            _, _, _, _, info = env.step(None)

            self.assertEqual(env.max_concurrent_overwhelmed, 5)
            self.assertLessEqual(len(info["status"]["active_overwhelmed_indices"]), 5)
            self.assertEqual(len(info["status"]["last_overwhelmed_trigger_indices"]), 5)
        finally:
            env.close()

    def test_fixed_cap_five_for_attack(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=1.0,
        )
        try:
            env.reset(seed=16)
            env.listen_wait_active = True
            _, _, _, _, info = env.step(None)

            self.assertEqual(env.max_concurrent_attack, 5)
            self.assertLessEqual(len(info["status"]["active_attack_indices"]), 5)
            self.assertEqual(len(info["status"]["last_attack_trigger_indices"]), 5)
        finally:
            env.close()

    def test_reset_step_signature_stable(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            reset_out = env.reset(seed=17)
            self.assertEqual(len(reset_out), 2)
            step_out = env.step(None)
            self.assertEqual(len(step_out), 5)
        finally:
            env.close()

    def test_random_waypoints_stay_inside_walkable_area(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=18)
            for human in env.humans:
                for _ in range(200):
                    waypoint = human._random_waypoint()
                    self._assert_in_walkable(human, waypoint)
        finally:
            env.close()

    def test_velocity_projection_keeps_predicted_position_walkable(self):
        env = self._make_env(
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=19)
            human = env.humans[0]
            current_xy = np.array([9.79, 5.0], dtype=np.float32)
            raw_v = np.array([1.5, 0.0], dtype=np.float32)
            safe_v = human._constrain_velocity_with_walkable(
                x=float(current_xy[0]),
                y=float(current_xy[1]),
                v_xy=raw_v,
                dt=float(env.timestep),
                margin=0.20,
            )
            next_xy = current_xy + float(env.timestep) * safe_v
            self._assert_in_walkable(human, next_xy)
            self.assertLessEqual(float(np.linalg.norm(safe_v)), float(human.max_speed) + 1e-6)
        finally:
            env.close()

    def test_high_trigger_run_keeps_all_humans_in_walkable_area(self):
        env = self._make_env(
            distracted_lambda_max_nd_per_sec=0.8,
            distracted_lambda_max_normal_per_sec=0.8,
            distracted_ramp_start_nd_seconds=0.0,
            distracted_ramp_start_normal_seconds=0.0,
            distracted_rise_nd_seconds=1.0,
            distracted_rise_normal_seconds=1.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=1.0,
            attack_wait_trigger_prob=1.0,
        )
        try:
            env.reset(seed=20)
            for _ in range(60):
                _, _, terminated, truncated, info = env.step(None)
                for idx, xy in enumerate(info["humans"]["pose_xy"]):
                    self._assert_in_walkable(env.humans[idx], xy, tol=2e-2)
                if terminated or truncated:
                    break

            env.listen_wait_active = True
            for _ in range(120):
                _, _, terminated, truncated, info = env.step(None)
                for idx, xy in enumerate(info["humans"]["pose_xy"]):
                    self._assert_in_walkable(env.humans[idx], xy, tol=2e-2)
                if terminated or truncated:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
