import unittest
import warnings
from unittest.mock import patch

import numpy as np

from museum_env.env import MOVE_BACK_SPEED, MuseumEnv
from museum_env.human import HumanMode, HumanProfile


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

    def test_distracted_prob_deprecated_warning_emitted(self):
        env = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            env = self._make_env(
                distracted_prob=0.25,
                impatient_prob=0.0,
                overwhelmed_wait_trigger_prob=0.0,
                attack_wait_trigger_prob=0.0,
            )
        try:
            env.reset(seed=81)
            self.assertTrue(any(w.category is DeprecationWarning for w in caught))
            self.assertTrue(any("distracted_prob" in str(w.message) for w in caught))
        finally:
            if env is not None:
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
