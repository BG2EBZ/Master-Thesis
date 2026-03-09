import unittest

import numpy as np

from museum_env.env import MOVE_BACK_SPEED, MuseumEnv
from museum_env.human import HumanMode


class TestSimplifiedTriggerProbabilities(unittest.TestCase):
    def _make_env(self, **kwargs):
        return MuseumEnv(
            render_mode=None,
            enable_event_logs=False,
            strict_action_validation=True,
            **kwargs,
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

    def test_distracted_prob_zero_never_triggers(self):
        env = self._make_env(
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=1)
            for human in env.humans:
                for _ in range(100):
                    self.assertIsNone(human._maybe_trigger_following_variant())
        finally:
            env.close()

    def test_distracted_prob_one_always_triggers(self):
        env = self._make_env(
            distracted_prob=1.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=2)
            for human in env.humans:
                self.assertEqual(human._maybe_trigger_following_variant(), HumanMode.DISTRACTED)
        finally:
            env.close()

    def test_impatient_prob_zero_never_triggers(self):
        env = self._make_env(
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=3)
            for human in env.humans:
                for _ in range(100):
                    self.assertIsNone(human._maybe_trigger_following_variant())
        finally:
            env.close()

    def test_impatient_prob_one_always_triggers(self):
        env = self._make_env(
            distracted_prob=0.0,
            impatient_prob=1.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=4)
            for human in env.humans:
                self.assertEqual(human._maybe_trigger_following_variant(), HumanMode.IMPATIENT)
        finally:
            env.close()

    def test_overwhelmed_only_triggers_in_wait_window(self):
        env = self._make_env(
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=1.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=5)
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
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=1.0,
        )
        try:
            env.reset(seed=6)
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
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=1.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=7)
            env.listen_wait_active = True
            _, _, _, _, info = env.step(None)

            self.assertEqual(env.max_concurrent_overwhelmed, 5)
            self.assertLessEqual(len(info["status"]["active_overwhelmed_indices"]), 5)
            self.assertEqual(len(info["status"]["last_overwhelmed_trigger_indices"]), 5)
        finally:
            env.close()

    def test_fixed_cap_five_for_attack(self):
        env = self._make_env(
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=1.0,
        )
        try:
            env.reset(seed=8)
            env.listen_wait_active = True
            _, _, _, _, info = env.step(None)

            self.assertEqual(env.max_concurrent_attack, 5)
            self.assertLessEqual(len(info["status"]["active_attack_indices"]), 5)
            self.assertEqual(len(info["status"]["last_attack_trigger_indices"]), 5)
        finally:
            env.close()

    def test_reset_step_signature_stable(self):
        env = self._make_env(
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            reset_out = env.reset(seed=9)
            self.assertEqual(len(reset_out), 2)
            step_out = env.step(None)
            self.assertEqual(len(step_out), 5)
        finally:
            env.close()

    def test_random_waypoints_stay_inside_walkable_area(self):
        env = self._make_env(
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=10)
            for human in env.humans:
                for _ in range(200):
                    waypoint = human._random_waypoint()
                    self._assert_in_walkable(human, waypoint)
        finally:
            env.close()

    def test_velocity_projection_keeps_predicted_position_walkable(self):
        env = self._make_env(
            distracted_prob=0.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=0.0,
            attack_wait_trigger_prob=0.0,
        )
        try:
            env.reset(seed=11)
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
            distracted_prob=1.0,
            impatient_prob=0.0,
            overwhelmed_wait_trigger_prob=1.0,
            attack_wait_trigger_prob=1.0,
        )
        try:
            env.reset(seed=12)
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
