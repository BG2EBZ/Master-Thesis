import unittest

from museum_env.env import MuseumEnv
from museum_env.human import HumanMode


class TestSimplifiedTriggerProbabilities(unittest.TestCase):
    def _make_env(self, **kwargs):
        return MuseumEnv(
            render_mode=None,
            enable_event_logs=False,
            strict_action_validation=True,
            **kwargs,
        )

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


if __name__ == "__main__":
    unittest.main()
