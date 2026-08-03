import unittest
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.guide_config import GuideBehaviorConfig
from scripts.eval_baseline import _policy_params_dict as eval_policy_params_dict
from train.rwr.policy_codec import guide_config_to_theta, summarize_theta, theta_to_guide_config


class PolicyCodecTests(unittest.TestCase):
    def test_from_theta_accepts_six_dimensions_and_clips_explanation_scale_and_cooldown(self):
        params = theta_to_guide_config(
            np.array([2.5, 3.5, 2.0, 0.7, 0.5, 45.0], dtype=np.float64)
        )

        self.assertAlmostEqual(params.explanation_time_scale, 0.7, places=7)
        self.assertAlmostEqual(params.explanation_wait_seconds, 42.0, places=7)
        self.assertAlmostEqual(params.callback_same_person_cooldown_seconds, 30.0, places=7)

    def test_from_theta_accepts_legacy_four_dimension_theta(self):
        params = theta_to_guide_config(np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64))

        self.assertAlmostEqual(params.explanation_time_scale, 1.0, places=7)
        self.assertAlmostEqual(params.explanation_wait_seconds, 60.0, places=7)
        self.assertAlmostEqual(params.callback_same_person_cooldown_seconds, 20.0, places=7)

    def test_from_theta_accepts_legacy_five_dimension_theta_with_default_cooldown(self):
        params = theta_to_guide_config(np.array([2.5, 3.5, 2.0, 0.7, 0.85], dtype=np.float64))

        self.assertAlmostEqual(params.explanation_time_scale, 0.85, places=7)
        self.assertAlmostEqual(params.callback_same_person_cooldown_seconds, 20.0, places=7)

    def test_to_theta_outputs_six_dimensions(self):
        params = GuideBehaviorConfig(
            explanation_time_scale=0.85,
            callback_same_person_cooldown_seconds=12.0,
        )
        theta = guide_config_to_theta(params)

        self.assertEqual(theta.shape, (6,))
        self.assertAlmostEqual(float(theta[4]), 0.85, places=7)
        self.assertAlmostEqual(float(theta[5]), 12.0, places=7)

    def test_policy_summary_dicts_include_explanation_and_cooldown_fields(self):
        legacy_theta = np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64)
        learned_theta = np.array([2.5, 3.5, 2.0, 0.7, 0.8, 12.0], dtype=np.float64)

        train_summary = summarize_theta(legacy_theta)
        eval_summary = eval_policy_params_dict(learned_theta)

        self.assertEqual(train_summary["explanation_time_scale"], 1.0)
        self.assertEqual(train_summary["explanation_wait_seconds"], 60.0)
        self.assertEqual(train_summary["callback_same_person_cooldown_seconds"], 20.0)
        self.assertEqual(eval_summary["explanation_time_scale"], 0.8)
        self.assertEqual(eval_summary["explanation_wait_seconds"], 48.0)
        self.assertEqual(eval_summary["callback_same_person_cooldown_seconds"], 12.0)


if __name__ == "__main__":
    unittest.main()
