import sys
from pathlib import Path
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.policy_search_params import PolicySearchParams
from training.common import policy_params_dict as train_policy_params_dict
from eval_baseline import _policy_params_dict as eval_policy_params_dict


class PolicySearchParamsTests(unittest.TestCase):
    def test_from_theta_accepts_six_dimensions_and_clips_explanation_scale_and_cooldown(self):
        params = PolicySearchParams.from_theta(
            np.array([2.5, 3.5, 2.0, 0.7, 0.5, 45.0], dtype=np.float64)
        )

        self.assertAlmostEqual(params.explanation_time_scale, 0.7, places=7)
        self.assertAlmostEqual(params.explanation_wait_seconds, 21.0, places=7)
        self.assertAlmostEqual(params.callback_same_person_cooldown_seconds, 30.0, places=7)

    def test_from_theta_accepts_legacy_four_dimension_theta(self):
        params = PolicySearchParams.from_theta(np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64))

        self.assertAlmostEqual(params.explanation_time_scale, 1.0, places=7)
        self.assertAlmostEqual(params.explanation_wait_seconds, 30.0, places=7)
        self.assertAlmostEqual(params.callback_same_person_cooldown_seconds, 20.0, places=7)

    def test_from_theta_accepts_legacy_five_dimension_theta_with_default_cooldown(self):
        params = PolicySearchParams.from_theta(np.array([2.5, 3.5, 2.0, 0.7, 0.85], dtype=np.float64))

        self.assertAlmostEqual(params.explanation_time_scale, 0.85, places=7)
        self.assertAlmostEqual(params.callback_same_person_cooldown_seconds, 20.0, places=7)

    def test_to_theta_outputs_six_dimensions(self):
        params = PolicySearchParams(
            explanation_time_scale=0.85,
            callback_same_person_cooldown_seconds=12.0,
        )
        theta = params.to_theta()

        self.assertEqual(theta.shape, (6,))
        self.assertAlmostEqual(float(theta[4]), 0.85, places=7)
        self.assertAlmostEqual(float(theta[5]), 12.0, places=7)

    def test_policy_summary_dicts_include_explanation_and_cooldown_fields(self):
        legacy_theta = np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64)
        learned_theta = np.array([2.5, 3.5, 2.0, 0.7, 0.8, 12.0], dtype=np.float64)

        train_summary = train_policy_params_dict(legacy_theta)
        eval_summary = eval_policy_params_dict(learned_theta)

        self.assertEqual(train_summary["explanation_time_scale"], 1.0)
        self.assertEqual(train_summary["explanation_wait_seconds"], 30.0)
        self.assertEqual(train_summary["callback_same_person_cooldown_seconds"], 20.0)
        self.assertEqual(eval_summary["explanation_time_scale"], 0.8)
        self.assertEqual(eval_summary["explanation_wait_seconds"], 24.0)
        self.assertEqual(eval_summary["callback_same_person_cooldown_seconds"], 12.0)


if __name__ == "__main__":
    unittest.main()
