import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train.rwr.algorithm import update_distribution_reps


class RepsDistributionUpdateTest(unittest.TestCase):
    def test_reps_update_is_finite_positive_and_shape_preserving(self):
        theta_batch = np.array(
            [
                [2.1, 3.0, 1.5, 0.65, 0.9],
                [2.8, 3.7, 2.2, 0.75, 0.95],
                [3.0, 4.0, 2.8, 0.85, 0.98],
                [2.4, 3.3, 1.8, 0.7, 0.88],
            ],
            dtype=np.float64,
        )
        returns = np.array([-35.0, -24.0, -28.0, -31.0], dtype=np.float64)

        mu, std = update_distribution_reps(theta_batch, returns, eps=0.1)

        self.assertEqual(mu.shape, (5,))
        self.assertEqual(std.shape, (5,))
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertTrue(np.all(std > 0.0))

    def test_reps_update_is_deterministic_for_fixed_inputs(self):
        theta_batch = np.array(
            [
                [2.2, 3.1, 1.4, 0.62, 0.92],
                [2.7, 3.9, 2.4, 0.82, 0.97],
                [3.2, 4.2, 2.1, 0.78, 0.86],
                [2.5, 3.4, 1.9, 0.72, 0.91],
            ],
            dtype=np.float64,
        )
        returns = np.array([-42.0, -23.0, -27.0, -33.0], dtype=np.float64)

        first_mu, first_std = update_distribution_reps(theta_batch, returns, eps=0.5)
        second_mu, second_std = update_distribution_reps(theta_batch, returns, eps=0.5)

        np.testing.assert_allclose(first_mu, second_mu)
        np.testing.assert_allclose(first_std, second_std)

    def test_reps_rejects_non_positive_eps(self):
        theta_batch = np.ones((3, 5), dtype=np.float64)
        returns = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        with self.assertRaises(ValueError):
            update_distribution_reps(theta_batch, returns, eps=0.0)


if __name__ == "__main__":
    unittest.main()
