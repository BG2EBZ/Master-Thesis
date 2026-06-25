import unittest
from unittest.mock import patch

import numpy as np

from museum_env import env_runtime


class EnvRuntimeTests(unittest.TestCase):
    def test_compute_human_pairwise_geometry_sets_diagonal_without_fill_diagonal(self):
        human_xy = np.array(
            [
                [0.0, 0.0],
                [3.0, 4.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        )

        with patch.object(
            env_runtime.np,
            "fill_diagonal",
            side_effect=AssertionError("np.fill_diagonal should not be used in this hot path"),
        ):
            pairwise_diff, pairwise_dist = env_runtime.compute_human_pairwise_geometry(human_xy)

        self.assertEqual(pairwise_diff.shape, (3, 3, 2))
        self.assertEqual(pairwise_dist.shape, (3, 3))
        self.assertTrue(np.all(np.isinf(np.diag(pairwise_dist))))
        self.assertAlmostEqual(float(pairwise_dist[0, 1]), 5.0)
        self.assertAlmostEqual(float(pairwise_dist[1, 0]), 5.0)
        self.assertAlmostEqual(float(pairwise_dist[0, 2]), 2.0)


if __name__ == "__main__":
    unittest.main()
