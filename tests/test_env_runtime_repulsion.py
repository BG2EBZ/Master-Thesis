from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.env_runtime import compute_social_repulsion


REFERENCE_DISTANCE = 0.25
REPULSION_DECAY = 0.2
REPULSION_CUTOFF_DISTANCE = 1.5
REPULSION_GAIN = 5.5


def _repulsion_for_distance(distance: float) -> np.ndarray:
    return compute_social_repulsion(
        np.array([[0.0, 0.0], [distance, 0.0]], dtype=np.float32),
        reference_distance=REFERENCE_DISTANCE,
        repulsion_decay=REPULSION_DECAY,
        repulsion_cutoff_distance=REPULSION_CUTOFF_DISTANCE,
        repulsion_gain=REPULSION_GAIN,
    )


class EnvRuntimeRepulsionTests(unittest.TestCase):
    def test_contact_distance_matches_calibrated_gain(self):
        repulsion = _repulsion_for_distance(REFERENCE_DISTANCE)

        self.assertAlmostEqual(float(np.linalg.norm(repulsion[0])), REPULSION_GAIN, places=6)
        np.testing.assert_allclose(repulsion[0], np.array([-REPULSION_GAIN, 0.0]), atol=1e-6)

    def test_repulsion_uses_helbing_exponential_decay(self):
        distance = 0.8
        repulsion = _repulsion_for_distance(distance)
        expected = REPULSION_GAIN * np.exp((REFERENCE_DISTANCE - distance) / REPULSION_DECAY)

        self.assertAlmostEqual(float(np.linalg.norm(repulsion[0])), float(expected), places=6)

    def test_cutoff_zeroes_repulsion_at_and_beyond_cutoff(self):
        at_cutoff = _repulsion_for_distance(REPULSION_CUTOFF_DISTANCE)
        beyond_cutoff = _repulsion_for_distance(REPULSION_CUTOFF_DISTANCE + 0.1)

        np.testing.assert_allclose(at_cutoff, np.zeros((2, 2), dtype=np.float32), atol=1e-7)
        np.testing.assert_allclose(beyond_cutoff, np.zeros((2, 2), dtype=np.float32), atol=1e-7)

    def test_pairwise_repulsion_is_equal_and_opposite(self):
        repulsion = _repulsion_for_distance(0.8)

        np.testing.assert_allclose(repulsion[0], -repulsion[1], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
