from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.env_constants import HUMAN_GOAL_THRESHOLD
from museum_env.human import (
    HUMAN_ARRIVAL_SLOW_RADIUS,
    HUMAN_ARRIVAL_STOP_RADIUS,
    _compute_arrival_velocity,
)


class HumanArrivalControlTests(unittest.TestCase):
    def test_arrival_velocity_uses_max_speed_outside_slow_radius(self):
        velocity = _compute_arrival_velocity(
            np.array([1.0, 0.0], dtype=np.float32),
            max_speed=1.2,
        )

        np.testing.assert_allclose(velocity, np.array([1.2, 0.0], dtype=np.float32))

    def test_arrival_velocity_scales_inside_slow_radius(self):
        target_distance = (HUMAN_ARRIVAL_STOP_RADIUS + HUMAN_ARRIVAL_SLOW_RADIUS) / 2.0
        velocity = _compute_arrival_velocity(
            np.array([target_distance, 0.0], dtype=np.float32),
            max_speed=1.2,
        )

        np.testing.assert_allclose(velocity, np.array([0.6, 0.0], dtype=np.float32))

    def test_arrival_velocity_stops_inside_stop_radius(self):
        velocity = _compute_arrival_velocity(
            np.array([HUMAN_ARRIVAL_STOP_RADIUS / 2.0, 0.0], dtype=np.float32),
            max_speed=1.2,
        )

        np.testing.assert_allclose(velocity, np.zeros(2, dtype=np.float32))

    def test_goal_threshold_covers_arrival_stop_radius(self):
        self.assertGreaterEqual(HUMAN_GOAL_THRESHOLD, HUMAN_ARRIVAL_STOP_RADIUS)


if __name__ == "__main__":
    unittest.main()
