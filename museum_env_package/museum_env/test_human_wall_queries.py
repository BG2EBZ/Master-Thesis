import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import museum_env.human as human_mod
from museum_env.human import Human


class HumanWallQueryTests(unittest.TestCase):
    def setUp(self):
        self.human = Human(
            name="person1",
            body_name="person1",
            qpos_idx=0,
            max_speed=1.0,
        )
        self.human.body_id = 0
        self.human._runtime_model = object()
        self.human._runtime_data = object()

    def test_same_step_identical_direction_hits_raycast_once(self):
        self.human._begin_wall_query_step(1)
        with patch("museum_env.human.raycast_hit_distance", return_value=0.25) as raycast_mock:
            first_hit = self.human._raycast_hit_distance(np.array([1.0, 0.0], dtype=np.float32))
            second_hit = self.human._raycast_hit_distance(np.array([1.0, 0.0], dtype=np.float32))

        self.assertEqual(first_hit, 0.25)
        self.assertEqual(second_hit, 0.25)
        self.assertEqual(raycast_mock.call_count, 1)

    def test_same_step_same_direction_different_magnitude_reuses_cache(self):
        self.human._begin_wall_query_step(1)
        with patch("museum_env.human.raycast_hit_distance", return_value=0.4) as raycast_mock:
            first_hit = self.human._raycast_hit_distance(np.array([1.0, 0.0], dtype=np.float32))
            second_hit = self.human._raycast_hit_distance(np.array([3.0, 0.0], dtype=np.float32))

        self.assertEqual(first_hit, 0.4)
        self.assertEqual(second_hit, 0.4)
        self.assertEqual(raycast_mock.call_count, 1)

    def test_cache_resets_when_step_changes(self):
        with patch("museum_env.human.raycast_hit_distance", return_value=0.5) as raycast_mock:
            self.human._begin_wall_query_step(1)
            self.human._raycast_hit_distance(np.array([1.0, 0.0], dtype=np.float32))
            self.human._begin_wall_query_step(2)
            self.human._raycast_hit_distance(np.array([1.0, 0.0], dtype=np.float32))

        self.assertEqual(raycast_mock.call_count, 2)

    def test_direction_cache_key_returns_integer_tuple_and_none_for_near_zero(self):
        cache_key = self.human._direction_cache_key(np.array([3.0, 4.0], dtype=np.float32))

        self.assertEqual(cache_key, (6000, 8000))
        self.assertTrue(all(isinstance(value, int) for value in cache_key))
        self.assertIsNone(
            self.human._direction_cache_key(np.array([1e-8, 0.0], dtype=np.float32))
        )

    def test_short_guide_skips_wall_spacing_raycast(self):
        with patch.object(self.human, "_raycast_hit_distance", side_effect=AssertionError("unexpected raycast")):
            wall_force = self.human._compute_wall_spacing_force(
                np.array([0.02, 0.0], dtype=np.float32)
            )

        np.testing.assert_allclose(wall_force, np.zeros(2, dtype=np.float32))

    def test_small_desired_speed_skips_forward_wall_raycast(self):
        desired_v = np.array([0.02, 0.0], dtype=np.float32)
        with patch.object(self.human, "_raycast_hit_distance", side_effect=AssertionError("unexpected raycast")):
            adjusted_v = self.human._adjust_target_velocity_for_walls(
                guide_xy=np.array([1.0, 0.0], dtype=np.float32),
                desired_v_xy=desired_v,
            )

        np.testing.assert_allclose(adjusted_v, desired_v)

    def test_short_guide_skips_forward_wall_raycast(self):
        desired_v = np.array([0.2, 0.0], dtype=np.float32)
        with patch.object(self.human, "_raycast_hit_distance", side_effect=AssertionError("unexpected raycast")):
            adjusted_v = self.human._adjust_target_velocity_for_walls(
                guide_xy=np.array([0.02, 0.0], dtype=np.float32),
                desired_v_xy=desired_v,
            )

        np.testing.assert_allclose(adjusted_v, desired_v)

    def test_blocked_forward_raycast_still_uses_detour_branch(self):
        with patch.object(human_mod, "WALL_DETOUR_ROTATIONS", ()):
            with patch.object(self.human, "_raycast_hit_distance", return_value=0.1):
                adjusted_v = self.human._adjust_target_velocity_for_walls(
                    guide_xy=np.array([1.0, 0.0], dtype=np.float32),
                    desired_v_xy=np.array([0.2, 0.0], dtype=np.float32),
                )

        np.testing.assert_allclose(adjusted_v, np.zeros(2, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
