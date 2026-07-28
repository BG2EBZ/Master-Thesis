from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.fuzzy import FollowingFuzzyEngine
from museum_env.fuzzy import human_states
from museum_env.human import HumanProfile


class HumanStatesTableDefinitionTests(unittest.TestCase):
    def test_input_specs_match_reference_table(self):
        expected = {
            (HumanProfile.NORMAL, "following"): {
                "following_time": {
                    "short": (0.0, 0.0, 25.0, 30.0),
                    "medium": (25.0, 30.0, 40.0, 45.0),
                    "long": (40.0, 45.0, 120.0, 120.0),
                },
                "hhd": {
                    "close": (0.0, 0.0, 0.6, 0.8),
                    "medium": (0.6, 0.8, 1.3, 1.5),
                    "far": (1.3, 1.5, 4.0, 4.0),
                },
                "hrd": {
                    "close": (0.0, 0.0, 0.8, 1.0),
                    "medium": (0.8, 1.0, 2.0, 2.2),
                    "far": (2.0, 2.2, 5.0, 5.0),
                },
                "density": {
                    "low": (0.0, 0.0, 3.0, 3.0),
                    "medium": (4.0, 4.0, 8.0, 8.0),
                    "crowded": (9.0, 9.0, 12.0, 12.0),
                },
                "angle": {"ahead": (-35.0, -20.0, 20.0, 35.0)},
            },
            (HumanProfile.NORMAL, "listening"): {
                "following_time": {
                    "short": (0.0, 0.0, 17.0, 20.0),
                    "medium": (17.0, 20.0, 27.0, 30.0),
                    "long": (27.0, 30.0, 120.0, 120.0),
                },
                "hhd": {
                    "close": (0.0, 0.0, 0.5, 0.7),
                    "medium": (0.5, 0.7, 1.0, 1.2),
                    "far": (1.0, 1.2, 4.0, 4.0),
                },
                "hrd": {
                    "close": (0.0, 0.0, 0.6, 0.8),
                    "medium": (0.6, 0.8, 1.8, 2.0),
                    "far": (1.8, 2.0, 5.0, 5.0),
                },
                "density": {
                    "low": (0.0, 0.0, 3.0, 3.0),
                    "medium": (4.0, 4.0, 8.0, 8.0),
                    "crowded": (9.0, 9.0, 12.0, 12.0),
                },
                "angle": {"ahead": (-35.0, -20.0, 20.0, 35.0)},
            },
            (HumanProfile.NEURODIVERGENT, "following"): {
                "following_time": {
                    "short": (0.0, 0.0, 20.0, 25.0),
                    "medium": (20.0, 25.0, 30.0, 35.0),
                    "long": (30.0, 35.0, 120.0, 120.0),
                },
                "hhd": {
                    "close": (0.0, 0.0, 0.7, 0.9),
                    "medium": (0.7, 0.9, 1.2, 1.4),
                    "far": (1.2, 1.4, 4.0, 4.0),
                },
                "hrd": {
                    "close": (0.0, 0.0, 1.0, 1.2),
                    "medium": (1.0, 1.2, 1.6, 1.8),
                    "far": (1.6, 1.8, 5.0, 5.0),
                },
                "density": {
                    "low": (0.0, 0.0, 2.0, 2.0),
                    "medium": (3.0, 3.0, 6.0, 6.0),
                    "crowded": (7.0, 7.0, 12.0, 12.0),
                },
                "angle": {"ahead": (-45.0, -30.0, 30.0, 45.0)},
            },
            (HumanProfile.NEURODIVERGENT, "listening"): {
                "following_time": {
                    "short": (0.0, 0.0, 13.0, 16.0),
                    "medium": (13.0, 16.0, 20.0, 23.0),
                    "long": (20.0, 23.0, 120.0, 120.0),
                },
                "hhd": {
                    "close": (0.0, 0.0, 0.6, 0.8),
                    "medium": (0.6, 0.8, 0.9, 1.1),
                    "far": (0.9, 1.1, 4.0, 4.0),
                },
                "hrd": {
                    "close": (0.0, 0.0, 0.8, 1.0),
                    "medium": (0.8, 1.0, 1.4, 1.6),
                    "far": (1.4, 1.6, 5.0, 5.0),
                },
                "density": {
                    "low": (0.0, 0.0, 2.0, 2.0),
                    "medium": (3.0, 3.0, 6.0, 6.0),
                    "crowded": (7.0, 7.0, 12.0, 12.0),
                },
                "angle": {"ahead": (-45.0, -30.0, 30.0, 45.0)},
            },
        }

        for (profile, context), expected_specs in expected.items():
            with self.subTest(profile=profile, context=context):
                actual_specs = human_states._build_input_specs(context, profile)
                actual_points = {
                    var_name: {
                        term_name: spec.points
                        for term_name, spec in term_specs.items()
                    }
                    for var_name, term_specs in actual_specs.items()
                }
                self.assertEqual(actual_points, expected_specs)

    def test_engine_clips_inputs_to_table_universes(self):
        clipped = FollowingFuzzyEngine.clip_inputs(
            following_time=999.0,
            hhd=999.0,
            hrd=999.0,
            density=999.0,
            angle=999.0,
        )

        self.assertEqual(clipped["following_time"], 120.0)
        self.assertEqual(clipped["hhd"], 4.0)
        self.assertEqual(clipped["hrd"], 5.0)
        self.assertEqual(clipped["density"], 12.0)
        self.assertEqual(clipped["angle"], 180.0)

    def test_ahead_region_uses_profile_boundaries(self):
        self.assertTrue(
            human_states.in_ahead_region(
                34.9,
                context="following",
                profile=HumanProfile.NORMAL,
            )
        )
        self.assertFalse(
            human_states.in_ahead_region(
                35.0,
                context="listening",
                profile=HumanProfile.NORMAL,
            )
        )
        self.assertTrue(
            human_states.in_ahead_region(
                44.9,
                context="following",
                profile=HumanProfile.NEURODIVERGENT,
            )
        )
        self.assertFalse(
            human_states.in_ahead_region(
                45.0,
                context="listening",
                profile=HumanProfile.NEURODIVERGENT,
            )
        )


if __name__ == "__main__":
    unittest.main()
