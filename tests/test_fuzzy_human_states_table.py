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


TIME_INPUT_NAMES = (
    "following_time",
    "listening_time",
    "total_duration_time",
    "pre_duration_time",
)
UNIFIED_TIME_SPECS = {
    "short": (0.0, 0.0, 20.0, 30.0),
    "medium": (20.0, 30.0, 40.0, 50.0),
    "long": (40.0, 50.0, 120.0, 120.0),
}


class HumanStatesTableDefinitionTests(unittest.TestCase):
    def test_input_specs_match_reference_table(self):
        expected = {
            HumanProfile.NORMAL: {
                "following_time": dict(UNIFIED_TIME_SPECS),
                "listening_time": dict(UNIFIED_TIME_SPECS),
                "total_duration_time": dict(UNIFIED_TIME_SPECS),
                "pre_duration_time": dict(UNIFIED_TIME_SPECS),
                "hhd": {
                    "close": (0.0, 0.0, 0.6, 0.8),
                    "medium": (0.6, 0.8, 1.2, 1.4),
                    "far": (1.2, 1.4, 4.0, 4.0),
                },
                "hrd": {
                    "close": (0.0, 0.0, 0.8, 1.0),
                    "medium": (0.8, 1.0, 2.0, 2.2),
                    "far": (2.0, 2.2, 5.0, 5.0),
                },
                "density": {
                    "low": (0.0, 0.0, 3.0, 3.0),
                    "medium": (4.0, 4.0, 6.0, 6.0),
                    "crowded": (7.0, 7.0, 12.0, 12.0),
                },
                "angle": {"ahead": (-35.0, -30.0, 30.0, 35.0)},
            },
            HumanProfile.NEURODIVERGENT: {
                "following_time": dict(UNIFIED_TIME_SPECS),
                "listening_time": dict(UNIFIED_TIME_SPECS),
                "total_duration_time": dict(UNIFIED_TIME_SPECS),
                "pre_duration_time": dict(UNIFIED_TIME_SPECS),
                "hhd": {
                    "close": (0.0, 0.0, 0.8, 1.0),
                    "medium": (0.8, 1.0, 1.2, 1.4),
                    "far": (1.2, 1.4, 4.0, 4.0),
                },
                "hrd": {
                    "close": (0.0, 0.0, 1.0, 1.2),
                    "medium": (1.0, 1.2, 1.6, 1.8),
                    "far": (1.6, 1.8, 5.0, 5.0),
                },
                "density": {
                    "low": (0.0, 0.0, 2.0, 2.0),
                    "medium": (3.0, 3.0, 4.0, 4.0),
                    "crowded": (5.0, 5.0, 12.0, 12.0),
                },
                "angle": {"ahead": (-45.0, -35.0, 35.0, 45.0)},
            },
        }

        for profile, expected_specs in expected.items():
            for context in ("following", "listening"):
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

    def test_input_specs_are_context_independent_within_profile(self):
        for profile in (HumanProfile.NORMAL, HumanProfile.NEURODIVERGENT):
            with self.subTest(profile=profile):
                following_specs = human_states._build_input_specs("following", profile)
                listening_specs = human_states._build_input_specs("listening", profile)
                self.assertEqual(following_specs, listening_specs)

    def test_time_specs_are_explicit_and_use_unified_placeholder_ranges(self):
        for profile in (HumanProfile.NORMAL, HumanProfile.NEURODIVERGENT):
            with self.subTest(profile=profile):
                actual_specs = human_states._build_input_specs("following", profile)
                self.assertEqual(
                    [key for key in actual_specs.keys() if key in TIME_INPUT_NAMES],
                    list(TIME_INPUT_NAMES),
                )
                for input_name in TIME_INPUT_NAMES:
                    actual_time_specs = {
                        term_name: spec.points
                        for term_name, spec in actual_specs[input_name].items()
                    }
                    self.assertEqual(actual_time_specs, UNIFIED_TIME_SPECS)

    def test_engine_clips_inputs_to_table_universes(self):
        clipped = FollowingFuzzyEngine.clip_inputs(
            following_time=999.0,
            listening_time=999.0,
            total_duration_time=999.0,
            pre_duration_time=999.0,
            hhd=999.0,
            hrd=999.0,
            density=999.0,
            angle=999.0,
        )

        self.assertEqual(clipped["following_time"], 120.0)
        self.assertEqual(clipped["listening_time"], 120.0)
        self.assertEqual(clipped["total_duration_time"], 120.0)
        self.assertEqual(clipped["pre_duration_time"], 120.0)
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
