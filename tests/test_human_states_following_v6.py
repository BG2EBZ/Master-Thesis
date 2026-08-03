from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.fuzzy import human_states
from museum_env.human import HumanProfile


class TestHumanStatesFuzzyBackend(unittest.TestCase):
    def test_batch_inputs_use_four_time_fields_before_distance_fields(self):
        rows = np.array(
            [
                [55.0, 0.0, 80.0, 80.0, 2.0, 3.5, 0.5, 0.0],
                [5.0, 0.0, 80.0, 80.0, 2.0, 3.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        )

        batch_results = human_states.compute_batch(
            rows,
            context="following",
            profile=HumanProfile.NORMAL,
        )
        single_result = human_states.compute(
            following_time=55.0,
            listening_time=0.0,
            total_duration_time=80.0,
            pre_duration_time=80.0,
            hhd=2.0,
            hrd=3.5,
            density=0.5,
            angle=0.0,
            context="following",
            profile=HumanProfile.NORMAL,
        )

        self.assertEqual(batch_results[0], single_result)
        self.assertNotEqual(
            batch_results[0]["dominant_value"],
            batch_results[1]["dominant_value"],
        )

    def test_following_context_uses_following_time_not_listening_time(self):
        base = dict(
            following_time=55.0,
            listening_time=0.0,
            total_duration_time=80.0,
            pre_duration_time=80.0,
            hhd=2.0,
            hrd=3.5,
            density=0.5,
            angle=0.0,
            context="following",
            profile=HumanProfile.NORMAL,
        )
        same_following_with_long_listening = dict(base, listening_time=55.0)
        short_following = dict(base, following_time=5.0)

        self.assertEqual(
            human_states.compute(**base),
            human_states.compute(**same_following_with_long_listening),
        )
        self.assertNotEqual(
            human_states.compute(**base)["dominant_value"],
            human_states.compute(**short_following)["dominant_value"],
        )

    def test_listening_context_uses_listening_time_not_following_time(self):
        base = dict(
            following_time=0.0,
            listening_time=55.0,
            total_duration_time=80.0,
            pre_duration_time=80.0,
            hhd=2.0,
            hrd=3.5,
            density=0.5,
            angle=0.0,
            context="listening",
            profile=HumanProfile.NORMAL,
        )
        same_listening_with_long_following = dict(base, following_time=55.0)
        short_listening = dict(base, listening_time=5.0)

        self.assertEqual(
            human_states.compute(**base),
            human_states.compute(**same_listening_with_long_following),
        )
        self.assertNotEqual(
            human_states.compute(**base)["dominant_value"],
            human_states.compute(**short_listening)["dominant_value"],
        )

    def test_total_duration_time_does_not_change_v1_rules(self):
        base = dict(
            following_time=55.0,
            listening_time=0.0,
            total_duration_time=0.0,
            pre_duration_time=80.0,
            hhd=2.0,
            hrd=3.5,
            density=0.5,
            angle=0.0,
            context="following",
            profile=HumanProfile.NEURODIVERGENT,
        )
        long_episode = dict(base, total_duration_time=120.0)

        self.assertEqual(human_states.compute(**base), human_states.compute(**long_episode))

    def test_pre_duration_time_controls_time_related_impatience_rule(self):
        base = dict(
            following_time=5.0,
            listening_time=0.0,
            total_duration_time=120.0,
            pre_duration_time=0.0,
            hhd=1.0,
            hrd=1.5,
            density=0.5,
            angle=90.0,
            context="following",
            profile=HumanProfile.NORMAL,
        )
        long_pre_duration = dict(base, pre_duration_time=80.0)

        self.assertNotEqual(
            human_states.compute(**base)["dominant_state"],
            "impatient",
        )
        self.assertEqual(
            human_states.compute(**long_pre_duration)["dominant_state"],
            "impatient",
        )

    def test_fast_backend_matches_scikit_reference_for_eight_input_rows(self):
        try:
            import skfuzzy  # noqa: F401
        except ImportError:
            self.skipTest("skfuzzy is not installed")

        rows = np.array(
            [
                [55.0, 0.0, 80.0, 80.0, 2.0, 3.5, 0.5, 0.0],
                [0.0, 35.0, 80.0, 80.0, 0.6, 0.7, 8.0, 0.0],
            ],
            dtype=np.float64,
        )

        for context, row in zip(("following", "listening"), rows):
            with self.subTest(context=context):
                fast = human_states.compute_batch(
                    np.array([row], dtype=np.float64),
                    context=context,
                    profile=HumanProfile.NORMAL,
                )[0]
                reference = human_states.compute_reference_batch(
                    np.array([row], dtype=np.float64),
                    context=context,
                    profile=HumanProfile.NORMAL,
                )[0]

                for key in ("overwhelmed", "distracted", "impatient", "engaged", "curiosity"):
                    self.assertAlmostEqual(float(fast[key]), float(reference[key]), places=3)
                self.assertEqual(fast["dominant_state"], reference["dominant_state"])


if __name__ == "__main__":
    unittest.main()
