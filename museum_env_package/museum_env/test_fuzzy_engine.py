import unittest
from collections import Counter

import gymnasium as gym
import museum_env.register_env
import numpy as np

from museum_env.fuzzy import human_states


OUTPUT_NAMES = ("overwhelmed", "distracted", "impatient", "engaged", "curiosity")
SYSTEM_KEYS = (
    ("following", "normal"),
    ("following", "neurodivergent"),
    ("listening", "normal"),
    ("listening", "neurodivergent"),
)


class FuzzyEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            human_states.compute_reference(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                context="following",
                profile="normal",
            )
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"scikit-fuzzy reference backend unavailable: {exc}") from exc

    def _assert_output_close(self, actual, reference, *, tolerance: float = 5e-2):
        self.assertEqual(actual["dominant_state"], reference["dominant_state"])
        for output_name in OUTPUT_NAMES:
            self.assertLessEqual(
                abs(float(actual[output_name]) - float(reference[output_name])),
                tolerance,
                msg=f"Output {output_name} drifted too far: {actual[output_name]} vs {reference[output_name]}",
            )

    def test_compute_matches_reference_on_curated_samples(self):
        curated_samples = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 0.5, 0.6, 3.0, 0.0],
                [10.0, 0.7, 0.8, 4.0, 10.0],
                [15.0, 0.9, 1.0, 6.0, 25.0],
                [20.0, 1.0, 1.3, 8.0, 40.0],
                [25.0, 1.2, 1.6, 9.0, -40.0],
                [30.0, 1.4, 2.0, 2.0, 0.0],
                [40.0, 0.3, 0.7, 10.0, 5.0],
                [50.0, 2.0, 2.4, 1.0, 0.0],
                [60.0, 4.0, 4.0, 10.0, 180.0],
                [3.1, 0.56, 1.05, 5.0, 0.0],
                [42.0, 0.55, 0.95, 4.5, 12.0],
            ],
            dtype=np.float64,
        )

        for context, profile in SYSTEM_KEYS:
            for row in curated_samples:
                with self.subTest(context=context, profile=profile, row=row.tolist()):
                    actual = human_states.compute(
                        following_time=float(row[0]),
                        hhd=float(row[1]),
                        hrd=float(row[2]),
                        density=float(row[3]),
                        angle=float(row[4]),
                        context=context,
                        profile=profile,
                    )
                    reference = human_states.compute_reference(
                        following_time=float(row[0]),
                        hhd=float(row[1]),
                        hrd=float(row[2]),
                        density=float(row[3]),
                        angle=float(row[4]),
                        context=context,
                        profile=profile,
                    )
                    self._assert_output_close(actual, reference)

    def test_compute_matches_reference_on_random_samples(self):
        rng = np.random.default_rng(7)
        for context, profile in SYSTEM_KEYS:
            rows = np.empty((32, 5), dtype=np.float64)
            rows[:, 0] = rng.uniform(0.0, 60.0, size=32)
            rows[:, 1] = rng.uniform(0.0, 4.0, size=32)
            rows[:, 2] = rng.uniform(0.0, 4.0, size=32)
            rows[:, 3] = rng.uniform(0.0, 10.0, size=32)
            rows[:, 4] = rng.uniform(-180.0, 180.0, size=32)
            for row in rows:
                with self.subTest(context=context, profile=profile):
                    actual = human_states.compute(
                        following_time=float(row[0]),
                        hhd=float(row[1]),
                        hrd=float(row[2]),
                        density=float(row[3]),
                        angle=float(row[4]),
                        context=context,
                        profile=profile,
                    )
                    reference = human_states.compute_reference(
                        following_time=float(row[0]),
                        hhd=float(row[1]),
                        hrd=float(row[2]),
                        density=float(row[3]),
                        angle=float(row[4]),
                        context=context,
                        profile=profile,
                    )
                    self._assert_output_close(actual, reference)

    def test_compute_batch_matches_scalar_backend(self):
        rows = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [12.5, 0.7, 0.9, 4.0, 15.0],
                [28.0, 1.1, 2.3, 7.0, -5.0],
                [50.0, 2.0, 3.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        for context, profile in SYSTEM_KEYS:
            batched = human_states.compute_batch(rows, context=context, profile=profile)
            scalar = [
                human_states.compute(
                    following_time=float(row[0]),
                    hhd=float(row[1]),
                    hrd=float(row[2]),
                    density=float(row[3]),
                    angle=float(row[4]),
                    context=context,
                    profile=profile,
                )
                for row in rows
            ]
            self.assertEqual(batched, scalar)


class FuzzyEnvironmentRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            human_states.compute_reference(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                context="following",
                profile="normal",
            )
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"scikit-fuzzy reference backend unavailable: {exc}") from exc

    def _run_episode(self, *, seed: int, backend: str, n_humans: int = 3) -> dict:
        env = gym.make("MuseumEnv-v0", render_mode=None, enable_event_logs=False, n_humans=n_humans)
        try:
            env.reset(seed=seed)
            base_env = env.unwrapped
            if backend == "reference":
                base_env.following_fuzzy_engine.compute = (
                    lambda following_time, hhd, hrd, density, angle, context="following", profile="normal": human_states.compute_reference(
                        following_time=following_time,
                        hhd=hhd,
                        hrd=hrd,
                        density=density,
                        angle=angle,
                        context=context,
                        profile=profile,
                    )
                )
            event_counts = Counter()
            terminated = False
            truncated = False
            final_info = None
            while True:
                _, _, terminated, truncated, info = env.step(None)
                final_info = info
                for name, value in info["events"].items():
                    if value:
                        event_counts[name] += 1
                if terminated or truncated:
                    break

            assert final_info is not None
            return {
                "terminated": terminated,
                "truncated": truncated,
                "terminated_reason": final_info["episode"]["terminated_reason"],
                "step": final_info["episode"]["step"],
                "events": dict(event_counts),
                "reward_components": dict(final_info["episode"]["reward_components"]),
                "return": float(final_info["episode"]["return"]),
                "overwhelmed_triggers": int(final_info["episode"]["overwhelmed_triggers"]),
                "impatient_triggers": int(final_info["episode"]["impatient_triggers"]),
                "distracted_triggers": int(final_info["episode"]["distracted_triggers"]),
                "info_keys": tuple(sorted(final_info.keys())),
                "crowd_keys": tuple(sorted(final_info["crowd"].keys())),
                "robot_keys": tuple(sorted(final_info["robot"].keys())),
                "phase_keys": tuple(sorted(final_info["phase"].keys())),
            }
        finally:
            env.close()

    def test_seeded_episode_regressions_match_reference_backend(self):
        for seed in (0, 1):
            with self.subTest(seed=seed):
                fast_summary = self._run_episode(seed=seed, backend="fast")
                reference_summary = self._run_episode(seed=seed, backend="reference")
                self.assertEqual(fast_summary, reference_summary)


if __name__ == "__main__":
    unittest.main()
