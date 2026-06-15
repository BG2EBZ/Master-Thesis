import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from museum_env.train_rwr import (
    DEFAULT_CSV_NAME,
    DEFAULT_PLOT_NAME,
    EpisodeResult,
    ThetaEvaluation,
    build_arg_parser,
    build_epoch_metrics,
    evaluate_theta,
    main,
    run_episode,
)


class _FakeEpisodeEnv:
    def __init__(self, steps):
        self._steps = list(steps)
        self._step_index = 0
        self.last_theta = None
        self.last_seed = None

    def set_policy_parameters(self, theta):
        self.last_theta = np.asarray(theta, dtype=np.float64)

    def reset(self, seed):
        self.last_seed = seed
        self._step_index = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, _action):
        step = self._steps[self._step_index]
        self._step_index += 1
        return (
            np.zeros(4, dtype=np.float32),
            step["reward"],
            step["terminated"],
            step["truncated"],
            step["info"],
        )


class _FakeTrainingEnv:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.current_theta = None
        self.current_seed = None

    def set_policy_parameters(self, theta):
        self.current_theta = np.asarray(theta, dtype=np.float64)

    def reset(self, seed):
        self.current_seed = int(seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, _action):
        reward = float(-np.sum((self.current_theta - np.array([2.5, 3.5, 2.0, 0.7])) ** 2))
        reward -= 0.05 * float(self.current_seed)
        success = bool(self.current_seed % 2 == 0)
        duration_seconds = 10.0 + float(self.current_seed) + float(self.current_theta[0])
        overwhelmed = 0 if success else 1
        impatient = int(self.current_seed % 3)
        distracted = int(self.current_seed % 2)
        terminated_reason = "final_listen_ready" if success else "max_steps"
        return (
            np.zeros(4, dtype=np.float32),
            reward,
            success,
            not success,
            {
                "episode": {
                    "step": 1,
                    "terminated_reason": terminated_reason,
                    "duration_seconds": duration_seconds,
                    "overwhelmed_triggers": overwhelmed,
                    "impatient_triggers": impatient,
                    "distracted_triggers": distracted,
                    "return": reward,
                    "reward_components": {},
                }
            },
        )

    def close(self):
        return None


class TrainRwrTests(unittest.TestCase):
    def test_run_episode_extracts_terminal_metrics(self):
        env = _FakeEpisodeEnv(
            steps=[
                {
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {"episode": {"step": 1, "terminated_reason": None}},
                },
                {
                    "reward": -3.5,
                    "terminated": True,
                    "truncated": False,
                    "info": {
                        "episode": {
                            "step": 2,
                            "terminated_reason": "final_listen_ready",
                            "duration_seconds": 12.5,
                            "overwhelmed_triggers": 1,
                            "impatient_triggers": 2,
                            "distracted_triggers": 3,
                            "return": -3.5,
                            "reward_components": {},
                        }
                    },
                },
            ]
        )
        theta = np.array([2.0, 3.0, 1.0, 0.5], dtype=np.float64)

        result = run_episode(env, theta=theta, seed=17)

        self.assertTrue(np.allclose(env.last_theta, theta))
        self.assertEqual(env.last_seed, 17)
        self.assertEqual(
            result,
            EpisodeResult(
                episode_return=-3.5,
                duration_seconds=12.5,
                overwhelmed_triggers=1,
                impatient_triggers=2,
                distracted_triggers=3,
                success=True,
            ),
        )

    def test_evaluate_theta_aggregates_across_seeds(self):
        episode_results = [
            EpisodeResult(-5.0, 10.0, 1, 0, 2, True),
            EpisodeResult(-7.0, 14.0, 0, 1, 1, False),
            EpisodeResult(-4.0, 12.0, 2, 1, 0, True),
        ]

        with patch("museum_env.train_rwr.run_episode", side_effect=episode_results):
            evaluation = evaluate_theta(
                env=object(),
                theta=np.array([1.0, 2.0, 3.0, 0.4], dtype=np.float64),
                seeds=[11, 22, 33],
            )

        self.assertAlmostEqual(evaluation.mean_return, -16.0 / 3.0, places=6)
        self.assertAlmostEqual(evaluation.success_rate, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(evaluation.mean_duration_seconds, 12.0, places=6)
        self.assertAlmostEqual(evaluation.mean_overwhelmed_triggers, 1.0, places=6)
        self.assertAlmostEqual(evaluation.mean_impatient_triggers, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(evaluation.mean_distracted_triggers, 1.0, places=6)

    def test_build_epoch_metrics_uses_fixed_field_names(self):
        metrics = build_epoch_metrics(
            epoch=3,
            evaluations=[
                ThetaEvaluation(-8.0, 0.0, 14.0, 1.0, 2.0, 3.0),
                ThetaEvaluation(-4.0, 1.0, 10.0, 0.0, 1.0, 1.0),
            ],
        )

        self.assertEqual(
            list(metrics.keys()),
            [
                "epoch",
                "mean_return",
                "best_return",
                "success_rate",
                "mean_duration_seconds",
                "mean_overwhelmed_triggers",
                "mean_impatient_triggers",
                "mean_distracted_triggers",
            ],
        )
        self.assertEqual(metrics["epoch"], 3)
        self.assertAlmostEqual(float(metrics["mean_return"]), -6.0, places=6)
        self.assertAlmostEqual(float(metrics["best_return"]), -4.0, places=6)
        self.assertAlmostEqual(float(metrics["success_rate"]), 0.5, places=6)

    def test_build_arg_parser_exposes_minimal_cli_surface(self):
        parser = build_arg_parser()
        public_args = {action.dest for action in parser._actions if action.dest != "help"}

        self.assertEqual(public_args, {"epochs", "samples_per_epoch", "seed", "output_dir"})
        args = parser.parse_args(["--output-dir", "/tmp/rwr_out"])
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.samples_per_epoch, 30)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.output_dir, Path("/tmp/rwr_out"))

    def test_main_writes_csv_and_plot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "artifacts"

            with patch("museum_env.train_rwr.MuseumEnv", _FakeTrainingEnv):
                exit_code = main(
                    [
                        "--epochs",
                        "1",
                        "--samples-per-epoch",
                        "2",
                        "--seed",
                        "7",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            csv_path = output_dir / DEFAULT_CSV_NAME
            plot_path = output_dir / DEFAULT_PLOT_NAME
            self.assertTrue(csv_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertGreater(csv_path.stat().st_size, 0)
            self.assertGreater(plot_path.stat().st_size, 0)

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], "1")
            self.assertIn("mean_return", rows[0])
            self.assertIn("success_rate", rows[0])


if __name__ == "__main__":
    unittest.main()
