import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import eval_baseline
from museum_env.policy_search_params import PolicySearchParams
from train_rwr import EpisodeResult


class _FakeExecutor:
    instances = []
    last_max_workers = None

    def __init__(self, *, max_workers):
        type(self).last_max_workers = int(max_workers)
        self.max_workers = int(max_workers)
        self.mapped_task_batches = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def map(self, func, iterable):
        tasks = list(iterable)
        self.mapped_task_batches.append(tasks)
        return [func(task) for task in tasks]


class EvalBaselineTests(unittest.TestCase):
    def setUp(self):
        _FakeExecutor.instances = []
        _FakeExecutor.last_max_workers = None

    def test_build_run_metrics_uses_per_run_raw_values(self):
        metrics = eval_baseline._build_run_metrics(
            [
                EpisodeResult(-10.0, 100.0, 1, 2, 3, True),
                EpisodeResult(-20.0, 120.0, 0, 4, 5, False),
                EpisodeResult(-5.0, 80.0, 2, 1, 0, True),
            ]
        )

        self.assertEqual([row["epoch"] for row in metrics], [1, 2, 3])
        self.assertAlmostEqual(metrics[0]["mean_return"], -10.0, places=6)
        self.assertAlmostEqual(metrics[1]["mean_return"], -20.0, places=6)
        self.assertAlmostEqual(metrics[2]["mean_return"], -5.0, places=6)
        self.assertAlmostEqual(metrics[2]["best_return"], -5.0, places=6)
        self.assertAlmostEqual(metrics[2]["mean_duration_seconds"], 80.0, places=6)
        self.assertAlmostEqual(metrics[2]["mean_overwhelmed_triggers"], 2.0, places=6)
        self.assertAlmostEqual(metrics[2]["mean_impatient_triggers"], 1.0, places=6)
        self.assertAlmostEqual(metrics[2]["mean_distracted_triggers"], 0.0, places=6)

    def test_plot_baseline_metrics_writes_non_empty_png(self):
        metrics = [
            {
                "epoch": 1,
                "mean_return": -10.0,
                "best_return": -10.0,
                "mean_duration_seconds": 100.0,
                "mean_overwhelmed_triggers": 1.0,
                "mean_impatient_triggers": 2.0,
                "mean_distracted_triggers": 3.0,
            },
            {
                "epoch": 2,
                "mean_return": -20.0,
                "best_return": -20.0,
                "mean_duration_seconds": 120.0,
                "mean_overwhelmed_triggers": 0.0,
                "mean_impatient_triggers": 4.0,
                "mean_distracted_triggers": 5.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "baseline_plot.png"
            eval_baseline.plot_baseline_metrics(metrics, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_evaluate_baseline_uses_default_policy_params_and_records_overall_summary(self):
        seen_tasks = []

        def _fake_evaluate_episode_task(task):
            seen_tasks.append(task)
            _theta, seed, _n_humans, _print_explanations = task
            reward = float(-0.1 * seed)
            success = bool(seed % 2 == 0)
            return EpisodeResult(
                episode_return=reward,
                duration_seconds=float(seed + 100),
                overwhelmed_triggers=int(seed % 3),
                impatient_triggers=int(seed % 4),
                distracted_triggers=int(seed % 5),
                success=success,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("eval_baseline.ProcessPoolExecutor", _FakeExecutor):
                with patch(
                    "eval_baseline._evaluate_episode_task",
                    side_effect=_fake_evaluate_episode_task,
                ):
                    metrics = eval_baseline.evaluate_baseline(
                        num_runs=4,
                        seed=7,
                        output_dir=Path(tmp_dir),
                        max_workers=8,
                    )

            csv_path = Path(tmp_dir) / eval_baseline.DEFAULT_CSV_NAME
            plot_path = Path(tmp_dir) / eval_baseline.DEFAULT_PLOT_NAME
            summary_path = Path(tmp_dir) / eval_baseline.DEFAULT_SUMMARY_NAME
            self.assertTrue(csv_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertGreater(csv_path.stat().st_size, 0)
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertGreater(summary_path.stat().st_size, 0)
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        baseline_theta = PolicySearchParams().to_theta()
        self.assertEqual(len(metrics), 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(_FakeExecutor.last_max_workers, 4)
        self.assertEqual(len(_FakeExecutor.instances), 1)
        self.assertEqual(len(_FakeExecutor.instances[0].mapped_task_batches), 1)
        self.assertEqual(len(seen_tasks), 4)
        self.assertTrue(all(np.allclose(task[0], baseline_theta) for task in seen_tasks))
        self.assertEqual(
            [task[1] for task in seen_tasks],
            eval_baseline._sample_evaluation_seeds(4, 7),
        )
        self.assertTrue(all(task[3] is False for task in seen_tasks))
        self.assertEqual(summary["baseline_theta"], [float(value) for value in baseline_theta])
        self.assertEqual(summary["baseline_policy_params"], {
            "slow_down_distance_m": 3.0,
            "callback_distance_m": 4.0,
            "callback_wait_seconds": 2.0,
            "slowdown_speed_scale": 0.7,
        })
        self.assertEqual(summary["seed"], 7)
        self.assertEqual(summary["evaluation_seeds"], [task[1] for task in seen_tasks])
        self.assertEqual(summary["num_runs"], 4)
        self.assertNotIn("success_rate", summary)
        self.assertEqual([int(row["epoch"]) for row in rows], [1, 2, 3, 4])

        per_run_returns = np.array([-0.1 * task[1] for task in seen_tasks], dtype=np.float64)
        self.assertAlmostEqual(float(rows[0]["mean_return"]), per_run_returns[0], places=6)
        self.assertAlmostEqual(float(rows[1]["mean_return"]), per_run_returns[1], places=6)
        self.assertAlmostEqual(float(rows[2]["best_return"]), per_run_returns[2], places=6)
        self.assertAlmostEqual(summary["mean_return"], float(np.mean(per_run_returns)), places=6)
        self.assertAlmostEqual(summary["best_return"], float(np.max(per_run_returns)), places=6)

    def test_main_runs_with_small_smoke_configuration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "eval_baseline.evaluate_baseline",
                return_value=[],
            ) as evaluate_mock:
                exit_code = eval_baseline.main(
                    [
                        "--num-runs",
                        "3",
                        "--seed",
                        "11",
                        "--output-dir",
                        str(Path(tmp_dir) / "baseline"),
                        "--max-workers",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 0)
        evaluate_mock.assert_called_once()
        kwargs = evaluate_mock.call_args.kwargs
        self.assertEqual(kwargs["num_runs"], 3)
        self.assertEqual(kwargs["seed"], 11)
        self.assertEqual(kwargs["max_workers"], 2)


if __name__ == "__main__":
    unittest.main()
