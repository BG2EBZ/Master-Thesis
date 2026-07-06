import csv
import json
import os
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

    def test_build_comparison_run_metrics_uses_paired_per_run_values(self):
        metrics = eval_baseline._build_comparison_run_metrics(
            baseline_results=[
                EpisodeResult(-10.0, 100.0, 1, 2, 3, True),
                EpisodeResult(-20.0, 120.0, 0, 4, 5, True),
            ],
            comparison_results=[
                EpisodeResult(-8.0, 90.0, 0, 1, 2, True),
                EpisodeResult(-15.0, 110.0, 2, 3, 4, True),
            ],
            evaluation_seeds=[101, 202],
        )

        self.assertEqual([row["run"] for row in metrics], [1, 2])
        self.assertEqual([row["seed"] for row in metrics], [101, 202])
        self.assertAlmostEqual(metrics[0]["baseline_return"], -10.0, places=6)
        self.assertAlmostEqual(metrics[0]["comparison_return"], -8.0, places=6)
        self.assertAlmostEqual(metrics[1]["baseline_duration_seconds"], 120.0, places=6)
        self.assertAlmostEqual(metrics[1]["comparison_duration_seconds"], 110.0, places=6)
        self.assertEqual(metrics[1]["baseline_impatient_triggers"], 4)
        self.assertEqual(metrics[1]["comparison_distracted_triggers"], 4)

    def test_plot_comparison_metrics_writes_non_empty_png(self):
        metrics = [
            {
                "run": 1,
                "seed": 101,
                "baseline_return": -10.0,
                "comparison_return": -8.0,
                "baseline_duration_seconds": 100.0,
                "comparison_duration_seconds": 90.0,
                "baseline_overwhelmed_triggers": 1,
                "comparison_overwhelmed_triggers": 0,
                "baseline_impatient_triggers": 2,
                "comparison_impatient_triggers": 1,
                "baseline_distracted_triggers": 3,
                "comparison_distracted_triggers": 2,
            },
            {
                "run": 2,
                "seed": 202,
                "baseline_return": -20.0,
                "comparison_return": -15.0,
                "baseline_duration_seconds": 120.0,
                "comparison_duration_seconds": 110.0,
                "baseline_overwhelmed_triggers": 0,
                "comparison_overwhelmed_triggers": 2,
                "baseline_impatient_triggers": 4,
                "comparison_impatient_triggers": 3,
                "baseline_distracted_triggers": 5,
                "comparison_distracted_triggers": 4,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "comparison_plot.png"
            eval_baseline.plot_comparison_metrics(metrics, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_evaluate_baseline_uses_fixed_evaluation_seeds_and_ignores_seed_metadata(self):
        seen_tasks = []
        baseline_theta = PolicySearchParams().to_theta()
        comparison_theta = np.array([2.9, 4.4, 2.2, 0.76], dtype=np.float64)
        best_theta_seen = np.array([2.4, 3.6, 1.8, 0.72], dtype=np.float64)
        heldout_evaluation_seeds = [301, 302, 303, 304, 305, 306]

        def _fake_evaluate_episode_task(task):
            seen_tasks.append(task)
            theta, seed, _n_humans, _print_explanations = task
            if np.allclose(theta, baseline_theta):
                reward_offset = 0.0
                duration_offset = 0.0
                trigger_offset = 0
            elif np.allclose(theta, comparison_theta):
                reward_offset = 5.0
                duration_offset = -10.0
                trigger_offset = 1
            else:
                raise AssertionError("Unexpected theta passed to evaluator")
            return EpisodeResult(
                episode_return=float(-0.1 * seed + reward_offset),
                duration_seconds=float(seed + 100 + duration_offset),
                overwhelmed_triggers=int(seed % 3) + trigger_offset,
                impatient_triggers=int(seed % 4) + trigger_offset,
                distracted_triggers=int(seed % 5) + trigger_offset,
                success=True,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            learned_params_json = Path(tmp_dir) / "best_params.json"
            with learned_params_json.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "master_seed": 77,
                        "final_theta": [float(value) for value in comparison_theta],
                        "best_theta_seen": [float(value) for value in best_theta_seen],
                        "heldout_evaluation_seeds": heldout_evaluation_seeds,
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")
            with patch("eval_baseline.ProcessPoolExecutor", _FakeExecutor):
                with patch(
                    "eval_baseline._evaluate_episode_task",
                    side_effect=_fake_evaluate_episode_task,
                ):
                    metrics = eval_baseline.evaluate_baseline(
                        learned_params_json=learned_params_json,
                        num_runs=4,
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

        self.assertEqual(len(metrics), 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            _FakeExecutor.last_max_workers,
            min(8, len(seen_tasks), os.cpu_count() or 1),
        )
        self.assertEqual(len(_FakeExecutor.instances), 1)
        self.assertEqual(len(_FakeExecutor.instances[0].mapped_task_batches), 1)
        self.assertEqual(len(seen_tasks), 8)
        baseline_tasks = seen_tasks[:4]
        comparison_tasks = seen_tasks[4:]
        self.assertTrue(all(np.allclose(task[0], baseline_theta) for task in baseline_tasks))
        self.assertTrue(all(np.allclose(task[0], comparison_theta) for task in comparison_tasks))
        expected_seeds = eval_baseline.FIXED_EVALUATION_SEEDS[:4]
        self.assertEqual(
            [task[1] for task in baseline_tasks],
            expected_seeds,
        )
        self.assertEqual(
            [task[1] for task in comparison_tasks],
            expected_seeds,
        )
        self.assertTrue(all(task[3] is False for task in seen_tasks))
        self.assertEqual(summary["baseline_theta"], [float(value) for value in baseline_theta])
        self.assertEqual(summary["comparison_theta"], [float(value) for value in comparison_theta])
        self.assertEqual(summary["baseline_policy_params"], {
            "slow_down_distance_m": 3.0,
            "callback_distance_m": 4.0,
            "callback_wait_seconds": 2.0,
            "slowdown_speed_scale": 0.7,
        })
        self.assertEqual(
            summary["comparison_policy_params"],
            eval_baseline._policy_params_dict(comparison_theta),
        )
        self.assertEqual(summary["learned_params_json"], str(learned_params_json))
        self.assertEqual(summary["evaluation_seeds"], expected_seeds)
        self.assertEqual(summary["num_runs"], 4)
        self.assertNotIn("seed", summary)
        self.assertNotIn("master_seed", summary)
        self.assertNotIn("success_rate", summary)
        self.assertEqual([int(row["run"]) for row in rows], [1, 2, 3, 4])
        self.assertEqual([int(row["seed"]) for row in rows], expected_seeds)

        baseline_returns = np.array([-0.1 * task[1] for task in baseline_tasks], dtype=np.float64)
        comparison_returns = baseline_returns + 5.0
        self.assertAlmostEqual(float(rows[0]["baseline_return"]), baseline_returns[0], places=6)
        self.assertAlmostEqual(float(rows[0]["comparison_return"]), comparison_returns[0], places=6)
        self.assertAlmostEqual(
            float(rows[1]["baseline_duration_seconds"]),
            float(expected_seeds[1] + 100),
            places=6,
        )
        self.assertAlmostEqual(
            float(rows[1]["comparison_duration_seconds"]),
            float(expected_seeds[1] + 90),
            places=6,
        )
        self.assertAlmostEqual(
            summary["baseline_mean_return"],
            float(np.mean(baseline_returns)),
            places=6,
        )
        self.assertAlmostEqual(
            summary["comparison_mean_return"],
            float(np.mean(comparison_returns)),
            places=6,
        )
        self.assertAlmostEqual(
            summary["baseline_best_return"],
            float(np.max(baseline_returns)),
            places=6,
        )
        self.assertAlmostEqual(
            summary["comparison_best_return"],
            float(np.max(comparison_returns)),
            places=6,
        )

    def test_evaluate_baseline_falls_back_to_best_theta_seen_with_fixed_seeds(self):
        seen_tasks = []
        baseline_theta = PolicySearchParams().to_theta()
        comparison_theta = np.array([2.6, 4.1, 2.3, 0.74], dtype=np.float64)
        expected_seeds = eval_baseline.FIXED_EVALUATION_SEEDS[:3]

        def _fake_evaluate_episode_task(task):
            seen_tasks.append(task)
            theta, seed, _n_humans, _print_explanations = task
            reward_offset = 0.0 if np.allclose(theta, baseline_theta) else 2.0
            return EpisodeResult(
                episode_return=float(seed + reward_offset),
                duration_seconds=float(seed + 10),
                overwhelmed_triggers=0,
                impatient_triggers=0,
                distracted_triggers=0,
                success=True,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            learned_params_json = Path(tmp_dir) / "legacy_with_master_seed.json"
            with learned_params_json.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "master_seed": 19,
                        "heldout_evaluation_seeds": [901, 902, 903],
                        "best_theta_seen": [float(value) for value in comparison_theta],
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")

            with patch("eval_baseline._evaluate_episode_task", side_effect=_fake_evaluate_episode_task):
                eval_baseline.evaluate_baseline(
                    learned_params_json=learned_params_json,
                    num_runs=3,
                    output_dir=Path(tmp_dir),
                    max_workers=1,
                )

            summary_path = Path(tmp_dir) / eval_baseline.DEFAULT_SUMMARY_NAME
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

        baseline_tasks = seen_tasks[:3]
        comparison_tasks = seen_tasks[3:]
        self.assertTrue(all(np.allclose(task[0], baseline_theta) for task in baseline_tasks))
        self.assertTrue(all(np.allclose(task[0], comparison_theta) for task in comparison_tasks))
        self.assertEqual([task[1] for task in baseline_tasks], expected_seeds)
        self.assertEqual([task[1] for task in comparison_tasks], expected_seeds)
        self.assertEqual(summary["comparison_theta"], [float(value) for value in comparison_theta])
        self.assertEqual(summary["evaluation_seeds"], expected_seeds)
        self.assertNotIn("master_seed", summary)

    def test_evaluate_baseline_uses_fixed_seeds_for_legacy_artifact_without_seed_metadata(self):
        seen_tasks = []
        comparison_theta = np.array([2.8, 4.0, 2.1, 0.75], dtype=np.float64)

        def _fake_evaluate_episode_task(task):
            seen_tasks.append(task)
            theta, seed, _n_humans, _print_explanations = task
            reward_offset = 1.0 if np.allclose(theta, comparison_theta) else 0.0
            return EpisodeResult(
                episode_return=float(seed + reward_offset),
                duration_seconds=float(seed),
                overwhelmed_triggers=0,
                impatient_triggers=0,
                distracted_triggers=0,
                success=True,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            learned_params_json = Path(tmp_dir) / "legacy_without_seed_metadata.json"
            with learned_params_json.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "best_theta_seen": [float(value) for value in comparison_theta],
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")

            with patch("eval_baseline._evaluate_episode_task", side_effect=_fake_evaluate_episode_task):
                eval_baseline.evaluate_baseline(
                    learned_params_json=learned_params_json,
                    num_runs=3,
                    output_dir=Path(tmp_dir),
                    max_workers=1,
                )

            summary_path = Path(tmp_dir) / eval_baseline.DEFAULT_SUMMARY_NAME
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

        expected_seeds = eval_baseline.FIXED_EVALUATION_SEEDS[:3]
        baseline_tasks = seen_tasks[:3]
        comparison_tasks = seen_tasks[3:]
        self.assertEqual([task[1] for task in baseline_tasks], expected_seeds)
        self.assertEqual([task[1] for task in comparison_tasks], expected_seeds)
        self.assertEqual(summary["evaluation_seeds"], expected_seeds)
        self.assertNotIn("master_seed", summary)

    def test_evaluate_baseline_errors_when_num_runs_exceeds_fixed_seed_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            learned_params_json = Path(tmp_dir) / "best_params.json"
            with learned_params_json.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "final_theta": [2.5, 3.5, 2.0, 0.7],
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")

            with self.assertRaisesRegex(ValueError, "fixed evaluation seed count of 20"):
                eval_baseline.evaluate_baseline(
                    learned_params_json=learned_params_json,
                    num_runs=21,
                    output_dir=Path(tmp_dir),
                    max_workers=1,
                )

    def test_main_runs_with_small_smoke_configuration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "eval_baseline.evaluate_baseline",
                return_value=[],
            ) as evaluate_mock:
                exit_code = eval_baseline.main(
                    [
                        "--learned-params-json",
                        str(Path(tmp_dir) / "best_params.json"),
                        "--num-runs",
                        "3",
                        "--output-dir",
                        str(Path(tmp_dir) / "baseline"),
                        "--max-workers",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 0)
        evaluate_mock.assert_called_once()
        kwargs = evaluate_mock.call_args.kwargs
        self.assertEqual(kwargs["learned_params_json"], Path(tmp_dir) / "best_params.json")
        self.assertEqual(kwargs["num_runs"], 3)
        self.assertNotIn("seed", kwargs)
        self.assertEqual(kwargs["max_workers"], 2)


if __name__ == "__main__":
    unittest.main()
