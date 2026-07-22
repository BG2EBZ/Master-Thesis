import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train.common.rollout import EpisodeResult
from train.rwr.algorithm import ThetaEvaluation, update_distribution
from train.rwr.defaults import (
    DEFAULT_BEST_PARAMS_NAME,
    DEFAULT_LEARNING_CURVE_MATRIX_CSV_NAME,
    DEFAULT_LEARNING_CURVE_RAW_CSV_NAME,
    DEFAULT_LEARNING_CURVE_SUMMARY_NAME,
    INITIAL_MU,
    LEARNING_CURVE_RAW_FIELDNAMES,
)
from train.rwr.plotting import _build_sparse_epoch_ticks
from train.rwr.training import (
    SingleSeedTrainingResult,
    _train_single_learning_seed,
    train,
    train_across_learning_seeds,
)
from train.rwr.rewarding import EpisodeRewardWeights
from scripts.plot_rwr_learning_curve import replot_learning_curve


class TrainRWREntrypointTests(unittest.TestCase):
    @staticmethod
    def _theta_evaluation(
        mean_return: float,
        duration_seconds: float = 12.0,
        overwhelmed_triggers: float = 1.0,
        impatient_triggers: float = 2.0,
        distracted_triggers: float = 3.0,
    ) -> ThetaEvaluation:
        return ThetaEvaluation(
            mean_return=mean_return,
            mean_duration_seconds=duration_seconds,
            mean_overwhelmed_triggers=overwhelmed_triggers,
            mean_impatient_triggers=impatient_triggers,
            mean_distracted_triggers=distracted_triggers,
        )

    @staticmethod
    def _learning_curve_row(
        learning_seed: int,
        epoch: int,
        mean_return: float,
        duration_seconds: float = 12.0,
        overwhelmed_triggers: float = 1.0,
        impatient_triggers: float = 2.0,
        distracted_triggers: float = 3.0,
    ) -> dict[str, float | int | str]:
        return {
            "policy": "rwr",
            "learning_seed": int(learning_seed),
            "evaluation_seed": -1,
            "epoch": int(epoch),
            "mean_return": float(mean_return),
            "mean_duration_seconds": float(duration_seconds),
            "mean_overwhelmed_triggers": float(overwhelmed_triggers),
            "mean_impatient_triggers": float(impatient_triggers),
            "mean_distracted_triggers": float(distracted_triggers),
        }

    @staticmethod
    def _episode_result(
        episode_return: float,
        duration_seconds: float = 12.0,
        overwhelmed_triggers: int = 1,
        impatient_triggers: int = 2,
        distracted_triggers: int = 3,
    ) -> EpisodeResult:
        return EpisodeResult(
            episode_return=episode_return,
            duration_seconds=duration_seconds,
            overwhelmed_triggers=overwhelmed_triggers,
            impatient_triggers=impatient_triggers,
            distracted_triggers=distracted_triggers,
            success=True,
        )

    def test_train_accepts_custom_reward_config_and_hyperparameters(self):
        reward_config = EpisodeRewardWeights(
            time_penalty_per_second=0.12,
            overwhelmed_trigger_penalty=5.0,
            impatient_trigger_penalty=2.5,
            distracted_trigger_penalty=1.5,
        )
        episode_result = self._episode_result(-10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "train"
            with patch("train.rwr.training._evaluate_episode_task", return_value=episode_result) as task_mock:
                with patch("train.rwr.training.plot_training_metrics"):
                    with patch("train.rwr.training.plot_exploration_metrics"):
                        metrics = train(
                            epochs=1,
                            samples_per_epoch=1,
                            seed=123,
                            output_dir=output_dir,
                            beta=0.2,
                            train_seeds_per_epoch=1,
                            n_humans=2,
                            reward_config=reward_config,
                        )

            self.assertEqual(len(metrics), 1)
            task = task_mock.call_args.args[0]
            self.assertEqual(len(task), 5)
            self.assertEqual(task[2], 2)
            self.assertEqual(task[4], reward_config)
            self.assertTrue((output_dir / DEFAULT_BEST_PARAMS_NAME).exists())

    def test_single_learning_seed_records_epoch_zero_and_evaluates_current_mu(self):
        evaluation_seeds = [101, 102]
        seen_tasks: list[tuple[np.ndarray, int, int, bool, EpisodeRewardWeights | None]] = []
        side_effect = iter(
            [
                self._episode_result(-30.0),
                self._episode_result(-20.0),
                self._episode_result(-100.0),
                self._episode_result(-50.0),
                self._episode_result(-10.0),
                self._episode_result(-5.0),
            ]
        )

        def fake_evaluate(task):
            seen_tasks.append(task)
            return next(side_effect)

        with patch("train.rwr.training.os.cpu_count", return_value=1):
            with patch("train.rwr.training._evaluate_episode_task", side_effect=fake_evaluate):
                result = _train_single_learning_seed(
                    epochs=1,
                    samples_per_epoch=2,
                    seed=123,
                    beta=0.2,
                    train_seeds_per_epoch=1,
                    n_humans=2,
                    reward_config=None,
                    evaluation_seeds=evaluation_seeds,
                )

        self.assertEqual([row["epoch"] for row in result.learning_curve_rows], [0, 1])
        self.assertEqual([row["policy"] for row in result.learning_curve_rows], ["rwr", "rwr"])
        eval_tasks = [task for task in seen_tasks if task[1] in evaluation_seeds]
        train_tasks = [task for task in seen_tasks if task[1] not in evaluation_seeds]
        self.assertEqual(len(eval_tasks), 4)
        self.assertEqual(len(train_tasks), 2)

        initial_theta = INITIAL_MU.copy()
        np.testing.assert_allclose(eval_tasks[0][0], initial_theta)
        np.testing.assert_allclose(eval_tasks[1][0], initial_theta)

        theta_batch = np.vstack([np.asarray(task[0], dtype=np.float64) for task in train_tasks])
        expected_mu, _expected_std = update_distribution(
            theta_batch=theta_batch,
            returns=np.array([-100.0, -50.0], dtype=np.float64),
            beta=0.2,
        )
        np.testing.assert_allclose(eval_tasks[2][0], expected_mu)
        np.testing.assert_allclose(eval_tasks[3][0], expected_mu)

    def test_train_across_learning_seeds_writes_learning_curve_outputs(self):
        first_result = SingleSeedTrainingResult(
            metrics=[{"epoch": 1, "mean_return": -1.0}],
            best_theta_seen=np.array([1.0], dtype=np.float64),
            best_evaluation=self._theta_evaluation(-1.0),
            best_return_seen=-1.0,
            best_epoch=1,
            best_sample_index=1,
            final_mu=np.array([1.0], dtype=np.float64),
            final_std=np.array([0.1], dtype=np.float64),
            learning_curve_rows=[
                self._learning_curve_row(11, 0, -100.0),
                self._learning_curve_row(11, 1, -60.0),
            ],
        )
        second_result = SingleSeedTrainingResult(
            metrics=[{"epoch": 1, "mean_return": -2.0}],
            best_theta_seen=np.array([2.0], dtype=np.float64),
            best_evaluation=self._theta_evaluation(-2.0),
            best_return_seen=-2.0,
            best_epoch=1,
            best_sample_index=1,
            final_mu=np.array([2.0], dtype=np.float64),
            final_std=np.array([0.2], dtype=np.float64),
            learning_curve_rows=[
                self._learning_curve_row(22, 0, -80.0),
                self._learning_curve_row(22, 1, -40.0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "curve"
            with patch(
                "train.rwr.training._train_single_learning_seed",
                side_effect=[first_result, second_result],
            ):
                with patch(
                    "train.rwr.training._evaluate_baseline_learning_curve",
                    return_value=[
                        self._episode_result(-70.0),
                        self._episode_result(-50.0),
                    ],
                ) as baseline_mock:
                    with patch("train.rwr.training.plot_learning_curve_metrics") as plot_mock:
                        rows = train_across_learning_seeds(
                            epochs=1,
                            samples_per_epoch=1,
                            seed=123,
                            output_dir=output_dir,
                            n_learning_seeds=2,
                            n_eval_seeds=2,
                        )

            self.assertEqual(len(rows), 8)
            raw_csv = output_dir / DEFAULT_LEARNING_CURVE_RAW_CSV_NAME
            matrix_csv = output_dir / DEFAULT_LEARNING_CURVE_MATRIX_CSV_NAME
            summary_json = output_dir / DEFAULT_LEARNING_CURVE_SUMMARY_NAME
            self.assertTrue(raw_csv.exists())
            self.assertTrue(matrix_csv.exists())
            self.assertTrue(summary_json.exists())
            raw_lines = raw_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(raw_lines), 1 + 8)
            self.assertIn("rwr", raw_lines[1])
            self.assertIn("baseline", "\n".join(raw_lines))

            with matrix_csv.open(newline="", encoding="utf-8") as handle:
                matrix_rows = list(csv.DictReader(handle))
            self.assertEqual(
                matrix_rows,
                [
                    {"learning_seed": "11", "epoch_0": "-100.0", "epoch_1": "-60.0"},
                    {"learning_seed": "22", "epoch_0": "-80.0", "epoch_1": "-40.0"},
                ],
            )

            summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["epochs"], [0, 1])
            self.assertEqual(summary_payload["n_learning_seeds"], 2)
            self.assertEqual(summary_payload["n_eval_seeds"], 2)
            self.assertEqual(summary_payload["mean_return"], [-90.0, -50.0])
            np.testing.assert_allclose(summary_payload["baseline_mean_return"], [-60.0, -60.0])
            np.testing.assert_allclose(summary_payload["baseline_ci95_low_return"], [-79.6, -79.6])
            np.testing.assert_allclose(summary_payload["baseline_ci95_high_return"], [-40.4, -40.4])
            self.assertIn("baseline_policy_params", summary_payload)
            self.assertEqual(summary_payload["learning_curve_matrix_csv"], str(matrix_csv))
            baseline_mock.assert_called_once()
            plot_mock.assert_called_once()
            return_matrix = plot_mock.call_args.kwargs["return_matrix"]
            baseline_return_matrix = plot_mock.call_args.kwargs["baseline_return_matrix"]
            self.assertEqual(return_matrix.shape, (2, 2))
            self.assertEqual(baseline_return_matrix.shape, (2, 2))

    def test_learning_curve_plot_uses_sparse_epoch_ticks(self):
        self.assertEqual(
            _build_sparse_epoch_ticks(range(61), max_x_ticks=8),
            [0, 10, 20, 30, 40, 50, 60],
        )

    def test_replot_learning_curve_loads_raw_csv_and_calls_plot(self):
        rows = [
            self._learning_curve_row(11, 0, -100.0),
            self._learning_curve_row(11, 1, -60.0),
            self._learning_curve_row(22, 0, -80.0),
            self._learning_curve_row(22, 1, -40.0),
            {
                "policy": "baseline",
                "learning_seed": -1,
                "evaluation_seed": 101,
                "epoch": 0,
                "mean_return": -70.0,
                "mean_duration_seconds": 12.0,
                "mean_overwhelmed_triggers": 1.0,
                "mean_impatient_triggers": 2.0,
                "mean_distracted_triggers": 3.0,
            },
            {
                "policy": "baseline",
                "learning_seed": -1,
                "evaluation_seed": 101,
                "epoch": 1,
                "mean_return": -70.0,
                "mean_duration_seconds": 12.0,
                "mean_overwhelmed_triggers": 1.0,
                "mean_impatient_triggers": 2.0,
                "mean_distracted_triggers": 3.0,
            },
            {
                "policy": "baseline",
                "learning_seed": -1,
                "evaluation_seed": 102,
                "epoch": 0,
                "mean_return": -50.0,
                "mean_duration_seconds": 12.0,
                "mean_overwhelmed_triggers": 1.0,
                "mean_impatient_triggers": 2.0,
                "mean_distracted_triggers": 3.0,
            },
            {
                "policy": "baseline",
                "learning_seed": -1,
                "evaluation_seed": 102,
                "epoch": 1,
                "mean_return": -50.0,
                "mean_duration_seconds": 12.0,
                "mean_overwhelmed_triggers": 1.0,
                "mean_impatient_triggers": 2.0,
                "mean_distracted_triggers": 3.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / DEFAULT_LEARNING_CURVE_RAW_CSV_NAME
            output_path = Path(tmpdir) / "learning_curve_plot_clean.png"
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_RAW_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            with patch("scripts.plot_rwr_learning_curve.plot_learning_curve_metrics") as plot_mock:
                plot_data = replot_learning_curve(
                    input_csv=input_csv,
                    output_path=output_path,
                    max_x_ticks=5,
                )

        self.assertEqual(plot_data.epochs, [0, 1])
        self.assertEqual(plot_data.learning_seeds, [11, 22])
        self.assertEqual(plot_data.evaluation_seeds, [101, 102])
        np.testing.assert_allclose(
            plot_data.return_matrix,
            np.array([[-100.0, -60.0], [-80.0, -40.0]], dtype=np.float64),
        )
        np.testing.assert_allclose(
            plot_data.baseline_return_matrix,
            np.array([[-70.0, -70.0], [-50.0, -50.0]], dtype=np.float64),
        )
        plot_mock.assert_called_once()
        self.assertEqual(plot_mock.call_args.kwargs["epochs"], [0, 1])
        self.assertEqual(plot_mock.call_args.kwargs["output_path"], output_path)
        self.assertEqual(plot_mock.call_args.kwargs["max_x_ticks"], 5)


if __name__ == "__main__":
    unittest.main()
