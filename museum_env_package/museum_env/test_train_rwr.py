import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from museum_env.policy_search_params import PolicySearchParams
from train_rwr import (
    DEFAULT_BETA,
    DEFAULT_BEST_PARAMS_NAME,
    DEFAULT_CSV_NAME,
    DEFAULT_EVALUATION_SEEDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOT_NAME,
    EpisodeResult,
    ThetaEvaluation,
    _aggregate_episode_results,
    _close_cached_env,
    _evaluate_episode_task,
    _round_json_floats,
    main,
    plot_training_metrics,
    run_episode,
    train,
    update_distribution,
)


class _FakeEpisodeEnv:
    def __init__(self, steps):
        self._steps = list(steps)
        self._step_index = 0
        self.last_theta = None
        self.last_seed = None
        self.step_count = 0
        self.dt = 0.002

    def set_policy_parameters(self, theta):
        self.last_theta = np.asarray(theta, dtype=np.float64)

    def reset(self, seed):
        self.last_seed = seed
        self._step_index = 0
        self.step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, _action):
        step = self._steps[self._step_index]
        self._step_index += 1
        self.step_count += 1
        return (
            np.zeros(4, dtype=np.float32),
            step["reward"],
            step["terminated"],
            step["truncated"],
            step["info"],
        )


class _FakeTrainingEnv:
    instances = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.current_theta = None
        self.current_seed = None
        self.closed = False
        self.step_count = 0
        self.dt = 0.002
        type(self).instances.append(self)

    def set_policy_parameters(self, theta):
        self.current_theta = np.asarray(theta, dtype=np.float64)

    def reset(self, seed):
        self.current_seed = int(seed)
        self.step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, _action):
        self.step_count += 1
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
                "events": {},
                "episode": {
                    "step": 1,
                    "terminated_reason": terminated_reason,
                    "duration_seconds": duration_seconds,
                    "overwhelmed_triggers": overwhelmed,
                    "impatient_triggers": impatient,
                    "distracted_triggers": distracted,
                    "return": reward,
                    "reward_components": {},
                },
            },
        )

    def close(self):
        self.closed = True


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


class TrainRwrTests(unittest.TestCase):
    def setUp(self):
        _close_cached_env()
        _FakeTrainingEnv.instances = []
        _FakeExecutor.instances = []
        _FakeExecutor.last_max_workers = None

    def tearDown(self):
        _close_cached_env()

    def test_default_evaluation_seeds_match_repo_default(self):
        self.assertEqual(DEFAULT_EVALUATION_SEEDS, (11, 22, 33))

    def test_run_episode_extracts_terminal_metrics(self):
        env = _FakeEpisodeEnv(
            steps=[
                {
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {
                        "events": {},
                        "episode": {"step": 1, "terminated_reason": None},
                    },
                },
                {
                    "reward": -3.5,
                    "terminated": True,
                    "truncated": False,
                    "info": {
                        "events": {},
                        "episode": {
                            "step": 2,
                            "terminated_reason": "final_listen_ready",
                            "duration_seconds": 12.5,
                            "overwhelmed_triggers": 1,
                            "impatient_triggers": 2,
                            "distracted_triggers": 3,
                            "return": -3.5,
                            "reward_components": {},
                        },
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

    def test_run_episode_prints_timestamped_explanation_progress(self):
        env = _FakeEpisodeEnv(
            steps=[
                {
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {
                        "events": {"started_listen_wait": True},
                        "episode": {"step": 1, "terminated_reason": None},
                    },
                },
                {
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {
                        "events": {"completed_listen_wait": True},
                        "episode": {"step": 2, "terminated_reason": None},
                    },
                },
                {
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {
                        "events": {"started_listen_wait": True},
                        "episode": {"step": 3, "terminated_reason": None},
                    },
                },
                {
                    "reward": -1.0,
                    "terminated": True,
                    "truncated": False,
                    "info": {
                        "events": {
                            "completed_listen_wait": True,
                            "final_listen_ready": True,
                        },
                        "episode": {
                            "step": 4,
                            "terminated_reason": "final_listen_ready",
                            "duration_seconds": 8.0,
                            "overwhelmed_triggers": 0,
                            "impatient_triggers": 0,
                            "distracted_triggers": 0,
                            "return": -1.0,
                            "reward_components": {},
                        },
                    },
                },
            ]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = run_episode(
                env,
                theta=np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64),
                seed=42,
            )

        self.assertTrue(result.success)
        self.assertEqual(
            [line.strip() for line in stdout.getvalue().splitlines() if line.strip()],
            [
                "[t=0.002s step=1] robot start explanation A",
                "[t=0.004s step=2] robot finish explanation A",
                "[t=0.006s step=3] robot start explanation B",
                "[t=0.008s step=4] robot finish explanation B",
            ],
        )

    def test_run_episode_can_suppress_explanation_progress_output(self):
        env = _FakeEpisodeEnv(
            steps=[
                {
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                    "info": {
                        "events": {"started_listen_wait": True},
                        "episode": {"step": 1, "terminated_reason": None},
                    },
                },
                {
                    "reward": -1.0,
                    "terminated": True,
                    "truncated": False,
                    "info": {
                        "events": {"completed_listen_wait": True},
                        "episode": {
                            "step": 2,
                            "terminated_reason": "final_listen_ready",
                            "duration_seconds": 4.0,
                            "overwhelmed_triggers": 0,
                            "impatient_triggers": 0,
                            "distracted_triggers": 0,
                            "return": -1.0,
                            "reward_components": {},
                        },
                    },
                },
            ]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            run_episode(
                env,
                theta=np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64),
                seed=42,
                print_explanations=False,
            )

        self.assertEqual(stdout.getvalue(), "")

    def test_evaluate_episode_task_returns_episode_result_and_cached_env_closes_on_cleanup(self):
        theta = np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64)

        with patch("train_rwr.MuseumEnv", _FakeTrainingEnv):
            result = _evaluate_episode_task((theta, 3, 15, False))
            self.assertEqual(len(_FakeTrainingEnv.instances), 1)
            self.assertFalse(_FakeTrainingEnv.instances[0].closed)
            _close_cached_env()

        self.assertAlmostEqual(result.episode_return, -0.15, places=6)
        self.assertEqual(result.duration_seconds, 15.5)
        self.assertEqual(result.overwhelmed_triggers, 1)
        self.assertEqual(result.impatient_triggers, 0)
        self.assertEqual(result.distracted_triggers, 1)
        self.assertFalse(result.success)
        self.assertTrue(_FakeTrainingEnv.instances[0].closed)

    def test_aggregate_episode_results_combines_multiple_seeds(self):
        evaluation = _aggregate_episode_results(
            [
                EpisodeResult(-0.05, 13.5, 1, 1, 1, False),
                EpisodeResult(-0.10, 14.5, 0, 2, 0, True),
                EpisodeResult(-0.15, 15.5, 1, 0, 1, False),
            ]
        )

        self.assertAlmostEqual(evaluation.mean_return, -0.1, places=6)
        self.assertAlmostEqual(evaluation.mean_duration_seconds, 14.5, places=6)
        self.assertAlmostEqual(evaluation.mean_overwhelmed_triggers, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(evaluation.mean_impatient_triggers, 1.0, places=6)
        self.assertAlmostEqual(evaluation.mean_distracted_triggers, 2.0 / 3.0, places=6)

    def test_evaluate_episode_task_reuses_cached_env_for_multiple_tasks(self):
        theta_a = np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64)
        theta_b = np.array([2.0, 3.0, 1.5, 0.6], dtype=np.float64)

        with patch("train_rwr.MuseumEnv", _FakeTrainingEnv):
            result_a = _evaluate_episode_task((theta_a, 1, 15, False))
            result_b = _evaluate_episode_task((theta_b, 2, 15, False))

        self.assertEqual(len(_FakeTrainingEnv.instances), 1)
        self.assertAlmostEqual(result_a.episode_return, -0.05, places=6)
        self.assertAlmostEqual(result_b.episode_return, -0.86, places=6)
        self.assertFalse(_FakeTrainingEnv.instances[0].closed)

    def test_round_json_floats_rounds_only_float_values(self):
        payload = {
            "nested": [1.23456, np.float64(2.34567), {"value": 3.45678}],
            "count": 4,
            "label": "ok",
            "tuple_data": (5.67891, 6),
        }

        rounded = _round_json_floats(payload)

        self.assertEqual(
            rounded,
            {
                "nested": [1.235, 2.346, {"value": 3.457}],
                "count": 4,
                "label": "ok",
                "tuple_data": [5.679, 6],
            },
        )

    def test_train_parallel_branch_reuses_one_executor_across_all_epochs(self):
        seen_tasks = []

        def _fake_evaluate_episode_task(task):
            seen_tasks.append(task)
            theta, seed, _n_humans, _print_explanations = task
            reward = float(np.sum(theta) - (0.1 * seed))
            return EpisodeResult(
                episode_return=reward,
                duration_seconds=float(seed),
                overwhelmed_triggers=int(seed % 2),
                impatient_triggers=int(seed % 3),
                distracted_triggers=int(seed % 4),
                success=bool(seed % 2 == 0),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("train_rwr.ProcessPoolExecutor", _FakeExecutor):
                with patch("train_rwr.DEFAULT_EVALUATION_SEEDS", (11, 12)):
                    with patch(
                        "train_rwr._evaluate_episode_task",
                        side_effect=_fake_evaluate_episode_task,
                    ):
                        with patch(
                            "train_rwr.update_distribution",
                            wraps=update_distribution,
                        ) as update_mock:
                            metrics = train(
                                epochs=5,
                                samples_per_epoch=4,
                                episodes_per_fit=2,
                                seed=7,
                                output_dir=Path(tmp_dir),
                            )
            best_params_path = Path(tmp_dir) / DEFAULT_BEST_PARAMS_NAME
            self.assertTrue(best_params_path.exists())
            with best_params_path.open("r", encoding="utf-8") as handle:
                best_params = json.load(handle)

        self.assertEqual(update_mock.call_count, 10)
        self.assertEqual(_FakeExecutor.last_max_workers, 4)
        self.assertEqual(len(_FakeExecutor.instances), 1)
        self.assertEqual(
            [len(batch) for batch in _FakeExecutor.instances[0].mapped_task_batches],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
        flattened_expected_tasks = [
            task
            for instance in _FakeExecutor.instances
            for batch in instance.mapped_task_batches
            for task in batch
        ]
        self.assertEqual(len(flattened_expected_tasks), 24)
        self.assertEqual(len(seen_tasks), 24)
        for expected_task, seen_task in zip(flattened_expected_tasks, seen_tasks):
            self.assertTrue(np.allclose(expected_task[0], seen_task[0]))
            self.assertEqual(expected_task[1:], seen_task[1:])

        training_tasks = seen_tasks[:20]
        report_tasks = seen_tasks[20:]
        self.assertTrue(all(task[3] is False for task in seen_tasks))
        self.assertTrue(all(isinstance(task[1], int) for task in seen_tasks))
        self.assertEqual([row["epoch"] for row in metrics], [1, 2, 3, 4, 5])

        training_returns = np.array(
            [float(np.sum(task[0]) - (0.1 * task[1])) for task in training_tasks],
            dtype=np.float64,
        )
        best_flat_idx = int(np.argmax(training_returns))
        best_epoch = (best_flat_idx // 4) + 1
        best_sample = (best_flat_idx % 4) + 1
        self.assertEqual(best_params["best_epoch"], best_epoch)
        self.assertEqual(best_params["best_sample_index_within_epoch"], best_sample)
        self.assertEqual(
            best_params["best_theta_seen"],
            _round_json_floats([float(value) for value in training_tasks[best_flat_idx][0]]),
        )
        self.assertEqual(
            best_params["best_return"],
            _round_json_floats(float(training_returns[best_flat_idx])),
        )

        expected_mu = None
        expected_std = None
        for block_start in range(0, len(training_tasks), 2):
            block_tasks = training_tasks[block_start : block_start + 2]
            block_theta = np.array([task[0] for task in block_tasks], dtype=np.float64)
            block_returns = np.array(
                [float(np.sum(task[0]) - (0.1 * task[1])) for task in block_tasks],
                dtype=np.float64,
            )
            expected_mu, expected_std = update_distribution(
                theta_batch=block_theta,
                returns=block_returns,
                beta=DEFAULT_BETA,
            )

        self.assertIsNotNone(expected_mu)
        self.assertIsNotNone(expected_std)
        self.assertEqual(
            best_params["final_theta"],
            _round_json_floats([float(value) for value in expected_mu]),
        )
        self.assertEqual(
            best_params["final_mu"],
            _round_json_floats([float(value) for value in expected_mu]),
        )
        self.assertEqual(
            best_params["final_std"],
            _round_json_floats([float(value) for value in expected_std]),
        )
        self.assertEqual(len(best_params["final_mu"]), 4)
        self.assertEqual(len(best_params["final_std"]), 4)
        self.assertNotIn("final_mu_policy_params", best_params)

        best_report_tasks = report_tasks[:2]
        final_report_tasks = report_tasks[2:]
        self.assertEqual([task[1] for task in best_report_tasks], [11, 12])
        self.assertEqual([task[1] for task in final_report_tasks], [11, 12])
        self.assertTrue(
            all(
                _round_json_floats([float(value) for value in task[0]]) == best_params["best_theta_seen"]
                for task in best_report_tasks
            )
        )
        self.assertTrue(
            all(
                _round_json_floats([float(value) for value in task[0]]) == best_params["final_theta"]
                for task in final_report_tasks
            )
        )
        self.assertEqual(
            best_params["best_eval_mean_return"],
            _round_json_floats(
                float(np.mean([float(np.sum(task[0]) - (0.1 * task[1])) for task in best_report_tasks]))
            ),
        )
        self.assertEqual(
            best_params["final_eval_mean_return"],
            _round_json_floats(
                float(np.mean([float(np.sum(task[0]) - (0.1 * task[1])) for task in final_report_tasks]))
            ),
        )

    def test_train_handles_partial_fit_batch(self):
        def _fake_evaluate_episode_task(task):
            theta, seed, _n_humans, _print_explanations = task
            return EpisodeResult(
                episode_return=float(np.sum(theta) - (0.1 * seed)),
                duration_seconds=float(seed),
                overwhelmed_triggers=int(seed % 2),
                impatient_triggers=int(seed % 3),
                distracted_triggers=int(seed % 4),
                success=bool(seed % 2 == 0),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("train_rwr.ProcessPoolExecutor", _FakeExecutor):
                with patch("train_rwr.DEFAULT_EVALUATION_SEEDS", (11,)):
                    with patch(
                        "train_rwr._evaluate_episode_task",
                        side_effect=_fake_evaluate_episode_task,
                    ):
                        with patch(
                            "train_rwr.update_distribution",
                            wraps=update_distribution,
                        ) as update_mock:
                            train(
                                epochs=1,
                                samples_per_epoch=3,
                                episodes_per_fit=2,
                                seed=7,
                                output_dir=Path(tmp_dir),
                            )

        self.assertEqual(update_mock.call_count, 2)

    def test_train_samples_next_fit_block_from_updated_distribution(self):
        seen_tasks = []
        initial_mu = np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64)
        first_block_theta = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        class _FakeBlockRng:
            def __init__(self):
                self.normal_calls: list[np.ndarray] = []
                self.integer_calls = 0

            def normal(self, *, loc, scale, size):
                del scale
                loc_array = np.asarray(loc, dtype=np.float64).copy()
                self.normal_calls.append(loc_array)
                if len(self.normal_calls) == 1:
                    return np.array(first_block_theta, dtype=np.float64, copy=True)
                if len(self.normal_calls) == 2:
                    if size != (2, len(initial_mu)):
                        raise AssertionError(f"Unexpected normal() size: {size!r}")
                    return np.tile(loc_array, (size[0], 1))
                raise AssertionError("Unexpected normal() call count")

            def integers(self, low, high=None, size=None):
                del low, high
                self.integer_calls += 1
                if self.integer_calls == 1:
                    return np.array([1, 2], dtype=np.int64)
                if self.integer_calls == 2:
                    return np.array([3, 4], dtype=np.int64)
                raise AssertionError("Unexpected integers() call count")

        fake_rng = _FakeBlockRng()

        def _fake_evaluate_episode_task(task):
            seen_tasks.append(task)
            theta, seed, _n_humans, _print_explanations = task
            return EpisodeResult(
                episode_return=float(theta[0]),
                duration_seconds=float(seed),
                overwhelmed_triggers=0,
                impatient_triggers=0,
                distracted_triggers=0,
                success=True,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("train_rwr.DEFAULT_MAX_WORKERS", 1):
                with patch("train_rwr.DEFAULT_EVALUATION_SEEDS", (11,)):
                    with patch("train_rwr.np.random.default_rng", return_value=fake_rng):
                        with patch(
                            "train_rwr._evaluate_episode_task",
                            side_effect=_fake_evaluate_episode_task,
                        ):
                            train(
                                epochs=1,
                                samples_per_epoch=4,
                                episodes_per_fit=2,
                                seed=7,
                                output_dir=Path(tmp_dir),
                            )

        expected_mu_after_first_block, _expected_std_after_first_block = update_distribution(
            theta_batch=first_block_theta,
            returns=np.array([0.0, 10.0], dtype=np.float64),
            beta=DEFAULT_BETA,
        )
        self.assertEqual(len(fake_rng.normal_calls), 2)
        self.assertTrue(np.allclose(fake_rng.normal_calls[0], initial_mu))
        self.assertTrue(np.allclose(fake_rng.normal_calls[1], expected_mu_after_first_block))

        training_tasks = seen_tasks[:4]
        second_block_tasks = training_tasks[2:4]
        self.assertTrue(
            all(np.allclose(task[0], expected_mu_after_first_block) for task in second_block_tasks)
        )

    def test_train_passes_raw_theta_to_episode_tasks_and_keeps_clipped_policy_artifact(self):
        raw_theta = np.array([0.1, 1.0, 12.0, 2.0], dtype=np.float64)
        seen_tasks = []

        def _fake_evaluate_episode_task(task):
            seen_tasks.append(task)
            return EpisodeResult(
                episode_return=1.0,
                duration_seconds=1.0,
                overwhelmed_triggers=0,
                impatient_triggers=0,
                distracted_triggers=0,
                success=True,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("train_rwr.ProcessPoolExecutor", _FakeExecutor):
                with patch("train_rwr.DEFAULT_EVALUATION_SEEDS", (42,)):
                    with patch("train_rwr.INITIAL_MU", raw_theta.copy()):
                        with patch("train_rwr.INITIAL_STD", np.zeros(4, dtype=np.float64)):
                            with patch(
                                "train_rwr._evaluate_episode_task",
                                side_effect=_fake_evaluate_episode_task,
                            ):
                                train(
                                    epochs=1,
                                    samples_per_epoch=1,
                                    episodes_per_fit=1,
                                    seed=7,
                                    output_dir=Path(tmp_dir),
                                )
            best_params_path = Path(tmp_dir) / DEFAULT_BEST_PARAMS_NAME
            with best_params_path.open("r", encoding="utf-8") as handle:
                best_params = json.load(handle)

        self.assertEqual(len(seen_tasks), 3)
        self.assertTrue(np.allclose(seen_tasks[0][0], raw_theta))
        self.assertTrue(all(np.allclose(task[0], raw_theta) for task in seen_tasks[1:]))
        self.assertEqual(
            best_params["best_theta_seen"],
            _round_json_floats([float(value) for value in raw_theta]),
        )
        self.assertEqual(
            best_params["final_theta"],
            _round_json_floats([float(value) for value in raw_theta]),
        )
        clipped_theta = PolicySearchParams.from_theta(raw_theta).to_theta()
        self.assertEqual(
            best_params["best_policy_params"],
            _round_json_floats({
                "slow_down_distance_m": float(clipped_theta[0]),
                "callback_distance_m": float(clipped_theta[1]),
                "callback_wait_seconds": float(clipped_theta[2]),
                "slowdown_speed_scale": float(clipped_theta[3]),
            }),
        )
        self.assertEqual(best_params["final_policy_params"], best_params["best_policy_params"])
        self.assertNotIn("final_mu_policy_params", best_params)
        self.assertEqual(
            best_params["final_mu"],
            _round_json_floats([float(value) for value in raw_theta]),
        )
        self.assertIn("best_eval_mean_return", best_params)
        self.assertIn("final_eval_mean_return", best_params)

    def test_plot_training_metrics_accepts_default_epoch_axis_label(self):
        metrics = [
            {
                "epoch": 1,
                "mean_return": -1.0,
                "best_return": -1.0,
                "mean_duration_seconds": 10.0,
                "mean_overwhelmed_triggers": 0.0,
                "mean_impatient_triggers": 1.0,
                "mean_distracted_triggers": 2.0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plot.png"
            plot_training_metrics(metrics, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_main_writes_csv_and_plot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "artifacts"

            with patch("train_rwr.MuseumEnv", _FakeTrainingEnv):
                exit_code = main(
                    [
                        "--epochs",
                        "1",
                        "--samples-per-epoch",
                        "1",
                        "--episodes-per-fit",
                        "1",
                        "--seed",
                        "7",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            csv_path = output_dir / DEFAULT_CSV_NAME
            plot_path = output_dir / DEFAULT_PLOT_NAME
            best_params_path = output_dir / DEFAULT_BEST_PARAMS_NAME
            self.assertTrue(csv_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertTrue(best_params_path.exists())
            self.assertGreater(csv_path.stat().st_size, 0)
            self.assertGreater(plot_path.stat().st_size, 0)
            self.assertGreater(best_params_path.stat().st_size, 0)

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with best_params_path.open("r", encoding="utf-8") as handle:
                best_params = json.load(handle)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], "1")
            self.assertIn("mean_return", rows[0])
            self.assertNotIn("success_rate", rows[0])
            self.assertEqual(len(best_params["best_theta_seen"]), 4)
            self.assertEqual(best_params["final_theta"], best_params["best_theta_seen"])
            self.assertEqual(best_params["final_policy_params"], best_params["best_policy_params"])
            self.assertEqual(best_params["final_mu"], best_params["final_theta"])
            self.assertNotIn("final_mu_policy_params", best_params)
            self.assertNotIn("best_success_rate", best_params)
            self.assertIn("best_eval_mean_return", best_params)
            self.assertIn("final_eval_mean_return", best_params)
            self.assertEqual(set(best_params["best_policy_params"].keys()), {
                "slow_down_distance_m",
                "callback_distance_m",
                "callback_wait_seconds",
                "slowdown_speed_scale",
            })
            self.assertEqual(set(best_params["final_policy_params"].keys()), {
                "slow_down_distance_m",
                "callback_distance_m",
                "callback_wait_seconds",
                "slowdown_speed_scale",
            })

    def test_default_output_dir_stays_under_repo_root(self):
        self.assertEqual(DEFAULT_OUTPUT_DIR.parent, Path("/home/tianci/Polimi/workspace/Master-Thesis/runs"))
        self.assertTrue(DEFAULT_OUTPUT_DIR.name.startswith("rwr_"))


if __name__ == "__main__":
    unittest.main()
