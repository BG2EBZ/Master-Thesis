import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from train_rwr import (
    DEFAULT_BEST_PARAMS_NAME,
    DEFAULT_CSV_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOT_NAME,
    EpisodeResult,
    ThetaEvaluation,
    _close_cached_env,
    _evaluate_theta_sample,
    main,
    run_episode,
    train,
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

    def test_evaluate_theta_sample_aggregates_seeds_and_cached_env_closes_on_cleanup(self):
        theta = np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64)

        with patch("train_rwr.MuseumEnv", _FakeTrainingEnv):
            evaluation = _evaluate_theta_sample((theta, (1, 2, 3), 15, False))
            self.assertEqual(len(_FakeTrainingEnv.instances), 1)
            self.assertFalse(_FakeTrainingEnv.instances[0].closed)
            _close_cached_env()

        self.assertAlmostEqual(evaluation.mean_return, -0.1, places=6)
        self.assertAlmostEqual(evaluation.success_rate, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(evaluation.mean_duration_seconds, 14.5, places=6)
        self.assertAlmostEqual(evaluation.mean_overwhelmed_triggers, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(evaluation.mean_impatient_triggers, 1.0, places=6)
        self.assertAlmostEqual(evaluation.mean_distracted_triggers, 2.0 / 3.0, places=6)
        self.assertTrue(_FakeTrainingEnv.instances[0].closed)

    def test_evaluate_theta_sample_reuses_cached_env_for_multiple_tasks(self):
        theta_a = np.array([2.5, 3.5, 2.0, 0.7], dtype=np.float64)
        theta_b = np.array([2.0, 3.0, 1.5, 0.6], dtype=np.float64)

        with patch("train_rwr.MuseumEnv", _FakeTrainingEnv):
            evaluation_a = _evaluate_theta_sample((theta_a, (1,), 15, False))
            evaluation_b = _evaluate_theta_sample((theta_b, (2,), 15, False))

        self.assertEqual(len(_FakeTrainingEnv.instances), 1)
        self.assertAlmostEqual(evaluation_a.mean_return, -0.05, places=6)
        self.assertAlmostEqual(evaluation_b.mean_return, -0.86, places=6)
        self.assertFalse(_FakeTrainingEnv.instances[0].closed)

    def test_train_parallel_branch_rebuilds_executor_every_two_epochs(self):
        seen_tasks = []

        def _fake_evaluate_theta_sample(task):
            seen_tasks.append(task)
            return ThetaEvaluation(
                mean_return=float(len(seen_tasks)),
                success_rate=0.5,
                mean_duration_seconds=1.0,
                mean_overwhelmed_triggers=0.0,
                mean_impatient_triggers=0.0,
                mean_distracted_triggers=0.0,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("train_rwr.ProcessPoolExecutor", _FakeExecutor):
                with patch(
                    "train_rwr._evaluate_theta_sample",
                    side_effect=_fake_evaluate_theta_sample,
                ):
                    metrics = train(
                        epochs=5,
                        samples_per_epoch=3,
                        seed=7,
                        output_dir=Path(tmp_dir),
                    )
            best_params_path = Path(tmp_dir) / DEFAULT_BEST_PARAMS_NAME
            self.assertTrue(best_params_path.exists())
            with best_params_path.open("r", encoding="utf-8") as handle:
                best_params = json.load(handle)

        self.assertEqual(_FakeExecutor.last_max_workers, 3)
        self.assertEqual(len(_FakeExecutor.instances), 3)
        self.assertEqual(
            [len(instance.mapped_task_batches) for instance in _FakeExecutor.instances],
            [2, 2, 1],
        )
        flattened_expected_tasks = [
            task
            for instance in _FakeExecutor.instances
            for batch in instance.mapped_task_batches
            for task in batch
        ]
        self.assertEqual(len(flattened_expected_tasks), 15)
        self.assertEqual(len(seen_tasks), 15)
        for expected_task, seen_task in zip(flattened_expected_tasks, seen_tasks):
            self.assertTrue(np.allclose(expected_task[0], seen_task[0]))
            self.assertEqual(expected_task[1:], seen_task[1:])
        self.assertTrue(all(len(batch) == 3 for instance in _FakeExecutor.instances for batch in instance.mapped_task_batches))
        self.assertTrue(all(task[3] is False for task in seen_tasks))
        self.assertEqual([row["epoch"] for row in metrics], [1, 2, 3, 4, 5])
        final_seen_task = seen_tasks[-1]
        self.assertEqual(best_params["best_epoch"], 5)
        self.assertEqual(best_params["best_sample_index_within_epoch"], 3)
        self.assertTrue(np.allclose(best_params["best_theta_seen"], final_seen_task[0]))
        self.assertAlmostEqual(best_params["best_return"], 15.0, places=6)
        self.assertEqual(len(best_params["final_mu"]), 4)
        self.assertEqual(len(best_params["final_std"]), 4)

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
            self.assertIn("success_rate", rows[0])
            self.assertEqual(len(best_params["best_theta_seen"]), 4)
            self.assertEqual(set(best_params["best_policy_params"].keys()), {
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
