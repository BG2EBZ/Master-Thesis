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
    DEFAULT_BEST_PARAMS_NAME,
    DEFAULT_CSV_NAME,
    DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    DEFAULT_HELDOUT_EVALUATION_SEED_COUNT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOT_NAME,
    EpisodeResult,
    ThetaEvaluation,
    _aggregate_episode_results,
    _build_epoch_training_seed_schedule,
    _close_cached_env,
    _evaluate_episode_task,
    _sample_heldout_evaluation_seeds,
    main,
    plot_training_metrics,
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

    def test_default_seed_constants_match_master_seed_contract(self):
        self.assertEqual(DEFAULT_EPOCH_TRAIN_SEED_COUNT, 3)
        self.assertEqual(DEFAULT_HELDOUT_EVALUATION_SEED_COUNT, 20)

    def test_master_seed_reproduces_heldout_and_training_seed_schedule(self):
        def _build_schedule(master_seed):
            _theta_seq, train_seed_seq, heldout_seed_seq = np.random.SeedSequence(
                master_seed
            ).spawn(3)
            heldout_seeds = _sample_heldout_evaluation_seeds(
                np.random.default_rng(heldout_seed_seq),
                DEFAULT_HELDOUT_EVALUATION_SEED_COUNT,
            )
            training_schedule = _build_epoch_training_seed_schedule(
                np.random.default_rng(train_seed_seq),
                epochs=4,
                seeds_per_epoch=DEFAULT_EPOCH_TRAIN_SEED_COUNT,
                excluded_seeds=heldout_seeds,
            )
            return heldout_seeds, training_schedule

        heldout_a, schedule_a = _build_schedule(master_seed=17)
        heldout_b, schedule_b = _build_schedule(master_seed=17)

        self.assertEqual(heldout_a, heldout_b)
        self.assertEqual(schedule_a, schedule_b)
        self.assertEqual(len(heldout_a), DEFAULT_HELDOUT_EVALUATION_SEED_COUNT)
        self.assertEqual(len(schedule_a), 4)
        self.assertTrue(all(len(epoch) == DEFAULT_EPOCH_TRAIN_SEED_COUNT for epoch in schedule_a))

    def test_different_master_seeds_produce_different_training_schedule(self):
        _theta_seq_a, train_seed_seq_a, heldout_seed_seq_a = np.random.SeedSequence(17).spawn(3)
        heldout_a = _sample_heldout_evaluation_seeds(
            np.random.default_rng(heldout_seed_seq_a),
            DEFAULT_HELDOUT_EVALUATION_SEED_COUNT,
        )
        schedule_a = _build_epoch_training_seed_schedule(
            np.random.default_rng(train_seed_seq_a),
            epochs=3,
            seeds_per_epoch=DEFAULT_EPOCH_TRAIN_SEED_COUNT,
            excluded_seeds=heldout_a,
        )

        _theta_seq_b, train_seed_seq_b, heldout_seed_seq_b = np.random.SeedSequence(18).spawn(3)
        heldout_b = _sample_heldout_evaluation_seeds(
            np.random.default_rng(heldout_seed_seq_b),
            DEFAULT_HELDOUT_EVALUATION_SEED_COUNT,
        )
        schedule_b = _build_epoch_training_seed_schedule(
            np.random.default_rng(train_seed_seq_b),
            epochs=3,
            seeds_per_epoch=DEFAULT_EPOCH_TRAIN_SEED_COUNT,
            excluded_seeds=heldout_b,
        )

        self.assertNotEqual(heldout_a, heldout_b)
        self.assertNotEqual(schedule_a, schedule_b)

    def test_build_epoch_training_seed_schedule_excludes_heldout_seeds(self):
        _theta_seq, train_seed_seq, heldout_seed_seq = np.random.SeedSequence(23).spawn(3)
        heldout_seeds = _sample_heldout_evaluation_seeds(
            np.random.default_rng(heldout_seed_seq),
            DEFAULT_HELDOUT_EVALUATION_SEED_COUNT,
        )
        training_schedule = _build_epoch_training_seed_schedule(
            np.random.default_rng(train_seed_seq),
            epochs=6,
            seeds_per_epoch=5,
            excluded_seeds=heldout_seeds,
        )

        heldout_seed_set = set(heldout_seeds)
        self.assertTrue(all(seed not in heldout_seed_set for epoch in training_schedule for seed in epoch))

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

    def test_train_parallel_branch_reuses_one_executor_across_all_epochs(self):
        seen_tasks = []
        heldout_seeds = [
            1001,
            1002,
            1003,
            1004,
            1005,
            1006,
            1007,
            1008,
            1009,
            1010,
            1011,
            1012,
            1013,
            1014,
            1015,
            1016,
            1017,
            1018,
            1019,
            1020,
        ]
        epoch_training_schedule = [
            [11, 12, 13],
            [21, 22, 23],
            [31, 32, 33],
            [41, 42, 43],
            [51, 52, 53],
        ]

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
                with patch(
                    "train_rwr._sample_heldout_evaluation_seeds",
                    return_value=heldout_seeds,
                ) as heldout_mock:
                    with patch(
                        "train_rwr._build_epoch_training_seed_schedule",
                        return_value=epoch_training_schedule,
                    ) as schedule_mock:
                        with patch(
                            "train_rwr._evaluate_episode_task",
                            side_effect=_fake_evaluate_episode_task,
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

        heldout_mock.assert_called_once()
        schedule_mock.assert_called_once_with(
            unittest.mock.ANY,
            epochs=5,
            seeds_per_epoch=DEFAULT_EPOCH_TRAIN_SEED_COUNT,
            excluded_seeds=heldout_seeds,
        )
        self.assertEqual(_FakeExecutor.last_max_workers, 3)
        self.assertEqual(len(_FakeExecutor.instances), 1)
        self.assertEqual(
            [len(instance.mapped_task_batches) for instance in _FakeExecutor.instances],
            [5],
        )
        flattened_expected_tasks = [
            task
            for instance in _FakeExecutor.instances
            for batch in instance.mapped_task_batches
            for task in batch
        ]
        self.assertEqual(len(flattened_expected_tasks), 45)
        self.assertEqual(len(seen_tasks), 45)
        for expected_task, seen_task in zip(flattened_expected_tasks, seen_tasks):
            self.assertTrue(np.allclose(expected_task[0], seen_task[0]))
            self.assertEqual(expected_task[1:], seen_task[1:])
        self.assertTrue(
            all(
                len(batch) == 9
                for instance in _FakeExecutor.instances
                for batch in instance.mapped_task_batches
            )
        )
        self.assertTrue(all(task[3] is False for task in seen_tasks))
        self.assertTrue(all(isinstance(task[1], int) for task in seen_tasks))
        self.assertEqual([row["epoch"] for row in metrics], [1, 2, 3, 4, 5])
        grouped_tasks = [seen_tasks[idx : idx + 3] for idx in range(0, len(seen_tasks), 3)]
        grouped_returns = [
            float(np.mean([float(np.sum(task[0]) - (0.1 * task[1])) for task in theta_tasks]))
            for theta_tasks in grouped_tasks
        ]
        best_group_idx = int(np.argmax(grouped_returns))
        best_theta_tasks = grouped_tasks[best_group_idx]
        best_epoch = (best_group_idx // 3) + 1
        best_sample = (best_group_idx % 3) + 1
        self.assertEqual(best_params["best_epoch"], best_epoch)
        self.assertEqual(best_params["best_sample_index_within_epoch"], best_sample)
        self.assertEqual(best_params["master_seed"], 7)
        self.assertEqual(best_params["epoch_training_seeds"], epoch_training_schedule)
        self.assertEqual(best_params["heldout_evaluation_seeds"], heldout_seeds)
        self.assertTrue(np.allclose(best_params["best_theta_seen"], best_theta_tasks[0][0]))
        self.assertTrue(np.allclose(best_params["final_theta"], best_theta_tasks[0][0]))
        self.assertAlmostEqual(best_params["best_return"], grouped_returns[best_group_idx], places=6)
        self.assertEqual(len(best_params["final_mu"]), 4)
        self.assertEqual(len(best_params["final_std"]), 4)
        self.assertEqual(best_params["final_policy_params"], best_params["best_policy_params"])
        self.assertNotIn("final_mu_policy_params", best_params)

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
                with patch("train_rwr.INITIAL_MU", raw_theta.copy()):
                    with patch("train_rwr.INITIAL_STD", np.zeros(4, dtype=np.float64)):
                        with patch(
                            "train_rwr._evaluate_episode_task",
                            side_effect=_fake_evaluate_episode_task,
                        ):
                            train(
                                epochs=1,
                                samples_per_epoch=1,
                                seed=7,
                                output_dir=Path(tmp_dir),
                            )
            best_params_path = Path(tmp_dir) / DEFAULT_BEST_PARAMS_NAME
            with best_params_path.open("r", encoding="utf-8") as handle:
                best_params = json.load(handle)

        self.assertEqual(len(seen_tasks), DEFAULT_EPOCH_TRAIN_SEED_COUNT)
        self.assertTrue(all(np.allclose(task[0], raw_theta) for task in seen_tasks))
        self.assertEqual(best_params["best_theta_seen"], [float(value) for value in raw_theta])
        self.assertEqual(best_params["final_theta"], [float(value) for value in raw_theta])
        clipped_theta = PolicySearchParams.from_theta(raw_theta).to_theta()
        self.assertEqual(best_params["best_policy_params"], {
            "slow_down_distance_m": float(clipped_theta[0]),
            "callback_distance_m": float(clipped_theta[1]),
            "callback_wait_seconds": float(clipped_theta[2]),
            "slowdown_speed_scale": float(clipped_theta[3]),
        })
        self.assertEqual(best_params["final_policy_params"], best_params["best_policy_params"])
        self.assertNotIn("final_mu_policy_params", best_params)

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
            self.assertEqual(best_params["master_seed"], 7)
            self.assertEqual(len(best_params["epoch_training_seeds"]), 1)
            self.assertEqual(len(best_params["epoch_training_seeds"][0]), DEFAULT_EPOCH_TRAIN_SEED_COUNT)
            self.assertEqual(
                len(best_params["heldout_evaluation_seeds"]),
                DEFAULT_HELDOUT_EVALUATION_SEED_COUNT,
            )
            self.assertNotIn("final_mu_policy_params", best_params)
            self.assertNotIn("best_success_rate", best_params)
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
