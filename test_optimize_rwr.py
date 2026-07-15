import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import optimize_rwr
from eval_baseline import DEFAULT_SUMMARY_NAME, evaluate_baseline
from museum_env.reward import RewardConfig
from train_rwr import DEFAULT_BEST_PARAMS_NAME, EpisodeResult, train


class RWREntrypointTests(unittest.TestCase):
    def test_train_accepts_custom_reward_config_and_hyperparameters(self):
        reward_config = RewardConfig(
            time_penalty_per_second=0.12,
            overwhelmed_trigger_penalty=5.0,
            impatient_trigger_penalty=2.5,
            distracted_trigger_penalty=1.5,
        )
        episode_result = EpisodeResult(
            episode_return=-10.0,
            duration_seconds=12.0,
            overwhelmed_triggers=1,
            impatient_triggers=2,
            distracted_triggers=3,
            success=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "train"
            with patch("train_rwr._evaluate_episode_task", return_value=episode_result) as task_mock:
                with patch("train_rwr.plot_training_metrics"):
                    with patch("train_rwr.plot_exploration_metrics"):
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

    def test_evaluate_baseline_accepts_n_humans_and_reward_config(self):
        reward_config = RewardConfig(
            time_penalty_per_second=0.12,
            overwhelmed_trigger_penalty=5.0,
            impatient_trigger_penalty=2.5,
            distracted_trigger_penalty=1.5,
        )
        baseline_theta = np.array([3.0, 4.0, 2.0, 0.7, 1.0], dtype=np.float64)
        learned_payload = {
            "final_theta": [2.5, 3.5, 2.0, 0.7, 1.0],
        }
        episode_results = [
            EpisodeResult(-20.0, 30.0, 1, 2, 3, True),
            EpisodeResult(-10.0, 20.0, 0, 1, 1, True),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            learned_params_json = root / "best_params.json"
            learned_params_json.write_text(json.dumps(learned_payload), encoding="utf-8")
            output_dir = root / "eval"
            with patch("eval_baseline._evaluate_episode_task", side_effect=episode_results) as task_mock:
                with patch("eval_baseline.plot_comparison_metrics"):
                    metrics = evaluate_baseline(
                        learned_params_json=learned_params_json,
                        num_runs=1,
                        output_dir=output_dir,
                        max_workers=1,
                        n_humans=2,
                        baseline_theta=baseline_theta,
                        reward_config=reward_config,
                    )
            self.assertEqual(len(metrics), 1)
            first_task = task_mock.call_args_list[0].args[0]
            self.assertEqual(first_task[2], 2)
            self.assertTrue(np.allclose(first_task[0], baseline_theta))
            self.assertEqual(first_task[4], reward_config)
            self.assertTrue((output_dir / DEFAULT_SUMMARY_NAME).exists())


class OptimizeRWRTests(unittest.TestCase):
    def test_select_top_results_prefers_higher_return_then_lower_triggers(self):
        config = optimize_rwr.SearchTrialConfig(
            epochs=20,
            samples_per_epoch=30,
            seed=42,
            beta=0.1,
            train_seeds_per_epoch=1,
            n_humans=15,
            time_penalty_per_second=0.1,
            overwhelmed_trigger_penalty=4.0,
            impatient_trigger_penalty=2.0,
            distracted_trigger_penalty=2.0,
        )
        worse_return = optimize_rwr.SearchTrialResult(
            phase="coarse",
            trial_id=1,
            config=config,
            train_output_dir=Path("/tmp/train1"),
            eval_output_dir=Path("/tmp/eval1"),
            learned_params_json=Path("/tmp/train1/best_params.json"),
            summary={
                "comparison_mean_return": -50.0,
                "comparison_mean_overwhelmed_triggers": 1.0,
                "comparison_mean_impatient_triggers": 5.0,
                "comparison_mean_distracted_triggers": 2.0,
                "comparison_mean_duration_seconds": 100.0,
            },
        )
        better_return = optimize_rwr.SearchTrialResult(
            phase="coarse",
            trial_id=2,
            config=config,
            train_output_dir=Path("/tmp/train2"),
            eval_output_dir=Path("/tmp/eval2"),
            learned_params_json=Path("/tmp/train2/best_params.json"),
            summary={
                "comparison_mean_return": -40.0,
                "comparison_mean_overwhelmed_triggers": 2.0,
                "comparison_mean_impatient_triggers": 6.0,
                "comparison_mean_distracted_triggers": 3.0,
                "comparison_mean_duration_seconds": 120.0,
            },
        )

        ranked = optimize_rwr._select_top_results([worse_return, better_return], 2)

        self.assertEqual([result.trial_id for result in ranked], [2, 1])

    def test_run_search_writes_trial_artifacts_with_mocked_train_and_eval(self):
        config = optimize_rwr.SearchTrialConfig(
            epochs=20,
            samples_per_epoch=30,
            seed=42,
            beta=0.1,
            train_seeds_per_epoch=1,
            n_humans=15,
            time_penalty_per_second=0.1,
            overwhelmed_trigger_penalty=4.0,
            impatient_trigger_penalty=2.0,
            distracted_trigger_penalty=2.0,
        )

        def fake_train(*, output_dir: Path, **_kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "best_theta_seen": [3.0, 4.0, 2.0, 0.7, 1.0],
                "final_theta": [3.0, 4.0, 2.0, 0.7, 1.0],
            }
            (output_dir / DEFAULT_BEST_PARAMS_NAME).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            return [{"epoch": 1, "mean_return": -1.0}]

        def fake_evaluate_baseline(*, output_dir: Path, num_runs: int, **_kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "comparison_mean_return": float(num_runs),
                "comparison_mean_overwhelmed_triggers": 1.0,
                "comparison_mean_impatient_triggers": 2.0,
                "comparison_mean_distracted_triggers": 3.0,
                "comparison_mean_duration_seconds": 4.0,
            }
            (output_dir / DEFAULT_SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "search"
            with patch("optimize_rwr._sample_trial_config", return_value=config):
                with patch("optimize_rwr.train", side_effect=fake_train):
                    with patch("optimize_rwr.evaluate_baseline", side_effect=fake_evaluate_baseline):
                        results = optimize_rwr.run_search(
                            output_dir=output_dir,
                            coarse_trials=1,
                            refine_top_k=1,
                            final_top_k=1,
                            coarse_eval_runs=5,
                            refine_eval_runs=10,
                            final_eval_runs=20,
                        )
            self.assertEqual(len(results), 3)
            self.assertTrue((output_dir / "trial_001" / "trial_config.json").exists())
            self.assertTrue((output_dir / "trial_001" / "coarse_trial_result.json").exists())
            self.assertTrue((output_dir / "trial_001" / "refine_trial_result.json").exists())
            self.assertTrue((output_dir / "trial_001" / "final_trial_result.json").exists())
            self.assertTrue((output_dir / "leaderboard.csv").exists())
            self.assertTrue((output_dir / "best_search_config.json").exists())


if __name__ == "__main__":
    unittest.main()
