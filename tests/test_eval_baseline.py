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

from scripts.eval_baseline import DEFAULT_SUMMARY_NAME, evaluate_baseline
from scripts.train_rwr import EpisodeResult
from train.rwr.rewarding import EpisodeRewardWeights


class EvaluateBaselineTests(unittest.TestCase):
    def test_evaluate_baseline_accepts_n_humans_and_reward_config(self):
        reward_config = EpisodeRewardWeights(
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
            with patch("scripts.eval_baseline._evaluate_episode_task", side_effect=episode_results) as task_mock:
                with patch("scripts.eval_baseline.plot_comparison_metrics"):
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


if __name__ == "__main__":
    unittest.main()
