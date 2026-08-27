"""RWR-only hyperparameter search for the guide policy-search trainer.

This script intentionally keeps the historical RWR search surface: it sweeps
RWR training and reward settings, but does not sweep REPS `eps`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.guide_config import GuideBehaviorConfig
from scripts.eval_baseline import (
    DEFAULT_MAX_WORKERS as DEFAULT_EVAL_MAX_WORKERS,
    DEFAULT_SUMMARY_NAME as DEFAULT_EVAL_SUMMARY_NAME,
    evaluate_baseline,
)
from train.policy_search.defaults import (
    DEFAULT_BEST_PARAMS_NAME,
    DEFAULT_N_HUMANS,
    DEFAULT_SEED,
)
from train.policy_search.policy_codec import guide_config_to_theta
from train.policy_search.rewarding import EpisodeRewardWeights
from train.policy_search.training import train

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "runs" / f"rwr_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

EPOCH_CHOICES = (15, 20, 30, 40)
SAMPLES_PER_EPOCH_CHOICES = (20, 30, 40, 60)
BETA_CHOICES = (0.03, 0.05, 0.1, 0.2, 0.4)
TRAIN_SEEDS_PER_EPOCH_CHOICES = (1, 3, 5)
MASTER_SEED_CHOICES = (42, 123, 777, 2025, 4096)
TIME_PENALTY_CHOICES = (0.05, 0.08, 0.1, 0.12, 0.15)
OVERWHELMED_PENALTY_CHOICES = (3.0, 4.0, 5.0, 6.0, 8.0)
IMPATIENT_PENALTY_CHOICES = (1.0, 1.5, 2.0, 2.5, 3.0)
DISTRACTED_PENALTY_CHOICES = (1.0, 1.5, 2.0, 2.5, 3.0)

DEFAULT_COARSE_TRIALS = 8
DEFAULT_REFINE_TOP_K = 3
DEFAULT_FINAL_TOP_K = 2
DEFAULT_COARSE_EVAL_RUNS = 5
DEFAULT_REFINE_EVAL_RUNS = 10
DEFAULT_FINAL_EVAL_RUNS = 20
LEADERBOARD_FIELDNAMES = (
    "phase",
    "rank_in_phase",
    "trial_id",
    "epochs",
    "samples_per_epoch",
    "seed",
    "beta",
    "train_seeds_per_epoch",
    "n_humans",
    "time_penalty_per_second",
    "overwhelmed_trigger_penalty",
    "impatient_trigger_penalty",
    "distracted_trigger_penalty",
    "comparison_mean_return",
    "comparison_mean_overwhelmed_triggers",
    "comparison_mean_impatient_triggers",
    "comparison_mean_distracted_triggers",
    "comparison_mean_duration_seconds",
    "train_output_dir",
    "eval_output_dir",
    "learned_params_json",
)


@dataclass(frozen=True)
class SearchTrialConfig:
    epochs: int
    samples_per_epoch: int
    seed: int
    beta: float
    train_seeds_per_epoch: int
    n_humans: int
    time_penalty_per_second: float
    overwhelmed_trigger_penalty: float
    impatient_trigger_penalty: float
    distracted_trigger_penalty: float

    @property
    def reward_weights(self) -> EpisodeRewardWeights:
        return EpisodeRewardWeights(
            time_penalty_per_second=float(self.time_penalty_per_second),
            overwhelmed_trigger_penalty=float(self.overwhelmed_trigger_penalty),
            impatient_trigger_penalty=float(self.impatient_trigger_penalty),
            distracted_trigger_penalty=float(self.distracted_trigger_penalty),
        )


@dataclass(frozen=True)
class SearchTrialResult:
    phase: str
    trial_id: int
    config: SearchTrialConfig
    train_output_dir: Path
    eval_output_dir: Path
    learned_params_json: Path
    summary: dict[str, object]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, received {type(payload)!r}.")
    return payload


def _write_json(payload: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _sample_trial_config(rng: np.random.Generator, *, n_humans: int) -> SearchTrialConfig:
    return SearchTrialConfig(
        epochs=int(rng.choice(EPOCH_CHOICES)),
        samples_per_epoch=int(rng.choice(SAMPLES_PER_EPOCH_CHOICES)),
        seed=int(rng.choice(MASTER_SEED_CHOICES)),
        beta=float(rng.choice(BETA_CHOICES)),
        train_seeds_per_epoch=int(rng.choice(TRAIN_SEEDS_PER_EPOCH_CHOICES)),
        n_humans=int(n_humans),
        time_penalty_per_second=float(rng.choice(TIME_PENALTY_CHOICES)),
        overwhelmed_trigger_penalty=float(rng.choice(OVERWHELMED_PENALTY_CHOICES)),
        impatient_trigger_penalty=float(rng.choice(IMPATIENT_PENALTY_CHOICES)),
        distracted_trigger_penalty=float(rng.choice(DISTRACTED_PENALTY_CHOICES)),
    )


def _trial_sort_key(result: SearchTrialResult) -> tuple[float, float, float, float, float]:
    summary = result.summary
    return (
        -float(summary["comparison_mean_return"]),
        float(summary["comparison_mean_overwhelmed_triggers"]),
        float(summary["comparison_mean_impatient_triggers"]),
        float(summary["comparison_mean_distracted_triggers"]),
        float(summary["comparison_mean_duration_seconds"]),
    )


def _select_top_results(results: Sequence[SearchTrialResult], top_k: int) -> list[SearchTrialResult]:
    return sorted(results, key=_trial_sort_key)[: max(0, int(top_k))]


def _trial_payload(result: SearchTrialResult) -> dict[str, object]:
    return {
        "phase": str(result.phase),
        "trial_id": int(result.trial_id),
        "config": asdict(result.config),
        "train_output_dir": str(result.train_output_dir),
        "eval_output_dir": str(result.eval_output_dir),
        "learned_params_json": str(result.learned_params_json),
        "summary": result.summary,
    }


def _leaderboard_rows(results: Sequence[SearchTrialResult], phase: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ranked_results = _select_top_results(results, len(results))
    for rank_idx, result in enumerate(ranked_results, start=1):
        row = {
            "phase": phase,
            "rank_in_phase": int(rank_idx),
            "trial_id": int(result.trial_id),
            "epochs": int(result.config.epochs),
            "samples_per_epoch": int(result.config.samples_per_epoch),
            "seed": int(result.config.seed),
            "beta": float(result.config.beta),
            "train_seeds_per_epoch": int(result.config.train_seeds_per_epoch),
            "n_humans": int(result.config.n_humans),
            "time_penalty_per_second": float(result.config.time_penalty_per_second),
            "overwhelmed_trigger_penalty": float(result.config.overwhelmed_trigger_penalty),
            "impatient_trigger_penalty": float(result.config.impatient_trigger_penalty),
            "distracted_trigger_penalty": float(result.config.distracted_trigger_penalty),
            "comparison_mean_return": float(result.summary["comparison_mean_return"]),
            "comparison_mean_overwhelmed_triggers": float(
                result.summary["comparison_mean_overwhelmed_triggers"]
            ),
            "comparison_mean_impatient_triggers": float(
                result.summary["comparison_mean_impatient_triggers"]
            ),
            "comparison_mean_distracted_triggers": float(
                result.summary["comparison_mean_distracted_triggers"]
            ),
            "comparison_mean_duration_seconds": float(
                result.summary["comparison_mean_duration_seconds"]
            ),
            "train_output_dir": str(result.train_output_dir),
            "eval_output_dir": str(result.eval_output_dir),
            "learned_params_json": str(result.learned_params_json),
        }
        rows.append(row)
    return rows


def _write_leaderboard(
    phase_results: dict[str, Sequence[SearchTrialResult]],
    output_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for phase_name in ("coarse", "refine", "final"):
        rows.extend(_leaderboard_rows(phase_results.get(phase_name, []), phase_name))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in LEADERBOARD_FIELDNAMES})


def _run_training_trial(
    *,
    config: SearchTrialConfig,
    trial_id: int,
    search_output_dir: Path,
) -> tuple[Path, Path]:
    trial_dir = search_output_dir / f"trial_{int(trial_id):03d}"
    train_output_dir = trial_dir / "train"
    train_output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(asdict(config), trial_dir / "trial_config.json")
    train(
        epochs=int(config.epochs),
        samples_per_epoch=int(config.samples_per_epoch),
        seed=int(config.seed),
        output_dir=train_output_dir,
        algorithm="rwr",
        beta=float(config.beta),
        train_seeds_per_epoch=int(config.train_seeds_per_epoch),
        n_humans=int(config.n_humans),
        reward_config=config.reward_weights,
    )
    learned_params_json = train_output_dir / DEFAULT_BEST_PARAMS_NAME
    return trial_dir, learned_params_json


def _evaluate_existing_trial(
    *,
    config: SearchTrialConfig,
    trial_id: int,
    phase: str,
    eval_num_runs: int,
    eval_max_workers: int,
    search_output_dir: Path,
    trial_dir: Path,
    learned_params_json: Path,
    baseline_theta: np.ndarray,
) -> SearchTrialResult:
    eval_output_dir = trial_dir / f"eval_{phase}"
    evaluate_baseline(
        learned_params_json=learned_params_json,
        num_runs=int(eval_num_runs),
        output_dir=eval_output_dir,
        max_workers=int(eval_max_workers),
        n_humans=int(config.n_humans),
        baseline_theta=baseline_theta,
        reward_config=config.reward_weights,
    )
    summary = _load_json(eval_output_dir / DEFAULT_EVAL_SUMMARY_NAME)
    result = SearchTrialResult(
        phase=str(phase),
        trial_id=int(trial_id),
        config=config,
        train_output_dir=trial_dir / "train",
        eval_output_dir=eval_output_dir,
        learned_params_json=learned_params_json,
        summary=summary,
    )
    _write_json(_trial_payload(result), trial_dir / f"{phase}_trial_result.json")
    _write_leaderboard(
        {
            phase: [result],
        },
        search_output_dir / "leaderboard_latest_trial.csv",
    )
    return result


def run_search(
    *,
    output_dir: Path,
    search_seed: int = DEFAULT_SEED,
    n_humans: int = DEFAULT_N_HUMANS,
    eval_max_workers: int = DEFAULT_EVAL_MAX_WORKERS,
    coarse_trials: int = DEFAULT_COARSE_TRIALS,
    refine_top_k: int = DEFAULT_REFINE_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    coarse_eval_runs: int = DEFAULT_COARSE_EVAL_RUNS,
    refine_eval_runs: int = DEFAULT_REFINE_EVAL_RUNS,
    final_eval_runs: int = DEFAULT_FINAL_EVAL_RUNS,
) -> list[SearchTrialResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(search_seed))
    baseline_theta = guide_config_to_theta(GuideBehaviorConfig())
    phase_results: dict[str, list[SearchTrialResult]] = {"coarse": [], "refine": [], "final": []}

    for trial_id in range(1, int(coarse_trials) + 1):
        config = _sample_trial_config(rng, n_humans=int(n_humans))
        trial_dir, learned_params_json = _run_training_trial(
            config=config,
            trial_id=int(trial_id),
            search_output_dir=output_dir,
        )
        phase_results["coarse"].append(
            _evaluate_existing_trial(
                config=config,
                trial_id=int(trial_id),
                phase="coarse",
                eval_num_runs=int(coarse_eval_runs),
                eval_max_workers=int(eval_max_workers),
                search_output_dir=output_dir,
                trial_dir=trial_dir,
                learned_params_json=learned_params_json,
                baseline_theta=baseline_theta,
            )
        )

    coarse_top = _select_top_results(
        phase_results["coarse"],
        min(int(refine_top_k), len(phase_results["coarse"])),
    )
    for result in coarse_top:
        phase_results["refine"].append(
            _evaluate_existing_trial(
                config=result.config,
                trial_id=int(result.trial_id),
                phase="refine",
                eval_num_runs=int(refine_eval_runs),
                eval_max_workers=int(eval_max_workers),
                search_output_dir=output_dir,
                trial_dir=result.train_output_dir.parent,
                learned_params_json=result.learned_params_json,
                baseline_theta=baseline_theta,
            )
        )

    final_candidates = _select_top_results(
        phase_results["refine"] if phase_results["refine"] else phase_results["coarse"],
        min(
            int(final_top_k),
            len(phase_results["refine"] if phase_results["refine"] else phase_results["coarse"]),
        ),
    )
    for result in final_candidates:
        phase_results["final"].append(
            _evaluate_existing_trial(
                config=result.config,
                trial_id=int(result.trial_id),
                phase="final",
                eval_num_runs=int(final_eval_runs),
                eval_max_workers=int(eval_max_workers),
                search_output_dir=output_dir,
                trial_dir=result.train_output_dir.parent,
                learned_params_json=result.learned_params_json,
                baseline_theta=baseline_theta,
            )
        )

    _write_leaderboard(phase_results, output_dir / "leaderboard.csv")
    best_pool = (
        phase_results["final"]
        or phase_results["refine"]
        or phase_results["coarse"]
    )
    best_result = _select_top_results(best_pool, 1)[0]
    _write_json(_trial_payload(best_result), output_dir / "best_search_config.json")
    return phase_results["coarse"] + phase_results["refine"] + phase_results["final"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal multi-phase search over RWR training and reward settings."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where search artifacts, leaderboard, and best_search_config.json are written.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Master seed for sampling trial configurations.",
    )
    parser.add_argument(
        "--n-humans",
        type=_positive_int,
        default=DEFAULT_N_HUMANS,
        help="Number of humans to simulate during training and evaluation.",
    )
    parser.add_argument(
        "--eval-max-workers",
        type=_positive_int,
        default=DEFAULT_EVAL_MAX_WORKERS,
        help="Maximum worker processes for held-out evaluation.",
    )
    parser.add_argument(
        "--coarse-trials",
        type=_positive_int,
        default=DEFAULT_COARSE_TRIALS,
        help="Number of random coarse-search trials to train and evaluate.",
    )
    parser.add_argument(
        "--refine-top-k",
        type=_positive_int,
        default=DEFAULT_REFINE_TOP_K,
        help="How many top coarse trials to reevaluate in the refine phase.",
    )
    parser.add_argument(
        "--final-top-k",
        type=_positive_int,
        default=DEFAULT_FINAL_TOP_K,
        help="How many top refine trials to reevaluate in the final phase.",
    )
    parser.add_argument(
        "--coarse-eval-runs",
        type=_positive_int,
        default=DEFAULT_COARSE_EVAL_RUNS,
        help="Held-out evaluation runs per coarse trial.",
    )
    parser.add_argument(
        "--refine-eval-runs",
        type=_positive_int,
        default=DEFAULT_REFINE_EVAL_RUNS,
        help="Held-out evaluation runs per refine trial.",
    )
    parser.add_argument(
        "--final-eval-runs",
        type=_positive_int,
        default=DEFAULT_FINAL_EVAL_RUNS,
        help="Held-out evaluation runs per final trial.",
    )
    args = parser.parse_args(argv)
    run_search(
        output_dir=args.output_dir,
        search_seed=int(args.search_seed),
        n_humans=int(args.n_humans),
        eval_max_workers=int(args.eval_max_workers),
        coarse_trials=int(args.coarse_trials),
        refine_top_k=int(args.refine_top_k),
        final_top_k=int(args.final_top_k),
        coarse_eval_runs=int(args.coarse_eval_runs),
        refine_eval_runs=int(args.refine_eval_runs),
        final_eval_runs=int(args.final_eval_runs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
