from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Keep BLAS/OpenMP libraries from oversubscribing CPU cores across workers
# unless the caller explicitly sets a different thread count.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from museum_env.guide_config import GuideBehaviorConfig
from train.common.artifacts import write_csv_rows, write_json
from train.common.plot_utils import compute_mean_confidence_band
from train.rwr.defaults import DEFAULT_MAX_WORKERS, DEFAULT_N_HUMANS, DEFAULT_SEED
from train.rwr.policy_codec import guide_config_to_theta, summarize_theta
from train.rwr.rewarding import DEFAULT_EPISODE_REWARD_WEIGHTS, EpisodeRewardWeights
from train.rwr.evaluation import _evaluate_episode_task

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DEFAULT_N_SEEDS = 10
DEFAULT_EPISODES_PER_SEED = 20
DEFAULT_OUTPUT_DIR = (
    ARTIFACTS_ROOT / "runs" / f"baseline_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
EPISODE_CSV_NAME = "baseline_episode_metrics.csv"
SEED_CSV_NAME = "baseline_seed_metrics.csv"
SUMMARY_JSON_NAME = "baseline_summary.json"
BASELINE_METRIC_NAMES = (
    "return",
    "duration_seconds",
    "overwhelmed_triggers",
    "impatient_triggers",
    "distracted_triggers",
)
EPISODE_FIELDNAMES = (
    "seed_index",
    "master_seed",
    "episode_index",
    "episode_seed",
    "return",
    "duration_seconds",
    "overwhelmed_triggers",
    "impatient_triggers",
    "distracted_triggers",
    "success",
)
SEED_FIELDNAMES = (
    "seed_index",
    "master_seed",
    "episode_count",
    "mean_return",
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return parsed


def _build_seed_schedule(
    *,
    seed: int,
    n_seeds: int,
    episodes_per_seed: int,
) -> list[dict[str, int | list[int]]]:
    master_seed_sequences = np.random.SeedSequence(int(seed)).spawn(int(n_seeds))
    master_seeds = [
        int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        for seed_sequence in master_seed_sequences
    ]

    schedule: list[dict[str, int | list[int]]] = []
    for seed_index, master_seed in enumerate(master_seeds, start=1):
        episode_seed_sequences = np.random.SeedSequence(int(master_seed)).spawn(
            int(episodes_per_seed)
        )
        episode_seeds = [
            int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
            for seed_sequence in episode_seed_sequences
        ]
        schedule.append(
            {
                "seed_index": int(seed_index),
                "master_seed": int(master_seed),
                "episode_seeds": episode_seeds,
            }
        )
    return schedule


def _resolve_worker_count(*, task_count: int, max_workers: int) -> int:
    if int(task_count) <= 0:
        raise ValueError("task_count must be positive")
    if int(max_workers) <= 0:
        raise ValueError("max_workers must be positive")
    return min(int(task_count), int(max_workers), os.cpu_count() or 1)


def _build_episode_rows_and_tasks(
    *,
    seed_schedule: Sequence[dict[str, int | list[int]]],
    baseline_theta: np.ndarray,
    n_humans: int,
    reward_config: EpisodeRewardWeights,
) -> tuple[
    list[dict[str, int | float | bool]],
    list[tuple[np.ndarray, int, int, bool, EpisodeRewardWeights]],
]:
    episode_rows: list[dict[str, int | float | bool]] = []
    episode_tasks: list[tuple[np.ndarray, int, int, bool, EpisodeRewardWeights]] = []
    for seed_entry in seed_schedule:
        seed_index = int(seed_entry["seed_index"])
        master_seed = int(seed_entry["master_seed"])
        episode_seeds = [int(value) for value in seed_entry["episode_seeds"]]
        for episode_index, episode_seed in enumerate(episode_seeds, start=1):
            episode_rows.append(
                {
                    "seed_index": int(seed_index),
                    "master_seed": int(master_seed),
                    "episode_index": int(episode_index),
                    "episode_seed": int(episode_seed),
                    "return": 0.0,
                    "duration_seconds": 0.0,
                    "overwhelmed_triggers": 0,
                    "impatient_triggers": 0,
                    "distracted_triggers": 0,
                    "success": False,
                }
            )
            episode_tasks.append(
                (
                    np.asarray(baseline_theta, dtype=np.float64),
                    int(episode_seed),
                    int(n_humans),
                    False,
                    reward_config,
                )
            )
    return episode_rows, episode_tasks


def _fill_episode_rows(
    episode_rows: Sequence[dict[str, int | float | bool]],
    episode_results,
) -> list[dict[str, int | float | bool]]:
    if len(episode_rows) != len(episode_results):
        raise ValueError("episode rows and results must align")

    filled_rows: list[dict[str, int | float | bool]] = []
    for row, result in zip(episode_rows, episode_results):
        filled_row = dict(row)
        filled_row["return"] = float(result.episode_return)
        filled_row["duration_seconds"] = float(result.duration_seconds)
        filled_row["overwhelmed_triggers"] = int(result.overwhelmed_triggers)
        filled_row["impatient_triggers"] = int(result.impatient_triggers)
        filled_row["distracted_triggers"] = int(result.distracted_triggers)
        filled_row["success"] = bool(result.success)
        filled_rows.append(filled_row)
    return filled_rows


def _build_seed_rows(
    episode_rows: Sequence[dict[str, int | float | bool]],
) -> list[dict[str, int | float]]:
    seed_rows: list[dict[str, int | float]] = []
    seed_indices = sorted({int(row["seed_index"]) for row in episode_rows})
    for seed_index in seed_indices:
        group_rows = [row for row in episode_rows if int(row["seed_index"]) == seed_index]
        if not group_rows:
            continue
        seed_rows.append(
            {
                "seed_index": int(seed_index),
                "master_seed": int(group_rows[0]["master_seed"]),
                "episode_count": int(len(group_rows)),
                "mean_return": float(np.mean([float(row["return"]) for row in group_rows])),
                "mean_duration_seconds": float(
                    np.mean([float(row["duration_seconds"]) for row in group_rows])
                ),
                "mean_overwhelmed_triggers": float(
                    np.mean([float(row["overwhelmed_triggers"]) for row in group_rows])
                ),
                "mean_impatient_triggers": float(
                    np.mean([float(row["impatient_triggers"]) for row in group_rows])
                ),
                "mean_distracted_triggers": float(
                    np.mean([float(row["distracted_triggers"]) for row in group_rows])
                ),
            }
        )
    return seed_rows


def _build_summary_payload(
    *,
    seed_rows: Sequence[dict[str, int | float]],
    seed_schedule: Sequence[dict[str, int | list[int]]],
    baseline_theta: np.ndarray,
    reward_config: EpisodeRewardWeights,
    n_humans: int,
    output_paths: dict[str, Path],
) -> dict[str, object]:
    summary_payload: dict[str, object] = {
        "baseline_theta": [float(value) for value in baseline_theta],
        "baseline_policy_params": summarize_theta(baseline_theta),
        "reward_weights": {
            "time_penalty_per_second": float(reward_config.time_penalty_per_second),
            "overwhelmed_trigger_penalty": float(reward_config.overwhelmed_trigger_penalty),
            "impatient_trigger_penalty": float(reward_config.impatient_trigger_penalty),
            "distracted_trigger_penalty": float(reward_config.distracted_trigger_penalty),
        },
        "n_humans": int(n_humans),
        "n_seeds": int(len(seed_rows)),
        "episodes_per_seed": int(seed_rows[0]["episode_count"]) if seed_rows else 0,
        "total_episodes": int(sum(int(row["episode_count"]) for row in seed_rows)),
        "seed_schedule": [
            {
                "seed_index": int(seed_entry["seed_index"]),
                "master_seed": int(seed_entry["master_seed"]),
                "episode_seeds": [int(value) for value in seed_entry["episode_seeds"]],
            }
            for seed_entry in seed_schedule
        ],
        "baseline_episode_metrics_csv": str(output_paths["episode_csv"]),
        "baseline_seed_metrics_csv": str(output_paths["seed_csv"]),
    }

    for metric_name in BASELINE_METRIC_NAMES:
        seed_metric_name = f"mean_{metric_name}"
        values = np.asarray(
            [[float(seed_row[seed_metric_name])] for seed_row in seed_rows],
            dtype=np.float64,
        )
        band = compute_mean_confidence_band(values)
        summary_payload[seed_metric_name] = float(band.mean[0])
        summary_payload[f"ci95_low_{metric_name}"] = float(band.low[0])
        summary_payload[f"ci95_high_{metric_name}"] = float(band.high[0])

    return summary_payload


def evaluate_baseline_only(
    *,
    seed: int,
    n_seeds: int,
    episodes_per_seed: int,
    output_dir: Path,
    max_workers: int,
    n_humans: int,
    reward_config: EpisodeRewardWeights,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_theta = guide_config_to_theta(GuideBehaviorConfig())
    seed_schedule = _build_seed_schedule(
        seed=int(seed),
        n_seeds=int(n_seeds),
        episodes_per_seed=int(episodes_per_seed),
    )
    episode_rows, episode_tasks = _build_episode_rows_and_tasks(
        seed_schedule=seed_schedule,
        baseline_theta=baseline_theta,
        n_humans=int(n_humans),
        reward_config=reward_config,
    )
    worker_count = _resolve_worker_count(
        task_count=len(episode_tasks),
        max_workers=int(max_workers),
    )

    print(
        "Running baseline-only evaluation: "
        f"n_seeds={int(n_seeds)}, episodes_per_seed={int(episodes_per_seed)}, "
        f"total_episodes={len(episode_tasks)}, workers={worker_count}",
        flush=True,
    )
    if worker_count == 1:
        try:
            episode_results = [_evaluate_episode_task(task) for task in episode_tasks]
        finally:
            from train.common.rollout import close_cached_env

            close_cached_env()
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            episode_results = list(executor.map(_evaluate_episode_task, episode_tasks))

    filled_episode_rows = _fill_episode_rows(episode_rows, episode_results)
    seed_rows = _build_seed_rows(filled_episode_rows)

    episode_csv_path = output_dir / EPISODE_CSV_NAME
    seed_csv_path = output_dir / SEED_CSV_NAME
    summary_json_path = output_dir / SUMMARY_JSON_NAME
    write_csv_rows(filled_episode_rows, EPISODE_FIELDNAMES, episode_csv_path)
    write_csv_rows(seed_rows, SEED_FIELDNAMES, seed_csv_path)
    summary_payload = _build_summary_payload(
        seed_rows=seed_rows,
        seed_schedule=seed_schedule,
        baseline_theta=baseline_theta,
        reward_config=reward_config,
        n_humans=int(n_humans),
        output_paths={
            "episode_csv": episode_csv_path,
            "seed_csv": seed_csv_path,
        },
    )
    write_json(summary_payload, summary_json_path)

    print(f"Saved baseline episode metrics CSV to {episode_csv_path}")
    print(f"Saved baseline seed metrics CSV to {seed_csv_path}")
    print(f"Saved baseline summary JSON to {summary_json_path}")
    print(
        "Baseline only: "
        f"mean_return={float(summary_payload['mean_return']):.3f}, "
        f"overwhelmed={float(summary_payload['mean_overwhelmed_triggers']):.3f}, "
        f"impatient={float(summary_payload['mean_impatient_triggers']):.3f}, "
        f"distracted={float(summary_payload['mean_distracted_triggers']):.3f}"
    )
    return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the default baseline guide policy without running RWR training."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Experiment master seed used to generate baseline seed groups.",
    )
    parser.add_argument(
        "--n-seeds",
        type=_positive_int,
        default=DEFAULT_N_SEEDS,
        help="Number of independent baseline seed groups.",
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=_positive_int,
        default=DEFAULT_EPISODES_PER_SEED,
        help="Number of baseline episodes generated per seed group.",
    )
    parser.add_argument(
        "--n-humans",
        type=_positive_int,
        default=DEFAULT_N_HUMANS,
        help="Number of humans to simulate during baseline rollouts.",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum worker processes for baseline rollout evaluation.",
    )
    parser.add_argument(
        "--time-penalty-per-second",
        type=_positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.time_penalty_per_second,
        help="Reward penalty applied per simulated second.",
    )
    parser.add_argument(
        "--overwhelmed-trigger-penalty",
        type=_positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.overwhelmed_trigger_penalty,
        help="Reward penalty applied per overwhelmed trigger.",
    )
    parser.add_argument(
        "--impatient-trigger-penalty",
        type=_positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.impatient_trigger_penalty,
        help="Reward penalty applied per impatient trigger.",
    )
    parser.add_argument(
        "--distracted-trigger-penalty",
        type=_positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.distracted_trigger_penalty,
        help="Reward penalty applied per distracted trigger.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where baseline CSV and JSON artifacts are written.",
    )
    args = parser.parse_args(argv)

    reward_config = EpisodeRewardWeights(
        time_penalty_per_second=float(args.time_penalty_per_second),
        overwhelmed_trigger_penalty=float(args.overwhelmed_trigger_penalty),
        impatient_trigger_penalty=float(args.impatient_trigger_penalty),
        distracted_trigger_penalty=float(args.distracted_trigger_penalty),
    )
    evaluate_baseline_only(
        seed=int(args.seed),
        n_seeds=int(args.n_seeds),
        episodes_per_seed=int(args.episodes_per_seed),
        output_dir=Path(args.output_dir),
        max_workers=int(args.max_workers),
        n_humans=int(args.n_humans),
        reward_config=reward_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
