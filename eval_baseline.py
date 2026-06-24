from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from museum_env.policy_search_params import PolicySearchParams
from train_rwr import (
    DEFAULT_MAX_WORKERS,
    _evaluate_episode_task,
    _policy_params_dict,
    write_best_params_json,
    write_metrics_csv,
)

if TYPE_CHECKING:
    from train_rwr import EpisodeResult

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NUM_RUNS = 20
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_CSV_NAME = "baseline_metrics.csv"
DEFAULT_PLOT_NAME = "baseline_metrics.png"
DEFAULT_SUMMARY_NAME = "baseline_summary.json"
DEFAULT_N_HUMANS = 15


def _sample_evaluation_seeds(num_runs: int, master_seed: int) -> list[int]:
    rng = np.random.default_rng(master_seed)
    sampled: list[int] = []
    seen: set[int] = set()
    while len(sampled) < int(num_runs):
        candidate = int(rng.integers(0, np.iinfo(np.int32).max))
        if candidate in seen:
            continue
        seen.add(candidate)
        sampled.append(candidate)
    return sampled


def _build_run_metrics(episode_results: Sequence[EpisodeResult]) -> list[dict[str, float | int]]:
    metrics: list[dict[str, float | int]] = []
    for idx, result in enumerate(episode_results, start=1):
        metrics.append(
            {
                "epoch": int(idx),
                "mean_return": float(result.episode_return),
                "best_return": float(result.episode_return),
                # Kept only to stay compatible with the shared training CSV schema.
                "success_rate": float(result.success),
                "mean_duration_seconds": float(result.duration_seconds),
                "mean_overwhelmed_triggers": float(result.overwhelmed_triggers),
                "mean_impatient_triggers": float(result.impatient_triggers),
                "mean_distracted_triggers": float(result.distracted_triggers),
            }
        )
    return metrics


def plot_baseline_metrics(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [int(row["epoch"]) for row in metrics]
    returns = [float(row["mean_return"]) for row in metrics]
    durations = [float(row["mean_duration_seconds"]) for row in metrics]
    overwhelmed = [float(row["mean_overwhelmed_triggers"]) for row in metrics]
    impatient = [float(row["mean_impatient_triggers"]) for row in metrics]
    distracted = [float(row["mean_distracted_triggers"]) for row in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax_return, ax_duration, ax_triggers = axes.flat

    ax_return.plot(runs, returns, label="return", linewidth=2)
    ax_return.set_title("Return")
    ax_return.set_xlabel("Run")
    ax_return.set_ylabel("Return")
    ax_return.grid(True, alpha=0.3)
    ax_return.legend()

    ax_duration.plot(runs, durations, color="tab:orange", linewidth=2)
    ax_duration.set_title("Guide Duration")
    ax_duration.set_xlabel("Run")
    ax_duration.set_ylabel("Seconds")
    ax_duration.grid(True, alpha=0.3)

    ax_triggers.plot(runs, overwhelmed, label="overwhelmed", linewidth=2)
    ax_triggers.plot(runs, impatient, label="impatient", linewidth=2)
    ax_triggers.plot(runs, distracted, label="distracted", linewidth=2)
    ax_triggers.set_title("Negative Trigger Counts")
    ax_triggers.set_xlabel("Run")
    ax_triggers.set_ylabel("Mean count")
    ax_triggers.grid(True, alpha=0.3)
    ax_triggers.legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_baseline(
    *,
    num_runs: int,
    seed: int,
    output_dir: Path,
    max_workers: int,
) -> list[dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = PolicySearchParams().to_theta()
    worker_count = min(int(num_runs), os.cpu_count() or 1, int(max_workers))
    evaluation_seeds = _sample_evaluation_seeds(int(num_runs), int(seed))
    tasks = [(theta, evaluation_seed, DEFAULT_N_HUMANS, False) for evaluation_seed in evaluation_seeds]

    if worker_count == 1:
        episode_results = [_evaluate_episode_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            episode_results = list(executor.map(_evaluate_episode_task, tasks))

    metrics = _build_run_metrics(episode_results)
    if not metrics:
        raise RuntimeError("Baseline evaluation completed without any episode results.")

    csv_path = output_dir / DEFAULT_CSV_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME

    returns = np.array([result.episode_return for result in episode_results], dtype=np.float64)
    durations = np.array([result.duration_seconds for result in episode_results], dtype=np.float64)
    overwhelmed = np.array(
        [result.overwhelmed_triggers for result in episode_results], dtype=np.float64
    )
    impatient = np.array(
        [result.impatient_triggers for result in episode_results], dtype=np.float64
    )
    distracted = np.array(
        [result.distracted_triggers for result in episode_results], dtype=np.float64
    )
    summary_payload = {
        "baseline_theta": [float(value) for value in theta],
        "baseline_policy_params": _policy_params_dict(theta),
        "seed": int(seed),
        "evaluation_seeds": [int(value) for value in evaluation_seeds],
        "num_runs": int(num_runs),
        "mean_return": float(np.mean(returns)),
        "best_return": float(np.max(returns)),
        "mean_duration_seconds": float(np.mean(durations)),
        "mean_overwhelmed_triggers": float(np.mean(overwhelmed)),
        "mean_impatient_triggers": float(np.mean(impatient)),
        "mean_distracted_triggers": float(np.mean(distracted)),
    }

    write_metrics_csv(metrics, csv_path)
    plot_baseline_metrics(metrics, plot_path)
    write_best_params_json(summary_payload, summary_path)

    print(f"Saved baseline metrics CSV to {csv_path}")
    print(f"Saved baseline metrics plot to {plot_path}")
    print(f"Saved baseline summary JSON to {summary_path}")
    print(
        "Baseline policy params: "
        f"slow_down_distance_m={float(summary_payload['baseline_policy_params']['slow_down_distance_m']):.3f}, "
        f"callback_distance_m={float(summary_payload['baseline_policy_params']['callback_distance_m']):.3f}, "
        f"callback_wait_seconds={float(summary_payload['baseline_policy_params']['callback_wait_seconds']):.3f}, "
        f"slowdown_speed_scale={float(summary_payload['baseline_policy_params']['slowdown_speed_scale']):.3f}, "
        f"mean_return={float(summary_payload['mean_return']):.3f}, "
        f"best_return={float(summary_payload['best_return']):.3f}, "
        f"mean_duration={float(summary_payload['mean_duration_seconds']):.3f}"
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the default baseline policy over multiple episodes."
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="Number of baseline episodes to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Master seed used to generate the per-run environment seeds for this evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where baseline_metrics.csv, baseline_metrics.png, and baseline_summary.json are written.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of worker processes used for parallel evaluation.",
    )
    args = parser.parse_args(argv)
    evaluate_baseline(
        num_runs=args.num_runs,
        seed=args.seed,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
