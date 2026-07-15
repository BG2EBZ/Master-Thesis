from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from museum_env.evaluation_seeds import FIXED_EVALUATION_SEEDS
from museum_env.policy_search_params import PolicySearchParams
from museum_env.reward import RewardConfig
from train_rwr import EpisodeResult, _evaluate_episode_task

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NUM_RUNS = 20
DEFAULT_N_HUMANS = 15
DEFAULT_MAX_WORKERS = 10
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_CSV_NAME = "comparison_metrics.csv"
DEFAULT_PLOT_NAME = "comparison_plot.png"
DEFAULT_SUMMARY_NAME = "comparison_summary.json"
COMPARISON_METRIC_FIELDNAMES = (
    "run",
    "seed",
    "baseline_return",
    "comparison_return",
    "baseline_duration_seconds",
    "comparison_duration_seconds",
    "baseline_overwhelmed_triggers",
    "comparison_overwhelmed_triggers",
    "baseline_impatient_triggers",
    "comparison_impatient_triggers",
    "baseline_distracted_triggers",
    "comparison_distracted_triggers",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_learned_params_payload(learned_params_json: Path) -> dict[str, object]:
    with learned_params_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected {learned_params_json} to contain a JSON object, received {type(payload)!r}."
        )
    return payload


def _load_comparison_theta(learned_params_payload: dict[str, object]) -> np.ndarray:
    raw_theta = learned_params_payload.get("final_theta")
    if raw_theta is None:
        raw_theta = learned_params_payload.get("best_theta_seen")
    if raw_theta is None:
        raise ValueError("learned params JSON must contain final_theta or best_theta_seen.")
    if not isinstance(raw_theta, list):
        raise ValueError("learned theta payload must be a JSON array.")
    return np.asarray(raw_theta, dtype=np.float64)


def _select_evaluation_seeds(num_runs: int) -> list[int]:
    max_runs = len(FIXED_EVALUATION_SEEDS)
    if num_runs > max_runs:
        raise ValueError(
            f"num_runs={num_runs} exceeds fixed evaluation seed count of {max_runs}."
        )
    return [int(value) for value in FIXED_EVALUATION_SEEDS[:num_runs]]


def _policy_params_dict(theta: np.ndarray) -> dict[str, float]:
    params = PolicySearchParams.from_theta(theta)
    return {
        "slow_down_distance_m": float(params.slow_down_distance_m),
        "callback_distance_m": float(params.callback_distance_m),
        "callback_wait_seconds": float(params.callback_wait_seconds),
        "slowdown_speed_scale": float(params.slowdown_speed_scale),
        "explanation_time_scale": float(params.explanation_time_scale),
        "explanation_wait_seconds": float(params.explanation_wait_seconds),
        "callback_same_person_cooldown_seconds": float(
            params.callback_same_person_cooldown_seconds
        ),
    }


def _build_comparison_run_metrics(
    baseline_results: Sequence[EpisodeResult],
    comparison_results: Sequence[EpisodeResult],
    evaluation_seeds: Sequence[int],
) -> list[dict[str, float | int]]:
    if not (
        len(baseline_results) == len(comparison_results) == len(evaluation_seeds)
    ):
        raise ValueError("baseline_results, comparison_results, and evaluation_seeds must align")

    metrics: list[dict[str, float | int]] = []
    for run_idx, (seed, baseline, comparison) in enumerate(
        zip(evaluation_seeds, baseline_results, comparison_results),
        start=1,
    ):
        metrics.append(
            {
                "run": int(run_idx),
                "seed": int(seed),
                "baseline_return": float(baseline.episode_return),
                "comparison_return": float(comparison.episode_return),
                "baseline_duration_seconds": float(baseline.duration_seconds),
                "comparison_duration_seconds": float(comparison.duration_seconds),
                "baseline_overwhelmed_triggers": int(baseline.overwhelmed_triggers),
                "comparison_overwhelmed_triggers": int(comparison.overwhelmed_triggers),
                "baseline_impatient_triggers": int(baseline.impatient_triggers),
                "comparison_impatient_triggers": int(comparison.impatient_triggers),
                "baseline_distracted_triggers": int(baseline.distracted_triggers),
                "comparison_distracted_triggers": int(comparison.distracted_triggers),
            }
        )
    return metrics


def _summarize_policy_results(results: Sequence[EpisodeResult]) -> dict[str, float]:
    returns = np.array([item.episode_return for item in results], dtype=np.float64)
    durations = np.array([item.duration_seconds for item in results], dtype=np.float64)
    overwhelmed = np.array([item.overwhelmed_triggers for item in results], dtype=np.float64)
    impatient = np.array([item.impatient_triggers for item in results], dtype=np.float64)
    distracted = np.array([item.distracted_triggers for item in results], dtype=np.float64)
    return {
        "mean_return": float(np.mean(returns)),
        "best_return": float(np.max(returns)),
        "mean_duration_seconds": float(np.mean(durations)),
        "mean_overwhelmed_triggers": float(np.mean(overwhelmed)),
        "mean_impatient_triggers": float(np.mean(impatient)),
        "mean_distracted_triggers": float(np.mean(distracted)),
    }


def _prefix_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in summary.items()}


def write_metrics_csv(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_METRIC_FIELDNAMES)
        writer.writeheader()
        for row in metrics:
            writer.writerow({field: row[field] for field in COMPARISON_METRIC_FIELDNAMES})


def write_summary_json(payload: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def plot_comparison_metrics(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [int(row["run"]) for row in metrics]
    baseline_returns = [float(row["baseline_return"]) for row in metrics]
    comparison_returns = [float(row["comparison_return"]) for row in metrics]
    baseline_durations = [float(row["baseline_duration_seconds"]) for row in metrics]
    comparison_durations = [float(row["comparison_duration_seconds"]) for row in metrics]
    baseline_overwhelmed = [float(row["baseline_overwhelmed_triggers"]) for row in metrics]
    comparison_overwhelmed = [float(row["comparison_overwhelmed_triggers"]) for row in metrics]
    baseline_impatient = [float(row["baseline_impatient_triggers"]) for row in metrics]
    comparison_impatient = [float(row["comparison_impatient_triggers"]) for row in metrics]
    baseline_distracted = [float(row["baseline_distracted_triggers"]) for row in metrics]
    comparison_distracted = [float(row["comparison_distracted_triggers"]) for row in metrics]
    return_deltas = [
        comparison - baseline
        for baseline, comparison in zip(baseline_returns, comparison_returns)
    ]
    return_delta_colors = [
        "tab:green" if delta >= 0.0 else "tab:red"
        for delta in return_deltas
    ]
    baseline_color = "tab:blue"
    comparison_color = "tab:orange"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    (
        ax_return,
        ax_duration,
        ax_return_delta,
        ax_overwhelmed,
        ax_impatient,
        ax_distracted,
    ) = axes.flat

    ax_return.plot(runs, baseline_returns, label="baseline", color=baseline_color, linewidth=2)
    ax_return.plot(
        runs,
        comparison_returns,
        label="comparison",
        color=comparison_color,
        linewidth=2,
    )
    ax_return.set_title("Return")
    ax_return.set_xlabel("Run")
    ax_return.set_ylabel("Return")
    ax_return.grid(True, alpha=0.3)
    ax_return.legend()

    ax_duration.plot(
        runs,
        baseline_durations,
        label="baseline",
        color=baseline_color,
        linewidth=2,
    )
    ax_duration.plot(
        runs,
        comparison_durations,
        label="comparison",
        color=comparison_color,
        linewidth=2,
    )
    ax_duration.set_title("Guide Duration")
    ax_duration.set_xlabel("Run")
    ax_duration.set_ylabel("Seconds")
    ax_duration.grid(True, alpha=0.3)
    ax_duration.legend()

    ax_return_delta.bar(runs, return_deltas, color=return_delta_colors, alpha=0.85)
    ax_return_delta.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_return_delta.set_title("Return Delta (comparison - baseline)")
    ax_return_delta.set_xlabel("Run")
    ax_return_delta.set_ylabel("Delta Return")
    ax_return_delta.grid(True, alpha=0.3)

    ax_overwhelmed.plot(
        runs,
        baseline_overwhelmed,
        label="baseline",
        color=baseline_color,
        linewidth=2,
    )
    ax_overwhelmed.plot(
        runs,
        comparison_overwhelmed,
        label="comparison",
        color=comparison_color,
        linestyle="--",
        linewidth=2,
    )
    ax_overwhelmed.set_title("Overwhelmed Triggers")
    ax_overwhelmed.set_xlabel("Run")
    ax_overwhelmed.set_ylabel("Count")
    ax_overwhelmed.grid(True, alpha=0.3)
    ax_overwhelmed.legend()

    ax_impatient.plot(
        runs,
        baseline_impatient,
        label="baseline",
        color=baseline_color,
        linewidth=2,
    )
    ax_impatient.plot(
        runs,
        comparison_impatient,
        label="comparison",
        color=comparison_color,
        linestyle="--",
        linewidth=2,
    )
    ax_impatient.set_title("Impatient Triggers")
    ax_impatient.set_xlabel("Run")
    ax_impatient.set_ylabel("Count")
    ax_impatient.grid(True, alpha=0.3)
    ax_impatient.legend()

    ax_distracted.plot(
        runs,
        baseline_distracted,
        label="baseline",
        color=baseline_color,
        linewidth=2,
    )
    ax_distracted.plot(
        runs,
        comparison_distracted,
        label="comparison",
        color=comparison_color,
        linestyle="--",
        linewidth=2,
    )
    ax_distracted.set_title("Distracted Triggers")
    ax_distracted.set_xlabel("Run")
    ax_distracted.set_ylabel("Count")
    ax_distracted.grid(True, alpha=0.3)
    ax_distracted.legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_baseline(
    *,
    learned_params_json: Path,
    num_runs: int,
    output_dir: Path,
    max_workers: int,
    n_humans: int = DEFAULT_N_HUMANS,
    baseline_theta: np.ndarray | None = None,
    reward_config: RewardConfig | None = None,
) -> list[dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    learned_params_payload = _load_learned_params_payload(learned_params_json)
    resolved_baseline_theta = (
        PolicySearchParams().to_theta()
        if baseline_theta is None
        else np.asarray(baseline_theta, dtype=np.float64)
    )
    comparison_theta = _load_comparison_theta(learned_params_payload)
    evaluation_seeds = _select_evaluation_seeds(num_runs)
    baseline_tasks = [
        (resolved_baseline_theta, int(evaluation_seed), int(n_humans), False, reward_config)
        for evaluation_seed in evaluation_seeds
    ]
    comparison_tasks = [
        (comparison_theta, int(evaluation_seed), int(n_humans), False, reward_config)
        for evaluation_seed in evaluation_seeds
    ]
    episode_tasks = baseline_tasks + comparison_tasks
    worker_count = min(len(episode_tasks), max_workers, os.cpu_count() or 1)

    if worker_count <= 1:
        episode_results = [_evaluate_episode_task(task) for task in episode_tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            episode_results = list(executor.map(_evaluate_episode_task, episode_tasks))

    baseline_results = episode_results[:num_runs]
    comparison_results = episode_results[num_runs:]
    metrics = _build_comparison_run_metrics(
        baseline_results=baseline_results,
        comparison_results=comparison_results,
        evaluation_seeds=evaluation_seeds,
    )

    csv_path = output_dir / DEFAULT_CSV_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    write_metrics_csv(metrics, csv_path)
    plot_comparison_metrics(metrics, plot_path)

    baseline_summary = _summarize_policy_results(baseline_results)
    comparison_summary = _summarize_policy_results(comparison_results)
    delta_summary = {
        key: float(comparison_summary[key] - baseline_summary[key])
        for key in baseline_summary
    }
    comparison_win_count = sum(
        comparison.episode_return > baseline.episode_return
        for baseline, comparison in zip(baseline_results, comparison_results)
    )
    summary_payload: dict[str, object] = {
        "baseline_theta": [float(value) for value in resolved_baseline_theta],
        "comparison_theta": [float(value) for value in comparison_theta],
        "baseline_policy_params": _policy_params_dict(resolved_baseline_theta),
        "comparison_policy_params": _policy_params_dict(comparison_theta),
        "learned_params_json": str(learned_params_json),
        "evaluation_seeds": [int(value) for value in evaluation_seeds],
        "num_runs": int(num_runs),
    }
    summary_payload.update(_prefix_summary("baseline", baseline_summary))
    summary_payload.update(_prefix_summary("comparison", comparison_summary))
    write_summary_json(summary_payload, summary_path)

    print(f"Saved comparison metrics CSV to {csv_path}")
    print(f"Saved comparison plot to {plot_path}")
    print(f"Saved comparison summary to {summary_path}")
    print(
        "Baseline vs comparison: "
        f"baseline_mean_return={float(summary_payload['baseline_mean_return']):.3f}, "
        f"comparison_mean_return={float(summary_payload['comparison_mean_return']):.3f}, "
        f"baseline_best_return={float(summary_payload['baseline_best_return']):.3f}, "
        f"comparison_best_return={float(summary_payload['comparison_best_return']):.3f}"
    )
    print(
        "Mean triggers: "
        f"baseline_overwhelmed={float(summary_payload['baseline_mean_overwhelmed_triggers']):.3f}, "
        f"comparison_overwhelmed={float(summary_payload['comparison_mean_overwhelmed_triggers']):.3f}, "
        f"baseline_impatient={float(summary_payload['baseline_mean_impatient_triggers']):.3f}, "
        f"comparison_impatient={float(summary_payload['comparison_mean_impatient_triggers']):.3f}, "
        f"baseline_distracted={float(summary_payload['baseline_mean_distracted_triggers']):.3f}, "
        f"comparison_distracted={float(summary_payload['comparison_mean_distracted_triggers']):.3f}"
    )
    print(
        "Mean deltas (comparison - baseline): "
        f"return={delta_summary['mean_return']:.3f}, "
        f"duration={delta_summary['mean_duration_seconds']:.3f}, "
        f"overwhelmed={delta_summary['mean_overwhelmed_triggers']:.3f}, "
        f"impatient={delta_summary['mean_impatient_triggers']:.3f}, "
        f"distracted={delta_summary['mean_distracted_triggers']:.3f}"
    )
    print(
        "Return win count: "
        f"comparison_wins={comparison_win_count}/{num_runs}, "
        f"baseline_wins={num_runs - comparison_win_count}/{num_runs}"
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and learned policy on fixed held-out seeds."
    )
    parser.add_argument(
        "--learned-params-json",
        type=Path,
        required=True,
        help="Training artifact JSON containing the learned theta.",
    )
    parser.add_argument(
        "--num-runs",
        type=_positive_int,
        default=DEFAULT_NUM_RUNS,
        help="Number of shared evaluation runs for baseline and comparison policy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where comparison_metrics.csv, comparison_plot.png, and comparison_summary.json are written.",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum worker processes for rollout evaluation.",
    )
    parser.add_argument(
        "--n-humans",
        type=_positive_int,
        default=DEFAULT_N_HUMANS,
        help="Number of humans to simulate during evaluation rollouts.",
    )
    args = parser.parse_args(argv)
    evaluate_baseline(
        learned_params_json=args.learned_params_json,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        n_humans=int(args.n_humans),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
