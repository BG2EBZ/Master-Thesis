from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from museum_env import MuseumEnv
from museum_env.policy_search_params import PolicySearchParams

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPOCHS = 30
DEFAULT_SAMPLES_PER_EPOCH = 30
DEFAULT_SEED = 42
DEFAULT_BETA = 0.02
DEFAULT_EVALUATION_SEEDS = (42,)
DEFAULT_N_HUMANS = 15
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "rwr_minimal"
DEFAULT_CSV_NAME = "training_metrics.csv"
DEFAULT_PLOT_NAME = "training_metrics.png"
METRIC_FIELDNAMES = (
    "epoch",
    "mean_return",
    "best_return",
    "success_rate",
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)
INITIAL_MU = np.array(
    [
        2.5,
        3.5,
        2.0,
        0.7,
    ],
    dtype=np.float64,
)
INITIAL_STD = np.array(
    [
        0.4,
        0.6,
        0.8,
        0.1,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    duration_seconds: float
    overwhelmed_triggers: int
    impatient_triggers: int
    distracted_triggers: int
    success: bool


@dataclass(frozen=True)
class ThetaEvaluation:
    mean_return: float
    success_rate: float
    mean_duration_seconds: float
    mean_overwhelmed_triggers: float
    mean_impatient_triggers: float
    mean_distracted_triggers: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def run_episode(
    env: MuseumEnv,
    theta: np.ndarray,
    seed: int,
    *,
    print_explanations: bool = True,
) -> EpisodeResult:
    env.set_policy_parameters(theta)
    _observation, _info = env.reset(seed=seed)

    terminated = False
    truncated = False
    step_info = {}
    explanation_started_count = 0
    explanation_finished_count = 0

    while not (terminated or truncated):
        observation, reward, terminated, truncated, step_info = env.step(None)

        # Print stage trainsition if enabled.
        if print_explanations:
            step = env.step_count
            sim_time = step * env.dt
            events = step_info["events"]

            if events.get("started_listen_wait"):
                explanation_started_count += 1
                label = "A" if explanation_started_count == 1 else "B"
                print(f"[t={sim_time:.3f}s step={step}] robot start explanation {label}")

            if events.get("completed_listen_wait"):
                explanation_finished_count += 1
                label = "A" if explanation_finished_count == 1 else "B"
                print(f"[t={sim_time:.3f}s step={step}] robot finish explanation {label}")

    episode_info = step_info["episode"]

    return EpisodeResult(
        episode_return=float(episode_info["return"]),
        duration_seconds=float(episode_info["duration_seconds"]),
        overwhelmed_triggers=int(episode_info["overwhelmed_triggers"]),
        impatient_triggers=int(episode_info["impatient_triggers"]),
        distracted_triggers=int(episode_info["distracted_triggers"]),
        success=episode_info["terminated_reason"] == "final_listen_ready",
    )


def _evaluate_theta_sample(task: tuple[np.ndarray, tuple[int, ...], int, bool]) -> ThetaEvaluation:
    theta, seeds, n_humans, print_explanations = task
    env = MuseumEnv(
        render_mode=None,
        enable_event_logs=False,
        n_humans=n_humans,
    )
    # Multi-seed evaluation
    try:
        episode_results = [
            run_episode(
                env,
                theta=theta,
                seed=seed,
                print_explanations=print_explanations,
            )
            for seed in seeds
        ]
    finally:
        env.close()

    return ThetaEvaluation(
        mean_return=float(np.mean([result.episode_return for result in episode_results])),
        success_rate=float(np.mean([result.success for result in episode_results])),
        mean_duration_seconds=float(np.mean([result.duration_seconds for result in episode_results])),
        mean_overwhelmed_triggers=float(
            np.mean([result.overwhelmed_triggers for result in episode_results])
        ),
        mean_impatient_triggers=float(
            np.mean([result.impatient_triggers for result in episode_results])
        ),
        mean_distracted_triggers=float(
            np.mean([result.distracted_triggers for result in episode_results])
        ),
    )

# Expectation-maximization update for Gaussian distribution
def update_distribution(
    theta_batch: np.ndarray,
    returns: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_returns = returns - np.max(returns)
    # Mapping weights to [0, 1]
    weights = np.exp(beta * shifted_returns)

    sum_weights = float(np.sum(weights))
    sum_weights_sq = float(np.sum(weights**2))
    # Unbiased variance estimator denominator
    denominator = sum_weights - (sum_weights_sq / sum_weights)

    # Weighted mean and std computation
    mu = weights @ theta_batch / sum_weights
    delta_sq = (theta_batch - mu) ** 2
    std = np.sqrt(weights @ delta_sq / max(denominator, 1e-8))

    return mu, std


def write_metrics_csv(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        for row in metrics:
            writer.writerow({field: row[field] for field in METRIC_FIELDNAMES})


def plot_training_metrics(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    epochs = [int(row["epoch"]) for row in metrics]
    mean_returns = [float(row["mean_return"]) for row in metrics]
    best_returns = [float(row["best_return"]) for row in metrics]
    success_rates = [float(row["success_rate"]) for row in metrics]
    mean_durations = [float(row["mean_duration_seconds"]) for row in metrics]
    mean_overwhelmed = [float(row["mean_overwhelmed_triggers"]) for row in metrics]
    mean_impatient = [float(row["mean_impatient_triggers"]) for row in metrics]
    mean_distracted = [float(row["mean_distracted_triggers"]) for row in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_return, ax_success, ax_duration, ax_triggers = axes.flat

    ax_return.plot(epochs, mean_returns, label="mean_return", linewidth=2)
    ax_return.plot(epochs, best_returns, label="best_return", linewidth=2)
    ax_return.set_title("Return")
    ax_return.set_xlabel("Epoch")
    ax_return.set_ylabel("Return")
    ax_return.grid(True, alpha=0.3)
    ax_return.legend()

    ax_success.plot(epochs, success_rates, color="tab:green", linewidth=2)
    ax_success.set_title("Success Rate")
    ax_success.set_xlabel("Epoch")
    ax_success.set_ylabel("Rate")
    ax_success.set_ylim(0.0, 1.0)
    ax_success.grid(True, alpha=0.3)

    ax_duration.plot(epochs, mean_durations, color="tab:orange", linewidth=2)
    ax_duration.set_title("Guide Duration")
    ax_duration.set_xlabel("Epoch")
    ax_duration.set_ylabel("Seconds")
    ax_duration.grid(True, alpha=0.3)

    ax_triggers.plot(epochs, mean_overwhelmed, label="overwhelmed", linewidth=2)
    ax_triggers.plot(epochs, mean_impatient, label="impatient", linewidth=2)
    ax_triggers.plot(epochs, mean_distracted, label="distracted", linewidth=2)
    ax_triggers.set_title("Negative Trigger Counts")
    ax_triggers.set_xlabel("Epoch")
    ax_triggers.set_ylabel("Mean count")
    ax_triggers.grid(True, alpha=0.3)
    ax_triggers.legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(
    *,
    epochs: int,
    samples_per_epoch: int,
    seed: int,
    output_dir: Path,
) -> list[dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=seed)
    mu = INITIAL_MU.copy()
    std = INITIAL_STD.copy()
    metrics: list[dict[str, float | int]] = []
    max_workers = min(samples_per_epoch, os.cpu_count() or 1)
    # Only print if running single-process
    print_explanations = max_workers == 1

    for epoch_idx in range(epochs):
        raw_theta_batch = rng.normal(
            loc=mu,
            scale=std,
            size=(samples_per_epoch, len(mu)),
        )
        theta_batch = np.array(
            [PolicySearchParams.from_theta(theta).to_theta() for theta in raw_theta_batch],
            dtype=np.float64,
        )
        tasks = [
            (theta, DEFAULT_EVALUATION_SEEDS, DEFAULT_N_HUMANS, print_explanations)
            for theta in theta_batch
        ]

        if max_workers == 1:
            evaluations = [_evaluate_theta_sample(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                evaluations = list(executor.map(_evaluate_theta_sample, tasks))

        returns = np.array(
            [evaluation.mean_return for evaluation in evaluations],
            dtype=np.float64,
        )
        mu, std = update_distribution(
            theta_batch=theta_batch,
            returns=returns,
            beta=DEFAULT_BETA,
        )

        epoch_metrics = {
            "epoch": int(epoch_idx + 1),
            "mean_return": float(np.mean([item.mean_return for item in evaluations])),
            "best_return": float(np.max([item.mean_return for item in evaluations])),
            "success_rate": float(np.mean([item.success_rate for item in evaluations])),
            "mean_duration_seconds": float(
                np.mean([item.mean_duration_seconds for item in evaluations])
            ),
            "mean_overwhelmed_triggers": float(
                np.mean([item.mean_overwhelmed_triggers for item in evaluations])
            ),
            "mean_impatient_triggers": float(
                np.mean([item.mean_impatient_triggers for item in evaluations])
            ),
            "mean_distracted_triggers": float(
                np.mean([item.mean_distracted_triggers for item in evaluations])
            ),
        }
        metrics.append(epoch_metrics)
        print(
            f"epoch={epoch_metrics['epoch']:02d} "
            f"mean_return={float(epoch_metrics['mean_return']):.3f} "
            f"best_return={float(epoch_metrics['best_return']):.3f} "
            f"success_rate={float(epoch_metrics['success_rate']):.3f} "
            f"mean_duration={float(epoch_metrics['mean_duration_seconds']):.3f}"
        )

    csv_path = output_dir / DEFAULT_CSV_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME
    write_metrics_csv(metrics, csv_path)
    plot_training_metrics(metrics, plot_path)

    print(f"Saved metrics CSV to {csv_path}")
    print(f"Saved metrics plot to {plot_path}")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a minimal RWR policy search loop.")
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=_positive_int,
        default=DEFAULT_SAMPLES_PER_EPOCH,
        help="Number of sampled policies per epoch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for policy sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where training_metrics.csv and training_metrics.png are written.",
    )
    args = parser.parse_args(argv)
    train(
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
