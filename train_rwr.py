from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Keep BLAS/OpenMP libraries from oversubscribing CPU cores across workers
# unless the caller explicitly sets a different thread count.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from museum_env import MuseumEnv
from museum_env.evaluation_seeds import FIXED_EVALUATION_SEEDS
from museum_env.policy_search_params import PolicySearchParams

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EPOCHS = 15
DEFAULT_SAMPLES_PER_EPOCH = 30
DEFAULT_SEED = 42
DEFAULT_BETA = 0.5
DEFAULT_EPOCH_TRAIN_SEED_COUNT = 3
DEFAULT_N_HUMANS = 15
DEFAULT_MAX_WORKERS = 10
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / f"rwr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_CSV_NAME = "training_metrics.csv"
DEFAULT_PLOT_NAME = "training_metrics.png"
DEFAULT_EXPLORATION_PLOT_NAME = "exploration_metrics.png"
DEFAULT_BEST_PARAMS_NAME = "best_params.json"
EXPLORATION_STD_FIELDNAMES = ("std_0", "std_1", "std_2", "std_3", "std_4")
EXPLORATION_PARAMETER_LABELS = (
    "slow_down_distance_m",
    "callback_distance_m",
    "callback_wait_seconds",
    "slowdown_speed_scale",
    "explanation_time_scale",
)
METRIC_FIELDNAMES = (
    "epoch",
    "mean_return",
    "best_return",
    "std_0",
    "std_1",
    "std_2",
    "std_3",
    "std_4",
    "distribution_entropy",
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
        1.0,
    ],
    dtype=np.float64,
)
INITIAL_STD = np.array(
    [
        0.4,
        0.6,
        0.8,
        0.1,
        0.08,
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
    mean_duration_seconds: float
    mean_overwhelmed_triggers: float
    mean_impatient_triggers: float
    mean_distracted_triggers: float


_CACHED_ENV: MuseumEnv | None = None
_CACHED_ENV_N_HUMANS: int | None = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _get_cached_env(n_humans: int) -> MuseumEnv:
    global _CACHED_ENV, _CACHED_ENV_N_HUMANS
    requested_n_humans = int(n_humans)
    if _CACHED_ENV is None:
        _CACHED_ENV = MuseumEnv(
            render_mode=None,
            enable_event_logs=False,
            n_humans=requested_n_humans,
        )
        _CACHED_ENV_N_HUMANS = requested_n_humans
        return _CACHED_ENV

    if _CACHED_ENV_N_HUMANS != requested_n_humans:
        raise RuntimeError(
            "Cached MuseumEnv was initialized with "
            f"n_humans={_CACHED_ENV_N_HUMANS}, but received n_humans={requested_n_humans}."
        )
    return _CACHED_ENV


def _close_cached_env() -> None:
    global _CACHED_ENV, _CACHED_ENV_N_HUMANS
    if _CACHED_ENV is not None:
        _CACHED_ENV.close()
    _CACHED_ENV = None
    _CACHED_ENV_N_HUMANS = None


atexit.register(_close_cached_env)


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


def _evaluate_episode_task(task: tuple[np.ndarray, int, int, bool]) -> EpisodeResult:
    theta, seed, n_humans, print_explanations = task
    env = _get_cached_env(n_humans)
    return run_episode(
        env,
        theta=theta,
        seed=seed,
        print_explanations=print_explanations,
    )


def _aggregate_episode_results(episode_results: Sequence[EpisodeResult]) -> ThetaEvaluation:
    return ThetaEvaluation(
        mean_return=float(np.mean([result.episode_return for result in episode_results])),
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


def _build_epoch_training_seed_schedule(
    rng: np.random.Generator,
    epochs: int,
    seeds_per_epoch: int,
) -> list[list[int]]:
    excluded_seed_set = {int(seed) for seed in FIXED_EVALUATION_SEEDS}
    epoch_training_seeds: list[list[int]] = []

    for _ in range(epochs):
        epoch_seeds: list[int] = []
        while len(epoch_seeds) < seeds_per_epoch:
            sampled_seed = int(rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
            if sampled_seed in excluded_seed_set:
                continue
            epoch_seeds.append(sampled_seed)
        epoch_training_seeds.append(epoch_seeds)

    return epoch_training_seeds


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


def _diagonal_gaussian_entropy(std: np.ndarray) -> float:
    clipped_std = np.maximum(np.asarray(std, dtype=np.float64), 1e-8)
    return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e * (clipped_std**2))))


def write_metrics_csv(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        for row in metrics:
            writer.writerow({field: row[field] for field in METRIC_FIELDNAMES})


def _policy_params_dict(theta: np.ndarray) -> dict[str, float]:
    params = PolicySearchParams.from_theta(theta)
    return {
        "slow_down_distance_m": float(params.slow_down_distance_m),
        "callback_distance_m": float(params.callback_distance_m),
        "callback_wait_seconds": float(params.callback_wait_seconds),
        "slowdown_speed_scale": float(params.slowdown_speed_scale),
        "explanation_time_scale": float(params.explanation_time_scale),
        "explanation_wait_seconds": float(params.explanation_wait_seconds),
    }


def write_best_params_json(payload: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def plot_training_metrics(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
    *,
    x_label: str = "Epoch",
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in metrics]
    mean_returns = [float(row["mean_return"]) for row in metrics]
    best_returns = [float(row["best_return"]) for row in metrics]
    mean_durations = [float(row["mean_duration_seconds"]) for row in metrics]
    mean_overwhelmed = [float(row["mean_overwhelmed_triggers"]) for row in metrics]
    mean_impatient = [float(row["mean_impatient_triggers"]) for row in metrics]
    mean_distracted = [float(row["mean_distracted_triggers"]) for row in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax_return, ax_duration, ax_triggers = axes.flat

    ax_return.plot(epochs, mean_returns, label="mean_return", linewidth=2)
    ax_return.plot(epochs, best_returns, label="best_return", linewidth=2)
    ax_return.set_title("Return")
    ax_return.set_xlabel(x_label)
    ax_return.set_ylabel("Return")
    ax_return.grid(True, alpha=0.3)
    ax_return.legend()

    ax_duration.plot(epochs, mean_durations, color="tab:orange", linewidth=2)
    ax_duration.set_title("Guide Duration")
    ax_duration.set_xlabel(x_label)
    ax_duration.set_ylabel("Seconds")
    ax_duration.grid(True, alpha=0.3)

    ax_triggers.plot(epochs, mean_overwhelmed, label="overwhelmed", linewidth=2)
    ax_triggers.plot(epochs, mean_impatient, label="impatient", linewidth=2)
    ax_triggers.plot(epochs, mean_distracted, label="distracted", linewidth=2)
    ax_triggers.set_title("Negative Trigger Counts")
    ax_triggers.set_xlabel(x_label)
    ax_triggers.set_ylabel("Mean count")
    ax_triggers.grid(True, alpha=0.3)
    ax_triggers.legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_exploration_metrics(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
    *,
    x_label: str = "Epoch",
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in metrics]
    std_series = {
        label: [float(row[field_name]) for row in metrics]
        for field_name, label in zip(EXPLORATION_STD_FIELDNAMES, EXPLORATION_PARAMETER_LABELS)
    }
    entropies = [float(row["distribution_entropy"]) for row in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_std, ax_entropy = axes.flat

    for label, values in std_series.items():
        ax_std.plot(epochs, values, label=label, linewidth=2)
    ax_std.set_title("Std Per Dimension")
    ax_std.set_xlabel(x_label)
    ax_std.set_ylabel("Std")
    ax_std.grid(True, alpha=0.3)
    ax_std.legend()

    ax_entropy.plot(epochs, entropies, color="tab:green", linewidth=2)
    ax_entropy.set_title("Distribution Entropy")
    ax_entropy.set_xlabel(x_label)
    ax_entropy.set_ylabel("Entropy")
    ax_entropy.grid(True, alpha=0.3)

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

    master_seed = int(seed)
    theta_seed_sequence, train_seed_sequence = np.random.SeedSequence(master_seed).spawn(2)
    theta_rng = np.random.default_rng(theta_seed_sequence)
    train_seed_rng = np.random.default_rng(train_seed_sequence)
    epoch_training_seeds = _build_epoch_training_seed_schedule(
        train_seed_rng,
        epochs=epochs,
        seeds_per_epoch=DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    )
    mu = INITIAL_MU.copy()
    std = INITIAL_STD.copy()
    metrics: list[dict[str, float | int]] = []
    best_theta_seen: np.ndarray | None = None
    best_evaluation: ThetaEvaluation | None = None
    best_return_seen = float("-inf")
    best_epoch = 0
    best_sample_index = 0
    max_workers = min(samples_per_epoch, os.cpu_count() or 1, DEFAULT_MAX_WORKERS)
    # Only print if running single-process
    print_explanations = max_workers == 1

    def _run_episode_batch(tasks, executor: ProcessPoolExecutor | None) -> list[EpisodeResult]:
        if executor is None:
            return [_evaluate_episode_task(task) for task in tasks]
        return list(executor.map(_evaluate_episode_task, tasks))

    def _run_training_loop(
        *,
        executor: ProcessPoolExecutor | None,
        start_epoch_idx: int,
        end_epoch_idx: int,
    ) -> None:
        nonlocal best_epoch, best_evaluation, best_return_seen, best_sample_index, best_theta_seen, mu, std
        for epoch_idx in range(start_epoch_idx, end_epoch_idx):
            sampling_std = np.array(std, dtype=np.float64, copy=True)
            theta_batch = theta_rng.normal(
                loc=mu,
                scale=sampling_std,
                size=(samples_per_epoch, len(mu)),
            )
            epoch_training_seed_batch = epoch_training_seeds[epoch_idx]
            episode_tasks = [
                (theta, int(seed), DEFAULT_N_HUMANS, print_explanations)
                for theta in theta_batch
                for seed in epoch_training_seed_batch
            ]
            episode_results = _run_episode_batch(episode_tasks, executor)
            seed_count = len(epoch_training_seed_batch)
            evaluations = [
                _aggregate_episode_results(episode_results[idx : idx + seed_count])
                for idx in range(0, len(episode_results), seed_count)
            ]

            returns = np.array(
                [evaluation.mean_return for evaluation in evaluations],
                dtype=np.float64,
            )
            best_idx = int(np.argmax(returns))
            best_return_this_epoch = float(returns[best_idx])
            if best_return_this_epoch > best_return_seen:
                best_theta_seen = np.array(theta_batch[best_idx], dtype=np.float64, copy=True)
                best_evaluation = evaluations[best_idx]
                best_return_seen = best_return_this_epoch
                best_epoch = int(epoch_idx + 1)
                best_sample_index = int(best_idx + 1)
            mu, std = update_distribution(
                theta_batch=theta_batch,
                returns=returns,
                beta=DEFAULT_BETA,
            )

            epoch_metrics = {
                "epoch": int(epoch_idx + 1),
                "mean_return": float(np.mean([item.mean_return for item in evaluations])),
                "best_return": float(np.max([item.mean_return for item in evaluations])),
                "std_0": float(sampling_std[0]),
                "std_1": float(sampling_std[1]),
                "std_2": float(sampling_std[2]),
                "std_3": float(sampling_std[3]),
                "std_4": float(sampling_std[4]),
                "distribution_entropy": _diagonal_gaussian_entropy(sampling_std),
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
                f"mean_duration={float(epoch_metrics['mean_duration_seconds']):.3f}"
            )

    if max_workers == 1:
        try:
            _run_training_loop(executor=None, start_epoch_idx=0, end_epoch_idx=epochs)
        finally:
            _close_cached_env()
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            _run_training_loop(
                executor=executor,
                start_epoch_idx=0,
                end_epoch_idx=epochs,
            )

    csv_path = output_dir / DEFAULT_CSV_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME
    exploration_plot_path = output_dir / DEFAULT_EXPLORATION_PLOT_NAME
    best_params_path = output_dir / DEFAULT_BEST_PARAMS_NAME
    if best_theta_seen is None or best_evaluation is None:
        raise RuntimeError("Training completed without evaluating any parameter samples.")

    best_params_payload = {
        "best_theta_seen": [float(value) for value in best_theta_seen],
        "best_policy_params": _policy_params_dict(best_theta_seen),
        "final_theta": [float(value) for value in best_theta_seen],
        "final_policy_params": _policy_params_dict(best_theta_seen),
        "best_return": float(best_return_seen),
        "best_mean_duration_seconds": float(best_evaluation.mean_duration_seconds),
        "best_mean_overwhelmed_triggers": float(best_evaluation.mean_overwhelmed_triggers),
        "best_mean_impatient_triggers": float(best_evaluation.mean_impatient_triggers),
        "best_mean_distracted_triggers": float(best_evaluation.mean_distracted_triggers),
        "best_epoch": int(best_epoch),
        "best_sample_index_within_epoch": int(best_sample_index),
        # These remain optimizer-space values; the environment clips them on application.
        "final_mu": [float(value) for value in mu],
        "final_std": [float(value) for value in std],
    }
    write_metrics_csv(metrics, csv_path)
    plot_training_metrics(metrics, plot_path)
    plot_exploration_metrics(metrics, exploration_plot_path)
    write_best_params_json(best_params_payload, best_params_path)

    print(f"Saved metrics CSV to {csv_path}")
    print(f"Saved metrics plot to {plot_path}")
    print(f"Saved exploration plot to {exploration_plot_path}")
    print(f"Saved best params JSON to {best_params_path}")
    final_policy_params = best_params_payload["final_policy_params"]
    print(
        "Final policy params: "
        f"slow_down_distance_m={float(final_policy_params['slow_down_distance_m']):.3f}, "
        f"callback_distance_m={float(final_policy_params['callback_distance_m']):.3f}, "
        f"callback_wait_seconds={float(final_policy_params['callback_wait_seconds']):.3f}, "
        f"slowdown_speed_scale={float(final_policy_params['slowdown_speed_scale']):.3f}, "
        f"explanation_time_scale={float(final_policy_params['explanation_time_scale']):.3f}, "
        f"explanation_wait_seconds={float(final_policy_params['explanation_wait_seconds']):.3f}, "
        f"best_return={float(best_params_payload['best_return']):.3f}, "
        f"epoch={int(best_params_payload['best_epoch'])}, "
        f"sample={int(best_params_payload['best_sample_index_within_epoch'])}"
    )
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
        help="Experiment master seed for policy sampling and rollout seeding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where training metrics, exploration plots, and best_params.json are written.",
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
