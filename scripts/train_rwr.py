from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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

from museum_env import MuseumEnv
from train.evaluation_seeds import FIXED_EVALUATION_SEEDS
from train.plot_utils import compute_mean_confidence_band
from train.rwr import (
    DEFAULT_EPISODE_REWARD_WEIGHTS,
    EpisodeRewardWeights,
    RWRRewardWrapper,
    plot_exploration_metrics,
    plot_learning_curve_metrics,
    plot_training_metrics,
    summarize_theta,
    theta_to_guide_config,
)

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DEFAULT_EPOCHS = 5
DEFAULT_SAMPLES_PER_EPOCH = 5
DEFAULT_SEED = 42
DEFAULT_BETA = 0.2
DEFAULT_EPOCH_TRAIN_SEED_COUNT = 1
DEFAULT_N_HUMANS = 15
DEFAULT_MAX_WORKERS = 10
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "runs" / f"rwr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_CSV_NAME = "training_metrics.csv"
DEFAULT_PLOT_NAME = "training_metrics.png"
DEFAULT_EXPLORATION_PLOT_NAME = "exploration_metrics.png"
DEFAULT_BEST_PARAMS_NAME = "best_params.json"
DEFAULT_LEARNING_CURVE_RAW_CSV_NAME = "learning_curve_raw.csv"
DEFAULT_LEARNING_CURVE_PLOT_NAME = "learning_curve_plot.png"
DEFAULT_LEARNING_CURVE_SUMMARY_NAME = "learning_curve_summary.json"
DEFAULT_N_LEARNING_SEEDS = 1
DEFAULT_N_EVAL_SEEDS = min(20, len(FIXED_EVALUATION_SEEDS))
METRIC_FIELDNAMES = (
    "epoch",
    "mean_return",
    "best_return",
    "std_0",
    "std_1",
    "std_2",
    "std_3",
    "std_4",
    # "std_5",  # Temporarily disabled: callback_same_person_cooldown_seconds
    "distribution_entropy",
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)
LEARNING_CURVE_RAW_FIELDNAMES = (
    "learning_seed",
    "epoch",
    "mean_return",
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
        # 20.0,  # Temporarily disabled: callback_same_person_cooldown_seconds
    ],
    dtype=np.float64,
)
INITIAL_STD = np.array(
    [
        0.4,
        0.6,
        0.8,
        0.1,
        0.05,
        # 5.0,  # Temporarily disabled: callback_same_person_cooldown_seconds
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


@dataclass(frozen=True)
class SingleSeedTrainingResult:
    metrics: list[dict[str, float | int]]
    best_theta_seen: np.ndarray
    best_evaluation: ThetaEvaluation
    best_return_seen: float
    best_epoch: int
    best_sample_index: int
    final_mu: np.ndarray
    final_std: np.ndarray
    learning_curve_rows: list[dict[str, float | int]]


_CACHED_ENV: RWRRewardWrapper | None = None
_CACHED_ENV_N_HUMANS: int | None = None
_CACHED_ENV_REWARD_CONFIG: EpisodeRewardWeights | None = None


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


def _resolve_reward_config(reward_config: EpisodeRewardWeights | None) -> EpisodeRewardWeights:
    return DEFAULT_EPISODE_REWARD_WEIGHTS if reward_config is None else reward_config


def _get_cached_env(
    n_humans: int,
    reward_config: EpisodeRewardWeights | None = None,
) -> RWRRewardWrapper:
    global _CACHED_ENV, _CACHED_ENV_N_HUMANS, _CACHED_ENV_REWARD_CONFIG
    requested_n_humans = int(n_humans)
    requested_reward_config = _resolve_reward_config(reward_config)
    if _CACHED_ENV is None:
        _CACHED_ENV = RWRRewardWrapper(
            MuseumEnv(
                render_mode=None,
                enable_event_logs=False,
                n_humans=requested_n_humans,
            ),
            reward_weights=requested_reward_config,
        )
        _CACHED_ENV_N_HUMANS = requested_n_humans
        _CACHED_ENV_REWARD_CONFIG = requested_reward_config
        return _CACHED_ENV

    if (
        _CACHED_ENV_N_HUMANS != requested_n_humans
        or _CACHED_ENV_REWARD_CONFIG != requested_reward_config
    ):
        _close_cached_env()
        _CACHED_ENV = RWRRewardWrapper(
            MuseumEnv(
                render_mode=None,
                enable_event_logs=False,
                n_humans=requested_n_humans,
            ),
            reward_weights=requested_reward_config,
        )
        _CACHED_ENV_N_HUMANS = requested_n_humans
        _CACHED_ENV_REWARD_CONFIG = requested_reward_config
    return _CACHED_ENV


def _close_cached_env() -> None:
    global _CACHED_ENV, _CACHED_ENV_N_HUMANS, _CACHED_ENV_REWARD_CONFIG
    if _CACHED_ENV is not None:
        _CACHED_ENV.close()
    _CACHED_ENV = None
    _CACHED_ENV_N_HUMANS = None
    _CACHED_ENV_REWARD_CONFIG = None


atexit.register(_close_cached_env)


def run_episode(
    env: RWRRewardWrapper,
    theta: np.ndarray,
    seed: int,
    *,
    print_explanations: bool = True,
) -> EpisodeResult:
    base_env = env.unwrapped
    base_env.set_guide_behavior_config(theta_to_guide_config(theta))
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
            step = base_env.step_count
            sim_time = step * base_env.dt
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


def _evaluate_episode_task(
    task: tuple[np.ndarray, int, int, bool]
    | tuple[np.ndarray, int, int, bool, EpisodeRewardWeights | None]
) -> EpisodeResult:
    if len(task) == 4:
        theta, seed, n_humans, print_explanations = task
        reward_config = None
    else:
        theta, seed, n_humans, print_explanations, reward_config = task
    env = _get_cached_env(n_humans, reward_config=reward_config)
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


def write_learning_curve_raw_csv(
    rows: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_RAW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in LEARNING_CURVE_RAW_FIELDNAMES})


def _build_learning_curve_row(
    *,
    learning_seed: int,
    epoch: int,
    evaluation: ThetaEvaluation,
) -> dict[str, float | int]:
    return {
        "learning_seed": int(learning_seed),
        "epoch": int(epoch),
        "mean_return": float(evaluation.mean_return),
        "mean_duration_seconds": float(evaluation.mean_duration_seconds),
        "mean_overwhelmed_triggers": float(evaluation.mean_overwhelmed_triggers),
        "mean_impatient_triggers": float(evaluation.mean_impatient_triggers),
        "mean_distracted_triggers": float(evaluation.mean_distracted_triggers),
    }


def _build_learning_curve_return_matrix(
    rows: Sequence[dict[str, float | int]],
) -> tuple[np.ndarray, list[int], list[int]]:
    learning_seeds = sorted({int(row["learning_seed"]) for row in rows})
    epochs = sorted({int(row["epoch"]) for row in rows})
    seed_to_idx = {seed: idx for idx, seed in enumerate(learning_seeds)}
    epoch_to_idx = {epoch: idx for idx, epoch in enumerate(epochs)}
    matrix = np.full((len(learning_seeds), len(epochs)), np.nan, dtype=np.float64)
    for row in rows:
        matrix[seed_to_idx[int(row["learning_seed"])], epoch_to_idx[int(row["epoch"])]] = float(
            row["mean_return"]
        )
    if np.isnan(matrix).any():
        raise ValueError("Learning curve rows do not form a complete seed x epoch matrix.")
    return matrix, learning_seeds, epochs


def write_learning_curve_summary_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _policy_params_dict(theta: np.ndarray) -> dict[str, float]:
    return summarize_theta(theta)


def write_best_params_json(payload: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _run_episode_batch(
    tasks: Sequence[tuple[Any, ...]],
    executor: ProcessPoolExecutor | None,
) -> list[EpisodeResult]:
    if executor is None:
        return [_evaluate_episode_task(task) for task in tasks]
    return list(executor.map(_evaluate_episode_task, tasks))


def _evaluate_theta_on_seeds(
    theta: np.ndarray,
    *,
    seeds: Sequence[int],
    n_humans: int,
    reward_config: EpisodeRewardWeights | None,
    executor: ProcessPoolExecutor | None,
) -> list[EpisodeResult]:
    eval_tasks = [
        (
            np.asarray(theta, dtype=np.float64),
            int(eval_seed),
            int(n_humans),
            False,
            reward_config,
        )
        for eval_seed in seeds
    ]
    return _run_episode_batch(eval_tasks, executor)


def _train_single_learning_seed(
    *,
    epochs: int,
    samples_per_epoch: int,
    seed: int,
    beta: float,
    train_seeds_per_epoch: int,
    n_humans: int,
    reward_config: EpisodeRewardWeights | None,
    evaluation_seeds: Sequence[int] | None = None,
) -> SingleSeedTrainingResult:
    resolved_reward_config = _resolve_reward_config(reward_config)
    master_seed = int(seed)
    theta_seed_sequence, train_seed_sequence = np.random.SeedSequence(master_seed).spawn(2)
    theta_rng = np.random.default_rng(theta_seed_sequence)
    train_seed_rng = np.random.default_rng(train_seed_sequence)
    epoch_training_seeds = _build_epoch_training_seed_schedule(
        train_seed_rng,
        epochs=epochs,
        seeds_per_epoch=int(train_seeds_per_epoch),
    )
    mu = INITIAL_MU.copy()
    std = INITIAL_STD.copy()
    metrics: list[dict[str, float | int]] = []
    learning_curve_rows: list[dict[str, float | int]] = []
    best_theta_seen: np.ndarray | None = None
    best_evaluation: ThetaEvaluation | None = None
    best_return_seen = float("-inf")
    best_epoch = 0
    best_sample_index = 0
    max_workers = min(samples_per_epoch, os.cpu_count() or 1, DEFAULT_MAX_WORKERS)
    print_explanations = max_workers == 1
    initial_theta = INITIAL_MU.copy()

    def _record_learning_curve_epoch(
        *,
        epoch: int,
        theta: np.ndarray,
        executor: ProcessPoolExecutor | None,
    ) -> None:
        if evaluation_seeds is None:
            return
        evaluation = _aggregate_episode_results(
            _evaluate_theta_on_seeds(
                np.asarray(theta, dtype=np.float64),
                seeds=evaluation_seeds,
                n_humans=int(n_humans),
                reward_config=resolved_reward_config,
                executor=executor,
            )
        )
        learning_curve_rows.append(
            _build_learning_curve_row(
                learning_seed=int(master_seed),
                epoch=int(epoch),
                evaluation=evaluation,
            )
        )

    def _run_training_loop(
        *,
        executor: ProcessPoolExecutor | None,
    ) -> None:
        nonlocal best_epoch, best_evaluation, best_return_seen, best_sample_index, best_theta_seen, mu, std
        _record_learning_curve_epoch(epoch=0, theta=initial_theta, executor=executor)
        for epoch_idx in range(epochs):
            sampling_std = np.array(std, dtype=np.float64, copy=True)
            theta_batch = theta_rng.normal(
                loc=mu,
                scale=sampling_std,
                size=(samples_per_epoch, len(mu)),
            )
            epoch_training_seed_batch = epoch_training_seeds[epoch_idx]
            episode_tasks = [
                (theta, int(train_seed), int(n_humans), print_explanations, resolved_reward_config)
                for theta in theta_batch
                for train_seed in epoch_training_seed_batch
            ]
            episode_results = _run_episode_batch(episode_tasks, executor)
            seed_count = len(epoch_training_seed_batch)
            evaluations = [
                _aggregate_episode_results(episode_results[idx : idx + seed_count])
                for idx in range(0, len(episode_results), seed_count)
            ]

            returns = np.array([evaluation.mean_return for evaluation in evaluations], dtype=np.float64)
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
                beta=float(beta),
            )
            _record_learning_curve_epoch(epoch=int(epoch_idx + 1), theta=mu, executor=executor)

            epoch_metrics = {
                "epoch": int(epoch_idx + 1),
                "mean_return": float(np.mean([item.mean_return for item in evaluations])),
                "best_return": float(np.max([item.mean_return for item in evaluations])),
                "std_0": float(sampling_std[0]),
                "std_1": float(sampling_std[1]),
                "std_2": float(sampling_std[2]),
                "std_3": float(sampling_std[3]),
                "std_4": float(sampling_std[4]),
                # "std_5": float(sampling_std[5]),  # Temporarily disabled
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
            _run_training_loop(executor=None)
        finally:
            _close_cached_env()
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            _run_training_loop(executor=executor)

    if best_theta_seen is None or best_evaluation is None:
        raise RuntimeError("Training completed without evaluating any parameter samples.")
    return SingleSeedTrainingResult(
        metrics=metrics,
        best_theta_seen=best_theta_seen,
        best_evaluation=best_evaluation,
        best_return_seen=float(best_return_seen),
        best_epoch=int(best_epoch),
        best_sample_index=int(best_sample_index),
        final_mu=np.array(mu, dtype=np.float64, copy=True),
        final_std=np.array(std, dtype=np.float64, copy=True),
        learning_curve_rows=learning_curve_rows,
    )


def train(
    *,
    epochs: int,
    samples_per_epoch: int,
    seed: int,
    output_dir: Path,
    beta: float = DEFAULT_BETA,
    train_seeds_per_epoch: int = DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    n_humans: int = DEFAULT_N_HUMANS,
    reward_config: EpisodeRewardWeights | None = None,
) -> list[dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _train_single_learning_seed(
        epochs=int(epochs),
        samples_per_epoch=int(samples_per_epoch),
        seed=int(seed),
        beta=float(beta),
        train_seeds_per_epoch=int(train_seeds_per_epoch),
        n_humans=int(n_humans),
        reward_config=reward_config,
        evaluation_seeds=None,
    )

    csv_path = output_dir / DEFAULT_CSV_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME
    exploration_plot_path = output_dir / DEFAULT_EXPLORATION_PLOT_NAME
    best_params_path = output_dir / DEFAULT_BEST_PARAMS_NAME

    best_params_payload = {
        "best_theta_seen": [float(value) for value in result.best_theta_seen],
        "best_policy_params": _policy_params_dict(result.best_theta_seen),
        "final_theta": [float(value) for value in result.best_theta_seen],
        "final_policy_params": _policy_params_dict(result.best_theta_seen),
        "best_return": float(result.best_return_seen),
        "best_mean_duration_seconds": float(result.best_evaluation.mean_duration_seconds),
        "best_mean_overwhelmed_triggers": float(result.best_evaluation.mean_overwhelmed_triggers),
        "best_mean_impatient_triggers": float(result.best_evaluation.mean_impatient_triggers),
        "best_mean_distracted_triggers": float(result.best_evaluation.mean_distracted_triggers),
        "best_epoch": int(result.best_epoch),
        "best_sample_index_within_epoch": int(result.best_sample_index),
        "final_mu": [float(value) for value in result.final_mu],
        "final_std": [float(value) for value in result.final_std],
    }
    write_metrics_csv(result.metrics, csv_path)
    plot_training_metrics(result.metrics, plot_path)
    plot_exploration_metrics(result.metrics, exploration_plot_path)
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
        f"callback_same_person_cooldown_seconds="
        f"{float(final_policy_params['callback_same_person_cooldown_seconds']):.3f}, "
        f"best_return={float(best_params_payload['best_return']):.3f}, "
        f"epoch={int(best_params_payload['best_epoch'])}, "
        f"sample={int(best_params_payload['best_sample_index_within_epoch'])}"
    )
    return result.metrics


def train_across_learning_seeds(
    *,
    epochs: int,
    samples_per_epoch: int,
    seed: int,
    output_dir: Path,
    beta: float = DEFAULT_BETA,
    train_seeds_per_epoch: int = DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    n_humans: int = DEFAULT_N_HUMANS,
    reward_config: EpisodeRewardWeights | None = None,
    n_learning_seeds: int = DEFAULT_N_LEARNING_SEEDS,
    n_eval_seeds: int = DEFAULT_N_EVAL_SEEDS,
) -> list[dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_n_learning_seeds = int(n_learning_seeds)
    resolved_n_eval_seeds = int(n_eval_seeds)
    if resolved_n_learning_seeds <= 0:
        raise ValueError("n_learning_seeds must be positive")
    if resolved_n_eval_seeds <= 0:
        raise ValueError("n_eval_seeds must be positive")
    if resolved_n_eval_seeds > len(FIXED_EVALUATION_SEEDS):
        raise ValueError(
            f"n_eval_seeds={resolved_n_eval_seeds} exceeds fixed evaluation seed count "
            f"of {len(FIXED_EVALUATION_SEEDS)}."
        )

    evaluation_seeds = [int(seed_value) for seed_value in FIXED_EVALUATION_SEEDS[:resolved_n_eval_seeds]]
    learning_seed_sequences = np.random.SeedSequence(int(seed)).spawn(resolved_n_learning_seeds)
    learning_seeds = [
        int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        for seed_sequence in learning_seed_sequences
    ]

    learning_curve_rows: list[dict[str, float | int]] = []
    for learning_seed in learning_seeds:
        seed_result = _train_single_learning_seed(
            epochs=int(epochs),
            samples_per_epoch=int(samples_per_epoch),
            seed=int(learning_seed),
            beta=float(beta),
            train_seeds_per_epoch=int(train_seeds_per_epoch),
            n_humans=int(n_humans),
            reward_config=reward_config,
            evaluation_seeds=evaluation_seeds,
        )
        learning_curve_rows.extend(seed_result.learning_curve_rows)

    raw_csv_path = output_dir / DEFAULT_LEARNING_CURVE_RAW_CSV_NAME
    plot_path = output_dir / DEFAULT_LEARNING_CURVE_PLOT_NAME
    summary_path = output_dir / DEFAULT_LEARNING_CURVE_SUMMARY_NAME
    write_learning_curve_raw_csv(learning_curve_rows, raw_csv_path)

    return_matrix, ordered_learning_seeds, ordered_epochs = _build_learning_curve_return_matrix(
        learning_curve_rows
    )
    band = compute_mean_confidence_band(return_matrix)
    plot_learning_curve_metrics(
        epochs=ordered_epochs,
        return_matrix=return_matrix,
        output_path=plot_path,
    )
    summary_payload = {
        "learning_seeds": [int(value) for value in ordered_learning_seeds],
        "evaluation_seeds": [int(value) for value in evaluation_seeds],
        "epochs": [int(value) for value in ordered_epochs],
        "n_learning_seeds": int(resolved_n_learning_seeds),
        "n_eval_seeds": int(resolved_n_eval_seeds),
        "mean_return": [float(value) for value in band.mean],
        "ci95_low_return": [float(value) for value in band.low],
        "ci95_high_return": [float(value) for value in band.high],
        "learning_curve_raw_csv": str(raw_csv_path),
        "learning_curve_plot": str(plot_path),
    }
    write_learning_curve_summary_json(summary_payload, summary_path)

    print(f"Saved learning curve raw CSV to {raw_csv_path}")
    print(f"Saved learning curve plot to {plot_path}")
    print(f"Saved learning curve summary to {summary_path}")
    return learning_curve_rows


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
        "--beta",
        type=_positive_float,
        default=DEFAULT_BETA,
        help="RWR reward-weight temperature used in the distribution update.",
    )
    parser.add_argument(
        "--train-seeds-per-epoch",
        type=_positive_int,
        default=DEFAULT_EPOCH_TRAIN_SEED_COUNT,
        help="Number of rollout seeds used to evaluate each sampled theta per epoch.",
    )
    parser.add_argument(
        "--n-humans",
        type=_positive_int,
        default=DEFAULT_N_HUMANS,
        help="Number of humans to simulate during training rollouts.",
    )
    parser.add_argument(
        "--n-learning-seeds",
        type=_positive_int,
        default=DEFAULT_N_LEARNING_SEEDS,
        help="Number of independent learning seeds used for benchmark learning curves.",
    )
    parser.add_argument(
        "--n-eval-seeds",
        type=_positive_int,
        default=DEFAULT_N_EVAL_SEEDS,
        help="Number of fixed evaluation seeds used to estimate each epoch point.",
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
        help="Directory where training artifacts and optional learning-curve outputs are written.",
    )
    args = parser.parse_args(argv)
    reward_config = EpisodeRewardWeights(
        time_penalty_per_second=float(args.time_penalty_per_second),
        overwhelmed_trigger_penalty=float(args.overwhelmed_trigger_penalty),
        impatient_trigger_penalty=float(args.impatient_trigger_penalty),
        distracted_trigger_penalty=float(args.distracted_trigger_penalty),
    )
    if int(args.n_learning_seeds) == 1:
        train(
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            output_dir=args.output_dir,
            beta=float(args.beta),
            train_seeds_per_epoch=int(args.train_seeds_per_epoch),
            n_humans=int(args.n_humans),
            reward_config=reward_config,
        )
    else:
        train_across_learning_seeds(
            epochs=int(args.epochs),
            samples_per_epoch=int(args.samples_per_epoch),
            seed=int(args.seed),
            output_dir=args.output_dir,
            beta=float(args.beta),
            train_seeds_per_epoch=int(args.train_seeds_per_epoch),
            n_humans=int(args.n_humans),
            reward_config=reward_config,
            n_learning_seeds=int(args.n_learning_seeds),
            n_eval_seeds=int(args.n_eval_seeds),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
