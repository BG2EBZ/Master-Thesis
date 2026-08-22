from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from museum_env.guide_config import GuideBehaviorConfig
from train.common.artifacts import build_dense_metric_matrix, write_csv_rows, write_json
from train.common.evaluation_seeds import FIXED_EVALUATION_SEEDS
from train.common.parallel import build_process_pool
from train.common.plot_utils import compute_mean_confidence_band
from train.common.rollout import run_episode_batch
from train.policy_search.algorithm import (
    EPPOConfig,
    ThetaEvaluation,
    aggregate_episode_results,
    diagonal_gaussian_entropy,
    update_distribution_by_algorithm,
)
from train.policy_search.defaults import (
    DEFAULT_BEST_PARAMS_NAME,
    DEFAULT_BETA,
    DEFAULT_CSV_NAME,
    DEFAULT_EPS,
    DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    DEFAULT_EXPLORATION_PLOT_NAME,
    DEFAULT_LEARNING_CURVE_MATRIX_CSV_NAME,
    DEFAULT_LEARNING_CURVE_PLOT_NAME,
    DEFAULT_LEARNING_CURVE_RAW_CSV_NAME,
    DEFAULT_LEARNING_CURVE_SUMMARY_NAME,
    DEFAULT_MAX_WORKERS,
    DEFAULT_N_EVAL_SEEDS,
    DEFAULT_N_HUMANS,
    DEFAULT_N_LEARNING_SEEDS,
    DEFAULT_PLOT_NAME,
    DEFAULT_SEED_PLAN_NAME,
    INITIAL_MU,
    INITIAL_STD,
    LEARNING_CURVE_RAW_FIELDNAMES,
    METRIC_FIELDNAMES,
)
from train.policy_search.evaluation import (
    _evaluate_episode_task,
    build_learning_curve_row as _build_learning_curve_row,
    evaluate_baseline_learning_curve_rows as _evaluate_baseline_learning_curve_rows,
    evaluate_theta_on_seeds as _evaluate_theta_on_seeds,
)
from train.policy_search.plotting import (
    plot_exploration_metrics,
    plot_learning_curve_metrics,
    plot_training_metrics,
)
from train.policy_search.policy_codec import (
    guide_config_to_theta,
    summarize_theta,
)
from train.policy_search.rewarding import (
    DEFAULT_EPISODE_REWARD_WEIGHTS,
    EpisodeRewardWeights,
)
from train.policy_search.seed_plan import (
    SeedPlan,
    build_seed_plan,
    copy_seed_schedule,
    seed_plan_hash,
    validate_seed_plan,
    write_seed_plan,
)
from train.policy_search.schedules import (
    build_epoch_training_seed_schedule as _build_epoch_training_seed_schedule,
    build_seed_schedule as _build_seed_schedule,
    resolve_worker_count as _resolve_worker_count,
)

SUPPORTED_POLICY_SEARCH_ALGORITHMS = ("rwr", "reps", "eppo")


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
    epoch_training_seeds: list[list[int]]
    learning_curve_rows: list[dict[str, float | int | str]]
    learning_curve_eval_seed_schedule: list[list[int]]


def _policy_params_dict(theta: np.ndarray) -> dict[str, float]:
    return summarize_theta(theta)


def _normalize_algorithm(algorithm: str) -> str:
    resolved_algorithm = str(algorithm).lower()
    if resolved_algorithm not in SUPPORTED_POLICY_SEARCH_ALGORITHMS:
        raise ValueError(
            "algorithm must be one of "
            + ", ".join(repr(item) for item in SUPPORTED_POLICY_SEARCH_ALGORITHMS)
        )
    return resolved_algorithm


def _algorithm_plot_label(algorithm: str) -> str:
    return _normalize_algorithm(algorithm).upper()


def _eppo_config_payload(config: EPPOConfig) -> dict[str, float | int]:
    return {
        key: int(value) if key in ("n_epochs_policy", "batch_size") else float(value)
        for key, value in asdict(config).items()
    }


def _build_learning_curve_matrix_rows(
    *,
    return_matrix: np.ndarray,
    learning_seeds: Sequence[int],
    epochs: Sequence[int],
) -> tuple[list[dict[str, float | int]], tuple[str, ...]]:
    epoch_fieldnames = tuple(f"epoch_{int(epoch)}" for epoch in epochs)
    rows: list[dict[str, float | int]] = []
    for row_idx, learning_seed in enumerate(learning_seeds):
        row: dict[str, float | int] = {"learning_seed": int(learning_seed)}
        for column_idx, epoch_fieldname in enumerate(epoch_fieldnames):
            row[epoch_fieldname] = float(return_matrix[row_idx, column_idx])
        rows.append(row)
    return rows, ("learning_seed", *epoch_fieldnames)


def _validate_training_seed_schedule(
    schedule: Sequence[Sequence[int]],
    *,
    epochs: int,
    train_seeds_per_epoch: int,
) -> None:
    if len(schedule) != int(epochs):
        raise ValueError("epoch_training_seeds must contain one row per epoch.")
    for epoch_seeds in schedule:
        if len(epoch_seeds) != int(train_seeds_per_epoch):
            raise ValueError(
                "epoch_training_seeds rows must match train_seeds_per_epoch."
            )


def _validate_learning_curve_seed_schedule(
    schedule: Sequence[Sequence[int]],
    *,
    epochs: int,
    n_eval_seeds: int,
) -> None:
    if len(schedule) != int(epochs) + 1:
        raise ValueError(
            "learning_curve_eval_seed_schedule must contain epochs + 1 rows."
        )
    for epoch_seeds in schedule:
        if len(epoch_seeds) != int(n_eval_seeds):
            raise ValueError(
                "learning_curve_eval_seed_schedule rows must match n_eval_seeds."
            )


def _train_single_learning_seed(
    *,
    epochs: int,
    samples_per_epoch: int,
    seed: int,
    beta: float,
    eps: float,
    algorithm: str,
    train_seeds_per_epoch: int,
    n_humans: int,
    reward_config: EpisodeRewardWeights | None,
    max_workers: int,
    eppo_config: EPPOConfig | None = None,
    learning_curve_eval_seeds_per_epoch: int | None = None,
    epoch_training_seeds: Sequence[Sequence[int]] | None = None,
    learning_curve_eval_seed_schedule: Sequence[Sequence[int]] | None = None,
) -> SingleSeedTrainingResult:
    resolved_reward_config = (
        DEFAULT_EPISODE_REWARD_WEIGHTS if reward_config is None else reward_config
    )
    resolved_algorithm = _normalize_algorithm(algorithm)
    resolved_eppo_config = EPPOConfig() if eppo_config is None else eppo_config
    master_seed = int(seed)
    # Generate three independent SeedSequences for theta sampling, training, and learning curve evaluation
    theta_seed_sequence, train_seed_sequence, curve_seed_sequence = np.random.SeedSequence(
        master_seed
    ).spawn(3)
    theta_rng = np.random.default_rng(theta_seed_sequence)
    train_seed_rng = np.random.default_rng(train_seed_sequence)
    learning_curve_eval_seed_rng = np.random.default_rng(curve_seed_sequence)
    resolved_epoch_training_seeds = (
        _build_epoch_training_seed_schedule(
            train_seed_rng,
            epochs=epochs,
            seeds_per_epoch=int(train_seeds_per_epoch),
            excluded_seeds=FIXED_EVALUATION_SEEDS,
        )
        if epoch_training_seeds is None
        else copy_seed_schedule(epoch_training_seeds)
    )
    _validate_training_seed_schedule(
        resolved_epoch_training_seeds,
        epochs=int(epochs),
        train_seeds_per_epoch=int(train_seeds_per_epoch),
    )
    resolved_learning_curve_eval_seed_schedule: list[list[int]] = []
    if learning_curve_eval_seeds_per_epoch is not None:
        if int(learning_curve_eval_seeds_per_epoch) <= 0:
            raise ValueError("learning_curve_eval_seeds_per_epoch must be positive")
        resolved_learning_curve_eval_seed_schedule = (
            _build_seed_schedule(
                learning_curve_eval_seed_rng,
                count=int(epochs) + 1,
                seeds_per_item=int(learning_curve_eval_seeds_per_epoch),
                excluded_seeds=FIXED_EVALUATION_SEEDS,
            )
            if learning_curve_eval_seed_schedule is None
            else copy_seed_schedule(learning_curve_eval_seed_schedule)
        )
        _validate_learning_curve_seed_schedule(
            resolved_learning_curve_eval_seed_schedule,
            epochs=int(epochs),
            n_eval_seeds=int(learning_curve_eval_seeds_per_epoch),
        )
    mu = INITIAL_MU.copy()
    std = INITIAL_STD.copy()
    metrics: list[dict[str, float | int]] = []
    learning_curve_rows: list[dict[str, float | int | str]] = []
    best_theta_seen: np.ndarray | None = None
    best_evaluation: ThetaEvaluation | None = None
    best_return_seen = float("-inf")
    best_epoch = 0
    best_sample_index = 0
    max_training_tasks_per_epoch = int(samples_per_epoch) * int(train_seeds_per_epoch)
    worker_count = _resolve_worker_count(
        task_count=max_training_tasks_per_epoch,
        max_workers=int(max_workers),
    )
    print_explanations = worker_count == 1
    initial_theta = INITIAL_MU.copy()

    def _record_learning_curve_epoch(
        *,
        epoch: int,
        theta: np.ndarray,
        eval_seeds: Sequence[int],
        executor: ProcessPoolExecutor | None,
    ) -> None:
        evaluation = aggregate_episode_results(
            _evaluate_theta_on_seeds(
                np.asarray(theta, dtype=np.float64),
                seeds=eval_seeds,
                n_humans=int(n_humans),
                reward_config=resolved_reward_config,
                executor=executor,
            )
        )
        learning_curve_rows.append(
            _build_learning_curve_row(
                policy=resolved_algorithm,
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
        if resolved_learning_curve_eval_seed_schedule:
            _record_learning_curve_epoch(
                epoch=0,
                theta=initial_theta,
                eval_seeds=resolved_learning_curve_eval_seed_schedule[0],
                executor=executor,
            )
        for epoch_idx in range(epochs):
            sampling_std = np.array(std, dtype=np.float64, copy=True)
            theta_batch = theta_rng.normal(
                loc=mu,
                scale=sampling_std,
                size=(samples_per_epoch, len(mu)),
            )
            epoch_training_seed_batch = resolved_epoch_training_seeds[epoch_idx]
            episode_tasks = [
                (theta, int(train_seed), int(n_humans), print_explanations, resolved_reward_config)
                for theta in theta_batch
                for train_seed in epoch_training_seed_batch
            ]
            episode_results = run_episode_batch(episode_tasks, executor, _evaluate_episode_task)
            seed_count = len(epoch_training_seed_batch)
            evaluations = [
                aggregate_episode_results(episode_results[idx : idx + seed_count])
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

            mu, std = update_distribution_by_algorithm(
                algorithm=resolved_algorithm,
                theta_batch=theta_batch,
                returns=returns,
                beta=float(beta),
                eps=float(eps),
                current_mu=mu,
                current_std=std,
                eppo_config=resolved_eppo_config,
            )
            if resolved_learning_curve_eval_seed_schedule:
                _record_learning_curve_epoch(
                    epoch=int(epoch_idx + 1),
                    theta=mu,
                    eval_seeds=resolved_learning_curve_eval_seed_schedule[int(epoch_idx + 1)],
                    executor=executor,
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
                "distribution_entropy": diagonal_gaussian_entropy(sampling_std),
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

    if worker_count == 1:
        try:
            _run_training_loop(executor=None)
        finally:
            from train.common.rollout import close_cached_env

            close_cached_env()
    else:
        with build_process_pool(max_workers=worker_count) as executor:
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
        epoch_training_seeds=copy_seed_schedule(resolved_epoch_training_seeds),
        learning_curve_rows=learning_curve_rows,
        learning_curve_eval_seed_schedule=copy_seed_schedule(
            resolved_learning_curve_eval_seed_schedule
        ),
    )


def train(
    *,
    epochs: int,
    samples_per_epoch: int,
    seed: int,
    output_dir: Path,
    beta: float = DEFAULT_BETA,
    eps: float = DEFAULT_EPS,
    algorithm: str = "rwr",
    train_seeds_per_epoch: int = DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    n_humans: int = DEFAULT_N_HUMANS,
    reward_config: EpisodeRewardWeights | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    eppo_config: EPPOConfig | None = None,
) -> list[dict[str, float | int]]:
    resolved_algorithm = _normalize_algorithm(algorithm)
    resolved_eppo_config = EPPOConfig() if eppo_config is None else eppo_config
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _train_single_learning_seed(
        epochs=int(epochs),
        samples_per_epoch=int(samples_per_epoch),
        seed=int(seed),
        beta=float(beta),
        eps=float(eps),
        algorithm=resolved_algorithm,
        train_seeds_per_epoch=int(train_seeds_per_epoch),
        n_humans=int(n_humans),
        reward_config=reward_config,
        max_workers=int(max_workers),
        eppo_config=resolved_eppo_config,
        learning_curve_eval_seeds_per_epoch=None,
    )

    csv_path = output_dir / DEFAULT_CSV_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME
    exploration_plot_path = output_dir / DEFAULT_EXPLORATION_PLOT_NAME
    best_params_path = output_dir / DEFAULT_BEST_PARAMS_NAME

    best_params_payload = {
        "algorithm": resolved_algorithm,
        "beta": float(beta),
        "eps": float(eps),
        "eppo_config": _eppo_config_payload(resolved_eppo_config),
        "best_theta_seen": [float(value) for value in result.best_theta_seen],
        "best_policy_params": _policy_params_dict(result.best_theta_seen),
        "final_theta": [float(value) for value in result.final_mu],
        "final_policy_params": _policy_params_dict(result.final_mu),
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
    write_csv_rows(result.metrics, METRIC_FIELDNAMES, csv_path)
    plot_training_metrics(result.metrics, plot_path)
    plot_exploration_metrics(result.metrics, exploration_plot_path)
    write_json(best_params_payload, best_params_path)

    print(f"Saved metrics CSV to {csv_path}")
    print(f"Saved metrics plot to {plot_path}")
    print(f"Saved exploration plot to {exploration_plot_path}")
    print(f"Saved best params JSON to {best_params_path}")
    final_policy_params = best_params_payload["final_policy_params"]
    print(
        f"Final {resolved_algorithm.upper()} policy params: "
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
    eps: float = DEFAULT_EPS,
    algorithm: str = "rwr",
    train_seeds_per_epoch: int = DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    n_humans: int = DEFAULT_N_HUMANS,
    reward_config: EpisodeRewardWeights | None = None,
    n_learning_seeds: int = DEFAULT_N_LEARNING_SEEDS,
    n_eval_seeds: int = DEFAULT_N_EVAL_SEEDS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    seed_plan: SeedPlan | None = None,
    eppo_config: EPPOConfig | None = None,
) -> list[dict[str, float | int | str]]:
    resolved_algorithm = _normalize_algorithm(algorithm)
    resolved_eppo_config = EPPOConfig() if eppo_config is None else eppo_config
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_n_learning_seeds = int(n_learning_seeds)
    resolved_n_eval_seeds = int(n_eval_seeds)
    if resolved_n_learning_seeds <= 0:
        raise ValueError("n_learning_seeds must be positive")
    if resolved_n_eval_seeds <= 0:
        raise ValueError("n_eval_seeds must be positive")

    resolved_seed_plan = (
        build_seed_plan(
            master_seed=int(seed),
            epochs=int(epochs),
            train_seeds_per_epoch=int(train_seeds_per_epoch),
            n_learning_seeds=resolved_n_learning_seeds,
            n_eval_seeds=resolved_n_eval_seeds,
        )
        if seed_plan is None
        else seed_plan
    )
    validate_seed_plan(
        resolved_seed_plan,
        epochs=int(epochs),
        train_seeds_per_epoch=int(train_seeds_per_epoch),
        n_learning_seeds=resolved_n_learning_seeds,
        n_eval_seeds=resolved_n_eval_seeds,
    )
    resolved_seed_plan_hash = seed_plan_hash(resolved_seed_plan)
    learning_seed_plans = list(resolved_seed_plan.learning_seed_plans)
    learning_seeds = [int(item.learning_seed) for item in learning_seed_plans]

    raw_csv_path = output_dir / DEFAULT_LEARNING_CURVE_RAW_CSV_NAME
    matrix_csv_path = output_dir / DEFAULT_LEARNING_CURVE_MATRIX_CSV_NAME
    plot_path = output_dir / DEFAULT_LEARNING_CURVE_PLOT_NAME
    summary_path = output_dir / DEFAULT_LEARNING_CURVE_SUMMARY_NAME
    seed_plan_path = output_dir / DEFAULT_SEED_PLAN_NAME
    write_seed_plan(resolved_seed_plan, seed_plan_path)
    print(f"Saved seed plan to {seed_plan_path} (sha256={resolved_seed_plan_hash})")

    learning_curve_rows: list[dict[str, float | int | str]] = []
    learning_seed_results: dict[int, SingleSeedTrainingResult] = {}
    eval_seed_schedule_by_learning_seed: dict[int, list[list[int]]] = {}
    for learning_seed_plan in learning_seed_plans:
        learning_seed = int(learning_seed_plan.learning_seed)
        seed_result = _train_single_learning_seed(
            epochs=int(epochs),
            samples_per_epoch=int(samples_per_epoch),
            seed=learning_seed,
            beta=float(beta),
            eps=float(eps),
            algorithm=resolved_algorithm,
            train_seeds_per_epoch=int(train_seeds_per_epoch),
            n_humans=int(n_humans),
            reward_config=reward_config,
            max_workers=int(max_workers),
            eppo_config=resolved_eppo_config,
            learning_curve_eval_seeds_per_epoch=resolved_n_eval_seeds,
            epoch_training_seeds=learning_seed_plan.train_seeds_by_epoch,
            learning_curve_eval_seed_schedule=learning_seed_plan.eval_seeds_by_epoch,
        )
        result_learning_seed = (
            int(seed_result.learning_curve_rows[0]["learning_seed"])
            if seed_result.learning_curve_rows
            else int(learning_seed)
        )
        learning_seed_results[result_learning_seed] = seed_result
        eval_seed_schedule_by_learning_seed[result_learning_seed] = (
            seed_result.learning_curve_eval_seed_schedule
        )
        learning_curve_rows.extend(seed_result.learning_curve_rows)
        write_csv_rows(learning_curve_rows, LEARNING_CURVE_RAW_FIELDNAMES, raw_csv_path)
        print(
            "Checkpointed learning curve raw CSV "
            f"after learning_seed={result_learning_seed} to {raw_csv_path}"
        )

    return_matrix, ordered_learning_seeds, ordered_epochs = build_dense_metric_matrix(
        learning_curve_rows,
        row_key="learning_seed",
        column_key="epoch",
        value_key="mean_return",
    )
    baseline_learning_curve_rows = _evaluate_baseline_learning_curve_rows(
        eval_seed_schedule_by_learning_seed=eval_seed_schedule_by_learning_seed,
        epochs=ordered_epochs,
        n_humans=int(n_humans),
        reward_config=reward_config,
        max_workers=int(max_workers),
    )
    all_learning_curve_rows = learning_curve_rows + baseline_learning_curve_rows

    write_csv_rows(all_learning_curve_rows, LEARNING_CURVE_RAW_FIELDNAMES, raw_csv_path)
    matrix_rows, matrix_fieldnames = _build_learning_curve_matrix_rows(
        return_matrix=return_matrix,
        learning_seeds=ordered_learning_seeds,
        epochs=ordered_epochs,
    )
    write_csv_rows(matrix_rows, matrix_fieldnames, matrix_csv_path)

    baseline_return_matrix, ordered_baseline_learning_seeds, baseline_epochs = build_dense_metric_matrix(
        baseline_learning_curve_rows,
        row_key="learning_seed",
        column_key="epoch",
        value_key="mean_return",
    )
    if baseline_epochs != ordered_epochs:
        raise ValueError("Baseline epochs do not match learned-policy learning curve epochs.")
    if ordered_baseline_learning_seeds != ordered_learning_seeds:
        raise ValueError("Baseline learning seeds do not match learned-policy learning seeds.")

    band = compute_mean_confidence_band(return_matrix)
    baseline_band = compute_mean_confidence_band(baseline_return_matrix)
    baseline_theta = guide_config_to_theta(GuideBehaviorConfig())
    plot_learning_curve_metrics(
        epochs=ordered_epochs,
        return_matrix=return_matrix,
        baseline_return_matrix=baseline_return_matrix,
        output_path=plot_path,
        learned_policy_label=_algorithm_plot_label(resolved_algorithm),
    )
    summary_payload = {
        "algorithm": resolved_algorithm,
        "master_seed": int(resolved_seed_plan.master_seed),
        "seed_plan_hash": resolved_seed_plan_hash,
        "seed_plan_json": str(seed_plan_path),
        "beta": float(beta),
        "eps": float(eps),
        "eppo_config": _eppo_config_payload(resolved_eppo_config),
        "learning_seeds": [int(value) for value in ordered_learning_seeds],
        "learning_curve_eval_seeds_by_learning_seed": [
            {
                "learning_seed": int(learning_seed),
                "epoch_eval_seeds": [
                    {
                        "epoch": int(epoch),
                        "seeds": [
                            int(seed)
                            for seed in eval_seed_schedule_by_learning_seed[int(learning_seed)][
                                epoch_idx
                            ]
                        ],
                    }
                    for epoch_idx, epoch in enumerate(ordered_epochs)
                ],
            }
            for learning_seed in ordered_learning_seeds
        ],
        "final_policy_params_by_learning_seed": [
            {
                "learning_seed": int(learning_seed),
                "best_theta_seen": [
                    float(value)
                    for value in learning_seed_results[int(learning_seed)].best_theta_seen
                ],
                "best_policy_params": _policy_params_dict(
                    learning_seed_results[int(learning_seed)].best_theta_seen
                ),
                "final_theta": [
                    float(value) for value in learning_seed_results[int(learning_seed)].final_mu
                ],
                "final_policy_params": _policy_params_dict(
                    learning_seed_results[int(learning_seed)].final_mu
                ),
                "final_mu": [
                    float(value) for value in learning_seed_results[int(learning_seed)].final_mu
                ],
                "final_std": [
                    float(value) for value in learning_seed_results[int(learning_seed)].final_std
                ],
                "best_return": float(
                    learning_seed_results[int(learning_seed)].best_return_seen
                ),
                "best_epoch": int(learning_seed_results[int(learning_seed)].best_epoch),
                "best_sample_index_within_epoch": int(
                    learning_seed_results[int(learning_seed)].best_sample_index
                ),
            }
            for learning_seed in ordered_learning_seeds
        ],
        "epochs": [int(value) for value in ordered_epochs],
        "n_learning_seeds": int(resolved_n_learning_seeds),
        "n_eval_seeds": int(resolved_n_eval_seeds),
        "mean_return": [float(value) for value in band.mean],
        "ci95_low_return": [float(value) for value in band.low],
        "ci95_high_return": [float(value) for value in band.high],
        "baseline_mean_return": [float(value) for value in baseline_band.mean],
        "baseline_ci95_low_return": [float(value) for value in baseline_band.low],
        "baseline_ci95_high_return": [float(value) for value in baseline_band.high],
        "baseline_policy_params": _policy_params_dict(baseline_theta),
        "learning_curve_raw_csv": str(raw_csv_path),
        "learning_curve_matrix_csv": str(matrix_csv_path),
        "learning_curve_plot": str(plot_path),
    }
    write_json(summary_payload, summary_path)

    print(f"Saved learning curve raw CSV to {raw_csv_path}")
    print(f"Saved learning curve matrix CSV to {matrix_csv_path}")
    print(f"Saved learning curve plot to {plot_path}")
    print(f"Saved learning curve summary to {summary_path}")
    return all_learning_curve_rows
