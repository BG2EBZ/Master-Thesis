from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np

from museum_env.guide_config import GuideBehaviorConfig
from train.common.rollout import (
    EpisodeResult,
    close_cached_env,
    get_cached_env,
    run_episode,
    run_episode_batch,
)
from train.policy_search.algorithm import ThetaEvaluation, aggregate_episode_results
from train.policy_search.policy_codec import guide_config_to_theta, theta_to_guide_config
from train.policy_search.rewarding import (
    DEFAULT_EPISODE_REWARD_WEIGHTS,
    EpisodeRewardWeights,
    RWRRewardWrapper,
)
from train.policy_search.schedules import resolve_worker_count


def _wrap_env(env, reward_config: EpisodeRewardWeights) -> RWRRewardWrapper:
    return RWRRewardWrapper(env, reward_weights=reward_config)


def _evaluate_episode_task(
    task: tuple[np.ndarray, int, int, bool]
    | tuple[np.ndarray, int, int, bool, EpisodeRewardWeights | None]
) -> EpisodeResult:
    if len(task) == 4:
        theta, seed, n_humans, print_explanations = task
        reward_config = None
    else:
        theta, seed, n_humans, print_explanations, reward_config = task
    env = get_cached_env(
        n_humans,
        reward_config=reward_config,
        default_reward_config=DEFAULT_EPISODE_REWARD_WEIGHTS,
        wrapper_factory=_wrap_env,
    )
    return run_episode(
        env,
        theta=theta,
        seed=seed,
        theta_to_guide_config=theta_to_guide_config,
        print_explanations=print_explanations,
    )


def build_learning_curve_row(
    *,
    policy: str = "rwr",
    learning_seed: int,
    epoch: int,
    evaluation: ThetaEvaluation,
) -> dict[str, float | int | str]:
    return {
        "policy": str(policy),
        "learning_seed": int(learning_seed),
        "evaluation_seed": -1,
        "epoch": int(epoch),
        "mean_return": float(evaluation.mean_return),
        "mean_duration_seconds": float(evaluation.mean_duration_seconds),
        "mean_overwhelmed_triggers": float(evaluation.mean_overwhelmed_triggers),
        "mean_impatient_triggers": float(evaluation.mean_impatient_triggers),
        "mean_distracted_triggers": float(evaluation.mean_distracted_triggers),
    }


def evaluate_theta_on_seeds(
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
    return run_episode_batch(eval_tasks, executor, _evaluate_episode_task)


def _build_baseline_learning_curve_row(
    *,
    learning_seed: int,
    epoch: int,
    episode_results: Sequence[EpisodeResult],
) -> dict[str, float | int | str]:
    return build_learning_curve_row(
        policy="baseline",
        learning_seed=int(learning_seed),
        epoch=int(epoch),
        evaluation=aggregate_episode_results(episode_results),
    )


def evaluate_baseline_learning_curve_rows(
    *,
    eval_seed_schedule_by_learning_seed: dict[int, list[list[int]]],
    epochs: Sequence[int],
    n_humans: int,
    reward_config: EpisodeRewardWeights | None,
    max_workers: int,
) -> list[dict[str, float | int | str]]:
    baseline_theta = guide_config_to_theta(GuideBehaviorConfig())
    ordered_epochs = [int(epoch) for epoch in epochs]
    ordered_learning_seeds = sorted(eval_seed_schedule_by_learning_seed)
    for learning_seed in ordered_learning_seeds:
        seed_schedule = eval_seed_schedule_by_learning_seed[int(learning_seed)]
        if len(seed_schedule) != len(ordered_epochs):
            raise ValueError("Baseline eval seed schedule does not match learning curve epochs.")

    total_episode_count = sum(
        len(eval_seed_schedule_by_learning_seed[int(learning_seed)][epoch_idx])
        for learning_seed in ordered_learning_seeds
        for epoch_idx in range(len(ordered_epochs))
    )
    worker_count = resolve_worker_count(
        task_count=total_episode_count,
        max_workers=int(max_workers),
    )

    def _evaluate_baseline_learning_seeds(
        executor: ProcessPoolExecutor | None,
    ) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for learning_seed_idx, learning_seed in enumerate(ordered_learning_seeds):
            seed_curve_points: list[tuple[int, list[int]]] = []
            seed_tasks: list[tuple[np.ndarray, int, int, bool, EpisodeRewardWeights | None]] = []
            for epoch_idx, epoch in enumerate(ordered_epochs):
                eval_seeds = [
                    int(seed)
                    for seed in eval_seed_schedule_by_learning_seed[int(learning_seed)][
                        epoch_idx
                    ]
                ]
                seed_curve_points.append((int(epoch), eval_seeds))
                seed_tasks.extend(
                    (
                        np.asarray(baseline_theta, dtype=np.float64),
                        int(eval_seed),
                        int(n_humans),
                        False,
                        reward_config,
                    )
                    for eval_seed in eval_seeds
                )

            episode_results = run_episode_batch(seed_tasks, executor, _evaluate_episode_task)
            result_offset = 0
            for curve_epoch, eval_seeds in seed_curve_points:
                seed_count = len(eval_seeds)
                rows.append(
                    _build_baseline_learning_curve_row(
                        learning_seed=int(learning_seed),
                        epoch=int(curve_epoch),
                        episode_results=episode_results[
                            result_offset : result_offset + seed_count
                        ],
                    )
                )
                result_offset += seed_count
            print(
                "Baseline evaluation completed "
                f"learning_seed={learning_seed} "
                f"({learning_seed_idx + 1}/{len(ordered_learning_seeds)}, "
                f"epochs={len(ordered_epochs)}, episodes={len(seed_tasks)})",
                flush=True,
            )
        return rows

    if worker_count == 1:
        try:
            rows = _evaluate_baseline_learning_seeds(executor=None)
        finally:
            close_cached_env()
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = _evaluate_baseline_learning_seeds(executor=executor)

    return rows
