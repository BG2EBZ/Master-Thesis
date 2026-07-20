from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from train.common.rollout import EpisodeResult


@dataclass(frozen=True)
class ThetaEvaluation:
    mean_return: float
    mean_duration_seconds: float
    mean_overwhelmed_triggers: float
    mean_impatient_triggers: float
    mean_distracted_triggers: float


def aggregate_episode_results(episode_results: Sequence[EpisodeResult]) -> ThetaEvaluation:
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


def update_distribution(
    theta_batch: np.ndarray,
    returns: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_returns = returns - np.max(returns)
    weights = np.exp(beta * shifted_returns)

    sum_weights = float(np.sum(weights))
    sum_weights_sq = float(np.sum(weights**2))
    denominator = sum_weights - (sum_weights_sq / sum_weights)

    mu = weights @ theta_batch / sum_weights
    delta_sq = (theta_batch - mu) ** 2
    std = np.sqrt(weights @ delta_sq / max(denominator, 1e-8))
    return mu, std


def diagonal_gaussian_entropy(std: np.ndarray) -> float:
    clipped_std = np.maximum(np.asarray(std, dtype=np.float64), 1e-8)
    return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e * (clipped_std**2))))
