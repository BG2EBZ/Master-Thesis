from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

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
    return _weighted_diagonal_gaussian_mle(theta_batch, weights)


def update_distribution_reps(
    theta_batch: np.ndarray,
    returns: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta_batch = np.asarray(theta_batch, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    resolved_eps = float(eps)
    if resolved_eps <= 0.0:
        raise ValueError("eps must be positive")

    # Dual Optimization
    res = minimize(
        _reps_dual_function,
        np.ones(1, dtype=np.float64),
        jac=_reps_dual_function_diff,
        bounds=((np.finfo(np.float32).eps, np.inf),),
        args=(resolved_eps, returns),
    )

    # Extract the optimal eta
    eta = float(res.x.item())
    # Compute the weights for the MLE update
    shifted_returns = returns - np.max(returns)
    weights = np.exp(shifted_returns / eta)
    return _weighted_diagonal_gaussian_mle(theta_batch, weights)


def update_distribution_by_algorithm(
    *,
    algorithm: str,
    theta_batch: np.ndarray,
    returns: np.ndarray,
    beta: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_algorithm = str(algorithm).lower()
    if resolved_algorithm == "rwr":
        return update_distribution(theta_batch=theta_batch, returns=returns, beta=beta)
    if resolved_algorithm == "reps":
        return update_distribution_reps(theta_batch=theta_batch, returns=returns, eps=eps)
    raise ValueError(f"Unsupported policy-search algorithm: {algorithm!r}")

# Weighted Maximum Likelihood Estimation for Diagonal Gaussian
def _weighted_diagonal_gaussian_mle(
    theta_batch: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sum_weights = float(np.sum(weights))
    sum_weights_sq = float(np.sum(weights**2))
    denominator = sum_weights - (sum_weights_sq / sum_weights)

    mu = weights @ theta_batch / sum_weights
    delta_sq = (theta_batch - mu) ** 2
    std = np.sqrt(weights @ delta_sq / max(denominator, 1e-8))
    return mu, std


def _reps_dual_function(eta_array: np.ndarray, eps: float, returns: np.ndarray) -> float:
    eta = float(eta_array.item())
    shifted_returns = returns - np.max(returns)
    mean_exp = np.mean(np.exp(shifted_returns / eta))
    return float(eta * eps + eta * np.log(mean_exp) + np.max(returns))


def _reps_dual_function_diff(
    eta_array: np.ndarray,
    eps: float,
    returns: np.ndarray,
) -> np.ndarray:
    eta = float(eta_array.item())
    shifted_returns = returns - np.max(returns)
    exp_returns = np.exp(shifted_returns / eta)
    mean_exp = np.mean(exp_returns)
    weighted_return_mean = np.mean(exp_returns * shifted_returns)
    gradient = eps + np.log(mean_exp) - weighted_return_mean / (eta * mean_exp)
    return np.array([gradient], dtype=np.float64)


def diagonal_gaussian_entropy(std: np.ndarray) -> float:
    clipped_std = np.maximum(np.asarray(std, dtype=np.float64), 1e-8)
    return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e * (clipped_std**2))))
