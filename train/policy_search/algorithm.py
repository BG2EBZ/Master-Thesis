from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from train.common.rollout import EpisodeResult


@dataclass(frozen=True)
class EPPOConfig:
    learning_rate: float = 1e-3
    n_epochs_policy: int = 5
    batch_size: int = 10
    eps_ppo: float = 0.2
    ent_coeff: float = 0.001
    min_std: float = 1e-4


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
    current_mu: np.ndarray | None = None,
    current_std: np.ndarray | None = None,
    eppo_config: EPPOConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_algorithm = str(algorithm).lower()
    if resolved_algorithm == "rwr":
        return update_distribution(theta_batch=theta_batch, returns=returns, beta=beta)
    if resolved_algorithm == "reps":
        return update_distribution_reps(theta_batch=theta_batch, returns=returns, eps=eps)
    if resolved_algorithm == "eppo":
        if current_mu is None or current_std is None:
            raise ValueError("ePPO requires current_mu and current_std")
        return update_distribution_eppo(
            theta_batch=theta_batch,
            returns=returns,
            current_mu=current_mu,
            current_std=current_std,
            config=EPPOConfig() if eppo_config is None else eppo_config,
        )
    raise ValueError(f"Unsupported policy-search algorithm: {algorithm!r}")


def update_distribution_eppo(
    *,
    theta_batch: np.ndarray,
    returns: np.ndarray,
    current_mu: np.ndarray,
    current_std: np.ndarray,
    config: EPPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_eppo_config(config)
    theta_batch = np.asarray(theta_batch, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    current_mu = np.asarray(current_mu, dtype=np.float64)
    current_std = np.asarray(current_std, dtype=np.float64)

    current_std = np.maximum(current_std, float(config.min_std))
    returns_std = np.std(returns, ddof=1) if returns.size > 1 else 0.0
    advantages = (returns - np.mean(returns)) / (returns_std + 1e-8)
    old_log_prob = _diagonal_gaussian_log_prob_numpy(
        theta_batch,
        current_mu,
        np.log(current_std),
        min_std=float(config.min_std),
    )

    mu = np.array(current_mu, dtype=np.float64, copy=True)
    log_std = np.log(current_std).astype(np.float64, copy=True)
    adam_state = _AdamState.zeros_like(mu, log_std)
    step_index = 0

    for _ in range(int(config.n_epochs_policy)):
        for start_idx in range(0, theta_batch.shape[0], int(config.batch_size)):
            end_idx = min(start_idx + int(config.batch_size), theta_batch.shape[0])
            theta_i = theta_batch[start_idx:end_idx]
            advantages_i = advantages[start_idx:end_idx]
            old_log_prob_i = old_log_prob[start_idx:end_idx]

            grad_mu, grad_log_std = _eppo_loss_gradients(
                theta=theta_i,
                advantages=advantages_i,
                old_log_prob=old_log_prob_i,
                mu=mu,
                log_std=log_std,
                config=config,
            )
            step_index += 1
            mu, log_std, adam_state = _adam_step(
                mu=mu,
                log_std=log_std,
                grad_mu=grad_mu,
                grad_log_std=grad_log_std,
                state=adam_state,
                step_index=step_index,
                learning_rate=float(config.learning_rate),
            )

    next_mu = mu.astype(np.float64)
    next_std = np.clip(np.exp(log_std), a_min=float(config.min_std), a_max=None)
    return next_mu, next_std


def _validate_eppo_config(config: EPPOConfig) -> None:
    if float(config.learning_rate) <= 0.0:
        raise ValueError("ePPO learning_rate must be positive")
    if int(config.n_epochs_policy) <= 0:
        raise ValueError("ePPO n_epochs_policy must be positive")
    if int(config.batch_size) <= 0:
        raise ValueError("ePPO batch_size must be positive")
    if float(config.eps_ppo) <= 0.0:
        raise ValueError("ePPO eps_ppo must be positive")
    if float(config.min_std) <= 0.0:
        raise ValueError("ePPO min_std must be positive")


@dataclass(frozen=True)
class _AdamState:
    mu_m: np.ndarray
    mu_v: np.ndarray
    log_std_m: np.ndarray
    log_std_v: np.ndarray

    @classmethod
    def zeros_like(cls, mu: np.ndarray, log_std: np.ndarray) -> "_AdamState":
        return cls(
            mu_m=np.zeros_like(mu, dtype=np.float64),
            mu_v=np.zeros_like(mu, dtype=np.float64),
            log_std_m=np.zeros_like(log_std, dtype=np.float64),
            log_std_v=np.zeros_like(log_std, dtype=np.float64),
        )


def _eppo_loss_gradients(
    *,
    theta: np.ndarray,
    advantages: np.ndarray,
    old_log_prob: np.ndarray,
    mu: np.ndarray,
    log_std: np.ndarray,
    config: EPPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    min_std = float(config.min_std)
    log_prob = _diagonal_gaussian_log_prob_numpy(
        theta,
        mu,
        log_std,
        min_std=min_std,
    )
    prob_ratio = np.exp(log_prob - old_log_prob)
    clipped_low = 1.0 - float(config.eps_ppo)
    clipped_high = 1.0 + float(config.eps_ppo)
    clipped_active = ((advantages >= 0.0) & (prob_ratio > clipped_high)) | (
        (advantages < 0.0) & (prob_ratio < clipped_low)
    )
    active_weight = np.where(clipped_active, 0.0, advantages * prob_ratio)

    std = np.maximum(np.exp(log_std), min_std)
    delta = theta - mu
    grad_log_prob_mu = delta / (std**2)
    unclipped_std_mask = (np.exp(log_std) >= min_std).astype(np.float64)
    grad_log_prob_log_std = ((delta**2 / (std**2)) - 1.0) * unclipped_std_mask

    grad_mu = -(active_weight[:, None] * grad_log_prob_mu).mean(axis=0)
    grad_log_std = -(active_weight[:, None] * grad_log_prob_log_std).mean(axis=0)
    grad_log_std -= float(config.ent_coeff) * _diagonal_gaussian_entropy_gradient_numpy(
        log_std,
        min_std=min_std,
    )
    return grad_mu, grad_log_std


def _adam_step(
    *,
    mu: np.ndarray,
    log_std: np.ndarray,
    grad_mu: np.ndarray,
    grad_log_std: np.ndarray,
    state: _AdamState,
    step_index: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, _AdamState]:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    mu_m = beta1 * state.mu_m + (1.0 - beta1) * grad_mu
    mu_v = beta2 * state.mu_v + (1.0 - beta2) * (grad_mu**2)
    log_std_m = beta1 * state.log_std_m + (1.0 - beta1) * grad_log_std
    log_std_v = beta2 * state.log_std_v + (1.0 - beta2) * (grad_log_std**2)

    bias_correction_1 = 1.0 - beta1**int(step_index)
    bias_correction_2 = 1.0 - beta2**int(step_index)
    next_mu = mu - float(learning_rate) * (mu_m / bias_correction_1) / (
        np.sqrt(mu_v / bias_correction_2) + eps
    )
    next_log_std = log_std - float(learning_rate) * (
        log_std_m / bias_correction_1
    ) / (np.sqrt(log_std_v / bias_correction_2) + eps)
    return (
        next_mu,
        next_log_std,
        _AdamState(
            mu_m=mu_m,
            mu_v=mu_v,
            log_std_m=log_std_m,
            log_std_v=log_std_v,
        ),
    )


def _diagonal_gaussian_log_prob_numpy(
    theta: np.ndarray,
    mu: np.ndarray,
    log_std: np.ndarray,
    *,
    min_std: float,
) -> np.ndarray:
    std = np.maximum(np.exp(log_std), float(min_std))
    normalized = (theta - mu) / std
    log_two_pi = np.log(2.0 * np.pi)
    return -0.5 * np.sum(normalized**2 + 2.0 * np.log(std) + log_two_pi, axis=1)


def _diagonal_gaussian_entropy_gradient_numpy(
    log_std: np.ndarray,
    *,
    min_std: float,
) -> np.ndarray:
    return (np.exp(log_std) >= float(min_std)).astype(np.float64)

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
