from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
import torch

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


    theta_t = torch.as_tensor(theta_batch, dtype=torch.float64)
    returns_t = torch.as_tensor(returns, dtype=torch.float64)
    current_mu_t = torch.as_tensor(current_mu, dtype=torch.float64)
    current_std_t = torch.clamp(
        torch.as_tensor(current_std, dtype=torch.float64),
        min=float(config.min_std),
    )

    returns_std = (
        torch.std(returns_t)
        if returns_t.numel() > 1
        else torch.zeros((), dtype=torch.float64)
    )
    advantages = (returns_t - torch.mean(returns_t)) / (returns_std + 1e-8)
    old_log_prob = _diagonal_gaussian_log_prob(
        theta_t,
        current_mu_t,
        torch.log(current_std_t),
        min_std=float(config.min_std),
    ).detach()

    mu = torch.nn.Parameter(current_mu_t.clone())
    log_std = torch.nn.Parameter(torch.log(current_std_t).clone())
    optimizer = torch.optim.Adam([mu, log_std], lr=float(config.learning_rate))

    for _ in range(int(config.n_epochs_policy)):
        for start_idx in range(0, theta_t.shape[0], int(config.batch_size)):
            end_idx = min(start_idx + int(config.batch_size), theta_t.shape[0])
            theta_i = theta_t[start_idx:end_idx]
            advantages_i = advantages[start_idx:end_idx]
            old_log_prob_i = old_log_prob[start_idx:end_idx]

            optimizer.zero_grad()
            log_prob_i = _diagonal_gaussian_log_prob(
                theta_i,
                mu,
                log_std,
                min_std=float(config.min_std),
            )
            prob_ratio = torch.exp(log_prob_i - old_log_prob_i)
            clipped_ratio = torch.clamp(
                prob_ratio,
                1.0 - float(config.eps_ppo),
                1.0 + float(config.eps_ppo),
            )
            loss = -torch.mean(
                torch.minimum(
                    prob_ratio * advantages_i,
                    clipped_ratio * advantages_i,
                )
            )
            loss -= float(config.ent_coeff) * _diagonal_gaussian_entropy(
                log_std,
                min_std=float(config.min_std),
            )
            loss.backward()
            optimizer.step()

    next_mu = mu.detach().cpu().numpy().astype(np.float64)
    next_std = np.clip(
        torch.exp(log_std).detach().cpu().numpy().astype(np.float64),
        a_min=float(config.min_std),
        a_max=None,
    )
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


def _diagonal_gaussian_log_prob(
    theta: torch.Tensor,
    mu: torch.Tensor,
    log_std: torch.Tensor,
    *,
    min_std: float,
) -> torch.Tensor:
    std = torch.clamp(torch.exp(log_std), min=float(min_std))
    normalized = (theta - mu) / std
    log_two_pi = torch.log(
        torch.as_tensor(2.0 * np.pi, dtype=theta.dtype, device=theta.device)
    )
    return -0.5 * torch.sum(normalized**2 + 2.0 * torch.log(std) + log_two_pi, dim=1)


def _diagonal_gaussian_entropy(
    log_std: torch.Tensor,
    *,
    min_std: float,
) -> torch.Tensor:
    std = torch.clamp(torch.exp(log_std), min=float(min_std))
    log_term = torch.log(std)
    entropy_constant = torch.log(
        torch.as_tensor(2.0 * np.pi * np.e, dtype=log_std.dtype, device=log_std.device)
    )
    return 0.5 * torch.sum(entropy_constant + 2.0 * log_term)

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
