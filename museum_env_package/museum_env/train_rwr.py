from __future__ import annotations
import numpy as np
from museum_env import MuseumEnv


def run_episode(env: MuseumEnv, theta: np.ndarray, seed: int) -> float:
    env.set_policy_parameters(theta)

    _observation, _info = env.reset(seed=seed)

    total_return = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        _observation, reward, terminated, truncated, _info = env.step(None)
        total_return += float(reward)

    return total_return


def evaluate_theta(
    env: MuseumEnv,
    theta: np.ndarray,
    seeds: list[int],
) -> float:
    returns = [
        run_episode(env, theta=theta, seed=seed)
        for seed in seeds
    ]

    return float(np.mean(returns))


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


def main() -> None:
    env = MuseumEnv(
        render_mode=None,
        enable_event_logs=False,
        n_humans=5,
    )

    rng = np.random.default_rng(seed=42)

    mu = np.array(
        [
            2.5,  # slow_down_distance_m
            3.5,  # callback_distance_m
            2.0,  # callback_wait_seconds
            0.7,  # slowdown_speed_scale
        ],
        dtype=np.float64,
    )

    std = np.array(
        [
            0.4,
            0.6,
            0.8,
            0.1,
        ],
        dtype=np.float64,
    )

    beta = 0.02
    epochs = 30
    samples_per_epoch = 30
    evaluation_seeds = [11, 22, 33]

    for epoch in range(epochs):
        theta_batch = rng.normal(
            loc=mu,
            scale=std,
            size=(samples_per_epoch, len(mu)),
        )

        returns = np.array(
            [
                evaluate_theta(env, theta=theta, seeds=evaluation_seeds)
                for theta in theta_batch
            ],
            dtype=np.float64,
        )

        mu, std = update_distribution(
            theta_batch=theta_batch,
            returns=returns,
            beta=beta,
        )

        print(
            f"epoch={epoch:02d} "
            f"mean_return={returns.mean():.3f} "
            f"best_return={returns.max():.3f}\n"
            f"mu={np.round(mu, 3)}\n"
            f"std={np.round(std, 3)}"
        )

    env.close()


if __name__ == "__main__":
    main()