from __future__ import annotations

import numpy as np

from museum_env.guide_config import GuideBehaviorConfig

# Decode a theta vector into a GuideBehaviorConfig object.
def theta_to_guide_config(theta) -> GuideBehaviorConfig:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape not in ((4,), (5,), (6,)):
        raise ValueError(f"Expected theta shape (4,), (5,), or (6,), got {theta.shape}")

    slow_down_distance_m = float(np.clip(theta[0], 1.5, 3.5))
    callback_distance_m = float(np.clip(theta[1], slow_down_distance_m + 0.4, 5.5))
    callback_wait_seconds = float(np.clip(theta[2], 0.5, 8.0))
    slowdown_speed_scale = float(np.clip(theta[3], 0.5, 0.95))

    explanation_time_scale = 1.0
    if theta.shape in ((5,), (6,)):
        explanation_time_scale = float(np.clip(theta[4], 0.75, 1.0))

    callback_same_person_cooldown_seconds = 20.0
    if theta.shape == (6,):
        callback_same_person_cooldown_seconds = float(np.clip(theta[5], 0.0, 30.0))

    return GuideBehaviorConfig(
        slow_down_distance_m=slow_down_distance_m,
        callback_distance_m=callback_distance_m,
        callback_wait_seconds=callback_wait_seconds,
        slowdown_speed_scale=slowdown_speed_scale,
        explanation_time_scale=explanation_time_scale,
        callback_same_person_cooldown_seconds=callback_same_person_cooldown_seconds,
    )

# Encode a GuideBehaviorConfig object into a theta vector.
def guide_config_to_theta(config: GuideBehaviorConfig) -> np.ndarray:
    return np.array(
        [
            float(config.slow_down_distance_m),
            float(config.callback_distance_m),
            float(config.callback_wait_seconds),
            float(config.slowdown_speed_scale),
            float(config.explanation_time_scale),
            float(config.callback_same_person_cooldown_seconds),
        ],
        dtype=np.float64,
    )

# Summarize a theta vector into a dictionary of human-readable parameters.
def summarize_theta(theta) -> dict[str, float]:
    config = theta_to_guide_config(theta)
    return {
        "slow_down_distance_m": float(config.slow_down_distance_m),
        "callback_distance_m": float(config.callback_distance_m),
        "callback_wait_seconds": float(config.callback_wait_seconds),
        "slowdown_speed_scale": float(config.slowdown_speed_scale),
        "explanation_time_scale": float(config.explanation_time_scale),
        "explanation_wait_seconds": float(config.explanation_wait_seconds),
        "callback_same_person_cooldown_seconds": float(
            config.callback_same_person_cooldown_seconds
        ),
    }
