from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env_constants import LISTEN_WAIT_SECONDS_DEFAULT


@dataclass
class PolicySearchParams:
    """Learnable parameters of the high-level museum guidance policy."""

    slow_down_distance_m: float = 3.0
    callback_distance_m: float = 4.0
    callback_wait_seconds: float = 2.0
    slowdown_speed_scale: float = 0.7
    explanation_time_scale: float = 1.0

    @classmethod
    def from_theta(cls, theta) -> "PolicySearchParams":
        """Decode raw policy-search parameters into a valid robot policy."""
        theta = np.asarray(theta, dtype=np.float64)

        if theta.shape not in ((4,), (5,)):
            raise ValueError(f"Expected theta shape (4,) or (5,), got {theta.shape}")

        slow_down_distance_m = float(np.clip(theta[0], 1.5, 3.5))

        # Callback must occur after slowdown, not before it.
        callback_distance_m = float(
            np.clip(theta[1], slow_down_distance_m + 0.4, 5.5)
        )

        callback_wait_seconds = float(np.clip(theta[2], 0.5, 8.0))
        slowdown_speed_scale = float(np.clip(theta[3], 0.3, 0.95))
        explanation_time_scale = 1.0
        if theta.shape == (5,):
            explanation_time_scale = float(np.clip(theta[4], 0.7, 1.0))

        return cls(
            slow_down_distance_m=slow_down_distance_m,
            callback_distance_m=callback_distance_m,
            callback_wait_seconds=callback_wait_seconds,
            slowdown_speed_scale=slowdown_speed_scale,
            explanation_time_scale=explanation_time_scale,
        )

    @property
    def explanation_wait_seconds(self) -> float:
        return float(LISTEN_WAIT_SECONDS_DEFAULT) * float(self.explanation_time_scale)

    def to_theta(self) -> np.ndarray:
        return np.array(
            [
                self.slow_down_distance_m,
                self.callback_distance_m,
                self.callback_wait_seconds,
                self.slowdown_speed_scale,
                self.explanation_time_scale,
            ],
            dtype=np.float64,
        )
