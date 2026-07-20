from __future__ import annotations

from dataclasses import dataclass

from .env_constants import LISTEN_WAIT_SECONDS_DEFAULT


@dataclass
class GuideBehaviorConfig:
    slow_down_distance_m: float = 3.0
    callback_distance_m: float = 4.0
    callback_wait_seconds: float = 2.0
    slowdown_speed_scale: float = 0.7
    explanation_time_scale: float = 1.0
    callback_same_person_cooldown_seconds: float = 10.0

    @property
    def explanation_wait_seconds(self) -> float:
        return float(LISTEN_WAIT_SECONDS_DEFAULT) * float(self.explanation_time_scale)
