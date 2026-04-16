import numpy as np

from .human_states_following import compute as _compute
from .human_states_following import compute_batch as _compute_batch


class FollowingFuzzyEngine:
    @staticmethod
    def clip_inputs(following_time: float, hhd: float, hrd: float, density: float):
        return {
            "following_time": float(np.clip(following_time, 0.0, 60.0)),
            "hhd": float(np.clip(hhd, 0.0, 4.0)),
            "hrd": float(np.clip(hrd, 0.0, 4.0)),
            "density": float(np.clip(density, 0.0, 10.0)),
        }

    def compute(self, following_time: float, hhd: float, hrd: float, density: float) -> dict:
        clipped = self.clip_inputs(
            following_time=following_time,
            hhd=hhd,
            hrd=hrd,
            density=density,
        )
        return _compute(**clipped)

    def compute_batch(self, inputs: np.ndarray) -> list[dict]:
        rows = np.asarray(inputs, dtype=np.float32)
        return [self.compute(*row) for row in rows]


def compute(following_time: float, hhd: float, hrd: float, density: float) -> dict:
    return _compute(
        following_time=following_time,
        hhd=hhd,
        hrd=hrd,
        density=density,
    )


def compute_batch(inputs: np.ndarray) -> list[dict]:
    return _compute_batch(inputs=inputs)


__all__ = ["FollowingFuzzyEngine", "compute", "compute_batch"]
