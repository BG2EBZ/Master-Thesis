import numpy as np

from .human_states import compute as _compute
from .human_states import compute_batch as _compute_batch


class FollowingFuzzyEngine:
    @staticmethod
    def clip_inputs(
        following_time: float,
        hhd: float,
        hrd: float,
        density: float,
        angle: float,
    ):
        return {
            "following_time": float(np.clip(following_time, 0.0, 120.0)),
            "hhd": float(np.clip(hhd, 0.0, 4.0)),
            "hrd": float(np.clip(hrd, 0.0, 5.0)),
            "density": float(np.clip(density, 0.0, 12.0)),
            "angle": float(np.clip(angle, -180.0, 180.0)),
        }

    def compute(
        self,
        following_time: float,
        hhd: float,
        hrd: float,
        density: float,
        angle: float,
        context: str = "following",
        profile: str = "normal",
    ) -> dict:
        clipped = self.clip_inputs(
            following_time=following_time,
            hhd=hhd,
            hrd=hrd,
            density=density,
            angle=angle,
        )
        return _compute(**clipped, context=context, profile=profile)

    def compute_batch(
        self,
        inputs: np.ndarray,
        context: str = "following",
        profile: str = "normal",
    ) -> list[dict]:
        rows = np.asarray(inputs, dtype=np.float32)
        return _compute_batch(rows, context=context, profile=profile)


def compute(
    following_time: float,
    hhd: float,
    hrd: float,
    density: float,
    angle: float,
    context: str = "following",
    profile: str = "normal",
) -> dict:
    return _compute(
        following_time=following_time,
        hhd=hhd,
        hrd=hrd,
        density=density,
        angle=angle,
        context=context,
        profile=profile,
    )


def compute_batch(
    inputs: np.ndarray,
    context: str = "following",
    profile: str = "normal",
) -> list[dict]:
    return _compute_batch(inputs=inputs, context=context, profile=profile)


__all__ = ["FollowingFuzzyEngine", "compute", "compute_batch"]
