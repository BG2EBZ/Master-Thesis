"""Shared geometry helpers for angle wrapping and MuJoCo raycasts."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np


def wrap_to_pi(ang: float) -> float:
    """Normalize an angle to the ``[-pi, pi)`` range."""
    return float((float(ang) + np.pi) % (2.0 * np.pi) - np.pi)


def raycast_hit_distance(model, data, body_id, direction_xy) -> Optional[float]:
    """Return the hit distance for a planar ray cast from one MuJoCo body."""
    direction_xy = np.asarray(direction_xy, dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction_xy))
    if direction_norm <= 1e-6:
        return None
    if model is None or data is None or body_id is None or not hasattr(data, "xpos"):
        return None

    ray_direction = np.zeros(3, dtype=np.float64)
    ray_direction[:2] = direction_xy[:2] / direction_norm
    ray_origin = np.array(data.xpos[int(body_id)], dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    hit_distance = float(
        mujoco.mj_ray(
            model,
            data,
            ray_origin,
            ray_direction,
            None,
            1,
            int(body_id),
            geomid,
        )
    )
    return hit_distance if hit_distance >= 0.0 else None
