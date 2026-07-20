from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceBand:
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray


def compute_mean_confidence_band(
    values: np.ndarray,
    *,
    z_value: float = 1.96,
) -> ConfidenceBand:
    """Compute mean and normal-approximation confidence band across runs."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape (n_runs, n_points), got {array.shape}")
    mean = np.mean(array, axis=0)
    if array.shape[0] <= 1:
        return ConfidenceBand(mean=mean, low=mean.copy(), high=mean.copy())

    sample_std = np.std(array, axis=0, ddof=1)
    half_width = float(z_value) * sample_std / np.sqrt(float(array.shape[0]))
    return ConfidenceBand(mean=mean, low=mean - half_width, high=mean + half_width)


def plot_mean_confidence_interval(
    ax,
    x: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    alpha: float = 0.2,
    linewidth: float = 2.0,
) -> ConfidenceBand:
    """Plot mean curve with a confidence interval, adapted to MushroomRL-style inputs."""
    band = compute_mean_confidence_band(values)
    ax.plot(
        x,
        band.mean,
        color=color,
        linewidth=linewidth,
        label=label,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.fill_between(x, band.low, band.high, color=color, alpha=alpha, linewidth=0.0)
    return band
