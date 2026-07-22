from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from train.common.plot_utils import plot_mean_confidence_interval

EXPLORATION_STD_FIELDNAMES = (
    "std_0",
    "std_1",
    "std_2",
    "std_3",
    "std_4",
)
EXPLORATION_PARAMETER_LABELS = (
    "slow_down_distance_m",
    "callback_distance_m",
    "callback_wait_seconds",
    "slowdown_speed_scale",
    "explanation_time_scale",
)


def _build_sparse_epoch_ticks(epochs: Sequence[int], max_x_ticks: int) -> list[int]:
    if int(max_x_ticks) <= 0:
        raise ValueError("max_x_ticks must be positive")

    ordered_epochs = sorted({int(epoch) for epoch in epochs})
    if len(ordered_epochs) <= int(max_x_ticks):
        return ordered_epochs

    start = int(ordered_epochs[0])
    end = int(ordered_epochs[-1])
    if int(max_x_ticks) == 1:
        return [start]
    span = max(1, end - start)
    raw_step = span / float(max(1, int(max_x_ticks) - 1))
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = int(max(1, magnitude))

    def _ticks_for_step(candidate_step: int) -> list[int]:
        first_tick = int(math.ceil(start / candidate_step) * candidate_step)
        candidate_ticks = list(range(first_tick, end + 1, candidate_step))
        if start not in candidate_ticks:
            candidate_ticks.insert(0, start)
        if end not in candidate_ticks:
            candidate_ticks.append(end)
        return candidate_ticks

    for multiplier in (1, 2, 5, 10):
        candidate = int(max(1, multiplier * magnitude))
        candidate_ticks = _ticks_for_step(candidate)
        if len(candidate_ticks) <= int(max_x_ticks):
            step = candidate
            break

    ticks = _ticks_for_step(step)
    if len(ticks) > int(max_x_ticks):
        return [start, end]
    return ticks


def plot_training_metrics(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
    *,
    x_label: str = "Epoch",
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in metrics]
    mean_returns = [float(row["mean_return"]) for row in metrics]
    best_returns = [float(row["best_return"]) for row in metrics]
    mean_durations = [float(row["mean_duration_seconds"]) for row in metrics]
    mean_overwhelmed = [float(row["mean_overwhelmed_triggers"]) for row in metrics]
    mean_impatient = [float(row["mean_impatient_triggers"]) for row in metrics]
    mean_distracted = [float(row["mean_distracted_triggers"]) for row in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax_return, ax_duration, ax_triggers = axes.flat

    ax_return.plot(epochs, mean_returns, label="mean_return", linewidth=2)
    ax_return.plot(epochs, best_returns, label="best_return", linewidth=2)
    ax_return.set_title("Return")
    ax_return.set_xlabel(x_label)
    ax_return.set_ylabel("Return")
    ax_return.grid(True, alpha=0.3)
    ax_return.legend()

    ax_duration.plot(epochs, mean_durations, color="tab:orange", linewidth=2)
    ax_duration.set_title("Guide Duration")
    ax_duration.set_xlabel(x_label)
    ax_duration.set_ylabel("Seconds")
    ax_duration.grid(True, alpha=0.3)

    ax_triggers.plot(epochs, mean_overwhelmed, label="overwhelmed", linewidth=2)
    ax_triggers.plot(epochs, mean_impatient, label="impatient", linewidth=2)
    ax_triggers.plot(epochs, mean_distracted, label="distracted", linewidth=2)
    ax_triggers.set_title("Negative Trigger Counts")
    ax_triggers.set_xlabel(x_label)
    ax_triggers.set_ylabel("Mean count")
    ax_triggers.grid(True, alpha=0.3)
    ax_triggers.legend()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_exploration_metrics(
    metrics: Sequence[dict[str, float | int]],
    output_path: Path,
    *,
    x_label: str = "Epoch",
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in metrics]
    std_series = {
        label: [float(row[field_name]) for row in metrics]
        for field_name, label in zip(EXPLORATION_STD_FIELDNAMES, EXPLORATION_PARAMETER_LABELS)
    }
    entropies = [float(row["distribution_entropy"]) for row in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_std, ax_entropy = axes.flat

    for label, values in std_series.items():
        ax_std.plot(epochs, values, label=label, linewidth=2)
    ax_std.set_title("Std Per Dimension")
    ax_std.set_xlabel(x_label)
    ax_std.set_ylabel("Std")
    ax_std.grid(True, alpha=0.3)
    ax_std.legend()

    ax_entropy.plot(epochs, entropies, color="tab:green", linewidth=2)
    ax_entropy.set_title("Distribution Entropy")
    ax_entropy.set_xlabel(x_label)
    ax_entropy.set_ylabel("Entropy")
    ax_entropy.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_learning_curve_metrics(
    *,
    epochs: Sequence[int],
    return_matrix: np.ndarray,
    baseline_return_matrix: np.ndarray | None = None,
    output_path: Path,
    x_label: str = "# Epochs",
    max_x_ticks: int = 8,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.8), constrained_layout=False)
    x_values = np.asarray(epochs, dtype=np.float64)
    plot_mean_confidence_interval(
        ax,
        x_values,
        np.asarray(return_matrix, dtype=np.float64),
        color="#f28e2b",
        label="RWR",
        alpha=0.24,
        linewidth=2.2,
    )
    has_baseline = baseline_return_matrix is not None
    if has_baseline:
        plot_mean_confidence_interval(
            ax,
            x_values,
            np.asarray(baseline_return_matrix, dtype=np.float64),
            color="#4e79a7",
            label="Baseline",
            alpha=0.18,
            linewidth=2.0,
        )

    ax.set_xlabel(x_label, fontsize=16, fontweight="semibold")
    ax.set_ylabel("J", fontsize=16, fontweight="semibold")
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    ax.set_xticks(_build_sparse_epoch_ticks(epochs, max_x_ticks=max_x_ticks))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, color="#d6d6d6", linewidth=1.0, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0b0b0")
    ax.spines["bottom"].set_color("#b0b0b0")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=2 if has_baseline else 1,
        fontsize=13,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.26)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_learning_curve_metric_panels(
    *,
    epochs: Sequence[int],
    metric_panels: Sequence[
        tuple[str, str, np.ndarray, np.ndarray | None]
    ],
    output_path: Path,
    x_label: str = "# Epochs",
    max_x_ticks: int = 8,
) -> None:
    if not metric_panels:
        raise ValueError("metric_panels must not be empty")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_count = len(metric_panels)
    if panel_count == 3:
        nrows, ncols = 1, 3
        figsize = (14.5, 4.8)
    elif panel_count == 4:
        nrows, ncols = 2, 2
        figsize = (12.5, 8.2)
    else:
        nrows, ncols = 1, panel_count
        figsize = (5.0 * panel_count, 4.8)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=False)
    axes_array = np.atleast_1d(axes).flat
    x_values = np.asarray(epochs, dtype=np.float64)
    x_ticks = _build_sparse_epoch_ticks(epochs, max_x_ticks=max_x_ticks)
    legend_handles = None
    legend_labels = None

    for ax, (title, y_label, rwr_matrix, baseline_matrix) in zip(axes_array, metric_panels):
        plot_mean_confidence_interval(
            ax,
            x_values,
            np.asarray(rwr_matrix, dtype=np.float64),
            color="#f28e2b",
            label="RWR",
            alpha=0.24,
            linewidth=2.0,
        )
        if baseline_matrix is not None:
            plot_mean_confidence_interval(
                ax,
                x_values,
                np.asarray(baseline_matrix, dtype=np.float64),
                color="#4e79a7",
                label="Baseline",
                alpha=0.18,
                linewidth=1.9,
            )
        ax.set_title(title, fontsize=14, fontweight="semibold")
        ax.set_xlabel(x_label, fontsize=12, fontweight="semibold")
        ax.set_ylabel(y_label, fontsize=12, fontweight="semibold")
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, color="#d6d6d6", linewidth=1.0, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b0b0b0")
        ax.spines["bottom"].set_color("#b0b0b0")
        if legend_handles is None or legend_labels is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in list(axes_array)[panel_count:]:
        ax.set_visible(False)

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            frameon=False,
            ncol=len(legend_labels),
            fontsize=12,
            handlelength=2.0,
        )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.16, hspace=0.34, wspace=0.24)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
