from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train.common.plot_utils import compute_mean_confidence_band, plot_mean_confidence_interval

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
BASELINE_EPISODE_CSV_NAME = "baseline_episode_metrics.csv"
BASELINE_SEED_CSV_NAME = "baseline_seed_metrics.csv"
SUMMARY_JSON_NAME = "baseline_summary.json"
DEFAULT_PLOT_NAME = "baseline_metrics_plot.png"
EPISODE_CURVE_METRICS = (
    "duration_seconds",
    "overwhelmed_triggers",
    "impatient_triggers",
    "distracted_triggers",
)
SEED_SUMMARY_METRICS = (
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)
METRIC_LABELS = {
    "duration_seconds": ("Episode Duration", "Seconds"),
    "mean_duration_seconds": ("Episode Duration", "Seconds"),
    "overwhelmed_triggers": ("Overwhelmed Triggers", "Mean count"),
    "mean_overwhelmed_triggers": ("Overwhelmed Triggers", "Mean count"),
    "impatient_triggers": ("Impatient Triggers", "Mean count"),
    "mean_impatient_triggers": ("Impatient Triggers", "Mean count"),
    "distracted_triggers": ("Distracted Triggers", "Mean count"),
    "mean_distracted_triggers": ("Distracted Triggers", "Mean count"),
}


@dataclass(frozen=True)
class CurveMetrics:
    seed_indices: list[int]
    episode_indices: list[int]
    metric_matrices: dict[str, np.ndarray]


@dataclass(frozen=True)
class SeedSummaryMetrics:
    seed_indices: list[int]
    episode_counts: list[int]
    metric_values: dict[str, np.ndarray]


def _resolve_latest_run_dir() -> Path:
    run_root = ARTIFACTS_ROOT / "runs"
    candidates = sorted(
        path for path in run_root.glob("baseline_only_*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"No baseline_only_* run directory found under {run_root}")
    return candidates[-1]


def _resolve_run_dir(run_dir: Path | None, input_csv: Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    if input_csv is not None:
        return Path(input_csv).parent
    return _resolve_latest_run_dir()


def _resolve_input_csv(
    *,
    run_dir: Path,
    input_csv: Path | None,
    plot_kind: str,
) -> Path:
    if input_csv is not None:
        return Path(input_csv)
    csv_name = (
        BASELINE_EPISODE_CSV_NAME
        if plot_kind == "episode-curve"
        else BASELINE_SEED_CSV_NAME
    )
    return Path(run_dir) / csv_name


def _resolve_output_path(run_dir: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    return Path(run_dir) / DEFAULT_PLOT_NAME


def _read_summary(run_dir: Path) -> dict[str, object]:
    summary_path = Path(run_dir) / SUMMARY_JSON_NAME
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def _infer_duration_seconds(row: dict[str, str], summary: dict[str, object]) -> float:
    weights = summary.get("reward_weights")
    if not isinstance(weights, dict):
        raise ValueError(
            "duration_seconds is missing from the episode CSV and reward_weights "
            "is missing from baseline_summary.json, so duration cannot be inferred."
        )
    time_weight = float(weights["time_penalty_per_second"])
    if time_weight <= 0.0:
        raise ValueError("time_penalty_per_second must be positive to infer duration.")
    return_value = float(row["return"])
    trigger_penalty = (
        float(weights["overwhelmed_trigger_penalty"]) * int(row["overwhelmed_triggers"])
        + float(weights["impatient_trigger_penalty"]) * int(row["impatient_triggers"])
        + float(weights["distracted_trigger_penalty"]) * int(row["distracted_triggers"])
    )
    return float(-(return_value + trigger_penalty) / time_weight)


def _read_episode_rows(input_csv: Path, summary: dict[str, object]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = (
            "seed_index",
            "episode_index",
            "return",
            "overwhelmed_triggers",
            "impatient_triggers",
            "distracted_triggers",
        )
        missing_columns = [
            column for column in required_columns if column not in (reader.fieldnames or ())
        ]
        if missing_columns:
            raise ValueError(f"{input_csv} is missing columns: {missing_columns}")
        has_duration = "duration_seconds" in (reader.fieldnames or ())
        for row in reader:
            rows.append(
                {
                    "seed_index": int(row["seed_index"]),
                    "episode_index": int(row["episode_index"]),
                    "duration_seconds": (
                        float(row["duration_seconds"])
                        if has_duration
                        else _infer_duration_seconds(row, summary)
                    ),
                    "overwhelmed_triggers": int(row["overwhelmed_triggers"]),
                    "impatient_triggers": int(row["impatient_triggers"]),
                    "distracted_triggers": int(row["distracted_triggers"]),
                }
            )
    if not rows:
        raise ValueError(f"{input_csv} does not contain any episode metric rows.")
    return rows


def _build_curve_metrics(rows: Sequence[dict[str, float | int]]) -> CurveMetrics:
    seed_indices = sorted({int(row["seed_index"]) for row in rows})
    episode_indices = sorted({int(row["episode_index"]) for row in rows})
    seed_to_idx = {seed_index: idx for idx, seed_index in enumerate(seed_indices)}
    episode_to_idx = {
        episode_index: idx for idx, episode_index in enumerate(episode_indices)
    }
    metric_matrices = {
        metric_name: np.full(
            (len(seed_indices), len(episode_indices)),
            np.nan,
            dtype=np.float64,
        )
        for metric_name in EPISODE_CURVE_METRICS
    }

    for row in rows:
        row_idx = seed_to_idx[int(row["seed_index"])]
        column_idx = episode_to_idx[int(row["episode_index"])]
        for metric_name in EPISODE_CURVE_METRICS:
            metric_matrices[metric_name][row_idx, column_idx] = float(row[metric_name])

    for metric_name, matrix in metric_matrices.items():
        if np.isnan(matrix).any():
            raise ValueError(f"Episode rows do not form a dense matrix for {metric_name}.")

    return CurveMetrics(
        seed_indices=seed_indices,
        episode_indices=episode_indices,
        metric_matrices=metric_matrices,
    )


def _read_seed_summary_metrics(input_csv: Path) -> SeedSummaryMetrics:
    rows: list[dict[str, str]] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [
            column
            for column in ("seed_index", "episode_count", *SEED_SUMMARY_METRICS)
            if column not in (reader.fieldnames or ())
        ]
        if missing_columns:
            raise ValueError(f"{input_csv} is missing columns: {missing_columns}")
        rows.extend(dict(row) for row in reader)
    if not rows:
        raise ValueError(f"{input_csv} does not contain any seed metric rows.")

    return SeedSummaryMetrics(
        seed_indices=[int(row["seed_index"]) for row in rows],
        episode_counts=[int(row["episode_count"]) for row in rows],
        metric_values={
            metric_name: np.asarray(
                [float(row[metric_name]) for row in rows],
                dtype=np.float64,
            )
            for metric_name in SEED_SUMMARY_METRICS
        },
    )


def _metric_summary(values: np.ndarray) -> tuple[float, float, float]:
    band = compute_mean_confidence_band(np.asarray(values, dtype=np.float64).reshape(-1, 1))
    return float(band.mean[0]), float(band.low[0]), float(band.high[0])


def _style_axis(ax, *, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="semibold")
    ax.set_xlabel(x_label, fontsize=12, fontweight="semibold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="semibold")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, color="#d6d6d6", linewidth=1.0, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0b0b0")
    ax.spines["bottom"].set_color("#b0b0b0")


def _build_sparse_ticks(values: Sequence[int], *, max_ticks: int = 8) -> list[int]:
    ordered = [int(value) for value in values]
    if len(ordered) <= max_ticks:
        return ordered
    tick_indices = np.linspace(0, len(ordered) - 1, num=max_ticks, dtype=int)
    return [ordered[int(idx)] for idx in tick_indices]


def _save_episode_curve_plot(
    *,
    metrics: CurveMetrics,
    output_path: Path,
    x_label: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=False)
    axes_array = axes.flat
    x_values = np.asarray(metrics.episode_indices, dtype=np.float64)

    for ax, metric_name in zip(axes_array, EPISODE_CURVE_METRICS):
        title, y_label = METRIC_LABELS[metric_name]
        plot_mean_confidence_interval(
            ax,
            x_values,
            metrics.metric_matrices[metric_name],
            color="#4e79a7",
            label="Baseline",
            alpha=0.18,
            linewidth=2.0,
        )
        _style_axis(ax, title=title, x_label=x_label, y_label=y_label)
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
        ax.set_xticks(_build_sparse_ticks(metrics.episode_indices))

    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        ncol=1,
        fontsize=12,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.12, hspace=0.34, wspace=0.24)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_seed_summary_plot(
    *,
    metrics: SeedSummaryMetrics,
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=False)
    axes_array = axes.flat
    x_values = np.asarray(metrics.seed_indices, dtype=np.int64)

    for ax, metric_name in zip(axes_array, SEED_SUMMARY_METRICS):
        values = metrics.metric_values[metric_name]
        mean, ci_low, ci_high = _metric_summary(values)
        title, y_label = METRIC_LABELS[metric_name]
        ax.plot(
            x_values,
            values,
            color="#4e79a7",
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label="Seed mean",
        )
        ax.axhline(mean, color="#f28e2b", linewidth=2.0, label="Mean")
        ax.fill_between(
            x_values,
            ci_low,
            ci_high,
            color="#f28e2b",
            alpha=0.18,
            linewidth=0.0,
            label="95% CI",
        )
        _style_axis(ax, title=f"{title}  mean={mean:.3f}", x_label="Seed index", y_label=y_label)
        ax.set_xticks(x_values)

    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        ncol=3,
        fontsize=11,
        handlelength=2.0,
    )
    fig.suptitle("Baseline Metrics Across Seeds", fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.12, hspace=0.34, wspace=0.24)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_baseline_only_metrics(
    *,
    run_dir: Path,
    input_csv: Path,
    output_path: Path,
    plot_kind: str,
    x_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plot_kind == "episode-curve":
        summary = _read_summary(run_dir)
        rows = _read_episode_rows(input_csv, summary)
        curve_metrics = _build_curve_metrics(rows)
        _save_episode_curve_plot(
            metrics=curve_metrics,
            output_path=output_path,
            x_label=x_label,
        )
        print(
            f"Saved baseline episode curve plot to {output_path} "
            f"(seeds={len(curve_metrics.seed_indices)}, "
            f"episodes={len(curve_metrics.episode_indices)})"
        )
        for metric_name in EPISODE_CURVE_METRICS:
            matrix = curve_metrics.metric_matrices[metric_name]
            band = compute_mean_confidence_band(matrix)
            print(
                f"{metric_name}: final_mean={float(band.mean[-1]):.3f}, "
                f"overall_mean={float(np.mean(matrix)):.3f}"
            )
        return

    seed_metrics = _read_seed_summary_metrics(input_csv)
    _save_seed_summary_plot(metrics=seed_metrics, output_path=output_path)
    print(
        f"Saved baseline seed summary plot to {output_path} "
        f"(seeds={len(seed_metrics.seed_indices)}, "
        f"episodes_per_seed={','.join(str(value) for value in seed_metrics.episode_counts)})"
    )
    for metric_name in SEED_SUMMARY_METRICS:
        mean, ci_low, ci_high = _metric_summary(seed_metrics.metric_values[metric_name])
        print(f"{metric_name}: mean={mean:.3f}, ci95=[{ci_low:.3f}, {ci_high:.3f}]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot baseline-only metrics from baseline episode or seed CSVs."
    )
    parser.add_argument(
        "--plot-kind",
        choices=("episode-curve", "seed-summary"),
        default="episode-curve",
        help="Plot an episode-index curve or the older seed-summary plot.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input CSV path. Defaults to the CSV required by --plot-kind in the run dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Baseline-only run directory. If omitted, the latest baseline_only_* run is used.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=f"Output PNG path. Defaults to {DEFAULT_PLOT_NAME} in the run dir.",
    )
    parser.add_argument(
        "--x-label",
        default="# Episodes",
        help="X-axis label for episode-curve plots.",
    )
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_dir, args.input_csv)
    input_csv = _resolve_input_csv(
        run_dir=run_dir,
        input_csv=args.input_csv,
        plot_kind=str(args.plot_kind),
    )
    output_path = _resolve_output_path(run_dir, args.output_path)
    plot_baseline_only_metrics(
        run_dir=run_dir,
        input_csv=input_csv,
        output_path=output_path,
        plot_kind=str(args.plot_kind),
        x_label=str(args.x_label),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
