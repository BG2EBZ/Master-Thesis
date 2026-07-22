from __future__ import annotations

import argparse
import csv
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

from train.common.artifacts import build_dense_metric_matrix
from train.rwr.defaults import (
    DEFAULT_LEARNING_CURVE_PLOT_NAME,
    DEFAULT_LEARNING_CURVE_RAW_CSV_NAME,
)
from train.rwr.plotting import plot_learning_curve_metrics


@dataclass(frozen=True)
class LearningCurvePlotData:
    epochs: list[int]
    learning_seeds: list[int]
    evaluation_seeds: list[int]
    return_matrix: np.ndarray
    baseline_return_matrix: np.ndarray | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_learning_curve_rows(input_csv: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "policy": str(row["policy"]),
                    "learning_seed": int(row["learning_seed"]),
                    "evaluation_seed": int(row["evaluation_seed"]),
                    "epoch": int(row["epoch"]),
                    "mean_return": float(row["mean_return"]),
                }
            )
    return rows


def load_learning_curve_plot_data(input_csv: Path) -> LearningCurvePlotData:
    rows = _read_learning_curve_rows(input_csv)
    rwr_rows = [row for row in rows if row["policy"] == "rwr"]
    if not rwr_rows:
        raise ValueError(f"No RWR rows found in {input_csv}")

    return_matrix, learning_seeds, epochs = build_dense_metric_matrix(
        rwr_rows,
        row_key="learning_seed",
        column_key="epoch",
        value_key="mean_return",
    )
    baseline_rows = [row for row in rows if row["policy"] == "baseline"]
    baseline_return_matrix: np.ndarray | None = None
    evaluation_seeds: list[int] = []
    if baseline_rows:
        baseline_return_matrix, evaluation_seeds, baseline_epochs = build_dense_metric_matrix(
            baseline_rows,
            row_key="evaluation_seed",
            column_key="epoch",
            value_key="mean_return",
        )
        if baseline_epochs != epochs:
            raise ValueError("Baseline epochs do not match RWR epochs.")

    return LearningCurvePlotData(
        epochs=epochs,
        learning_seeds=learning_seeds,
        evaluation_seeds=evaluation_seeds,
        return_matrix=return_matrix,
        baseline_return_matrix=baseline_return_matrix,
    )


def _default_output_path(input_csv: Path) -> Path:
    if input_csv.name == DEFAULT_LEARNING_CURVE_RAW_CSV_NAME:
        return input_csv.with_name(DEFAULT_LEARNING_CURVE_PLOT_NAME)
    return input_csv.with_suffix(".png")


def replot_learning_curve(
    *,
    input_csv: Path,
    output_path: Path,
    max_x_ticks: int = 8,
) -> LearningCurvePlotData:
    plot_data = load_learning_curve_plot_data(input_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_learning_curve_metrics(
        epochs=plot_data.epochs,
        return_matrix=plot_data.return_matrix,
        baseline_return_matrix=plot_data.baseline_return_matrix,
        output_path=output_path,
        max_x_ticks=int(max_x_ticks),
    )
    return plot_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replot a saved RWR learning curve from learning_curve_raw.csv."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Saved learning_curve_raw.csv containing RWR and optional baseline rows.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Destination plot path. Defaults to learning_curve_plot.png next to the input CSV.",
    )
    parser.add_argument(
        "--max-x-ticks",
        type=_positive_int,
        default=8,
        help="Maximum number of integer x-axis ticks to show.",
    )
    args = parser.parse_args(argv)

    input_csv = Path(args.input_csv)
    output_path = (
        _default_output_path(input_csv)
        if args.output_path is None
        else Path(args.output_path)
    )
    plot_data = replot_learning_curve(
        input_csv=input_csv,
        output_path=output_path,
        max_x_ticks=int(args.max_x_ticks),
    )
    print(
        f"Saved replot to {output_path} "
        f"(learning_seeds={len(plot_data.learning_seeds)}, epochs={len(plot_data.epochs)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
