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
)
from train.rwr.plotting import (
    plot_learning_curve_metric_panels,
    plot_learning_curve_metrics,
)

RETURN_METRIC = "mean_return"
METRIC_COLUMNS = (
    RETURN_METRIC,
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)
METRIC_LABELS = {
    "mean_duration_seconds": ("Episode Duration", "Seconds"),
    "mean_overwhelmed_triggers": ("Overwhelmed Triggers", "Mean count"),
    "mean_impatient_triggers": ("Impatient Triggers", "Mean count"),
    "mean_distracted_triggers": ("Distracted Triggers", "Mean count"),
}
PLOT_KIND_METRICS = {
    "return": (RETURN_METRIC,),
    "other-metrics": (
        "mean_duration_seconds",
        "mean_overwhelmed_triggers",
        "mean_impatient_triggers",
        "mean_distracted_triggers",
    ),
}
DEFAULT_PLOT_NAMES = {
    "return": DEFAULT_LEARNING_CURVE_PLOT_NAME,
    "other-metrics": "learning_curve_other_metrics.png",
}


@dataclass(frozen=True)
class LearningCurvePlotData:
    policy: str
    epochs: list[int]
    learning_seeds: list[int]
    evaluation_seeds: list[int]
    metric_matrices: dict[str, np.ndarray]
    baseline_metric_matrices: dict[str, np.ndarray]

    @property
    def return_matrix(self) -> np.ndarray:
        return self.metric_matrices[RETURN_METRIC]

    @property
    def baseline_return_matrix(self) -> np.ndarray | None:
        return self.baseline_metric_matrices.get(RETURN_METRIC)


def _read_learning_curve_rows(
    input_csv: Path,
    metric_names: Sequence[str],
) -> list[dict[str, float | int | str]]:
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
                    **{metric: float(row[metric]) for metric in metric_names},
                }
            )
    return rows


def _build_metric_matrix_bundle(
    rows: list[dict[str, float | int | str]],
    *,
    row_key: str,
    metric_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    matrices: dict[str, np.ndarray] = {}
    ordered_rows: list[int] | None = None
    ordered_epochs: list[int] | None = None
    for metric_name in metric_names:
        matrix, metric_ordered_rows, metric_ordered_epochs = build_dense_metric_matrix(
            rows,
            row_key=row_key,
            column_key="epoch",
            value_key=metric_name,
        )
        if ordered_rows is None:
            ordered_rows = metric_ordered_rows
            ordered_epochs = metric_ordered_epochs
        elif ordered_rows != metric_ordered_rows or ordered_epochs != metric_ordered_epochs:
            raise ValueError(f"Matrix keys for {metric_name} do not match previous metrics.")
        matrices[metric_name] = matrix

    if ordered_rows is None or ordered_epochs is None:
        raise ValueError("metric_names must not be empty")
    return matrices, ordered_rows, ordered_epochs


def load_learning_curve_plot_data(
    input_csv: Path,
    *,
    metric_names: Sequence[str] = (RETURN_METRIC,),
    learned_policy: str | None = None,
) -> LearningCurvePlotData:
    resolved_metric_names = tuple(metric_names)
    unknown_metric_names = [
        metric_name for metric_name in resolved_metric_names if metric_name not in METRIC_COLUMNS
    ]
    if unknown_metric_names:
        raise ValueError(f"Unknown metric columns: {unknown_metric_names}")

    rows = _read_learning_curve_rows(input_csv, resolved_metric_names)
    learned_policies = sorted(
        {str(row["policy"]) for row in rows if str(row["policy"]) != "baseline"}
    )
    if learned_policy is None:
        if len(learned_policies) != 1:
            raise ValueError(
                "Expected exactly one learned policy in the CSV; "
                f"found {learned_policies}."
            )
        resolved_policy = learned_policies[0]
    else:
        resolved_policy = str(learned_policy).lower()
    learned_rows = [row for row in rows if row["policy"] == resolved_policy]
    if not learned_rows:
        raise ValueError(f"No {resolved_policy} rows found in {input_csv}")

    metric_matrices, learning_seeds, epochs = _build_metric_matrix_bundle(
        learned_rows,
        row_key="learning_seed",
        metric_names=resolved_metric_names,
    )
    baseline_rows = [row for row in rows if row["policy"] == "baseline"]
    baseline_metric_matrices: dict[str, np.ndarray] = {}
    evaluation_seeds: list[int] = []
    if baseline_rows:
        baseline_row_key = (
            "evaluation_seed"
            if all(int(row["learning_seed"]) == -1 for row in baseline_rows)
            else "learning_seed"
        )
        baseline_metric_matrices, baseline_rows_order, baseline_epochs = _build_metric_matrix_bundle(
            baseline_rows,
            row_key=baseline_row_key,
            metric_names=resolved_metric_names,
        )
        if baseline_epochs != epochs:
            raise ValueError("Baseline epochs do not match learned-policy epochs.")
        if baseline_row_key == "learning_seed":
            if baseline_rows_order != learning_seeds:
                raise ValueError(
                    "Baseline learning seeds do not match learned-policy learning seeds."
                )
        else:
            evaluation_seeds = baseline_rows_order

    return LearningCurvePlotData(
        policy=resolved_policy,
        epochs=epochs,
        learning_seeds=learning_seeds,
        evaluation_seeds=evaluation_seeds,
        metric_matrices=metric_matrices,
        baseline_metric_matrices=baseline_metric_matrices,
    )


def _default_output_dir(input_csv: Path, output_path: Path | None) -> Path:
    return input_csv.parent if output_path is None else output_path


def replot_learning_curve(
    *,
    input_csv: Path,
    output_path: Path,
    max_x_ticks: int = 8,
    plot_kind: str = "return",
) -> LearningCurvePlotData:
    metric_names = PLOT_KIND_METRICS[plot_kind]
    plot_data = load_learning_curve_plot_data(input_csv, metric_names=metric_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plot_kind == "return":
        plot_learning_curve_metrics(
            epochs=plot_data.epochs,
            return_matrix=plot_data.return_matrix,
            baseline_return_matrix=plot_data.baseline_return_matrix,
            output_path=output_path,
            max_x_ticks=int(max_x_ticks),
            learned_policy_label=plot_data.policy.upper(),
        )
    else:
        metric_panels = [
            (
                METRIC_LABELS[metric_name][0],
                METRIC_LABELS[metric_name][1],
                plot_data.metric_matrices[metric_name],
                plot_data.baseline_metric_matrices.get(metric_name),
            )
            for metric_name in metric_names
        ]
        plot_learning_curve_metric_panels(
            epochs=plot_data.epochs,
            metric_panels=metric_panels,
            output_path=output_path,
            max_x_ticks=int(max_x_ticks),
            learned_policy_label=plot_data.policy.upper(),
        )
    return plot_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replot a saved policy-search learning curve from learning_curve_raw.csv."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Saved learning_curve_raw.csv containing one learned policy and optional baseline rows.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Directory where both plot PNGs are written. Defaults to the input CSV directory.",
    )
    args = parser.parse_args(argv)

    input_csv = Path(args.input_csv)
    output_dir = _default_output_dir(input_csv, args.output_path)
    for plot_kind in PLOT_KIND_METRICS:
        output_path = output_dir / DEFAULT_PLOT_NAMES[plot_kind]
        plot_data = replot_learning_curve(
            input_csv=input_csv,
            output_path=output_path,
            plot_kind=plot_kind,
        )
        print(
            f"Saved replot to {output_path} "
            f"(plot_kind={plot_kind}, "
            f"learning_seeds={len(plot_data.learning_seeds)}, epochs={len(plot_data.epochs)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
