from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train.common.artifacts import build_dense_metric_matrix, write_csv_rows, write_json
from train.common.plot_utils import compute_mean_confidence_band
from train.rwr.defaults import (
    ARTIFACTS_ROOT,
    DEFAULT_LEARNING_CURVE_SUMMARY_NAME,
    LEARNING_CURVE_RAW_FIELDNAMES,
)
from train.rwr.plotting import plot_multi_policy_learning_curve_metrics

DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "runs" / (
    f"policy_search_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
DEFAULT_RAW_CSV_NAME = "policy_search_comparison_raw.csv"
DEFAULT_SUMMARY_NAME = "policy_search_comparison_summary.json"
DEFAULT_PLOT_NAME = "policy_search_comparison_plot.png"
RETURN_METRIC = "mean_return"
METRIC_COLUMNS = (
    RETURN_METRIC,
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)


@dataclass(frozen=True)
class PolicyRunSpec:
    policy: str
    csv_path: Path


@dataclass(frozen=True)
class PolicySearchComparison:
    epochs: list[int]
    learning_seeds: list[int]
    seed_plan_hash: str | None
    output_raw_csv: Path
    output_summary_json: Path
    output_plot: Path


def parse_run_spec(value: str) -> PolicyRunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run spec must use policy=path")
    raw_policy, raw_path = value.split("=", 1)
    policy = raw_policy.strip().lower()
    if not policy:
        raise argparse.ArgumentTypeError("run spec policy must not be empty")
    if policy == "baseline":
        raise argparse.ArgumentTypeError("baseline is loaded from each run; do not pass it as --run")
    csv_path = Path(raw_path.strip())
    if not str(csv_path):
        raise argparse.ArgumentTypeError("run spec path must not be empty")
    return PolicyRunSpec(policy=policy, csv_path=csv_path)


def _read_learning_curve_rows(input_csv: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with Path(input_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [
            field for field in LEARNING_CURVE_RAW_FIELDNAMES if field not in (reader.fieldnames or ())
        ]
        if missing_fields:
            raise ValueError(f"{input_csv} is missing columns: {missing_fields}")
        for row in reader:
            rows.append(
                {
                    "policy": str(row["policy"]).lower(),
                    "learning_seed": int(row["learning_seed"]),
                    "evaluation_seed": int(row["evaluation_seed"]),
                    "epoch": int(row["epoch"]),
                    **{metric: float(row[metric]) for metric in METRIC_COLUMNS},
                }
            )
    return rows


def _build_metric_matrix(
    rows: Sequence[dict[str, float | int | str]],
    *,
    metric_name: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    return build_dense_metric_matrix(
        rows,
        row_key="learning_seed",
        column_key="epoch",
        value_key=metric_name,
    )


def _load_seed_plan_hash(input_csv: Path) -> str | None:
    summary_path = Path(input_csv).parent / DEFAULT_LEARNING_CURVE_SUMMARY_NAME
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    seed_hash = payload.get("seed_plan_hash")
    return str(seed_hash) if seed_hash is not None else None


def _validate_unique_policies(run_specs: Sequence[PolicyRunSpec]) -> None:
    seen: set[str] = set()
    for run_spec in run_specs:
        if run_spec.policy in seen:
            raise ValueError(f"Duplicate policy label: {run_spec.policy}")
        seen.add(run_spec.policy)


def _ordered_policy_matrices(
    *,
    baseline_matrix: np.ndarray,
    policy_matrices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    ordered: dict[str, np.ndarray] = {"baseline": baseline_matrix}
    for policy in ("rwr", "reps"):
        if policy in policy_matrices:
            ordered[policy] = policy_matrices[policy]
    for policy in sorted(policy_matrices):
        if policy not in ordered:
            ordered[policy] = policy_matrices[policy]
    return ordered


def compare_policy_search_runs(
    *,
    run_specs: Sequence[PolicyRunSpec],
    output_dir: Path,
    require_seed_plan_hash: bool = True,
    max_x_ticks: int = 8,
) -> PolicySearchComparison:
    if not run_specs:
        raise ValueError("At least one --run policy=path is required.")
    _validate_unique_policies(run_specs)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_dir / DEFAULT_RAW_CSV_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME

    reference_learning_seeds: list[int] | None = None
    reference_epochs: list[int] | None = None
    baseline_return_matrix: np.ndarray | None = None
    baseline_rows_for_output: list[dict[str, float | int | str]] | None = None
    policy_return_matrices: dict[str, np.ndarray] = {}
    combined_rows: list[dict[str, float | int | str]] = []
    input_csvs: dict[str, str] = {}
    seed_plan_hashes: dict[str, str | None] = {}

    for run_spec in run_specs:
        rows = _read_learning_curve_rows(run_spec.csv_path)
        input_csvs[run_spec.policy] = str(run_spec.csv_path)
        seed_plan_hashes[run_spec.policy] = _load_seed_plan_hash(run_spec.csv_path)
        learned_rows = [row for row in rows if row["policy"] == run_spec.policy]
        if not learned_rows:
            raise ValueError(f"No rows with policy={run_spec.policy!r} found in {run_spec.csv_path}")
        learned_matrix, learning_seeds, epochs = _build_metric_matrix(
            learned_rows,
            metric_name=RETURN_METRIC,
        )
        if reference_learning_seeds is None:
            reference_learning_seeds = learning_seeds
            reference_epochs = epochs
        elif learning_seeds != reference_learning_seeds or epochs != reference_epochs:
            raise ValueError("Compared policy runs do not share learning seeds and epochs.")
        policy_return_matrices[run_spec.policy] = learned_matrix
        combined_rows.extend(learned_rows)

        baseline_rows = [row for row in rows if row["policy"] == "baseline"]
        if not baseline_rows:
            raise ValueError(f"No baseline rows found in {run_spec.csv_path}")
        current_baseline_matrix, baseline_learning_seeds, baseline_epochs = _build_metric_matrix(
            baseline_rows,
            metric_name=RETURN_METRIC,
        )
        if baseline_learning_seeds != learning_seeds or baseline_epochs != epochs:
            raise ValueError(
                f"Baseline rows in {run_spec.csv_path} do not match learned-policy seeds."
            )
        if baseline_return_matrix is None:
            baseline_return_matrix = current_baseline_matrix
            baseline_rows_for_output = baseline_rows
        elif not np.allclose(baseline_return_matrix, current_baseline_matrix):
            raise ValueError("Baseline return matrices differ across runs.")

    if reference_learning_seeds is None or reference_epochs is None:
        raise ValueError("No comparable policy rows found.")
    if baseline_return_matrix is None or baseline_rows_for_output is None:
        raise ValueError("No baseline rows found.")

    observed_hashes = {value for value in seed_plan_hashes.values() if value is not None}
    if require_seed_plan_hash:
        missing_seed_hash_policies = [
            policy for policy, value in seed_plan_hashes.items() if value is None
        ]
        if missing_seed_hash_policies:
            raise ValueError(
                "Missing seed_plan_hash for policies "
                f"{missing_seed_hash_policies}. Re-run training or pass --allow-missing-seed-plan."
            )
        if len(observed_hashes) != 1:
            raise ValueError("Compared runs do not share the same seed_plan_hash.")

    combined_rows = baseline_rows_for_output + combined_rows
    write_csv_rows(combined_rows, LEARNING_CURVE_RAW_FIELDNAMES, raw_csv_path)
    ordered_return_matrices = _ordered_policy_matrices(
        baseline_matrix=baseline_return_matrix,
        policy_matrices=policy_return_matrices,
    )
    plot_multi_policy_learning_curve_metrics(
        epochs=reference_epochs,
        policy_return_matrices=ordered_return_matrices,
        output_path=plot_path,
        max_x_ticks=int(max_x_ticks),
    )

    summary_payload = {
        "policies": list(ordered_return_matrices),
        "input_csvs": input_csvs,
        "seed_plan_hashes": seed_plan_hashes,
        "seed_plan_hash": next(iter(observed_hashes)) if len(observed_hashes) == 1 else None,
        "epochs": [int(value) for value in reference_epochs],
        "learning_seeds": [int(value) for value in reference_learning_seeds],
        "mean_return_by_policy": {},
        "ci95_low_return_by_policy": {},
        "ci95_high_return_by_policy": {},
        "comparison_raw_csv": str(raw_csv_path),
        "comparison_plot": str(plot_path),
    }
    for policy, matrix in ordered_return_matrices.items():
        band = compute_mean_confidence_band(matrix)
        summary_payload["mean_return_by_policy"][policy] = [
            float(value) for value in band.mean
        ]
        summary_payload["ci95_low_return_by_policy"][policy] = [
            float(value) for value in band.low
        ]
        summary_payload["ci95_high_return_by_policy"][policy] = [
            float(value) for value in band.high
        ]
    write_json(summary_payload, summary_path)

    return PolicySearchComparison(
        epochs=reference_epochs,
        learning_seeds=reference_learning_seeds,
        seed_plan_hash=summary_payload["seed_plan_hash"],
        output_raw_csv=raw_csv_path,
        output_summary_json=summary_path,
        output_plot=plot_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple policy-search learning_curve_raw.csv files."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        help="Policy run in policy=path form, e.g. rwr=artifacts/runs/rwr/learning_curve_raw.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where combined CSV, summary JSON, and comparison plot are written.",
    )
    parser.add_argument(
        "--allow-missing-seed-plan",
        action="store_true",
        help="Allow comparing old runs without seed_plan_hash metadata.",
    )
    parser.add_argument(
        "--max-x-ticks",
        type=int,
        default=8,
        help="Maximum number of x-axis epoch ticks in the output plot.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = compare_policy_search_runs(
        run_specs=args.run,
        output_dir=args.output_dir,
        require_seed_plan_hash=not bool(args.allow_missing_seed_plan),
        max_x_ticks=int(args.max_x_ticks),
    )
    print(f"Saved comparison raw CSV to {result.output_raw_csv}")
    print(f"Saved comparison summary to {result.output_summary_json}")
    print(f"Saved comparison plot to {result.output_plot}")
    if result.seed_plan_hash is not None:
        print(f"Verified shared seed plan sha256={result.seed_plan_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
