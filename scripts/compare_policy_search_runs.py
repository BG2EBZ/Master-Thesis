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
from train.rwr.plotting import (
    plot_multi_policy_learning_curve_metric_panels,
    plot_multi_policy_learning_curve_metrics,
)
from train.rwr.seed_plan import load_seed_plan, seed_plan_hash

DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "runs" / (
    f"policy_search_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
DEFAULT_RAW_CSV_NAME = "policy_search_comparison_raw.csv"
DEFAULT_SUMMARY_NAME = "policy_search_comparison_summary.json"
DEFAULT_PLOT_NAME = "policy_search_comparison_plot.png"
DEFAULT_OTHER_METRICS_PLOT_NAME = "policy_search_comparison_other_metrics.png"
RETURN_METRIC = "mean_return"
METRIC_COLUMNS = (
    RETURN_METRIC,
    "mean_duration_seconds",
    "mean_overwhelmed_triggers",
    "mean_impatient_triggers",
    "mean_distracted_triggers",
)
OTHER_METRICS = tuple(metric for metric in METRIC_COLUMNS if metric != RETURN_METRIC)
METRIC_LABELS = {
    "mean_duration_seconds": ("Episode Duration", "Seconds"),
    "mean_overwhelmed_triggers": ("Overwhelmed Triggers", "Mean count"),
    "mean_impatient_triggers": ("Impatient Triggers", "Mean count"),
    "mean_distracted_triggers": ("Distracted Triggers", "Mean count"),
}


@dataclass(frozen=True)
class PolicyRunSpec:
    policy: str
    csv_path: Path


@dataclass(frozen=True)
class PolicySearchComparison:
    epochs: list[int]
    learning_seeds: list[int]
    seed_plan_hash: str | None
    max_epoch: int | None
    truncated_seed_schedule_verified: bool
    output_raw_csv: Path
    output_summary_json: Path
    output_plot: Path
    output_other_metrics_plot: Path


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


def _build_metric_matrices(
    rows: Sequence[dict[str, float | int | str]],
    *,
    metric_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    matrices: dict[str, np.ndarray] = {}
    reference_rows: list[int] | None = None
    reference_epochs: list[int] | None = None
    for metric_name in metric_names:
        matrix, ordered_rows, ordered_epochs = _build_metric_matrix(
            rows,
            metric_name=metric_name,
        )
        if reference_rows is None:
            reference_rows = ordered_rows
            reference_epochs = ordered_epochs
        elif ordered_rows != reference_rows or ordered_epochs != reference_epochs:
            raise ValueError(f"Matrix keys for {metric_name} do not match previous metrics.")
        matrices[metric_name] = matrix

    if reference_rows is None or reference_epochs is None:
        raise ValueError("metric_names must not be empty.")
    return matrices, reference_rows, reference_epochs


def _load_seed_plan_hash(input_csv: Path) -> str | None:
    summary_path = Path(input_csv).parent / DEFAULT_LEARNING_CURVE_SUMMARY_NAME
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            seed_hash = payload.get("seed_plan_hash")
            if seed_hash is not None:
                return str(seed_hash)

    seed_plan_path = Path(input_csv).parent / "seed_plan.json"
    if seed_plan_path.exists():
        return seed_plan_hash(load_seed_plan(seed_plan_path))
    return None


def _filter_rows_by_max_epoch(
    rows: Sequence[dict[str, float | int | str]],
    *,
    max_epoch: int | None,
) -> list[dict[str, float | int | str]]:
    if max_epoch is None:
        return list(rows)
    return [row for row in rows if int(row["epoch"]) <= int(max_epoch)]


def _load_truncated_eval_seed_schedule_from_seed_plan(
    input_csv: Path,
    *,
    epochs: Sequence[int],
) -> dict[int, dict[int, tuple[int, ...]]] | None:
    seed_plan_path = Path(input_csv).parent / "seed_plan.json"
    if not seed_plan_path.exists():
        return None
    seed_plan = load_seed_plan(seed_plan_path)
    schedule: dict[int, dict[int, tuple[int, ...]]] = {}
    for learning_seed_plan in seed_plan.learning_seed_plans:
        epoch_schedule: dict[int, tuple[int, ...]] = {}
        for epoch in epochs:
            if int(epoch) >= len(learning_seed_plan.eval_seeds_by_epoch):
                raise ValueError(
                    f"{seed_plan_path} does not contain evaluation seeds for epoch {epoch}."
                )
            epoch_schedule[int(epoch)] = tuple(
                int(seed) for seed in learning_seed_plan.eval_seeds_by_epoch[int(epoch)]
            )
        schedule[int(learning_seed_plan.learning_seed)] = epoch_schedule
    return schedule


def _load_truncated_eval_seed_schedule_from_summary(
    input_csv: Path,
    *,
    epochs: Sequence[int],
) -> dict[int, dict[int, tuple[int, ...]]] | None:
    summary_path = Path(input_csv).parent / DEFAULT_LEARNING_CURVE_SUMMARY_NAME
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    raw_schedules = payload.get("learning_curve_eval_seeds_by_learning_seed")
    if not isinstance(raw_schedules, list):
        return None

    required_epochs = {int(epoch) for epoch in epochs}
    schedule: dict[int, dict[int, tuple[int, ...]]] = {}
    for item in raw_schedules:
        if not isinstance(item, dict) or "learning_seed" not in item:
            return None
        raw_epoch_schedules = item.get("epoch_eval_seeds")
        if not isinstance(raw_epoch_schedules, list):
            return None
        epoch_schedule: dict[int, tuple[int, ...]] = {}
        for epoch_item in raw_epoch_schedules:
            if not isinstance(epoch_item, dict):
                return None
            epoch = int(epoch_item["epoch"])
            if epoch in required_epochs:
                epoch_schedule[epoch] = tuple(int(seed) for seed in epoch_item["seeds"])
        if set(epoch_schedule) != required_epochs:
            raise ValueError(
                f"{summary_path} does not contain evaluation seeds for all compared epochs."
            )
        schedule[int(item["learning_seed"])] = epoch_schedule
    return schedule


def _load_truncated_eval_seed_schedule(
    input_csv: Path,
    *,
    epochs: Sequence[int],
) -> dict[int, dict[int, tuple[int, ...]]] | None:
    return _load_truncated_eval_seed_schedule_from_seed_plan(
        input_csv,
        epochs=epochs,
    ) or _load_truncated_eval_seed_schedule_from_summary(
        input_csv,
        epochs=epochs,
    )


def _describe_axis(values: Sequence[int]) -> str:
    if not values:
        return "0 values"
    if len(values) == 1:
        return f"1 value [{values[0]}]"
    return f"{len(values)} values [{values[0]}..{values[-1]}]"


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
    max_epoch: int | None = None,
) -> PolicySearchComparison:
    if not run_specs:
        raise ValueError("At least one --run policy=path is required.")
    if max_epoch is not None and int(max_epoch) < 0:
        raise ValueError("max_epoch must be non-negative.")
    _validate_unique_policies(run_specs)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_dir / DEFAULT_RAW_CSV_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    plot_path = output_dir / DEFAULT_PLOT_NAME
    other_metrics_plot_path = output_dir / DEFAULT_OTHER_METRICS_PLOT_NAME

    reference_learning_seeds: list[int] | None = None
    reference_epochs: list[int] | None = None
    reference_policy: str | None = None
    reference_csv: Path | None = None
    baseline_metric_matrices: dict[str, np.ndarray] | None = None
    baseline_rows_for_output: list[dict[str, float | int | str]] | None = None
    policy_metric_matrices: dict[str, dict[str, np.ndarray]] = {}
    combined_rows: list[dict[str, float | int | str]] = []
    input_csvs: dict[str, str] = {}
    seed_plan_hashes: dict[str, str | None] = {}
    truncated_seed_schedule_verified = False

    for run_spec in run_specs:
        rows = _filter_rows_by_max_epoch(
            _read_learning_curve_rows(run_spec.csv_path),
            max_epoch=max_epoch,
        )
        input_csvs[run_spec.policy] = str(run_spec.csv_path)
        seed_plan_hashes[run_spec.policy] = _load_seed_plan_hash(run_spec.csv_path)
        learned_rows = [row for row in rows if row["policy"] == run_spec.policy]
        if not learned_rows:
            raise ValueError(f"No rows with policy={run_spec.policy!r} found in {run_spec.csv_path}")
        learned_matrices, learning_seeds, epochs = _build_metric_matrices(
            learned_rows,
            metric_names=METRIC_COLUMNS,
        )
        if reference_learning_seeds is None:
            reference_learning_seeds = learning_seeds
            reference_epochs = epochs
            reference_policy = run_spec.policy
            reference_csv = run_spec.csv_path
        elif learning_seeds != reference_learning_seeds or epochs != reference_epochs:
            raise ValueError(
                "Compared policy runs do not share learning seeds and epochs. "
                f"Reference {reference_policy} ({reference_csv}) has "
                f"learning seeds {_describe_axis(reference_learning_seeds)} and "
                f"epochs {_describe_axis(reference_epochs or [])}; "
                f"{run_spec.policy} ({run_spec.csv_path}) has "
                f"learning seeds {_describe_axis(learning_seeds)} and "
                f"epochs {_describe_axis(epochs)}."
            )
        policy_metric_matrices[run_spec.policy] = learned_matrices
        combined_rows.extend(learned_rows)

        baseline_rows = [row for row in rows if row["policy"] == "baseline"]
        if not baseline_rows:
            raise ValueError(f"No baseline rows found in {run_spec.csv_path}")
        current_baseline_matrices, baseline_learning_seeds, baseline_epochs = _build_metric_matrices(
            baseline_rows,
            metric_names=METRIC_COLUMNS,
        )
        if baseline_learning_seeds != learning_seeds or baseline_epochs != epochs:
            raise ValueError(
                f"Baseline rows in {run_spec.csv_path} do not match learned-policy seeds."
            )
        if baseline_metric_matrices is None:
            baseline_metric_matrices = current_baseline_matrices
            baseline_rows_for_output = baseline_rows
        else:
            for metric_name in METRIC_COLUMNS:
                if not np.allclose(
                    baseline_metric_matrices[metric_name],
                    current_baseline_matrices[metric_name],
                ):
                    raise ValueError(f"Baseline {metric_name} matrices differ across runs.")

    if reference_learning_seeds is None or reference_epochs is None:
        raise ValueError("No comparable policy rows found.")
    if baseline_metric_matrices is None or baseline_rows_for_output is None:
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
            if max_epoch is None:
                raise ValueError("Compared runs do not share the same seed_plan_hash.")
            reference_schedule: dict[int, dict[int, tuple[int, ...]]] | None = None
            reference_schedule_policy: str | None = None
            for run_spec in run_specs:
                schedule = _load_truncated_eval_seed_schedule(
                    run_spec.csv_path,
                    epochs=reference_epochs,
                )
                if schedule is None:
                    raise ValueError(
                        "Compared runs do not share the same seed_plan_hash, and "
                        f"{run_spec.csv_path} does not contain enough evaluation seed "
                        "metadata to verify the truncated comparison."
                    )
                missing_schedule_seeds = [
                    int(seed) for seed in reference_learning_seeds if int(seed) not in schedule
                ]
                if missing_schedule_seeds:
                    raise ValueError(
                        "Compared runs do not share the same seed_plan_hash, and "
                        f"{run_spec.csv_path} is missing evaluation seed schedules for "
                        f"learning seeds {missing_schedule_seeds}."
                    )
                compared_schedule = {
                    int(seed): schedule[int(seed)] for seed in reference_learning_seeds
                }
                if reference_schedule is None:
                    reference_schedule = compared_schedule
                    reference_schedule_policy = run_spec.policy
                elif compared_schedule != reference_schedule:
                    raise ValueError(
                        "Compared runs do not share the same seed_plan_hash, and their "
                        f"evaluation seed schedules differ through epoch {max_epoch} "
                        f"between {reference_schedule_policy} and {run_spec.policy}."
                    )
            truncated_seed_schedule_verified = True

    combined_rows = baseline_rows_for_output + combined_rows
    write_csv_rows(combined_rows, LEARNING_CURVE_RAW_FIELDNAMES, raw_csv_path)
    ordered_return_matrices = _ordered_policy_matrices(
        baseline_matrix=baseline_metric_matrices[RETURN_METRIC],
        policy_matrices={
            policy: matrices[RETURN_METRIC]
            for policy, matrices in policy_metric_matrices.items()
        },
    )
    plot_multi_policy_learning_curve_metrics(
        epochs=reference_epochs,
        policy_return_matrices=ordered_return_matrices,
        output_path=plot_path,
        max_x_ticks=int(max_x_ticks),
    )
    other_metric_panels = [
        (
            METRIC_LABELS[metric_name][0],
            METRIC_LABELS[metric_name][1],
            _ordered_policy_matrices(
                baseline_matrix=baseline_metric_matrices[metric_name],
                policy_matrices={
                    policy: matrices[metric_name]
                    for policy, matrices in policy_metric_matrices.items()
                },
            ),
        )
        for metric_name in OTHER_METRICS
    ]
    plot_multi_policy_learning_curve_metric_panels(
        epochs=reference_epochs,
        metric_panels=other_metric_panels,
        output_path=other_metrics_plot_path,
        max_x_ticks=int(max_x_ticks),
    )

    summary_payload = {
        "policies": list(ordered_return_matrices),
        "input_csvs": input_csvs,
        "max_epoch": int(max_epoch) if max_epoch is not None else None,
        "truncated_seed_schedule_verified": bool(truncated_seed_schedule_verified),
        "seed_plan_hashes": seed_plan_hashes,
        "seed_plan_hash": next(iter(observed_hashes)) if len(observed_hashes) == 1 else None,
        "epochs": [int(value) for value in reference_epochs],
        "learning_seeds": [int(value) for value in reference_learning_seeds],
        "mean_return_by_policy": {},
        "ci95_low_return_by_policy": {},
        "ci95_high_return_by_policy": {},
        "mean_by_policy": {},
        "ci95_low_by_policy": {},
        "ci95_high_by_policy": {},
        "comparison_raw_csv": str(raw_csv_path),
        "comparison_plot": str(plot_path),
        "comparison_other_metrics_plot": str(other_metrics_plot_path),
    }
    ordered_metric_matrices = {
        RETURN_METRIC: ordered_return_matrices,
        **{
            metric_name: panel_policy_matrices
            for metric_name, (_, _, panel_policy_matrices) in zip(
                OTHER_METRICS,
                other_metric_panels,
            )
        },
    }
    for metric_name, policy_matrices in ordered_metric_matrices.items():
        summary_payload["mean_by_policy"][metric_name] = {}
        summary_payload["ci95_low_by_policy"][metric_name] = {}
        summary_payload["ci95_high_by_policy"][metric_name] = {}
        for policy, matrix in policy_matrices.items():
            band = compute_mean_confidence_band(matrix)
            mean_values = [float(value) for value in band.mean]
            low_values = [float(value) for value in band.low]
            high_values = [float(value) for value in band.high]
            summary_payload["mean_by_policy"][metric_name][policy] = mean_values
            summary_payload["ci95_low_by_policy"][metric_name][policy] = low_values
            summary_payload["ci95_high_by_policy"][metric_name][policy] = high_values
            if metric_name == RETURN_METRIC:
                summary_payload["mean_return_by_policy"][policy] = mean_values
                summary_payload["ci95_low_return_by_policy"][policy] = low_values
                summary_payload["ci95_high_return_by_policy"][policy] = high_values
    write_json(summary_payload, summary_path)

    return PolicySearchComparison(
        epochs=reference_epochs,
        learning_seeds=reference_learning_seeds,
        seed_plan_hash=summary_payload["seed_plan_hash"],
        max_epoch=summary_payload["max_epoch"],
        truncated_seed_schedule_verified=truncated_seed_schedule_verified,
        output_raw_csv=raw_csv_path,
        output_summary_json=summary_path,
        output_plot=plot_path,
        output_other_metrics_plot=other_metrics_plot_path,
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
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Only compare rows with epoch <= this value, e.g. --max-epoch 50.",
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
        max_epoch=args.max_epoch,
    )
    print(f"Saved comparison raw CSV to {result.output_raw_csv}")
    print(f"Saved comparison summary to {result.output_summary_json}")
    print(f"Saved comparison plot to {result.output_plot}")
    print(f"Saved comparison other-metrics plot to {result.output_other_metrics_plot}")
    if result.seed_plan_hash is not None:
        print(f"Verified shared seed plan sha256={result.seed_plan_hash}")
    elif result.truncated_seed_schedule_verified:
        print(
            "Verified matching truncated evaluation seed schedules "
            f"through epoch {result.max_epoch}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
