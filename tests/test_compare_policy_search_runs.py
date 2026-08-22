import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.compare_policy_search_runs import (
    METRIC_COLUMNS,
    PolicyRunSpec,
    compare_policy_search_runs,
    parse_run_spec,
)
from train.policy_search.defaults import (
    DEFAULT_LEARNING_CURVE_SUMMARY_NAME,
    LEARNING_CURVE_RAW_FIELDNAMES,
)
from train.policy_search.seed_plan import (
    LearningSeedPlan,
    SeedPlan,
    build_seed_plan,
    seed_plan_hash,
    write_seed_plan,
)


def _row(policy, learning_seed, epoch, mean_return):
    return {
        "policy": policy,
        "learning_seed": learning_seed,
        "evaluation_seed": -1,
        "epoch": epoch,
        "mean_return": mean_return,
        "mean_duration_seconds": 12.0,
        "mean_overwhelmed_triggers": 0.0,
        "mean_impatient_triggers": 1.0,
        "mean_distracted_triggers": 2.0,
    }


def _write_run(
    run_dir: Path,
    *,
    policy: str,
    seed_hash: str,
    offset: float = 0.0,
    learning_seeds=(101, 202),
    epochs=(0, 1),
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for learning_seed in learning_seeds:
        for epoch in epochs:
            rows.append(_row("baseline", learning_seed, epoch, -50.0 + epoch))
            rows.append(_row(policy, learning_seed, epoch, -40.0 + epoch + offset))

    csv_path = run_dir / "learning_curve_raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with (run_dir / DEFAULT_LEARNING_CURVE_SUMMARY_NAME).open("w", encoding="utf-8") as handle:
        json.dump({"seed_plan_hash": seed_hash}, handle)
        handle.write("\n")
    return csv_path


def _write_seed_plan_run(
    run_dir: Path,
    *,
    policy: str,
    seed_plan: SeedPlan,
    offset: float = 0.0,
    epochs: tuple[int, ...] | None = None,
) -> Path:
    learning_seeds = tuple(
        int(learning_seed_plan.learning_seed)
        for learning_seed_plan in seed_plan.learning_seed_plans
    )
    if epochs is None:
        epochs = tuple(range(int(seed_plan.epochs) + 1))
    csv_path = _write_run(
        run_dir,
        policy=policy,
        seed_hash=seed_plan_hash(seed_plan),
        offset=offset,
        learning_seeds=learning_seeds,
        epochs=epochs,
    )
    write_seed_plan(seed_plan, run_dir / "seed_plan.json")
    return csv_path


class ComparePolicySearchRunsTest(unittest.TestCase):
    def test_parse_run_spec(self):
        spec = parse_run_spec("rwr=artifacts/runs/rwr/learning_curve_raw.csv")

        self.assertEqual(spec.policy, "rwr")
        self.assertEqual(spec.csv_path.name, "learning_curve_raw.csv")

    def test_compare_writes_combined_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rwr_csv = _write_run(root / "rwr", policy="rwr", seed_hash="same")
            reps_csv = _write_run(root / "reps", policy="reps", seed_hash="same", offset=2.0)

            result = compare_policy_search_runs(
                run_specs=[
                    PolicyRunSpec("rwr", rwr_csv),
                    PolicyRunSpec("reps", reps_csv),
                ],
                output_dir=root / "compare",
            )

            self.assertEqual(result.seed_plan_hash, "same")
            self.assertTrue(result.output_raw_csv.exists())
            self.assertTrue(result.output_summary_json.exists())
            self.assertTrue(result.output_plot.exists())
            self.assertTrue(result.output_other_metrics_plot.exists())
            with result.output_summary_json.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["policies"], ["baseline", "rwr", "reps"])
            self.assertEqual(
                summary["comparison_other_metrics_plot"],
                str(result.output_other_metrics_plot),
            )
            self.assertIn("baseline", summary["mean_return_by_policy"])
            self.assertIn("rwr", summary["mean_return_by_policy"])
            self.assertIn("reps", summary["mean_return_by_policy"])
            self.assertEqual(set(summary["mean_by_policy"]), set(METRIC_COLUMNS))
            for metric_name in METRIC_COLUMNS:
                self.assertEqual(
                    set(summary["mean_by_policy"][metric_name]),
                    {"baseline", "rwr", "reps"},
                )
                self.assertEqual(
                    set(summary["ci95_low_by_policy"][metric_name]),
                    {"baseline", "rwr", "reps"},
                )
                self.assertEqual(
                    set(summary["ci95_high_by_policy"][metric_name]),
                    {"baseline", "rwr", "reps"},
                )

    def test_compare_rejects_mismatched_seed_plan_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rwr_csv = _write_run(root / "rwr", policy="rwr", seed_hash="first")
            reps_csv = _write_run(root / "reps", policy="reps", seed_hash="second")

            with self.assertRaises(ValueError):
                compare_policy_search_runs(
                    run_specs=[
                        PolicyRunSpec("rwr", rwr_csv),
                        PolicyRunSpec("reps", reps_csv),
                    ],
                    output_dir=root / "compare",
                )

    def test_compare_loads_seed_plan_hash_from_seed_plan_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rwr_dir = root / "rwr"
            reps_dir = root / "reps"
            rwr_csv = _write_run(rwr_dir, policy="rwr", seed_hash="old")
            reps_csv = _write_run(reps_dir, policy="reps", seed_hash="old", offset=2.0)
            (rwr_dir / DEFAULT_LEARNING_CURVE_SUMMARY_NAME).unlink()
            (reps_dir / DEFAULT_LEARNING_CURVE_SUMMARY_NAME).unlink()

            seed_plan = build_seed_plan(
                master_seed=42,
                epochs=1,
                train_seeds_per_epoch=1,
                n_learning_seeds=2,
                n_eval_seeds=1,
            )
            write_seed_plan(seed_plan, rwr_dir / "seed_plan.json")
            write_seed_plan(seed_plan, reps_dir / "seed_plan.json")

            result = compare_policy_search_runs(
                run_specs=[
                    PolicyRunSpec("rwr", rwr_csv),
                    PolicyRunSpec("reps", reps_csv),
                ],
                output_dir=root / "compare",
            )

            self.assertIsNotNone(result.seed_plan_hash)

    def test_compare_max_epoch_truncates_longer_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            longer_plan = build_seed_plan(
                master_seed=42,
                epochs=2,
                train_seeds_per_epoch=1,
                n_learning_seeds=2,
                n_eval_seeds=1,
            )
            shorter_plan = build_seed_plan(
                master_seed=42,
                epochs=1,
                train_seeds_per_epoch=1,
                n_learning_seeds=2,
                n_eval_seeds=1,
            )
            rwr_csv = _write_seed_plan_run(root / "rwr", policy="rwr", seed_plan=longer_plan)
            reps_csv = _write_seed_plan_run(
                root / "reps",
                policy="reps",
                seed_plan=shorter_plan,
                offset=2.0,
            )

            result = compare_policy_search_runs(
                run_specs=[
                    PolicyRunSpec("rwr", rwr_csv),
                    PolicyRunSpec("reps", reps_csv),
                ],
                output_dir=root / "compare",
                max_epoch=1,
            )

            self.assertEqual(result.epochs, [0, 1])
            self.assertIsNone(result.seed_plan_hash)
            self.assertTrue(result.truncated_seed_schedule_verified)
            self.assertTrue(result.output_other_metrics_plot.exists())
            with result.output_summary_json.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["max_epoch"], 1)
            self.assertTrue(summary["truncated_seed_schedule_verified"])
            self.assertEqual(summary["epochs"], [0, 1])
            with result.output_raw_csv.open(newline="", encoding="utf-8") as handle:
                epochs = {int(row["epoch"]) for row in csv.DictReader(handle)}
            self.assertEqual(epochs, {0, 1})

    def test_compare_max_epoch_still_rejects_mismatched_epochs_after_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rwr_csv = _write_run(root / "rwr", policy="rwr", seed_hash="same", epochs=(0, 1))
            reps_csv = _write_run(root / "reps", policy="reps", seed_hash="same", epochs=(0,))

            with self.assertRaisesRegex(ValueError, "learning seeds and epochs"):
                compare_policy_search_runs(
                    run_specs=[
                        PolicyRunSpec("rwr", rwr_csv),
                        PolicyRunSpec("reps", reps_csv),
                    ],
                    output_dir=root / "compare",
                    max_epoch=1,
                )

    def test_compare_max_epoch_rejects_mismatched_truncated_eval_seed_schedule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rwr_plan = build_seed_plan(
                master_seed=42,
                epochs=1,
                train_seeds_per_epoch=1,
                n_learning_seeds=2,
                n_eval_seeds=1,
            )
            tampered_learning_seed_plans = []
            for index, item in enumerate(rwr_plan.learning_seed_plans):
                eval_seeds_by_epoch = [list(epoch_seeds) for epoch_seeds in item.eval_seeds_by_epoch]
                if index == 0:
                    eval_seeds_by_epoch[1][0] = int(eval_seeds_by_epoch[1][0]) + 1
                tampered_learning_seed_plans.append(
                    LearningSeedPlan(
                        learning_seed=item.learning_seed,
                        train_seeds_by_epoch=[
                            list(epoch_seeds) for epoch_seeds in item.train_seeds_by_epoch
                        ],
                        eval_seeds_by_epoch=eval_seeds_by_epoch,
                    )
                )
            reps_plan = SeedPlan(
                master_seed=rwr_plan.master_seed,
                epochs=rwr_plan.epochs,
                train_seeds_per_epoch=rwr_plan.train_seeds_per_epoch,
                n_learning_seeds=rwr_plan.n_learning_seeds,
                n_eval_seeds=rwr_plan.n_eval_seeds,
                learning_seed_plans=tampered_learning_seed_plans,
            )
            rwr_csv = _write_seed_plan_run(root / "rwr", policy="rwr", seed_plan=rwr_plan)
            reps_csv = _write_seed_plan_run(
                root / "reps",
                policy="reps",
                seed_plan=reps_plan,
                offset=2.0,
            )

            with self.assertRaisesRegex(ValueError, "evaluation seed schedules differ"):
                compare_policy_search_runs(
                    run_specs=[
                        PolicyRunSpec("rwr", rwr_csv),
                        PolicyRunSpec("reps", reps_csv),
                    ],
                    output_dir=root / "compare",
                    max_epoch=1,
                )


if __name__ == "__main__":
    unittest.main()
