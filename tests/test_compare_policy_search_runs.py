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
    PolicyRunSpec,
    compare_policy_search_runs,
    parse_run_spec,
)
from train.rwr.defaults import (
    DEFAULT_LEARNING_CURVE_SUMMARY_NAME,
    LEARNING_CURVE_RAW_FIELDNAMES,
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


def _write_run(run_dir: Path, *, policy: str, seed_hash: str, offset: float = 0.0) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for learning_seed in (101, 202):
        for epoch in (0, 1):
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
            with result.output_summary_json.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["policies"], ["baseline", "rwr", "reps"])
            self.assertIn("baseline", summary["mean_return_by_policy"])
            self.assertIn("rwr", summary["mean_return_by_policy"])
            self.assertIn("reps", summary["mean_return_by_policy"])

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


if __name__ == "__main__":
    unittest.main()
