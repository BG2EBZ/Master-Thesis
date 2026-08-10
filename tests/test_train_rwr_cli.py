import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import train_reps, train_rwr
from train.rwr.defaults import DEFAULT_BETA, DEFAULT_EPS, DEFAULT_OUTPUT_DIR


class TrainRwrCliTest(unittest.TestCase):
    def test_defaults_keep_rwr_behavior(self):
        args = train_rwr.build_arg_parser().parse_args([])

        self.assertEqual(float(args.beta), DEFAULT_BETA)
        self.assertFalse(hasattr(args, "eps"))
        self.assertIsNone(args.output_dir)
        self.assertIsNone(args.seed_plan)
        self.assertEqual(train_rwr._default_output_dir_for_algorithm("rwr"), DEFAULT_OUTPUT_DIR)

    def test_reps_script_parses_eps(self):
        args = train_reps.build_arg_parser().parse_args(["--eps", "0.5"])
        output_dir = train_reps._default_output_dir_for_algorithm("reps")

        self.assertFalse(hasattr(args, "beta"))
        self.assertEqual(float(args.eps), 0.5)
        self.assertTrue(output_dir.name.startswith("reps_"))
        self.assertEqual(output_dir.parent.name, "runs")

    def test_rwr_script_rejects_algorithm_flag(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                train_rwr.build_arg_parser().parse_args(["--algorithm", "reps"])


if __name__ == "__main__":
    unittest.main()
