import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train.rwr.seed_plan import (
    build_seed_plan,
    load_seed_plan,
    seed_plan_hash,
    seed_plan_to_dict,
    validate_seed_plan,
    write_seed_plan,
)


class PolicySearchSeedPlanTest(unittest.TestCase):
    def test_seed_plan_is_deterministic_and_shape_checked(self):
        first = build_seed_plan(
            master_seed=123,
            epochs=2,
            train_seeds_per_epoch=1,
            n_learning_seeds=2,
            n_eval_seeds=3,
        )
        second = build_seed_plan(
            master_seed=123,
            epochs=2,
            train_seeds_per_epoch=1,
            n_learning_seeds=2,
            n_eval_seeds=3,
        )

        self.assertEqual(seed_plan_to_dict(first), seed_plan_to_dict(second))
        self.assertEqual(seed_plan_hash(first), seed_plan_hash(second))
        self.assertEqual(len(first.learning_seed_plans), 2)
        self.assertEqual(len(first.learning_seed_plans[0].train_seeds_by_epoch), 2)
        self.assertEqual(len(first.learning_seed_plans[0].eval_seeds_by_epoch), 3)

    def test_seed_plan_round_trips_json(self):
        seed_plan = build_seed_plan(
            master_seed=99,
            epochs=1,
            train_seeds_per_epoch=2,
            n_learning_seeds=1,
            n_eval_seeds=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seed_plan.json"
            write_seed_plan(seed_plan, path)
            loaded = load_seed_plan(path)

        self.assertEqual(seed_plan_to_dict(loaded), seed_plan_to_dict(seed_plan))
        self.assertEqual(seed_plan_hash(loaded), seed_plan_hash(seed_plan))

    def test_seed_plan_rejects_mismatched_request(self):
        seed_plan = build_seed_plan(
            master_seed=123,
            epochs=2,
            train_seeds_per_epoch=1,
            n_learning_seeds=2,
            n_eval_seeds=3,
        )

        with self.assertRaises(ValueError):
            validate_seed_plan(seed_plan, epochs=3)


if __name__ == "__main__":
    unittest.main()
