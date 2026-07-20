from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Keep BLAS/OpenMP libraries from oversubscribing CPU cores across workers
# unless the caller explicitly sets a different thread count.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from train.rwr.defaults import (
    DEFAULT_BETA,
    DEFAULT_EPOCHS,
    DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    DEFAULT_N_EVAL_SEEDS,
    DEFAULT_N_HUMANS,
    DEFAULT_N_LEARNING_SEEDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLES_PER_EPOCH,
    DEFAULT_SEED,
)
from train.rwr.rewarding import DEFAULT_EPISODE_REWARD_WEIGHTS, EpisodeRewardWeights
from train.rwr.training import train, train_across_learning_seeds


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a minimal RWR policy search loop.")
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=_positive_int,
        default=DEFAULT_SAMPLES_PER_EPOCH,
        help="Number of sampled policies per epoch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Experiment master seed for policy sampling and rollout seeding.",
    )
    parser.add_argument(
        "--beta",
        type=_positive_float,
        default=DEFAULT_BETA,
        help="RWR reward-weight temperature used in the distribution update.",
    )
    parser.add_argument(
        "--train-seeds-per-epoch",
        type=_positive_int,
        default=DEFAULT_EPOCH_TRAIN_SEED_COUNT,
        help="Number of rollout seeds used to evaluate each sampled theta per epoch.",
    )
    parser.add_argument(
        "--n-humans",
        type=_positive_int,
        default=DEFAULT_N_HUMANS,
        help="Number of humans to simulate during training rollouts.",
    )
    parser.add_argument(
        "--n-learning-seeds",
        type=_positive_int,
        default=DEFAULT_N_LEARNING_SEEDS,
        help="Number of independent learning seeds used for benchmark learning curves.",
    )
    parser.add_argument(
        "--n-eval-seeds",
        type=_positive_int,
        default=DEFAULT_N_EVAL_SEEDS,
        help="Number of fixed evaluation seeds used to estimate each epoch point.",
    )
    parser.add_argument(
        "--time-penalty-per-second",
        type=_positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.time_penalty_per_second,
        help="Reward penalty applied per simulated second.",
    )
    parser.add_argument(
        "--overwhelmed-trigger-penalty",
        type=_positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.overwhelmed_trigger_penalty,
        help="Reward penalty applied per overwhelmed trigger.",
    )
    parser.add_argument(
        "--impatient-trigger-penalty",
        type=_positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.impatient_trigger_penalty,
        help="Reward penalty applied per impatient trigger.",
    )
    parser.add_argument(
        "--distracted-trigger-penalty",
        type=_positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.distracted_trigger_penalty,
        help="Reward penalty applied per distracted trigger.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where training artifacts and optional learning-curve outputs are written.",
    )
    args = parser.parse_args(argv)
    reward_config = EpisodeRewardWeights(
        time_penalty_per_second=float(args.time_penalty_per_second),
        overwhelmed_trigger_penalty=float(args.overwhelmed_trigger_penalty),
        impatient_trigger_penalty=float(args.impatient_trigger_penalty),
        distracted_trigger_penalty=float(args.distracted_trigger_penalty),
    )
    if int(args.n_learning_seeds) == 1:
        train(
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            output_dir=args.output_dir,
            beta=float(args.beta),
            train_seeds_per_epoch=int(args.train_seeds_per_epoch),
            n_humans=int(args.n_humans),
            reward_config=reward_config,
        )
    else:
        train_across_learning_seeds(
            epochs=int(args.epochs),
            samples_per_epoch=int(args.samples_per_epoch),
            seed=int(args.seed),
            output_dir=args.output_dir,
            beta=float(args.beta),
            train_seeds_per_epoch=int(args.train_seeds_per_epoch),
            n_humans=int(args.n_humans),
            reward_config=reward_config,
            n_learning_seeds=int(args.n_learning_seeds),
            n_eval_seeds=int(args.n_eval_seeds),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
