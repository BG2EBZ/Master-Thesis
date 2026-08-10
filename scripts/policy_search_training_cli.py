from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

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
    ARTIFACTS_ROOT,
    DEFAULT_BETA,
    DEFAULT_EPS,
    DEFAULT_EPOCHS,
    DEFAULT_EPOCH_TRAIN_SEED_COUNT,
    DEFAULT_MAX_WORKERS,
    DEFAULT_N_EVAL_SEEDS,
    DEFAULT_N_HUMANS,
    DEFAULT_N_LEARNING_SEEDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLES_PER_EPOCH,
    DEFAULT_SEED,
)
from train.rwr.rewarding import DEFAULT_EPISODE_REWARD_WEIGHTS, EpisodeRewardWeights
from train.rwr.seed_plan import load_seed_plan
from train.rwr.training import train, train_across_learning_seeds

PolicySearchAlgorithm = Literal["rwr", "reps"]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return parsed


def default_output_dir_for_algorithm(algorithm: str) -> Path:
    resolved_algorithm = str(algorithm).lower()
    if resolved_algorithm == "rwr":
        return DEFAULT_OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ARTIFACTS_ROOT / "runs" / f"{resolved_algorithm}_{timestamp}"


def build_training_arg_parser(
    *,
    algorithm: PolicySearchAlgorithm,
) -> argparse.ArgumentParser:
    resolved_algorithm = str(algorithm).lower()
    if resolved_algorithm not in ("rwr", "reps"):
        raise ValueError("algorithm must be 'rwr' or 'reps'")
    parser = argparse.ArgumentParser(
        description=f"Train the guide policy-search loop with {resolved_algorithm.upper()}."
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=positive_int,
        default=DEFAULT_SAMPLES_PER_EPOCH,
        help="Number of sampled policies per epoch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Experiment master seed for policy sampling and rollout seeding.",
    )
    if resolved_algorithm == "rwr":
        parser.add_argument(
            "--beta",
            type=positive_float,
            default=DEFAULT_BETA,
            help="RWR reward-weight temperature.",
        )
    else:
        parser.add_argument(
            "--eps",
            type=positive_float,
            default=DEFAULT_EPS,
            help="REPS KL-divergence update bound.",
        )
    parser.add_argument(
        "--train-seeds-per-epoch",
        type=positive_int,
        default=DEFAULT_EPOCH_TRAIN_SEED_COUNT,
        help="Number of rollout seeds used to evaluate each sampled theta per epoch.",
    )
    parser.add_argument(
        "--n-humans",
        type=positive_int,
        default=DEFAULT_N_HUMANS,
        help="Number of humans to simulate during training rollouts.",
    )
    parser.add_argument(
        "--n-learning-seeds",
        type=positive_int,
        default=DEFAULT_N_LEARNING_SEEDS,
        help="Number of independent learning seeds used for benchmark learning curves.",
    )
    parser.add_argument(
        "--n-eval-seeds",
        type=positive_int,
        default=DEFAULT_N_EVAL_SEEDS,
        help="Number of fresh learning-curve evaluation episodes generated per epoch.",
    )
    parser.add_argument(
        "--seed-plan",
        type=Path,
        default=None,
        help=(
            "Optional seed_plan.json to reuse for multi-seed training. "
            "If omitted, a matching plan is generated and saved."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of worker processes used for parallel training rollouts.",
    )
    parser.add_argument(
        "--time-penalty-per-second",
        type=positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.time_penalty_per_second,
        help="Reward penalty applied per simulated second.",
    )
    parser.add_argument(
        "--overwhelmed-trigger-penalty",
        type=positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.overwhelmed_trigger_penalty,
        help="Reward penalty applied per overwhelmed trigger.",
    )
    parser.add_argument(
        "--impatient-trigger-penalty",
        type=positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.impatient_trigger_penalty,
        help="Reward penalty applied per impatient trigger.",
    )
    parser.add_argument(
        "--distracted-trigger-penalty",
        type=positive_float,
        default=DEFAULT_EPISODE_REWARD_WEIGHTS.distracted_trigger_penalty,
        help="Reward penalty applied per distracted trigger.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where training artifacts and optional learning-curve outputs "
            "are written."
        ),
    )
    return parser


def run_training_from_args(
    *,
    args: argparse.Namespace,
    algorithm: PolicySearchAlgorithm,
) -> None:
    resolved_algorithm = str(algorithm).lower()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else default_output_dir_for_algorithm(resolved_algorithm)
    )
    reward_config = EpisodeRewardWeights(
        time_penalty_per_second=float(args.time_penalty_per_second),
        overwhelmed_trigger_penalty=float(args.overwhelmed_trigger_penalty),
        impatient_trigger_penalty=float(args.impatient_trigger_penalty),
        distracted_trigger_penalty=float(args.distracted_trigger_penalty),
    )
    seed_plan = load_seed_plan(args.seed_plan) if args.seed_plan is not None else None
    beta = float(getattr(args, "beta", DEFAULT_BETA))
    eps = float(getattr(args, "eps", DEFAULT_EPS))

    if int(args.n_learning_seeds) == 1 and seed_plan is None:
        train(
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            output_dir=output_dir,
            beta=beta,
            eps=eps,
            algorithm=resolved_algorithm,
            train_seeds_per_epoch=int(args.train_seeds_per_epoch),
            n_humans=int(args.n_humans),
            reward_config=reward_config,
            max_workers=int(args.max_workers),
        )
    else:
        train_across_learning_seeds(
            epochs=int(args.epochs),
            samples_per_epoch=int(args.samples_per_epoch),
            seed=int(args.seed),
            output_dir=output_dir,
            beta=beta,
            eps=eps,
            algorithm=resolved_algorithm,
            train_seeds_per_epoch=int(args.train_seeds_per_epoch),
            n_humans=int(args.n_humans),
            reward_config=reward_config,
            n_learning_seeds=int(args.n_learning_seeds),
            n_eval_seeds=int(args.n_eval_seeds),
            max_workers=int(args.max_workers),
            seed_plan=seed_plan,
        )


def main_for_algorithm(
    *,
    algorithm: PolicySearchAlgorithm,
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_training_arg_parser(algorithm=algorithm)
    args = parser.parse_args(argv)
    run_training_from_args(args=args, algorithm=algorithm)
    return 0
