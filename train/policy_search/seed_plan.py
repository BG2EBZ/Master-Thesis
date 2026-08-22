from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from train.common.evaluation_seeds import FIXED_EVALUATION_SEEDS
from train.policy_search.schedules import (
    build_epoch_training_seed_schedule,
    build_seed_schedule,
)


@dataclass(frozen=True)
class LearningSeedPlan:
    learning_seed: int
    train_seeds_by_epoch: list[list[int]]
    eval_seeds_by_epoch: list[list[int]]


@dataclass(frozen=True)
class SeedPlan:
    master_seed: int
    epochs: int
    train_seeds_per_epoch: int
    n_learning_seeds: int
    n_eval_seeds: int
    learning_seed_plans: list[LearningSeedPlan]


def build_seed_plan(
    *,
    master_seed: int,
    epochs: int,
    train_seeds_per_epoch: int,
    n_learning_seeds: int,
    n_eval_seeds: int,
) -> SeedPlan:
    resolved_epochs = int(epochs)
    resolved_train_seeds_per_epoch = int(train_seeds_per_epoch)
    resolved_n_learning_seeds = int(n_learning_seeds)
    resolved_n_eval_seeds = int(n_eval_seeds)
    if resolved_epochs <= 0:
        raise ValueError("epochs must be positive")
    if resolved_train_seeds_per_epoch <= 0:
        raise ValueError("train_seeds_per_epoch must be positive")
    if resolved_n_learning_seeds <= 0:
        raise ValueError("n_learning_seeds must be positive")
    if resolved_n_eval_seeds <= 0:
        raise ValueError("n_eval_seeds must be positive")

    learning_seed_sequences = np.random.SeedSequence(int(master_seed)).spawn(
        resolved_n_learning_seeds
    )
    learning_seeds = [
        int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        for seed_sequence in learning_seed_sequences
    ]

    learning_seed_plans: list[LearningSeedPlan] = []
    for learning_seed in learning_seeds:
        _, train_seed_sequence, curve_seed_sequence = np.random.SeedSequence(
            int(learning_seed)
        ).spawn(3)
        train_seed_rng = np.random.default_rng(train_seed_sequence)
        eval_seed_rng = np.random.default_rng(curve_seed_sequence)
        learning_seed_plans.append(
            LearningSeedPlan(
                learning_seed=int(learning_seed),
                train_seeds_by_epoch=build_epoch_training_seed_schedule(
                    train_seed_rng,
                    epochs=resolved_epochs,
                    seeds_per_epoch=resolved_train_seeds_per_epoch,
                    excluded_seeds=FIXED_EVALUATION_SEEDS,
                ),
                eval_seeds_by_epoch=build_seed_schedule(
                    eval_seed_rng,
                    count=resolved_epochs + 1,
                    seeds_per_item=resolved_n_eval_seeds,
                    excluded_seeds=FIXED_EVALUATION_SEEDS,
                ),
            )
        )

    return SeedPlan(
        master_seed=int(master_seed),
        epochs=resolved_epochs,
        train_seeds_per_epoch=resolved_train_seeds_per_epoch,
        n_learning_seeds=resolved_n_learning_seeds,
        n_eval_seeds=resolved_n_eval_seeds,
        learning_seed_plans=learning_seed_plans,
    )


def seed_plan_to_dict(seed_plan: SeedPlan) -> dict[str, Any]:
    return {
        "master_seed": int(seed_plan.master_seed),
        "epochs": int(seed_plan.epochs),
        "train_seeds_per_epoch": int(seed_plan.train_seeds_per_epoch),
        "n_learning_seeds": int(seed_plan.n_learning_seeds),
        "n_eval_seeds": int(seed_plan.n_eval_seeds),
        "learning_seed_plans": [
            {
                "learning_seed": int(item.learning_seed),
                "train_seeds_by_epoch": [
                    [int(seed) for seed in epoch_seeds]
                    for epoch_seeds in item.train_seeds_by_epoch
                ],
                "eval_seeds_by_epoch": [
                    [int(seed) for seed in epoch_seeds]
                    for epoch_seeds in item.eval_seeds_by_epoch
                ],
            }
            for item in seed_plan.learning_seed_plans
        ],
    }


def seed_plan_from_dict(payload: dict[str, Any]) -> SeedPlan:
    try:
        learning_seed_plan_payloads = payload["learning_seed_plans"]
    except KeyError as exc:
        raise ValueError("Seed plan is missing learning_seed_plans.") from exc
    if not isinstance(learning_seed_plan_payloads, list):
        raise ValueError("Seed plan learning_seed_plans must be a list.")

    learning_seed_plans = [
        LearningSeedPlan(
            learning_seed=int(item["learning_seed"]),
            train_seeds_by_epoch=[
                [int(seed) for seed in epoch_seeds]
                for epoch_seeds in item["train_seeds_by_epoch"]
            ],
            eval_seeds_by_epoch=[
                [int(seed) for seed in epoch_seeds]
                for epoch_seeds in item["eval_seeds_by_epoch"]
            ],
        )
        for item in learning_seed_plan_payloads
    ]
    seed_plan = SeedPlan(
        master_seed=int(payload["master_seed"]),
        epochs=int(payload["epochs"]),
        train_seeds_per_epoch=int(payload["train_seeds_per_epoch"]),
        n_learning_seeds=int(payload["n_learning_seeds"]),
        n_eval_seeds=int(payload["n_eval_seeds"]),
        learning_seed_plans=learning_seed_plans,
    )
    validate_seed_plan(seed_plan)
    return seed_plan


def load_seed_plan(path: Path) -> SeedPlan:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Seed plan JSON must contain an object.")
    return seed_plan_from_dict(payload)


def write_seed_plan(seed_plan: SeedPlan, path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(seed_plan_to_dict(seed_plan), handle, indent=2)
        handle.write("\n")


def seed_plan_hash(seed_plan: SeedPlan) -> str:
    canonical = json.dumps(seed_plan_to_dict(seed_plan), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_seed_plan(
    seed_plan: SeedPlan,
    *,
    epochs: int | None = None,
    train_seeds_per_epoch: int | None = None,
    n_learning_seeds: int | None = None,
    n_eval_seeds: int | None = None,
) -> None:
    if int(seed_plan.epochs) <= 0:
        raise ValueError("Seed plan epochs must be positive.")
    if int(seed_plan.train_seeds_per_epoch) <= 0:
        raise ValueError("Seed plan train_seeds_per_epoch must be positive.")
    if int(seed_plan.n_learning_seeds) <= 0:
        raise ValueError("Seed plan n_learning_seeds must be positive.")
    if int(seed_plan.n_eval_seeds) <= 0:
        raise ValueError("Seed plan n_eval_seeds must be positive.")

    if epochs is not None and int(seed_plan.epochs) != int(epochs):
        raise ValueError("Seed plan epochs do not match requested epochs.")
    if (
        train_seeds_per_epoch is not None
        and int(seed_plan.train_seeds_per_epoch) != int(train_seeds_per_epoch)
    ):
        raise ValueError(
            "Seed plan train_seeds_per_epoch does not match requested train_seeds_per_epoch."
        )
    if n_learning_seeds is not None and int(seed_plan.n_learning_seeds) != int(n_learning_seeds):
        raise ValueError("Seed plan n_learning_seeds does not match requested n_learning_seeds.")
    if n_eval_seeds is not None and int(seed_plan.n_eval_seeds) != int(n_eval_seeds):
        raise ValueError("Seed plan n_eval_seeds does not match requested n_eval_seeds.")

    if len(seed_plan.learning_seed_plans) != int(seed_plan.n_learning_seeds):
        raise ValueError("Seed plan learning_seed_plans count does not match n_learning_seeds.")

    seen_learning_seeds: set[int] = set()
    for item in seed_plan.learning_seed_plans:
        if int(item.learning_seed) in seen_learning_seeds:
            raise ValueError("Seed plan contains duplicate learning seeds.")
        seen_learning_seeds.add(int(item.learning_seed))
        _validate_schedule(
            item.train_seeds_by_epoch,
            expected_epochs=int(seed_plan.epochs),
            expected_seeds_per_epoch=int(seed_plan.train_seeds_per_epoch),
            schedule_name="train_seeds_by_epoch",
        )
        _validate_schedule(
            item.eval_seeds_by_epoch,
            expected_epochs=int(seed_plan.epochs) + 1,
            expected_seeds_per_epoch=int(seed_plan.n_eval_seeds),
            schedule_name="eval_seeds_by_epoch",
        )


def learning_seed_plan_by_seed(seed_plan: SeedPlan) -> dict[int, LearningSeedPlan]:
    validate_seed_plan(seed_plan)
    return {int(item.learning_seed): item for item in seed_plan.learning_seed_plans}


def copy_seed_schedule(schedule: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(seed) for seed in epoch_seeds] for epoch_seeds in schedule]


def _validate_schedule(
    schedule: Sequence[Sequence[int]],
    *,
    expected_epochs: int,
    expected_seeds_per_epoch: int,
    schedule_name: str,
) -> None:
    if len(schedule) != int(expected_epochs):
        raise ValueError(f"Seed plan {schedule_name} epoch count is invalid.")
    for epoch_seeds in schedule:
        if len(epoch_seeds) != int(expected_seeds_per_epoch):
            raise ValueError(f"Seed plan {schedule_name} seed count is invalid.")
        for seed in epoch_seeds:
            int(seed)
