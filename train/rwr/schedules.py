from __future__ import annotations

import os
from typing import Sequence

import numpy as np


def build_epoch_training_seed_schedule(
    rng: np.random.Generator,
    epochs: int,
    seeds_per_epoch: int,
    *,
    excluded_seeds: Sequence[int],
) -> list[list[int]]:
    return build_seed_schedule(
        rng,
        count=int(epochs),
        seeds_per_item=int(seeds_per_epoch),
        excluded_seeds=excluded_seeds,
    )


def build_seed_schedule(
    rng: np.random.Generator,
    *,
    count: int,
    seeds_per_item: int,
    excluded_seeds: Sequence[int] = (),
) -> list[list[int]]:
    excluded_seed_set = {int(seed) for seed in excluded_seeds}
    seed_schedule: list[list[int]] = []

    for _ in range(int(count)):
        item_seeds: list[int] = []
        while len(item_seeds) < int(seeds_per_item):
            sampled_seed = int(rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
            if sampled_seed in excluded_seed_set:
                continue
            item_seeds.append(sampled_seed)
        seed_schedule.append(item_seeds)

    return seed_schedule


def resolve_worker_count(*, task_count: int, max_workers: int) -> int:
    resolved_task_count = int(task_count)
    resolved_max_workers = int(max_workers)
    if resolved_max_workers <= 0:
        raise ValueError("max_workers must be positive")
    return min(resolved_task_count, os.cpu_count() or 1, resolved_max_workers)
