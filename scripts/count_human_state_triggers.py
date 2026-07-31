from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env import MuseumEnv
from museum_env.human import HumanMode

DEFAULT_EPISODES = 20
DEFAULT_MASTER_SEED = 42
DEFAULT_N_HUMANS = 15
DEFAULT_STATES = (
    HumanMode.OVERWHELMED,
    HumanMode.IMPATIENT,
    HumanMode.DISTRACTED,
    HumanMode.CURIOSITY,
)
ALL_HUMAN_MODES = (
    HumanMode.WANDERING,
    HumanMode.FOLLOWING,
    HumanMode.LISTENING,
    HumanMode.CURIOSITY,
    HumanMode.DISTRACTED,
    HumanMode.OVERWHELMED,
    HumanMode.IMPATIENT,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _normalize_states(raw_states: Sequence[str]) -> tuple[str, ...]:
    valid_modes = set(ALL_HUMAN_MODES)
    states: list[str] = []
    for raw_state in raw_states:
        for part in str(raw_state).split(","):
            state = part.strip().lower()
            if not state:
                continue
            if state not in valid_modes:
                valid = ", ".join(sorted(valid_modes))
                raise argparse.ArgumentTypeError(
                    f"unknown human state {state!r}; expected one of: {valid}"
                )
            if state not in states:
                states.append(state)
    if not states:
        raise argparse.ArgumentTypeError("at least one human state must be provided")
    return tuple(states)


def _build_episode_seeds(master_seed: int, episodes: int) -> list[int]:
    seed_sequences = np.random.SeedSequence(int(master_seed)).spawn(int(episodes))
    return [
        int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        for seed_sequence in seed_sequences
    ]


def _current_human_modes(env: MuseumEnv) -> list[str]:
    return [str(human.mode) for human in env.humans]


def _count_mode_entries(
    previous_modes: Sequence[str],
    current_modes: Sequence[str],
    target_states: Sequence[str],
) -> dict[str, int]:
    target_set = set(target_states)
    counts = {state: 0 for state in target_states}
    for previous_mode, current_mode in zip(previous_modes, current_modes):
        current = str(current_mode)
        if current in target_set and str(previous_mode) != current:
            counts[current] += 1
    return counts


def run_episode(
    env: MuseumEnv,
    seed: int,
    target_states: Sequence[str],
) -> dict[str, int | float | str]:
    _obs, _info = env.reset(seed=int(seed))
    del _obs, _info

    previous_modes = _current_human_modes(env)
    state_counts = {state: 0 for state in target_states}
    terminated = False
    truncated = False
    info = {}
    while not (terminated or truncated):
        _obs, _reward, terminated, truncated, info = env.step(None)
        del _obs, _reward
        current_modes = [str(mode) for mode in info["crowd"]["modes"]]
        entry_counts = _count_mode_entries(previous_modes, current_modes, target_states)
        for state in target_states:
            state_counts[state] += int(entry_counts[state])
        previous_modes = current_modes

    episode_info = info["episode"]
    total = int(sum(state_counts.values()))
    return {
        "steps": int(episode_info["step"]),
        "duration_seconds": float(episode_info.get("duration_seconds", 0.0)),
        "terminated_reason": str(episode_info["terminated_reason"]),
        **state_counts,
        "total": total,
    }


def _print_episode_table(rows: Sequence[dict[str, int | float | str]], states: Sequence[str]) -> None:
    fieldnames = ("episode", "seed", "steps", "duration_s", "reason", *states, "total")
    print(",".join(fieldnames))
    for row in rows:
        values = [
            str(int(row["episode"])),
            str(int(row["seed"])),
            str(int(row["steps"])),
            f"{float(row['duration_seconds']):.2f}",
            str(row["terminated_reason"]),
        ]
        values.extend(str(int(row[state])) for state in states)
        values.append(str(int(row["total"])))
        print(",".join(values))


def _print_summary(rows: Sequence[dict[str, int | float | str]], states: Sequence[str]) -> None:
    print("SUMMARY")
    for state in (*states, "total"):
        values = np.array([int(row[state]) for row in rows], dtype=np.float64)
        print(
            f"{state}: "
            f"mean={values.mean():.3f}, "
            f"std={values.std(ddof=0):.3f}, "
            f"min={values.min():.0f}, "
            f"max={values.max():.0f}, "
            f"sum={values.sum():.0f}"
        )
    success_count = sum(
        str(row["terminated_reason"]) == "final_listen_ready"
        for row in rows
    )
    print(f"success_count={success_count}/{len(rows)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MuseumEnv episodes and count human state triggers as mode-entry transitions."
        )
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=_positive_int,
        default=DEFAULT_EPISODES,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help="Master seed used to generate one rollout seed per episode.",
    )
    parser.add_argument(
        "--n-humans",
        type=_positive_int,
        default=DEFAULT_N_HUMANS,
        help="Number of humans to simulate.",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        help=(
            "Human modes to count. Values may be space-separated or comma-separated; "
            "default: overwhelmed impatient distracted curiosity."
        ),
    )
    args = parser.parse_args(argv)
    try:
        args.states = _normalize_states(args.states)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_states = tuple(args.states)
    episode_seeds = _build_episode_seeds(
        master_seed=int(args.master_seed),
        episodes=int(args.episodes),
    )

    rows: list[dict[str, int | float | str]] = []
    env = MuseumEnv(
        render_mode=None,
        enable_event_logs=False,
        n_humans=int(args.n_humans),
    )
    try:
        for episode_idx, seed in enumerate(episode_seeds, start=1):
            row = run_episode(env, seed=seed, target_states=target_states)
            row["episode"] = int(episode_idx)
            row["seed"] = int(seed)
            rows.append(row)
    finally:
        env.close()

    _print_episode_table(rows, target_states)
    _print_summary(rows, target_states)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
