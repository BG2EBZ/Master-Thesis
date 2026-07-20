from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np

from museum_env import MuseumEnv


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    duration_seconds: float
    overwhelmed_triggers: int
    impatient_triggers: int
    distracted_triggers: int
    success: bool


_CACHED_ENV: gym.Env | None = None
_CACHED_ENV_N_HUMANS: int | None = None
_CACHED_ENV_REWARD_CONFIG: Any = None


def resolve_reward_config(reward_config: Any, default_reward_config: Any) -> Any:
    return default_reward_config if reward_config is None else reward_config


def get_cached_env(
    n_humans: int,
    reward_config: Any = None,
    *,
    default_reward_config: Any,
    wrapper_factory: Callable[[MuseumEnv, Any], gym.Env],
) -> gym.Env:
    global _CACHED_ENV, _CACHED_ENV_N_HUMANS, _CACHED_ENV_REWARD_CONFIG
    requested_n_humans = int(n_humans)
    requested_reward_config = resolve_reward_config(reward_config, default_reward_config)
    if _CACHED_ENV is None:
        _CACHED_ENV = wrapper_factory(
            MuseumEnv(
                render_mode=None,
                enable_event_logs=False,
                n_humans=requested_n_humans,
            ),
            requested_reward_config,
        )
        _CACHED_ENV_N_HUMANS = requested_n_humans
        _CACHED_ENV_REWARD_CONFIG = requested_reward_config
        return _CACHED_ENV

    if (
        _CACHED_ENV_N_HUMANS != requested_n_humans
        or _CACHED_ENV_REWARD_CONFIG != requested_reward_config
    ):
        close_cached_env()
        _CACHED_ENV = wrapper_factory(
            MuseumEnv(
                render_mode=None,
                enable_event_logs=False,
                n_humans=requested_n_humans,
            ),
            requested_reward_config,
        )
        _CACHED_ENV_N_HUMANS = requested_n_humans
        _CACHED_ENV_REWARD_CONFIG = requested_reward_config
    return _CACHED_ENV


def close_cached_env() -> None:
    global _CACHED_ENV, _CACHED_ENV_N_HUMANS, _CACHED_ENV_REWARD_CONFIG
    if _CACHED_ENV is not None:
        _CACHED_ENV.close()
    _CACHED_ENV = None
    _CACHED_ENV_N_HUMANS = None
    _CACHED_ENV_REWARD_CONFIG = None


atexit.register(close_cached_env)


def run_episode(
    env: gym.Env,
    theta: np.ndarray,
    seed: int,
    *,
    theta_to_guide_config: Callable[[np.ndarray], Any],
    print_explanations: bool = True,
) -> EpisodeResult:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    base_env.set_guide_behavior_config(theta_to_guide_config(theta))
    _observation, _info = env.reset(seed=seed)

    terminated = False
    truncated = False
    step_info = {}
    explanation_started_count = 0
    explanation_finished_count = 0

    while not (terminated or truncated):
        _observation, _reward, terminated, truncated, step_info = env.step(None)

        if print_explanations:
            step = base_env.step_count
            sim_time = step * base_env.dt
            events = step_info["events"]

            if events.get("started_listen_wait"):
                explanation_started_count += 1
                label = "A" if explanation_started_count == 1 else "B"
                print(f"[t={sim_time:.3f}s step={step}] robot start explanation {label}")

            if events.get("completed_listen_wait"):
                explanation_finished_count += 1
                label = "A" if explanation_finished_count == 1 else "B"
                print(f"[t={sim_time:.3f}s step={step}] robot finish explanation {label}")

    episode_info = step_info["episode"]
    return EpisodeResult(
        episode_return=float(episode_info["return"]),
        duration_seconds=float(episode_info["duration_seconds"]),
        overwhelmed_triggers=int(episode_info["overwhelmed_triggers"]),
        impatient_triggers=int(episode_info["impatient_triggers"]),
        distracted_triggers=int(episode_info["distracted_triggers"]),
        success=episode_info["terminated_reason"] == "final_listen_ready",
    )


def run_episode_batch(
    tasks: Sequence[tuple[Any, ...]],
    executor: ProcessPoolExecutor | None,
    evaluate_episode_task: Callable[[tuple[Any, ...]], EpisodeResult],
) -> list[EpisodeResult]:
    if executor is None:
        return [evaluate_episode_task(task) for task in tasks]
    return list(executor.map(evaluate_episode_task, tasks))
