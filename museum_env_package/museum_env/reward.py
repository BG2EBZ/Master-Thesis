from __future__ import annotations
from dataclasses import dataclass
from .env_state import EpisodeMetrics


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the first episodic policy-search reward."""

    # completion_reward: float = 100.0
    # timeout_penalty: float = -100.0

    time_penalty_per_second: float = 0.1

    overwhelmed_trigger_penalty: float = 4.0
    impatient_trigger_penalty: float = 2.0
    distracted_trigger_penalty: float = 2.0


DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_episode_reward(
    *,
    completed: bool,
    truncated: bool,
    duration_seconds: float,
    metrics: EpisodeMetrics,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> tuple[float, dict[str, float]]:
    """
    Compute one scalar return for episodic policy search.

    Intermediate simulation steps receive zero reward.
    The full return is emitted only when the tour completes or times out.
    """

    # if completed:
    #     outcome_term = float(config.completion_reward)
    # elif truncated:
    #     outcome_term = float(config.timeout_penalty)
    # else:
    #     raise ValueError(
    #         "compute_episode_reward() should only be called when the episode ends."
    #     )

    time_term = -float(config.time_penalty_per_second) * float(duration_seconds)

    overwhelmed_term = (
        -float(config.overwhelmed_trigger_penalty)
        * int(metrics.overwhelmed_triggers)
    )

    impatient_term = (
        -float(config.impatient_trigger_penalty)
        * int(metrics.impatient_triggers)
    )

    distracted_term = (
        -float(config.distracted_trigger_penalty)
        * int(metrics.distracted_triggers)
    )

    components = {
        # "outcome": outcome_term,
        "time": time_term,
        "overwhelmed": overwhelmed_term,
        "impatient": impatient_term,
        "distracted": distracted_term,
    }

    reward = float(sum(components.values()))
    return reward, components