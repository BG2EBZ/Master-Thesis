from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class EpisodeRewardWeights:
    time_penalty_per_second: float = 0.1
    overwhelmed_trigger_penalty: float = 2.0
    impatient_trigger_penalty: float = 4.0
    distracted_trigger_penalty: float = 2.0


DEFAULT_EPISODE_REWARD_WEIGHTS = EpisodeRewardWeights()


def compute_episode_reward(
    *,
    duration_seconds: float,
    overwhelmed_triggers: int,
    impatient_triggers: int,
    distracted_triggers: int,
    weights: EpisodeRewardWeights = DEFAULT_EPISODE_REWARD_WEIGHTS,
) -> tuple[float, dict[str, float]]:
    time_term = -float(weights.time_penalty_per_second) * float(duration_seconds)
    overwhelmed_term = -float(weights.overwhelmed_trigger_penalty) * int(overwhelmed_triggers)
    impatient_term = -float(weights.impatient_trigger_penalty) * int(impatient_triggers)
    distracted_term = -float(weights.distracted_trigger_penalty) * int(distracted_triggers)

    components = {
        "time": time_term,
        "overwhelmed": overwhelmed_term,
        "impatient": impatient_term,
        "distracted": distracted_term,
    }
    return float(sum(components.values())), components


class RWRRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_weights: EpisodeRewardWeights | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_weights = (
            DEFAULT_EPISODE_REWARD_WEIGHTS if reward_weights is None else reward_weights
        )

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            episode_info = info.setdefault("episode", {})
            reward, reward_components = compute_episode_reward(
                duration_seconds=float(episode_info["duration_seconds"]),
                overwhelmed_triggers=int(episode_info["overwhelmed_triggers"]),
                impatient_triggers=int(episode_info["impatient_triggers"]),
                distracted_triggers=int(episode_info["distracted_triggers"]),
                weights=self.reward_weights,
            )
            episode_info["return"] = float(reward)
            episode_info["reward_components"] = reward_components
        else:
            reward = 0.0
        return observation, float(reward), terminated, truncated, info
