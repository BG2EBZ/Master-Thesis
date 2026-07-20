from pathlib import Path
import sys
import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train.rwr.rewarding import EpisodeRewardWeights, RWRRewardWrapper


class _DummyEpisodeEnv(gym.Env):
    metadata = {}

    def __init__(self, *, terminated: bool, truncated: bool):
        super().__init__()
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        del seed, options
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        del action
        info = {
            "episode": {
                "duration_seconds": 12.0,
                "overwhelmed_triggers": 1,
                "impatient_triggers": 2,
                "distracted_triggers": 3,
            }
        }
        return np.zeros((1,), dtype=np.float32), 999.0, self.terminated, self.truncated, info


class RWRRewardWrapperTests(unittest.TestCase):
    def test_non_terminal_step_returns_zero_reward(self):
        env = RWRRewardWrapper(_DummyEpisodeEnv(terminated=False, truncated=False))

        _obs, reward, terminated, truncated, info = env.step(np.zeros((1,), dtype=np.float32))

        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotIn("return", info["episode"])
        self.assertNotIn("reward_components", info["episode"])

    def test_terminal_step_computes_reward_and_populates_episode_info(self):
        weights = EpisodeRewardWeights(
            time_penalty_per_second=0.1,
            overwhelmed_trigger_penalty=4.0,
            impatient_trigger_penalty=2.0,
            distracted_trigger_penalty=2.0,
        )
        env = RWRRewardWrapper(
            _DummyEpisodeEnv(terminated=True, truncated=False),
            reward_weights=weights,
        )

        _obs, reward, terminated, truncated, info = env.step(np.zeros((1,), dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, -15.2, places=7)
        self.assertAlmostEqual(info["episode"]["return"], -15.2, places=7)
        self.assertEqual(
            info["episode"]["reward_components"],
            {
                "time": -1.2000000000000002,
                "overwhelmed": -4.0,
                "impatient": -4.0,
                "distracted": -6.0,
            },
        )

    def test_truncated_step_uses_same_terminal_reward_path(self):
        env = RWRRewardWrapper(_DummyEpisodeEnv(terminated=False, truncated=True))

        _obs, reward, terminated, truncated, info = env.step(np.zeros((1,), dtype=np.float32))

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, -15.2, places=7)
        self.assertAlmostEqual(info["episode"]["return"], -15.2, places=7)


if __name__ == "__main__":
    unittest.main()
