import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from museum_env import env_control
from museum_env.env import MuseumEnv
from museum_env.env_state import FOLLOW_PHASE_TRANSIT, EpisodeMetrics
from museum_env.human import HumanMode
from museum_env.reward import RewardConfig, compute_episode_reward


class MuseumEnvRewardTests(unittest.TestCase):
    def _make_env(self, **kwargs):
        defaults = {
            "render_mode": None,
            "enable_event_logs": False,
        }
        defaults.update(kwargs)
        return MuseumEnv(**defaults)

    def test_compute_episode_reward_uses_time_penalty_only_when_no_triggers(self):
        reward, components = compute_episode_reward(
            completed=True,
            truncated=False,
            duration_seconds=12.5,
            metrics=EpisodeMetrics(),
        )

        self.assertAlmostEqual(reward, -1.25, places=6)
        self.assertEqual(
            components,
            {
                "time": -1.25,
                "overwhelmed": -0.0,
                "impatient": -0.0,
                "distracted": -0.0,
            },
        )

    def test_compute_episode_reward_counts_each_trigger_penalty(self):
        reward, components = compute_episode_reward(
            completed=False,
            truncated=True,
            duration_seconds=10.0,
            metrics=EpisodeMetrics(
                overwhelmed_triggers=1,
                impatient_triggers=2,
                distracted_triggers=3,
            ),
        )

        self.assertAlmostEqual(reward, -15.0, places=6)
        self.assertEqual(components["time"], -1.0)
        self.assertEqual(components["overwhelmed"], -4.0)
        self.assertEqual(components["impatient"], -4.0)
        self.assertEqual(components["distracted"], -6.0)

    def test_compute_episode_reward_uses_custom_config(self):
        reward, components = compute_episode_reward(
            completed=True,
            truncated=False,
            duration_seconds=8.0,
            metrics=EpisodeMetrics(
                overwhelmed_triggers=2,
                impatient_triggers=1,
                distracted_triggers=4,
            ),
            config=RewardConfig(
                time_penalty_per_second=0.25,
                overwhelmed_trigger_penalty=1.5,
                impatient_trigger_penalty=0.75,
                distracted_trigger_penalty=0.5,
            ),
        )

        self.assertAlmostEqual(reward, -7.75, places=6)
        self.assertEqual(components["time"], -2.0)
        self.assertEqual(components["overwhelmed"], -3.0)
        self.assertEqual(components["impatient"], -0.75)
        self.assertEqual(components["distracted"], -2.0)

    def test_step_returns_zero_reward_before_episode_end_and_keeps_compact_info(self):
        env = self._make_env(n_humans=0)
        try:
            env.reset(seed=7)

            obs, reward, terminated, truncated, info = env.step(None)

            self.assertEqual(obs.shape, (4,))
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(
                sorted(info.keys()),
                ["crowd", "episode", "events", "phase", "robot"],
            )
            self.assertEqual(
                sorted(info["episode"].keys()),
                ["step", "terminated_reason"],
            )
        finally:
            env.close()

    def test_terminal_success_step_returns_episode_reward(self):
        env = self._make_env(n_humans=0)
        try:
            env.reset(seed=8)
            with patch(
                "museum_env.env_flow.progress_listening_phase",
                side_effect=lambda _env, events, _frame: setattr(
                    events, "final_listen_ready", True
                ),
            ):
                _, reward, terminated, truncated, info = env.step(None)

            expected_reward, expected_components = compute_episode_reward(
                completed=True,
                truncated=False,
                duration_seconds=float(env.step_count) * float(env.dt),
                metrics=env.episode_metrics,
                config=env.reward_config,
            )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(reward, expected_reward, places=6)
            self.assertEqual(info["episode"]["terminated_reason"], "final_listen_ready")
            self.assertAlmostEqual(
                info["episode"]["duration_seconds"],
                float(env.step_count) * float(env.dt),
                places=6,
            )
            self.assertEqual(info["episode"]["overwhelmed_triggers"], 0)
            self.assertEqual(info["episode"]["impatient_triggers"], 0)
            self.assertEqual(info["episode"]["distracted_triggers"], 0)
            self.assertAlmostEqual(info["episode"]["return"], expected_reward, places=6)
            self.assertEqual(info["episode"]["reward_components"], expected_components)
        finally:
            env.close()

    def test_truncated_step_returns_episode_reward(self):
        env = self._make_env(n_humans=0)
        try:
            env.reset(seed=9)
            env.max_steps = 1

            _, reward, terminated, truncated, info = env.step(None)

            expected_reward, expected_components = compute_episode_reward(
                completed=False,
                truncated=True,
                duration_seconds=float(env.step_count) * float(env.dt),
                metrics=env.episode_metrics,
                config=env.reward_config,
            )
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["episode"]["terminated_reason"], "max_steps")
            self.assertAlmostEqual(reward, expected_reward, places=6)
            self.assertEqual(info["episode"]["reward_components"], expected_components)
        finally:
            env.close()

    def test_custom_reward_config_changes_terminal_reward(self):
        custom_config = RewardConfig(
            time_penalty_per_second=1.0,
            overwhelmed_trigger_penalty=10.0,
            impatient_trigger_penalty=5.0,
            distracted_trigger_penalty=3.0,
        )
        env = self._make_env(n_humans=0, reward_config=custom_config)
        try:
            env.reset(seed=10)
            env.episode_metrics.distracted_triggers = 2
            env.episode_metrics.impatient_triggers = 1
            with patch(
                "museum_env.env_flow.progress_listening_phase",
                side_effect=lambda _env, events, _frame: setattr(
                    events, "final_listen_ready", True
                ),
            ):
                _, reward, terminated, truncated, info = env.step(None)

            expected_reward, expected_components = compute_episode_reward(
                completed=True,
                truncated=False,
                duration_seconds=float(env.step_count) * float(env.dt),
                metrics=env.episode_metrics,
                config=custom_config,
            )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(reward, expected_reward, places=6)
            self.assertEqual(info["episode"]["distracted_triggers"], 2)
            self.assertEqual(info["episode"]["impatient_triggers"], 1)
            self.assertEqual(info["episode"]["reward_components"], expected_components)
        finally:
            env.close()

    def test_episode_metrics_reset_on_env_reset(self):
        env = self._make_env(n_humans=0)
        try:
            env.reset(seed=11)
            env.episode_metrics.overwhelmed_triggers = 1
            env.episode_metrics.impatient_triggers = 2
            env.episode_metrics.distracted_triggers = 3

            env.reset(seed=12)

            self.assertEqual(env.episode_metrics.overwhelmed_triggers, 0)
            self.assertEqual(env.episode_metrics.impatient_triggers, 0)
            self.assertEqual(env.episode_metrics.distracted_triggers, 0)
        finally:
            env.close()

    def test_distracted_trigger_is_counted_once(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=13)
            human = env.humans[0]
            human.set_mode(HumanMode.FOLLOWING)

            env_control.apply_fuzzy_transition(
                env,
                human,
                idx=0,
                context="following",
                fuzzy_result={"dominant_state": "distracted"},
                fuzzy_inputs={
                    "following_time": 0.0,
                    "hhd": 0.0,
                    "hrd": 0.0,
                    "density": 0.0,
                    "angle": 0.0,
                },
                world_frame=SimpleNamespace(robot_xy=np.array([0.0, 0.0], dtype=np.float32)),
            )
            self.assertEqual(env.episode_metrics.distracted_triggers, 1)

            env.follow_phase = FOLLOW_PHASE_TRANSIT
            env_control._maybe_apply_fuzzy(
                env,
                human,
                idx=0,
                context="following",
                session_steps=0,
                world_frame=None,
            )
            self.assertEqual(env.episode_metrics.distracted_triggers, 1)
        finally:
            env.close()

    def test_impatient_trigger_is_counted_once(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=14)
            human = env.humans[0]
            human.set_mode(HumanMode.FOLLOWING)

            env_control.apply_fuzzy_transition(
                env,
                human,
                idx=0,
                context="following",
                fuzzy_result={"dominant_state": "impatient"},
                fuzzy_inputs={
                    "following_time": 0.0,
                    "hhd": 0.0,
                    "hrd": 0.0,
                    "density": 0.0,
                    "angle": 0.0,
                },
                world_frame=SimpleNamespace(robot_xy=np.array([0.0, 0.0], dtype=np.float32)),
            )
            self.assertEqual(env.episode_metrics.impatient_triggers, 1)

            env.follow_phase = FOLLOW_PHASE_TRANSIT
            env_control._maybe_apply_fuzzy(
                env,
                human,
                idx=0,
                context="following",
                session_steps=0,
                world_frame=None,
            )
            self.assertEqual(env.episode_metrics.impatient_triggers, 1)
        finally:
            env.close()

    def test_overwhelmed_trigger_is_counted_once(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=15)
            human = env.humans[0]
            human.set_mode(HumanMode.FOLLOWING)

            env_control.apply_fuzzy_transition(
                env,
                human,
                idx=0,
                context="following",
                fuzzy_result={"dominant_state": "overwhelmed"},
                fuzzy_inputs={
                    "following_time": 0.0,
                    "hhd": 0.0,
                    "hrd": 0.0,
                    "density": 0.0,
                    "angle": 0.0,
                },
                world_frame=SimpleNamespace(robot_xy=np.array([0.0, 0.0], dtype=np.float32)),
            )
            self.assertEqual(env.episode_metrics.overwhelmed_triggers, 1)

            env.follow_phase = FOLLOW_PHASE_TRANSIT
            env_control._maybe_apply_fuzzy(
                env,
                human,
                idx=0,
                context="following",
                session_steps=0,
                world_frame=None,
            )
            self.assertEqual(env.episode_metrics.overwhelmed_triggers, 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
