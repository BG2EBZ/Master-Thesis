import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from museum_env import env_control, env_flow
from museum_env.env import MuseumEnv
from museum_env.env_reporting import (
    HUMAN_SPEAKING_HALO_RGBA_OFF,
    HUMAN_SPEAKING_HALO_RGBA_ON,
    ROBOT_COLOR_NATURAL,
    ROBOT_COLOR_SAD,
    ROBOT_FOLLOWME_LABEL_GROUP,
    ROBOT_ANSWER_LABEL_GROUP,
    resolve_robot_visual_state,
)
from museum_env.env_state import (
    LISTEN_QUESTION_PHASE_ANSWER,
    LISTEN_QUESTION_PHASE_TURN_BACK,
    LISTEN_QUESTION_PHASE_TURN_TO_HUMAN,
    LISTEN_QUESTION_TIMING_MID_RANDOM,
    LISTEN_QUESTION_TIMING_POST_WAIT,
    StepEvents,
)
from museum_env.human import (
    DEFAULT_SIM_TIMESTEP_SECONDS,
    DISTRACTED_BEHAVIOR_CONVERSATION,
    DISTRACTED_BEHAVIOR_STOP_AND_GO_FOLLOWING,
    DISTRACTED_CONVERSATION_STOP_DISTANCE,
    DISTRACTED_SOURCE_FOLLOWING,
    DISTRACTED_SOURCE_LISTENING,
    DISTRACTED_SPEED_SCALE,
    DISTRACTED_TARGET_DISTANCE_MIN,
    HUMAN_YAW_RATE_GAIN,
    WALL_REPULSION_DISTANCE_METERS,
    WALL_REPULSION_GAIN,
    Human,
    HumanMode,
    HumanProfile,
)
from museum_env.map_layouts import AxisAlignedRect, MapLayout
from museum_env.spatial_utils import raycast_hit_distance, wrap_to_pi


class MuseumEnvRefactorTests(unittest.TestCase):
    def _make_env(self, **kwargs):
        defaults = {
            "render_mode": None,
            "enable_event_logs": False,
        }
        defaults.update(kwargs)
        return MuseumEnv(**defaults)

    def _run_until(self, env, predicate, max_steps):
        last_info = None
        for step in range(max_steps):
            _, _, terminated, truncated, info = env.step(None)
            last_info = info
            if predicate(info):
                return step, info
            if terminated or truncated:
                return step, info
        self.fail(f"condition not met within {max_steps} steps; last_info={last_info}")

    def _make_pose_data(self, pose):
        return SimpleNamespace(qpos=np.array(pose, dtype=np.float32))

    def _prime_listening_question(
        self,
        env,
        *,
        timing_mode,
        trigger_step=None,
        is_final=False,
    ):
        env.listening_state.enter_wait(is_final)
        env.robot.listen_mode = True
        env.listening_state.session_has_question = True
        env.listening_state.question_timing_mode = timing_mode
        env.listening_state.question_trigger_step = trigger_step
        env.listening_state.question_fired = False
        env.listening_state.question_human_idx = None

    def _set_robot_and_human_poses(self, env, robot_pose, human_poses):
        robot_origin_xy = np.array(env.model.body("robot").pos[:2], dtype=np.float32)
        robot_pose = np.array(robot_pose, dtype=np.float32)
        env.data.qpos[0:2] = robot_pose[:2] - robot_origin_xy
        env.data.qpos[2] = robot_pose[2]
        env.data.qvel[0:3] = 0.0
        for human, pose in zip(env.humans, human_poses):
            env.data.qpos[human.qpos_idx : human.qpos_idx + 3] = np.array(
                pose,
                dtype=np.float32,
            )
            env.data.qvel[human.qpos_idx : human.qpos_idx + 3] = 0.0
        mujoco.mj_forward(env.model, env.data)

    def _invalidate_observation_cache(self, env):
        env.runtime_cache.observations = None
        env.runtime_cache.sample_age_steps = 0

    def test_reset_contract_and_default_profile(self):
        env = self._make_env(n_humans=5)
        try:
            obs, info = env.reset(seed=10)
            self.assertEqual(obs.shape, (4,))
            self.assertEqual(info, {})
            self.assertEqual(len(env.humans), 5)
            self.assertEqual(env.humans[0].profile, HumanProfile.NEURODIVERGENT)
            for human in env.humans[1:]:
                self.assertEqual(human.profile, HumanProfile.NORMAL)
            self.assertEqual(env.listening_state.phase, "idle")
        finally:
            env.close()

    def test_step_info_uses_compact_contract(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=11)
            _, _, _, _, info = env.step(None)
            self.assertEqual(
                sorted(info.keys()),
                ["crowd", "episode", "events", "phase", "robot"],
            )
            self.assertNotIn("metrics", info)
            self.assertNotIn("status", info)
            self.assertEqual(
                sorted(info["episode"].keys()),
                [
                    "step",
                    "terminated_reason",
                ],
            )
            self.assertEqual(
                sorted(info["phase"].keys()),
                ["follow", "listen"],
            )
            self.assertEqual(
                sorted(info["robot"].keys()),
                [
                    "action",
                    "callback_phase",
                    "dist_to_goal",
                    "emotion",
                    "goal_xy",
                    "mode",
                    "pose_xy",
                    "speaker_active",
                    "yaw",
                ],
            )
            self.assertEqual(
                sorted(info["crowd"].keys()),
                [
                    "distracted_indices",
                    "goal_xy",
                    "human_robot_distance",
                    "modes",
                    "pose_xy",
                    "profiles",
                    "reached_goal_indices",
                ],
            )
        finally:
            env.close()

    def test_step_info_exposes_human_robot_distance_consistent_with_pose_xy(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=13)
            _, _, _, _, info = env.step(None)

            human_xy = np.asarray(info["crowd"]["pose_xy"], dtype=np.float32)
            robot_xy = np.asarray(info["robot"]["pose_xy"], dtype=np.float32)
            expected = np.linalg.norm(human_xy - robot_xy[None, :], axis=1)

            np.testing.assert_allclose(
                info["crowd"]["human_robot_distance"],
                expected,
                atol=1e-6,
            )
        finally:
            env.close()

    def test_callback_visual_state_uses_please_rejoin_label(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=113)
            visual_state = resolve_robot_visual_state(robot=env.robot, callback_visual_active=True)
            self.assertTrue(visual_state.show_follow_me)
            self.assertEqual(visual_state.text_label, "please rejoin")
        finally:
            env.close()

    def test_first_listen_flow_reaches_intro_and_wait(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=12)
            _, info = self._run_until(env, lambda info: info["events"]["entered_listen"], 6000)
            self.assertEqual(info["phase"]["listen"], "intro")

            _, info = self._run_until(env, lambda info: info["events"]["started_listen_wait"], 2500)
            self.assertEqual(info["phase"]["listen"], "wait")
            self.assertTrue(info["robot"]["speaker_active"])
        finally:
            env.close()

    def test_listening_wait_shortens_once_per_person_and_accumulates(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=31)
            env.listen_wait_steps = 10
            env.listen_distance_shorten_steps = 2
            env.listening_state.enter_wait(False)
            env.robot.listen_mode = True

            env_flow.maybe_shorten_listening_wait(
                env,
                SimpleNamespace(
                    observations=SimpleNamespace(
                        human_robot_distance=np.array([2.5, 1.5], dtype=np.float32),
                    )
                )
            )
            self.assertEqual(env.listening_state.wait_target_steps, 8)
            self.assertEqual(env.listening_state.distance_shorten_triggered_indices, {0})

            env_flow.maybe_shorten_listening_wait(
                env,
                SimpleNamespace(
                    observations=SimpleNamespace(
                        human_robot_distance=np.array([2.8, 1.4], dtype=np.float32),
                    )
                )
            )
            self.assertEqual(env.listening_state.wait_target_steps, 8)
            self.assertEqual(env.listening_state.distance_shorten_triggered_indices, {0})

            env_flow.maybe_shorten_listening_wait(
                env,
                SimpleNamespace(
                    observations=SimpleNamespace(
                        human_robot_distance=np.array([2.8, 2.3], dtype=np.float32),
                    )
                )
            )
            self.assertEqual(env.listening_state.wait_target_steps, 6)
            self.assertEqual(env.listening_state.distance_shorten_triggered_indices, {0, 1})
        finally:
            env.close()

    def test_listening_wait_shorten_tracking_resets_each_session(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=32)
            env.listen_wait_steps = 10
            env.listen_distance_shorten_steps = 2
            env.listening_state.enter_wait(False)
            env.robot.listen_mode = True

            env_flow.maybe_shorten_listening_wait(
                env,
                SimpleNamespace(
                    observations=SimpleNamespace(
                        human_robot_distance=np.array([2.5], dtype=np.float32),
                    )
                )
            )
            self.assertEqual(env.listening_state.wait_target_steps, 8)
            self.assertEqual(env.listening_state.distance_shorten_triggered_indices, {0})

            env.listening_state.enter_idle()
            env.listening_state.enter_wait(False)
            env.robot.listen_mode = True

            self.assertEqual(env.listening_state.wait_target_steps, 0)
            self.assertEqual(env.listening_state.distance_shorten_triggered_indices, set())

            env_flow.maybe_shorten_listening_wait(
                env,
                SimpleNamespace(
                    observations=SimpleNamespace(
                        human_robot_distance=np.array([2.6], dtype=np.float32),
                    )
                )
            )
            self.assertEqual(env.listening_state.wait_target_steps, 8)
            self.assertEqual(env.listening_state.distance_shorten_triggered_indices, {0})
        finally:
            env.close()

    def test_listening_wait_shortening_can_skip_post_wait_question(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=33)
            env.listen_wait_steps = 6
            env.listen_distance_shorten_steps = 6
            env.listen_question_pause_steps = 1
            env.humans[0].set_profile(HumanProfile.NORMAL)
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((3.0, 0.0, 0.0),),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_POST_WAIT,
                is_final=False,
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, info = env.step(None)

            self.assertFalse(info["events"]["question_started"])
            self.assertTrue(info["events"]["completed_listen_wait"])
            self.assertEqual(info["phase"]["listen"], "idle")
            self.assertTrue(env.post_explanation_state.active)
        finally:
            env.close()

    def test_listening_question_completion_finishes_shortened_wait(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=34)
            env.listen_wait_steps = 6
            env.listen_distance_shorten_steps = 3
            env.listen_question_pause_steps = 1
            env.humans[0].set_profile(HumanProfile.NORMAL)
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((3.0, 0.0, 0.0),),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
            )
            self._invalidate_observation_cache(env)

            env.step(None)
            self.assertEqual(env.listening_state.wait_target_steps, 3)

            env.step(None)
            _, _, _, _, info = env.step(None)
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertTrue(info["events"]["question_started"])

            _, info = self._run_until(env, lambda _info: _info["events"]["question_completed"], 200)
            self.assertTrue(info["events"]["question_completed"])
            self.assertTrue(info["events"]["completed_listen_wait"])
            self.assertEqual(info["phase"]["listen"], "idle")
            self.assertTrue(env.post_explanation_state.active)
        finally:
            env.close()

    def test_listening_questions_pause_and_highlight_single_human(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=21)
            env.listen_wait_steps = 6
            env.listen_question_pause_steps = 2
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
            )
            env.listening_state.counter = (env.listen_wait_steps // 2) - 1

            _, _, _, _, info = env.step(None)

            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertTrue(info["events"]["question_started"])
            self.assertFalse(info["robot"]["speaker_active"])
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_TURN_TO_HUMAN)
            active_idx = env.listening_state.question_human_idx
            self.assertIn(active_idx, (0, 1))
            self.assertEqual(
                [human.speaking_active for human in env.humans],
                [idx == active_idx for idx in range(len(env.humans))],
            )
            np.testing.assert_allclose(
                env.model.geom_rgba[env.all_human_speaking_halo_geom_ids[active_idx]],
                HUMAN_SPEAKING_HALO_RGBA_ON,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                env.model.geom_rgba[env.all_human_speaking_halo_geom_ids[1 - active_idx]],
                HUMAN_SPEAKING_HALO_RGBA_OFF,
                atol=1e-6,
            )
        finally:
            env.close()

    def test_listening_questions_pause_freezes_counter_and_resumes(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=22)
            env.listen_wait_steps = 6
            env.listen_question_pause_steps = 1
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.0, 0.0, 0.0),),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
            )
            env.listening_state.counter = (env.listen_wait_steps // 2) - 1

            env.step(None)
            paused_counter = env.listening_state.paused_counter

            _, info = self._run_until(
                env,
                lambda _info: env.listening_state.question_phase == LISTEN_QUESTION_PHASE_ANSWER,
                200,
            )
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_ANSWER)
            self.assertFalse(info["events"]["question_completed"])
            self.assertEqual(env.listening_state.paused_counter, paused_counter)
            self.assertEqual(env.listening_state.question_answer_steps_remaining, 1)
            self.assertFalse(env.humans[0].speaking_active)
            self.assertTrue(info["robot"]["speaker_active"])
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_ANSWER_LABEL_GROUP], 1)

            _, info = self._run_until(
                env,
                lambda _info: env.listening_state.question_phase == LISTEN_QUESTION_PHASE_TURN_BACK,
                200,
            )
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_TURN_BACK)
            self.assertFalse(info["events"]["question_completed"])
            self.assertFalse(info["robot"]["speaker_active"])

            _, info = self._run_until(env, lambda _info: _info["events"]["question_completed"], 200)
            self.assertEqual(info["phase"]["listen"], "wait")
            self.assertTrue(info["events"]["question_completed"])
            self.assertEqual(env.listening_state.counter, paused_counter)
            self.assertEqual(env.listening_state.question_answer_steps_remaining, 0)
            self.assertFalse(env.humans[0].speaking_active)
            self.assertTrue(info["robot"]["speaker_active"])
            self.assertIsNone(env.listening_state.question_human_idx)
        finally:
            env.close()

    def test_listening_question_plan_stays_disabled_when_probability_zero(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=23)
            env.listening_state.enter_wait(False)
            env.robot.listen_mode = True
            env.listen_wait_steps = 6
            env.listen_distance_shorten_steps = 0
            env.listen_question_probability = 0.0
            env.listen_question_after_explanation_probability = 0.0
            env.listen_question_pause_steps = 2
            env_flow.prepare_listening_question_plan(env)

            _, _, _, _, info = env.step(None)
            self.assertEqual(info["phase"]["listen"], "wait")
            self.assertFalse(info["events"]["question_started"])
            self.assertFalse(env.listening_state.session_has_question)
            self.assertIsNone(env.listening_state.question_timing_mode)
            self.assertTrue(env.listening_state.question_fired)

            env.listen_question_probability = 1.0
            _, _, _, _, info = env.step(None)
            self.assertEqual(info["phase"]["listen"], "wait")
            self.assertFalse(info["events"]["question_started"])
            self.assertFalse(env.listening_state.session_has_question)
        finally:
            env.close()

    def test_listening_question_mid_random_plan_is_seed_stable(self):
        env1 = self._make_env(n_humans=1)
        env2 = self._make_env(n_humans=1)
        try:
            for env in (env1, env2):
                env.reset(seed=24)
                env.listening_state.enter_wait(False)
                env.robot.listen_mode = True
                env.listen_wait_steps = 10
                env.listen_question_probability = 1.0
                env.listen_question_after_explanation_probability = 0.0
                env_flow.prepare_listening_question_plan(env)

            self.assertEqual(env1.listening_state.question_timing_mode, LISTEN_QUESTION_TIMING_MID_RANDOM)
            self.assertEqual(env2.listening_state.question_timing_mode, LISTEN_QUESTION_TIMING_MID_RANDOM)
            self.assertEqual(
                env1.listening_state.question_trigger_step,
                env2.listening_state.question_trigger_step,
            )
            self.assertGreaterEqual(env1.listening_state.question_trigger_step, env1.listen_wait_steps // 2)
            self.assertLess(env1.listening_state.question_trigger_step, env1.listen_wait_steps)
        finally:
            env1.close()
            env2.close()

    def test_listening_questions_ignore_non_listening_humans(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=24)
            env.listen_wait_steps = 6
            env.listen_question_pause_steps = 2
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
            )
            env.listening_state.counter = (env.listen_wait_steps // 2) - 1
            env.humans[0].start_impatient(recovery_mode=HumanMode.LISTENING)
            env.humans[0].impatient_duration = 100

            _, _, _, _, info = env.step(None)

            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_human_idx, 1)
            self.assertFalse(env.humans[0].speaking_active)
            self.assertTrue(env.humans[1].speaking_active)
        finally:
            env.close()

    def test_listening_question_turn_rate_is_capped_while_turning_to_human(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=25)
            env.listen_wait_steps = 6
            env.listen_question_pause_steps = 1
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 1.0, 0.0),),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
            )
            env.listening_state.counter = (env.listen_wait_steps // 2) - 1

            env.step(None)
            _, _, _, _, info = env.step(None)

            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_TURN_TO_HUMAN)
            self.assertAlmostEqual(abs(info["robot"]["action"]["yaw_rate"]), 1.0, places=6)
            self.assertLessEqual(abs(info["robot"]["action"]["yaw_rate"]), 1.0 + 1e-6)
            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
        finally:
            env.close()

    def test_listening_question_human_speaks_for_configured_duration(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=26)
            env.listen_wait_steps = 6
            env.listen_question_pause_steps = 2
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.0, 0.0, 0.0),),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
            )
            env.listening_state.counter = (env.listen_wait_steps // 2) - 1

            env.step(None)

            _, _, _, _, info = env.step(None)
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_TURN_TO_HUMAN)
            self.assertEqual(env.listening_state.question_ask_steps_remaining, 1)
            self.assertTrue(env.humans[0].speaking_active)
            self.assertFalse(info["robot"]["speaker_active"])

            _, _, _, _, info = env.step(None)
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_ANSWER)
            self.assertEqual(env.listening_state.question_ask_steps_remaining, 0)
            self.assertFalse(env.humans[0].speaking_active)
            self.assertTrue(info["robot"]["speaker_active"])
        finally:
            env.close()

    def test_listening_question_freezes_listener_robot_yaw_during_turn(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=27)
            env.listen_wait_steps = 6
            env.listen_question_pause_steps = 1
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
            )
            env.listening_state.counter = (env.listen_wait_steps // 2) - 1

            env.step(None)
            asking_idx = int(env.listening_state.question_human_idx)
            listener_idx = 1 - asking_idx

            for _ in range(10):
                env.step(None)

            captured = {}
            original_step = env.humans[listener_idx].step

            def capture_step(model, data, ctx):
                captured["robot_yaw"] = float(ctx["robot_yaw"])
                captured["live_robot_yaw"] = float(ctx["robot_pose"][2])
                return original_step(model, data, ctx)

            with patch.object(env.humans[listener_idx], "step", side_effect=capture_step):
                env.step(None)

            self.assertIn("robot_yaw", captured)
            self.assertIn("live_robot_yaw", captured)
            self.assertAlmostEqual(
                captured["robot_yaw"],
                float(env.listening_state.question_return_yaw),
                places=6,
            )
            self.assertGreater(
                abs(captured["live_robot_yaw"] - captured["robot_yaw"]),
                1e-4,
            )
        finally:
            env.close()

    def test_final_listening_questions_resume_final_session(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=28)
            env.listen_wait_steps = 6
            env.listen_question_pause_steps = 1
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.0, 0.0, 0.0),),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
                is_final=True,
            )
            env.listening_state.counter = (env.listen_wait_steps // 2) - 1

            _, _, _, _, info = env.step(None)
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertTrue(env.listening_state.paused_is_final)

            _, info = self._run_until(
                env,
                lambda _info: env.listening_state.question_phase == LISTEN_QUESTION_PHASE_ANSWER,
                200,
            )
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_ANSWER)

            _, info = self._run_until(
                env,
                lambda _info: env.listening_state.question_phase == LISTEN_QUESTION_PHASE_TURN_BACK,
                200,
            )
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_TURN_BACK)

            _, info = self._run_until(env, lambda _info: _info["events"]["question_completed"], 200)
            self.assertEqual(info["phase"]["listen"], "wait")
            self.assertTrue(info["events"]["question_completed"])
            self.assertTrue(env.listening_state.is_final)
        finally:
            env.close()

    def test_listening_question_post_wait_pauses_before_completion_and_then_finishes(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=29)
            env.listen_wait_steps = 6
            env.listen_question_pause_steps = 1
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.0, 0.0, 0.0),),
            )
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_POST_WAIT,
                is_final=False,
            )
            env.listening_state.counter = env.listen_wait_steps - 1

            _, _, _, _, info = env.step(None)
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertTrue(info["events"]["question_started"])
            self.assertFalse(info["events"]["completed_listen_wait"])
            self.assertEqual(env.listening_state.question_human_idx, 0)

            _, info = self._run_until(
                env,
                lambda _info: env.listening_state.question_phase == LISTEN_QUESTION_PHASE_ANSWER,
                200,
            )
            self.assertEqual(info["phase"]["listen"], "paused")
            self.assertEqual(env.listening_state.question_phase, LISTEN_QUESTION_PHASE_ANSWER)
            self.assertTrue(info["robot"]["speaker_active"])
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_ANSWER_LABEL_GROUP], 1)
            self.assertFalse(env.humans[0].speaking_active)

            _, info = self._run_until(env, lambda _info: _info["events"]["question_completed"], 200)
            self.assertTrue(info["events"]["question_completed"])
            self.assertTrue(info["events"]["completed_listen_wait"])
            self.assertEqual(info["phase"]["listen"], "idle")
            self.assertTrue(env.post_explanation_state.active)
            self.assertFalse(env.humans[0].speaking_active)
        finally:
            env.close()

    def test_listening_question_probability_split_selects_expected_branch(self):
        env_post = self._make_env(n_humans=1)
        env_mid = self._make_env(n_humans=1)
        try:
            env_post.reset(seed=30)
            env_post.listening_state.enter_wait(False)
            env_post.robot.listen_mode = True
            env_post.listen_wait_steps = 8
            env_post.listen_question_probability = 1.0
            env_post.listen_question_after_explanation_probability = 1.0
            env_flow.prepare_listening_question_plan(env_post)

            env_mid.reset(seed=30)
            env_mid.listening_state.enter_wait(False)
            env_mid.robot.listen_mode = True
            env_mid.listen_wait_steps = 8
            env_mid.listen_question_probability = 1.0
            env_mid.listen_question_after_explanation_probability = 0.0
            env_flow.prepare_listening_question_plan(env_mid)

            self.assertEqual(
                env_post.listening_state.question_timing_mode,
                LISTEN_QUESTION_TIMING_POST_WAIT,
            )
            self.assertIsNone(env_post.listening_state.question_trigger_step)
            self.assertEqual(
                env_mid.listening_state.question_timing_mode,
                LISTEN_QUESTION_TIMING_MID_RANDOM,
            )
            self.assertIsNotNone(env_mid.listening_state.question_trigger_step)
        finally:
            env_post.close()
            env_mid.close()

    def test_listening_question_cancels_when_no_listening_candidate_at_trigger(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=28)
            env.listen_wait_steps = 6
            env.listen_distance_shorten_steps = 0
            self._prime_listening_question(
                env,
                timing_mode=LISTEN_QUESTION_TIMING_MID_RANDOM,
                trigger_step=env.listen_wait_steps // 2,
            )
            env.listening_state.counter = (env.listen_wait_steps // 2) - 1
            env.humans[0].start_impatient(recovery_mode=HumanMode.LISTENING)
            env.humans[0].impatient_duration = 100

            _, _, _, _, info = env.step(None)
            self.assertEqual(info["phase"]["listen"], "wait")
            self.assertFalse(info["events"]["question_started"])
            self.assertTrue(env.listening_state.question_fired)
            self.assertIsNone(env.listening_state.question_human_idx)
        finally:
            env.close()

    def test_post_explanation_resumes_transit_follow_after_wait_completion(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=13)
            env.listen_intro_delay_steps = 2
            env.listen_wait_steps = 3
            env.robot.v_max = 3.0

            _, info = self._run_until(
                env,
                lambda info: info["events"]["completed_listen_wait"],
                22000,
            )
            self.assertEqual(info["phase"]["listen"], "idle")

            _, info = self._run_until(
                env,
                lambda info: info["phase"]["follow"] == "transit_follow",
                8000,
            )
            self.assertEqual(info["phase"]["listen"], "idle")
        finally:
            env.close()

    def test_listening_distracted_does_not_pause_or_trigger_callback(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=14)
            human = env.humans[0]
            env.listening_state.enter_wait(False)
            env.robot.listen_mode = True
            human.set_mode(HumanMode.DISTRACTED)
            human.distracted_source = DISTRACTED_SOURCE_LISTENING
            human.distracted_recovery_mode = HumanMode.LISTENING

            _, _, _, _, info = env.step(None)

            self.assertEqual(env.listening_state.phase, "wait")
            self.assertTrue(env.robot.listen_mode)
            self.assertFalse(info["events"]["callback_triggered"])
            self.assertFalse(env.robot.callback_active)
            self.assertIsNone(info["robot"]["callback_phase"])
        finally:
            env.close()

    def test_distracted_does_not_turn_robot_blue_without_callback(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=15)
            human = env.humans[0]
            env.follow_phase = "transit_follow"
            human.set_mode(HumanMode.DISTRACTED)
            human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
            human.distracted_recovery_mode = HumanMode.FOLLOWING
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((-0.5, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, info = env.step(None)

            self.assertEqual(info["robot"]["emotion"], "natural")
            self.assertFalse(info["events"]["callback_triggered"])
            self.assertFalse(info["events"]["callback_completed"])
            self.assertFalse(info["events"]["callback_success"])
            self.assertFalse(info["events"]["callback_ignored"])
            self.assertFalse(env.robot.callback_active)
            self.assertIsNone(info["robot"]["callback_phase"])
            np.testing.assert_allclose(
                env.model.geom_rgba[env.robot_base_geom_id],
                ROBOT_COLOR_NATURAL,
                atol=1e-6,
            )
        finally:
            env.close()

    def test_active_callback_state_drives_callback_mode_and_visual_label(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=16)
            env.robot.callback_active = True
            env.robot.callback_phase = "cue"
            env.robot.callback_target_idx = 0
            env.robot.callback_target_xy = np.array(env.humans[0].get_pose(env.data)[:2], dtype=np.float32)
            env.robot.callback_cue_total_steps = 3
            env.robot.callback_cue_elapsed_steps = 0
            env.robot.mode = "callback"

            _, _, _, _, info = env.step(None)

            self.assertTrue(env.robot.callback_active)
            self.assertEqual(info["robot"]["emotion"], "sad")
            self.assertEqual(env.robot.callback_phase, "cue")
            self.assertEqual(info["robot"]["mode"], "callback")
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP], 1)
            np.testing.assert_allclose(
                env.model.geom_rgba[env.robot_base_geom_id],
                ROBOT_COLOR_SAD,
                atol=1e-6,
            )
        finally:
            env.close()

    def test_step_enables_human_speaking_halos_for_conversation(self):
        env = self._make_env(n_humans=2, callback_trigger_distance_meters=10.0)
        try:
            env.reset(seed=18)
            first_human = env.humans[0]
            second_human = env.humans[1]

            env.data.qpos[first_human.qpos_idx : first_human.qpos_idx + 3] = np.array(
                [5.0, 5.8, 0.0],
                dtype=np.float32,
            )
            env.data.qvel[first_human.qpos_idx : first_human.qpos_idx + 3] = 0.0
            env.data.qpos[second_human.qpos_idx : second_human.qpos_idx + 3] = np.array(
                [5.6, 5.8, np.pi],
                dtype=np.float32,
            )
            env.data.qvel[second_human.qpos_idx : second_human.qpos_idx + 3] = 0.0
            mujoco.mj_forward(env.model, env.data)

            for human in env.humans:
                human.set_mode(HumanMode.DISTRACTED)
                human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

            env.step(None)

            np.testing.assert_allclose(
                env.model.geom_rgba[env.all_human_speaking_halo_geom_ids[0]],
                HUMAN_SPEAKING_HALO_RGBA_ON,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                env.model.geom_rgba[env.all_human_speaking_halo_geom_ids[1]],
                HUMAN_SPEAKING_HALO_RGBA_ON,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                env.model.geom_rgba[env.all_human_speaking_halo_geom_ids[2]],
                HUMAN_SPEAKING_HALO_RGBA_OFF,
                atol=1e-6,
            )
        finally:
            env.close()

    def test_sync_human_visual_state_clears_halo_after_exit_and_reset(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=19)
            human = env.humans[0]
            halo_geom_id = env.all_human_speaking_halo_geom_ids[0]
            inactive_halo_geom_id = env.all_human_speaking_halo_geom_ids[1]

            human.set_mode(HumanMode.DISTRACTED)
            human.speaking_active = True
            env._sync_human_visual_state()
            np.testing.assert_allclose(env.model.geom_rgba[halo_geom_id], HUMAN_SPEAKING_HALO_RGBA_ON, atol=1e-6)

            human.set_mode(HumanMode.FOLLOWING)
            env._sync_human_visual_state()
            np.testing.assert_allclose(env.model.geom_rgba[halo_geom_id], HUMAN_SPEAKING_HALO_RGBA_OFF, atol=1e-6)

            env.all_humans[1].speaking_active = True
            env._sync_human_visual_state()
            np.testing.assert_allclose(
                env.model.geom_rgba[inactive_halo_geom_id],
                HUMAN_SPEAKING_HALO_RGBA_OFF,
                atol=1e-6,
            )

            human.set_mode(HumanMode.DISTRACTED)
            human.speaking_active = True
            env._sync_human_visual_state()
            env.reset(seed=20)
            np.testing.assert_allclose(env.model.geom_rgba[halo_geom_id], HUMAN_SPEAKING_HALO_RGBA_OFF, atol=1e-6)
        finally:
            env.close()

    def test_fuzzy_evaluates_once_per_observation_refresh_cycle(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=17)
            env.follow_phase = "transit_follow"
            env.observation_update_period_steps = 2
            engaged_result = {
                "overwhelmed": 0.1,
                "distracted": 0.1,
                "impatient": 0.1,
                "engaged": 0.9,
                "dominant_state": "engaged",
                "dominant_value": 0.9,
            }

            with patch.object(env.following_fuzzy_engine, "compute", return_value=engaged_result) as mock_compute:
                env.step(None)
                self.assertEqual(mock_compute.call_count, 1)
                self.assertEqual(
                    mock_compute.call_args.kwargs["profile"],
                    HumanProfile.NEURODIVERGENT,
                )
                self.assertIn("angle", mock_compute.call_args.kwargs)
                self.assertGreaterEqual(mock_compute.call_args.kwargs["angle"], -180.0)
                self.assertLessEqual(mock_compute.call_args.kwargs["angle"], 180.0)

                env.step(None)
                self.assertEqual(mock_compute.call_count, 1)

                env.step(None)
                self.assertEqual(mock_compute.call_count, 2)
                self.assertEqual(env.fuzzy_debug[0].context, "following")
        finally:
            env.close()

    def test_human_states_module_import_and_compute_accepts_angle(self):
        module = importlib.import_module("museum_env.fuzzy.human_states")
        result = module.compute(3.1, 0.56, 1.05, 5.0, 0.0, context="following")

        self.assertIn("dominant_state", result)
        self.assertIn("engaged", result)

    def test_compute_human_fuzzy_debug_reports_robot_relative_angle_degrees(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=19)
            cases = [
                ((1.0, 0.0, 0.0), 0.0),
                ((0.0, -1.0, 0.0), -90.0),
                ((0.0, 1.0, 0.0), 90.0),
                ((-1.0, 0.0, 0.0), 180.0),
            ]

            for human_pose, expected_angle in cases:
                with self.subTest(human_pose=human_pose):
                    self._set_robot_and_human_poses(
                        env,
                        robot_pose=(0.0, 0.0, 0.0),
                        human_poses=[human_pose],
                    )
                    self._invalidate_observation_cache(env)
                    world_frame = env._build_world_frame(force=True)
                    fuzzy_debug = env_control.compute_human_fuzzy_debug(
                        env,
                        idx=0,
                        context="following",
                        session_steps=0,
                        world_frame=world_frame,
                    )
                    measured_angle = float(fuzzy_debug["inputs"]["angle"])

                    if abs(expected_angle) == 180.0:
                        self.assertAlmostEqual(abs(measured_angle), 180.0, places=4)
                    else:
                        self.assertAlmostEqual(measured_angle, expected_angle, places=4)
        finally:
            env.close()

    def test_following_distracted_moves_toward_distracted_target_with_speed_limit(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
        human.distracted_target_xy = np.array([6.0, 6.0], dtype=np.float32)
        human.distracted_target_yaw = float(np.pi / 2.0)

        pose = (6.0, 5.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[6.0, 5.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        action = human.step(None, data, ctx)

        expected_velocity = np.array(
            [0.0, DISTRACTED_SPEED_SCALE * human.max_speed],
            dtype=np.float32,
        )
        np.testing.assert_allclose(action[:2], expected_velocity, atol=1e-6)
        self.assertLessEqual(
            float(np.linalg.norm(action[:2])),
            DISTRACTED_SPEED_SCALE * human.max_speed + 1e-6,
        )
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * (np.pi / 2.0), places=5)

    def test_following_distracted_preserves_forward_progress_when_hr_force_reverses_velocity(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
        human.distracted_target_xy = np.array([6.0, 6.0], dtype=np.float32)
        human.distracted_target_yaw = float(np.pi / 2.0)

        pose = (6.0, 5.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (6.0, 1.0, 0.0),
            "robot_xy": np.array([6.0, 1.0], dtype=np.float32),
            "human_xy": np.array([[6.0, 5.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        action = human.step(None, data, ctx)

        to_target_xy = human.distracted_target_xy - np.array(pose[:2], dtype=np.float32)
        self.assertGreater(float(np.dot(action[:2], to_target_xy)), 0.0)
        np.testing.assert_allclose(
            action[:2],
            np.array([0.0, DISTRACTED_SPEED_SCALE * human.max_speed], dtype=np.float32),
            atol=1e-6,
        )
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * (np.pi / 2.0), places=5)

    def test_following_distracted_orients_toward_nearest_exhibit_while_moving(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (6.2, 6.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 6.0, 0.0),
            "robot_xy": np.array([5.0, 6.0], dtype=np.float32),
            "human_xy": np.array([[6.2, 6.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        action = human.step(None, data, ctx)

        expected_exhibit = np.array([9.786, 6.65], dtype=np.float32)
        exhibit_dir = expected_exhibit - np.array(pose[:2], dtype=np.float32)
        expected_focus = (
            expected_exhibit
            - DISTRACTED_TARGET_DISTANCE_MIN * (exhibit_dir / np.linalg.norm(exhibit_dir))
        )
        expected_yaw = float(np.arctan2(expected_exhibit[1] - pose[1], expected_exhibit[0] - pose[0]))
        expected_yaw_rate = HUMAN_YAW_RATE_GAIN * expected_yaw
        expected_velocity = (
            DISTRACTED_SPEED_SCALE
            * human.max_speed
            * (expected_focus - np.array(pose[:2], dtype=np.float32))
            / np.linalg.norm(expected_focus - np.array(pose[:2], dtype=np.float32))
        )

        np.testing.assert_allclose(human.distracted_target_xy, expected_focus, atol=1e-6)
        np.testing.assert_allclose(action[:2], expected_velocity, atol=1e-5)
        np.testing.assert_allclose(action[2], expected_yaw_rate, atol=1e-5)
        self.assertGreater(float(action[0]), 0.0)

    def test_following_distracted_falls_back_to_synthetic_focus_target(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (6.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 0.0, 0.0),
            "robot_xy": np.array([5.0, 0.0], dtype=np.float32),
            "human_xy": np.array([[6.0, 0.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        action = human.step(None, data, ctx)

        focus_distance = float(
            np.linalg.norm(human.distracted_target_xy - np.array(pose[:2], dtype=np.float32))
        )
        self.assertAlmostEqual(focus_distance, 1.0, places=6)
        expected_velocity = (
            DISTRACTED_SPEED_SCALE
            * human.max_speed
            * (human.distracted_target_xy - np.array(pose[:2], dtype=np.float32))
            / np.linalg.norm(human.distracted_target_xy - np.array(pose[:2], dtype=np.float32))
        )
        np.testing.assert_allclose(action[:2], expected_velocity, atol=1e-6)
        self.assertTrue(np.isfinite(action).all())

    def test_nd_following_distracted_stops_immediately_before_moving_to_target(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_profile(HumanProfile.NEURODIVERGENT)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (6.2, 6.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 6.0, 0.0),
            "robot_xy": np.array([5.0, 6.0], dtype=np.float32),
            "human_xy": np.array([[6.2, 6.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        first_action = human.step(None, data, ctx)

        expected_exhibit = np.array([9.786, 6.65], dtype=np.float32)
        exhibit_dir = expected_exhibit - np.array(pose[:2], dtype=np.float32)
        expected_focus = (
            expected_exhibit
            - DISTRACTED_TARGET_DISTANCE_MIN * (exhibit_dir / np.linalg.norm(exhibit_dir))
        )

        np.testing.assert_allclose(first_action, np.zeros(3, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(human.distracted_target_xy, expected_focus, atol=1e-6)

        second_action = human.step(None, data, ctx)
        self.assertGreater(float(np.linalg.norm(second_action[:2])), 0.0)

    def test_nd_following_distracted_uses_stop_and_go_when_no_focus_target_exists(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_profile(HumanProfile.NEURODIVERGENT)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
        human.nd_distracted_stop_and_go_stop_steps = 1
        human.nd_distracted_stop_and_go_move_steps = 1

        pose = (1.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (0.0, 0.0, 0.0),
            "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
            "human_xy": np.array([[1.0, 0.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        with (
            patch("numpy.random.uniform", return_value=45.0),
            patch("numpy.random.rand", return_value=1.0),
        ):
            first_action = human.step(None, data, ctx)

        np.testing.assert_allclose(first_action, np.zeros(3, dtype=np.float32), atol=1e-6)
        self.assertIsNone(human.distracted_target_xy)
        self.assertEqual(human.distracted_behavior_kind, DISTRACTED_BEHAVIOR_STOP_AND_GO_FOLLOWING)
        self.assertTrue(human.speaking_active)

        second_action = human.step(None, data, ctx)
        np.testing.assert_allclose(second_action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)

        third_action = human.step(None, data, ctx)
        self.assertIsNone(human.distracted_target_xy)
        self.assertTrue(human.speaking_active)
        self.assertGreater(float(np.linalg.norm(third_action[:2])), 0.0)
        self.assertLess(float(third_action[0]), 0.0)

    def test_following_distracted_stops_at_target_then_recovers_after_hold(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
        human.distracted_target_xy = np.array([1.1, 1.0], dtype=np.float32)
        human.distracted_target_yaw = float(np.pi / 2.0)
        human.distracted_stop_duration = 3
        human.distracted_recovery_mode = HumanMode.FOLLOWING

        pose = (1.0, 1.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[1.0, 1.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        first_action = human.step(None, data, ctx)
        self.assertTrue(human.distracted_stop_reached)
        np.testing.assert_allclose(first_action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertEqual(human.mode, HumanMode.DISTRACTED)

        second_action = human.step(None, data, ctx)
        np.testing.assert_allclose(second_action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertEqual(human.mode, HumanMode.DISTRACTED)

        third_action = human.step(None, data, ctx)

        np.testing.assert_allclose(third_action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertEqual(human.mode, HumanMode.FOLLOWING)

    def test_following_distracted_force_recovers_after_max_duration(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
        human.distracted_target_xy = np.array([3.0, 1.0], dtype=np.float32)
        human.distracted_target_yaw = 0.0
        human.distracted_duration = 2
        human.distracted_recovery_mode = HumanMode.FOLLOWING

        pose = (1.0, 1.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[1.0, 1.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        human.step(None, data, ctx)
        self.assertEqual(human.mode, HumanMode.DISTRACTED)
        self.assertFalse(human.distracted_stop_reached)

        human.step(None, data, ctx)
        self.assertEqual(human.mode, HumanMode.FOLLOWING)

    def test_listening_distracted_force_recovers_after_max_duration(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_LISTENING
        human.distracted_duration = 3
        human.distracted_recovery_mode = HumanMode.LISTENING

        pose = (6.2, 6.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (5.0, 6.0, 0.0),
            "robot_xy": np.array([5.0, 6.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([[6.2, 6.0], [6.4, 6.1]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        human.step(None, data, ctx)
        self.assertEqual(human.mode, HumanMode.DISTRACTED)

        human.step(None, data, ctx)
        self.assertEqual(human.mode, HumanMode.DISTRACTED)

        human.step(None, data, ctx)
        self.assertEqual(human.mode, HumanMode.LISTENING)

    def test_listening_distracted_prioritizes_nearest_exhibit_over_nearby_person(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_LISTENING

        pose = (6.2, 6.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (5.0, 6.0, 0.0),
            "robot_xy": np.array([5.0, 6.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([[6.2, 6.0], [6.4, 6.1]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        action = human.step(None, data, ctx)

        expected_exhibit = np.array([9.786, 6.65], dtype=np.float32)
        exhibit_dir = expected_exhibit - np.array(pose[:2], dtype=np.float32)
        expected_focus = (
            expected_exhibit
            - DISTRACTED_TARGET_DISTANCE_MIN * (exhibit_dir / np.linalg.norm(exhibit_dir))
        )
        expected_yaw = float(np.arctan2(expected_exhibit[1] - pose[1], expected_exhibit[0] - pose[0]))

        np.testing.assert_allclose(human.distracted_target_xy, expected_focus, atol=1e-6)
        np.testing.assert_allclose(action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * expected_yaw, places=5)

    def test_listening_distracted_looks_toward_nearby_person_when_no_exhibit_exists(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_LISTENING

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (1.0, 0.0, 0.0),
            "robot_xy": np.array([1.0, 0.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        action = human.step(None, data, ctx)

        expected_focus = np.array([0.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(human.distracted_target_xy, expected_focus, atol=1e-6)
        np.testing.assert_allclose(action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * (np.pi / 2.0), places=5)

    def test_listening_distracted_falls_back_to_robot_relative_glance_when_no_focus_exists(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_LISTENING

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (1.0, 0.0, 0.0),
            "robot_xy": np.array([1.0, 0.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([[0.0, 0.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        with (
            patch("numpy.random.uniform", return_value=45.0),
            patch("numpy.random.rand", return_value=0.0),
        ):
            action = human.step(None, data, ctx)

        np.testing.assert_allclose(
            human.distracted_target_xy,
            np.array(pose[:2], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * (-np.pi / 4.0), places=5)

    def test_distracted_conversation_prioritizes_nearby_distracted_partner_over_exhibit(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (6.2, 6.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (5.0, 6.0, 0.0),
            "robot_xy": np.array([5.0, 6.0], dtype=np.float32),
            "human_xy": np.array([[6.2, 6.0], [6.4, 6.1]], dtype=np.float32),
            "human_modes": [HumanMode.DISTRACTED, HumanMode.DISTRACTED],
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        action = human.step(None, data, ctx)

        self.assertEqual(human.distracted_behavior_kind, DISTRACTED_BEHAVIOR_CONVERSATION)
        self.assertEqual(human.distracted_partner_index, 1)
        self.assertTrue(human.speaking_active)
        np.testing.assert_allclose(
            human.distracted_target_xy,
            np.array(pose[:2], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(
            float(action[2]),
            HUMAN_YAW_RATE_GAIN * float(np.arctan2(0.1, 0.2)),
            places=4,
        )

    def test_distracted_conversation_selects_one_nearest_partner_and_keeps_it_while_valid(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        first_ctx = {
            "index": 0,
            "n_humans": 3,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[0.0, 0.0], [2.5, 0.0], [1.5, 0.0]], dtype=np.float32),
            "human_modes": [
                HumanMode.DISTRACTED,
                HumanMode.DISTRACTED,
                HumanMode.DISTRACTED,
            ],
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        human.step(None, data, first_ctx)
        self.assertEqual(human.distracted_partner_index, 2)

        sticky_ctx = {
            **first_ctx,
            "human_xy": np.array([[0.0, 0.0], [0.9, 0.0], [1.5, 0.0]], dtype=np.float32),
        }
        human.step(None, data, sticky_ctx)
        self.assertEqual(human.distracted_partner_index, 2)
        self.assertEqual(human.distracted_behavior_kind, DISTRACTED_BEHAVIOR_CONVERSATION)

    def test_distracted_conversation_uses_compose_move_velocity_for_target_motion(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        adjusted_v_xy = np.array([0.0, 0.3], dtype=np.float32)
        ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            "human_modes": [HumanMode.DISTRACTED, HumanMode.DISTRACTED],
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        with patch.object(human, "_compose_move_velocity", return_value=adjusted_v_xy) as compose_mock:
            action = human.step(None, data, ctx)

        np.testing.assert_allclose(
            human.distracted_target_xy,
            np.array([2.0 - DISTRACTED_CONVERSATION_STOP_DISTANCE, 0.0], dtype=np.float32),
            atol=1e-6,
        )
        compose_mock.assert_called_once()
        np.testing.assert_allclose(
            compose_mock.call_args.kwargs["guide_xy"],
            np.array([2.0 - DISTRACTED_CONVERSATION_STOP_DISTANCE, 0.0], dtype=np.float32),
            atol=1e-6,
        )
        self.assertEqual(compose_mock.call_args.kwargs["speed_limit"], DISTRACTED_SPEED_SCALE * human.max_speed)
        np.testing.assert_allclose(action[:2], adjusted_v_xy, atol=1e-6)
        self.assertTrue(human.speaking_active)

    def test_distracted_conversation_stops_and_rotates_when_within_stop_distance(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[0.0, 0.0], [0.0, 0.5]], dtype=np.float32),
            "human_modes": [HumanMode.DISTRACTED, HumanMode.DISTRACTED],
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }

        action = human.step(None, data, ctx)
        np.testing.assert_allclose(action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * (np.pi / 2.0), places=5)

        aligned_action = human.step(None, self._make_pose_data((0.0, 0.0, np.pi / 2.0)), ctx)
        np.testing.assert_allclose(aligned_action, np.zeros(3, dtype=np.float32), atol=1e-6)

    def test_distracted_conversation_falls_back_to_regular_focus_when_partner_invalid(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        conversation_ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            "human_modes": [HumanMode.DISTRACTED, HumanMode.DISTRACTED],
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }
        human.step(None, data, conversation_ctx)
        self.assertTrue(human.speaking_active)

        lost_partner_ctx = {
            **conversation_ctx,
            "human_xy": np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32),
            "human_modes": [HumanMode.DISTRACTED, HumanMode.FOLLOWING],
        }
        with (
            patch("numpy.random.uniform", return_value=45.0),
            patch("numpy.random.rand", return_value=1.0),
        ):
            action = human.step(None, data, lost_partner_ctx)

        self.assertFalse(human.speaking_active)
        self.assertIsNone(human.distracted_partner_index)
        self.assertNotEqual(human.distracted_behavior_kind, DISTRACTED_BEHAVIOR_CONVERSATION)
        self.assertAlmostEqual(
            float(np.linalg.norm(human.distracted_target_xy - np.array(pose[:2], dtype=np.float32))),
            1.0,
            places=6,
        )
        self.assertTrue(np.isfinite(action).all())

    def test_listening_distracted_conversation_times_out_back_to_listening(self):
        empty_layout = MapLayout(
            name="test_layout_without_exhibits",
            default_xml_asset="museum_scene.xml",
            spawn_rects=(AxisAlignedRect(0.0, 1.0, 0.0, 1.0),),
            robot_waypoints=((0.0, 0.0),),
            metadata={},
        )
        human = Human("person1", "person1", 0, max_speed=1.0, map_layout=empty_layout)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_LISTENING
        human.distracted_recovery_mode = HumanMode.LISTENING
        human.distracted_duration = 2

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (1.0, 0.0, 0.0),
            "robot_xy": np.array([1.0, 0.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            "human_modes": [HumanMode.DISTRACTED, HumanMode.DISTRACTED],
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        human.step(None, data, ctx)
        self.assertEqual(human.mode, HumanMode.DISTRACTED)
        self.assertTrue(human.speaking_active)

        human.step(None, data, ctx)
        self.assertEqual(human.mode, HumanMode.LISTENING)
        self.assertFalse(human.speaking_active)

    def test_adjust_target_velocity_for_walls_keeps_clear_velocity(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        guide_xy = np.array([1.0, 0.0], dtype=np.float32)
        desired_v_xy = np.array([0.4, 0.1], dtype=np.float32)

        with patch.object(human, "_raycast_hit_distance", return_value=1.0):
            adjusted_v_xy = human._adjust_target_velocity_for_walls(
                guide_xy=guide_xy,
                desired_v_xy=desired_v_xy,
            )

        np.testing.assert_allclose(adjusted_v_xy, desired_v_xy, atol=1e-6)

    def test_adjust_target_velocity_for_walls_selects_side_detour_with_progress(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        guide_xy = np.array([1.0, 0.0], dtype=np.float32)
        desired_v_xy = np.array([1.0, 0.0], dtype=np.float32)

        def allow_only_positive_y(v_xy):
            v_xy = np.asarray(v_xy, dtype=np.float32)
            if float(v_xy[1]) > 0.0:
                return v_xy
            return np.zeros(2, dtype=np.float32)

        with (
            patch.object(human, "_raycast_hit_distance", return_value=0.1),
            patch.object(human, "_constrain_velocity_with_walkable", side_effect=allow_only_positive_y),
        ):
            adjusted_v_xy = human._adjust_target_velocity_for_walls(
                guide_xy=guide_xy,
                desired_v_xy=desired_v_xy,
            )

        self.assertGreater(float(adjusted_v_xy[0]), 0.0)
        self.assertGreater(float(adjusted_v_xy[1]), 0.0)
        self.assertGreater(float(np.dot(adjusted_v_xy, guide_xy)), 0.0)

    def test_adjust_target_velocity_for_walls_stops_when_no_detour_advances(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        guide_xy = np.array([1.0, 0.0], dtype=np.float32)
        desired_v_xy = np.array([1.0, 0.0], dtype=np.float32)

        with (
            patch.object(human, "_raycast_hit_distance", return_value=0.1),
            patch.object(
                human,
                "_constrain_velocity_with_walkable",
                return_value=np.zeros(2, dtype=np.float32),
            ),
        ):
            adjusted_v_xy = human._adjust_target_velocity_for_walls(
                guide_xy=guide_xy,
                desired_v_xy=desired_v_xy,
            )

        np.testing.assert_allclose(adjusted_v_xy, np.zeros(2, dtype=np.float32), atol=1e-6)

    def test_shared_wrap_to_pi_matches_previous_range(self):
        self.assertAlmostEqual(wrap_to_pi(3.0 * np.pi), -np.pi, places=6)
        self.assertAlmostEqual(wrap_to_pi(-1.5 * np.pi), 0.5 * np.pi, places=6)
        self.assertAlmostEqual(wrap_to_pi(0.25 * np.pi), 0.25 * np.pi, places=6)

    def test_shared_raycast_hit_distance_uses_single_height_probe(self):
        model = object()
        data = SimpleNamespace(xpos=np.array([[1.0, 2.0, 0.125]], dtype=np.float64))

        with patch("museum_env.spatial_utils.mujoco.mj_ray", return_value=0.2) as ray_mock:
            hit_distance = raycast_hit_distance(
                model,
                data,
                0,
                np.array([1.0, 0.0], dtype=np.float32),
            )

        self.assertEqual(ray_mock.call_count, 1)
        np.testing.assert_allclose(ray_mock.call_args.args[2], np.array([1.0, 2.0, 0.125]), atol=1e-6)
        self.assertAlmostEqual(hit_distance, 0.2, places=6)

    def test_shared_raycast_hit_distance_handles_invalid_and_miss_cases(self):
        data = SimpleNamespace(xpos=np.array([[1.0, 2.0, 0.125]], dtype=np.float64))
        self.assertIsNone(raycast_hit_distance(object(), data, 0, np.zeros(2, dtype=np.float32)))
        self.assertIsNone(raycast_hit_distance(None, data, 0, np.array([1.0, 0.0], dtype=np.float32)))
        with patch("museum_env.spatial_utils.mujoco.mj_ray", return_value=-1.0):
            self.assertIsNone(
                raycast_hit_distance(
                    object(),
                    data,
                    0,
                    np.array([1.0, 0.0], dtype=np.float32),
                )
            )

    def test_wall_spacing_force_is_zero_when_side_probes_are_clear(self):
        human = Human("person1", "person1", 0, max_speed=1.0)

        with patch.object(human, "_raycast_hit_distance", return_value=1.0):
            wall_force = human._compute_wall_spacing_force(np.array([1.0, 0.0], dtype=np.float32))

        np.testing.assert_allclose(wall_force, np.zeros(2, dtype=np.float32), atol=1e-6)

    def test_wall_spacing_force_pushes_right_when_left_wall_is_close(self):
        human = Human("person1", "person1", 0, max_speed=1.0)

        with patch.object(human, "_raycast_hit_distance", side_effect=[0.2, 1.0]):
            wall_force = human._compute_wall_spacing_force(np.array([1.0, 0.0], dtype=np.float32))

        expected_force = np.array(
            [0.0, -WALL_REPULSION_GAIN * (WALL_REPULSION_DISTANCE_METERS - 0.2)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(wall_force, expected_force, atol=1e-6)

    def test_wall_spacing_force_pushes_left_when_right_wall_is_close(self):
        human = Human("person1", "person1", 0, max_speed=1.0)

        with patch.object(human, "_raycast_hit_distance", side_effect=[1.0, 0.2]):
            wall_force = human._compute_wall_spacing_force(np.array([1.0, 0.0], dtype=np.float32))

        expected_force = np.array(
            [0.0, WALL_REPULSION_GAIN * (WALL_REPULSION_DISTANCE_METERS - 0.2)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(wall_force, expected_force, atol=1e-6)

    def test_compose_move_velocity_adds_wall_force_before_wall_adjustment(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        current_xy = np.array([0.0, 0.0], dtype=np.float32)
        guide_xy = np.array([1.0, 0.0], dtype=np.float32)
        wall_force = np.array([0.0, 0.2], dtype=np.float32)
        expected_pre_adjust_v_xy = human._limit_speed(
            np.array([1.0, 0.2], dtype=np.float32),
            human.max_speed,
        )

        with (
            patch.object(human, "_compute_wall_spacing_force", return_value=wall_force),
            patch.object(
                human,
                "_adjust_target_velocity_for_walls",
                side_effect=lambda guide_xy, desired_v_xy: np.asarray(desired_v_xy, dtype=np.float32),
            ) as adjust_mock,
        ):
            v_total = human._compose_move_velocity(
                current_xy=current_xy,
                guide_xy=guide_xy,
                goal_v_xy=np.array([1.0, 0.0], dtype=np.float32),
                speed_limit=human.max_speed,
            )

        np.testing.assert_allclose(adjust_mock.call_args.kwargs["guide_xy"], guide_xy, atol=1e-6)
        np.testing.assert_allclose(
            adjust_mock.call_args.kwargs["desired_v_xy"],
            expected_pre_adjust_v_xy,
            atol=1e-6,
        )
        np.testing.assert_allclose(v_total, expected_pre_adjust_v_xy, atol=1e-6)

    def test_compose_move_velocity_combines_goal_repulsion_hr_and_wall_before_adjustment(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        current_xy = np.array([0.0, 0.0], dtype=np.float32)
        guide_xy = np.array([1.0, 0.0], dtype=np.float32)
        goal_v_xy = np.array([0.4, 0.0], dtype=np.float32)
        repulsion_xy = np.array([0.1, 0.0], dtype=np.float32)
        hr_force = np.array([0.0, 0.2], dtype=np.float32)
        wall_force = np.array([0.0, 0.1], dtype=np.float32)
        expected_pre_adjust_v_xy = np.array([0.5, 0.3], dtype=np.float32)

        with (
            patch.object(human, "_compute_hr_spacing_force", return_value=hr_force) as hr_mock,
            patch.object(human, "_compute_wall_spacing_force", return_value=wall_force),
            patch.object(
                human,
                "_adjust_target_velocity_for_walls",
                side_effect=lambda guide_xy, desired_v_xy: np.asarray(desired_v_xy, dtype=np.float32),
            ) as adjust_mock,
        ):
            v_total = human._compose_move_velocity(
                current_xy=current_xy,
                guide_xy=guide_xy,
                goal_v_xy=goal_v_xy,
                speed_limit=2.0,
                repulsion_xy=repulsion_xy,
                robot_xy=np.array([2.0, 0.0], dtype=np.float32),
                hr_distance_min=human.hr_distance_min,
                hr_distance_max=human.hr_distance_max,
            )

        hr_mock.assert_called_once()
        np.testing.assert_allclose(adjust_mock.call_args.kwargs["guide_xy"], guide_xy, atol=1e-6)
        np.testing.assert_allclose(
            adjust_mock.call_args.kwargs["desired_v_xy"],
            expected_pre_adjust_v_xy,
            atol=1e-6,
        )
        np.testing.assert_allclose(v_total, expected_pre_adjust_v_xy, atol=1e-6)

    def test_move_uses_compose_move_velocity_helper(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        current_xy = np.array([0.0, 0.0], dtype=np.float32)
        to_target_xy = np.array([1.0, 0.0], dtype=np.float32)
        adjusted_v_xy = np.array([0.0, 0.5], dtype=np.float32)
        ctx = {
            "robot_xy": np.array([2.0, 0.0], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
        }

        with patch.object(human, "_compose_move_velocity", return_value=adjusted_v_xy) as compose_mock:
            action = human._move(to_target_xy, 0.0, ctx, current_xy)

        compose_mock.assert_called_once()
        np.testing.assert_allclose(compose_mock.call_args.kwargs["guide_xy"], to_target_xy, atol=1e-6)
        np.testing.assert_allclose(
            compose_mock.call_args.kwargs["goal_v_xy"],
            np.array([1.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )
        self.assertEqual(compose_mock.call_args.kwargs["speed_limit"], human.max_speed)
        np.testing.assert_allclose(action[:2], adjusted_v_xy, atol=1e-6)

    def test_listening_uses_compose_move_velocity_for_target_motion(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.LISTENING)
        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        adjusted_v_xy = np.array([0.0, 0.3], dtype=np.float32)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (1.0, 0.0, 0.0),
            "robot_xy": np.array([1.0, 0.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([[0.0, 0.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        with patch.object(human, "_compose_move_velocity", return_value=adjusted_v_xy) as compose_mock:
            action = human.step(None, data, ctx)

        compose_mock.assert_called_once()
        self.assertGreater(float(np.linalg.norm(compose_mock.call_args.kwargs["guide_xy"])), 0.0)
        self.assertIsNone(compose_mock.call_args.kwargs["hr_distance_max"])
        np.testing.assert_allclose(action[:2], adjusted_v_xy, atol=1e-6)

    def test_following_distracted_uses_compose_move_velocity_and_preserves_forward_progress(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.DISTRACTED)
        human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
        human.distracted_target_xy = np.array([6.0, 6.0], dtype=np.float32)
        human.distracted_target_yaw = float(np.pi / 2.0)

        pose = (6.0, 5.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "human_xy": np.array([[6.0, 5.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "follow_radius": 1.0,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
        }
        reverse_v_xy = np.array([0.0, -0.5], dtype=np.float32)
        fallback_v_xy = np.array([0.0, 0.5], dtype=np.float32)

        with (
            patch.object(human, "_compose_move_velocity", return_value=reverse_v_xy) as compose_mock,
            patch.object(human, "_constrain_velocity_with_walkable", return_value=fallback_v_xy),
        ):
            action = human.step(None, data, ctx)

        compose_mock.assert_called_once()
        np.testing.assert_allclose(
            compose_mock.call_args.kwargs["guide_xy"],
            np.array([0.0, 1.0], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(action[:2], fallback_v_xy, atol=1e-6)

    def test_listening_impatient_rotates_in_place_without_repulsion(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.start_impatient(recovery_mode=HumanMode.LISTENING)

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (1.0, 0.0, 0.0),
            "robot_xy": np.array([1.0, 0.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([[0.0, 0.0]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        with (
            patch("numpy.random.uniform", return_value=45.0),
            patch("numpy.random.rand", return_value=1.0),
        ):
            human.start_impatient(recovery_mode=HumanMode.LISTENING)
            action = human.step(None, data, ctx)

        np.testing.assert_allclose(action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * (np.pi / 4.0), places=5)

    def test_listening_impatient_ignores_repulsion_translation(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        repulsion = np.array([0.2, -0.1], dtype=np.float32)

        pose = (0.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 2,
            "robot_pose": (1.0, 0.0, 0.0),
            "robot_xy": np.array([1.0, 0.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
            "repulsion": repulsion,
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        with (
            patch("numpy.random.uniform", return_value=45.0),
            patch("numpy.random.rand", return_value=0.0),
        ):
            human.start_impatient(recovery_mode=HumanMode.LISTENING)
            action = human.step(None, data, ctx)

        np.testing.assert_allclose(action[:2], np.zeros(2, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * (-np.pi / 4.0), places=5)

    def test_apply_runtime_config_sets_listening_impatient_glance_steps(self):
        human = Human("person1", "person1", 0, max_speed=1.0)

        human.apply_runtime_config(
            dt=0.1,
            max_distracted_duration_seconds=10.0,
            impatient_duration_seconds=6.0,
            impatient_speed_multiplier=1.5,
            impatient_front_offset=1.0,
            listening_impatient_glance_seconds=2.0,
        )

        self.assertEqual(human.impatient_duration, 60)
        self.assertEqual(human.listening_impatient_glance_steps, 20)

    def test_following_impatient_uses_dedicated_narrower_fan_half_angle(self):
        robot_pose = (5.0, 5.0, 0.0)
        impatient_fan_half_angle = np.deg2rad(30.0)
        for index, expected_angle_deg in ((0, -30.0), (1, 30.0)):
            with self.subTest(index=index):
                human = Human(f"person{index + 1}", f"person{index + 1}", index, max_speed=1.0)
                human.start_impatient(recovery_mode=HumanMode.FOLLOWING)
                ctx = {
                    "index": index,
                    "n_humans": 2,
                    "robot_pose": robot_pose,
                    "robot_xy": np.array(robot_pose[:2], dtype=np.float32),
                    "human_xy": np.zeros((2, 2), dtype=np.float32),
                    "repulsion": np.zeros(2, dtype=np.float32),
                    "fan_half_angle": np.deg2rad(80.0),
                    "impatient_fan_half_angle": impatient_fan_half_angle,
                    "impatient_front_offset": human.impatient_front_offset,
                }

                human.assign_target_from_context(ctx)

                expected_waypoint = np.array(
                    [
                        robot_pose[0] + human.impatient_front_offset * np.cos(np.deg2rad(expected_angle_deg)),
                        robot_pose[1] + human.impatient_front_offset * np.sin(np.deg2rad(expected_angle_deg)),
                    ],
                    dtype=np.float32,
                )
                np.testing.assert_allclose(human.current_waypoint, expected_waypoint, atol=1e-6)

    def test_listening_impatient_moves_toward_corridor_midpoint_after_look_phase(self):
        cases = (
            ((1.0, 5.0, 0.0), np.array([5.0, 5.0], dtype=np.float32), np.array([8.25, 0.0], dtype=np.float32)),
            (
                (9.25, -12.5, 0.0),
                np.array([9.25, -11.0], dtype=np.float32),
                np.array([8.25, -10.0], dtype=np.float32),
            ),
        )
        for pose, robot_xy, expected_target in cases:
            with self.subTest(pose=pose, expected_target=tuple(expected_target.tolist())):
                human = Human("person1", "person1", 0, max_speed=1.0)
                human.impatient_speed_multiplier = 1.0
                with (
                    patch("numpy.random.uniform", return_value=45.0),
                    patch("numpy.random.rand", return_value=1.0),
                ):
                    human.start_impatient(recovery_mode=HumanMode.LISTENING)

                human.impatient_duration = 6
                human.listening_impatient_glance_steps = 3
                human.impatient_timer = human.listening_impatient_glance_steps

                data = self._make_pose_data(pose)
                ctx = {
                    "index": 0,
                    "n_humans": 1,
                    "robot_pose": (float(robot_xy[0]), float(robot_xy[1]), 0.0),
                    "robot_xy": robot_xy,
                    "robot_yaw": 0.0,
                    "human_xy": np.array([pose[:2]], dtype=np.float32),
                    "repulsion": np.zeros(2, dtype=np.float32),
                    "fan_half_angle": np.deg2rad(80.0),
                    "impatient_front_offset": human.impatient_front_offset,
                    "listen_radius": 1.0,
                    "listening_sector_half_angle": np.deg2rad(80.0),
                }

                action = human.step(None, data, ctx)

                direction = expected_target - np.asarray(pose[:2], dtype=np.float32)
                expected_velocity = direction / np.linalg.norm(direction)
                expected_yaw_rate = HUMAN_YAW_RATE_GAIN * wrap_to_pi(
                    float(np.arctan2(expected_velocity[1], expected_velocity[0])) - float(pose[2])
                )

                np.testing.assert_allclose(action[:2], expected_velocity, atol=1e-6)
                self.assertAlmostEqual(float(action[2]), expected_yaw_rate, places=5)

    def test_listening_impatient_move_uses_compose_move_velocity(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.impatient_speed_multiplier = 1.0
        with (
            patch("numpy.random.uniform", return_value=45.0),
            patch("numpy.random.rand", return_value=1.0),
        ):
            human.start_impatient(recovery_mode=HumanMode.LISTENING)

        human.impatient_duration = 6
        human.listening_impatient_glance_steps = 3
        human.impatient_timer = human.listening_impatient_glance_steps

        pose = (1.0, 5.0, 0.0)
        data = self._make_pose_data(pose)
        adjusted_v_xy = np.array([0.2, 0.1], dtype=np.float32)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([pose[:2]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        with patch.object(human, "_compose_move_velocity", return_value=adjusted_v_xy) as compose_mock:
            action = human.step(None, data, ctx)

        compose_mock.assert_called_once()
        self.assertIsNone(compose_mock.call_args.kwargs["hr_distance_max"])
        np.testing.assert_allclose(action[:2], adjusted_v_xy, atol=1e-6)
        expected_yaw_rate = HUMAN_YAW_RATE_GAIN * wrap_to_pi(
            float(np.arctan2(adjusted_v_xy[1], adjusted_v_xy[0])) - float(pose[2])
        )
        self.assertAlmostEqual(float(action[2]), expected_yaw_rate, places=6)

    def test_listening_impatient_preserves_forward_progress_with_fallback_velocity(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.impatient_speed_multiplier = 1.0
        with (
            patch("numpy.random.uniform", return_value=45.0),
            patch("numpy.random.rand", return_value=1.0),
        ):
            human.start_impatient(recovery_mode=HumanMode.LISTENING)

        human.impatient_duration = 6
        human.listening_impatient_glance_steps = 3
        human.impatient_timer = human.listening_impatient_glance_steps

        pose = (8.25, -11.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (9.25, -11.0, 0.0),
            "robot_xy": np.array([9.25, -11.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([pose[:2]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }
        reverse_v_xy = np.array([0.0, -0.5], dtype=np.float32)
        fallback_v_xy = np.array([0.0, 0.5], dtype=np.float32)

        with (
            patch.object(human, "_compose_move_velocity", return_value=reverse_v_xy) as compose_mock,
            patch.object(human, "_constrain_velocity_with_walkable", return_value=fallback_v_xy),
        ):
            action = human.step(None, data, ctx)

        compose_mock.assert_called_once()
        np.testing.assert_allclose(
            compose_mock.call_args.kwargs["guide_xy"],
            np.array([0.0, 1.0], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(action[:2], fallback_v_xy, atol=1e-6)
        self.assertAlmostEqual(float(action[2]), HUMAN_YAW_RATE_GAIN * (np.pi / 2.0), places=5)

    def test_listening_impatient_holds_at_corridor_midpoint_until_duration_ends(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.impatient_speed_multiplier = 1.0
        with (
            patch("numpy.random.uniform", return_value=45.0),
            patch("numpy.random.rand", return_value=1.0),
        ):
            human.start_impatient(recovery_mode=HumanMode.LISTENING)

        human.impatient_duration = 5
        human.listening_impatient_glance_steps = 2
        human.impatient_timer = human.listening_impatient_glance_steps

        pose = (8.25, -10.0, 0.7)
        data = self._make_pose_data(pose)
        ctx = {
            "index": 0,
            "n_humans": 1,
            "robot_pose": (5.0, 5.0, 0.0),
            "robot_xy": np.array([5.0, 5.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "human_xy": np.array([pose[:2]], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
            "fan_half_angle": np.deg2rad(80.0),
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(80.0),
        }

        action = human.step(None, data, ctx)
        np.testing.assert_allclose(action, np.zeros(3, dtype=np.float32), atol=1e-6)
        self.assertEqual(human.mode, HumanMode.IMPATIENT)

        human.impatient_timer = human.impatient_duration - 1
        action = human.step(None, data, ctx)
        np.testing.assert_allclose(action, np.zeros(3, dtype=np.float32), atol=1e-6)
        self.assertEqual(human.mode, HumanMode.LISTENING)

    def test_listening_impatient_corridor_fallback_uses_human_nearest_room(self):
        for pose, robot_xy, expected_target in (
            ((8.0, -1.0, 0.0), np.array([9.25, -12.5], dtype=np.float32), np.array([8.25, 0.0], dtype=np.float32)),
            ((8.0, -5.0, 0.0), np.array([5.0, 5.0], dtype=np.float32), np.array([8.25, -10.0], dtype=np.float32)),
        ):
            with self.subTest(
                pose=pose,
                robot_xy=tuple(robot_xy.tolist()),
                expected_target=tuple(expected_target.tolist()),
            ):
                human = Human("person1", "person1", 0, max_speed=1.0)
                human.impatient_speed_multiplier = 1.0
                with (
                    patch("numpy.random.uniform", return_value=45.0),
                    patch("numpy.random.rand", return_value=1.0),
                ):
                    human.start_impatient(recovery_mode=HumanMode.LISTENING)

                human.impatient_duration = 6
                human.listening_impatient_glance_steps = 3
                human.impatient_timer = human.listening_impatient_glance_steps

                data = self._make_pose_data(pose)
                ctx = {
                    "index": 0,
                    "n_humans": 1,
                    "robot_pose": (float(robot_xy[0]), float(robot_xy[1]), 0.0),
                    "robot_xy": robot_xy,
                    "robot_yaw": 0.0,
                    "human_xy": np.array([pose[:2]], dtype=np.float32),
                    "repulsion": np.zeros(2, dtype=np.float32),
                    "fan_half_angle": np.deg2rad(80.0),
                    "impatient_front_offset": human.impatient_front_offset,
                    "listen_radius": 1.0,
                    "listening_sector_half_angle": np.deg2rad(80.0),
                }

                action = human.step(None, data, ctx)
                expected_velocity = expected_target - np.asarray(pose[:2], dtype=np.float32)
                expected_velocity = expected_velocity / np.linalg.norm(expected_velocity)
                expected_yaw_rate = HUMAN_YAW_RATE_GAIN * wrap_to_pi(
                    float(np.arctan2(expected_velocity[1], expected_velocity[0])) - float(pose[2])
                )
                np.testing.assert_allclose(action[:2], expected_velocity, atol=1e-6)
                self.assertAlmostEqual(float(action[2]), expected_yaw_rate, places=5)

    def test_general_phase_impatient_recovers_to_current_phase_mode(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=19)
            env.follow_phase = "transit_follow"
            human = env.humans[0]
            with (
                patch("numpy.random.uniform", return_value=45.0),
                patch("numpy.random.rand", return_value=1.0),
            ):
                human.start_impatient(recovery_mode=HumanMode.LISTENING)

            human.impatient_duration = 1
            human.impatient_timer = 0

            current_xy = np.array(human.get_pose(env.data)[:2], dtype=np.float32)
            world_frame = SimpleNamespace(
                repulsion_vectors=np.zeros((1, 2), dtype=np.float32),
                robot_pose=(5.0, 5.0, 0.0),
                robot_xy=np.array([5.0, 5.0], dtype=np.float32),
                human_xy=current_xy.reshape(1, 2),
            )

            env_control.apply_general_phase_strategy(env, human, 0, world_frame)
            self.assertEqual(human.mode, HumanMode.FOLLOWING)
        finally:
            env.close()

    def test_transit_follow_slowdown_applies_when_any_human_exceeds_distance_threshold(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=29)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.robot.v_max = 3.0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 1.0, 0.0), (0.0, 3.0, 0.0)),
            )
            self._invalidate_observation_cache(env)
            baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(
                info["robot"]["action"]["vx"],
                0.7 * float(baseline_action[0]),
                places=6,
            )
            self.assertAlmostEqual(
                info["robot"]["action"]["vy"],
                0.7 * float(baseline_action[1]),
                places=6,
            )
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], float(baseline_action[2]), places=6)
            self.assertEqual(info["robot"]["mode"], "move")
        finally:
            env.close()

    def test_transit_follow_waits_when_any_human_exceeds_wait_distance_threshold(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=129)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.robot.v_max = 3.0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 1.0, 0.0), (0.0, 4.1, 0.0)),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], 0.0, places=6)
            self.assertEqual(info["robot"]["mode"], "stop")
        finally:
            env.close()

    def test_transit_follow_does_not_slow_when_all_humans_are_within_threshold(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=30)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.robot.v_max = 3.0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 1.0, 0.0), (0.0, 2.4, 0.0)),
            )
            self._invalidate_observation_cache(env)
            baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], float(baseline_action[0]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], float(baseline_action[1]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], float(baseline_action[2]), places=6)
            self.assertEqual(info["robot"]["mode"], "move")
        finally:
            env.close()

    def test_pre_listen_follow_does_not_slow_when_human_exceeds_distance_threshold(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=31)
            env.follow_phase = "pre_listen_engage"
            env.robot.listen_done = False
            env.robot.v_max = 3.0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((3.0, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], float(baseline_action[0]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], float(baseline_action[1]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], float(baseline_action[2]), places=6)
            self.assertEqual(info["robot"]["mode"], "move")
        finally:
            env.close()

    def test_transit_follow_slowdown_recovers_immediately_when_distance_returns_within_threshold(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=32)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.robot.v_max = 3.0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 3.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            slowed_baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, slowed_info = env.step(None)
            self.assertAlmostEqual(
                slowed_info["robot"]["action"]["vx"],
                0.7 * float(slowed_baseline_action[0]),
                places=6,
            )
            self.assertAlmostEqual(
                slowed_info["robot"]["action"]["vy"],
                0.7 * float(slowed_baseline_action[1]),
                places=6,
            )
            self.assertAlmostEqual(
                slowed_info["robot"]["action"]["yaw_rate"],
                float(slowed_baseline_action[2]),
                places=6,
            )

            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 2.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            recovered_baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, recovered_info = env.step(None)
            self.assertAlmostEqual(recovered_info["robot"]["action"]["vx"], float(recovered_baseline_action[0]), places=6)
            self.assertAlmostEqual(recovered_info["robot"]["action"]["vy"], float(recovered_baseline_action[1]), places=6)
            self.assertAlmostEqual(
                recovered_info["robot"]["action"]["yaw_rate"],
                float(recovered_baseline_action[2]),
                places=6,
            )
        finally:
            env.close()

    def test_pre_listen_follow_does_not_wait_when_human_exceeds_wait_distance_threshold(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=131)
            env.follow_phase = "pre_listen_engage"
            env.robot.listen_done = False
            env.robot.v_max = 3.0
            env.following_callback_wait_steps = 1
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((4.1, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], float(baseline_action[0]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], float(baseline_action[1]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], float(baseline_action[2]), places=6)
            self.assertEqual(info["robot"]["mode"], "move")
            self.assertFalse(info["events"]["callback_triggered"])
        finally:
            env.close()

    def test_transit_follow_wait_recovers_to_slowdown_when_distance_returns_within_wait_threshold(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=132)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.robot.v_max = 3.0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 4.1, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, waiting_info = env.step(None)
            self.assertAlmostEqual(waiting_info["robot"]["action"]["vx"], 0.0, places=6)
            self.assertAlmostEqual(waiting_info["robot"]["action"]["vy"], 0.0, places=6)
            self.assertAlmostEqual(waiting_info["robot"]["action"]["yaw_rate"], 0.0, places=6)
            self.assertEqual(waiting_info["robot"]["mode"], "stop")

            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 3.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            slowed_baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, recovered_info = env.step(None)
            self.assertAlmostEqual(
                recovered_info["robot"]["action"]["vx"],
                0.7 * float(slowed_baseline_action[0]),
                places=6,
            )
            self.assertAlmostEqual(
                recovered_info["robot"]["action"]["vy"],
                0.7 * float(slowed_baseline_action[1]),
                places=6,
            )
            self.assertAlmostEqual(
                recovered_info["robot"]["action"]["yaw_rate"],
                float(slowed_baseline_action[2]),
                places=6,
            )
            self.assertEqual(recovered_info["robot"]["mode"], "move")
        finally:
            env.close()

    def test_transit_follow_personal_space_backoff_moves_away_from_close_human(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=138)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            baseline_action, _, _, _ = env.robot._waypoint_action((5.0, 5.0, 0.0))

            with patch("museum_env.env_control.raycast_hit_distance", return_value=1.0):
                _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.5, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], float(baseline_action[2]), places=6)
            self.assertEqual(info["robot"]["mode"], "move")
        finally:
            env.close()

    def test_transit_follow_personal_space_backoff_targets_nearest_close_human(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=139)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0), (5.0, 4.5, 0.0)),
            )
            self._invalidate_observation_cache(env)

            with patch("museum_env.env_control.raycast_hit_distance", return_value=1.0):
                _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.5, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
        finally:
            env.close()

    def test_transit_follow_personal_space_backoff_freezes_when_away_is_unsafe(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=140)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((5.0, 5.55, 0.0),),
            )
            self._invalidate_observation_cache(env)

            def raycast_side_effect(_model, _data, _body_id, direction_xy):
                direction_xy = np.asarray(direction_xy, dtype=np.float32)
                direction_xy = direction_xy / np.linalg.norm(direction_xy)
                if direction_xy[1] < -0.99:
                    return 0.1
                if direction_xy[0] < -0.99:
                    return 0.1
                return 1.0

            with patch("museum_env.env_control.raycast_hit_distance", side_effect=raycast_side_effect):
                _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
        finally:
            env.close()

    def test_transit_follow_personal_space_backoff_does_not_start_lateral_session_when_away_is_unsafe(
        self,
    ):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=141)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((5.0, 5.55, 0.0),),
            )
            self._invalidate_observation_cache(env)
            world_frame = env._build_world_frame(force=True)
            call_count = {"value": 0}

            def safety_side_effect(_env, world_frame, target_idx, direction_xy):
                del world_frame, target_idx
                call_count["value"] += 1
                direction_xy = np.asarray(direction_xy, dtype=np.float32)
                direction_xy = direction_xy / np.linalg.norm(direction_xy)
                if direction_xy[1] < -0.99:
                    return False, float("-inf"), float("-inf"), float("-inf")
                if direction_xy[0] > 0.99:
                    return True, 0.30, 0.60, 0.80
                return True, 0.10, 0.95, 10.0

            with patch.object(env, "_log_event") as log_mock, patch(
                "museum_env.env_control.is_robot_backoff_direction_safe",
                side_effect=safety_side_effect,
            ):
                action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )

            self.assertEqual(call_count["value"], 1)
            self.assertEqual(log_mock.call_count, 0)
            self.assertFalse(env.personal_space_backoff_state.active)
            self.assertEqual(env.personal_space_backoff_state.direction_label, None)
            self.assertAlmostEqual(action[0], 0.0, places=6)
            self.assertAlmostEqual(action[1], 0.0, places=6)
        finally:
            env.close()

    def test_transit_follow_personal_space_backoff_freezes_translation_when_all_directions_are_unsafe(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=142)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            baseline_action, _, _, _ = env.robot._waypoint_action((5.0, 5.0, 0.0))

            with patch(
                "museum_env.env_control.is_robot_backoff_direction_safe",
                return_value=(False, float("-inf"), float("-inf"), float("-inf")),
            ):
                _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], float(baseline_action[2]), places=6)
        finally:
            env.close()

    def test_listening_wait_personal_space_backoff_overrides_stop(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=143)
            env.listening_state.enter_wait(False)
            env.robot.listen_mode = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((5.4, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            with patch("museum_env.env_control.raycast_hit_distance", return_value=1.0):
                _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], -0.5, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
            self.assertEqual(info["robot"]["mode"], "move")
        finally:
            env.close()

    def test_listening_paused_personal_space_backoff_preserves_turn_yaw_rate(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=144)
            env.robot.listen_mode = True
            env.listening_state.enter_wait(False)
            env.listening_state.pause()
            env.listening_state.question_phase = LISTEN_QUESTION_PHASE_TURN_TO_HUMAN
            env.listening_state.question_human_idx = 0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((5.0, 5.55, 0.0),),
            )
            self._invalidate_observation_cache(env)

            with patch("museum_env.env_control.raycast_hit_distance", return_value=1.0):
                _, _, _, _, info = env.step(None)

            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], -0.5, places=6)
            self.assertAlmostEqual(abs(info["robot"]["action"]["yaw_rate"]), 1.0, places=6)
            self.assertEqual(info["robot"]["mode"], "move")
        finally:
            env.close()

    def test_active_callback_personal_space_backoff_does_not_interrupt_callback(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=145)
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((5.4, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            env.robot.start_callback(
                target_idx=0,
                target_xy=np.array([5.4, 5.0], dtype=np.float32),
                cue_steps=2,
            )

            with patch("museum_env.env_control.raycast_hit_distance", return_value=1.0):
                _, _, _, _, info = env.step(None)

            self.assertEqual(info["robot"]["mode"], "callback")
            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
        finally:
            env.close()

    def test_non_listening_stop_personal_space_backoff_does_not_override_protected_stop(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=146)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 99
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0), (5.0, 9.1, 0.0)),
            )
            self._invalidate_observation_cache(env)

            with patch("museum_env.env_control.raycast_hit_distance", return_value=1.0):
                _, _, _, _, info = env.step(None)

            self.assertEqual(info["robot"]["mode"], "stop")
            self.assertAlmostEqual(info["robot"]["action"]["vx"], 0.0, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
        finally:
            env.close()

    def test_personal_space_backoff_session_keeps_direction_until_target_reaches_release_distance(
        self,
    ):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=1461)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            world_frame = env._build_world_frame(force=True)

            with patch.object(env, "_log_event") as log_mock, patch(
                "museum_env.env_control.raycast_hit_distance",
                return_value=1.0,
            ):
                first_action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )

                self._set_robot_and_human_poses(
                    env,
                    robot_pose=(5.10, 5.0, 0.0),
                    human_poses=((4.6, 5.0, 0.0),),
                )
                self._invalidate_observation_cache(env)
                mid_frame = env._build_world_frame(force=True)
                second_action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    mid_frame,
                )

                self._set_robot_and_human_poses(
                    env,
                    robot_pose=(5.25, 5.0, 0.0),
                    human_poses=((4.6, 5.0, 0.0),),
                )
                self._invalidate_observation_cache(env)
                end_frame = env._build_world_frame(force=True)
                third_action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    end_frame,
                )

                self._set_robot_and_human_poses(
                    env,
                    robot_pose=(5.35, 5.0, 0.0),
                    human_poses=((4.6, 5.0, 0.0),),
                )
                self._invalidate_observation_cache(env)
                release_frame = env._build_world_frame(force=True)
                fourth_action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    release_frame,
                )

            self.assertEqual(log_mock.call_count, 2)
            self.assertIn("Robot personal-space backoff near person1", log_mock.call_args_list[0].args[0])
            self.assertIn("direction=away", log_mock.call_args_list[0].args[0])
            self.assertEqual(log_mock.call_args_list[1].args[0], ">>> Robot personal-space backoff ended.")
            self.assertFalse(env.personal_space_backoff_state.active)
            self.assertEqual(env.personal_space_backoff_state.direction_label, None)
            self.assertGreater(first_action[0], 0.0)
            self.assertAlmostEqual(first_action[1], 0.0, places=6)
            self.assertGreater(second_action[0], 0.0)
            self.assertAlmostEqual(second_action[1], 0.0, places=6)
            self.assertGreater(third_action[0], 0.0)
            self.assertAlmostEqual(third_action[1], 0.0, places=6)
            self.assertAlmostEqual(fourth_action[0], 0.0, places=6)
            self.assertAlmostEqual(fourth_action[1], 0.0, places=6)
        finally:
            env.close()

    def test_personal_space_backoff_ends_when_tracked_target_reaches_release_distance(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=14614)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.31, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0),),
            )
            env.personal_space_backoff_state.start(
                target_idx=0,
                direction_label="away",
                direction_xy=np.array([1.0, 0.0], dtype=np.float32),
                start_xy=np.array([5.0, 5.0], dtype=np.float32),
            )
            self._invalidate_observation_cache(env)
            world_frame = env._build_world_frame(force=True)

            with patch.object(env, "_log_event") as log_mock:
                action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )

            self.assertEqual(log_mock.call_count, 1)
            self.assertEqual(log_mock.call_args_list[0].args[0], ">>> Robot personal-space backoff ended.")
            self.assertFalse(env.personal_space_backoff_state.active)
            self.assertEqual(env.personal_space_backoff_state.direction_label, None)
            self.assertAlmostEqual(action[0], 0.0, places=6)
            self.assertAlmostEqual(action[1], 0.0, places=6)
        finally:
            env.close()

    def test_personal_space_backoff_keeps_full_speed_just_before_hard_retreat_cap(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=14615)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            near_end_progress = float(env_control.ROBOT_PERSONAL_SPACE_MAX_RETREAT_DISTANCE_METERS) - 1e-4
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0 + near_end_progress, 5.0, 0.0),
                human_poses=((5.0, 5.0, 0.0),),
            )
            env.personal_space_backoff_state.start(
                target_idx=0,
                direction_label="away",
                direction_xy=np.array([1.0, 0.0], dtype=np.float32),
                start_xy=np.array([5.0, 5.0], dtype=np.float32),
            )
            self._invalidate_observation_cache(env)
            world_frame = env._build_world_frame(force=True)

            with patch(
                "museum_env.env_control.is_robot_backoff_direction_safe",
                return_value=(True, 0.2, 1.0, 1.0),
            ):
                action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )

            self.assertTrue(env.personal_space_backoff_state.active)
            self.assertAlmostEqual(
                action[0],
                float(env_control.ROBOT_PERSONAL_SPACE_BACKOFF_SPEED_METERS),
                places=6,
            )
            self.assertAlmostEqual(action[1], 0.0, places=6)
            self.assertEqual(env.robot.mode, "move")
        finally:
            env.close()

    def test_personal_space_backoff_keeps_same_session_when_retreat_exceeds_old_threshold_but_target_is_still_close(
        self,
    ):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=1462)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.7, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            first_frame = env._build_world_frame(force=True)

            with patch.object(env, "_log_event") as log_mock, patch(
                "museum_env.env_control.raycast_hit_distance",
                return_value=1.0,
            ):
                env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    first_frame,
                )

                self._set_robot_and_human_poses(
                    env,
                    robot_pose=(5.25, 5.0, 0.0),
                    human_poses=((4.7, 5.0, 0.0),),
                )
                self._invalidate_observation_cache(env)
                second_frame = env._build_world_frame(force=True)
                second_action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    second_frame,
                )

            self.assertEqual(log_mock.call_count, 1)
            self.assertIn("direction=away", log_mock.call_args_list[0].args[0])
            self.assertTrue(env.personal_space_backoff_state.active)
            np.testing.assert_allclose(
                env.personal_space_backoff_state.start_xy,
                np.array([5.0, 5.0], dtype=np.float32),
                atol=1e-6,
            )
            self.assertGreater(second_action[0], 0.0)
            self.assertAlmostEqual(second_action[1], 0.0, places=6)
        finally:
            env.close()

    def test_personal_space_backoff_ends_at_hard_retreat_cap_and_can_restart_later(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=14621)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            hard_cap_progress = float(env_control.ROBOT_PERSONAL_SPACE_MAX_RETREAT_DISTANCE_METERS)
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0 + hard_cap_progress, 5.0, 0.0),
                human_poses=((5.05, 5.0, 0.0),),
            )
            env.personal_space_backoff_state.start(
                target_idx=0,
                direction_label="away",
                direction_xy=np.array([1.0, 0.0], dtype=np.float32),
                start_xy=np.array([5.0, 5.0], dtype=np.float32),
            )
            self._invalidate_observation_cache(env)
            world_frame = env._build_world_frame(force=True)

            with patch.object(env, "_log_event") as log_mock, patch(
                "museum_env.env_control.is_robot_backoff_direction_safe",
                return_value=(True, 0.2, 1.0, 1.0),
            ):
                end_action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )
                restart_action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )

            self.assertEqual(log_mock.call_count, 2)
            self.assertEqual(log_mock.call_args_list[0].args[0], ">>> Robot personal-space backoff ended.")
            self.assertIn("Robot personal-space backoff near person1", log_mock.call_args_list[1].args[0])
            self.assertTrue(env.personal_space_backoff_state.active)
            self.assertAlmostEqual(end_action[0], 0.0, places=6)
            self.assertAlmostEqual(end_action[1], 0.0, places=6)
            self.assertGreater(restart_action[0], 0.0)
            self.assertAlmostEqual(restart_action[1], 0.0, places=6)
        finally:
            env.close()

    def test_personal_space_backoff_cancels_unsafe_session_without_lateral_reselection(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=1463)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            world_frame = env._build_world_frame(force=True)
            call_index = {"value": 0}

            def safety_side_effect(_env, _world_frame, target_idx, direction_xy):
                del target_idx
                direction_xy = np.asarray(direction_xy, dtype=np.float32)
                direction_xy = direction_xy / np.linalg.norm(direction_xy)
                call_number = int(call_index["value"])
                call_index["value"] += 1
                if call_number == 0:
                    return True, 0.2, 1.0, 1.0
                if call_number == 1:
                    return False, float("-inf"), float("-inf"), float("-inf")
                return False, float("-inf"), float("-inf"), float("-inf")

            with patch.object(env, "_log_event") as log_mock, patch(
                "museum_env.env_control.is_robot_backoff_direction_safe",
                side_effect=safety_side_effect,
            ):
                env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )
                second_action = env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )

            self.assertEqual(call_index["value"], 2)
            self.assertEqual(log_mock.call_count, 2)
            self.assertIn("direction=away", log_mock.call_args_list[0].args[0])
            self.assertEqual(log_mock.call_args_list[1].args[0], ">>> Robot personal-space backoff ended.")
            self.assertFalse(env.personal_space_backoff_state.active)
            self.assertEqual(env.personal_space_backoff_state.direction_label, None)
            self.assertAlmostEqual(second_action[0], 0.0, places=6)
            self.assertAlmostEqual(second_action[1], 0.0, places=6)
        finally:
            env.close()

    def test_personal_space_backoff_logs_exit_once_when_callback_suppresses_backoff(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=1465)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            world_frame = env._build_world_frame(force=True)

            with patch.object(env, "_log_event") as log_mock, patch(
                "museum_env.env_control.raycast_hit_distance",
                return_value=1.0,
            ):
                env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )
                env.robot.callback_active = True
                env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )
                env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )

            self.assertEqual(log_mock.call_count, 2)
            self.assertEqual(log_mock.call_args_list[1].args[0], ">>> Robot personal-space backoff ended.")
        finally:
            env.close()

    def test_personal_space_backoff_logs_exit_once_when_protected_stop_suppresses_backoff(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=1466)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            self._set_robot_and_human_poses(
                env,
                robot_pose=(5.0, 5.0, 0.0),
                human_poses=((4.6, 5.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            world_frame = env._build_world_frame(force=True)

            with patch.object(env, "_log_event") as log_mock, patch(
                "museum_env.env_control.raycast_hit_distance",
                return_value=1.0,
            ):
                env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )
                env.robot.mode = "stop"
                env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )
                env_control.apply_robot_personal_space_backoff_if_needed(
                    env,
                    np.zeros(3, dtype=np.float32),
                    world_frame,
                )

            self.assertEqual(log_mock.call_count, 2)
            self.assertEqual(log_mock.call_args_list[1].args[0], ">>> Robot personal-space backoff ended.")
        finally:
            env.close()

    def test_transit_follow_front_sector_rejoin_is_suppressed_when_nearest_front_human_is_too_close(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=147)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 99
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.5, 0.0, 0.0), (1.0, 0.2, 0.0)),
            )
            self._invalidate_observation_cache(env)

            with patch("museum_env.env_control.raycast_hit_distance", return_value=1.0):
                _, _, _, _, info = env.step(None)

            self.assertFalse(info["events"]["callback_triggered"])
            self.assertIsNone(env._following_callback_override_target_idx)
            self.assertFalse(env.robot.callback_active)
            self.assertEqual(info["robot"]["mode"], "move")
            self.assertAlmostEqual(info["robot"]["action"]["vx"], -0.5, places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], 0.0, places=6)
        finally:
            env.close()

    def test_transit_follow_front_sector_rejoin_is_suppressed_inside_callback_trigger_distance(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=148)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 99
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.9, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, info = env.step(None)

            self.assertFalse(info["events"]["callback_triggered"])
            self.assertFalse(env.robot.callback_active)
            self.assertIsNone(info["robot"]["callback_phase"])
            self.assertAlmostEqual(info["robot"]["action"]["vx"], float(baseline_action[0]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], float(baseline_action[1]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], float(baseline_action[2]), places=6)
        finally:
            env.close()

    def test_transit_follow_front_sector_rejoin_is_suppressed_at_callback_trigger_distance(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=149)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 99
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((2.0, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)
            baseline_action, _, _, _ = env.robot._waypoint_action((0.0, 0.0, 0.0))

            _, _, _, _, info = env.step(None)

            self.assertFalse(info["events"]["callback_triggered"])
            self.assertFalse(env.robot.callback_active)
            self.assertIsNone(info["robot"]["callback_phase"])
            self.assertAlmostEqual(info["robot"]["action"]["vx"], float(baseline_action[0]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["vy"], float(baseline_action[1]), places=6)
            self.assertAlmostEqual(info["robot"]["action"]["yaw_rate"], float(baseline_action[2]), places=6)
        finally:
            env.close()

    def test_transit_follow_front_sector_rejoin_still_triggers_above_callback_trigger_distance(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=149)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 99
            env.following_callback_cue_steps = 2
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((2.01, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, info = env.step(None)

            self.assertTrue(info["events"]["callback_triggered"])
            self.assertEqual(info["robot"]["mode"], "callback")
            self.assertEqual(env.robot.callback_target_idx, 0)
        finally:
            env.close()

    def test_transit_follow_far_callback_still_works_when_nearest_front_human_is_too_close(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=150)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.5, 0.0, 0.0), (0.0, 4.1, 0.0)),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, info = env.step(None)

            self.assertTrue(info["events"]["callback_triggered"])
            self.assertEqual(info["robot"]["mode"], "callback")
            self.assertEqual(env.robot.callback_target_idx, 1)
        finally:
            env.close()

    def test_transit_follow_front_sector_triggers_immediate_callback_for_nearest_human(self):
        env = self._make_env(n_humans=3)
        try:
            env.reset(seed=136)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 99
            env.following_callback_cue_steps = 2
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((2.6, 0.4, 0.0), (2.1, 0.0, 0.0), (0.0, 4.5, 0.0)),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, info = env.step(None)

            self.assertTrue(info["events"]["callback_triggered"])
            self.assertEqual(info["robot"]["mode"], "callback")
            self.assertEqual(info["robot"]["callback_phase"], "cue")
            self.assertEqual(env.robot.callback_target_idx, 1)
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP], 1)
        finally:
            env.close()

    def test_transit_follow_front_sector_callback_wait_episode_resets_after_sector_clears(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=137)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 3
            env.following_callback_cue_steps = 1
            env.following_callback_resume_grace_steps = 0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((2.1, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, first_trigger_info = env.step(None)
            self.assertTrue(first_trigger_info["events"]["callback_triggered"])
            self.assertEqual(first_trigger_info["robot"]["mode"], "callback")

            _, _, _, _, completed_info = env.step(None)
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertFalse(env.robot.callback_active)

            _, _, _, _, blocked_info = env.step(None)
            self.assertFalse(blocked_info["events"]["callback_triggered"])
            self.assertEqual(blocked_info["robot"]["mode"], "move")

            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 1.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, cleared_info = env.step(None)
            self.assertFalse(cleared_info["events"]["callback_triggered"])
            self.assertEqual(cleared_info["robot"]["mode"], "move")

            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((2.1, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, second_trigger_info = env.step(None)
            self.assertTrue(second_trigger_info["events"]["callback_triggered"])
            self.assertEqual(second_trigger_info["robot"]["mode"], "callback")
        finally:
            env.close()

    def test_transit_follow_front_sector_respects_custom_callback_trigger_distance(self):
        env = self._make_env(n_humans=1, callback_trigger_distance_meters=3.0)
        try:
            env.reset(seed=151)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 99
            env.following_callback_cue_steps = 1
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((2.5, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, blocked_info = env.step(None)

            self.assertFalse(blocked_info["events"]["callback_triggered"])
            self.assertEqual(blocked_info["robot"]["mode"], "move")

            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((3.1, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)

            self.assertTrue(triggered_info["events"]["callback_triggered"])
            self.assertEqual(triggered_info["robot"]["mode"], "callback")
            self.assertEqual(env.robot.callback_target_idx, 0)
        finally:
            env.close()

    def test_transit_follow_callback_triggers_for_farthest_human_after_wait_threshold(self):
        env = self._make_env(n_humans=2)
        try:
            env.reset(seed=133)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 3
            env.following_callback_cue_steps = 2
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((0.0, 3.6, 0.0), (0.0, 4.2, 0.0)),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, first_info = env.step(None)
            self.assertEqual(first_info["robot"]["mode"], "stop")
            self.assertFalse(first_info["events"]["callback_triggered"])

            _, _, _, _, second_info = env.step(None)
            self.assertEqual(second_info["robot"]["mode"], "stop")
            self.assertFalse(second_info["events"]["callback_triggered"])

            _, _, _, _, third_info = env.step(None)
            self.assertTrue(third_info["events"]["callback_triggered"])
            self.assertEqual(third_info["robot"]["mode"], "callback")
            self.assertEqual(env.robot.callback_target_idx, 1)
            self.assertEqual(third_info["robot"]["callback_phase"], "turn")
            self.assertEqual(env._label_scene_option.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP], 1)
        finally:
            env.close()

    def test_transit_follow_callback_completes_and_does_not_repeat_in_same_wait_episode(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=134)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 2
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)
            self.assertTrue(triggered_info["events"]["callback_triggered"])
            self.assertEqual(triggered_info["robot"]["mode"], "callback")
            self.assertTrue(env.robot.callback_active)
            self.assertEqual(triggered_info["robot"]["emotion"], "sad")

            _, completed_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertFalse(completed_info["events"]["callback_success"])
            self.assertFalse(completed_info["events"]["callback_ignored"])
            self.assertFalse(env.robot.callback_active)
            self.assertEqual(completed_info["robot"]["mode"], "move")
            self.assertEqual(completed_info["robot"]["emotion"], "natural")

            _, _, _, _, waiting_again_info = env.step(None)
            self.assertFalse(waiting_again_info["events"]["callback_triggered"])
            self.assertFalse(env.robot.callback_active)
        finally:
            env.close()

    def test_transit_follow_callback_can_trigger_again_after_distance_recovers(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=135)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            env.following_callback_resume_grace_steps = 0
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, first_trigger_info = env.step(None)
            self.assertTrue(first_trigger_info["events"]["callback_triggered"])

            _, first_complete_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(first_complete_info["events"]["callback_completed"])

            _, _, _, _, same_episode_wait_info = env.step(None)
            self.assertFalse(same_episode_wait_info["events"]["callback_triggered"])
            self.assertFalse(env.robot.callback_active)

            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.0260605, 2.8190778, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, recovered_info = env.step(None)
            self.assertFalse(recovered_info["events"]["callback_triggered"])
            self.assertEqual(recovered_info["robot"]["mode"], "move")

            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, second_trigger_info = env.step(None)
            self.assertTrue(second_trigger_info["events"]["callback_triggered"])
            self.assertEqual(second_trigger_info["robot"]["mode"], "callback")
        finally:
            env.close()

    def test_transit_follow_callback_completion_grants_resume_grace_period(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=152)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 99
            env.following_callback_cue_steps = 1
            env.following_callback_resume_grace_steps = 3
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((2.1, 0.0, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)
            self.assertTrue(triggered_info["events"]["callback_triggered"])
            self.assertEqual(triggered_info["robot"]["mode"], "callback")

            _, _, _, _, completed_info = env.step(None)
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertEqual(env._following_callback_resume_grace_steps_remaining, 3)

            for expected_remaining in (2, 1, 0):
                _, _, _, _, grace_info = env.step(None)
                self.assertFalse(grace_info["events"]["callback_triggered"])
                self.assertFalse(env.robot.callback_active)
                self.assertEqual(grace_info["robot"]["mode"], "move")
                self.assertEqual(
                    env._following_callback_resume_grace_steps_remaining,
                    expected_remaining,
                )
        finally:
            env.close()

    def test_transit_follow_callback_resume_grace_resets_wait_episode(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=153)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            env.following_callback_resume_grace_steps = 2
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((3.4, 3.6, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, first_trigger_info = env.step(None)
            self.assertTrue(first_trigger_info["events"]["callback_triggered"])

            _, completed_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(completed_info["events"]["callback_completed"])

            env.following_callback_wait_steps = 3
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((3.4, 3.6, 0.0),),
            )
            self._invalidate_observation_cache(env)

            for expected_remaining in (1, 0):
                _, _, _, _, grace_info = env.step(None)
                self.assertFalse(grace_info["events"]["callback_triggered"])
                self.assertEqual(grace_info["robot"]["mode"], "move")
                self.assertEqual(
                    env._following_callback_resume_grace_steps_remaining,
                    expected_remaining,
                )
                self.assertEqual(env._following_wait_elapsed_steps, 0)

            _, _, _, _, post_grace_stop1 = env.step(None)
            self.assertFalse(post_grace_stop1["events"]["callback_triggered"])
            self.assertEqual(post_grace_stop1["robot"]["mode"], "stop")
            self.assertEqual(env._following_wait_elapsed_steps, 1)

            _, _, _, _, post_grace_stop2 = env.step(None)
            self.assertFalse(post_grace_stop2["events"]["callback_triggered"])
            self.assertEqual(post_grace_stop2["robot"]["mode"], "stop")
            self.assertEqual(env._following_wait_elapsed_steps, 2)

            _, _, _, _, second_trigger_info = env.step(None)
            self.assertTrue(second_trigger_info["events"]["callback_triggered"])
            self.assertEqual(second_trigger_info["robot"]["mode"], "callback")
        finally:
            env.close()

    def test_callback_response_defaults_use_updated_nd_split(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=136)
            self.assertEqual(
                env.callback_response_profile_probs_by_mode[HumanMode.IMPATIENT][HumanProfile.NORMAL]["rejoin"],
                0.50,
            )
            self.assertEqual(
                env.callback_response_profile_probs_by_mode[HumanMode.IMPATIENT][HumanProfile.NORMAL]["ignore"],
                0.50,
            )
            self.assertEqual(
                env.callback_response_profile_probs_by_mode[HumanMode.IMPATIENT][HumanProfile.NEURODIVERGENT]["rejoin"],
                0.30,
            )
            self.assertEqual(
                env.callback_response_profile_probs_by_mode[HumanMode.IMPATIENT][HumanProfile.NEURODIVERGENT]["ignore"],
                0.70,
            )
            self.assertEqual(
                env.callback_response_profile_probs[HumanProfile.NEURODIVERGENT]["rejoin"],
                0.50,
            )
            self.assertEqual(
                env.callback_response_profile_probs[HumanProfile.NEURODIVERGENT]["ignore"],
                0.50,
            )
            self.assertEqual(
                env.callback_response_profile_probs_by_mode[HumanMode.OVERWHELMED][HumanProfile.NORMAL]["rejoin"],
                0.30,
            )
            self.assertEqual(
                env.callback_response_profile_probs_by_mode[HumanMode.OVERWHELMED][HumanProfile.NORMAL]["ignore"],
                0.70,
            )
            self.assertEqual(
                env.callback_response_profile_probs_by_mode[HumanMode.OVERWHELMED][HumanProfile.NEURODIVERGENT]["rejoin"],
                0.10,
            )
            self.assertEqual(
                env.callback_response_profile_probs_by_mode[HumanMode.OVERWHELMED][HumanProfile.NEURODIVERGENT]["ignore"],
                0.90,
            )
        finally:
            env.close()

    def test_callback_completion_rejoins_distracted_normal_human_in_follow_stage(self):
        env = self._make_env(
            n_humans=1,
            callback_rejoin_prob_normal=1.0,
            callback_ignore_prob_normal=0.0,
        )
        try:
            env.reset(seed=137)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.set_mode(HumanMode.DISTRACTED)
            human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
            human.distracted_recovery_mode = HumanMode.FOLLOWING
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)
            self.assertTrue(triggered_info["events"]["callback_triggered"])
            self.assertEqual(env.robot.callback_target_idx, 0)

            _, completed_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertTrue(completed_info["events"]["callback_success"])
            self.assertFalse(completed_info["events"]["callback_ignored"])
            self.assertEqual(human.mode, HumanMode.FOLLOWING)
        finally:
            env.close()

    def test_callback_completion_rejoins_impatient_normal_human_in_follow_stage(self):
        env = self._make_env(
            n_humans=1,
            callback_impatient_rejoin_prob_normal=1.0,
            callback_impatient_ignore_prob_normal=0.0,
        )
        try:
            env.reset(seed=1371)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.start_impatient(recovery_mode=HumanMode.FOLLOWING)
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)
            self.assertTrue(triggered_info["events"]["callback_triggered"])
            self.assertEqual(env.robot.callback_target_idx, 0)

            _, completed_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertTrue(completed_info["events"]["callback_success"])
            self.assertFalse(completed_info["events"]["callback_ignored"])
            self.assertEqual(human.mode, HumanMode.FOLLOWING)
            self.assertAlmostEqual(human.max_speed, human.base_max_speed, places=6)
        finally:
            env.close()

    def test_callback_completion_can_be_ignored_by_distracted_normal_human(self):
        env = self._make_env(
            n_humans=1,
            callback_rejoin_prob_normal=0.0,
            callback_ignore_prob_normal=1.0,
        )
        try:
            env.reset(seed=138)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.set_mode(HumanMode.DISTRACTED)
            human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
            human.distracted_recovery_mode = HumanMode.FOLLOWING
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)
            self.assertTrue(triggered_info["events"]["callback_triggered"])

            _, completed_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertFalse(completed_info["events"]["callback_success"])
            self.assertTrue(completed_info["events"]["callback_ignored"])
            self.assertEqual(human.mode, HumanMode.DISTRACTED)
        finally:
            env.close()

    def test_callback_completion_can_be_ignored_by_impatient_normal_human(self):
        env = self._make_env(
            n_humans=1,
            callback_impatient_rejoin_prob_normal=0.0,
            callback_impatient_ignore_prob_normal=1.0,
        )
        try:
            env.reset(seed=1381)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.start_impatient(recovery_mode=HumanMode.FOLLOWING)
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)
            self.assertTrue(triggered_info["events"]["callback_triggered"])

            _, completed_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertFalse(completed_info["events"]["callback_success"])
            self.assertTrue(completed_info["events"]["callback_ignored"])
            self.assertEqual(human.mode, HumanMode.IMPATIENT)
        finally:
            env.close()

    def test_callback_response_rejoins_to_listening_when_listening_stage_is_active(self):
        env = self._make_env(
            n_humans=1,
            callback_rejoin_prob_normal=1.0,
            callback_ignore_prob_normal=0.0,
        )
        try:
            env.reset(seed=139)
            env.listening_state.enter_wait(False)
            env.robot.callback_target_idx = 0
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.set_mode(HumanMode.DISTRACTED)
            human.distracted_source = DISTRACTED_SOURCE_LISTENING
            human.distracted_recovery_mode = HumanMode.LISTENING
            events = StepEvents()

            env_control.apply_callback_response_if_needed(env, events)

            self.assertTrue(events.callback_success)
            self.assertFalse(events.callback_ignored)
            self.assertEqual(human.mode, HumanMode.LISTENING)
        finally:
            env.close()

    def test_callback_response_rejoins_impatient_human_to_listening_when_listening_stage_is_active(self):
        env = self._make_env(
            n_humans=1,
            callback_impatient_rejoin_prob_normal=1.0,
            callback_impatient_ignore_prob_normal=0.0,
        )
        try:
            env.reset(seed=1391)
            env.listening_state.enter_wait(False)
            env.robot.callback_target_idx = 0
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.start_impatient(recovery_mode=HumanMode.LISTENING)
            events = StepEvents()

            env_control.apply_callback_response_if_needed(env, events)

            self.assertTrue(events.callback_success)
            self.assertFalse(events.callback_ignored)
            self.assertEqual(human.mode, HumanMode.LISTENING)
            self.assertAlmostEqual(human.max_speed, human.base_max_speed, places=6)
        finally:
            env.close()

    def test_callback_response_ignores_non_distracted_target_human(self):
        env = self._make_env(
            n_humans=1,
            callback_rejoin_prob_normal=1.0,
            callback_ignore_prob_normal=0.0,
        )
        try:
            env.reset(seed=140)
            env.robot.callback_target_idx = 0
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.set_mode(HumanMode.FOLLOWING)
            events = StepEvents()

            env_control.apply_callback_response_if_needed(env, events)

            self.assertFalse(events.callback_success)
            self.assertFalse(events.callback_ignored)
            self.assertEqual(human.mode, HumanMode.FOLLOWING)
        finally:
            env.close()

    def test_callback_response_ignore_preserves_impatient_runtime(self):
        env = self._make_env(
            n_humans=1,
            callback_impatient_rejoin_prob_normal=0.0,
            callback_impatient_ignore_prob_normal=1.0,
        )
        try:
            env.reset(seed=1401)
            env.robot.callback_target_idx = 0
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.start_impatient(recovery_mode=HumanMode.FOLLOWING)
            human.impatient_timer = 17
            impatient_speed = float(human.max_speed)
            events = StepEvents()

            env_control.apply_callback_response_if_needed(env, events)

            self.assertFalse(events.callback_success)
            self.assertTrue(events.callback_ignored)
            self.assertEqual(human.mode, HumanMode.IMPATIENT)
            self.assertEqual(human.impatient_timer, 17)
            self.assertAlmostEqual(human.max_speed, impatient_speed, places=6)
            self.assertEqual(human.impatient_recovery_mode, HumanMode.FOLLOWING)
        finally:
            env.close()

    def test_callback_completion_rejoins_overwhelmed_normal_human_in_follow_stage(self):
        env = self._make_env(
            n_humans=1,
            callback_overwhelmed_rejoin_prob_normal=1.0,
            callback_overwhelmed_ignore_prob_normal=0.0,
        )
        try:
            env.reset(seed=141)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.set_mode(HumanMode.OVERWHELMED)
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)
            self.assertTrue(triggered_info["events"]["callback_triggered"])

            _, completed_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertTrue(completed_info["events"]["callback_success"])
            self.assertFalse(completed_info["events"]["callback_ignored"])
            self.assertEqual(human.mode, HumanMode.FOLLOWING)
        finally:
            env.close()

    def test_callback_completion_can_be_ignored_by_overwhelmed_normal_human(self):
        env = self._make_env(
            n_humans=1,
            callback_overwhelmed_rejoin_prob_normal=0.0,
            callback_overwhelmed_ignore_prob_normal=1.0,
        )
        try:
            env.reset(seed=142)
            env.follow_phase = "transit_follow"
            env.robot.listen_done = True
            env.following_callback_wait_steps = 1
            env.following_callback_cue_steps = 1
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.set_mode(HumanMode.OVERWHELMED)
            self._set_robot_and_human_poses(
                env,
                robot_pose=(0.0, 0.0, 0.0),
                human_poses=((1.3680806, 3.7587705, 0.0),),
            )
            self._invalidate_observation_cache(env)

            _, _, _, _, triggered_info = env.step(None)
            self.assertTrue(triggered_info["events"]["callback_triggered"])

            _, completed_info = self._run_until(
                env,
                lambda _info: _info["events"]["callback_completed"],
                200,
            )
            self.assertTrue(completed_info["events"]["callback_completed"])
            self.assertFalse(completed_info["events"]["callback_success"])
            self.assertTrue(completed_info["events"]["callback_ignored"])
            self.assertEqual(human.mode, HumanMode.OVERWHELMED)
        finally:
            env.close()

    def test_callback_response_rejoins_overwhelmed_human_to_listening_when_listening_stage_is_active(self):
        env = self._make_env(
            n_humans=1,
            callback_overwhelmed_rejoin_prob_normal=1.0,
            callback_overwhelmed_ignore_prob_normal=0.0,
        )
        try:
            env.reset(seed=143)
            env.listening_state.enter_wait(False)
            env.robot.callback_target_idx = 0
            human = env.humans[0]
            human.set_profile(HumanProfile.NORMAL)
            human.set_mode(HumanMode.OVERWHELMED)
            events = StepEvents()

            env_control.apply_callback_response_if_needed(env, events)

            self.assertTrue(events.callback_success)
            self.assertFalse(events.callback_ignored)
            self.assertEqual(human.mode, HumanMode.LISTENING)
        finally:
            env.close()

    def test_overwhelmed_duration_defaults_use_expected_steps(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        self.assertEqual(
            human.overwhelmed_leave_duration,
            round(human.max_overwhelmed_leave_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS),
        )
        self.assertEqual(
            human.overwhelmed_pause_duration,
            round(human.max_overwhelmed_pause_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS),
        )

    def test_overwhelmed_transitions_from_backoff_to_leave_to_pause(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.OVERWHELMED)
        human.overwhelmed_stage = "backoff"
        human.overwhelmed_backoff_start_xy = np.array([0.0, 0.0], dtype=np.float32)
        human.overwhelmed_leave_dir = np.array([1.0, 0.0], dtype=np.float32)
        human.overwhelmed_recovery_mode = HumanMode.FOLLOWING

        backoff_pose = (human.overwhelmed_backoff_dist, 0.0, 0.0)
        backoff_data = self._make_pose_data(backoff_pose)
        leave_data = self._make_pose_data((0.0, 0.0, 0.0))

        backoff_action = human.step(None, backoff_data, {})
        self.assertEqual(human.overwhelmed_stage, "leave")
        np.testing.assert_allclose(backoff_action, np.zeros(3, dtype=np.float32), atol=1e-6)

        human.overwhelmed_leave_timer = human.overwhelmed_leave_duration - 1
        leave_action = human.step(None, leave_data, {})
        self.assertEqual(human.mode, HumanMode.OVERWHELMED)
        self.assertEqual(human.overwhelmed_stage, "pause")
        np.testing.assert_allclose(
            leave_action[:2],
            np.array([human.max_speed, 0.0], dtype=np.float32),
            atol=1e-6,
        )

        pause_action = human.step(None, leave_data, {})
        self.assertEqual(human.mode, HumanMode.OVERWHELMED)
        self.assertEqual(human.overwhelmed_pause_timer, 1)
        np.testing.assert_allclose(pause_action, np.zeros(3, dtype=np.float32), atol=1e-6)

    def test_overwhelmed_backoff_uses_compose_move_velocity_without_hr_or_repulsion(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.OVERWHELMED)
        human.overwhelmed_stage = "backoff"
        human.overwhelmed_backoff_start_xy = np.array([0.0, 0.0], dtype=np.float32)
        human.overwhelmed_leave_dir = np.array([1.0, 0.0], dtype=np.float32)
        adjusted_v_xy = np.array([0.2, 0.1], dtype=np.float32)

        data = self._make_pose_data((0.0, 0.0, 0.0))
        with patch.object(human, "_compose_move_velocity", return_value=adjusted_v_xy) as compose_mock:
            action = human.step(None, data, {})

        compose_mock.assert_called_once()
        self.assertNotIn("repulsion_xy", compose_mock.call_args.kwargs)
        self.assertNotIn("robot_xy", compose_mock.call_args.kwargs)
        self.assertNotIn("hr_distance_min", compose_mock.call_args.kwargs)
        self.assertNotIn("hr_distance_max", compose_mock.call_args.kwargs)
        np.testing.assert_allclose(action[:2], adjusted_v_xy, atol=1e-6)

    def test_overwhelmed_leave_uses_compose_move_velocity_without_hr_or_repulsion(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.OVERWHELMED)
        human.overwhelmed_stage = "leave"
        human.overwhelmed_leave_dir = np.array([1.0, 0.0], dtype=np.float32)
        adjusted_v_xy = np.array([0.2, 0.1], dtype=np.float32)

        data = self._make_pose_data((0.0, 0.0, 0.0))
        with patch.object(human, "_compose_move_velocity", return_value=adjusted_v_xy) as compose_mock:
            action = human.step(None, data, {})

        compose_mock.assert_called_once()
        self.assertNotIn("repulsion_xy", compose_mock.call_args.kwargs)
        self.assertNotIn("robot_xy", compose_mock.call_args.kwargs)
        self.assertNotIn("hr_distance_min", compose_mock.call_args.kwargs)
        self.assertNotIn("hr_distance_max", compose_mock.call_args.kwargs)
        np.testing.assert_allclose(action[:2], adjusted_v_xy, atol=1e-6)

    def test_overwhelmed_pause_recovers_to_following_and_listening(self):
        for recovery_mode in (HumanMode.FOLLOWING, HumanMode.LISTENING):
            with self.subTest(recovery_mode=recovery_mode):
                human = Human("person1", "person1", 0, max_speed=1.0)
                human.set_mode(HumanMode.OVERWHELMED)
                human.overwhelmed_stage = "pause"
                human.overwhelmed_recovery_mode = recovery_mode
                human.overwhelmed_pause_timer = human.overwhelmed_pause_duration - 1
                data = self._make_pose_data((0.0, 0.0, 0.0))

                action = human.step(None, data, {})

                self.assertEqual(human.mode, recovery_mode)
                np.testing.assert_allclose(action, np.zeros(3, dtype=np.float32), atol=1e-6)

    def test_post_explanation_yield_uses_behavior_kind_through_step(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.FOLLOWING)

        pose = (2.0, 2.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "behavior_kind": "post_explanation_yield",
            "target_xy": np.array([4.0, 2.0], dtype=np.float32),
            "robot_pose": (1.0, 2.0, 0.0),
            "robot_xy": np.array([1.0, 2.0], dtype=np.float32),
            "repulsion": np.zeros(2, dtype=np.float32),
        }

        action = human.step(None, data, ctx)

        self.assertGreater(float(action[0]), 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, places=6)
        np.testing.assert_allclose(human.current_waypoint, ctx["target_xy"], atol=1e-6)

    def test_post_explanation_listening_anchor_uses_behavior_kind_through_step(self):
        human = Human("person1", "person1", 0, max_speed=1.0)
        human.set_mode(HumanMode.LISTENING)

        pose = (1.0, 0.0, 0.0)
        data = self._make_pose_data(pose)
        ctx = {
            "behavior_kind": "post_explanation_listening_anchor",
            "robot_xy": np.array([0.0, 0.0], dtype=np.float32),
            "robot_yaw": 0.0,
            "repulsion": np.zeros(2, dtype=np.float32),
            "listen_radius": 1.0,
            "listening_sector_half_angle": np.deg2rad(70.0),
            "anchor_robot_xy": np.array([0.0, 1.0], dtype=np.float32),
            "anchor_robot_yaw": 0.0,
            "live_robot_xy": np.array([0.0, 0.0], dtype=np.float32),
        }

        action = human.step(None, data, ctx)

        expected_yaw = float(np.arctan2(1.0, -1.0))
        expected_yaw_rate = HUMAN_YAW_RATE_GAIN * expected_yaw
        np.testing.assert_allclose(action[2], expected_yaw_rate, atol=1e-5)

    def test_full_flow_reaches_final_listen_ready(self):
        env = self._make_env(n_humans=1)
        try:
            env.reset(seed=18)
            env.listen_intro_delay_steps = 1
            env.listen_wait_steps = 2
            env.robot.v_max = 3.0
            env.max_steps = 60000

            _, info = self._run_until(
                env,
                lambda info: info["events"]["final_listen_ready"],
                30000,
            )

            self.assertEqual(info["episode"]["terminated_reason"], "final_listen_ready")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
