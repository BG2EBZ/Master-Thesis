from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (PACKAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.run_env as run_env


_STEP_INFO = {
    "events": {
        "question_started": False,
        "question_completed": False,
        "final_listen_ready": False,
    },
    "episode": {
        "step": 0,
        "terminated_reason": None,
    },
    "phase": {
        "follow": "idle",
        "listen": "idle",
    },
    "robot": {
        "mode": "move",
        "dist_to_goal": 0.0,
        "speaker_active": False,
    },
    "crowd": {
        "human_robot_distance": [],
    },
}


class _FakeClock:
    def __init__(self, after_sleep_hooks=None):
        self.current = 0.0
        self.after_sleep_hooks = list(after_sleep_hooks or [])
        self.sleep_calls = []

    def advance(self, seconds):
        self.current += float(seconds)

    def perf_counter(self):
        return self.current

    def sleep(self, seconds):
        self.sleep_calls.append(float(seconds))
        self.current += float(seconds)
        if self.after_sleep_hooks:
            self.after_sleep_hooks.pop(0)()


class _FakeBaseEnv:
    def __init__(self, dt):
        self.dt = float(dt)
        self.viewer = object()


class _FakeEnv:
    def __init__(self, dt, work_durations, clock):
        self.unwrapped = _FakeBaseEnv(dt)
        self._work_durations = list(work_durations)
        self._clock = clock
        self.render_calls = 0
        self.step_calls = 0

    def reset(self):
        self.step_calls = 0
        return None, {}

    def step(self, _action):
        work_duration = self._work_durations[self.step_calls]
        self._clock.advance(work_duration)
        self.step_calls += 1
        info = deepcopy(_STEP_INFO)
        info["episode"]["step"] = self.step_calls
        return None, 0.0, False, False, info

    def render(self):
        self.render_calls += 1


class TestEnvRealtimePacingTests(unittest.TestCase):
    def _run_loop(
        self,
        *,
        realtime,
        sleep_scale,
        dt,
        work_durations,
        after_sleep_hooks=None,
        controller_holder=None,
    ):
        clock = _FakeClock(after_sleep_hooks=after_sleep_hooks)
        env = _FakeEnv(dt=dt, work_durations=work_durations, clock=clock)
        controller_holder = {} if controller_holder is None else controller_holder
        real_pause_controller = run_env._PauseController

        def build_controller(*args, **kwargs):
            controller = real_pause_controller(*args, **kwargs)
            controller_holder["controller"] = controller
            return controller

        with ExitStack() as stack:
            stack.enter_context(patch("scripts.run_env.time.perf_counter", side_effect=clock.perf_counter))
            stack.enter_context(patch("scripts.run_env.time.sleep", side_effect=clock.sleep))
            stack.enter_context(patch("scripts.run_env._PauseController", side_effect=build_controller))
            run_env._run_loop(
                env,
                max_steps=len(work_durations),
                print_every=0,
                realtime=realtime,
                sleep_scale=sleep_scale,
                print_human_robot_distance_periodically=False,
            )

        return env, clock, controller_holder.get("controller")

    def test_pause_controller_handles_pause_and_speed_hotkeys_with_bounds(self):
        controller = run_env._PauseController(
            toggle_key=run_env.mujoco.viewer.glfw.KEY_P,
            increase_keys=(
                run_env.mujoco.viewer.glfw.KEY_EQUAL,
                run_env.mujoco.viewer.glfw.KEY_KP_ADD,
            ),
            decrease_keys=(
                run_env.mujoco.viewer.glfw.KEY_MINUS,
                run_env.mujoco.viewer.glfw.KEY_KP_SUBTRACT,
            ),
            initial_speed_multiplier=1.0,
        )

        controller.on_key(run_env.mujoco.viewer.glfw.KEY_P)
        self.assertTrue(controller.paused)
        controller.on_key(run_env.mujoco.viewer.glfw.KEY_P)
        self.assertFalse(controller.paused)

        controller.on_key(run_env.mujoco.viewer.glfw.KEY_EQUAL)
        self.assertEqual(controller.speed_multiplier, 2.0)
        controller.on_key(run_env.mujoco.viewer.glfw.KEY_KP_ADD)
        self.assertEqual(controller.speed_multiplier, 4.0)
        controller.on_key(run_env.mujoco.viewer.glfw.KEY_EQUAL)
        self.assertEqual(controller.speed_multiplier, 8.0)
        controller.on_key(run_env.mujoco.viewer.glfw.KEY_KP_ADD)
        self.assertEqual(controller.speed_multiplier, 8.0)

        controller = run_env._PauseController(
            toggle_key=run_env.mujoco.viewer.glfw.KEY_P,
            increase_keys=(
                run_env.mujoco.viewer.glfw.KEY_EQUAL,
                run_env.mujoco.viewer.glfw.KEY_KP_ADD,
            ),
            decrease_keys=(
                run_env.mujoco.viewer.glfw.KEY_MINUS,
                run_env.mujoco.viewer.glfw.KEY_KP_SUBTRACT,
            ),
            initial_speed_multiplier=1.0,
        )
        controller.on_key(run_env.mujoco.viewer.glfw.KEY_MINUS)
        self.assertEqual(controller.speed_multiplier, 0.5)
        controller.on_key(run_env.mujoco.viewer.glfw.KEY_KP_SUBTRACT)
        self.assertEqual(controller.speed_multiplier, 0.25)
        controller.on_key(run_env.mujoco.viewer.glfw.KEY_MINUS)
        self.assertEqual(controller.speed_multiplier, 0.125)
        controller.on_key(run_env.mujoco.viewer.glfw.KEY_KP_SUBTRACT)
        self.assertEqual(controller.speed_multiplier, 0.125)

    def test_realtime_demo_sleeps_to_match_sim_time(self):
        env, clock, controller = self._run_loop(
            realtime=True,
            sleep_scale=1.0,
            dt=0.5,
            work_durations=[0.1, 0.2],
        )

        self.assertEqual(env.step_calls, 2)
        self.assertEqual(env.render_calls, 2)
        self.assertIsNotNone(controller)
        self.assertEqual(len(clock.sleep_calls), 2)
        self.assertAlmostEqual(clock.sleep_calls[0], 0.4, places=7)
        self.assertAlmostEqual(clock.sleep_calls[1], 0.3, places=7)

    def test_realtime_demo_skips_sleep_when_already_slower_than_real_time(self):
        env, clock, _ = self._run_loop(
            realtime=True,
            sleep_scale=1.0,
            dt=0.5,
            work_durations=[0.6],
        )

        self.assertEqual(env.step_calls, 1)
        self.assertEqual(env.render_calls, 1)
        self.assertEqual(clock.sleep_calls, [])

    def test_sleep_scale_controls_pacing_target(self):
        faster_env, faster_clock, _ = self._run_loop(
            realtime=True,
            sleep_scale=0.5,
            dt=0.5,
            work_durations=[0.1],
        )
        slower_env, slower_clock, _ = self._run_loop(
            realtime=True,
            sleep_scale=2.0,
            dt=0.5,
            work_durations=[0.1],
        )

        self.assertEqual(faster_env.step_calls, 1)
        self.assertEqual(slower_env.step_calls, 1)
        self.assertEqual(len(faster_clock.sleep_calls), 1)
        self.assertEqual(len(slower_clock.sleep_calls), 1)
        self.assertAlmostEqual(faster_clock.sleep_calls[0], 0.15, places=7)
        self.assertAlmostEqual(slower_clock.sleep_calls[0], 0.9, places=7)

    def test_speed_up_only_changes_future_pacing(self):
        controller_holder = {}

        def speed_up():
            controller_holder["controller"].on_key(run_env.mujoco.viewer.glfw.KEY_EQUAL)

        env, clock, controller = self._run_loop(
            realtime=True,
            sleep_scale=1.0,
            dt=0.5,
            work_durations=[0.1, 0.1],
            after_sleep_hooks=[speed_up],
            controller_holder=controller_holder,
        )

        self.assertEqual(env.step_calls, 2)
        self.assertEqual(controller.speed_multiplier, 2.0)
        self.assertEqual(len(clock.sleep_calls), 2)
        self.assertAlmostEqual(clock.sleep_calls[0], 0.4, places=7)
        self.assertAlmostEqual(clock.sleep_calls[1], 0.15, places=7)

    def test_slow_down_only_changes_future_pacing(self):
        controller_holder = {}

        def slow_down():
            controller_holder["controller"].on_key(run_env.mujoco.viewer.glfw.KEY_MINUS)

        env, clock, controller = self._run_loop(
            realtime=True,
            sleep_scale=1.0,
            dt=0.5,
            work_durations=[0.1, 0.1],
            after_sleep_hooks=[slow_down],
            controller_holder=controller_holder,
        )

        self.assertEqual(env.step_calls, 2)
        self.assertEqual(controller.speed_multiplier, 0.5)
        self.assertEqual(len(clock.sleep_calls), 2)
        self.assertAlmostEqual(clock.sleep_calls[0], 0.4, places=7)
        self.assertAlmostEqual(clock.sleep_calls[1], 0.9, places=7)

    def test_non_realtime_mode_does_not_sleep(self):
        env, clock, controller = self._run_loop(
            realtime=False,
            sleep_scale=1.0,
            dt=0.5,
            work_durations=[0.1, 0.2],
        )

        self.assertEqual(env.step_calls, 2)
        self.assertEqual(env.render_calls, 0)
        self.assertEqual(clock.sleep_calls, [])
        self.assertIsNone(controller)


if __name__ == "__main__":
    unittest.main()
