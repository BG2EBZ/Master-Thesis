import unittest
from copy import deepcopy
from unittest.mock import patch

import test_env


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
    def __init__(self):
        self.current = 0.0
        self.sleep_calls = []

    def advance(self, seconds):
        self.current += float(seconds)

    def perf_counter(self):
        return self.current

    def sleep(self, seconds):
        self.sleep_calls.append(float(seconds))
        self.current += float(seconds)


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
    def _run_loop(self, *, realtime, sleep_scale, dt, work_durations):
        clock = _FakeClock()
        env = _FakeEnv(dt=dt, work_durations=work_durations, clock=clock)

        with (
            patch("test_env.time.perf_counter", side_effect=clock.perf_counter),
            patch("test_env.time.sleep", side_effect=clock.sleep),
        ):
            test_env._run_loop(
                env,
                max_steps=len(work_durations),
                print_every=0,
                realtime=realtime,
                sleep_scale=sleep_scale,
                print_human_robot_distance_periodically=False,
            )

        return env, clock

    def test_realtime_demo_sleeps_to_match_sim_time(self):
        env, clock = self._run_loop(
            realtime=True,
            sleep_scale=1.0,
            dt=0.5,
            work_durations=[0.1, 0.2],
        )

        self.assertEqual(env.step_calls, 2)
        self.assertEqual(env.render_calls, 2)
        self.assertEqual(len(clock.sleep_calls), 2)
        self.assertAlmostEqual(clock.sleep_calls[0], 0.4, places=7)
        self.assertAlmostEqual(clock.sleep_calls[1], 0.3, places=7)

    def test_realtime_demo_skips_sleep_when_already_slower_than_real_time(self):
        env, clock = self._run_loop(
            realtime=True,
            sleep_scale=1.0,
            dt=0.5,
            work_durations=[0.6],
        )

        self.assertEqual(env.step_calls, 1)
        self.assertEqual(env.render_calls, 1)
        self.assertEqual(clock.sleep_calls, [])

    def test_sleep_scale_controls_pacing_target(self):
        faster_env, faster_clock = self._run_loop(
            realtime=True,
            sleep_scale=0.5,
            dt=0.5,
            work_durations=[0.1],
        )
        slower_env, slower_clock = self._run_loop(
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

    def test_non_realtime_mode_does_not_sleep(self):
        env, clock = self._run_loop(
            realtime=False,
            sleep_scale=1.0,
            dt=0.5,
            work_durations=[0.1, 0.2],
        )

        self.assertEqual(env.step_calls, 2)
        self.assertEqual(env.render_calls, 0)
        self.assertEqual(clock.sleep_calls, [])


if __name__ == "__main__":
    unittest.main()
