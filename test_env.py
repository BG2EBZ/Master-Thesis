import argparse
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import mujoco.viewer
import museum_env.register_env
import numpy as np

DEFAULT_MAX_STEPS = 300000
DEFAULT_RENDER_FPS = 60
DEFAULT_RENDER_EVERY_STEPS = 1
DEFAULT_VIDEO_FPS = 500
DEFAULT_SLEEP_SCALE = 1.0
DEFAULT_RTF_PRINT_EVERY = 2500
DEFAULT_VIDEO_ROOT = "videos"
DEFAULT_NAME_PREFIX = "museum_full_run"


class _PauseController:
    def __init__(self, toggle_key):
        self.toggle_key = int(toggle_key)
        self.paused = False

    def on_key(self, key):
        if int(key) != self.toggle_key:
            return
        self.paused = not self.paused
        status = "paused" if self.paused else "resumed"
        print(f"[control] {status}")


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return parsed


def _build_video_folder(video_root, use_timestamp_subfolder, name_prefix):
    root = Path(video_root)
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = root / f"{name_prefix}_{run_id}"
    else:
        folder = root
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def _summarize_info(step, info):
    phase = info["phase"]
    robot = info["robot"]
    return (
        f"[step {step}] "
        f"robot_mode={robot['mode']} "
        f"follow_phase={phase['follow']} "
        f"listen_phase={phase['listen']} "
        f"dist_to_goal={robot['dist_to_goal']:.2f} "
        f"speaker={robot['speaker_active']} "
    )


def _report_step(step, terminated, truncated, info, base_env=None, wall_start=None, sim_dt=None):
    suffix = ""
    if wall_start is not None and sim_dt is not None:
        real_time = time.perf_counter() - wall_start
        sim_time = step * sim_dt
        suffix = f", sim_time={sim_time:.3f}s, real_time={real_time:.3f}s"

    if info["events"]["question_started"]:
        active_idx = (
            getattr(base_env.listening_state, "question_human_idx", None)
            if base_env is not None
            else None
        )
        timing_mode = (
            getattr(base_env.listening_state, "question_timing_mode", None)
            if base_env is not None
            else None
        )
        person_label = "none" if active_idx is None else str(int(active_idx) + 1)
        print(
            f"[step {step}] question started: timing={timing_mode}, person={person_label}{suffix}"
        )

    if info["events"]["question_completed"]:
        print(
            f"[step {step}] question completed: listen_phase={info['phase']['listen']}{suffix}"
        )

    if info["events"]["final_listen_ready"]:
        print(f"[step {step}] final listen completed{suffix}")

    if terminated:
        print(f"Episode terminated at step {step}{suffix}")
        return True

    if truncated:
        print(f"Episode truncated at step {step}{suffix}")
        return True

    return False


def _print_human_robot_distance(step, info):
    distances = info["crowd"]["human_robot_distance"]
    if len(distances) == 0:
        print(f"[step {step}] hr_distance none")
        return

    listen_phase = info["phase"]["listen"]
    listen_phase_prefix = f"listen_phase={listen_phase} " if listen_phase != "idle" else ""
    parts = [f"person{idx + 1}={float(distance):.2f}" for idx, distance in enumerate(distances)]
    print(f"[step {step}] hr_distance {listen_phase_prefix}{' '.join(parts)}")


def _print_following_crowd_regulation_status(step, info):
    distances = np.asarray(info["crowd"]["human_robot_distance"], dtype=np.float32)
    max_hr_distance = float(np.max(distances)) if distances.size != 0 else 0.0
    print(
        f"[step {step}] robot_mode={info['robot']['mode']} "
        f"max_hr_distance={max_hr_distance:.2f}"
    )


def _print_rtf(tag, steps_done, sim_dt, wall_start):
    real_time = max(1e-9, time.perf_counter() - wall_start)
    sim_time = steps_done * sim_dt
    rtf = sim_time / real_time
    print(
        f"[{tag}] steps={steps_done}, sim_time={sim_time:.3f}s, "
        f"real_time={real_time:.3f}s, rtf={rtf:.3f}"
    )


def _sleep_until_realtime_target(steps_done, sim_dt, wall_start, sleep_scale, wall_time_offset=0.0):
    target_wall_time = wall_start + wall_time_offset + (steps_done * sim_dt * float(sleep_scale))
    remaining = target_wall_time - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)


def _make_env(mode):
    render_mode = None
    if mode == "demo":
        render_mode = "human"
    elif mode == "record":
        render_mode = "rgb_array"
    return gym.make("MuseumEnv-v0", render_mode=render_mode)


def _run_loop(
    env,
    *,
    max_steps,
    print_every,
    realtime=False,
    sleep_scale=1.0,
    print_human_robot_distance_periodically=False,
):
    base_env = env.unwrapped
    obs, info = env.reset()
    del obs, info

    sim_dt = float(base_env.dt)
    wall_start = time.perf_counter()
    wall_time_offset = 0.0
    pause_controller = None

    if realtime and base_env.viewer is None:
        pause_controller = _PauseController(mujoco.viewer.glfw.KEY_P)
        base_env.viewer = mujoco.viewer.launch_passive(
            base_env.model,
            base_env.data,
            key_callback=pause_controller.on_key,
        )
        if base_env.viewer.user_scn is not None:
            base_env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            base_env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        env.render()

    step = 0
    while step < max_steps:
        if pause_controller is not None and pause_controller.paused:
            pause_start = time.perf_counter()
            env.render()
            time.sleep(sim_dt * float(sleep_scale))
            wall_time_offset += time.perf_counter() - pause_start
            continue

        _, _, terminated, truncated, info = env.step(None)
        periodic_print = bool(print_every) and ((step + 1) % print_every == 0)
        if periodic_print and print_human_robot_distance_periodically:
            # _print_human_robot_distance(step + 1, info)
            _print_following_crowd_regulation_status(step + 1, info)
        if periodic_print:
            # print(_summarize_info(step, info))
            _print_rtf("loop", step + 1, sim_dt, wall_start)

        rendered_this_step = False
        if realtime:
            if (step + 1) % DEFAULT_RENDER_EVERY_STEPS == 0:
                env.render()
                rendered_this_step = True

            if not rendered_this_step and (terminated or truncated):
                env.render()

            _sleep_until_realtime_target(
                step + 1,
                sim_dt,
                wall_start,
                sleep_scale,
                wall_time_offset,
            )

        if _report_step(step, terminated, truncated, info, base_env, wall_start, sim_dt):
            break

        step += 1


def run_demo(args):
    env = _make_env("demo")
    try:
        _run_loop(
            env,
            max_steps=args.max_steps,
            print_every=args.print_every,
            realtime=True,
            sleep_scale=args.sleep_scale,
            print_human_robot_distance_periodically=True,
        )
    finally:
        env.close()


def run_train(args):
    env = _make_env("train")
    try:
        _run_loop(
            env,
            max_steps=args.max_steps,
            print_every=args.print_every,
            realtime=False,
            print_human_robot_distance_periodically=False,
        )
    finally:
        env.close()


def run_record(args):
    video_folder = _build_video_folder(
        args.video_root,
        args.use_timestamp_subfolder,
        args.name_prefix,
    )
    env = _make_env("record")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=args.name_prefix,
        episode_trigger=lambda episode_idx: episode_idx == 0,
        fps=args.video_fps,
    )
    try:
        _run_loop(
            env,
            max_steps=args.max_steps,
            print_every=args.print_every,
            realtime=False,
            print_human_robot_distance_periodically=False,
        )
        print(f"Video saved to {video_folder}")
    finally:
        env.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the slimmed MuseumEnv demo/train/record loop.")
    parser.add_argument(
        "--mode",
        choices=("demo", "train", "record"),
        default="demo",
        help="Execution mode.",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of steps to simulate.",
    )
    parser.add_argument(
        "--print-every",
        type=_positive_int,
        default=DEFAULT_RTF_PRINT_EVERY,
        help="Print periodic environment summaries every N steps.",
    )
    parser.add_argument(
        "--sleep-scale",
        type=_positive_float,
        default=DEFAULT_SLEEP_SCALE,
        help="Only for demo mode. Multiplier on simulation-time pacing (1.0 = real time).",
    )
    parser.add_argument(
        "--video-fps",
        type=_positive_int,
        default=DEFAULT_VIDEO_FPS,
        help="Only for record mode. Encoded output video fps.",
    )
    parser.add_argument(
        "--video-root",
        default=DEFAULT_VIDEO_ROOT,
        help="Only for record mode. Root output folder.",
    )
    parser.add_argument(
        "--name-prefix",
        default=DEFAULT_NAME_PREFIX,
        help="Only for record mode. Video filename prefix.",
    )
    parser.add_argument(
        "--use-timestamp-subfolder",
        action="store_true",
        help="Only for record mode. Create a timestamped output folder.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "demo":
        run_demo(args)
        return
    if args.mode == "train":
        run_train(args)
        return
    run_record(args)


if __name__ == "__main__":
    main()
