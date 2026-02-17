import argparse
import inspect
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import museum_env.register_env


DEFAULT_MAX_STEPS = 30000
DEFAULT_RENDER_FPS = 60
DEFAULT_VIDEO_FPS = 500

# 1.0 means target real-time, <1.0 means faster than real-time, >1.0 means slower than real-time 
DEFAULT_SLEEP_SCALE = 1.0
# my timestep="0.002" in .xml files, so 500 steps ~ 1 second of sim time 
DEFAULT_RTF_PRINT_EVERY = 500
# resolution (in pixels) for the simulation window and the recorded video. 
DEFAULT_RENDER_WIDTH = 1920
DEFAULT_RENDER_HEIGHT = 1080

DEFAULT_OFFSAMPLES = 1

# Folder where videos will be saved
DEFAULT_VIDEO_ROOT = "videos"
# Starting string for video filenames
DEFAULT_NAME_PREFIX = "museum_full_run"


def _positive_int(value):
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def _positive_float(value):
    v = float(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return v


def _build_video_folder(video_root, use_timestamp_subfolder, name_prefix):
    root = Path(video_root)
    if use_timestamp_subfolder:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = root / f"{name_prefix}_{run_id}"
    else:
        folder = root
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def _report_step(step, terminated, truncated, info):
    if info["events"]["final_listen_ready"]:
        print(
            f"[step {step}] >>> Termination gate met: "
            "final waypoint reached + all humans reached final listen goals."
        )

    if terminated:
        print(
            f"Episode terminated at step {step}. "
            f"final_listen_ready={info['events']['final_listen_ready']}"
        )
        return True

    if truncated:
        print(f"Episode truncated at step {step}.")
        return True

    return False


def _print_rtf(tag, steps_done, sim_dt, wall_start, extra=""):
    real_time = max(1e-9, time.perf_counter() - wall_start)
    sim_time = steps_done * sim_dt
    rtf = sim_time / real_time
    suffix = f", {extra}" if extra else ""
    print(
        f"[{tag}] steps={steps_done}, sim_time={sim_time:.3f}s, "
        f"real_time={real_time:.3f}s, rtf={rtf:.3f}{suffix}"
    )


def _configure_base_env(base_env, args):
    base_env.max_steps = args.max_steps
    base_env.render_width = args.render_width
    base_env.render_height = args.render_height
    base_env.model.vis.quality.offsamples = args.offsamples
    base_env.metadata["render_fps"] = args.render_fps


def _run_demo_stable(env, args, sim_dt):
    sim_hz = 1.0 / sim_dt
    render_fps = max(1, args.render_fps)
    steps_per_frame = max(1, round(sim_hz / render_fps))
    target_frame_sec = (steps_per_frame * sim_dt) / args.sleep_scale

    obs, _ = env.reset()
    print("Initial obs:", obs)
    print(
        f"[demo-stable] dt={sim_dt:.6f}s, render_fps={render_fps}, "
        f"steps_per_frame={steps_per_frame}, target_frame={target_frame_sec:.6f}s"
    )

    wall_start = time.perf_counter()
    steps_done = 0

    while steps_done < args.max_steps:
        frame_start = time.perf_counter()
        terminated = False
        truncated = False
        info = {"events": {"final_listen_ready": False}}

        for _ in range(steps_per_frame):
            if steps_done >= args.max_steps:
                break

            obs, reward, terminated, truncated, info = env.step(None)
            steps_done += 1

            # if steps_done % args.rtf_print_every == 0:
            #     _print_rtf("demo-stable", steps_done, sim_dt, wall_start)

            # Checks if the goal was met or the simulation ended during these steps.
            if _report_step(steps_done - 1, terminated, truncated, info):
                break

        env.render()
        
        elapsed = time.perf_counter() - frame_start
        sleep_sec = target_frame_sec - elapsed
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        if terminated or truncated:
            break


def _run_demo_strict(env, args, sim_dt):
    sim_hz = 1.0 / sim_dt
    render_fps = max(1, args.render_fps)
    steps_per_frame = max(1, round(sim_hz / render_fps))

    obs, _ = env.reset()
    print("Initial obs:", obs)
    print(
        f"[demo-strict] dt={sim_dt:.6f}s, render_fps={render_fps}, "
        f"steps_per_frame={steps_per_frame}, sleep_scale={args.sleep_scale:.3f}"
    )

    wall_start = time.perf_counter()
    lag_steps = 0

    for step in range(args.max_steps):
        obs, reward, terminated, truncated, info = env.step(None)

        target_tick = wall_start + ((step + 1) * sim_dt) / args.sleep_scale
        sleep_sec = target_tick - time.perf_counter()
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        else:
            lag_steps += 1

        if ((step + 1) % steps_per_frame == 0) or terminated or truncated:
            env.render()

        if (step + 1) % args.rtf_print_every == 0:
            _print_rtf(
                "demo-strict",
                step + 1,
                sim_dt,
                wall_start,
                extra=f"lag_steps={lag_steps}",
            )

        if _report_step(step, terminated, truncated, info):
            break


def _run_fast_loop(env, args, sim_dt, tag):
    obs, _ = env.reset()
    print("Initial obs:", obs)

    wall_start = time.perf_counter()
    for step in range(args.max_steps):
        obs, reward, terminated, truncated, info = env.step(None)

        if (step + 1) % args.rtf_print_every == 0:
            _print_rtf(tag, step + 1, sim_dt, wall_start)

        if _report_step(step, terminated, truncated, info):
            break


def _run_record_mode(args):
    env = None
    video_folder = _build_video_folder(
        video_root=args.video_root,
        use_timestamp_subfolder=(not args.no_timestamp_subfolder),
        name_prefix=args.name_prefix,
    )
    try:
        env = gym.make("MuseumEnv-v0", render_mode="rgb_array")
        base_env = env.unwrapped
        _configure_base_env(base_env, args)
        sim_dt = float(base_env.timestep)

        record_kwargs = {
            "env": env,
            "video_folder": video_folder,
            "episode_trigger": lambda episode_id: True,
            "name_prefix": args.name_prefix,
        }
        # Keep compatibility across gymnasium versions.
        if "fps" in inspect.signature(RecordVideo.__init__).parameters:
            record_kwargs["fps"] = args.video_fps

        env = RecordVideo(**record_kwargs)
        print(
            f"[record] width={args.render_width}, height={args.render_height}, "
            f"offsamples={args.offsamples}, video_fps={args.video_fps}, dt={sim_dt:.6f}"
        )
        _run_fast_loop(env, args, sim_dt, tag="record")
    finally:
        if env is not None:
            env.close()
        print(f"Video saved under: {video_folder}/")


def run(args):
    if args.mode == "record":
        _run_record_mode(args)
        return

    render_mode = "human" if args.mode == "demo" else None
    env = None
    try:
        env = gym.make("MuseumEnv-v0", render_mode=render_mode)
        base_env = env.unwrapped
        _configure_base_env(base_env, args)
        sim_dt = float(base_env.timestep)

        if args.mode == "train":
            print(f"[train] dt={sim_dt:.6f}, no render, no sleep")
            _run_fast_loop(env, args, sim_dt, tag="train")
            return

        if args.realtime_policy == "strict":
            _run_demo_strict(env, args, sim_dt)
        else:
            _run_demo_stable(env, args, sim_dt)
    finally:
        if env is not None:
            env.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="MuseumEnv runner with demo/train/record modes."
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "train", "record"],
        default="demo",
        help="demo: near real-time rendering, train: max speed, record: video capture",
    )
    parser.add_argument(
        "--realtime-policy",
        choices=["stable", "strict"],
        default="stable",
        help="only used in demo mode",
    )
    parser.add_argument("--render-fps", type=_positive_int, default=DEFAULT_RENDER_FPS)
    parser.add_argument("--max-steps", type=_positive_int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--video-fps", type=_positive_int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--sleep-scale", type=_positive_float, default=DEFAULT_SLEEP_SCALE)
    parser.add_argument("--rtf-print-every", type=_positive_int, default=DEFAULT_RTF_PRINT_EVERY)
    parser.add_argument("--render-width", type=_positive_int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--render-height", type=_positive_int, default=DEFAULT_RENDER_HEIGHT)
    parser.add_argument("--offsamples", type=_positive_int, default=DEFAULT_OFFSAMPLES)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--no-timestamp-subfolder", action="store_true")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
