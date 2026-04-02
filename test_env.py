import argparse
import inspect
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import museum_env.register_env

try:
    import glfw
except ImportError:  # pragma: no cover - optional runtime dependency
    glfw = None


DEFAULT_MAX_STEPS = 300000
DEFAULT_RENDER_FPS = 60
DEFAULT_VIDEO_FPS = 500

# 1.0 means target real-time, >1.0 means faster than real-time, <1.0 means slower than real-time.
DEFAULT_SLEEP_SCALE = 1.0
SLEEP_SCALE_MIN = 0.0625
SLEEP_SCALE_MAX = 16.0
SLEEP_SCALE_STEP_FACTOR = 2.0
# my timestep="0.002" in .xml files, so 500 steps ~ 1 second of sim time 
DEFAULT_RTF_PRINT_EVERY = 2500
# resolution (in pixels) for the simulation window and the recorded video. 
DEFAULT_RENDER_WIDTH = 640
DEFAULT_RENDER_HEIGHT = 480

DEFAULT_OFFSAMPLES = 1

# Folder where videos will be saved
DEFAULT_VIDEO_ROOT = "videos"
# Starting string for video filenames
DEFAULT_NAME_PREFIX = "museum_full_run"
R_KEYCODE_FALLBACKS = (ord("R"), ord("r"))
P_KEYCODE_FALLBACKS = (ord("P"), ord("p"))
SPEED_UP_KEYCODE_FALLBACKS = (ord("="), ord("+"))
SPEED_DOWN_KEYCODE_FALLBACKS = (ord("-"), ord("_"))


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


def _report_step(step, terminated, truncated, info, wall_start=None, sim_dt=None):
    time_str = ""
    if wall_start is not None and sim_dt is not None:
        real_time = time.perf_counter() - wall_start
        sim_time = step * sim_dt
        time_str = f", sim_time={sim_time:.3f}s, real_time={real_time:.3f}s"

    if info["events"]["final_listen_ready"]:
        print(
            f"[step {step}] >>> Termination gate met: "
            "final waypoint reached + all humans reached final listen goals." + time_str
        )

    if terminated:
        print(
            f"Episode terminated at step {step}. "
            f"final_listen_ready={info['events']['final_listen_ready']}" + time_str
        )
        return True

    if truncated:
        print(f"Episode truncated at step {step}." + time_str)
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


def _build_distracted_progress_extra(base_env, info, sim_dt):
    if info is None:
        return ""
    if base_env is None or not hasattr(base_env, "humans"):
        return ""

    humans_info = info.get("humans", {})
    status_info = info.get("status", {})
    listen_status = status_info.get("listen_wait", {})
    listening_active = bool(status_info.get("listen_mode", False) or listen_status.get("active", False))
    following_steps = humans_info.get("following_steps", None)
    impatient_steps = humans_info.get("following_low_robot_speed_steps", None)
    listening_steps = humans_info.get("listening_steps", None)

    if not listening_active:
        if following_steps is None or impatient_steps is None:
            return ""
        try:
            follow_step_list = [int(v) for v in following_steps]
            impatient_step_list = [int(v) for v in impatient_steps]
        except TypeError:
            return ""
        follow_window_active = bool(status_info.get("distracted_follow_window_active", False))
        follow_chunks = []
        impatient_chunks = []
        for i, human in enumerate(base_env.humans):
            steps_i = follow_step_list[i] if i < len(follow_step_list) else 0
            follow_sec = float(steps_i) * float(sim_dt)
            ramp_start_sec = float(getattr(human, "following_distracted_ramp_start_seconds", 0.0))
            follow_chunks.append(f"h{i+1}:{follow_sec:.1f}/{ramp_start_sec:.1f}s")
            impatient_steps_i = impatient_step_list[i] if i < len(impatient_step_list) else 0
            impatient_sec = float(impatient_steps_i) * float(sim_dt)
            impatient_ramp_start_sec = float(
                getattr(human, "following_impatient_ramp_start_seconds", 0.0)
            )
            impatient_chunks.append(
                f"h{i+1}:{impatient_sec:.1f}/{impatient_ramp_start_sec:.1f}s"
            )
        return (
            f"distr_win={int(follow_window_active)}, "
            f"follow_eff/start=[{', '.join(follow_chunks)}], "
            f"impatient_eff/start=[{', '.join(impatient_chunks)}]"
        )

    if listening_steps is None:
        return ""
    try:
        listening_step_list = [int(v) for v in listening_steps]
    except TypeError:
        return ""
    listening_window_active = status_info.get("listening_distracted_window_active", [])
    listening_chunks = []
    for i, human in enumerate(base_env.humans):
        steps_i = listening_step_list[i] if i < len(listening_step_list) else 0
        listen_sec = float(steps_i) * float(sim_dt)
        ramp_start_sec = float(getattr(human, "listening_distracted_ramp_start_seconds", 0.0))
        window_i = int(bool(listening_window_active[i])) if i < len(listening_window_active) else 0
        listening_chunks.append(
            f"h{i+1}:{listen_sec:.1f}/{ramp_start_sec:.1f}s(w={window_i})"
        )

    return f"listen_eff/start=[{', '.join(listening_chunks)}]"


def _build_hh_distance_mean_1s_extra(info):
    if info is None:
        return ""

    try:
        hh_mean_1s = info["metrics"]["humans"]["nearest_human_distance_mean_1s"]
    except (KeyError, TypeError):
        return ""

    try:
        values = list(hh_mean_1s)
    except TypeError:
        return ""
    if not values:
        return ""

    chunks = []
    for idx, value in enumerate(values, start=1):
        value_float = float(value)
        value_str = "nan" if value_float != value_float else f"{value_float:.2f}"
        chunks.append(f"h{idx}:{value_str}")
    return f"hh_mean_1s=[{', '.join(chunks)}]"


def _build_hr_distance_mean_1s_extra(info):
    if info is None:
        return ""

    try:
        hr_mean_1s = info["metrics"]["humans"]["human_robot_distance_mean_1s"]
    except (KeyError, TypeError):
        return ""

    try:
        values = list(hr_mean_1s)
    except TypeError:
        return ""
    if not values:
        return ""

    chunks = []
    for idx, value in enumerate(values, start=1):
        value_float = float(value)
        value_str = "nan" if value_float != value_float else f"{value_float:.2f}"
        chunks.append(f"h{idx}:{value_str}")
    return f"hr_mean_1s=[{', '.join(chunks)}]"


def _build_local_crowding_count_1m_extra(info):
    if info is None:
        return ""

    try:
        local_crowding = info["metrics"]["humans"]["local_crowding_count_1m"]
    except (KeyError, TypeError):
        return ""

    try:
        values = list(local_crowding)
    except TypeError:
        return ""
    if not values:
        return ""

    chunks = []
    for idx, value in enumerate(values, start=1):
        chunks.append(f"h{idx}:{int(value)}")
    return f"crowd_1m=[{', '.join(chunks)}]"


def _combine_periodic_extras(*extras):
    parts = [str(extra) for extra in extras if str(extra)]
    return ", ".join(parts)


def _configure_base_env(base_env, args):
    base_env.max_steps = args.max_steps
    base_env.render_width = args.render_width
    base_env.render_height = args.render_height
    base_env.model.vis.quality.offsamples = args.offsamples
    base_env.metadata["render_fps"] = args.render_fps


def _print_initial_obs_and_goal(env, obs):
    print("Initial obs:", obs)
    base_env = env.unwrapped
    if hasattr(base_env, "_get_goal_xy"):
        gx, gy = base_env._get_goal_xy()
        print(f"Initial robot goal_xy: ({gx:.3f}, {gy:.3f})")


def _make_demo_key_callback(
    reset_event,
    pause_toggle_event,
    speed_up_event=None,
    speed_down_event=None,
):
    reset_keycodes = set(R_KEYCODE_FALLBACKS)
    pause_keycodes = set(P_KEYCODE_FALLBACKS)
    speed_up_keycodes = set(SPEED_UP_KEYCODE_FALLBACKS)
    speed_down_keycodes = set(SPEED_DOWN_KEYCODE_FALLBACKS)
    if glfw is not None and hasattr(glfw, "KEY_R"):
        reset_keycodes.add(int(glfw.KEY_R))
    if glfw is not None and hasattr(glfw, "KEY_P"):
        pause_keycodes.add(int(glfw.KEY_P))
    if glfw is not None and hasattr(glfw, "KEY_EQUAL"):
        speed_up_keycodes.add(int(glfw.KEY_EQUAL))
    if glfw is not None and hasattr(glfw, "KEY_KP_ADD"):
        speed_up_keycodes.add(int(glfw.KEY_KP_ADD))
    if glfw is not None and hasattr(glfw, "KEY_MINUS"):
        speed_down_keycodes.add(int(glfw.KEY_MINUS))
    if glfw is not None and hasattr(glfw, "KEY_KP_SUBTRACT"):
        speed_down_keycodes.add(int(glfw.KEY_KP_SUBTRACT))

    def _on_key(keycode):
        try:
            key = int(keycode)
            if key in reset_keycodes:
                reset_event.set()
            elif key in pause_keycodes:
                pause_toggle_event.set()
            elif speed_up_event is not None and key in speed_up_keycodes:
                speed_up_event.set()
            elif speed_down_event is not None and key in speed_down_keycodes:
                speed_down_event.set()
        except (TypeError, ValueError):
            return

    return _on_key


def _handle_requested_reset_if_any(env, reset_event):
    if reset_event is None or not reset_event.is_set():
        return False
    reset_event.clear()
    obs, _ = env.reset()
    _print_initial_obs_and_goal(env, obs)
    return True


def _handle_pause_toggle_if_any(pause_toggle_event, paused):
    if pause_toggle_event is None or not pause_toggle_event.is_set():
        return paused
    pause_toggle_event.clear()
    paused = not paused
    print("[demo] Paused" if paused else "[demo] Resumed")
    return paused


def _update_sleep_scale_from_events(current_sleep_scale, speed_up_event=None, speed_down_event=None):
    speed_up = bool(speed_up_event is not None and speed_up_event.is_set())
    speed_down = bool(speed_down_event is not None and speed_down_event.is_set())
    if speed_up_event is not None:
        speed_up_event.clear()
    if speed_down_event is not None:
        speed_down_event.clear()

    new_sleep_scale = float(current_sleep_scale)
    if speed_up and not speed_down:
        new_sleep_scale *= SLEEP_SCALE_STEP_FACTOR
    elif speed_down and not speed_up:
        new_sleep_scale /= SLEEP_SCALE_STEP_FACTOR

    new_sleep_scale = float(min(SLEEP_SCALE_MAX, max(SLEEP_SCALE_MIN, new_sleep_scale)))
    if new_sleep_scale != float(current_sleep_scale):
        print(f"[demo] speed x{new_sleep_scale:.3f} (sleep_scale={new_sleep_scale:.4f})")
    return new_sleep_scale


def _run_demo_stable(
    env,
    args,
    sim_dt,
    reset_event=None,
    pause_toggle_event=None,
    speed_up_event=None,
    speed_down_event=None,
):
    base_env = env.unwrapped
    sim_hz = 1.0 / sim_dt
    render_fps = max(1, args.render_fps)
    steps_per_frame = max(1, round(sim_hz / render_fps))
    current_sleep_scale = float(args.sleep_scale)

    obs, _ = env.reset()
    _print_initial_obs_and_goal(env, obs)
    print(
        f"[demo-stable] dt={sim_dt:.6f}s, render_fps={render_fps}, "
        f"steps_per_frame={steps_per_frame}, sleep_scale={current_sleep_scale:.4f}"
    )

    wall_start = time.perf_counter()
    steps_done = 0
    paused = False

    while steps_done < args.max_steps:
        current_sleep_scale = _update_sleep_scale_from_events(
            current_sleep_scale,
            speed_up_event=speed_up_event,
            speed_down_event=speed_down_event,
        )
        paused = _handle_pause_toggle_if_any(pause_toggle_event, paused)
        if paused:
            _handle_requested_reset_if_any(env, reset_event)
            env.render()
            time.sleep(0.01)
            continue

        frame_start = time.perf_counter()
        terminated = False
        truncated = False
        reset_happened = False
        info = {"events": {"final_listen_ready": False}}

        for _ in range(steps_per_frame):
            if steps_done >= args.max_steps:
                break
            if _handle_requested_reset_if_any(env, reset_event):
                reset_happened = True
                break

            obs, reward, terminated, truncated, info = env.step(None)
            steps_done += 1

            if steps_done % args.rtf_print_every == 0:
                _print_rtf(
                    "demo-stable",
                    steps_done,
                    sim_dt,
                    wall_start,
                    extra=_combine_periodic_extras(
                        _build_distracted_progress_extra(base_env, info, sim_dt),
                        (
                            _build_hh_distance_mean_1s_extra(info)
                            if args.print_hh_distance_mean_1s
                            else ""
                        ),
                        (
                            _build_hr_distance_mean_1s_extra(info)
                            if args.print_hr_distance_mean_1s
                            else ""
                        ),
                        (
                            _build_local_crowding_count_1m_extra(info)
                            if args.print_local_crowding_count_1m
                            else ""
                        ),
                    ),
                )

            # Checks if the goal was met or the simulation ended during these steps.
            if _report_step(steps_done - 1, terminated, truncated, info, wall_start, sim_dt):
                break

        env.render()
        if reset_happened:
            continue
        
        target_frame_sec = (steps_per_frame * sim_dt) / current_sleep_scale
        elapsed = time.perf_counter() - frame_start
        sleep_sec = target_frame_sec - elapsed
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        if terminated or truncated:
            break


def _run_demo_strict(
    env,
    args,
    sim_dt,
    reset_event=None,
    pause_toggle_event=None,
    speed_up_event=None,
    speed_down_event=None,
):
    base_env = env.unwrapped
    sim_hz = 1.0 / sim_dt
    render_fps = max(1, args.render_fps)
    steps_per_frame = max(1, round(sim_hz / render_fps))
    current_sleep_scale = float(args.sleep_scale)

    obs, _ = env.reset()
    _print_initial_obs_and_goal(env, obs)
    print(
        f"[demo-strict] dt={sim_dt:.6f}s, render_fps={render_fps}, "
        f"steps_per_frame={steps_per_frame}, sleep_scale={current_sleep_scale:.4f}"
    )

    wall_start = time.perf_counter()
    lag_steps = 0
    steps_done = 0
    paused = False
    pause_started_wall = None

    while steps_done < args.max_steps:
        was_paused = paused
        paused = _handle_pause_toggle_if_any(pause_toggle_event, paused)
        if paused and not was_paused:
            pause_started_wall = time.perf_counter()
        elif (not paused) and was_paused and pause_started_wall is not None:
            wall_start += time.perf_counter() - pause_started_wall
            pause_started_wall = None

        prev_sleep_scale = current_sleep_scale
        current_sleep_scale = _update_sleep_scale_from_events(
            current_sleep_scale,
            speed_up_event=speed_up_event,
            speed_down_event=speed_down_event,
        )
        if current_sleep_scale != prev_sleep_scale:
            wall_start = time.perf_counter() - (steps_done * sim_dt) / current_sleep_scale
            if paused:
                pause_started_wall = time.perf_counter()

        if paused:
            _handle_requested_reset_if_any(env, reset_event)
            env.render()
            time.sleep(0.01)
            continue

        if _handle_requested_reset_if_any(env, reset_event):
            wall_start = time.perf_counter()
            lag_steps = 0
            pause_started_wall = None
            env.render()
            continue

        obs, reward, terminated, truncated, info = env.step(None)
        steps_done += 1

        target_tick = wall_start + (steps_done * sim_dt) / current_sleep_scale
        sleep_sec = target_tick - time.perf_counter()
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        else:
            lag_steps += 1

        if (steps_done % steps_per_frame == 0) or terminated or truncated:
            env.render()

        if steps_done % args.rtf_print_every == 0:
            _print_rtf(
                "demo-strict",
                steps_done,
                sim_dt,
                wall_start,
                extra=_combine_periodic_extras(
                    f"lag_steps={lag_steps}",
                    _build_distracted_progress_extra(base_env, info, sim_dt),
                    (
                        _build_hh_distance_mean_1s_extra(info)
                        if args.print_hh_distance_mean_1s
                        else ""
                    ),
                    (
                        _build_hr_distance_mean_1s_extra(info)
                        if args.print_hr_distance_mean_1s
                        else ""
                    ),
                    (
                        _build_local_crowding_count_1m_extra(info)
                        if args.print_local_crowding_count_1m
                        else ""
                    ),
                ),
            )

        if _report_step(steps_done - 1, terminated, truncated, info, wall_start, sim_dt):
            break


def _run_fast_loop(env, args, sim_dt, tag):
    base_env = env.unwrapped
    obs, _ = env.reset()
    _print_initial_obs_and_goal(env, obs)

    wall_start = time.perf_counter()
    for step in range(args.max_steps):
        obs, reward, terminated, truncated, info = env.step(None)

        if (step + 1) % args.rtf_print_every == 0:
            _print_rtf(
                tag,
                step + 1,
                sim_dt,
                wall_start,
                extra=_combine_periodic_extras(
                    _build_distracted_progress_extra(base_env, info, sim_dt),
                    (
                        _build_hh_distance_mean_1s_extra(info)
                        if args.print_hh_distance_mean_1s
                        else ""
                    ),
                    (
                        _build_hr_distance_mean_1s_extra(info)
                        if args.print_hr_distance_mean_1s
                        else ""
                    ),
                    (
                        _build_local_crowding_count_1m_extra(info)
                        if args.print_local_crowding_count_1m
                        else ""
                    ),
                ),
            )

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
        demo_reset_event = None
        demo_pause_toggle_event = None
        demo_speed_up_event = None
        demo_speed_down_event = None

        if args.mode == "train":
            print(f"[train] dt={sim_dt:.6f}, no render, no sleep")
            _run_fast_loop(env, args, sim_dt, tag="train")
            return

        if args.mode == "demo" and hasattr(base_env, "set_viewer_key_callback"):
            demo_reset_event = threading.Event()
            demo_pause_toggle_event = threading.Event()
            demo_speed_up_event = threading.Event()
            demo_speed_down_event = threading.Event()
            base_env.set_viewer_key_callback(
                _make_demo_key_callback(
                    demo_reset_event,
                    demo_pause_toggle_event,
                    demo_speed_up_event,
                    demo_speed_down_event,
                )
            )
            print("[demo] keys: R reset, P pause/resume, =/+ faster, - slower")

        if args.realtime_policy == "strict":
            _run_demo_strict(
                env,
                args,
                sim_dt,
                reset_event=demo_reset_event,
                pause_toggle_event=demo_pause_toggle_event,
                speed_up_event=demo_speed_up_event,
                speed_down_event=demo_speed_down_event,
            )
        else:
            _run_demo_stable(
                env,
                args,
                sim_dt,
                reset_event=demo_reset_event,
                pause_toggle_event=demo_pause_toggle_event,
                speed_up_event=demo_speed_up_event,
                speed_down_event=demo_speed_down_event,
            )
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
    parser.add_argument(
        "--print-hh-distance-mean-1s",
        action="store_true",
        help="print per-human nearest_human_distance_mean_1s at each rtf_print_every interval",
    )
    parser.add_argument(
        "--print-hr-distance-mean-1s",
        action="store_true",
        help="print per-human human_robot_distance_mean_1s at each rtf_print_every interval",
    )
    parser.add_argument(
        "--print-local-crowding-count-1m",
        action="store_true",
        help="print per-human local_crowding_count_1m at each rtf_print_every interval",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
