import inspect
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import museum_env.register_env

# -----------------------------
# Recording knobs
# -----------------------------
MAX_STEPS = 30000
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
OFFSAMPLES = 1
# MuJoCo dt is 0.002s, so real-time video is about 1/dt = 500 FPS.
# Set lower/higher based on how fast you want playback to look.
VIDEO_FPS = 500
VIDEO_ROOT = "videos"
USE_TIMESTAMP_SUBFOLDER = True


def _build_video_folder():
    root = Path(VIDEO_ROOT)
    if USE_TIMESTAMP_SUBFOLDER:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = root / f"museum_full_run_{run_id}"
    else:
        folder = root
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def main():
    env = None
    video_folder = _build_video_folder()
    last_step = -1

    try:
        env = gym.make("MuseumEnv-v0", render_mode="rgb_array")
        base_env = env.unwrapped
        base_env.max_steps = MAX_STEPS
        base_env.render_width = RENDER_WIDTH
        base_env.render_height = RENDER_HEIGHT
        base_env.model.vis.quality.offsamples = OFFSAMPLES

        record_kwargs = {
            "env": env,
            "video_folder": video_folder,
            "episode_trigger": lambda episode_id: True,
            "name_prefix": "museum_full_run",
        }
        # Keep compatibility across gymnasium versions.
        if "fps" in inspect.signature(RecordVideo.__init__).parameters:
            record_kwargs["fps"] = VIDEO_FPS

        env = RecordVideo(**record_kwargs)

        obs, _ = env.reset()
        print("Initial obs:", obs)
        print(
            f"Recording settings: {RENDER_WIDTH}x{RENDER_HEIGHT}, "
            f"offsamples={OFFSAMPLES}, video_fps={VIDEO_FPS}"
        )

        for step in range(MAX_STEPS):
            last_step = step
            obs, reward, terminated, truncated, info = env.step(None)

            # Example debug reads with the new nested info schema:
            # robot = info["robot"]
            # humans = info["humans"]
            # status = info["status"]
            # events = info["events"]
            # print(robot["mode"], humans["all_reached"], status["listen_wait"]["remaining"])

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
                break

            if truncated:
                print(f"Episode truncated at step {step}.")
                break
    except KeyboardInterrupt:
        print(f"\nInterrupted by user at step {last_step}.")
    finally:
        if env is not None:
            env.close()
        print(f"Video saved under: {video_folder}/")


if __name__ == "__main__":
    main()
