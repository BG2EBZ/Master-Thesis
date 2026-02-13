import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import inspect
from datetime import datetime
from pathlib import Path
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

        # action = np.zeros(env.action_space.shape, dtype=np.float32)
        # action[0] = 20.0  # command full speed forward on X axis

        for step in range(MAX_STEPS):
            last_step = step
            obs, reward, terminated, truncated, info = env.step(None)

            # if step % 100 == 0:
            #     vx = info["robot_vx"]
            #     vy = info["robot_vy"]
            #     v_yaw = info["robot_v_yaw"]
            #     desired_yaw = info["desired_yaw"]
            #     actual_yaw = info["actual_yaw"]
            #     print(f"[step {step}] robot velocity: vx={vx:.4f}, vy={vy:.4f}, v_yaw={v_yaw:.4f}")
            #     print(f"           robot yaw: desired={desired_yaw:.4f}, actual={actual_yaw:.4f}")
            #     human_desired_yaw = info["human_desired_yaw"]
            #     human_actual_yaw = info["human_actual_yaw"]
            #     human_vx = info["human_vx"]
            #     human_vy = info["human_vy"]
            #     human_v_yaw = info["human_v_yaw"]
            #     for i, (dyaw, ayaw) in enumerate(zip(human_desired_yaw, human_actual_yaw)):
            #         print(
            #             f"           person{i+1} velocity: "
            #             f"vx={human_vx[i]:.4f}, vy={human_vy[i]:.4f}, v_yaw={human_v_yaw[i]:.4f}"
            #         )
            #         print(f"           person{i+1} yaw: desired={dyaw:.4f}, actual={ayaw:.4f}")
            #     human_goals = info["human_goals"]
            #     for i, (gx, gy) in enumerate(human_goals):
            #         print(f"           person{i+1} goal: x={gx:.4f}, y={gy:.4f}")

            # if step % 200 == 0:
            #     human_v_follow = info["human_v_follow"]
            #     human_v_repulsion = info["human_v_repulsion"]
            #     human_v_hr = info["human_v_hr"]
            #     human_v_total = human_v_follow + human_v_repulsion + human_v_hr
            #     if np.any(np.linalg.norm(human_v_repulsion, axis=1) > 1e-6) or np.any(np.linalg.norm(human_v_hr, axis=1) > 1e-6):
            #         print(f"[step {step}]")
            #         for i, (vf, vr, vhr) in enumerate(zip(human_v_follow, human_v_repulsion, human_v_hr)):
            #             print(
            #                 f"           person{i+1} v_follow=({vf[0]:.4f}, {vf[1]:.4f}) "
            #                 f"v_repulsion=({vr[0]:.4f}, {vr[1]:.4f}) "
            #                 f"v_hr=({vhr[0]:.4f}, {vhr[1]:.4f})"
            #                 f" v_total=({human_v_total[i][0]:.4f}, {human_v_total[i][1]:.4f})")

            # if step % 500 == 0:
            #     rx, ry = info["robot_xy"]
            #     gx, gy = info["robot_goal_xy"]
            #     human_xy = info["human_xy"]
            #     human_goals = info["human_goals"]

            #     print(f"\n[step {step}]")
            #     print(f"  robot pos : ({rx:.2f}, {ry:.2f})")
            #     print(f"  robot goal: ({gx:.2f}, {gy:.2f})")

            #     for i, ((hx, hy), (hgx, hgy)) in enumerate(zip(human_xy, human_goals)):
            #         print(
            #             f"  person{i+1}: "
            #             f"pos=({hx:.2f}, {hy:.2f}) "
            #             f"goal=({hgx:.2f}, {hgy:.2f})"
            #         )

            # # ---- event-based print: any human reaches goal ----
            # if info["human_reached_goal"]:
            #     for i in info["human_reached_goal"]:
            #         print(f"[step {step}] >>> person{i+1} reached their goal!")

            if info.get("final_listen_ready", False):
                print(
                    f"[step {step}] >>> Termination gate met: "
                    "final waypoint reached + all humans reached final listen goals."
                )

            if terminated:
                print(
                    f"Episode terminated at step {step}. "
                    f"final_listen_ready={info.get('final_listen_ready', False)}"
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
