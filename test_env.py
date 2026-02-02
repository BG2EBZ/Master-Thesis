import gymnasium as gym
import numpy as np
import time
import museum_env.register_env

env = gym.make("MuseumEnv-v0", render_mode="human")

obs, _ = env.reset()
print("Initial obs:", obs)

# action = np.zeros(env.action_space.shape, dtype=np.float32)
# action[0] = 20.0  # command full speed forward on X axis

for step in range(30000):
    obs, reward, terminated, truncated, info = env.step(None)
    env.render()
    time.sleep(0.02)

    if step % 50 == 0:
        vx = info["robot_vx"]
        vy = info["robot_vy"]
        v_yaw = info["robot_v_yaw"]
        desired_yaw = info["desired_yaw"]
        actual_yaw = info["actual_yaw"]
        print(f"[step {step}] robot velocity: vx={vx:.4f}, vy={vy:.4f}, v_yaw={v_yaw:.4f}")
        print(f"           robot yaw: desired={desired_yaw:.4f}, actual={actual_yaw:.4f}")

    if step % 500 == 0:
        rx, ry = info["robot_xy"]
        gx, gy = info["robot_goal_xy"]
        human_xy = info["human_xy"]
        human_goals = info["human_goals"]

        print(f"\n[step {step}]")
        print(f"  robot pos : ({rx:.2f}, {ry:.2f})")
        print(f"  robot goal: ({gx:.2f}, {gy:.2f})")

        for i, ((hx, hy), (hgx, hgy)) in enumerate(zip(human_xy, human_goals)):
            print(
                f"  person{i+1}: "
                f"pos=({hx:.2f}, {hy:.2f}) "
                f"goal=({hgx:.2f}, {hgy:.2f})"
            )
    
    # # ---- event-based print: any human reaches goal ----
    # if info["human_reached_goal"]:
    #     for i in info["human_reached_goal"]:
    #         print(f"[step {step}] >>> person{i+1} reached their goal!")

    if terminated:
        print(f"Robot reached goal at step {step}!")
        break

env.close()