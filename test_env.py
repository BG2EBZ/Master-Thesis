import time

import gymnasium as gym
import museum_env.register_env


env = gym.make("MuseumEnv-v0", render_mode="human")

obs, _ = env.reset()
print("Initial obs:", obs)

for step in range(30000):
    obs, reward, terminated, truncated, info = env.step(None)
    env.render()
    time.sleep(0.01)

    # Example debug reads with the new nested info schema:
    # robot = info["robot"]
    # humans = info["humans"]
    # status = info["status"]
    # events = info["events"]
    # print(robot["action"]["vx"], humans["action"]["vx"][0], status["listen_wait"]["active"])

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

env.close()
