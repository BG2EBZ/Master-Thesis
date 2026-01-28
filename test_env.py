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

    if step % 500 == 0:
        x, y, dx, dy = obs
        gx = x + dx
        gy = y + dy
        print(
            f"[step {step}] "
            f"robot=({x:.2f}, {y:.2f}) "
            f"goal=({gx:.2f}, {gy:.2f})"
        )
    
    if terminated:
        print(f"Robot reached goal at step {step}!")
        break

env.close()