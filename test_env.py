import gymnasium as gym
import numpy as np
import time
import museum_env.register_env

env = gym.make("MuseumEnv-v0", render_mode="human")

obs, _ = env.reset()
print("Initial obs:", obs)

action = np.zeros(env.action_space.shape, dtype=np.float32)
action[0] = 20.0  # command full speed forward on X axis

for _ in range(3000):
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.02)

env.close()