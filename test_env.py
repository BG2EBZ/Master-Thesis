from museum_env import MuseumEnv
import numpy as np
import time

env = MuseumEnv("museum_scene.xml", render_mode="human")

obs, _ = env.reset()
print("Initial obs:", obs)

env.data.ctrl[:] = 0.0
env.data.ctrl[0] = 10.0

for _ in range(3000):
    # action = env.action_space.sample()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.02)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()