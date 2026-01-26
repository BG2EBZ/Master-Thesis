import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="MuseumEnv-v0",
    entry_point="museum_env:MuseumEnv",  # format: <module>:<class>
    kwargs={"xml_path": "museum_scene.xml"},
)
