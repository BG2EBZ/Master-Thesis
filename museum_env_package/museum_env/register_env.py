from gymnasium.envs.registration import register

register(
    id="MuseumEnv-v0",
    entry_point="museum_env:MuseumEnv",  # format: <module>:<class>
    kwargs={},
)
