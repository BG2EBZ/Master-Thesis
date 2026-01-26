import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class MuseumEnv(gym.Env):
    """
    Minimal runnable Gymnasium environment for a MuJoCo museum scene.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path="museum_scene.xml", render_mode=None):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        # --- Action space ---
        # Assume N actuators (e.g. wheel velocities)
        self.nu = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=np.float32,
        )

        # --- Observation space ---
        # Minimal: robot x, y, orientation (theta)
        # We read these from qpos
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )

        self.timestep = self.model.opt.timestep
        self.max_steps = 1000
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1

        # Apply action to actuators
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # --- Very simple reward ---
        reward = -0.01  # time penalty only

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Minimal observation:
        [x, y, theta]
        """
        # Assumes robot pose is in qpos[0:3]
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        theta = self.data.qpos[2]

        return np.array([x, y, theta], dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
