import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
from importlib import resources 


class MuseumEnv(gym.Env):
    """
    Minimal runnable Gymnasium environment for a MuJoCo museum scene.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path=None, render_mode=None):
        super().__init__()

        if xml_path is None:
            with resources.path("museum_env.assets", "museum_scene.xml") as xml_file:
                xml_path = str(xml_file)
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        # --- Action space ---
        # Assume N actuators (3 for robot)
        self.nu = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=np.float32,
        )

        # print("=== Actuator order ===")
        # for i in range(self.model.nu):
        #     print(i, self.model.actuator(i).name)
        # print("======================")

        # --- Observation space ---
        # Minimal: robot x, y, orientation (theta)
        # Read these from qpos
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )

        self.timestep = self.model.opt.timestep
        self.max_steps = 1000
        self.step_count = 0

    def _wrap_to_pi(self, ang: float) -> float:
        return (ang + np.pi) % (2 * np.pi) - np.pi

    def _get_robot_pose(self):
        # Use qpos[0:3] for robot x,y,yaw now
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        yaw = float(self.data.qpos[2])
        return x, y, yaw

    def _get_goal_xy(self):
        # Requires <site name="goal_site" .../> in XML
        goal_xy = self.data.site("goal_site").xpos[:2]
        return float(goal_xy[0]), float(goal_xy[1])

    def _rule_based_action(self):
        """
        Minimal go-to-goal controller.
        Returns action = [vx, vy, yaw_rate] in actuator units (m/s, m/s, rad/s).
        """
        x, y, yaw = self._get_robot_pose()
        gx, gy = self._get_goal_xy()

        dx = gx - x
        dy = gy - y
        dist = np.hypot(dx, dy) + 1e-8

        # --- Tunable gains (start with these) ---
        k_v = 5.0      # translation gain
        k_yaw = 2.0    # heading gain

        # Desired heading toward goal
        desired_yaw = np.arctan2(dy, dx)
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        # Move toward goal in world frame (since your actuators control x/y slides)
        vx = k_v * dx
        vy = k_v * dy

        # Slow down near goal (prevents overshoot)
        if dist < 0.5:
            scale = dist / 0.5
            vx *= scale
            vy *= scale

        yaw_rate = k_yaw * yaw_err

        # --- Clip to actuator limits (match your XML ctrlrange) ---
        vx = np.clip(vx, -1.5, 1.5)
        vy = np.clip(vy, -1.5, 1.5)
        yaw_rate = np.clip(yaw_rate, -2.0, 2.0)

        return np.array([vx, vy, yaw_rate], dtype=np.float32), dist
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action=None):
        """
        Rule-based navigation step now.
        """
        self.step_count += 1

        rb_action, dist = self._rule_based_action()

        # Apply to robot actuators only
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = rb_action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Reward is optional for rule-based baseline, but keep it consistent
        reward = -dist

        terminated = dist < 0.3

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {"dist_to_goal": dist}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Minimal observation:
        [x, y, theta]
        """
        # Assumes robot pose is in qpos[0:3]
        x, y, yaw = self._get_robot_pose()
        gx, gy = self._get_goal_xy()
        return np.array([x, y, gx - x, gy - y], dtype=np.float32)

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
