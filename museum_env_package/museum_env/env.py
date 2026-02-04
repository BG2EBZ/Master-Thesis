import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
from importlib import resources 
from .human import Human


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
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.timestep = self.model.opt.timestep
        self.max_steps = 1000
        self.step_count = 0

        # Waypoints: room A → corridor → room B
        # self.waypoints = [
        #     (5, 5),      # Start in room A
        #     (8.5, 0.0),      # Reach corridor entrance
        #     (8.5, -10.0),     # Traverse corridor
        #     (10, -12.5),    # Stop in room B
        # ]

        # Single target: middle position on the left wall (assume a display here).
        self.display_xy = (1.0, 5.0)
        self.waypoints = [self.display_xy]
        self.current_waypoint_idx = 0

        # Human follow switch (start with random walking)
        self.follow_humans = False
        self.robot_start_xy = None
        self.human_follow_distance = 0.5
        
        # --- Initialize humans ---
        # 5 people in XML at positions in qpos
        # Robot: qpos[0:3] = [x, y, yaw]
        # Person1: qpos[3:6], Person2: qpos[6:9], Person3: qpos[9:12], Person4: qpos[12:15], Person5: qpos[15:18]
        self.humans = [
            Human("person1", "person1", qpos_idx=3),
            Human("person2", "person2", qpos_idx=6),
            Human("person3", "person3", qpos_idx=9),
            Human("person4", "person4", qpos_idx=12),
            Human("person5", "person5", qpos_idx=15),
        ]
        for human in self.humans:
            human.external_waypoint = False

    def _wrap_to_pi(self, ang: float) -> float:
        return (ang + np.pi) % (2 * np.pi) - np.pi

    def _get_robot_pose(self):
        # Use body xpos (world coordinates) instead of qpos (joint values)
        robot_body_id = self.model.body("robot").id
        x = float(self.data.xpos[robot_body_id, 0])
        y = float(self.data.xpos[robot_body_id, 1])
        # Get yaw from qpos (rotation around z-axis)
        yaw = float(self.data.qpos[2])
        return x, y, yaw

    def _get_human_poses(self):
        """
        Returns:
            humans_xy: (N, 2) array of human positions in world frame
        """
        humans_xy = []
        for human in self.humans:
            human_body_id = self.model.body(human.body_name).id
            x = float(self.data.xpos[human_body_id, 0])
            y = float(self.data.xpos[human_body_id, 1])
            humans_xy.append([x, y])
        return np.array(humans_xy, dtype=np.float32)

    def _get_human_yaws(self):
        """
        Returns:
            humans_yaw: (N,) array of human yaw angles (rad)
        """
        humans_yaw = []
        for human in self.humans:
            yaw = float(self.data.qpos[human.qpos_idx + 2])
            humans_yaw.append(yaw)
        return np.array(humans_yaw, dtype=np.float32)

    def _get_goal_xy(self):
        # Requires <site name="goal_site" .../> in XML
        goal_xy = self.data.site("goal_site").xpos[:2]
        return float(goal_xy[0]), float(goal_xy[1])

    def _rule_based_action(self):
        """
        Waypoint-following controller.
        Navigates through predefined waypoints from room A to room B.
        """
        x, y, yaw = self._get_robot_pose()
        # Get current target waypoint
        wx, wy = self.waypoints[self.current_waypoint_idx]
 
        dx = wx - x
        dy = wy - y
        dist = np.hypot(dx, dy) + 1e-8

        # Switch to next waypoint if close enough (threshold: 0.3m)
        if dist < 0.3 and self.current_waypoint_idx < len(self.waypoints) - 1:
            self.current_waypoint_idx += 1
            wx, wy = self.waypoints[self.current_waypoint_idx]
            dx = wx - x
            dy = wy - y
            dist = np.hypot(dx, dy) + 1e-8

        # Tunable gains
        k_v = 100.0      # translation gain
        k_yaw = 100.0    # heading gain

        # Desired heading toward current waypoint
        desired_yaw = np.arctan2(dy, dx)
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        # Move toward waypoint
        vx = k_v * dx
        vy = k_v * dy

        # Slow down near goal (prevents overshoot)
        if dist < 0.5:
            scale = dist / 0.5
            vx *= scale
            vy *= scale

        yaw_rate = k_yaw * yaw_err

        # Clip to actuator limits (match XML ctrlrange)
        v = np.clip(k_v * dist, 0, 50.0)
        vx = v * np.cos(yaw)
        vy = v * np.sin(yaw)
        yaw_rate = np.clip(k_yaw * yaw_err, -50.0, 50.0)

        return np.array([vx, vy, yaw_rate], dtype=np.float32), dist, desired_yaw, yaw
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        self.current_waypoint_idx = 0
        self.step_count = 0

        # Store robot start position (for follow trigger)
        rx, ry, _ = self._get_robot_pose()
        self.robot_start_xy = np.array([rx, ry], dtype=np.float32)
        self.follow_humans = False
        
        # Reset humans
        for human in self.humans:
            human.step_count = 0
            human.external_waypoint = False
            human.current_waypoint = human._random_waypoint()

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action=None):
        """
        Rule-based navigation step + human random walking.
        """
        self.step_count += 1

        rb_action, dist, desired_yaw, actual_yaw = self._rule_based_action()

        # stop after reaching the display
        if dist < 0.3:
            rb_action[0:2] = 0.0
            rb_action[2] = 0.0

        human_goal_threshold = 0.5

        # Slow down / stop if a human is too close in front (±60 deg)
        human_xy = self._get_human_poses()
        rx, ry, ryaw = self._get_robot_pose()
        rel = human_xy - np.array([rx, ry], dtype=np.float32)
        dists = np.linalg.norm(rel, axis=1)
        # if dists.size:
        #     angles = np.arctan2(rel[:, 1], rel[:, 0])
        #     angle_err = np.array([self._wrap_to_pi(a - ryaw) for a in angles], dtype=np.float32)
        #     in_front = np.abs(angle_err) <= (np.pi / 3.0)
        #     front_dists = dists[in_front]
        #     min_dist = float(np.min(front_dists)) if front_dists.size else float("inf")
        # else:
        #     min_dist = float("inf")
        # if min_dist < 1.0:
        #     rb_action[0:2] = 0.0
        #     rb_action[2] = 0.0
        # elif min_dist < 2.0:
        #     scale = (min_dist - 1.0) / 1.0
        #     rb_action[0:2] *= scale
        #     rb_action[2] *= scale

        # Apply robot action
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = rb_action
        
        # Update humans
        human_actions = []

        # Switch to follow once the robot has started moving toward the display
        rx, ry, _ = self._get_robot_pose()
        if not self.follow_humans:
            moved_dist = float(np.hypot(rx - self.robot_start_xy[0], ry - self.robot_start_xy[1]))
            if moved_dist >= self.human_follow_distance:
                self.follow_humans = True

        # If following, set external waypoints around the robot
        n_humans = len(self.humans)
        follow_radius = 0.6
        for i, human in enumerate(self.humans):
            human.external_waypoint = self.follow_humans
            if self.follow_humans:
                angle = (2.0 * np.pi / max(n_humans, 1)) * i
                offset = np.array(
                    [follow_radius * np.cos(angle), follow_radius * np.sin(angle)],
                    dtype=np.float32,
                )
                human.current_waypoint = np.array([rx, ry], dtype=np.float32) + offset
            human_action = human.update(self.model, self.data, self.timestep)
            human_actions.append(human_action)
            # Person controls: person1 at indices 3-5, person2 at 6-8, etc.
            ctrl_idx = 3 + i * 3
            self.data.ctrl[ctrl_idx:ctrl_idx+3] = human_action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # ---- DEBUG: human positions ----
        # if self.step_count % 100 == 0:   # avoid spamming
        #     human_xy = self._get_human_poses()
        #     print(f"[step {self.step_count}] Humans:", human_xy)
        # --------------------------------

        rx, ry, _ = self._get_robot_pose()
        gx, gy = self._get_goal_xy()
        human_actual_yaw = self._get_human_yaws()
        human_goals = np.array(
            [h.current_waypoint for h in self.humans],
            dtype=np.float32
        )
        human_desired_yaw = np.arctan2(
            human_goals[:, 1] - human_xy[:, 1],
            human_goals[:, 0] - human_xy[:, 0],
        ).astype(np.float32)
        human_actions = np.array(human_actions, dtype=np.float32) if human_actions else np.zeros((0, 3), dtype=np.float32)
        human_reached_goal = []

        for i, (pos, goal) in enumerate(zip(human_xy, human_goals)):
            dist_to_goal = np.linalg.norm(pos - goal)
            if dist_to_goal < human_goal_threshold:
                human_reached_goal.append(i)

        min_dist = float(np.min(dists))

        obs = self._get_obs()

        # Reward is optional for rule-based baseline, but keep it consistent
        reward = -dist

        # Terminate when robot reaches the final waypoint
        terminated = (dist < 0.3 and self.current_waypoint_idx == len(self.waypoints) - 1)
        truncated = self.step_count >= self.max_steps

        info = {
            "dist_to_goal": dist,
            "robot_xy": np.array([rx, ry], dtype=np.float32),
            "robot_goal_xy": np.array([gx, gy], dtype=np.float32),
            "robot_vx": float(rb_action[0]),
            "robot_vy": float(rb_action[1]),
            "robot_v_yaw": float(rb_action[2]),
            "desired_yaw": float(desired_yaw),
            "actual_yaw": float(actual_yaw),
            "human_xy": human_xy,          # (N, 2)
            "human_goals": human_goals,    # (N, 2)
            "human_desired_yaw": human_desired_yaw,  # (N,)
            "human_actual_yaw": human_actual_yaw,    # (N,)
            "human_vx": human_actions[:, 0],         # (N,)
            "human_vy": human_actions[:, 1],         # (N,)
            "human_v_yaw": human_actions[:, 2],      # (N,)
            "human_reached_goal": human_reached_goal,
            # "min_human_dist": min_dist,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Minimal observation: [x, y, goal_dx, goal_dy]
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
