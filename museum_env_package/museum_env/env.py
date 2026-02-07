import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
from importlib import resources 
from .human import Human, HumanMode


class RobotMode:
    MOVE = "move"
    STOP = "stop"


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
        # Social distance (repulsion) parameters
        self.social_distance = 1.0
        self.repulsion_gain = 2.0
        # Listening formation (fan around robot after it stops)
        self.listen_mode = False
        self.listen_fan_half_angle = np.deg2rad(75.0)  # 150-degree fan
        self.listen_fan_radius = 0.6
        self.listen_stand_threshold = 0.6
        # Turn to face people after reaching the display
        self.turn_target_yaw = None
        self.turn_done = False
        self.robot_mode = RobotMode.MOVE
        
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
            human.set_mode(HumanMode.WANDERING)

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
            humans_xyz: (N, 3) array of human [x, y, yaw] in world frame
        """
        humans_xyz = []
        for human in self.humans:
            human_body_id = self.model.body(human.body_name).id
            x = float(self.data.xpos[human_body_id, 0])
            y = float(self.data.xpos[human_body_id, 1])
            yaw = float(self.data.qpos[human.qpos_idx + 2])
            humans_xyz.append([x, y, yaw])
        return np.array(humans_xyz, dtype=np.float32)

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
        k_v = 20.0      # translation gain
        k_yaw = 20.0    # heading gain

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
        self.turn_target_yaw = None
        self.turn_done = False
        self.listen_mode = False
        self.robot_mode = RobotMode.MOVE
        
        # Reset humans
        for human in self.humans:
            human.step_count = 0
            human.external_waypoint = False
            human.current_waypoint = human._random_waypoint()
            human.set_mode(HumanMode.WANDERING)

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action=None):
        """
        Rule-based navigation step + human random walking.
        """
        self.step_count += 1

        rb_action, dist, desired_yaw, actual_yaw = self._rule_based_action()

        # Get human poses once per step
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        # stop after reaching the display and turn to face people
        rx, ry, ryaw = self._get_robot_pose()
        self.robot_mode = RobotMode.MOVE
        if dist < 0.2:
            self.robot_mode = RobotMode.STOP
            if self.turn_target_yaw is None:
                # Face the crowd: use the mean human position as target
                if human_xyz.size:
                    mean_hx = float(np.mean(human_xyz[:, 0]))
                    mean_hy = float(np.mean(human_xyz[:, 1]))
                    dx = mean_hx - rx
                    dy = mean_hy - ry
                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        self.turn_target_yaw = self._wrap_to_pi(ryaw + np.pi)
                    else:
                        self.turn_target_yaw = np.arctan2(dy, dx)
                else:
                    # Fallback if no humans are detected
                    self.turn_target_yaw = self._wrap_to_pi(ryaw + np.pi)
            yaw_err = self._wrap_to_pi(self.turn_target_yaw - ryaw)
            rb_action[0:2] = 0.0
            if abs(yaw_err) < 0.05:
                rb_action[2] = 0.0
                self.turn_done = True
            else:
                k_turn = 50.0
                rb_action[2] = np.clip(k_turn * yaw_err, -50.0, 50.0)

        # Enter listening mode after turning is done at the display
        if dist < 0.2 and self.turn_done and not self.listen_mode:
            self.listen_mode = True
            print(">>> Robot entering LISTEN mode.")

        # Apply robot action
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = rb_action
        
        # Update humans
        human_actions = []

        # Switch to follow once the robot has started moving toward the display
        if not self.listen_mode:
            if not self.follow_humans:
                moved_dist = float(np.hypot(rx - self.robot_start_xy[0], ry - self.robot_start_xy[1]))
                if moved_dist >= self.human_follow_distance:
                    self.follow_humans = True

        # Compute repulsion vectors for social distance
        repulsion_vectors = []
        if human_xy.size:
            for i in range(human_xy.shape[0]):
                pos = human_xy[i]
                diff = pos - human_xy
                neighbor_dist = np.linalg.norm(diff, axis=1)
                mask = (neighbor_dist > 1e-6) & (neighbor_dist < self.social_distance)
                if np.any(mask):
                    directions = diff[mask] / neighbor_dist[mask][:, None]
                    strengths = (self.social_distance - neighbor_dist[mask]) / self.social_distance
                    repulsion = (directions * strengths[:, None]).sum(axis=0)
                else:
                    repulsion = np.zeros(2, dtype=np.float32)
                repulsion_vectors.append(self.repulsion_gain * repulsion)
        else:
            repulsion_vectors = [np.zeros(2, dtype=np.float32) for _ in self.humans]

        
        n_humans = len(self.humans)
        follow_radius = 1.0
        fan_angle = 2.0 * self.listen_fan_half_angle
        for i, human in enumerate(self.humans):

            repulsion_vec = repulsion_vectors[i] if i < len(repulsion_vectors) else np.zeros(2, dtype=np.float32)
            relative_angle = 0.0

            ctx = {
                "robot_xy": np.array([rx, ry], dtype=np.float32),
                "robot_yaw": ryaw,
                "repulsion": repulsion_vec,
                "follow_radius": follow_radius,
                "angle_offset": relative_angle,
                "stand_threshold": self.listen_stand_threshold,
            }

            if self.listen_mode:
                human.set_mode(HumanMode.LISTEN)

                if n_humans > 1:
                    relative_angle = (i / (n_humans - 1)) * fan_angle - fan_angle / 2.0
                else:
                    relative_angle = 0.0

                angle = ryaw + relative_angle
                target = np.array(
                    [
                        rx + self.listen_fan_radius * np.cos(angle),
                        ry + self.listen_fan_radius * np.sin(angle),
                    ],
                    dtype=np.float32,
                )
                human.current_waypoint = target

            else:
                human.set_mode(HumanMode.FOLLOWING if self.follow_humans else HumanMode.WANDERING)

                if self.follow_humans:
                    if n_humans > 1:
                        relative_angle = (i / (n_humans - 1)) * fan_angle - fan_angle / 2.0
                    else:
                        relative_angle = 0.0

                    angle = ryaw + np.pi + relative_angle
                    offset = np.array(
                        [follow_radius * np.cos(angle), follow_radius * np.sin(angle)],
                        dtype=np.float32,
                    )
                    human.current_waypoint = np.array([rx, ry], dtype=np.float32) + offset
                else:
                    relative_angle = 0.0


            human_action = human.step(self.model, self.data, ctx)
            human_actions.append(human_action)

            ctrl_idx = 3 + i * 3
            self.data.ctrl[ctrl_idx:ctrl_idx+3] = human_action


        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Refresh poses after the step for reporting
        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        gx, gy = self._get_goal_xy()
        human_goals = np.array(
            [h.current_waypoint for h in self.humans],
            dtype=np.float32
        )
        human_desired_yaw = np.arctan2(
            human_goals[:, 1] - human_xy[:, 1],
            human_goals[:, 0] - human_xy[:, 0],
        ).astype(np.float32)
        human_actions = np.array(human_actions, dtype=np.float32) if human_actions else np.zeros((0, 3), dtype=np.float32)
        human_goal_threshold = 0.5
        human_reached_goal = []
        human_v_follow = np.array([h.last_v_follow for h in self.humans], dtype=np.float32)
        human_v_repulsion = np.array([h.last_v_repulsion for h in self.humans], dtype=np.float32)

        for i, (pos, goal) in enumerate(zip(human_xy, human_goals)):
            dist_to_goal = np.linalg.norm(pos - goal)
            if dist_to_goal < human_goal_threshold:
                human_reached_goal.append(i)

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
            "robot_mode": str(self.robot_mode),
            "listen_mode": bool(self.listen_mode),
            "human_xy": human_xy,          # (N, 2)
            "human_goals": human_goals,    # (N, 2)
            "human_desired_yaw": human_desired_yaw,  # (N,)
            "human_actual_yaw": human_actual_yaw,    # (N,)
            "human_vx": human_actions[:, 0],         # (N,)
            "human_vy": human_actions[:, 1],         # (N,)
            "human_v_yaw": human_actions[:, 2],      # (N,)
            "human_v_follow": human_v_follow,         # (N, 2)
            "human_v_repulsion": human_v_repulsion,   # (N, 2)
            "human_reached_goal": human_reached_goal,
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
