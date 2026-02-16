import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from importlib import resources

from .human import Human, HumanMode
from .robot import Robot, RobotMode


class MuseumEnv(gym.Env):
    """
    Minimal runnable Gymnasium environment for a MuJoCo museum scene.
    """
    # metadata = {"render_modes": ["human"], "render_fps": 60}
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

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
        self.renderer = None
        self.render_width = 1920
        self.render_height = 1080

        # --- Action space ---
        self.nu = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=np.float32,
        )

        # --- Observation space ---
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.timestep = self.model.opt.timestep
        self.max_steps = 100000
        self.step_count = 0

        # Waypoints: room A → corridor → room B
        waypoints = [
            (1.0, 5.0),
            (0.6, 5.0),
            (1.0, 2.0),
            (8.5, 2.0),
            (8.5, -10.0),
            (8.5, -12.5),
            (11, -12.5),
        ]

        # Robot agent
        self.robot = Robot(waypoints=waypoints, v_max=3.0, k_v=20.0, k_yaw=20.0)

        # Human follow switch (start with random walking)
        self.follow_humans = False
        self.robot_start_xy = None
        self.human_follow_distance = 1.0

        # Social distance (repulsion) parameters
        self.social_distance = 0.8
        self.repulsion_gain = 6.0

        # Listening formation (fan around robot after it stops)
        self.follow_fan_half_angle = np.deg2rad(85.0)  # following fan
        self.listen_fan_half_angle = np.deg2rad(75.0)  # listening fan
        self.listen_fan_radius = 0.8
        self.listen_stand_threshold = 0.2
        self.listen_reached_logged = set()

        # --- Initialize humans ---
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

    def _get_robot_pose(self):
        robot_body_id = self.model.body("robot").id
        x = float(self.data.xpos[robot_body_id, 0])
        y = float(self.data.xpos[robot_body_id, 1])
        yaw = float(self.data.qpos[2])
        return x, y, yaw

    def _get_human_poses(self):
        humans_xyz = []
        for human in self.humans:
            human_body_id = self.model.body(human.body_name).id
            x = float(self.data.xpos[human_body_id, 0])
            y = float(self.data.xpos[human_body_id, 1])
            yaw = float(self.data.qpos[human.qpos_idx + 2])
            humans_xyz.append([x, y, yaw])
        return np.array(humans_xyz, dtype=np.float32)

    def _get_goal_xy(self):
        goal_xy = self.data.site("goal_site").xpos[:2]
        return float(goal_xy[0]), float(goal_xy[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0

        # Reset robot agent
        self.robot.reset()

        # Store robot start position (for follow trigger)
        rx, ry, _ = self._get_robot_pose()
        self.robot_start_xy = np.array([rx, ry], dtype=np.float32)
        self.follow_humans = False

        # Reset listening logging
        self.listen_reached_logged = set()

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
        Robot rule-based navigation (via Robot class) + human walking.
        """
        self.step_count += 1

        # Get human poses once per step (needed for robot to face crowd)
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        # --- Robot decision (NEW) ---
        robot_pose = self._get_robot_pose()
        robot_out = self.robot.step(robot_pose=robot_pose, human_xyz=human_xyz)

        rb_action = robot_out["action"]
        dist = robot_out["dist"]
        desired_yaw = robot_out["desired_yaw"]
        actual_yaw = robot_out["actual_yaw"]
        robot_mode = robot_out["mode"]
        enter_listen = robot_out["enter_listen"]

        # If robot just entered listen, assign listen targets
        if enter_listen:
            rx, ry, ryaw = robot_pose
            self.listen_reached_logged = set()
            n_humans = len(self.humans)

            print(f">>> Robot entering LISTEN mode. robot=({rx:.2f}, {ry:.2f}, yaw={ryaw:.2f})")
            for i, human in enumerate(self.humans):
                human.assign_listen_target(
                    index=i,
                    n_humans=n_humans,
                    robot_pose=(rx, ry, ryaw),
                    listen_radius=self.listen_fan_radius,
                    fan_half_angle=self.listen_fan_half_angle,
                )
                gx, gy = human.current_waypoint
                print(f"    person{i+1} listen_goal=({gx:.3f}, {gy:.3f})")

        # Apply robot action
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = rb_action

        # --- Update humans (unchanged logic, but listen flag now from self.robot.listen_mode) ---
        human_actions = []

        rx, ry, ryaw = self._get_robot_pose()

        # Switch to follow once the robot has started moving toward the display
        if not self.robot.listen_mode:
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

        for i, human in enumerate(self.humans):
            repulsion_vec = repulsion_vectors[i] if i < len(repulsion_vectors) else np.zeros(2, dtype=np.float32)

            ctx = {
                "robot_xy": np.array([rx, ry], dtype=np.float32),
                "robot_yaw": ryaw,
                "repulsion": repulsion_vec,
                "stand_threshold": self.listen_stand_threshold,
            }

            if self.robot.listen_mode:
                human.set_mode(HumanMode.LISTENING)
            else:
                human.set_mode(HumanMode.FOLLOWING if self.follow_humans else HumanMode.WANDERING)
                if self.follow_humans:
                    human.set_context(
                        index=i,
                        n_humans=n_humans,
                        robot_pose=(rx, ry, ryaw),
                        follow_radius=follow_radius,
                        fan_half_angle=self.follow_fan_half_angle,
                    )

            human_action = human.step(self.model, self.data, ctx)
            human_actions.append(human_action)

            ctrl_idx = 3 + i * 3
            self.data.ctrl[ctrl_idx:ctrl_idx + 3] = human_action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Refresh poses after the step for reporting
        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        gx, gy = self._get_goal_xy()
        human_goals = np.array([h.current_waypoint for h in self.humans], dtype=np.float32)
        human_desired_yaw = np.arctan2(
            human_goals[:, 1] - human_xy[:, 1],
            human_goals[:, 0] - human_xy[:, 0],
        ).astype(np.float32)

        human_actions = np.array(human_actions, dtype=np.float32) if human_actions else np.zeros((0, 3), dtype=np.float32)
        human_goal_threshold = 0.2
        human_reached_goal = []

        human_v_follow = np.array([h.last_v_follow for h in self.humans], dtype=np.float32)
        human_v_repulsion = np.array([h.last_v_repulsion for h in self.humans], dtype=np.float32)
        human_v_hr = np.array([h.last_v_hr for h in self.humans], dtype=np.float32)

        for i, (pos, goal) in enumerate(zip(human_xy, human_goals)):
            dist_to_goal = float(np.linalg.norm(pos - goal))
            if dist_to_goal < human_goal_threshold:
                human_reached_goal.append(i)
                if self.robot.listen_mode and i not in self.listen_reached_logged:
                    self.listen_reached_logged.add(i)
                    print(f">>> person{i+1} reached their goal at step {self.step_count}!")

        final_waypoint_reached = self.robot.is_final_reached(dist)
        all_humans_reached = len(self.humans) > 0 and len(human_reached_goal) == len(self.humans)

        # Listening complete condition: all humans reached
        if self.robot.listen_mode and all_humans_reached:
            if not final_waypoint_reached:
                self.robot.on_listening_complete()

                # Reuse startup behavior: robot departs first, humans follow after 0.5m.
                self.follow_humans = False
                self.robot_start_xy = np.array([rx, ry], dtype=np.float32)

                print(">>> Listening complete. Resume MOVE to Room B.")

        final_listen_ready = final_waypoint_reached and self.robot.listen_mode and all_humans_reached

        obs = self._get_obs()

        reward = -dist
        terminated = final_listen_ready
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
            "robot_mode": str(robot_mode),
            "listen_mode": bool(self.robot.listen_mode),
            "human_xy": human_xy,
            "human_goals": human_goals,
            "human_desired_yaw": human_desired_yaw,
            "human_actual_yaw": human_actual_yaw,
            "human_vx": human_actions[:, 0],
            "human_vy": human_actions[:, 1],
            "human_v_yaw": human_actions[:, 2],
            "human_v_follow": human_v_follow,
            "human_v_repulsion": human_v_repulsion,
            "human_v_hr": human_v_hr,
            "human_v_total": human_v_follow + human_v_repulsion + human_v_hr,
            "human_reached_goal": human_reached_goal,
            "final_waypoint_reached": bool(final_waypoint_reached),
            "all_humans_reached": bool(all_humans_reached),
            "final_listen_ready": bool(final_listen_ready),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        x, y, yaw = self._get_robot_pose()
        gx, gy = self._get_goal_xy()
        return np.array([x, y, gx - x, gy - y], dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                vis = getattr(self.model, "vis", None)
                global_vis = getattr(vis, "global_", None) if vis is not None else None
                offwidth = int(getattr(global_vis, "offwidth", 640))
                offheight = int(getattr(global_vis, "offheight", 480))

                width = max(1, min(self.render_width, offwidth))
                height = max(1, min(self.render_height, offheight))

                try:
                    self.renderer = mujoco.Renderer(self.model, height=height, width=width)
                except ValueError:
                    # Last-resort fallback to MuJoCo defaults when XML/driver limits are tighter.
                    self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
