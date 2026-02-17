import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from importlib import resources

from .human import Human, HumanMode
from .robot import Robot


class MuseumEnv(gym.Env):
    """
    Minimal runnable Gymnasium environment for a MuJoCo museum scene.
    """

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

        # Waypoints: room A -> corridor -> room B
        waypoints = [
            (1.0, 5.0),
            (0.6, 4.5),
            (1.0, 2.0),
            (8.5, 2.0),
            (8.5, -10.0),
            (8.5, -12.5),
            (11, -12.5),
        ]

        # Robot agent
        self.robot = Robot(waypoints=waypoints, v_max=1.5, k_v=20.0, k_yaw=20.0)

        # Human follow switch (start with random walking)
        self.follow_humans = False
        self.robot_start_xy = None
        self.human_follow_distance = 1.0

        # Social distance (repulsion) parameters
        self.social_distance = 0.8
        self.repulsion_gain = 6.0

        # Listening formation (fan around robot after it stops)
        self.follow_fan_half_angle = np.deg2rad(85.0)
        self.listen_fan_half_angle = np.deg2rad(75.0)
        self.listen_fan_radius = 1.0
        self.listen_stand_threshold = 0.2
        self.listen_reached_logged = set()

        # Listening wait window
        self.listen_wait_steps = 400
        self.listen_wait_active = False
        self.listen_wait_counter = 0
        self.listen_wait_is_final = False
        self.listen_session_count = 0
        self.overwhelmed_triggered_once = False
        self.overwhelmed_target_idx = 1  # person2
        self.overwhelmed_trigger_wait_step = 200

        # --- Initialize humans ---
        self.humans = [
            Human("person1", "person1", qpos_idx=3, max_speed=1.67),
            Human("person2", "person2", qpos_idx=6, max_speed=1.67),
            Human("person3", "person3", qpos_idx=9, max_speed=1.67),
            Human("person4", "person4", qpos_idx=12, max_speed=1.67),
            Human("person5", "person5", qpos_idx=15, max_speed=1.67),
        ]
        for human in self.humans:
            human.external_waypoint = False
            human.set_mode(HumanMode.WANDERING)

        self.humans[0].can_be_distracted = True
        self.humans[0].distracted_probability = 0.0005
        self.humans[1].can_be_overwhelmed = True

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

        # Reset listening state
        self.listen_reached_logged = set()
        self.listen_wait_active = False
        self.listen_wait_counter = 0
        self.listen_wait_is_final = False
        self.listen_session_count = 0
        self.overwhelmed_triggered_once = False

        # Reset humans
        for human in self.humans:
            human.reset_episode_state()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action=None):
        """
        Robot rule-based navigation (via Robot class) + human walking.
        """
        self.step_count += 1

        if self.listen_wait_active:
            return self._step_waiting_branch()

        return self._step_active_branch()

    def _step_waiting_branch(self):
        events = {
            "entered_listen": False,
            "started_listen_wait": False,
            "completed_listen_wait": False,
            "final_listen_ready": False,
            "overwhelmed_triggered": False,
        }

        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)

        robot_xy = np.array([rx, ry], dtype=np.float32)
        events["overwhelmed_triggered"] = self._maybe_trigger_overwhelmed_in_wait(
            robot_xy=robot_xy,
            human_xy=human_xy,
        )

        # Freeze everyone during explanation window.
        self.data.ctrl[:] = 0.0
        human_actions = np.zeros((len(self.humans), 3), dtype=np.float32)

        # Let only person2 move if they are overwhelmed.
        tgt_idx = self.overwhelmed_target_idx
        if 0 <= tgt_idx < len(self.humans):
            tgt_human = self.humans[tgt_idx]
            if tgt_human.mode == HumanMode.OVERWHELMED:
                ctx = {
                    "robot_xy": robot_xy,
                    "robot_yaw": ryaw,
                    "repulsion": np.zeros(2, dtype=np.float32),
                    "stand_threshold": self.listen_stand_threshold,
                }
                tgt_action = tgt_human.step(self.model, self.data, ctx)
                human_actions[tgt_idx] = tgt_action
                ctrl_idx = 3 + tgt_idx * 3
                self.data.ctrl[ctrl_idx:ctrl_idx + 3] = tgt_action

        mujoco.mj_step(self.model, self.data)
        self.listen_wait_counter += 1

        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        wx, wy = self.robot.get_current_waypoint()
        dist = float(np.hypot(wx - rx, wy - ry) + 1e-8)
        desired_yaw = float(np.arctan2(wy - ry, wx - rx))
        actual_yaw = float(ryaw)

        human_goals = np.array([h.current_waypoint for h in self.humans], dtype=np.float32)
        human_reached_goal = self._check_human_goals(human_xy, human_goals)

        final_waypoint_reached = self.robot.is_final_reached(dist)
        all_humans_reached = len(self.humans) > 0 and len(human_reached_goal) == len(self.humans)

        if self.listen_wait_counter >= self.listen_wait_steps:
            events["completed_listen_wait"] = True
            if self.listen_wait_is_final:
                events["final_listen_ready"] = True
                print(">>> Listening wait complete at final display.")
            else:
                self.robot.on_listening_complete()
                self.follow_humans = False
                self.robot_start_xy = np.array([rx, ry], dtype=np.float32)
                print(">>> Listening wait complete. Resume MOVE to Room B.")

            self.listen_wait_active = False
            self.listen_wait_counter = 0
            self.listen_wait_is_final = False

        human_v_follow = np.zeros((len(self.humans), 2), dtype=np.float32)
        human_v_repulsion = np.zeros((len(self.humans), 2), dtype=np.float32)
        human_v_hr = np.zeros((len(self.humans), 2), dtype=np.float32)

        snapshot = self._collect_step_snapshot(
            robot_pose=(rx, ry, ryaw),
            dist=dist,
            desired_yaw=desired_yaw,
            actual_yaw=actual_yaw,
            robot_mode=str(self.robot.mode),
            robot_action=np.zeros(3, dtype=np.float32),
            human_xy=human_xy,
            human_actual_yaw=human_actual_yaw,
            human_goals=human_goals,
            human_actions=human_actions,
            human_v_follow=human_v_follow,
            human_v_repulsion=human_v_repulsion,
            human_v_hr=human_v_hr,
            human_reached_goal=human_reached_goal,
            final_waypoint_reached=final_waypoint_reached,
            all_humans_reached=all_humans_reached,
        )

        obs = self._get_obs()
        reward = -dist
        terminated = events["final_listen_ready"]
        truncated = self.step_count >= self.max_steps

        info = self._build_info(snapshot, events, truncated)
        return obs, reward, terminated, truncated, info

    def _maybe_trigger_overwhelmed_in_wait(self, robot_xy, human_xy):
        # Trigger on the 200th waiting step of the first listen session only once per episode.
        if self.overwhelmed_triggered_once:
            return False
        if self.listen_session_count != 1:
            return False
        if (self.listen_wait_counter + 1) != self.overwhelmed_trigger_wait_step:
            return False

        idx = self.overwhelmed_target_idx
        if idx < 0 or idx >= len(self.humans) or idx >= human_xy.shape[0]:
            return False

        human = self.humans[idx]
        if not human.can_be_overwhelmed:
            return False

        human.start_overwhelmed(robot_xy=robot_xy, current_xy=human_xy[idx])
        self.overwhelmed_triggered_once = True
        return True

    def _step_active_branch(self):
        events = {
            "entered_listen": False,
            "started_listen_wait": False,
            "completed_listen_wait": False,
            "final_listen_ready": False,
            "overwhelmed_triggered": False,
        }

        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        # --- Robot decision ---
        robot_pose = self._get_robot_pose()
        robot_out = self.robot.step(robot_pose=robot_pose, human_xyz=human_xyz)

        rb_action = robot_out["action"]
        dist = robot_out["dist"]
        desired_yaw = robot_out["desired_yaw"]
        actual_yaw = robot_out["actual_yaw"]
        robot_mode = robot_out["mode"]
        enter_listen = robot_out["enter_listen"]
        events["entered_listen"] = bool(enter_listen)

        # If robot just entered listen, assign listen targets
        if enter_listen:
            rx, ry, ryaw = robot_pose
            self.listen_reached_logged = set()
            self.listen_wait_active = False
            self.listen_wait_counter = 0
            self.listen_wait_is_final = False
            self.listen_session_count += 1
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

        rx, ry, ryaw = self._get_robot_pose()

        # Switch to follow once the robot has started moving toward the display
        if not self.robot.listen_mode and not self.follow_humans:
            moved_dist = float(np.hypot(rx - self.robot_start_xy[0], ry - self.robot_start_xy[1]))
            if moved_dist >= self.human_follow_distance:
                self.follow_humans = True

        repulsion_vectors = self._compute_social_repulsion(human_xy)
        human_actions = self._update_humans_and_apply_ctrl(
            rx=rx,
            ry=ry,
            ryaw=ryaw,
            repulsion_vectors=repulsion_vectors,
        )

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Refresh poses after the step for reporting
        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        human_goals = np.array([h.current_waypoint for h in self.humans], dtype=np.float32)
        human_reached_goal = self._check_human_goals(human_xy, human_goals)

        human_v_follow = np.array([h.last_v_follow for h in self.humans], dtype=np.float32)
        human_v_repulsion = np.array([h.last_v_repulsion for h in self.humans], dtype=np.float32)
        human_v_hr = np.array([h.last_v_hr for h in self.humans], dtype=np.float32)

        final_waypoint_reached = self.robot.is_final_reached(dist)
        all_humans_reached = len(self.humans) > 0 and len(human_reached_goal) == len(self.humans)
        events.update(self._handle_listen_transitions(final_waypoint_reached, all_humans_reached))

        snapshot = self._collect_step_snapshot(
            robot_pose=(rx, ry, ryaw),
            dist=float(dist),
            desired_yaw=float(desired_yaw),
            actual_yaw=float(actual_yaw),
            robot_mode=str(robot_mode),
            robot_action=np.array(rb_action, dtype=np.float32),
            human_xy=human_xy,
            human_actual_yaw=human_actual_yaw,
            human_goals=human_goals,
            human_actions=human_actions,
            human_v_follow=human_v_follow,
            human_v_repulsion=human_v_repulsion,
            human_v_hr=human_v_hr,
            human_reached_goal=human_reached_goal,
            final_waypoint_reached=final_waypoint_reached,
            all_humans_reached=all_humans_reached,
        )

        obs = self._get_obs()
        reward = -float(dist)
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = self._build_info(snapshot, events, truncated)
        return obs, reward, terminated, truncated, info

    def _compute_social_repulsion(self, human_xy):
        if not human_xy.size:
            return [np.zeros(2, dtype=np.float32) for _ in self.humans]

        repulsion_vectors = []
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

        return repulsion_vectors

    def _update_humans_and_apply_ctrl(self, rx, ry, ryaw, repulsion_vectors):
        human_actions = []
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
                if human.mode != HumanMode.OVERWHELMED:
                    human.set_mode(HumanMode.LISTENING)
            else:
                if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED):
                    human.set_mode(HumanMode.FOLLOWING if self.follow_humans else HumanMode.WANDERING)

                if self.follow_humans and human.mode == HumanMode.FOLLOWING:
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

        if human_actions:
            return np.array(human_actions, dtype=np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    def _check_human_goals(self, human_xy, human_goals):
        human_goal_threshold = 0.2
        human_reached_goal = []
        for i, (pos, goal) in enumerate(zip(human_xy, human_goals)):
            dist_to_goal = float(np.linalg.norm(pos - goal))
            if dist_to_goal < human_goal_threshold:
                human_reached_goal.append(i)
                if self.robot.listen_mode and i not in self.listen_reached_logged:
                    self.listen_reached_logged.add(i)
                    print(f">>> person{i+1} reached their goal at step {self.step_count}!")
        return human_reached_goal

    def _handle_listen_transitions(self, final_waypoint_reached, all_humans_reached):
        events = {
            "started_listen_wait": False,
            "completed_listen_wait": False,
            "final_listen_ready": False,
        }

        # Listening complete condition: all humans reached.
        # Start waiting window instead of switching immediately.
        if self.robot.listen_mode and all_humans_reached and not self.listen_wait_active:
            self.listen_wait_active = True
            self.listen_wait_counter = 0
            self.listen_wait_is_final = bool(final_waypoint_reached)
            events["started_listen_wait"] = True
            print(f">>> Listening targets reached. Start wait for {self.listen_wait_steps} steps.")

        return events

    def _collect_step_snapshot(
        self,
        robot_pose,
        dist,
        desired_yaw,
        actual_yaw,
        robot_mode,
        robot_action,
        human_xy,
        human_actual_yaw,
        human_goals,
        human_actions,
        human_v_follow,
        human_v_repulsion,
        human_v_hr,
        human_reached_goal,
        final_waypoint_reached,
        all_humans_reached,
    ):
        rx, ry, _ = robot_pose
        gx, gy = self._get_goal_xy()

        human_desired_yaw = np.arctan2(
            human_goals[:, 1] - human_xy[:, 1],
            human_goals[:, 0] - human_xy[:, 0],
        ).astype(np.float32)

        return {
            "robot_xy": np.array([rx, ry], dtype=np.float32),
            "robot_goal_xy": np.array([gx, gy], dtype=np.float32),
            "dist_to_goal": float(dist),
            "robot_yaw": float(actual_yaw),
            "robot_desired_yaw": float(desired_yaw),
            "robot_mode": str(robot_mode),
            "robot_action": np.array(robot_action, dtype=np.float32),
            "human_xy": human_xy,
            "human_goals": human_goals,
            "human_actual_yaw": human_actual_yaw,
            "human_desired_yaw": human_desired_yaw,
            "human_actions": human_actions,
            "human_v_follow": human_v_follow,
            "human_v_repulsion": human_v_repulsion,
            "human_v_hr": human_v_hr,
            "human_v_total": human_v_follow + human_v_repulsion + human_v_hr,
            "human_mode": [h.mode for h in self.humans],
            "human_distracted_timer": np.array([h.distracted_timer for h in self.humans], dtype=np.int32),
            "human_overwhelmed_stage": [h.overwhelmed_stage for h in self.humans],
            "human_overwhelmed_leave_timer": np.array(
                [h.overwhelmed_leave_timer for h in self.humans], dtype=np.int32
            ),
            "human_reached_goal": human_reached_goal,
            "final_waypoint_reached": bool(final_waypoint_reached),
            "all_humans_reached": bool(all_humans_reached),
        }

    def _build_info(self, snapshot, events, truncated):
        listen_wait_remaining = (
            max(0, self.listen_wait_steps - self.listen_wait_counter) if self.listen_wait_active else 0
        )

        terminated_reason = None
        if events["final_listen_ready"]:
            terminated_reason = "final_listen_ready"
        elif truncated:
            terminated_reason = "max_steps"

        robot_action = snapshot["robot_action"]
        human_actions = snapshot["human_actions"]

        return {
            "events": {
                "entered_listen": bool(events["entered_listen"]),
                "started_listen_wait": bool(events["started_listen_wait"]),
                "completed_listen_wait": bool(events["completed_listen_wait"]),
                "final_listen_ready": bool(events["final_listen_ready"]),
                "overwhelmed_triggered": bool(events["overwhelmed_triggered"]),
            },
            "status": {
                "step_count": int(self.step_count),
                "listen_mode": bool(self.robot.listen_mode),
                "listen_wait": {
                    "active": bool(self.listen_wait_active),
                    "counter": int(self.listen_wait_counter),
                    "steps": int(self.listen_wait_steps),
                    "remaining": int(listen_wait_remaining),
                    "is_final": bool(self.listen_wait_is_final),
                },
                "terminated_reason": terminated_reason,
            },
            "robot": {
                "pose_xy": snapshot["robot_xy"],
                "goal_xy": snapshot["robot_goal_xy"],
                "dist_to_goal": float(snapshot["dist_to_goal"]),
                "yaw": float(snapshot["robot_yaw"]),
                "desired_yaw": float(snapshot["robot_desired_yaw"]),
                "mode": str(snapshot["robot_mode"]),
                "action": {
                    "vx": float(robot_action[0]),
                    "vy": float(robot_action[1]),
                    "yaw_rate": float(robot_action[2]),
                },
                "final_waypoint_reached": bool(snapshot["final_waypoint_reached"]),
            },
            "humans": {
                "pose_xy": snapshot["human_xy"],
                "goal_xy": snapshot["human_goals"],
                "actual_yaw": snapshot["human_actual_yaw"],
                "desired_yaw": snapshot["human_desired_yaw"],
                "mode": snapshot["human_mode"],
                "distracted_timer": snapshot["human_distracted_timer"],
                "overwhelmed_stage": snapshot["human_overwhelmed_stage"],
                "overwhelmed_leave_timer": snapshot["human_overwhelmed_leave_timer"],
                "reached_goal_indices": snapshot["human_reached_goal"],
                "all_reached": bool(snapshot["all_humans_reached"]),
                "action": {
                    "vx": human_actions[:, 0],
                    "vy": human_actions[:, 1],
                    "yaw_rate": human_actions[:, 2],
                },
                "velocity_components": {
                    "follow": snapshot["human_v_follow"],
                    "repulsion": snapshot["human_v_repulsion"],
                    "human_robot": snapshot["human_v_hr"],
                    "total": snapshot["human_v_total"],
                },
            },
        }

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
