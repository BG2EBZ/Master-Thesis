import numpy as np


class HumanMode:
    WANDERING = "wandering"
    FOLLOWING = "following"
    LISTENING = "listening"


class Human:
    """
    Minimal human behavior: random walking in the museum.
    """
    
    def __init__(self, name, body_name, qpos_idx, max_speed=2.5, waypoint_threshold=0.2):
        """
        Args:
            name: Human identifier (e.g., "person1")
            body_name: MuJoCo body name (e.g., "person1")
            qpos_idx: Starting index in qpos for this human's [x, y, yaw]
            max_speed: Maximum walking speed (m/s)
            waypoint_threshold: Distance to reach waypoint before picking new one
        """
        self.name = name
        self.body_name = body_name
        self.qpos_idx = qpos_idx  # qpos[qpos_idx:qpos_idx+3] = [x, y, yaw]
        self.max_speed = max_speed
        self.waypoint_threshold = waypoint_threshold
        # If True, current_waypoint is managed externally (e.g., follow robot)
        self.external_waypoint = False
        self.mode = HumanMode.WANDERING

        self.context = {}
        
        # Store body_id (will be set when we have access to model)
        self.body_id = None
        self.x_dof_idx = None
        self.y_dof_idx = None
        
        # Current target waypoint
        self.current_waypoint = self._random_waypoint()
        self.step_count = 0
        # Cache for debugging/metrics
        self.last_v_follow = np.zeros(2, dtype=np.float32)
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)  # human–robot force

    def set_mode(self, mode: str):
        if mode not in (HumanMode.WANDERING, HumanMode.FOLLOWING, HumanMode.LISTENING):
            raise ValueError(f"Unknown human mode: {mode}")
        self.mode = mode


    def set_context(self, **kwargs):
        """
        Store high-level social context provided by env.
        Example:
            mode="listening"
            index=0
            n_humans=3
            robot_pose=(rx, ry, ryaw)
        """
        self.context = kwargs

    def _assign_target_from_context(self):
        """
        Decide where to stand based on social context.
        """
        if not self.context:
            return

        index = self.context.get("index", 0)
        n_humans = self.context.get("n_humans", 1)
        robot_pose = self.context.get("robot_pose", None)

        if robot_pose is None:
            return

        rx, ry, ryaw = robot_pose

        if self.mode == HumanMode.LISTENING:
            fan_half_angle = self.context.get("fan_half_angle", np.pi / 6)
            radius = self.context.get("listen_radius", 1.2)

            if n_humans > 1:
                rel = (index / (n_humans - 1)) * (2 * fan_half_angle) - fan_half_angle
            else:
                rel = 0.0

            angle = ryaw + rel
            self.current_waypoint = np.array(
                [rx + radius * np.cos(angle),
                ry + radius * np.sin(angle)],
                dtype=np.float32
            )

        elif self.mode == HumanMode.FOLLOWING:
            fan_half_angle = self.context.get("fan_half_angle", np.pi / 6)
            radius = self.context.get("follow_radius", 1.0)

            if n_humans > 1:
                rel = (index / (n_humans - 1)) * (2 * fan_half_angle) - fan_half_angle
            else:
                rel = 0.0

            angle = ryaw + np.pi + rel
            self.current_waypoint = np.array(
                [rx + radius * np.cos(angle),
                ry + radius * np.sin(angle)],
                dtype=np.float32
            )
    
    def step(self, model, data, ctx):
        """
        ctx: dict provided by env, e.g.
        {
            "dt": timestep,
            "robot_xy": np.array([x, y]),
            "robot_yaw": yaw,
            "repulsion": np.array([rx, ry]),
        }
        """
        if self.body_id is None:
            self.body_id = model.body(self.body_name).id

        if self.x_dof_idx is None:
            self.x_dof_idx = model.jnt_dofadr[model.joint(f"{self.name}_x").id]
            self.y_dof_idx = model.jnt_dofadr[model.joint(f"{self.name}_y").id]

        if self.mode == HumanMode.WANDERING:
            return self._step_wandering(data, ctx)

        if self.mode == HumanMode.FOLLOWING:
            self._assign_target_from_context()
            return self._step_following(data, ctx)

        if self.mode == HumanMode.LISTENING:
            return self._step_listening(data, ctx)

        raise ValueError(f"Unknown human mode {self.mode}")
        
    def assign_follow_target(self, index, n_humans, robot_pose, follow_radius, fan_half_angle):
        """
        Compute and assign target for FOLLOWING behavior.
        """
        rx, ry, ryaw = robot_pose

        if n_humans > 1:
            relative_angle = (index / (n_humans - 1)) * (2 * fan_half_angle) - fan_half_angle
        else:
            relative_angle = 0.0

        angle = ryaw + np.pi + relative_angle
        offset = np.array(
            [
                follow_radius * np.cos(angle),
                follow_radius * np.sin(angle),
            ],
            dtype=np.float32,
        )

        self.current_waypoint = np.array([rx, ry], dtype=np.float32) + offset


    def assign_listen_target(self, index, n_humans, robot_pose, listen_radius, fan_half_angle):
        """
        Compute and assign target for LISTEN behavior.
        """
        rx, ry, ryaw = robot_pose

        if n_humans > 1:
            relative_angle = (index / (n_humans - 1)) * (2 * fan_half_angle) - fan_half_angle
        else:
            relative_angle = 0.0

        angle = ryaw + relative_angle
        self.current_waypoint = np.array(
            [
                rx + listen_radius * np.cos(angle),
                ry + listen_radius * np.sin(angle),
            ],
            dtype=np.float32,
        )


    def _step_wandering(self, data, ctx):
        x, y, yaw = self._get_pose(data)

        dx = self.current_waypoint[0] - x
        dy = self.current_waypoint[1] - y
        dist = np.hypot(dx, dy)

        if dist < self.waypoint_threshold:
            self.current_waypoint = self._random_waypoint()
            dx = self.current_waypoint[0] - x
            dy = self.current_waypoint[1] - y

        return self._move(dx, dy, yaw, data, ctx)

    def _step_following(self, data, ctx):
        x, y, yaw = self._get_pose(data)
        dx = self.current_waypoint[0] - x
        dy = self.current_waypoint[1] - y
        return self._move(dx, dy, yaw, data, ctx)
    
    def _step_listening(self, data, ctx):
        x, y, yaw = self._get_pose(data)
        dx = self.current_waypoint[0] - x
        dy = self.current_waypoint[1] - y
        dist = np.hypot(dx, dy)

        stand_threshold = ctx.get("stand_threshold")

        if dist >= stand_threshold:
            return self._move(dx, dy, yaw, data, ctx)
        
        robot_xy = ctx.get("robot_xy")
        if robot_xy is None:
            return np.zeros(3, dtype=np.float32)
        
        rx, ry = robot_xy[0], robot_xy[1]
        desired_yaw = np.arctan2(ry - y, rx - x)
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        if abs(yaw_err) < np.deg2rad(3.0):
            return np.zeros(3, dtype=np.float32)
        
        return np.array([0.0, 0.0, 20.0 * yaw_err])

    def _move(self, dx, dy, yaw, data, ctx):
        robot_xy = ctx.get("robot_xy", None)

        repulsion = ctx.get("repulsion", np.zeros(2))
        v_hr = np.zeros(2, dtype=np.float32)

        if robot_xy is not None:
            # current human position
            hx, hy, _ = self._get_pose(data)

            diff_hr = np.array([hx, hy], dtype=np.float32) - robot_xy
            dist_hr = np.linalg.norm(diff_hr) + 1e-6
            dir_hr = diff_hr / dist_hr

        # preferred human–robot distance (meters)
        d_pref = 1.0
        d_min = 1.0   # too close → repulsion
        d_max = 2.0   # too far → attraction

        if dist_hr < d_min:
            # too close → repulsion (slow down / move away)
            k_rep_hr = 1.5
            v_hr = k_rep_hr * (d_min - dist_hr) * dir_hr

        elif dist_hr > d_max:
            # too far → attraction (move towards robot)
            k_att_hr = 0.8
            v_hr = -k_att_hr * (dist_hr - d_max) * dir_hr

        dist = np.hypot(dx, dy)
        if dist > 1e-6:
            v_follow = self.max_speed * np.array([dx, dy]) / dist
        else:
            v_follow = np.zeros(2)

        v_repulsion = np.array(repulsion, dtype=np.float32)

        self.last_v_follow = v_follow.copy()
        self.last_v_repulsion = v_repulsion.copy()
        self.last_v_hr = v_hr.copy()

        v_total = v_follow + v_repulsion + v_hr
        speed = np.linalg.norm(v_total)

        if speed > self.max_speed:
            v_total = v_total / speed * self.max_speed

        desired_yaw = np.arctan2(v_total[1], v_total[0]) if speed > 1e-6 else yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        # if abs(yaw_err) > np.deg2rad(5):
        #     # hard stop translation while turning
        #     # data.qvel[self.x_dof_idx] = 0.0
        #     # data.qvel[self.y_dof_idx] = 0.0
        #     return np.array([0.0, 0.0, 20.0 * yaw_err])

        return np.array([v_total[0], v_total[1], 20.0 * yaw_err])
    
    # -------------------------
    # Helpers
    # -------------------------

    def _get_pose(self, data):
        x = float(data.xpos[self.body_id, 0])
        y = float(data.xpos[self.body_id, 1])
        yaw = float(data.qpos[self.qpos_idx + 2])
        return x, y, yaw

    def _random_waypoint(self):
        """Generate random waypoint within museum bounds"""
        # Room A: x[0,10], y[0,10]
        # Corridor: x[7,10], y[-10,0]
        # Room B: x[7,12], y[-15,-10]

        wx = np.random.uniform(1, 9)
        wy = np.random.uniform(1, 9)        
        
        return np.array([wx, wy], dtype=np.float32)
    
    def _wrap_to_pi(self, ang):
        return (ang + np.pi) % (2 * np.pi) - np.pi
    
