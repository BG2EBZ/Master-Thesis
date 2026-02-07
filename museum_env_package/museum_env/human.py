import numpy as np


class HumanMode:
    WANDERING = "wandering"
    FOLLOWING = "following"
    LISTEN = "listen"


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
        
        # Store body_id (will be set when we have access to model)
        self.body_id = None
        
        # Current target waypoint
        self.current_waypoint = self._random_waypoint()
        self.step_count = 0
        # Cache for debugging/metrics
        self.last_v_follow = np.zeros(2, dtype=np.float32)
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)

    def set_mode(self, mode: str):
        if mode not in (HumanMode.WANDERING, HumanMode.FOLLOWING, HumanMode.LISTEN):
            raise ValueError(f"Unknown human mode: {mode}")
        self.mode = mode

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

        if self.mode == HumanMode.WANDERING:
            return self._step_wandering(data, ctx)

        elif self.mode == HumanMode.FOLLOWING:
            return self._step_following(data, ctx)

        elif self.mode == HumanMode.LISTEN:
            return self._step_listening(data, ctx)

        else:
            raise ValueError(f"Unknown human mode {self.mode}")


    def _step_wandering(self, data, ctx):
        x, y, yaw = self._get_pose(data)

        dx = self.current_waypoint[0] - x
        dy = self.current_waypoint[1] - y
        dist = np.hypot(dx, dy)

        if dist < self.waypoint_threshold:
            self.current_waypoint = self._random_waypoint()
            dx = self.current_waypoint[0] - x
            dy = self.current_waypoint[1] - y

        return self._move(dx, dy, yaw, ctx)

    def _step_following(self, data, ctx):
        robot_xy = ctx["robot_xy"]
        robot_yaw = ctx["robot_yaw"]

        follow_radius = ctx.get("follow_radius", 1.0)
        angle_offset = ctx.get("angle_offset", 0.0)

        target = robot_xy + follow_radius * np.array([
            np.cos(robot_yaw + np.pi + angle_offset),
            np.sin(robot_yaw + np.pi + angle_offset),
        ])

        self.current_waypoint = target
        x, y, yaw = self._get_pose(data)
        dx = target[0] - x
        dy = target[1] - y

        return self._move(dx, dy, yaw, ctx)
    
    def _step_listening(self, data, ctx):
        x, y, yaw = self._get_pose(data)
        dx = self.current_waypoint[0] - x
        dy = self.current_waypoint[1] - y
        dist = np.hypot(dx, dy)

        if dist < ctx.get("stand_threshold", 0.4):
            return np.zeros(3)

        return self._move(dx, dy, yaw, ctx)

    def _move(self, dx, dy, yaw, ctx):
        repulsion = ctx.get("repulsion", np.zeros(2))

        dist = np.hypot(dx, dy)
        if dist > 1e-6:
            v_follow = self.max_speed * np.array([dx, dy]) / dist
        else:
            v_follow = np.zeros(2)

        v_repulsion = np.array(repulsion, dtype=np.float32)

        self.last_v_follow = v_follow.copy()
        self.last_v_repulsion = v_repulsion.copy()

        v_total = v_follow + v_repulsion
        speed = np.linalg.norm(v_total)

        if speed > self.max_speed:
            v_total = v_total / speed * self.max_speed

        desired_yaw = np.arctan2(v_total[1], v_total[0]) if speed > 1e-6 else yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        if abs(yaw_err) > np.deg2rad(5):
            return np.array([0.0, 0.0, 20.0 * yaw_err])

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
    
