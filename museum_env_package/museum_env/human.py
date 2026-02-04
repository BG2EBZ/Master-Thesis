import numpy as np


class Human:
    """
    Minimal human behavior: random walking in the museum.
    """
    
    def __init__(self, name, body_name, qpos_idx, max_speed=2, waypoint_threshold=0.5):
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
        
        # Store body_id (will be set when we have access to model)
        self.body_id = None
        
        # Current target waypoint
        self.current_waypoint = self._random_waypoint()
        self.step_count = 0
    
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
    
    def update_behavior(self, robot_xy, display_xy, dt):
        hx, hy = self.get_xy()
        rx, ry = robot_xy

        dist_to_robot = np.linalg.norm([hx - rx, hy - ry])
        dist_to_display = np.linalg.norm([hx - display_xy[0], hy - display_xy[1]])

        # standing → wandering (after some time)
        if self.behavior == "standing":
            self.stand_timer -= dt
            if self.stand_timer <= 0:
                self.behavior = "wandering"

        # wandering → walking (robot nearby)
        elif self.behavior == "wandering":
            if dist_to_robot < 2.0:
                self.behavior = "walking"

        # walking → standing (reach display)
        elif self.behavior == "walking":
            if dist_to_display < 0.8:
                self.behavior = "standing"
                self.stand_timer = np.random.uniform(3, 6)
    
    def update(self, model, data, timestep, repulsion_vec=None):
        """
        Update human motion toward current waypoint with optional repulsion.
        
        Args:
            model: MuJoCo model (for body ID lookup)
            data: MuJoCo data (for state and control)
            timestep: Simulation timestep
            repulsion_vec: Optional 2D velocity vector pushing away from other humans
        """
        self.step_count += 1
        
        # Get body ID on first call
        if self.body_id is None:
            self.body_id = model.body(self.body_name).id
        
        # Current pose using world coordinates (xpos) instead of qpos
        x = float(data.xpos[self.body_id, 0])
        y = float(data.xpos[self.body_id, 1])
        yaw = float(data.qpos[self.qpos_idx + 2])  # Only yaw from qpos
        
        # Vector to waypoint (follow component)
        dx = self.current_waypoint[0] - x
        dy = self.current_waypoint[1] - y
        dist = np.hypot(dx, dy)
        
        # Pick new waypoint if reached current one
        if dist < self.waypoint_threshold:
            if not self.external_waypoint:
                self.current_waypoint = self._random_waypoint()
                dx = self.current_waypoint[0] - x
                dy = self.current_waypoint[1] - y
                dist = np.hypot(dx, dy) + 1e-8
        
        if repulsion_vec is None:
            repulsion_vec = np.zeros(2, dtype=np.float32)

        # v_follow: move toward waypoint
        if dist > 1e-6:
            v_follow = self.max_speed * np.array([dx, dy], dtype=np.float32) / dist
        else:
            v_follow = np.zeros(2, dtype=np.float32)

        # v_repulsion: push away from nearby humans
        v_repulsion = np.array(repulsion_vec, dtype=np.float32)

        # Total desired velocity
        v_total = v_follow + v_repulsion
        speed = float(np.linalg.norm(v_total))
        if speed > self.max_speed and speed > 1e-6:
            v_total = v_total / speed * self.max_speed
            speed = self.max_speed

        # Desired yaw based on total velocity
        if speed > 1e-6:
            desired_yaw = float(np.arctan2(v_total[1], v_total[0]))
        else:
            desired_yaw = yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        vx = float(v_total[0])
        vy = float(v_total[1])
        yaw_rate = 50.0 * yaw_err  # Heading gain
        
        # Clip to actuator limits
        vx = np.clip(vx, -self.max_speed, self.max_speed)
        vy = np.clip(vy, -self.max_speed, self.max_speed)
        yaw_rate = np.clip(yaw_rate, -50.0, 50.0)
        
        # Return control commands [vx, vy, yaw_rate]
        return np.array([vx, vy, yaw_rate], dtype=np.float32)
