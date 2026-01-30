import numpy as np


class Human:
    """
    Minimal human behavior: random walking in the museum.
    """
    
    def __init__(self, name, body_name, qpos_idx, max_speed=1.2, waypoint_threshold=0.5):
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
        
        # room_choice = np.random.randint(0, 3)
        
        # if room_choice == 0:  # Room A
        #     wx = np.random.uniform(1, 9)
        #     wy = np.random.uniform(1, 9)
        # elif room_choice == 1:  # Corridor
        #     wx = np.random.uniform(7.5, 9.5)
        #     wy = np.random.uniform(-9, -1)
        # else:  # Room B
        #     wx = np.random.uniform(7.5, 11.5)
        #     wy = np.random.uniform(-14, -11)
        
        return np.array([wx, wy], dtype=np.float32)
    
    def _wrap_to_pi(self, ang):
        return (ang + np.pi) % (2 * np.pi) - np.pi
    
    def update(self, qpos, timestep):
        """
        Update human motion toward current waypoint.
        
        Args:
            qpos: MuJoCo qpos array (read/write)
            timestep: Simulation timestep
        """
        self.step_count += 1
        
        # Current pose
        x = qpos[self.qpos_idx]
        y = qpos[self.qpos_idx + 1]
        yaw = qpos[self.qpos_idx + 2]
        
        # Vector to waypoint
        dx = self.current_waypoint[0] - x
        dy = self.current_waypoint[1] - y
        dist = np.hypot(dx, dy)
        
        # Pick new waypoint if reached current one
        if dist < self.waypoint_threshold:
            self.current_waypoint = self._random_waypoint()
            dx = self.current_waypoint[0] - x
            dy = self.current_waypoint[1] - y
            dist = np.hypot(dx, dy) + 1e-8
        
        # Simple proportional controller toward waypoint
        desired_yaw = np.arctan2(dy, dx)
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)
        
        # Velocity commands
        vx = self.max_speed * (dx / dist) if dist > 1e-6 else 0.0
        vy = self.max_speed * (dy / dist) if dist > 1e-6 else 0.0
        yaw_rate = 2.0 * yaw_err  # Heading gain
        
        # Clip to actuator limits
        vx = np.clip(vx, -self.max_speed, self.max_speed)
        vy = np.clip(vy, -self.max_speed, self.max_speed)
        yaw_rate = np.clip(yaw_rate, -2.0, 2.0)
        
        # Return control commands [vx, vy, yaw_rate]
        return np.array([vx, vy, yaw_rate], dtype=np.float32)