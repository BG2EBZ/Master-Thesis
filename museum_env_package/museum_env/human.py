import numpy as np


class HumanMode:
    WANDERING = "wandering"
    FOLLOWING = "following"
    LISTENING = "listening"
    DISTRACTED = "distracted"
    OVERWHELMED = "overwhelmed"


class Human:
    """
    Minimal human behavior: random walking in the museum.
    """
    
    def __init__(self, name, body_name, qpos_idx, max_speed, waypoint_threshold=0.2):
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

        self.can_be_distracted = False
        self.distracted_timer = 0
        self.distracted_duration = np.random.randint(1000, 1500)
        self.distracted_probability = 0.000  # small chance per step

        self.can_be_overwhelmed = False
        self.overwhelmed_stage = None  # "backoff" | "leave"
        self.overwhelmed_backoff_dist = 0.3
        self.overwhelmed_leave_speed = 1.5
        self.overwhelmed_leave_duration = 1000
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_leave_dir = np.zeros(2, dtype=np.float32)
        self.overwhelmed_robot_ref_xy = None
        self.overwhelmed_backoff_start_xy = None

    def set_mode(self, mode: str):
        if mode not in (
            HumanMode.WANDERING,
            HumanMode.FOLLOWING,
            HumanMode.LISTENING,
            HumanMode.DISTRACTED,
            HumanMode.OVERWHELMED,
        ):
            raise ValueError(f"Unknown human mode: {mode}")
        self.mode = mode

    def reset_overwhelmed_state(self):
        self.overwhelmed_stage = None
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_leave_dir = np.zeros(2, dtype=np.float32)
        self.overwhelmed_robot_ref_xy = None
        self.overwhelmed_backoff_start_xy = None

    def start_overwhelmed(self, robot_xy, current_xy=None):
        if not self.can_be_overwhelmed:
            return

        if current_xy is None:
            current_xy = np.array(self.current_waypoint, dtype=np.float32)
        else:
            current_xy = np.array(current_xy, dtype=np.float32)

        robot_xy = np.array(robot_xy, dtype=np.float32)
        diff = current_xy - robot_xy
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            leave_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            leave_dir = diff / dist

        self.mode = HumanMode.OVERWHELMED
        self.overwhelmed_stage = "backoff"
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_leave_dir = leave_dir.astype(np.float32)
        self.overwhelmed_robot_ref_xy = robot_xy
        self.overwhelmed_backoff_start_xy = current_xy
        print(f">>> {self.name} became OVERWHELMED!")


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
            # probabilistic switch to distracted
            if self.can_be_distracted and np.random.rand() < self.distracted_probability:
                self.mode = HumanMode.DISTRACTED
                self.distracted_timer = 0
                print(f">>> {self.name} became DISTRACTED!")
                return self._step_distracted(data, ctx)

            self._assign_target_from_context()
            return self._step_following(data, ctx)

        if self.mode == HumanMode.LISTENING:
            return self._step_listening(data, ctx)
        
        if self.mode == HumanMode.DISTRACTED:
            return self._step_distracted(data, ctx)

        if self.mode == HumanMode.OVERWHELMED:
            return self._step_overwhelmed(data, ctx)


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
    
    def _step_distracted(self, data, ctx):
        x, y, yaw = self._get_pose(data)

        """
        Distracted behavior:
        - Ignore robot attraction
        - Wander locally
        - Move slower than normal wandering
        - Recover automatically after duration
        """

        # Increment internal timer
        self.distracted_timer += 1

        # On first distracted step → choose a new random waypoint
        if self.distracted_timer == 1:
            self.current_waypoint = self._random_waypoint()

        # Create modified context that ignores robot
        distracted_ctx = ctx.copy()
        distracted_ctx["robot_xy"] = None  # disable human-robot attraction

        # Reuse wandering behavior
        action = self._step_wandering(data, distracted_ctx)

        # Slow down movement to make distraction visible
        action[0:2] *= 0.5  # reduce translation speed

        # Optional: slight random yaw noise for realism
        action[2] += np.random.uniform(-2.0, 2.0)

        # Recover after duration
        if self.distracted_timer > self.distracted_duration:
            self.mode = HumanMode.FOLLOWING
            self.distracted_timer = 0
            print(f">>> {self.name} recovered → FOLLOWING")

        return action

    def _step_overwhelmed(self, data, ctx):
        x, y, yaw = self._get_pose(data)
        pos_xy = np.array([x, y], dtype=np.float32)
        self.last_v_follow = np.zeros(2, dtype=np.float32)
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)

        if self.overwhelmed_stage is None:
            self.mode = HumanMode.FOLLOWING
            self.reset_overwhelmed_state()
            return np.zeros(3, dtype=np.float32)

        leave_dir = np.array(self.overwhelmed_leave_dir, dtype=np.float32)
        leave_norm = float(np.linalg.norm(leave_dir))
        if leave_norm < 1e-6:
            robot_xy = self.overwhelmed_robot_ref_xy
            if robot_xy is None:
                robot_xy = np.array(ctx.get("robot_xy", np.array([x - 1.0, y], dtype=np.float32)), dtype=np.float32)
            diff = pos_xy - robot_xy
            diff_norm = float(np.linalg.norm(diff))
            leave_dir = diff / diff_norm if diff_norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
            self.overwhelmed_leave_dir = leave_dir
        else:
            leave_dir = leave_dir / leave_norm

        desired_yaw = np.arctan2(leave_dir[1], leave_dir[0])

        if self.overwhelmed_stage == "backoff":
            if self.overwhelmed_backoff_start_xy is None:
                self.overwhelmed_backoff_start_xy = pos_xy.copy()

            backoff_target = (
                self.overwhelmed_backoff_start_xy
                + self.overwhelmed_backoff_dist * leave_dir
            )
            to_target = backoff_target - pos_xy
            dist_to_target = float(np.linalg.norm(to_target))

            if dist_to_target < 0.02:
                self.overwhelmed_stage = "leave"
                to_target = np.zeros(2, dtype=np.float32)
                dist_to_target = 0.0

            if dist_to_target > 1e-6:
                backoff_speed = min(self.overwhelmed_leave_speed, self.max_speed)
                v_xy = backoff_speed * (to_target / dist_to_target)
            else:
                v_xy = np.zeros(2, dtype=np.float32)

            yaw_err = self._wrap_to_pi(desired_yaw - yaw)
            return np.array([v_xy[0], v_xy[1], 20.0 * yaw_err], dtype=np.float32)

        # Leave stage: keep moving away for a fixed duration.
        leave_speed = min(self.overwhelmed_leave_speed, self.max_speed)
        v_xy = leave_speed * leave_dir
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)
        self.overwhelmed_leave_timer += 1

        if self.overwhelmed_leave_timer >= self.overwhelmed_leave_duration:
            self.mode = HumanMode.FOLLOWING
            self.reset_overwhelmed_state()
            print(f">>> {self.name} recovered from OVERWHELMED -> FOLLOWING")

        return np.array([v_xy[0], v_xy[1], 20.0 * yaw_err], dtype=np.float32)


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
            d_min = 1.0   # too close → repulsion
            d_max = 2.0   # too far → attraction

            if dist_hr < d_min:
                # too close → repulsion (slow down / move away)
                k_rep_hr = 2.0
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
    
