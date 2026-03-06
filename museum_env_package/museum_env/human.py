import logging
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False

DEFAULT_WAYPOINT_THRESHOLD = 0.2
DEFAULT_FAN_HALF_ANGLE = np.pi / 6
DEFAULT_LISTEN_RADIUS = 1.2
DEFAULT_FOLLOW_RADIUS = 1.0
DEFAULT_IMPATIENT_FRONT_OFFSET = 1.2
HUMAN_YAW_RATE_GAIN = 20.0
HUMAN_ROTATION_STOP_DEG = 3.0
DISTRACTED_SPEED_SCALE = 0.5
DISTRACTED_YAW_NOISE_MIN = -2.0
DISTRACTED_YAW_NOISE_MAX = 2.0
OVERWHELMED_FALLBACK_X_OFFSET = 1.0
OVERWHELMED_STAGE_SWITCH_DIST = 0.02
HR_DISTANCE_MIN = 0.8
HR_DISTANCE_MAX = 2.0
HR_REPULSION_GAIN = 6.0
HR_ATTRACTION_GAIN = 0.8
NORM_EPS = 1e-6
ATTACK_DEFAULT_SPEED = 1.0
ATTACK_HIT_DISTANCE_DEFAULT = 0.33
WALL_CLEARANCE = 0.20
MIN_SPEED_EPS = 1e-6

@dataclass
class HumanContext:
    index: int = 0
    n_humans: int = 1
    robot_pose: Optional[Tuple[float, float, float]] = None
    fan_half_angle: float = DEFAULT_FAN_HALF_ANGLE
    follow_radius: float = DEFAULT_FOLLOW_RADIUS
    listen_radius: float = DEFAULT_LISTEN_RADIUS
    impatient_front_offset: Optional[float] = None
    robot_xy: Optional[np.ndarray] = None
    robot_yaw: Optional[float] = None

    @classmethod
    def from_kwargs(cls, **kwargs):
        allowed = {f.name for f in fields(cls)}
        unknown = set(kwargs.keys()) - allowed
        if unknown:
            unknown_str = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown context field(s): {unknown_str}")
        ctx = cls()
        for key, value in kwargs.items():
            setattr(ctx, key, value)
        return ctx


class HumanMode:
    WANDERING = "wandering"
    FOLLOWING = "following"
    LISTENING = "listening"
    DISTRACTED = "distracted"
    OVERWHELMED = "overwhelmed"
    IMPATIENT = "impatient"
    ATTACK = "attack"


class Human:
    """
    Minimal human behavior: random walking in the museum.
    """
    
    def __init__(
        self,
        name,
        body_name,
        qpos_idx,
        max_speed,
        waypoint_threshold=DEFAULT_WAYPOINT_THRESHOLD,
    ):
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
        self.max_speed = float(max_speed)
        self.base_max_speed = float(max_speed)
        self.waypoint_threshold = waypoint_threshold
        # If True, current_waypoint is managed externally (e.g., follow robot)
        self.external_waypoint = False
        setattr(self, "mode", None)

        self.context = HumanContext()
        self.enable_event_logs = True
        
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

        self.distracted_timer = 0
        self.distracted_duration = np.random.randint(1000, 1500)
        self.callback_response_mode = None  # None | "stay" | "ignore"
        self.callback_stay_steps_remaining = 0
        self.callback_ignore_last_dir = np.zeros(2, dtype=np.float32)

        self.can_be_impatient = False
        self.impatient_duration = 800
        self.impatient_timer = 0
        self.impatient_speed_multiplier = 1.3
        self.impatient_front_offset = DEFAULT_IMPATIENT_FRONT_OFFSET
        self.impatient_original_max_speed = None
        self.following_distracted_probability = 0.0
        self.following_impatient_probability = 0.0

        self.can_be_overwhelmed = False
        self.overwhelmed_stage = None  # "backoff" | "leave"
        self.overwhelmed_backoff_dist = 0.3
        self.overwhelmed_leave_speed = 1.5
        self.overwhelmed_leave_duration = 1000
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_leave_dir = np.zeros(2, dtype=np.float32)
        self.overwhelmed_robot_ref_xy = None
        self.overwhelmed_backoff_start_xy = None

        self.can_attack = False
        self.attack_speed = ATTACK_DEFAULT_SPEED
        self.attack_hit_distance = ATTACK_HIT_DISTANCE_DEFAULT
        self.attack_hit_this_step = False
        self.attack_origin_listen_waypoint = None
        self.transition_to(HumanMode.WANDERING, reason="init", force=True)

    @staticmethod
    def _validate_mode(mode: str):
        if mode not in (
            HumanMode.WANDERING,
            HumanMode.FOLLOWING,
            HumanMode.LISTENING,
            HumanMode.DISTRACTED,
            HumanMode.OVERWHELMED,
            HumanMode.IMPATIENT,
            HumanMode.ATTACK,
        ):
            raise ValueError(f"Unknown human mode: {mode}")

    def _on_exit_mode(self, prev_mode: str, next_mode: str, reason: Optional[str] = None):
        if prev_mode == HumanMode.IMPATIENT and next_mode != HumanMode.IMPATIENT:
            self._stop_impatient()
        if prev_mode == HumanMode.DISTRACTED and next_mode != HumanMode.DISTRACTED:
            self.distracted_timer = 0
            self._clear_callback_response_state()
        if prev_mode == HumanMode.OVERWHELMED and next_mode != HumanMode.OVERWHELMED:
            self.reset_overwhelmed_state()
        if prev_mode == HumanMode.ATTACK and next_mode != HumanMode.ATTACK:
            self.attack_hit_this_step = False

    def _on_enter_mode(self, prev_mode: Optional[str], next_mode: str, reason: Optional[str] = None):
        if next_mode == HumanMode.DISTRACTED:
            self.distracted_timer = 0
            self._clear_callback_response_state()
        if next_mode == HumanMode.ATTACK:
            self.attack_hit_this_step = False

    def transition_to(self, next_mode: str, reason: Optional[str] = None, force: bool = False) -> bool:
        self._validate_mode(next_mode)
        prev_mode = self.mode
        if (not force) and prev_mode == next_mode:
            return False
        if prev_mode is not None:
            self._on_exit_mode(prev_mode=prev_mode, next_mode=next_mode, reason=reason)
        self.mode = next_mode
        self._on_enter_mode(prev_mode=prev_mode, next_mode=next_mode, reason=reason)
        return True

    def set_mode(self, mode: str):
        self.transition_to(mode, reason="set_mode")

    def reset_overwhelmed_state(self):
        self.overwhelmed_stage = None
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_leave_dir = np.zeros(2, dtype=np.float32)
        self.overwhelmed_robot_ref_xy = None
        self.overwhelmed_backoff_start_xy = None

    def reset_episode_state(self):
        """Reset per-episode dynamic state while keeping static config."""
        self.step_count = 0
        self.external_waypoint = False
        self.transition_to(HumanMode.WANDERING, reason="episode_reset", force=True)
        self.context = HumanContext()
        self.current_waypoint = self._random_waypoint()

        self.distracted_timer = 0
        self.distracted_duration = np.random.randint(1000, 1500)
        self._clear_callback_response_state()

        self.impatient_timer = 0
        self.impatient_original_max_speed = None

        self.last_v_follow = np.zeros(2, dtype=np.float32)
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)

        self.max_speed = float(self.base_max_speed)
        self.reset_overwhelmed_state()
        self.attack_hit_this_step = False
        self.attack_origin_listen_waypoint = None

    def set_event_logging(self, enabled: bool):
        self.enable_event_logs = bool(enabled)
        logger.setLevel(logging.INFO if self.enable_event_logs else logging.CRITICAL + 1)

    def configure_following_variant(self, variant_mode, probability):
        allowed = (None, HumanMode.DISTRACTED, HumanMode.IMPATIENT)
        if variant_mode not in allowed:
            raise ValueError(f"Invalid following variant mode: {variant_mode}")
        p = float(probability)
        if p < 0.0 or p > 1.0:
            raise ValueError(f"following variant probability must be in [0, 1], got {probability}")
        if variant_mode is None:
            self.following_distracted_probability = 0.0
            self.following_impatient_probability = 0.0
            return
        if variant_mode == HumanMode.DISTRACTED:
            self.following_distracted_probability = p
            self.following_impatient_probability = 0.0
            return
        self.following_distracted_probability = 0.0
        self.following_impatient_probability = p

    def configure_following_variant_probs(self, distracted_prob, impatient_prob):
        # In the simplified setup, use impatient_prob=0.0 to model "distracted-only".
        p_d = float(distracted_prob)
        p_i = float(impatient_prob)
        if p_d < 0.0 or p_d > 1.0:
            raise ValueError(f"distracted probability must be in [0, 1], got {distracted_prob}")
        if p_i < 0.0 or p_i > 1.0:
            raise ValueError(f"impatient probability must be in [0, 1], got {impatient_prob}")
        self.following_distracted_probability = p_d
        self.following_impatient_probability = p_i

    def _log_event(self, msg: str):
        if self.enable_event_logs:
            logger.info(msg)

    @staticmethod
    def _compute_fan_relative_angle(index, n_humans, fan_half_angle):
        if n_humans > 1:
            return (index / (n_humans - 1)) * (2 * fan_half_angle) - fan_half_angle
        return 0.0

    @staticmethod
    def _compute_fan_target(robot_pose, radius, relative_angle, base_angle_offset):
        rx, ry, ryaw = robot_pose
        angle = ryaw + base_angle_offset + relative_angle
        return np.array(
            [rx + radius * np.cos(angle), ry + radius * np.sin(angle)],
            dtype=np.float32,
        )

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
        if dist < NORM_EPS:
            leave_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            leave_dir = diff / dist

        self.transition_to(HumanMode.OVERWHELMED, reason="trigger_overwhelmed")
        self.overwhelmed_stage = "backoff"
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_leave_dir = leave_dir.astype(np.float32)
        self.overwhelmed_robot_ref_xy = robot_xy
        self.overwhelmed_backoff_start_xy = current_xy
        self._log_event(f">>> {self.name} became OVERWHELMED!")

    def start_impatient(self, robot_pose=None, index=None, n_humans=None):
        if not self.can_be_impatient:
            return False
        if self.mode == HumanMode.IMPATIENT:
            return True

        if robot_pose is not None:
            self.context.robot_pose = robot_pose
        if index is not None:
            self.context.index = index
        if n_humans is not None:
            self.context.n_humans = n_humans
        if self.context.impatient_front_offset is None:
            self.context.impatient_front_offset = self.impatient_front_offset

        self.impatient_original_max_speed = float(self.max_speed)
        self.max_speed = float(self.impatient_original_max_speed * self.impatient_speed_multiplier)
        self.impatient_timer = 0
        self.transition_to(HumanMode.IMPATIENT, reason="trigger_impatient")
        self._log_event(f">>> {self.name} became IMPATIENT!")
        return True

    def _stop_impatient(self):
        if self.impatient_original_max_speed is not None:
            self.max_speed = float(self.impatient_original_max_speed)
        else:
            self.max_speed = float(self.base_max_speed)
        self.impatient_original_max_speed = None
        self.impatient_timer = 0

    def _clear_callback_response_state(self):
        self.callback_response_mode = None
        self.callback_stay_steps_remaining = 0
        self.callback_ignore_last_dir = np.zeros(2, dtype=np.float32)

    def start_attack(self):
        if not self.can_attack:
            return False
        self.attack_origin_listen_waypoint = np.array(self.current_waypoint, dtype=np.float32)
        self.transition_to(HumanMode.ATTACK, reason="trigger_attack")
        self.attack_hit_this_step = False
        self._log_event(f">>> {self.name} became ATTACK!")
        return True

    def force_recover_from_callback(self) -> bool:
        return self.apply_callback_response(response="rejoin", stay_steps=0)

    def apply_callback_response(self, response: str, stay_steps: int = 0) -> bool:
        if self.mode != HumanMode.DISTRACTED:
            return False

        if response == "rejoin":
            self.transition_to(HumanMode.FOLLOWING, reason="callback_rejoin")
            return True

        if response == "stay":
            self.callback_response_mode = "stay"
            self.callback_stay_steps_remaining = max(1, int(stay_steps))
            return True

        if response == "ignore":
            self.callback_response_mode = "ignore"
            return True

        raise ValueError(f"Unknown callback response: {response}")

    def _maybe_trigger_following_variant(self):
        trigger_distracted = (
            self.following_distracted_probability > 0.0
            and np.random.rand() < self.following_distracted_probability
        )
        trigger_impatient = (
            self.following_impatient_probability > 0.0
            and self.can_be_impatient
            and np.random.rand() < self.following_impatient_probability
        )
        if trigger_distracted:
            return HumanMode.DISTRACTED
        if trigger_impatient:
            return HumanMode.IMPATIENT
        return None


    def set_context(self, **kwargs):
        """
        Store high-level social context provided by env.
        Example:
            mode="listening"
            index=0
            n_humans=3
            robot_pose=(rx, ry, ryaw)
        """
        self.context = HumanContext.from_kwargs(**kwargs)

    def _assign_target_from_context(self):
        """
        Decide where to stand based on social context.
        """
        index = self.context.index
        n_humans = self.context.n_humans
        robot_pose = self.context.robot_pose

        if robot_pose is None:
            return

        fan_half_angle = self.context.fan_half_angle
        relative_angle = self._compute_fan_relative_angle(index, n_humans, fan_half_angle)
        if self.mode == HumanMode.LISTENING:
            radius = self.context.listen_radius
            self.current_waypoint = self._compute_fan_target(
                robot_pose=robot_pose,
                radius=radius,
                relative_angle=relative_angle,
                base_angle_offset=0.0,
            )

        elif self.mode == HumanMode.FOLLOWING:
            radius = self.context.follow_radius
            self.current_waypoint = self._compute_fan_target(
                robot_pose=robot_pose,
                radius=radius,
                relative_angle=relative_angle,
                base_angle_offset=np.pi,
            )

        elif self.mode == HumanMode.IMPATIENT:
            radius = self.context.impatient_front_offset
            if radius is None:
                radius = self.impatient_front_offset
            self.current_waypoint = self._compute_fan_target(
                robot_pose=robot_pose,
                radius=radius,
                relative_angle=relative_angle,
                base_angle_offset=0.0,
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
            # may trigger a variant behavior (e.g., distracted or impatient)
            variant = self._maybe_trigger_following_variant()
            if variant == HumanMode.IMPATIENT:
                robot_xy = self.context.robot_xy if self.context.robot_xy is not None else ctx.get("robot_xy")
                robot_yaw = self.context.robot_yaw if self.context.robot_yaw is not None else ctx.get("robot_yaw", 0.0)
                if robot_xy is not None:
                    robot_pose = (float(robot_xy[0]), float(robot_xy[1]), float(robot_yaw))
                else:
                    robot_pose = None
                started = self.start_impatient(
                    robot_pose=robot_pose,
                    index=self.context.index,
                    n_humans=self.context.n_humans,
                )
                if started:
                    return self._step_impatient(data, ctx)

            if variant == HumanMode.DISTRACTED:
                self.transition_to(HumanMode.DISTRACTED, reason="following_variant_distracted")
                self._log_event(f">>> {self.name} became DISTRACTED!")
                return self._step_distracted(data, ctx)

            self._assign_target_from_context()
            return self._step_following(data, ctx)

        if self.mode == HumanMode.LISTENING:
            return self._step_listening(data, ctx)
        
        if self.mode == HumanMode.DISTRACTED:
            return self._step_distracted(data, ctx)

        if self.mode == HumanMode.OVERWHELMED:
            return self._step_overwhelmed(data, ctx)

        if self.mode == HumanMode.IMPATIENT:
            return self._step_impatient(data, ctx)

        if self.mode == HumanMode.ATTACK:
            return self._step_attack(data, ctx)


        raise ValueError(f"Unknown human mode {self.mode}")
        
    def assign_follow_target(self, index, n_humans, robot_pose, follow_radius, fan_half_angle):
        """
        Compute and assign target for FOLLOWING behavior.
        """
        relative_angle = self._compute_fan_relative_angle(index, n_humans, fan_half_angle)
        self.current_waypoint = self._compute_fan_target(
            robot_pose=robot_pose,
            radius=follow_radius,
            relative_angle=relative_angle,
            base_angle_offset=np.pi,
        )


    def assign_listen_target(self, index, n_humans, robot_pose, listen_radius, fan_half_angle):
        """
        Compute and assign target for LISTEN behavior.
        """
        relative_angle = self._compute_fan_relative_angle(index, n_humans, fan_half_angle)
        self.current_waypoint = self._compute_fan_target(
            robot_pose=robot_pose,
            radius=listen_radius,
            relative_angle=relative_angle,
            base_angle_offset=0.0,
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

        stand_threshold = ctx.get("stand_threshold", self.waypoint_threshold)

        if dist >= stand_threshold:
            return self._move(dx, dy, yaw, data, ctx)
        
        robot_xy = ctx.get("robot_xy")
        if robot_xy is None:
            return np.zeros(3, dtype=np.float32)
        
        rx, ry = robot_xy[0], robot_xy[1]
        desired_yaw = np.arctan2(ry - y, rx - x)
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        if abs(yaw_err) < np.deg2rad(HUMAN_ROTATION_STOP_DEG):
            return np.zeros(3, dtype=np.float32)
        
        return np.array([0.0, 0.0, HUMAN_YAW_RATE_GAIN * yaw_err])
    
    def _step_distracted(self, data, ctx):
        x, y, yaw = self._get_pose(data)

        """
        Distracted behavior:
        - Ignore robot attraction
        - Wander locally
        - Move slower than normal wandering
        - Recover automatically after duration
        """

        if self.callback_response_mode == "stay":
            self.last_v_follow = np.zeros(2, dtype=np.float32)
            self.last_v_repulsion = np.zeros(2, dtype=np.float32)
            self.last_v_hr = np.zeros(2, dtype=np.float32)
            self.callback_stay_steps_remaining = max(0, int(self.callback_stay_steps_remaining) - 1)
            if self.callback_stay_steps_remaining <= 0:
                self.callback_response_mode = None
            return np.zeros(3, dtype=np.float32)

        if self.callback_response_mode == "ignore":
            self.distracted_timer += 1

            robot_xy = ctx.get("robot_xy", None)
            if robot_xy is not None:
                away = np.array([x - float(robot_xy[0]), y - float(robot_xy[1])], dtype=np.float32)
            else:
                away = np.array(self.callback_ignore_last_dir, dtype=np.float32)

            away_norm = float(np.linalg.norm(away))
            if away_norm > NORM_EPS:
                away_dir = away / away_norm
            else:
                last_norm = float(np.linalg.norm(self.callback_ignore_last_dir))
                if last_norm > NORM_EPS:
                    away_dir = self.callback_ignore_last_dir / last_norm
                else:
                    away_dir = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

            self.callback_ignore_last_dir = np.array(away_dir, dtype=np.float32)
            v_away = float(DISTRACTED_SPEED_SCALE * self.max_speed) * away_dir
            v_repulsion = np.array(ctx.get("repulsion", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            v_total = v_away + v_repulsion
            speed = float(np.linalg.norm(v_total))
            if speed > self.max_speed and speed > NORM_EPS:
                v_total = v_total / speed * self.max_speed
                speed = float(np.linalg.norm(v_total))

            desired_yaw = np.arctan2(v_total[1], v_total[0]) if speed > NORM_EPS else yaw
            yaw_err = self._wrap_to_pi(desired_yaw - yaw)
            self.last_v_follow = np.array(v_away, dtype=np.float32)
            self.last_v_repulsion = np.array(v_repulsion, dtype=np.float32)
            self.last_v_hr = np.zeros(2, dtype=np.float32)
            action = np.array([v_total[0], v_total[1], HUMAN_YAW_RATE_GAIN * yaw_err], dtype=np.float32)
        else:
            # Increment internal timer (also used by env callback trigger logic).
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
            action[0:2] *= DISTRACTED_SPEED_SCALE  # reduce translation speed

            # Optional: slight random yaw noise for realism
            action[2] += np.random.uniform(DISTRACTED_YAW_NOISE_MIN, DISTRACTED_YAW_NOISE_MAX)

        # Recover after duration
        if self.distracted_timer > self.distracted_duration:
            self.transition_to(HumanMode.FOLLOWING, reason="distracted_timeout_recover")
            self._log_event(f">>> {self.name} recovered -> FOLLOWING")

        return self._apply_wall_constraint_to_action(action, data, ctx)

    def _step_overwhelmed(self, data, ctx):
        x, y, yaw = self._get_pose(data)
        pos_xy = np.array([x, y], dtype=np.float32)
        self.last_v_follow = np.zeros(2, dtype=np.float32)
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)

        if self.overwhelmed_stage is None:
            self.transition_to(HumanMode.FOLLOWING, reason="overwhelmed_invalid_stage_recover")
            return np.zeros(3, dtype=np.float32)

        leave_dir = np.array(self.overwhelmed_leave_dir, dtype=np.float32)
        leave_norm = float(np.linalg.norm(leave_dir))
        if leave_norm < NORM_EPS:
            robot_xy = self.overwhelmed_robot_ref_xy
            if robot_xy is None:
                robot_xy = np.array(
                    ctx.get("robot_xy", np.array([x - OVERWHELMED_FALLBACK_X_OFFSET, y], dtype=np.float32)),
                    dtype=np.float32,
                )
            diff = pos_xy - robot_xy
            diff_norm = float(np.linalg.norm(diff))
            leave_dir = diff / diff_norm if diff_norm > NORM_EPS else np.array([1.0, 0.0], dtype=np.float32)
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

            if dist_to_target < OVERWHELMED_STAGE_SWITCH_DIST:
                self.overwhelmed_stage = "leave"
                to_target = np.zeros(2, dtype=np.float32)
                dist_to_target = 0.0

            if dist_to_target > NORM_EPS:
                backoff_speed = min(self.overwhelmed_leave_speed, self.max_speed)
                v_xy = backoff_speed * (to_target / dist_to_target)
            else:
                v_xy = np.zeros(2, dtype=np.float32)

            yaw_err = self._wrap_to_pi(desired_yaw - yaw)
            action = np.array([v_xy[0], v_xy[1], HUMAN_YAW_RATE_GAIN * yaw_err], dtype=np.float32)
            return self._apply_wall_constraint_to_action(action, data, ctx)

        # Leave stage: keep moving away for a fixed duration.
        leave_speed = min(self.overwhelmed_leave_speed, self.max_speed)
        v_xy = leave_speed * leave_dir
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)
        self.overwhelmed_leave_timer += 1

        if self.overwhelmed_leave_timer >= self.overwhelmed_leave_duration:
            self.transition_to(HumanMode.FOLLOWING, reason="overwhelmed_timeout_recover")
            self._log_event(f">>> {self.name} recovered from OVERWHELMED -> FOLLOWING")

        action = np.array([v_xy[0], v_xy[1], HUMAN_YAW_RATE_GAIN * yaw_err], dtype=np.float32)
        return self._apply_wall_constraint_to_action(action, data, ctx)

    def _step_impatient(self, data, ctx):
        self.impatient_timer += 1
        self._assign_target_from_context()
        action = self._step_following(data, ctx)

        if self.impatient_timer >= self.impatient_duration:
            self.set_mode(HumanMode.FOLLOWING)
            self._log_event(f">>> {self.name} recovered from IMPATIENT -> FOLLOWING")

        return action

    def _step_attack(self, data, ctx):
        self.attack_hit_this_step = False
        robot_xy = ctx.get("robot_xy", None)
        if robot_xy is None:
            self.last_v_follow = np.zeros(2, dtype=np.float32)
            self.last_v_repulsion = np.zeros(2, dtype=np.float32)
            self.last_v_hr = np.zeros(2, dtype=np.float32)
            return np.zeros(3, dtype=np.float32)

        x, y, yaw = self._get_pose(data)
        to_robot = np.array([robot_xy[0] - x, robot_xy[1] - y], dtype=np.float32)
        dist = float(np.linalg.norm(to_robot))

        if dist <= float(self.attack_hit_distance):
            self.attack_hit_this_step = True
            self.transition_to(HumanMode.LISTENING, reason="attack_hit_recover")
            self.last_v_follow = np.zeros(2, dtype=np.float32)
            self.last_v_repulsion = np.zeros(2, dtype=np.float32)
            self.last_v_hr = np.zeros(2, dtype=np.float32)
            self._log_event(f">>> {self.name} hit robot -> LISTENING")
            return np.zeros(3, dtype=np.float32)

        if dist > NORM_EPS:
            direction = to_robot / dist
            v_follow = float(self.attack_speed) * direction
        else:
            v_follow = np.zeros(2, dtype=np.float32)

        self.last_v_follow = v_follow.copy()
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)

        desired_yaw = np.arctan2(v_follow[1], v_follow[0]) if float(np.linalg.norm(v_follow)) > NORM_EPS else yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)
        action = np.array([v_follow[0], v_follow[1], HUMAN_YAW_RATE_GAIN * yaw_err], dtype=np.float32)
        return self._apply_wall_constraint_to_action(action, data, ctx)


    def _move(self, dx, dy, yaw, data, ctx):
        robot_xy = ctx.get("robot_xy", None)

        repulsion = ctx.get("repulsion", np.zeros(2))
        v_hr = np.zeros(2, dtype=np.float32)

        if robot_xy is not None:
            # current human position
            hx, hy, _ = self._get_pose(data)

            diff_hr = np.array([hx, hy], dtype=np.float32) - robot_xy
            dist_hr = np.linalg.norm(diff_hr) + NORM_EPS
            dir_hr = diff_hr / dist_hr

            # preferred human–robot distance (meters)
            if dist_hr < HR_DISTANCE_MIN:
                # too close → repulsion (slow down / move away)
                v_hr = HR_REPULSION_GAIN * (HR_DISTANCE_MIN - dist_hr) * dir_hr

            elif dist_hr > HR_DISTANCE_MAX:
                # too far → attraction (move towards robot)
                v_hr = -HR_ATTRACTION_GAIN * (dist_hr - HR_DISTANCE_MAX) * dir_hr

        dist = np.hypot(dx, dy)
        if dist > NORM_EPS:
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

        desired_yaw = np.arctan2(v_total[1], v_total[0]) if speed > NORM_EPS else yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        # if abs(yaw_err) > np.deg2rad(5):
        #     # hard stop translation while turning
        #     # data.qvel[self.x_dof_idx] = 0.0
        #     # data.qvel[self.y_dof_idx] = 0.0
        #     return np.array([0.0, 0.0, 20.0 * yaw_err])

        action = np.array([v_total[0], v_total[1], HUMAN_YAW_RATE_GAIN * yaw_err], dtype=np.float32)
        return self._apply_wall_constraint_to_action(action, data, ctx)
    
    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _walkable_rects(margin: float):
        m = max(0.0, float(margin))
        rects = [
            # Room A: x[0,10], y[0,10]
            (0.0 + m, 10.0 - m, 0.0 + m, 10.0 - m),
            # Corridor: x[7,10], y[-10,0]
            (7.0 + m, 10.0 - m, -10.0 + m, 0.0 - m),
            # Room B: x[7,12], y[-15,-10]
            (7.0 + m, 12.0 - m, -15.0 + m, -10.0 - m),
            # Opening near y=0 between room A and corridor.
            (7.0 + m, 10.0 - m, -m, m),
            # Opening near y=-10 between corridor and room B.
            (7.0 + m, 10.0 - m, -10.0 - m, -10.0 + m),
        ]
        valid = []
        for xmin, xmax, ymin, ymax in rects:
            if xmin <= xmax and ymin <= ymax:
                valid.append((xmin, xmax, ymin, ymax))
        return valid

    @staticmethod
    def _is_point_in_rect(x: float, y: float, rect) -> bool:
        xmin, xmax, ymin, ymax = rect
        return bool((xmin <= x <= xmax) and (ymin <= y <= ymax))

    def _is_point_in_walkable(self, xy, margin: float) -> bool:
        x = float(xy[0])
        y = float(xy[1])
        return any(self._is_point_in_rect(x, y, rect) for rect in self._walkable_rects(margin))

    def _project_point_to_walkable(self, xy, margin: float):
        point = np.array(xy, dtype=np.float32)
        rects = self._walkable_rects(margin)
        if not rects:
            return point
        if self._is_point_in_walkable(point, margin):
            return point

        best_proj = None
        best_dist_sq = None
        for xmin, xmax, ymin, ymax in rects:
            proj_x = float(np.clip(point[0], xmin, xmax))
            proj_y = float(np.clip(point[1], ymin, ymax))
            proj = np.array([proj_x, proj_y], dtype=np.float32)
            dist_sq = float(np.sum((proj - point) ** 2))
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_proj = proj
        if best_proj is None:
            return point
        return best_proj

    def _constrain_velocity_with_walkable(self, x: float, y: float, v_xy, dt: float, margin: float):
        v_xy = np.array(v_xy, dtype=np.float32)
        speed = float(np.linalg.norm(v_xy))
        if speed > self.max_speed and speed > MIN_SPEED_EPS:
            v_xy = v_xy / speed * float(self.max_speed)

        dt = float(dt)
        if dt <= MIN_SPEED_EPS:
            return v_xy

        current_xy = np.array([float(x), float(y)], dtype=np.float32)
        next_xy = current_xy + dt * v_xy
        if self._is_point_in_walkable(next_xy, margin):
            return v_xy

        projected_xy = self._project_point_to_walkable(next_xy, margin)
        safe_v = (projected_xy - current_xy) / dt
        safe_speed = float(np.linalg.norm(safe_v))
        if safe_speed > self.max_speed and safe_speed > MIN_SPEED_EPS:
            safe_v = safe_v / safe_speed * float(self.max_speed)
        return np.array(safe_v, dtype=np.float32)

    def _apply_wall_constraint_to_action(self, action, data, ctx):
        constrained_action = np.array(action, dtype=np.float32)
        if constrained_action.shape[0] < 2:
            return constrained_action

        x, y, _ = self._get_pose(data)
        dt = float(ctx.get("dt", 0.002))
        constrained_action[0:2] = self._constrain_velocity_with_walkable(
            x=x,
            y=y,
            v_xy=constrained_action[0:2],
            dt=dt,
            margin=WALL_CLEARANCE,
        )
        return constrained_action

    def _get_pose(self, data):
        x = float(data.xpos[self.body_id, 0])
        y = float(data.xpos[self.body_id, 1])
        yaw = float(data.qpos[self.qpos_idx + 2])
        return x, y, yaw

    def _random_waypoint(self):
        """Generate random waypoint within walkable museum bounds."""
        rects = self._walkable_rects(WALL_CLEARANCE)
        if not rects:
            return np.array([0.0, 0.0], dtype=np.float32)

        areas = np.array(
            [
                max(0.0, float(xmax - xmin)) * max(0.0, float(ymax - ymin))
                for xmin, xmax, ymin, ymax in rects
            ],
            dtype=np.float64,
        )
        if float(np.sum(areas)) <= 0.0:
            xmin, xmax, ymin, ymax = rects[0]
            return np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)], dtype=np.float32)

        probs = areas / float(np.sum(areas))
        rect_idx = int(np.random.choice(len(rects), p=probs))
        xmin, xmax, ymin, ymax = rects[rect_idx]
        wx = float(np.random.uniform(xmin, xmax))
        wy = float(np.random.uniform(ymin, ymax))
        return np.array([wx, wy], dtype=np.float32)
    
    def _wrap_to_pi(self, ang):
        return (ang + np.pi) % (2 * np.pi) - np.pi
    
