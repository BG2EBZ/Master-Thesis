import logging
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np

# Human agent state machine and local motion logic.
# Env owns high-level scheduling; this class turns mode/context into low-level actions.
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
DISTRACTED_YAW_DEVIATION_MIN_DEG = 45.0
DISTRACTED_YAW_DEVIATION_MAX_DEG = 90.0
DISTRACTED_TARGET_DISTANCE_MIN = 0.5
DISTRACTED_TARGET_DISTANCE_MAX = 1.5
DEFAULT_SIM_TIMESTEP_SECONDS = 0.002
DISTRACTED_HAZARD_SIGMOID_K = float(2.0 * np.log(9.0))
DISTRACTED_LAMBDA_MAX_PER_SEC_DEFAULT = 0.08
DISTRACTED_RAMP_START_SECONDS_DEFAULT = 40.0
DISTRACTED_RISE_SECONDS_DEFAULT = 20.0
IMPATIENT_LAMBDA_MAX_PER_SEC_DEFAULT = 0.08
IMPATIENT_RAMP_START_SECONDS_DEFAULT = 10.0
IMPATIENT_RISE_SECONDS_DEFAULT = 10.0
IMPATIENT_ROBOT_SPEED_THRESHOLD_DEFAULT = 0.2
LISTENING_DISTRACTED_LAMBDA_MAX_PER_SEC_DEFAULT = 0.05
LISTENING_DISTRACTED_RAMP_START_SECONDS_DEFAULT = 40.0
LISTENING_DISTRACTED_RISE_SECONDS_DEFAULT = 30.0
DISTRACTED_DURATION_SECONDS_DEFAULT = 10.0
LISTENING_DISTRACTED_YAW_DEVIATION_MIN_DEG = 10.0
LISTENING_DISTRACTED_YAW_DEVIATION_MAX_DEG = 40.0
LISTENING_DISTRACTED_SPEED_METERS_PER_SEC = 0.3
LISTENING_DISTRACTED_MOVE_SECONDS = 2.0
OVERWHELMED_FALLBACK_X_OFFSET = 1.0
OVERWHELMED_STAGE_SWITCH_DIST = 0.02
HR_DISTANCE_MIN = 0.8
HR_DISTANCE_MAX = 2.0
HR_DISTANCE_MAX_NORMAL_DEFAULT = 1.5
HR_DISTANCE_MIN_ND_DEFAULT = 1.0
HR_REPULSION_GAIN = 4.0
HR_ATTRACTION_GAIN = 2.0
NORM_EPS = 1e-6
ATTACK_DEFAULT_SPEED = 1.0
ATTACK_HIT_DISTANCE_DEFAULT = 0.33
HUMAN_WALL_FOOTPRINT_RADIUS = 0.25
SEGMENT_CHECK_SPACING = 0.05
MIN_SPEED_EPS = 1e-6
ROOM_A_X_MIN = 0.0
ROOM_A_X_MAX = 10.0
ROOM_A_Y_MIN = 0.0
ROOM_A_Y_MAX = 10.0
DISTRACTED_SOURCE_FOLLOWING = "following"
DISTRACTED_SOURCE_LISTENING = "listening"

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
        """Create context from kwargs and reject unknown fields early."""
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


class HumanProfile:
    NORMAL = "normal"
    NEURODIVERGENT = "neurodivergent"


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
        self.hr_distance_min = float(HR_DISTANCE_MIN)
        self.hr_distance_max = float(HR_DISTANCE_MAX)

        self.distracted_timer = 0
        self.max_distracted_duration_seconds = float(DISTRACTED_DURATION_SECONDS_DEFAULT)
        self.distracted_duration = 0
        self.configure_distracted_duration(
            max_duration_seconds=self.max_distracted_duration_seconds,
            dt=DEFAULT_SIM_TIMESTEP_SECONDS,
        )
        self.callback_response_mode = None
        self.distracted_target_xy = None
        self.distracted_stop_reached = False
        self.distracted_target_yaw = None

        self.can_be_impatient = False
        self.impatient_duration = 800
        self.impatient_timer = 0
        self.impatient_speed_multiplier = 1.3
        self.impatient_front_offset = DEFAULT_IMPATIENT_FRONT_OFFSET
        self.impatient_original_max_speed = None
        self.following_impatient_probability = 0.0
        self.following_low_robot_speed_steps = 0
        self.following_impatient_lambda_max_per_sec = IMPATIENT_LAMBDA_MAX_PER_SEC_DEFAULT
        self.following_impatient_ramp_start_seconds = IMPATIENT_RAMP_START_SECONDS_DEFAULT
        self.following_impatient_rise_seconds = IMPATIENT_RISE_SECONDS_DEFAULT
        self.following_impatient_robot_speed_threshold = IMPATIENT_ROBOT_SPEED_THRESHOLD_DEFAULT
        self.following_distracted_lambda_max_per_sec = DISTRACTED_LAMBDA_MAX_PER_SEC_DEFAULT
        self.following_distracted_ramp_start_seconds = DISTRACTED_RAMP_START_SECONDS_DEFAULT
        self.following_distracted_rise_seconds = DISTRACTED_RISE_SECONDS_DEFAULT
        self.profile = HumanProfile.NORMAL
        self.following_steps = 0
        self.following_distracted_window_active = False
        self.listening_steps = 0
        self.listening_started_this_session = False
        self.listening_target_reached_once = False
        self.listening_distracted_window_active = False
        self.listening_distracted_lambda_max_per_sec = LISTENING_DISTRACTED_LAMBDA_MAX_PER_SEC_DEFAULT
        self.listening_distracted_ramp_start_seconds = LISTENING_DISTRACTED_RAMP_START_SECONDS_DEFAULT
        self.listening_distracted_rise_seconds = LISTENING_DISTRACTED_RISE_SECONDS_DEFAULT
        self.distracted_source = None
        self.listening_distracted_move_steps = max(
            1,
            int(round(LISTENING_DISTRACTED_MOVE_SECONDS / DEFAULT_SIM_TIMESTEP_SECONDS)),
        )
        self.listening_distracted_move_elapsed_steps = 0
        self.listening_distracted_hold_until_session_end = False

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
        """Validate that mode belongs to HumanMode."""
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

    @staticmethod
    def _validate_profile(profile: str):
        """Validate that profile belongs to HumanProfile."""
        if profile not in (HumanProfile.NORMAL, HumanProfile.NEURODIVERGENT):
            raise ValueError(f"Unknown human profile: {profile}")

    def _on_exit_mode(self, prev_mode: str, next_mode: str, reason: Optional[str] = None):
        """Run cleanup hooks when leaving a mode."""
        if prev_mode == HumanMode.IMPATIENT and next_mode != HumanMode.IMPATIENT:
            self._stop_impatient()
        if prev_mode == HumanMode.DISTRACTED and next_mode != HumanMode.DISTRACTED:
            self.distracted_timer = 0
            self._clear_callback_response_state()
            self._clear_distracted_navigation_state()
            self.distracted_source = None
        if prev_mode == HumanMode.FOLLOWING and next_mode != HumanMode.FOLLOWING:
            self.reset_following_duration()
            self.reset_impatient_trigger_progress()
        if prev_mode == HumanMode.OVERWHELMED and next_mode != HumanMode.OVERWHELMED:
            self.reset_overwhelmed_state()
        if prev_mode == HumanMode.ATTACK and next_mode != HumanMode.ATTACK:
            self.attack_hit_this_step = False

    def _on_enter_mode(self, prev_mode: Optional[str], next_mode: str, reason: Optional[str] = None):
        """Run initialization hooks when entering a mode."""
        if next_mode == HumanMode.DISTRACTED:
            self.distracted_timer = 0
            self._clear_callback_response_state()
            self._clear_distracted_navigation_state()
        if next_mode == HumanMode.ATTACK:
            self.attack_hit_this_step = False

    def transition_to(self, next_mode: str, reason: Optional[str] = None, force: bool = False) -> bool:
        """Switch mode and execute enter/exit side-effects."""
        # Centralized mode transition so enter/exit side-effects stay in one place.
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
        """Public wrapper for mode switch."""
        self.transition_to(mode, reason="set_mode")

    def reset_overwhelmed_state(self):
        """Clear internal state used by overwhelmed behavior."""
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
        self._clear_callback_response_state()
        self._clear_distracted_navigation_state()

        self.impatient_timer = 0
        self.impatient_original_max_speed = None

        self.last_v_follow = np.zeros(2, dtype=np.float32)
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)

        self.max_speed = float(self.base_max_speed)
        self.reset_following_duration()
        self.reset_impatient_trigger_progress()
        self.following_distracted_window_active = False
        self.reset_listening_session_state()
        self.reset_overwhelmed_state()
        self.attack_hit_this_step = False
        self.attack_origin_listen_waypoint = None

    def set_event_logging(self, enabled: bool):
        """Enable/disable per-human event logs."""
        self.enable_event_logs = bool(enabled)
        logger.setLevel(logging.INFO if self.enable_event_logs else logging.CRITICAL + 1)

    def set_profile(self, profile: str):
        """Set behavior profile (normal or neurodivergent)."""
        self._validate_profile(profile)
        self.profile = profile

    def configure_hr_distance_band(self, hr_distance_min: float, hr_distance_max: float):
        """Configure preferred human-robot spacing band for this human."""
        distance_min = float(hr_distance_min)
        distance_max = float(hr_distance_max)
        if distance_min <= 0.0:
            raise ValueError(f"hr_distance_min must be > 0, got {hr_distance_min}")
        if distance_max < distance_min:
            raise ValueError(
                "hr_distance_max must be >= hr_distance_min, "
                f"got min={hr_distance_min}, max={hr_distance_max}"
            )
        self.hr_distance_min = distance_min
        self.hr_distance_max = distance_max

    def reset_following_duration(self):
        """Reset timer counting consecutive eligible following steps."""
        self.following_steps = 0

    def reset_impatient_trigger_progress(self):
        """Reset timer counting consecutive low-speed robot following steps."""
        self.following_low_robot_speed_steps = 0

    def set_following_distracted_window_active(self, active: bool):
        """Tell the human whether distracted-follow hazard should be active now."""
        self.following_distracted_window_active = bool(active)

    def reset_listening_session_state(self):
        """Reset per-session listening hazard state."""
        self.listening_steps = 0
        self.listening_started_this_session = False
        self.listening_target_reached_once = False
        self.listening_distracted_window_active = False
        self.listening_distracted_move_elapsed_steps = 0
        self.listening_distracted_hold_until_session_end = False

    def set_listening_distracted_window_active(self, active: bool):
        """Tell the human whether listening->distracted hazard should be active now."""
        self.listening_distracted_window_active = bool(active)

    def configure_distracted_duration(
        self,
        max_duration_seconds: float,
        dt: float = DEFAULT_SIM_TIMESTEP_SECONDS,
    ):
        """Configure maximum distracted duration and cache the converted step count."""
        duration_seconds = float(max_duration_seconds)
        dt_safe = float(dt)
        if duration_seconds <= 0.0:
            raise ValueError(
                f"distracted max_duration_seconds must be > 0, got {max_duration_seconds}"
            )
        if dt_safe <= 0.0:
            raise ValueError(f"distracted dt must be > 0, got {dt}")

        self.max_distracted_duration_seconds = duration_seconds
        self.distracted_duration = max(1, int(round(duration_seconds / dt_safe)))

    def update_following_duration(self, eligible_following: bool):
        """Accumulate following duration only when current step is eligible."""
        if eligible_following:
            self.following_steps += 1
            return
        self.reset_following_duration()

    def update_impatient_trigger_progress(self, eligible_following: bool, robot_speed: float):
        """Accumulate low-speed robot duration only during eligible following steps."""
        if (not eligible_following) or robot_speed >= self.following_impatient_robot_speed_threshold:
            self.reset_impatient_trigger_progress()
            return
        self.following_low_robot_speed_steps += 1

    def update_listening_session_progress(self, reached_target: bool):
        """Accumulate listening duration from first target reach until session end."""
        if reached_target and not self.listening_target_reached_once:
            self.listening_target_reached_once = True
            self.listening_started_this_session = True

        if self.listening_started_this_session:
            self.listening_steps += 1

    @staticmethod
    def _normalize_hazard_parameters(
        hazard_name: str,
        lambda_max_per_sec: float,
        ramp_start_seconds: float,
        rise_seconds: float,
    ):
        """Validate common sigmoid-hazard parameters and return normalized floats."""
        lambda_max = float(lambda_max_per_sec)
        ramp_start = float(ramp_start_seconds)
        rise = float(rise_seconds)
        if lambda_max < 0.0:
            raise ValueError(f"{hazard_name} lambda_max_per_sec must be >= 0, got {lambda_max_per_sec}")
        if ramp_start < 0.0:
            raise ValueError(f"{hazard_name} ramp_start_seconds must be >= 0, got {ramp_start_seconds}")
        if rise <= 0.0:
            raise ValueError(f"{hazard_name} rise_seconds must be > 0, got {rise_seconds}")
        return lambda_max, ramp_start, rise

    # Use sigmoid-based hazard increase probability over time
    def configure_distracted_follow_hazard(
        self,
        lambda_max_per_sec: float,
        ramp_start_seconds: float,
        rise_seconds: float,
    ):
        """Configure time-varying hazard used to trigger distracted-from-following."""
        lambda_max, ramp_start, rise = self._normalize_hazard_parameters(
            hazard_name="distracted",
            lambda_max_per_sec=lambda_max_per_sec,
            ramp_start_seconds=ramp_start_seconds,
            rise_seconds=rise_seconds,
        )
        self.following_distracted_lambda_max_per_sec = lambda_max
        self.following_distracted_ramp_start_seconds = ramp_start
        self.following_distracted_rise_seconds = rise

    def configure_distracted_listening_hazard(
        self,
        lambda_max_per_sec: float,
        ramp_start_seconds: float,
        rise_seconds: float,
    ):
        """Configure time-varying hazard used to trigger distracted-from-listening."""
        lambda_max, ramp_start, rise = self._normalize_hazard_parameters(
            hazard_name="distracted",
            lambda_max_per_sec=lambda_max_per_sec,
            ramp_start_seconds=ramp_start_seconds,
            rise_seconds=rise_seconds,
        )
        self.listening_distracted_lambda_max_per_sec = lambda_max
        self.listening_distracted_ramp_start_seconds = ramp_start
        self.listening_distracted_rise_seconds = rise

    def configure_impatient_follow_hazard(
        self,
        lambda_max_per_sec: float,
        ramp_start_seconds: float,
        rise_seconds: float,
        robot_speed_threshold: float,
    ):
        """Configure time-varying hazard used to trigger impatient-from-following."""
        lambda_max, ramp_start, rise = self._normalize_hazard_parameters(
            hazard_name="impatient",
            lambda_max_per_sec=lambda_max_per_sec,
            ramp_start_seconds=ramp_start_seconds,
            rise_seconds=rise_seconds,
        )
        speed_threshold = float(robot_speed_threshold)
        if speed_threshold <= 0.0:
            raise ValueError(
                "impatient robot_speed_threshold must be > 0, "
                f"got {robot_speed_threshold}"
            )
        self.following_impatient_lambda_max_per_sec = lambda_max
        self.following_impatient_ramp_start_seconds = ramp_start
        self.following_impatient_rise_seconds = rise
        self.following_impatient_robot_speed_threshold = speed_threshold

    def configure_listening_distracted_motion(
        self,
        dt: float = DEFAULT_SIM_TIMESTEP_SECONDS,
        move_seconds: float = LISTENING_DISTRACTED_MOVE_SECONDS,
    ):
        """Cache fixed listening-distracted move duration in simulation steps."""
        dt_safe = self._normalize_dt(dt)
        move_seconds = float(move_seconds)
        if move_seconds <= 0.0:
            raise ValueError(f"listening distracted move_seconds must be > 0, got {move_seconds}")
        self.listening_distracted_move_steps = max(1, int(round(move_seconds / dt_safe)))

    def _log_event(self, msg: str):
        """Emit log line only when logging is enabled."""
        if self.enable_event_logs:
            logger.info(msg)

    @staticmethod
    def _compute_fan_relative_angle(index, n_humans, fan_half_angle):
        """Map human index to a symmetric angle inside the formation fan."""
        if n_humans > 1:
            return (index / (n_humans - 1)) * (2 * fan_half_angle) - fan_half_angle
        return 0.0

    @staticmethod
    def _compute_fan_target(robot_pose, radius, relative_angle, base_angle_offset):
        """Compute Cartesian target around robot from polar fan parameters."""
        rx, ry, ryaw = robot_pose
        angle = ryaw + base_angle_offset + relative_angle
        return np.array(
            [rx + radius * np.cos(angle), ry + radius * np.sin(angle)],
            dtype=np.float32,
        )

    def start_overwhelmed(self, robot_xy, current_xy=None):
        """Enter OVERWHELMED and initialize two-stage retreat state."""
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
        """Enter IMPATIENT and temporarily increase max speed."""
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
        """Restore speed and timers when leaving impatient mode."""
        if self.impatient_original_max_speed is not None:
            self.max_speed = float(self.impatient_original_max_speed)
        else:
            self.max_speed = float(self.base_max_speed)
        self.impatient_original_max_speed = None
        self.impatient_timer = 0

    def _clear_callback_response_state(self):
        """Clear temporary state used by callback responses in distracted mode."""
        self.callback_response_mode = None

    def _clear_distracted_navigation_state(self):
        """Clear one-shot distracted target state for the current episode/mode."""
        self.distracted_target_xy = None
        self.distracted_stop_reached = False
        self.distracted_target_yaw = None
        self.listening_distracted_move_elapsed_steps = 0

    def _set_distracted_target_state(
        self,
        target_yaw: float,
        target_xy,
        *,
        reset_listening_move_elapsed_steps: bool = False,
    ):
        """Store one-shot distracted navigation state shared by distracted variants."""
        target_xy = np.array(target_xy, dtype=np.float32)
        self.distracted_target_yaw = float(target_yaw)
        self.distracted_target_xy = target_xy
        self.current_waypoint = target_xy.copy()
        self.distracted_stop_reached = False
        if reset_listening_move_elapsed_steps:
            self.listening_distracted_move_elapsed_steps = 0

    def _initialize_distracted_target(self, current_xy, current_yaw: float):
        """Sample one local distracted goal and deviated heading from the current pose."""
        current_xy = np.array(current_xy, dtype=np.float32)
        target_yaw, sampled_target_xy = self._sample_distracted_target_candidate(
            current_xy=current_xy,
            current_yaw=current_yaw,
        )
        target_xy = self._find_farthest_walkable_point_on_segment(
            start_xy=current_xy,
            end_xy=sampled_target_xy,
            margin=HUMAN_WALL_FOOTPRINT_RADIUS,
        )
        self._set_distracted_target_state(target_yaw=target_yaw, target_xy=target_xy)

    def _sample_distracted_target_candidate(self, current_xy, current_yaw: float):
        """Sample one distracted target candidate before wall/segment validity checks."""
        deviation_deg = float(
            np.random.uniform(
                DISTRACTED_YAW_DEVIATION_MIN_DEG,
                DISTRACTED_YAW_DEVIATION_MAX_DEG,
            )
        )
        deviation_sign = -1.0 if np.random.rand() < 0.5 else 1.0
        deviation_rad = np.deg2rad(deviation_deg) * deviation_sign
        target_yaw = self._wrap_to_pi(float(current_yaw) + deviation_rad)
        target_distance = float(
            np.random.uniform(
                DISTRACTED_TARGET_DISTANCE_MIN,
                DISTRACTED_TARGET_DISTANCE_MAX,
            )
        )
        raw_target_xy = np.array(current_xy, dtype=np.float32) + target_distance * np.array(
            [np.cos(target_yaw), np.sin(target_yaw)],
            dtype=np.float32,
        )
        return float(target_yaw), np.array(raw_target_xy, dtype=np.float32)

    def start_attack(self):
        """Enter ATTACK mode if this human is allowed to attack."""
        if not self.can_attack:
            return False
        self.attack_origin_listen_waypoint = np.array(self.current_waypoint, dtype=np.float32)
        self.transition_to(HumanMode.ATTACK, reason="trigger_attack")
        self.attack_hit_this_step = False
        self._log_event(f">>> {self.name} became ATTACK!")
        return True

    def force_recover_from_callback(self) -> bool:
        """Force callback outcome to immediate rejoin."""
        return self.apply_callback_response(response="rejoin", stay_steps=0)

    def apply_callback_response(self, response: str, stay_steps: int = 0) -> bool:
        """Apply callback response strategy while currently distracted."""
        if self.mode != HumanMode.DISTRACTED:
            return False
        if self.distracted_source != DISTRACTED_SOURCE_FOLLOWING:
            return False

        if response == "rejoin":
            self.transition_to(HumanMode.FOLLOWING, reason="callback_rejoin")
            return True

        if response == "ignore":
            return True

        raise ValueError(f"Unknown callback response: {response}")

    @staticmethod
    def _normalize_dt(dt: float) -> float:
        """Clamp dt to a small positive epsilon to avoid numerical issues."""
        return max(float(dt), MIN_SPEED_EPS)

    def _get_distracted_follow_threshold_steps(self, dt: float = DEFAULT_SIM_TIMESTEP_SECONDS) -> int:
        """Convert distracted ramp-start time (sec) to step threshold."""
        dt_safe = self._normalize_dt(dt)
        return int(np.floor(self.following_distracted_ramp_start_seconds / dt_safe))

    @staticmethod
    def _compute_distracted_lambda_per_sec_with_dt_safe(
        elapsed_steps: int,
        lambda_max_per_sec: float,
        ramp_start_seconds: float,
        rise_seconds: float,
        dt_safe: float,
    ) -> float:
        """Compute instantaneous distracted hazard rate (per second)."""
        if lambda_max_per_sec <= 0.0:
            return 0.0
        elapsed_seconds = float(elapsed_steps) * dt_safe
        if elapsed_seconds <= ramp_start_seconds:
            return 0.0
        x = (
            (elapsed_seconds - ramp_start_seconds)
            / rise_seconds
        )
        z = np.clip(-DISTRACTED_HAZARD_SIGMOID_K * (x - 0.5), -60.0, 60.0)
        sig = 1.0 / (1.0 + np.exp(z))
        progress = float(np.clip((sig - 0.1) / 0.8, 0.0, 1.0))
        return float(lambda_max_per_sec * progress)

    @staticmethod
    def _compute_step_probability_from_lambda(lambda_t: float, dt_safe: float) -> float:
        """Convert instantaneous hazard rate to per-step Bernoulli probability."""
        if lambda_t <= 0.0:
            return 0.0
        return float(1.0 - np.exp(-lambda_t * dt_safe))

    def _compute_distracted_step_probability(
        self,
        source: str,
        dt: float = DEFAULT_SIM_TIMESTEP_SECONDS,
    ) -> float:
        """Compute distracted trigger probability for one step from the selected source."""
        dt_safe = self._normalize_dt(dt)
        if source == DISTRACTED_SOURCE_FOLLOWING:
            elapsed_steps = self.following_steps
            lambda_max_per_sec = self.following_distracted_lambda_max_per_sec
            ramp_start_seconds = self.following_distracted_ramp_start_seconds
            rise_seconds = self.following_distracted_rise_seconds
        elif source == DISTRACTED_SOURCE_LISTENING:
            elapsed_steps = self.listening_steps
            lambda_max_per_sec = self.listening_distracted_lambda_max_per_sec
            ramp_start_seconds = self.listening_distracted_ramp_start_seconds
            rise_seconds = self.listening_distracted_rise_seconds
        else:
            raise ValueError(f"Unknown distracted source: {source}")

        lambda_t = self._compute_distracted_lambda_per_sec_with_dt_safe(
            elapsed_steps=elapsed_steps,
            lambda_max_per_sec=lambda_max_per_sec,
            ramp_start_seconds=ramp_start_seconds,
            rise_seconds=rise_seconds,
            dt_safe=dt_safe,
        )
        return self._compute_step_probability_from_lambda(lambda_t=lambda_t, dt_safe=dt_safe)

    def _compute_impatient_follow_step_probability(
        self,
        dt: float = DEFAULT_SIM_TIMESTEP_SECONDS,
    ) -> float:
        """Compute per-step Bernoulli trigger probability for impatient-from-following."""
        dt_safe = self._normalize_dt(dt)
        lambda_t = self._compute_distracted_lambda_per_sec_with_dt_safe(
            elapsed_steps=self.following_low_robot_speed_steps,
            lambda_max_per_sec=self.following_impatient_lambda_max_per_sec,
            ramp_start_seconds=self.following_impatient_ramp_start_seconds,
            rise_seconds=self.following_impatient_rise_seconds,
            dt_safe=dt_safe,
        )
        return self._compute_step_probability_from_lambda(lambda_t=lambda_t, dt_safe=dt_safe)

    # whether to trigger distracted following variant based on current probability
    def _maybe_trigger_distracted_following_variant(self, dt: float = DEFAULT_SIM_TIMESTEP_SECONDS):
        """Sample whether FOLLOWING should switch to DISTRACTED at this step."""
        step_prob = self._compute_distracted_step_probability(
            source=DISTRACTED_SOURCE_FOLLOWING,
            dt=dt,
        )
        trigger_distracted = (
            self.following_distracted_window_active
            and step_prob > 0.0
            and np.random.rand() < step_prob
        )
        if trigger_distracted:
            return HumanMode.DISTRACTED
        return None

    def _maybe_trigger_distracted_listening_variant(self, dt: float = DEFAULT_SIM_TIMESTEP_SECONDS):
        """Sample whether LISTENING should switch to DISTRACTED at this step."""
        step_prob = self._compute_distracted_step_probability(
            source=DISTRACTED_SOURCE_LISTENING,
            dt=dt,
        )
        trigger_distracted = (
            self.listening_distracted_window_active
            and (not self.listening_distracted_hold_until_session_end)
            and step_prob > 0.0
            and np.random.rand() < step_prob
        )
        if trigger_distracted:
            return HumanMode.DISTRACTED
        return None

    def _maybe_trigger_impatient_following_variant(self, dt: float = DEFAULT_SIM_TIMESTEP_SECONDS):
        """Sample whether FOLLOWING should switch to IMPATIENT at this step."""
        step_prob = self._compute_impatient_follow_step_probability(dt=dt)
        trigger_impatient = (
            self.can_be_impatient
            and step_prob > 0.0
            and np.random.rand() < step_prob
        )
        if trigger_impatient:
            return HumanMode.IMPATIENT
        return None

    def _maybe_trigger_following_variant(self, dt: float = DEFAULT_SIM_TIMESTEP_SECONDS):
        """Sample higher-priority following variants in deterministic priority order."""
        # Distracted has higher priority than impatient when both are eligible.
        distracted_variant = self._maybe_trigger_distracted_following_variant(dt=dt)
        if distracted_variant is not None:
            return distracted_variant
        return self._maybe_trigger_impatient_following_variant(dt=dt)


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

    def _assign_target_from_context(self, mode: Optional[str] = None):
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
        target_mode = self.mode if mode is None else mode
        if target_mode == HumanMode.LISTENING:
            radius = self.context.listen_radius
            self.current_waypoint = self._compute_fan_target(
                robot_pose=robot_pose,
                radius=radius,
                relative_angle=relative_angle,
                base_angle_offset=0.0,
            )

        elif target_mode == HumanMode.FOLLOWING:
            radius = self.context.follow_radius
            self.current_waypoint = self._compute_fan_target(
                robot_pose=robot_pose,
                radius=radius,
                relative_angle=relative_angle,
                base_angle_offset=np.pi,
            )

        elif target_mode == HumanMode.IMPATIENT:
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
        # Cache MuJoCo ids once; avoids repeated model lookups every step.
        if self.body_id is None:
            self.body_id = model.body(self.body_name).id

        if self.x_dof_idx is None:
            self.x_dof_idx = model.jnt_dofadr[model.joint(f"{self.name}_x").id]
            self.y_dof_idx = model.jnt_dofadr[model.joint(f"{self.name}_y").id]

        if self.mode == HumanMode.WANDERING:
            return self._step_wandering(data, ctx)

        if self.mode == HumanMode.FOLLOWING:
            # FOLLOWING may stochastically branch to IMPATIENT or DISTRACTED.
            variant = self._maybe_trigger_following_variant(dt=float(ctx.get("dt", DEFAULT_SIM_TIMESTEP_SECONDS)))
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
                self.distracted_source = DISTRACTED_SOURCE_FOLLOWING
                self.transition_to(HumanMode.DISTRACTED, reason="following_variant_distracted")
                self._log_event(f">>> {self.name} became DISTRACTED!")
                return self._step_distracted(data, ctx)

            self._assign_target_from_context()
            return self._step_following(data, ctx)

        if self.mode == HumanMode.LISTENING:
            variant = self._maybe_trigger_distracted_listening_variant(
                dt=float(ctx.get("dt", DEFAULT_SIM_TIMESTEP_SECONDS))
            )
            if variant == HumanMode.DISTRACTED:
                self.distracted_source = DISTRACTED_SOURCE_LISTENING
                self.listening_distracted_hold_until_session_end = True
                self.transition_to(HumanMode.DISTRACTED, reason="listening_variant_distracted")
                self._log_event(f">>> {self.name} became DISTRACTED during LISTENING!")
                return self._step_distracted(data, ctx)
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
        
    def _step_wandering(self, data, ctx):
        """WANDERING: move to random waypoint, and resample when reached."""
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
        """FOLLOWING: move toward current follow target."""
        x, y, yaw = self._get_pose(data)
        dx = self.current_waypoint[0] - x
        dy = self.current_waypoint[1] - y
        return self._move(dx, dy, yaw, data, ctx)
    
    def _step_listening(self, data, ctx):
        """LISTENING: approach listen slot, then rotate to face robot."""
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
        """DISTRACTED: make one local deviated move, then stop until recovery/callback."""
        if self.distracted_source == DISTRACTED_SOURCE_LISTENING:
            return self._step_listening_distracted(data, ctx)

        x, y, yaw = self._get_pose(data)
        self.distracted_timer += 1

        current_xy = np.array([x, y], dtype=np.float32)
        # Generate one-shot distracted target
        if self.distracted_target_xy is None:
            self._initialize_distracted_target(current_xy=current_xy, current_yaw=yaw)

        target_xy = np.array(self.distracted_target_xy, dtype=np.float32)
        to_target = target_xy - current_xy
        dist_to_target = float(np.linalg.norm(to_target))
        if dist_to_target < self.waypoint_threshold:
            self.distracted_stop_reached = True

        if self.distracted_stop_reached:
            self.last_v_follow = np.zeros(2, dtype=np.float32)
            self.last_v_repulsion = np.zeros(2, dtype=np.float32)
            self.last_v_hr = np.zeros(2, dtype=np.float32)
            action = np.zeros(3, dtype=np.float32)
        else:
            move_speed_limit = float(DISTRACTED_SPEED_SCALE * self.max_speed)
            if dist_to_target > NORM_EPS:
                v_goal = move_speed_limit * (to_target / dist_to_target)
            else:
                v_goal = np.zeros(2, dtype=np.float32)

            v_repulsion = np.array(ctx.get("repulsion", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            v_total = v_goal + v_repulsion
            speed = float(np.linalg.norm(v_total))
            if speed > move_speed_limit and speed > NORM_EPS:
                v_total = v_total / speed * move_speed_limit
                speed = float(np.linalg.norm(v_total))

            desired_yaw = (
                np.arctan2(v_total[1], v_total[0])
                if speed > NORM_EPS
                else float(self.distracted_target_yaw if self.distracted_target_yaw is not None else yaw)
            )
            yaw_err = self._wrap_to_pi(desired_yaw - yaw)
            self.last_v_follow = np.array(v_goal, dtype=np.float32)
            self.last_v_repulsion = np.array(v_repulsion, dtype=np.float32)
            self.last_v_hr = np.zeros(2, dtype=np.float32)
            action = np.array([v_total[0], v_total[1], HUMAN_YAW_RATE_GAIN * yaw_err], dtype=np.float32)

        # Recover after duration
        if self.distracted_timer >= self.distracted_duration:
            self.transition_to(HumanMode.FOLLOWING, reason="distracted_timeout_recover")
            self._log_event(f">>> {self.name} recovered -> FOLLOWING")

        return self._apply_wall_constraint_to_action(action, data, ctx)

    def _initialize_listening_distracted_target(self, current_xy, current_yaw: float):
        """Sample one slight yaw offset, then a backward move target for listening distraction."""
        current_xy = np.array(current_xy, dtype=np.float32)
        deviation_deg = float(
            np.random.uniform(
                LISTENING_DISTRACTED_YAW_DEVIATION_MIN_DEG,
                LISTENING_DISTRACTED_YAW_DEVIATION_MAX_DEG,
            )
        )
        deviation_sign = -1.0 if np.random.rand() < 0.5 else 1.0
        target_yaw = self._wrap_to_pi(float(current_yaw) + deviation_sign * np.deg2rad(deviation_deg))
        target_distance = float(LISTENING_DISTRACTED_SPEED_METERS_PER_SEC * LISTENING_DISTRACTED_MOVE_SECONDS)
        backward_dir = -np.array(
            [np.cos(target_yaw), np.sin(target_yaw)],
            dtype=np.float32,
        )
        sampled_target_xy = current_xy + target_distance * backward_dir
        target_xy = self._find_farthest_walkable_point_on_segment(
            start_xy=current_xy,
            end_xy=sampled_target_xy,
            margin=HUMAN_WALL_FOOTPRINT_RADIUS,
        )
        self._set_distracted_target_state(
            target_yaw=target_yaw,
            target_xy=target_xy,
            reset_listening_move_elapsed_steps=True,
        )

    def _step_listening_distracted(self, data, ctx):
        """DISTRACTED-from-listening: move away for 1 second, then freeze until session end."""
        x, y, yaw = self._get_pose(data)
        self.distracted_timer += 1
        current_xy = np.array([x, y], dtype=np.float32)

        if self.distracted_target_yaw is None:
            self._initialize_listening_distracted_target(
                current_xy=current_xy,
                current_yaw=yaw,
            )

        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)
        desired_yaw = float(self.distracted_target_yaw)
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        if abs(yaw_err) >= np.deg2rad(HUMAN_ROTATION_STOP_DEG):
            self.last_v_follow = np.zeros(2, dtype=np.float32)
            return np.array([0.0, 0.0, HUMAN_YAW_RATE_GAIN * yaw_err], dtype=np.float32)

        if self.listening_distracted_move_elapsed_steps >= self.listening_distracted_move_steps:
            self.last_v_follow = np.zeros(2, dtype=np.float32)
            return np.zeros(3, dtype=np.float32)

        move_dir = -np.array(
            [np.cos(self.distracted_target_yaw), np.sin(self.distracted_target_yaw)],
            dtype=np.float32,
        )
        v_goal = float(LISTENING_DISTRACTED_SPEED_METERS_PER_SEC) * move_dir
        self.last_v_follow = np.array(v_goal, dtype=np.float32)
        self.listening_distracted_move_elapsed_steps += 1
        action = np.array([v_goal[0], v_goal[1], HUMAN_YAW_RATE_GAIN * yaw_err], dtype=np.float32)
        return self._apply_wall_constraint_to_action(action, data, ctx)

    def _step_overwhelmed(self, data, ctx):
        """OVERWHELMED: first back off, then keep leaving for fixed duration."""
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
        """IMPATIENT: fast following toward front slot, with timeout recovery."""
        self.impatient_timer += 1
        self._assign_target_from_context()
        action = self._step_following(data, ctx)

        if self.impatient_timer >= self.impatient_duration:
            self.set_mode(HumanMode.FOLLOWING)
            self._log_event(f">>> {self.name} recovered from IMPATIENT -> FOLLOWING")

        return action

    def _step_attack(self, data, ctx):
        """ATTACK: chase robot until hit distance, then return to listening."""
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
        """Shared low-level controller that fuses follow, repulsion and HR spacing forces."""
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
            if dist_hr < self.hr_distance_min:
                # too close → repulsion (slow down / move away)
                v_hr = HR_REPULSION_GAIN * (self.hr_distance_min - dist_hr) * dir_hr

            elif dist_hr > self.hr_distance_max:
                # too far → attraction (move towards robot)
                v_hr = -HR_ATTRACTION_GAIN * (dist_hr - self.hr_distance_max) * dir_hr

        dist = np.hypot(dx, dy)
        if dist > NORM_EPS:
            v_follow = self.max_speed * np.array([dx, dy]) / dist
        else:
            v_follow = np.zeros(2)

        v_repulsion = np.array(repulsion, dtype=np.float32)

        self.last_v_follow = v_follow.copy()
        self.last_v_repulsion = v_repulsion.copy()
        self.last_v_hr = v_hr.copy()

        # Final translational command is a blend of:
        # (1) target following, (2) human-human repulsion, (3) human-robot spacing force.
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
        """Return walkable map rectangles after shrinking by safety margin."""
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
    def sample_point_in_rects(rects, rng=None):
        """Sample one point uniformly over a set of axis-aligned rectangles."""
        if not rects:
            return np.array([0.0, 0.0], dtype=np.float32)

        rng_choice = np.random if rng is None else rng
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
        rect_idx = int(rng_choice.choice(len(rects), p=probs))
        xmin, xmax, ymin, ymax = rects[rect_idx]
        wx = float(rng_choice.uniform(xmin, xmax))
        wy = float(rng_choice.uniform(ymin, ymax))
        return np.array([wx, wy], dtype=np.float32)

    @classmethod
    def sample_walkable_point(cls, margin: float = HUMAN_WALL_FOOTPRINT_RADIUS, rng=None):
        """Sample one point from walkable space."""
        rects = cls._walkable_rects(margin)
        return cls.sample_point_in_rects(rects, rng=rng)

    @classmethod
    def sample_room_a_point(cls, margin: float = HUMAN_WALL_FOOTPRINT_RADIUS, rng=None):
        """Sample one point inside Room A."""
        m = max(0.0, float(margin))
        rects = [(
            ROOM_A_X_MIN + m,
            ROOM_A_X_MAX - m,
            ROOM_A_Y_MIN + m,
            ROOM_A_Y_MAX - m,
        )]
        valid_rects = []
        for xmin, xmax, ymin, ymax in rects:
            if xmin <= xmax and ymin <= ymax:
                valid_rects.append((xmin, xmax, ymin, ymax))
        if not valid_rects:
            return np.array(
                [0.5 * (ROOM_A_X_MIN + ROOM_A_X_MAX), 0.5 * (ROOM_A_Y_MIN + ROOM_A_Y_MAX)],
                dtype=np.float32,
            )
        return cls.sample_point_in_rects(valid_rects, rng=rng)

    @staticmethod
    def _is_point_in_rect(x: float, y: float, rect) -> bool:
        """Check if point lies inside one axis-aligned rectangle."""
        xmin, xmax, ymin, ymax = rect
        return bool((xmin <= x <= xmax) and (ymin <= y <= ymax))

    def _is_point_in_walkable(self, xy, margin: float) -> bool:
        """Check if point belongs to any walkable rectangle."""
        x = float(xy[0])
        y = float(xy[1])
        return any(self._is_point_in_rect(x, y, rect) for rect in self._walkable_rects(margin))

    def _project_point_to_walkable(self, xy, margin: float):
        """Project point to nearest point inside walkable region."""
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

    def _is_segment_walkable(self, start_xy, end_xy, margin: float):
        """Check whether the full straight segment stays inside walkable space."""
        start_xy = np.array(start_xy, dtype=np.float32)
        end_xy = np.array(end_xy, dtype=np.float32)
        if not self._is_point_in_walkable(start_xy, margin):
            return False
        if not self._is_point_in_walkable(end_xy, margin):
            return False

        segment = end_xy - start_xy
        dist = float(np.linalg.norm(segment))
        if dist <= MIN_SPEED_EPS:
            return True

        n_steps = max(1, int(np.ceil(dist / SEGMENT_CHECK_SPACING)))
        for alpha in np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32):
            point = start_xy + alpha * segment
            if not self._is_point_in_walkable(point, margin):
                return False
        return True

    def _find_farthest_walkable_point_on_segment(self, start_xy, end_xy, margin: float):
        """Return the farthest point from start that keeps the segment walkable."""
        start_xy = np.array(start_xy, dtype=np.float32)
        end_xy = np.array(end_xy, dtype=np.float32)
        if not self._is_point_in_walkable(start_xy, margin):
            return self._project_point_to_walkable(start_xy, margin)
        if self._is_segment_walkable(start_xy, end_xy, margin):
            return end_xy

        best_point = start_xy.copy()
        lo = 0.0
        hi = 1.0
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            candidate = start_xy + mid * (end_xy - start_xy)
            if self._is_segment_walkable(start_xy, candidate, margin):
                best_point = candidate
                lo = mid
            else:
                hi = mid
        return np.array(best_point, dtype=np.float32)

    def _constrain_velocity_with_walkable(self, x: float, y: float, v_xy, dt: float, margin: float):
        """Adjust velocity so the next-step position stays in walkable area."""
        # Project next-step position back to walkable area when command exits map bounds.
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
        """Apply wall constraint to translational action components."""
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
            margin=HUMAN_WALL_FOOTPRINT_RADIUS,
        )
        return constrained_action

    def _get_pose(self, data):
        """Read current (x, y, yaw) of this human from MuJoCo state."""
        x = float(data.xpos[self.body_id, 0])
        y = float(data.xpos[self.body_id, 1])
        yaw = float(data.qpos[self.qpos_idx + 2])
        return x, y, yaw

    def _random_waypoint(self):
        """Generate random waypoint within walkable museum bounds."""
        return self.sample_walkable_point(HUMAN_WALL_FOOTPRINT_RADIUS, rng=np.random)
    
    def _wrap_to_pi(self, ang):
        """Normalize angle to [-pi, pi)."""
        return (ang + np.pi) % (2 * np.pi) - np.pi
    
