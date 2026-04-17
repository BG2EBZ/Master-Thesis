import logging
from typing import Optional, Tuple

import mujoco
import numpy as np

from .map_layouts import DEFAULT_MUSEUM_LAYOUT, MapLayout

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
LISTENING_RING_GAIN = 4.0
LISTENING_FRONT_SECTOR_HALF_ANGLE_DEG = 80.0
LISTENING_FRONT_SECTOR_HALF_ANGLE = np.deg2rad(LISTENING_FRONT_SECTOR_HALF_ANGLE_DEG)
LISTENING_SECTOR_PROJECTION_EPS = 1e-2
DISTRACTED_SPEED_SCALE = 0.5
DISTRACTED_YAW_DEVIATION_MIN_DEG = 45.0
DISTRACTED_YAW_DEVIATION_MAX_DEG = 90.0
DISTRACTED_TARGET_DISTANCE_MIN = 0.5
DISTRACTED_TARGET_DISTANCE_MAX = 1.5
DEFAULT_SIM_TIMESTEP_SECONDS = 0.002
DISTRACTED_DURATION_SECONDS_DEFAULT = 10.0
LISTENING_DISTRACTED_YAW_DEVIATION_MIN_DEG = 10.0
LISTENING_DISTRACTED_YAW_DEVIATION_MAX_DEG = 40.0
LISTENING_DISTRACTED_SPEED_METERS_PER_SEC = 0.3
LISTENING_DISTRACTED_MOVE_SECONDS = 2.0
LISTENING_IMPATIENT_YAW_DEVIATION_MIN_DEG = 45.0
LISTENING_IMPATIENT_YAW_DEVIATION_MAX_DEG = 90.0
LISTENING_IMPATIENT_SWAY_SPEED_METERS_PER_SEC = 0.08
LISTENING_IMPATIENT_TARGET_REACHED_DEG = 5.0
OVERWHELMED_FALLBACK_X_OFFSET = 1.0
OVERWHELMED_STAGE_SWITCH_DIST = 0.02
HR_DISTANCE_MIN = 0.8
HR_DISTANCE_MAX = 2.0
HR_DISTANCE_MAX_NORMAL_DEFAULT = 1.5
HR_DISTANCE_MIN_ND_DEFAULT = 1.0
HR_REPULSION_GAIN = 4.0
HR_ATTRACTION_GAIN = 2.0
HR_REPULSION_GAIN_MID_DISTANCE = 1.2
HR_REPULSION_GAIN_NEAR_DISTANCE = 0.8
HR_REPULSION_GAIN_MID_MULTIPLIER = 4.0
HR_REPULSION_GAIN_NEAR_MULTIPLIER = 10.0
NORM_EPS = 1e-6
HUMAN_WALL_FOOTPRINT_RADIUS = 0.25
MIN_SPEED_EPS = 1e-6
RAYCAST_CLEARANCE_EPS = 1e-3
RAYCAST_SLOWDOWN_DISTANCE_METERS = 0.3
DISTRACTED_SOURCE_FOLLOWING = "following"
DISTRACTED_SOURCE_LISTENING = "listening"


class HumanMode:
    WANDERING = "wandering"
    FOLLOWING = "following"
    LISTENING = "listening"
    DISTRACTED = "distracted"
    OVERWHELMED = "overwhelmed"
    IMPATIENT = "impatient"


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
        map_layout: Optional[MapLayout] = None,
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
        self.map_layout = DEFAULT_MUSEUM_LAYOUT if map_layout is None else map_layout
        setattr(self, "mode", None)

        self.enable_event_logs = True
        
        # Store body_id (will be set when we have access to model)
        self.body_id = None
        self.x_dof_idx = None
        self.y_dof_idx = None
        self._runtime_model = None
        self._runtime_data = None
        
        # Current target waypoint
        self.current_waypoint = self._random_waypoint()
        self.step_count = 0
        # Cache for debugging/metrics
        self.last_v_follow = np.zeros(2, dtype=np.float32)
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)  # human-robot force
        self.last_in_listening_front_sector = False
        self.hr_distance_min = float(HR_DISTANCE_MIN)
        self.hr_distance_max = float(HR_DISTANCE_MAX)

        self.distracted_timer = 0
        self.max_distracted_duration_seconds = float(DISTRACTED_DURATION_SECONDS_DEFAULT)
        self.distracted_duration = 0
        self.configure_distracted_duration(
            max_duration_seconds=self.max_distracted_duration_seconds,
            dt=DEFAULT_SIM_TIMESTEP_SECONDS,
        )
        self.distracted_target_xy = None
        self.distracted_stop_reached = False
        self.distracted_target_yaw = None
        self.distracted_recovery_mode = HumanMode.FOLLOWING

        self.can_be_impatient = False
        self.impatient_duration = 800
        self.impatient_timer = 0
        self.impatient_speed_multiplier = 1.3
        self.impatient_front_offset = DEFAULT_IMPATIENT_FRONT_OFFSET
        self.impatient_original_max_speed = None
        self.impatient_recovery_mode = HumanMode.FOLLOWING
        self.listening_impatient_yaw_deviation = 0.0
        self.listening_impatient_turn_sign = 1.0
        self.profile = HumanProfile.NORMAL
        self.following_steps = 0
        self.listening_steps = 0
        self.listening_started_this_session = False
        self.listening_distracted_window_active = False
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
        self.overwhelmed_recovery_mode = HumanMode.FOLLOWING

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
            self._clear_distracted_navigation_state()
            self.distracted_source = None
            self.distracted_recovery_mode = HumanMode.FOLLOWING
        if prev_mode == HumanMode.FOLLOWING and next_mode != HumanMode.FOLLOWING:
            self.reset_following_duration()
        if prev_mode == HumanMode.OVERWHELMED and next_mode != HumanMode.OVERWHELMED:
            self.reset_overwhelmed_state()

    def _on_enter_mode(self, prev_mode: Optional[str], next_mode: str, reason: Optional[str] = None):
        """Run initialization hooks when entering a mode."""
        if next_mode == HumanMode.DISTRACTED:
            self.distracted_timer = 0
            self._clear_distracted_navigation_state()

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
        self.overwhelmed_recovery_mode = HumanMode.FOLLOWING

    def reset_episode_state(self):
        """Reset per-episode dynamic state while keeping static config."""
        self.step_count = 0
        self.transition_to(HumanMode.WANDERING, reason="episode_reset", force=True)
        self.current_waypoint = self._random_waypoint()

        self.distracted_timer = 0
        self._clear_distracted_navigation_state()

        self.impatient_timer = 0
        self.impatient_original_max_speed = None
        self.impatient_recovery_mode = HumanMode.FOLLOWING
        self.listening_impatient_yaw_deviation = 0.0
        self.listening_impatient_turn_sign = 1.0

        self.last_v_follow = np.zeros(2, dtype=np.float32)
        self.last_v_repulsion = np.zeros(2, dtype=np.float32)
        self.last_v_hr = np.zeros(2, dtype=np.float32)
        self.last_in_listening_front_sector = False

        self.max_speed = float(self.base_max_speed)
        self.reset_following_duration()
        self.reset_listening_session_state()
        self.reset_overwhelmed_state()

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

    def reset_listening_session_state(self):
        """Reset per-session listening hazard state."""
        self.listening_steps = 0
        self.listening_started_this_session = False
        self.listening_distracted_window_active = False
        self.listening_distracted_move_elapsed_steps = 0
        self.listening_distracted_hold_until_session_end = False

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

    def update_listening_session_progress(self, active: bool):
        """Accumulate listening duration for the active listening session."""
        if not active:
            return
        self.listening_started_this_session = True
        self.listening_steps += 1

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
        target_xy = np.empty(2, dtype=np.float32)
        target_xy[0] = rx + radius * np.cos(angle)
        target_xy[1] = ry + radius * np.sin(angle)
        return target_xy

    def start_overwhelmed(self, robot_xy, current_xy=None, recovery_mode: str = HumanMode.FOLLOWING):
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
        self.overwhelmed_recovery_mode = recovery_mode
        self.overwhelmed_stage = "backoff"
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_leave_dir = leave_dir.astype(np.float32)
        self.overwhelmed_robot_ref_xy = robot_xy
        self.overwhelmed_backoff_start_xy = current_xy
        self._log_event(f">>> {self.name} became OVERWHELMED!")

    def start_impatient(
        self,
        robot_pose=None,
        index=None,
        n_humans=None,
        recovery_mode: str = HumanMode.FOLLOWING,
    ):
        """Enter IMPATIENT and temporarily increase max speed."""
        if not self.can_be_impatient:
            return False
        if self.mode == HumanMode.IMPATIENT:
            return True

        self.impatient_original_max_speed = float(self.max_speed)
        self.max_speed = float(self.impatient_original_max_speed * self.impatient_speed_multiplier)
        self.impatient_timer = 0
        self.impatient_recovery_mode = recovery_mode
        if recovery_mode == HumanMode.LISTENING:
            self.listening_impatient_yaw_deviation = np.deg2rad(
                float(
                    np.random.uniform(
                        LISTENING_IMPATIENT_YAW_DEVIATION_MIN_DEG,
                        LISTENING_IMPATIENT_YAW_DEVIATION_MAX_DEG,
                    )
                )
            )
            self.listening_impatient_turn_sign = 1.0 if np.random.rand() >= 0.5 else -1.0
        else:
            self.listening_impatient_yaw_deviation = 0.0
            self.listening_impatient_turn_sign = 1.0
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
        self.impatient_recovery_mode = HumanMode.FOLLOWING
        self.listening_impatient_yaw_deviation = 0.0
        self.listening_impatient_turn_sign = 1.0

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
        target_xy = np.asarray(target_xy, dtype=np.float32)
        self.distracted_target_yaw = float(target_yaw)
        self.distracted_target_xy = target_xy
        self.current_waypoint = target_xy.copy()
        self.distracted_stop_reached = False
        if reset_listening_move_elapsed_steps:
            self.listening_distracted_move_elapsed_steps = 0

    def _initialize_distracted_target(self, current_xy, current_yaw: float):
        """Sample one local distracted goal and deviated heading from the current pose."""
        current_xy = np.asarray(current_xy, dtype=np.float32)
        target_yaw, sampled_target_xy = self._sample_distracted_target_candidate(
            current_xy=current_xy,
            current_yaw=current_yaw,
        )
        self._set_distracted_target_state(target_yaw=target_yaw, target_xy=sampled_target_xy)

    def _sample_distracted_target_candidate(self, current_xy, current_yaw: float):
        """Sample one distracted target candidate before wall/segment validity checks."""
        current_xy = np.asarray(current_xy, dtype=np.float32)
        deviation_deg = np.random.uniform(
            DISTRACTED_YAW_DEVIATION_MIN_DEG,
            DISTRACTED_YAW_DEVIATION_MAX_DEG,
        )
        deviation_sign = -1.0 if np.random.rand() < 0.5 else 1.0
        deviation_rad = np.deg2rad(deviation_deg) * deviation_sign
        target_yaw = self._wrap_to_pi(current_yaw + deviation_rad)
        target_distance = np.random.uniform(
            DISTRACTED_TARGET_DISTANCE_MIN,
            DISTRACTED_TARGET_DISTANCE_MAX,
        )
        direction_xy = np.array([np.cos(target_yaw), np.sin(target_yaw)], dtype=np.float32)
        raw_target_xy = current_xy + target_distance * direction_xy
        return float(target_yaw), np.asarray(raw_target_xy, dtype=np.float32)

    def force_recover_from_callback(self) -> bool:
        """Force callback outcome to immediate rejoin."""
        return self.apply_callback_response(response="rejoin", stay_steps=0)

    def apply_callback_response(self, response: str, stay_steps: int = 0) -> bool:
        """Apply callback response strategy while currently distracted."""
        if self.mode != HumanMode.DISTRACTED:
            return False

        if response == "rejoin":
            self.transition_to(self.distracted_recovery_mode, reason="callback_rejoin")
            return True

        if response == "ignore":
            return True

        raise ValueError(f"Unknown callback response: {response}")

    @staticmethod
    def _normalize_dt(dt: float) -> float:
        """Clamp dt to a small positive epsilon to avoid numerical issues."""
        return max(float(dt), MIN_SPEED_EPS)

    def _reset_motion_debug(self):
        """Reset cached motion components without reallocating the debug arrays."""
        self.last_v_follow.fill(0.0)
        self.last_v_repulsion.fill(0.0)
        self.last_v_hr.fill(0.0)

    def _set_motion_debug(self, *, v_follow=None, v_repulsion=None, v_hr=None):
        """Update cached motion components in place for diagnostics/tests."""
        if v_follow is None:
            self.last_v_follow.fill(0.0)
        else:
            self.last_v_follow[:] = v_follow

        if v_repulsion is None:
            self.last_v_repulsion.fill(0.0)
        else:
            self.last_v_repulsion[:] = v_repulsion

        if v_hr is None:
            self.last_v_hr.fill(0.0)
        else:
            self.last_v_hr[:] = v_hr

    @staticmethod
    def _compose_action(v_xy, yaw_rate):
        """Pack XY velocity and yaw rate into the expected float32 action vector."""
        action = np.zeros(3, dtype=np.float32)
        action[:2] = v_xy
        action[2] = np.float32(yaw_rate)
        return action

    def _assign_target_from_context(self, ctx: dict, mode: Optional[str] = None, current_xy=None):
        """
        Determine target waypoint based on social context dict.
        
        Args:
            ctx: context dict with keys like 'index', 'n_humans', 'robot_pose', 
                 'fan_half_angle', 'follow_radius', 'impatient_front_offset', etc.
            mode: optionally override the target computation mode
            current_xy: ignored (kept for signature compatibility)
        """
        del current_xy
        index = ctx.get("index", 0)
        n_humans = ctx.get("n_humans", 1)
        robot_pose = ctx.get("robot_pose")

        if robot_pose is None:
            return

        fan_half_angle = ctx.get("fan_half_angle", DEFAULT_FAN_HALF_ANGLE)
        relative_angle = self._compute_fan_relative_angle(index, n_humans, fan_half_angle)
        target_mode = self.mode if mode is None else mode
        raw_target_xy = None
        if target_mode == HumanMode.FOLLOWING:
            radius = ctx.get("follow_radius", DEFAULT_FOLLOW_RADIUS)
            raw_target_xy = self._compute_fan_target(
                robot_pose=robot_pose,
                radius=radius,
                relative_angle=relative_angle,
                base_angle_offset=np.pi,
            )

        elif target_mode == HumanMode.IMPATIENT:
            radius = ctx.get("impatient_front_offset")
            if radius is None:
                radius = self.impatient_front_offset
            raw_target_xy = self._compute_fan_target(
                robot_pose=robot_pose,
                radius=radius,
                relative_angle=relative_angle,
                base_angle_offset=0.0,
            )

        if raw_target_xy is None:
            return

        self.current_waypoint = np.asarray(raw_target_xy, dtype=np.float32)
    
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
        self._runtime_model = model
        self._runtime_data = data

        pose = self._get_pose(data)
        if self.mode == HumanMode.WANDERING:
            return self._step_wandering(data, ctx, pose)

        if self.mode == HumanMode.FOLLOWING:
            current_xy = np.asarray(pose[:2], dtype=np.float32)
            self._assign_target_from_context(ctx, current_xy=current_xy)
            return self._step_following(data, ctx, pose)

        if self.mode == HumanMode.LISTENING:
            return self._step_listening(data, ctx, pose)
        
        if self.mode == HumanMode.DISTRACTED:
            return self._step_distracted(data, ctx, pose)

        if self.mode == HumanMode.OVERWHELMED:
            return self._step_overwhelmed(data, ctx, pose)

        if self.mode == HumanMode.IMPATIENT:
            return self._step_impatient(data, ctx, pose)

        raise ValueError(f"Unknown human mode {self.mode}")
        
    def _step_wandering(self, data, ctx, pose):
        """WANDERING: move to random waypoint, and resample when reached."""
        _, _, yaw = pose
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        to_waypoint = self.current_waypoint - current_xy
        dist = np.linalg.norm(to_waypoint)

        if dist < self.waypoint_threshold:
            self.current_waypoint = self._random_waypoint()
            to_waypoint = self.current_waypoint - current_xy

        return self._move(to_waypoint, yaw, ctx, current_xy=current_xy)

    def _step_following(self, data, ctx, pose):
        """FOLLOWING: move toward current follow target."""
        _, _, yaw = pose
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        to_waypoint = self.current_waypoint - current_xy
        return self._move(to_waypoint, yaw, ctx, current_xy=current_xy)
    
    def _step_listening(self, data, ctx, pose):
        """LISTENING: low-cost motion toward the nearest front-sector point on the listening ring."""
        del data
        _, _, yaw = pose
        robot_xy = ctx.get("robot_xy")
        if robot_xy is None:
            self._reset_motion_debug()
            self.last_in_listening_front_sector = False
            return np.zeros(3, dtype=np.float32)

        current_xy = np.asarray(pose[:2], dtype=np.float32)
        robot_yaw = ctx.get("robot_yaw", yaw)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        to_robot = robot_xy - current_xy
        dist_to_robot = np.linalg.norm(to_robot)
        listen_radius = ctx.get("listen_radius", DEFAULT_LISTEN_RADIUS)
        sector_half_angle = ctx.get(
            "listening_sector_half_angle",
            LISTENING_FRONT_SECTOR_HALF_ANGLE,
        )
        desired_yaw = np.arctan2(to_robot[1], to_robot[0]) if dist_to_robot > NORM_EPS else robot_yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)
        self.last_in_listening_front_sector = self.is_within_listening_front_sector(
            point_xy=current_xy,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            sector_half_angle=sector_half_angle,
        )

        target_xy = self._compute_listening_sector_target_point(
            current_xy=current_xy,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            listen_radius=listen_radius,
            sector_half_angle=sector_half_angle,
        )
        v_goal = LISTENING_RING_GAIN * (target_xy - current_xy)
        repulsion = ctx.get("repulsion")
        if repulsion is None:
            v_repulsion = np.zeros(2, dtype=np.float32)
        else:
            v_repulsion = np.asarray(repulsion, dtype=np.float32)
        v_hr = self._compute_hr_spacing_force(
            current_xy=current_xy,
            robot_xy=robot_xy,
            distance_min=self.hr_distance_min,
            distance_max=None,
        )

        v_total = v_goal + v_repulsion + v_hr
        speed = np.linalg.norm(v_total)
        if speed > self.max_speed and speed > NORM_EPS:
            v_total = v_total / speed * self.max_speed

        self._set_motion_debug(v_follow=v_goal, v_repulsion=v_repulsion, v_hr=v_hr)
        action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
        return self._apply_wall_constraint_to_action(action, ctx, current_xy=current_xy)

    def _step_listening_with_anchor_target_and_live_repulsion(
        self,
        data,
        ctx,
        pose,
        *,
        anchor_robot_xy,
        anchor_robot_yaw: float,
        live_robot_xy,
    ):
        """LISTENING variant using an anchor pose for target geometry and a live pose for HR repulsion."""
        del data
        _, _, yaw = pose
        if anchor_robot_xy is None:
            self._reset_motion_debug()
            self.last_in_listening_front_sector = False
            return np.zeros(3, dtype=np.float32)

        current_xy = np.asarray(pose[:2], dtype=np.float32)
        anchor_robot_xy = np.asarray(anchor_robot_xy, dtype=np.float32)
        live_robot_xy = (
            anchor_robot_xy
            if live_robot_xy is None
            else np.asarray(live_robot_xy, dtype=np.float32)
        )
        listen_radius = ctx.get("listen_radius", DEFAULT_LISTEN_RADIUS)
        sector_half_angle = ctx.get(
            "listening_sector_half_angle",
            LISTENING_FRONT_SECTOR_HALF_ANGLE,
        )
        to_anchor_robot = anchor_robot_xy - current_xy
        dist_to_anchor_robot = np.linalg.norm(to_anchor_robot)
        desired_yaw = (
            np.arctan2(to_anchor_robot[1], to_anchor_robot[0])
            if dist_to_anchor_robot > NORM_EPS
            else anchor_robot_yaw
        )
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)
        self.last_in_listening_front_sector = self.is_within_listening_front_sector(
            point_xy=current_xy,
            robot_xy=anchor_robot_xy,
            robot_yaw=anchor_robot_yaw,
            sector_half_angle=sector_half_angle,
        )

        target_xy = self._compute_listening_sector_target_point(
            current_xy=current_xy,
            robot_xy=anchor_robot_xy,
            robot_yaw=anchor_robot_yaw,
            listen_radius=listen_radius,
            sector_half_angle=sector_half_angle,
        )
        v_goal = LISTENING_RING_GAIN * (target_xy - current_xy)
        repulsion = ctx.get("repulsion")
        if repulsion is None:
            v_repulsion = np.zeros(2, dtype=np.float32)
        else:
            v_repulsion = np.asarray(repulsion, dtype=np.float32)
        v_hr = self._compute_hr_spacing_force(
            current_xy=current_xy,
            robot_xy=live_robot_xy,
            distance_min=self.hr_distance_min,
            distance_max=None,
        )

        v_total = v_goal + v_repulsion + v_hr
        speed = np.linalg.norm(v_total)
        if speed > self.max_speed and speed > NORM_EPS:
            v_total = v_total / speed * self.max_speed

        self._set_motion_debug(v_follow=v_goal, v_repulsion=v_repulsion, v_hr=v_hr)
        action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
        return self._apply_wall_constraint_to_action(action, ctx, current_xy=current_xy)

    def _step_distracted(self, data, ctx, pose):
        """DISTRACTED: make one local deviated move, then stop until recovery/callback."""
        if self.distracted_source == DISTRACTED_SOURCE_LISTENING:
            return self._step_listening_distracted(data, ctx, pose)

        _, _, yaw = pose
        self.distracted_timer += 1

        current_xy = np.asarray(pose[:2], dtype=np.float32)
        # Generate one-shot distracted target
        if self.distracted_target_xy is None:
            self._initialize_distracted_target(current_xy=current_xy, current_yaw=yaw)

        target_xy = np.asarray(self.distracted_target_xy, dtype=np.float32)
        to_target = target_xy - current_xy
        dist_to_target = np.linalg.norm(to_target)
        if dist_to_target < self.waypoint_threshold:
            self.distracted_stop_reached = True

        if self.distracted_stop_reached:
            self._reset_motion_debug()
            action = np.zeros(3, dtype=np.float32)
        else:
            move_speed_limit = DISTRACTED_SPEED_SCALE * self.max_speed
            if dist_to_target > NORM_EPS:
                v_goal = move_speed_limit * (to_target / dist_to_target)
            else:
                v_goal = np.zeros(2, dtype=np.float32)

            repulsion = ctx.get("repulsion")
            if repulsion is None:
                v_repulsion = np.zeros(2, dtype=np.float32)
            else:
                v_repulsion = np.asarray(repulsion, dtype=np.float32)
            v_total = v_goal + v_repulsion
            speed = np.linalg.norm(v_total)
            if speed > move_speed_limit and speed > NORM_EPS:
                v_total = v_total / speed * move_speed_limit
                speed = np.linalg.norm(v_total)

            desired_yaw = (
                np.arctan2(v_total[1], v_total[0])
                if speed > NORM_EPS
                else (self.distracted_target_yaw if self.distracted_target_yaw is not None else yaw)
            )
            yaw_err = self._wrap_to_pi(desired_yaw - yaw)
            self._set_motion_debug(v_follow=v_goal, v_repulsion=v_repulsion, v_hr=None)
            action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)

        # Recover after duration
        if self.distracted_timer >= self.distracted_duration:
            self.transition_to(self.distracted_recovery_mode, reason="distracted_timeout_recover")
            self._log_event(f">>> {self.name} recovered -> {self.mode.upper()}")

        return self._apply_wall_constraint_to_action(action, ctx, current_xy=current_xy)

    def _initialize_listening_distracted_target(
        self,
        current_xy,
        current_yaw: float,
        robot_xy,
        robot_yaw: float,
        sector_half_angle: float,
    ):
        """Sample one gaze-away yaw target while staying in place."""
        current_xy = np.asarray(current_xy, dtype=np.float32)
        del current_yaw
        del robot_yaw
        del sector_half_angle
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        deviation_deg = np.random.uniform(45.0, 90.0)
        deviation_sign = -1.0 if np.random.rand() < 0.5 else 1.0
        robot_facing_yaw = np.arctan2(robot_xy[1] - current_xy[1], robot_xy[0] - current_xy[0])
        target_yaw = self._wrap_to_pi(robot_facing_yaw + deviation_sign * np.deg2rad(deviation_deg))
        self._set_distracted_target_state(
            target_yaw=target_yaw,
            target_xy=current_xy,
            reset_listening_move_elapsed_steps=True,
        )

    def _step_listening_distracted(self, data, ctx, pose):
        """DISTRACTED-from-listening: turn gaze away from robot and then hold still."""
        del data
        _, _, yaw = pose
        self.distracted_timer += 1
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        robot_xy = np.asarray(ctx.get("robot_xy", current_xy), dtype=np.float32)
        robot_yaw = ctx.get("robot_yaw", yaw)
        sector_half_angle = ctx.get(
            "listening_sector_half_angle",
            LISTENING_FRONT_SECTOR_HALF_ANGLE,
        )
        self.last_in_listening_front_sector = self.is_within_listening_front_sector(
            point_xy=current_xy,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            sector_half_angle=sector_half_angle,
        )

        if self.distracted_target_yaw is None:
            self._initialize_listening_distracted_target(
                current_xy=current_xy,
                current_yaw=yaw,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                sector_half_angle=sector_half_angle,
            )

        self.last_v_repulsion.fill(0.0)
        self.last_v_hr.fill(0.0)
        desired_yaw = self.distracted_target_yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        if abs(yaw_err) >= np.deg2rad(HUMAN_ROTATION_STOP_DEG):
            self.last_v_follow.fill(0.0)
            return self._apply_wall_constraint_to_action(
                self._compose_action(np.zeros(2, dtype=np.float32), HUMAN_YAW_RATE_GAIN * yaw_err),
                ctx,
                current_xy=current_xy,
            )

        self.last_v_follow.fill(0.0)
        return np.zeros(3, dtype=np.float32)

    def _step_overwhelmed(self, data, ctx, pose):
        """OVERWHELMED: first back off, then keep leaving for fixed duration."""
        del data
        _, _, yaw = pose
        pos_xy = np.asarray(pose[:2], dtype=np.float32)
        self._reset_motion_debug()

        if self.overwhelmed_stage is None:
            self.transition_to(self.overwhelmed_recovery_mode, reason="overwhelmed_invalid_stage_recover")
            return np.zeros(3, dtype=np.float32)

        leave_dir = np.asarray(self.overwhelmed_leave_dir, dtype=np.float32)
        leave_norm = np.linalg.norm(leave_dir)
        if leave_norm < NORM_EPS:
            robot_xy = self.overwhelmed_robot_ref_xy
            if robot_xy is None:
                fallback_xy = np.array(
                    [pos_xy[0] - OVERWHELMED_FALLBACK_X_OFFSET, pos_xy[1]],
                    dtype=np.float32,
                )
                robot_xy = np.asarray(ctx.get("robot_xy", fallback_xy), dtype=np.float32)
            else:
                robot_xy = np.asarray(robot_xy, dtype=np.float32)
            diff = pos_xy - robot_xy
            diff_norm = np.linalg.norm(diff)
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
            dist_to_target = np.linalg.norm(to_target)

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
            action = self._compose_action(v_xy, HUMAN_YAW_RATE_GAIN * yaw_err)
            return self._apply_wall_constraint_to_action(action, ctx, current_xy=pos_xy)

        # Leave stage: keep moving away for a fixed duration.
        leave_speed = min(self.overwhelmed_leave_speed, self.max_speed)
        v_xy = leave_speed * leave_dir
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)
        self.overwhelmed_leave_timer += 1

        if self.overwhelmed_leave_timer >= self.overwhelmed_leave_duration:
            self.transition_to(self.overwhelmed_recovery_mode, reason="overwhelmed_timeout_recover")
            self._log_event(
                f">>> {self.name} recovered from OVERWHELMED -> {self.mode.upper()}"
            )

        action = self._compose_action(v_xy, HUMAN_YAW_RATE_GAIN * yaw_err)
        return self._apply_wall_constraint_to_action(action, ctx, current_xy=pos_xy)

    def _step_impatient(self, data, ctx, pose):
        """IMPATIENT: fast following toward front slot, with timeout recovery."""
        self.impatient_timer += 1
        _, _, yaw = pose
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        if self.impatient_recovery_mode == HumanMode.LISTENING:
            robot_xy = np.asarray(ctx.get("robot_xy", current_xy), dtype=np.float32)
            to_robot = robot_xy - current_xy
            base_yaw = (
                np.arctan2(to_robot[1], to_robot[0])
                if np.linalg.norm(to_robot) > NORM_EPS
                else yaw
            )
            deviation = self.listening_impatient_yaw_deviation
            if deviation <= 0.0:
                deviation = np.deg2rad(
                    0.5
                    * (
                        LISTENING_IMPATIENT_YAW_DEVIATION_MIN_DEG
                        + LISTENING_IMPATIENT_YAW_DEVIATION_MAX_DEG
                    )
                )
                self.listening_impatient_yaw_deviation = deviation
            desired_yaw = self._wrap_to_pi(base_yaw + self.listening_impatient_turn_sign * deviation)
            yaw_err = self._wrap_to_pi(desired_yaw - yaw)
            if abs(yaw_err) <= np.deg2rad(LISTENING_IMPATIENT_TARGET_REACHED_DEG):
                self.listening_impatient_turn_sign *= -1.0
                desired_yaw = self._wrap_to_pi(base_yaw + self.listening_impatient_turn_sign * deviation)
                yaw_err = self._wrap_to_pi(desired_yaw - yaw)

            perp = np.array([-np.sin(base_yaw), np.cos(base_yaw)], dtype=np.float32)
            v_goal = self.listening_impatient_turn_sign * LISTENING_IMPATIENT_SWAY_SPEED_METERS_PER_SEC * perp
            repulsion = ctx.get("repulsion")
            if repulsion is None:
                v_repulsion = np.zeros(2, dtype=np.float32)
            else:
                v_repulsion = np.asarray(repulsion, dtype=np.float32)
            v_total = v_goal + v_repulsion
            speed = np.linalg.norm(v_total)
            if speed > self.max_speed and speed > NORM_EPS:
                v_total = v_total / speed * self.max_speed
            self._set_motion_debug(v_follow=v_goal, v_repulsion=v_repulsion, v_hr=None)
            action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
            action = self._apply_wall_constraint_to_action(action, ctx, current_xy=current_xy)
        else:
            self._assign_target_from_context(ctx, current_xy=current_xy)
            action = self._step_following(data, ctx, pose)

        if self.impatient_timer >= self.impatient_duration:
            self.set_mode(self.impatient_recovery_mode)
            self._log_event(f">>> {self.name} recovered from IMPATIENT -> {self.mode.upper()}")

        return action

    def _move(self, to_target_xy, yaw, ctx, current_xy):
        """Shared low-level controller that fuses follow, repulsion and HR spacing forces."""
        robot_xy = ctx.get("robot_xy", None)

        repulsion = ctx.get("repulsion")
        if repulsion is None:
            v_repulsion = np.zeros(2, dtype=np.float32)
        else:
            v_repulsion = np.asarray(repulsion, dtype=np.float32)
        v_hr = np.zeros(2, dtype=np.float32)

        if robot_xy is not None:
            v_hr = self._compute_hr_spacing_force(
                current_xy=current_xy,
                robot_xy=np.asarray(robot_xy, dtype=np.float32),
                distance_min=self.hr_distance_min,
                distance_max=self.hr_distance_max,
            )

        to_target_xy = np.asarray(to_target_xy, dtype=np.float32)
        dist = np.linalg.norm(to_target_xy)
        if dist > NORM_EPS:
            v_follow = self.max_speed * (to_target_xy / dist)
        else:
            v_follow = np.zeros(2, dtype=np.float32)

        self._set_motion_debug(v_follow=v_follow, v_repulsion=v_repulsion, v_hr=v_hr)

        # Final translational command is a blend of:
        # (1) target following, (2) human-human repulsion, (3) human-robot spacing force.
        v_total = v_follow + v_repulsion + v_hr
        speed = np.linalg.norm(v_total)

        if speed > self.max_speed:
            v_total = v_total / speed * self.max_speed

        desired_yaw = np.arctan2(v_total[1], v_total[0]) if speed > NORM_EPS else yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
        return self._apply_wall_constraint_to_action(action, ctx, current_xy=current_xy)
    
    # -------------------------
    # Helpers
    # -------------------------

    @classmethod
    def sample_walkable_point(cls, margin: float = HUMAN_WALL_FOOTPRINT_RADIUS, rng=None):
        """Compatibility wrapper returning a sample from the default spawn region."""
        return DEFAULT_MUSEUM_LAYOUT.sample_spawn_point(margin=margin, rng=rng)

    @classmethod
    def sample_room_a_point(cls, margin: float = HUMAN_WALL_FOOTPRINT_RADIUS, rng=None):
        """Compatibility wrapper returning a spawn sample from the default map layout."""
        return DEFAULT_MUSEUM_LAYOUT.sample_spawn_point(margin=margin, rng=rng)

    @staticmethod
    def _iter_raycast_origin_heights(body_z: float) -> tuple[float, ...]:
        primary_height = float(body_z)
        secondary_height = min(primary_height, 0.10)
        if abs(primary_height - secondary_height) <= MIN_SPEED_EPS:
            return (primary_height,)
        return (primary_height, secondary_height)

    def _raycast_hit_distance(self, direction_xy):
        """Return the nearest live collision distance along one XY direction, if any."""
        direction_xy = np.asarray(direction_xy, dtype=np.float32)
        desired_speed = np.linalg.norm(direction_xy)
        if desired_speed <= MIN_SPEED_EPS:
            return None
        if self._runtime_model is None or self._runtime_data is None or self.body_id is None:
            return None

        ray_direction = np.zeros(3, dtype=np.float64)
        ray_direction[:2] = direction_xy / desired_speed
        ray_origin = np.array(self._runtime_data.xpos[self.body_id], dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        best_hit_distance = None
        for ray_height in self._iter_raycast_origin_heights(float(ray_origin[2])):
            ray_origin_with_height = ray_origin.copy()
            ray_origin_with_height[2] = ray_height
            geomid[0] = -1
            hit_distance = float(
                mujoco.mj_ray(
                    self._runtime_model,
                    self._runtime_data,
                    ray_origin_with_height,
                    ray_direction,
                    None,
                    1,
                    int(self.body_id),
                    geomid,
                )
            )
            if hit_distance < 0.0:
                continue
            if best_hit_distance is None or hit_distance < best_hit_distance:
                best_hit_distance = hit_distance
        return best_hit_distance

    @staticmethod
    def _clip_listening_sector_relative_angle(relative_angle: float, sector_half_angle: float):
        """Clamp one relative polar angle to the front listening sector."""
        half_angle = max(0.0, sector_half_angle - LISTENING_SECTOR_PROJECTION_EPS)
        return float(np.clip(relative_angle, -half_angle, half_angle))

    def _compute_listening_sector_relative_angle(self, point_xy, robot_xy, robot_yaw: float) -> float:
        """Return point angle relative to robot heading in [-pi, pi)."""
        point_xy = np.asarray(point_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        rel_xy = point_xy - robot_xy
        if np.dot(rel_xy, rel_xy) <= (NORM_EPS * NORM_EPS):
            return 0.0
        absolute_angle = np.arctan2(rel_xy[1], rel_xy[0])
        return float(self._wrap_to_pi(absolute_angle - robot_yaw))

    def is_within_listening_front_sector(
        self,
        point_xy,
        robot_xy,
        robot_yaw: float,
        sector_half_angle: float = LISTENING_FRONT_SECTOR_HALF_ANGLE,
    ) -> bool:
        """Return whether a point lies inside the robot-front listening sector."""
        rel_angle = self._compute_listening_sector_relative_angle(
            point_xy=point_xy,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
        )
        return bool(abs(rel_angle) <= sector_half_angle + LISTENING_SECTOR_PROJECTION_EPS)

    def _clamp_absolute_angle_to_listening_sector(
        self,
        absolute_angle: float,
        robot_yaw: float,
        sector_half_angle: float,
    ) -> float:
        """Clamp one world-frame angle so it stays inside the robot-front listening sector."""
        relative_angle = self._wrap_to_pi(absolute_angle - robot_yaw)
        clipped_relative_angle = self._clip_listening_sector_relative_angle(
            relative_angle=relative_angle,
            sector_half_angle=sector_half_angle,
        )
        return float(self._wrap_to_pi(robot_yaw + clipped_relative_angle))

    def _compute_listening_sector_target_point(
        self,
        current_xy,
        robot_xy,
        robot_yaw: float,
        listen_radius: float,
        sector_half_angle: float,
    ):
        """Return the nearest low-cost front-sector target point on the listening ring."""
        current_xy = np.asarray(current_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        rel_xy = current_xy - robot_xy
        if np.dot(rel_xy, rel_xy) <= (NORM_EPS * NORM_EPS):
            absolute_angle = robot_yaw
        else:
            absolute_angle = np.arctan2(rel_xy[1], rel_xy[0])
        clamped_angle = self._clamp_absolute_angle_to_listening_sector(
            absolute_angle=absolute_angle,
            robot_yaw=robot_yaw,
            sector_half_angle=sector_half_angle,
        )
        target_xy = np.empty(2, dtype=np.float32)
        target_xy[0] = robot_xy[0] + listen_radius * np.cos(clamped_angle)
        target_xy[1] = robot_xy[1] + listen_radius * np.sin(clamped_angle)
        return target_xy

    def _compute_hr_spacing_force(
        self,
        current_xy,
        robot_xy,
        *,
        distance_min: Optional[float],
        distance_max: Optional[float],
    ):
        """Compute continuous human-robot spacing force from a preferred distance band."""
        if robot_xy is None:
            return np.zeros(2, dtype=np.float32)

        current_xy = np.asarray(current_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        diff = current_xy - robot_xy
        dist_hr = np.linalg.norm(diff)
        if dist_hr <= NORM_EPS:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = diff / dist_hr

        if distance_min is not None and dist_hr < distance_min:
            if dist_hr <= HR_REPULSION_GAIN_NEAR_DISTANCE:
                repulsion_gain = HR_REPULSION_GAIN * HR_REPULSION_GAIN_NEAR_MULTIPLIER
            elif dist_hr <= HR_REPULSION_GAIN_MID_DISTANCE:
                repulsion_gain = HR_REPULSION_GAIN * HR_REPULSION_GAIN_MID_MULTIPLIER
            else:
                repulsion_gain = HR_REPULSION_GAIN
            magnitude = repulsion_gain * (distance_min - dist_hr)
            return np.asarray(magnitude * direction, dtype=np.float32)

        if distance_max is not None and dist_hr > distance_max:
            magnitude = -HR_ATTRACTION_GAIN * (dist_hr - distance_max)
            return np.asarray(magnitude * direction, dtype=np.float32)

        return np.zeros(2, dtype=np.float32)

    def _constrain_velocity_with_walkable(
        self,
        current_xy=None,
        v_xy=None,
        dt: float = DEFAULT_SIM_TIMESTEP_SECONDS,
        margin: float = HUMAN_WALL_FOOTPRINT_RADIUS,
        *,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ):
        """Adjust XY velocity using live MuJoCo ray-cast clearance only."""
        del dt, margin
        if current_xy is None:
            if x is None or y is None:
                raise ValueError("current_xy or both x/y must be provided.")
        if v_xy is None:
            raise ValueError("v_xy must be provided.")
        v_xy = np.asarray(v_xy, dtype=np.float32)
        speed = np.linalg.norm(v_xy)
        if speed > self.max_speed and speed > MIN_SPEED_EPS:
            v_xy = v_xy / speed * self.max_speed
            speed = np.linalg.norm(v_xy)

        if speed <= MIN_SPEED_EPS:
            return v_xy

        hit_distance = self._raycast_hit_distance(v_xy)
        if hit_distance is None or hit_distance >= RAYCAST_SLOWDOWN_DISTANCE_METERS:
            return v_xy

        clearance = max(0.0, hit_distance - RAYCAST_CLEARANCE_EPS)
        if clearance <= MIN_SPEED_EPS:
            return np.zeros(2, dtype=np.float32)

        speed_scale = min(1.0, clearance / RAYCAST_SLOWDOWN_DISTANCE_METERS)
        return np.asarray(v_xy * speed_scale, dtype=np.float32)

    def _apply_wall_constraint_to_action(self, action, ctx, current_xy):
        """Apply live ray-cast wall/obstacle constraints to translational action components."""
        constrained_action = np.array(action, dtype=np.float32)
        if constrained_action.shape[0] < 2:
            return constrained_action

        dt = ctx.get("dt", 0.002)
        constrained_v = self._constrain_velocity_with_walkable(
            current_xy=current_xy,
            v_xy=constrained_action[0:2],
            dt=dt,
            margin=HUMAN_WALL_FOOTPRINT_RADIUS,
        )
        constrained_action[0:2] = constrained_v
        return constrained_action

    def _get_pose(self, data):
        """Read current (x, y, yaw) of this human from MuJoCo state."""
        x = float(data.qpos[self.qpos_idx])
        y = float(data.qpos[self.qpos_idx + 1])
        yaw = float(data.qpos[self.qpos_idx + 2])
        return x, y, yaw

    def _random_waypoint(self):
        """Generate random waypoint inside the configured spawn region."""
        return self.map_layout.sample_spawn_point(HUMAN_WALL_FOOTPRINT_RADIUS, rng=np.random)
    
    def _wrap_to_pi(self, ang):
        """Normalize angle to [-pi, pi)."""
        return (ang + np.pi) % (2 * np.pi) - np.pi
    
