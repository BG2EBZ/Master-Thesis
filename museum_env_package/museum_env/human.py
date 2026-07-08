import logging
from typing import Optional

import numpy as np

from .env_constants import (
    FOLLOW_RADIUS_DEFAULT,
    HUMAN_WALL_FOOTPRINT_RADIUS,
)
from .map_layouts import DEFAULT_MUSEUM_LAYOUT, MapLayout
from .spatial_utils import raycast_hit_distance, wrap_to_pi

logger = logging.getLogger(__name__)

DEFAULT_WAYPOINT_THRESHOLD = 0.2
DEFAULT_IMPATIENT_FRONT_OFFSET = 1.2
HUMAN_YAW_RATE_GAIN = 1.7
HUMAN_ROTATION_STOP_DEG = 3.0
LISTENING_RING_GAIN = 4.0
LISTENING_SECTOR_PROJECTION_EPS = 1e-2
DISTRACTED_SPEED_SCALE = 0.5
DISTRACTED_YAW_DEVIATION_MIN_DEG = 45.0
DISTRACTED_YAW_DEVIATION_MAX_DEG = 90.0
DISTRACTED_TARGET_DISTANCE_MIN = 0.5
DISTRACTED_TARGET_DISTANCE_MAX = 1.5
DEFAULT_SIM_TIMESTEP_SECONDS = 0.05
DISTRACTED_DURATION_SECONDS_DEFAULT = 10.0
DISTRACTED_STOP_DURATION_SECONDS = 5.0
DISTRACTED_EXHIBIT_LOOK_RADIUS = 4.0
DISTRACTED_HUMAN_LOOK_RADIUS = 2.0
DISTRACTED_FALLBACK_DISTANCE = 1.0
DISTRACTED_CONVERSATION_STOP_DISTANCE = 0.8
ND_DISTRACTED_STOP_AND_GO_STOP_SECONDS = 0.6
ND_DISTRACTED_STOP_AND_GO_MOVE_SECONDS = 0.4
LISTENING_IMPATIENT_GLANCE_SECONDS_DEFAULT = 2.0
LISTENING_IMPATIENT_YAW_DEVIATION_MIN_DEG = 45.0
LISTENING_IMPATIENT_YAW_DEVIATION_MAX_DEG = 90.0
LISTENING_IMPATIENT_TARGET_REACHED_DEG = 5.0
CURIOUS_STOP_DURATION_SECONDS_DEFAULT = 5.0
CURIOSITY_RETRIGGER_COOLDOWN_SECONDS_DEFAULT = 10.0
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
NORM_EPS = 1e-3
MIN_SPEED_EPS = 1e-3
RAYCAST_CLEARANCE_EPS = 1e-3
RAYCAST_SLOWDOWN_DISTANCE_METERS = 0.3
WALL_RAYCAST_GUIDE_SKIP_METERS = 0.1
WALL_RAYCAST_SPEED_SKIP_MPS = 0.05
WALL_RAYCAST_CACHE_KEY_DECIMALS = 4
WALL_REPULSION_DISTANCE_METERS = 0.45
WALL_REPULSION_GAIN = 20.0
WALL_DETOUR_ANGLES_DEG = (60.0, -60.0, 90.0, -90.0, 120.0, -120.0)
WALL_DETOUR_ROTATIONS = tuple(
    (float(np.cos(np.deg2rad(angle_deg))), float(np.sin(np.deg2rad(angle_deg))))
    for angle_deg in WALL_DETOUR_ANGLES_DEG
)
DISTRACTED_BEHAVIOR_FOCUS = "focus"
DISTRACTED_BEHAVIOR_CONVERSATION = "conversation"
DISTRACTED_BEHAVIOR_STOP_AND_GO_FOLLOWING = "stop_and_go_following"
DISTRACTED_SOURCE_FOLLOWING = "following"
DISTRACTED_SOURCE_LISTENING = "listening"
DEFAULT_FOLLOW_RADIUS = FOLLOW_RADIUS_DEFAULT


class HumanMode:
    WANDERING = "wandering"
    FOLLOWING = "following"
    LISTENING = "listening"
    CURIOSITY = "curiosity"
    DISTRACTED = "distracted"
    OVERWHELMED = "overwhelmed"
    IMPATIENT = "impatient"


class HumanProfile:
    NORMAL = "normal"
    NEURODIVERGENT = "neurodivergent"


class Human:
    def __init__(
        self,
        name,
        body_name,
        qpos_idx,
        max_speed,
        waypoint_threshold=DEFAULT_WAYPOINT_THRESHOLD,
        map_layout: Optional[MapLayout] = None,
    ):
        self.name = name
        self.body_name = body_name
        self.qpos_idx = qpos_idx
        self.max_speed = float(max_speed)
        self.base_max_speed = float(max_speed)
        self.waypoint_threshold = waypoint_threshold
        self.map_layout = DEFAULT_MUSEUM_LAYOUT if map_layout is None else map_layout
        self.mode = None

        self.enable_event_logs = True
        self.body_id = None
        self._runtime_model = None
        self._runtime_data = None
        self.rng = np.random
        self._wall_raycast_cache_step_id = -1
        self._wall_raycast_cache: dict[tuple[float, float], Optional[float]] = {}

        self.current_waypoint = self._random_waypoint()
        self.hr_distance_min = float(HR_DISTANCE_MIN)
        self.hr_distance_max = float(HR_DISTANCE_MAX)

        self.distracted_timer = 0
        self.distracted_elapsed_steps = 0
        self.max_distracted_duration_seconds = float(DISTRACTED_DURATION_SECONDS_DEFAULT)
        self.distracted_duration = (round(self.max_distracted_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS))
        self.distracted_stop_duration_seconds = float(DISTRACTED_STOP_DURATION_SECONDS)
        self.distracted_stop_duration = round(self.distracted_stop_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS)

        self.distracted_target_xy = None
        self.distracted_stop_reached = False
        self.distracted_target_yaw = None
        self.distracted_behavior_kind = None
        self.distracted_partner_index = None
        self.distracted_recovery_mode = HumanMode.FOLLOWING
        self.distracted_source = None
        self.speaking_active = False
        self.nd_distracted_stop_and_go_stop_seconds = float(ND_DISTRACTED_STOP_AND_GO_STOP_SECONDS)
        self.nd_distracted_stop_and_go_stop_steps = round(
            self.nd_distracted_stop_and_go_stop_seconds / DEFAULT_SIM_TIMESTEP_SECONDS
        )
        self.nd_distracted_stop_and_go_move_seconds = float(ND_DISTRACTED_STOP_AND_GO_MOVE_SECONDS)
        self.nd_distracted_stop_and_go_move_steps = round(
            self.nd_distracted_stop_and_go_move_seconds / DEFAULT_SIM_TIMESTEP_SECONDS
        )

        self.impatient_duration = 800
        self.impatient_timer = 0
        self.impatient_speed_multiplier = 1.3
        self.impatient_front_offset = DEFAULT_IMPATIENT_FRONT_OFFSET
        self.listening_impatient_glance_steps = round(LISTENING_IMPATIENT_GLANCE_SECONDS_DEFAULT / DEFAULT_SIM_TIMESTEP_SECONDS)
        self.impatient_original_max_speed = None
        self.impatient_recovery_mode = HumanMode.FOLLOWING
        self.listening_impatient_yaw_deviation = 0.0
        self.listening_impatient_turn_sign = 1.0

        self.profile = None
        self.following_steps = 0
        self.listening_steps = 0
        self.curiosity_duration_seconds = float(CURIOUS_STOP_DURATION_SECONDS_DEFAULT)
        self.curiosity_duration = round(self.curiosity_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS)
        self.curiosity_timer = 0
        self.curiosity_recovery_mode = HumanMode.FOLLOWING
        self.curiosity_retrigger_cooldown_seconds = float(CURIOSITY_RETRIGGER_COOLDOWN_SECONDS_DEFAULT)
        self.curiosity_retrigger_cooldown_steps = round(
            self.curiosity_retrigger_cooldown_seconds / DEFAULT_SIM_TIMESTEP_SECONDS
        )
        self.curiosity_retrigger_cooldown_steps_remaining = 0

        self.overwhelmed_stage = None
        self.overwhelmed_backoff_dist = 0.3
        self.overwhelmed_leave_speed = 1.0
        self.max_overwhelmed_leave_duration_seconds = 2.0
        self.overwhelmed_leave_duration = round(self.max_overwhelmed_leave_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS)
        self.overwhelmed_leave_timer = 0
        self.max_overwhelmed_pause_duration_seconds = 10.0
        self.overwhelmed_pause_duration = round(self.max_overwhelmed_pause_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS)
        self.overwhelmed_pause_timer = 0
        self.overwhelmed_leave_dir = np.zeros(2, dtype=np.float32)
        self.overwhelmed_backoff_start_xy = None
        self.overwhelmed_recovery_mode = HumanMode.FOLLOWING

        self.set_profile(HumanProfile.NORMAL)
        self.set_mode(HumanMode.WANDERING)

    def set_mode(self, mode: str) -> None:
        prev_mode = self.mode
        if prev_mode == mode:
            return
        if prev_mode == HumanMode.IMPATIENT:
            self._stop_impatient()
        if prev_mode == HumanMode.DISTRACTED:
            self.distracted_timer = 0
            self.distracted_elapsed_steps = 0
            self._clear_distracted_navigation_state()
            self.distracted_source = None
            self.distracted_recovery_mode = HumanMode.FOLLOWING
        if prev_mode == HumanMode.CURIOSITY:
            self.reset_curiosity_state()
        if prev_mode == HumanMode.FOLLOWING:
            self.reset_following_duration()
        if prev_mode == HumanMode.OVERWHELMED:
            self.reset_overwhelmed_state()
        self.mode = mode
        if prev_mode == HumanMode.CURIOSITY and mode != HumanMode.CURIOSITY:
            self.curiosity_retrigger_cooldown_steps_remaining = int(self.curiosity_retrigger_cooldown_steps)
        if mode == HumanMode.DISTRACTED:
            self.distracted_timer = 0
            self.distracted_elapsed_steps = 0
            self._clear_distracted_navigation_state()
        if mode == HumanMode.CURIOSITY:
            self.curiosity_timer = 0

    def reset_overwhelmed_state(self):
        self.overwhelmed_stage = None
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_pause_timer = 0
        self.overwhelmed_leave_dir = np.zeros(2, dtype=np.float32)
        self.overwhelmed_backoff_start_xy = None

    def reset_curiosity_state(self):
        self.curiosity_timer = 0
        self.curiosity_recovery_mode = HumanMode.FOLLOWING
        self.curiosity_retrigger_cooldown_steps_remaining = 0

    def reset_episode_state(self):
        self.mode = None
        self.set_mode(HumanMode.WANDERING)
        self.current_waypoint = self._random_waypoint()

        self.distracted_timer = 0
        self.distracted_elapsed_steps = 0
        self._clear_distracted_navigation_state()
        self.distracted_source = None
        self.distracted_recovery_mode = HumanMode.FOLLOWING

        self.impatient_timer = 0
        self.impatient_original_max_speed = None
        self.impatient_recovery_mode = HumanMode.FOLLOWING
        self.listening_impatient_yaw_deviation = 0.0
        self.listening_impatient_turn_sign = 1.0

        self.max_speed = float(self.base_max_speed)
        self.reset_following_duration()
        self.listening_steps = 0
        self.reset_curiosity_state()
        self.reset_overwhelmed_state()
        self._reset_wall_query_state()

    def set_profile(self, profile: str):
        self.profile = profile
        if profile == HumanProfile.NEURODIVERGENT:
            self.hr_distance_min = float(HR_DISTANCE_MIN_ND_DEFAULT)
            self.hr_distance_max = float(HR_DISTANCE_MAX)
        else:
            self.hr_distance_min = float(HR_DISTANCE_MIN)
            self.hr_distance_max = float(HR_DISTANCE_MAX_NORMAL_DEFAULT)

    def apply_runtime_config(
        self,
        *,
        dt: float,
        max_distracted_duration_seconds: float,
        impatient_duration_seconds: float,
        impatient_speed_multiplier: float,
        impatient_front_offset: float,
        listening_impatient_glance_seconds: float,
        rng=None,
    ) -> None:
        dt = float(dt)
        if rng is not None:
            self.rng = rng
        self.max_distracted_duration_seconds = float(max_distracted_duration_seconds)
        self.distracted_duration = round(self.max_distracted_duration_seconds / dt)
        self.distracted_stop_duration = round(self.distracted_stop_duration_seconds / dt)
        self.nd_distracted_stop_and_go_stop_steps = round(
            self.nd_distracted_stop_and_go_stop_seconds / dt
        )
        self.nd_distracted_stop_and_go_move_steps = round(
            self.nd_distracted_stop_and_go_move_seconds / dt
        )
        self.impatient_duration = round(float(impatient_duration_seconds) / dt)
        self.impatient_speed_multiplier = float(impatient_speed_multiplier)
        self.impatient_front_offset = float(impatient_front_offset)
        self.listening_impatient_glance_steps = float((listening_impatient_glance_seconds) / dt)
        self.curiosity_duration = round(self.curiosity_duration_seconds / dt)
        self.curiosity_retrigger_cooldown_steps = round(self.curiosity_retrigger_cooldown_seconds / dt)
        self.curiosity_retrigger_cooldown_steps_remaining = min(
            int(self.curiosity_retrigger_cooldown_steps_remaining),
            int(self.curiosity_retrigger_cooldown_steps),
        )

    def reset_following_duration(self):
        self.following_steps = 0

    def update_following_duration(self, eligible_following: bool):
        if eligible_following:
            self.following_steps += 1
        else:
            self.reset_following_duration()

    def start_overwhelmed(self, robot_xy, current_xy, recovery_mode: str = HumanMode.FOLLOWING):
        current_xy = np.array(current_xy, dtype=np.float32)
        robot_xy = np.array(robot_xy, dtype=np.float32)
        diff = current_xy - robot_xy
        dist = float(np.linalg.norm(diff))
        leave_dir = np.array([1.0, 0.0], dtype=np.float32) if dist < NORM_EPS else diff / dist

        self.set_mode(HumanMode.OVERWHELMED)
        self.overwhelmed_recovery_mode = recovery_mode
        self.overwhelmed_stage = "backoff"
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_pause_timer = 0
        self.overwhelmed_leave_dir = leave_dir
        self.overwhelmed_backoff_start_xy = current_xy

    def start_impatient(self, recovery_mode: str = HumanMode.FOLLOWING):
        self.impatient_original_max_speed = float(self.max_speed)
        self.max_speed = float(self.impatient_original_max_speed * self.impatient_speed_multiplier)
        self.impatient_timer = 0
        self.impatient_recovery_mode = recovery_mode
        if recovery_mode == HumanMode.LISTENING:
            self.listening_impatient_yaw_deviation = np.deg2rad(
                float(
                    self.rng.uniform(
                        LISTENING_IMPATIENT_YAW_DEVIATION_MIN_DEG,
                        LISTENING_IMPATIENT_YAW_DEVIATION_MAX_DEG,
                    )
                )
            )
            self.listening_impatient_turn_sign = 1.0 if self.rng.random() >= 0.5 else -1.0
        else:
            self.listening_impatient_yaw_deviation = 0.0
            self.listening_impatient_turn_sign = 1.0
        self.set_mode(HumanMode.IMPATIENT)

    def start_curiosity(self, recovery_mode: str = HumanMode.FOLLOWING):
        self.curiosity_timer = 0
        self.curiosity_recovery_mode = recovery_mode
        self.set_mode(HumanMode.CURIOSITY)

    def _stop_impatient(self):
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
        self.distracted_target_xy = None
        self.distracted_stop_reached = False
        self.distracted_target_yaw = None
        self.distracted_behavior_kind = None
        self.distracted_partner_index = None
        self.speaking_active = False

    def _reset_wall_query_state(self) -> None:
        self._wall_raycast_cache_step_id = -1
        self._wall_raycast_cache = {}

    def _begin_wall_query_step(self, step_id: int) -> None:
        step_id = int(step_id)
        if self._wall_raycast_cache_step_id == step_id:
            return
        self._wall_raycast_cache_step_id = step_id
        self._wall_raycast_cache = {}

    def _direction_cache_key(self, direction_xy) -> Optional[tuple[float, float]]:
        direction_xy = np.asarray(direction_xy, dtype=np.float32)
        direction_norm = float(np.linalg.norm(direction_xy))
        if direction_norm <= 1e-6:
            return None
        unit_direction = direction_xy[:2] / direction_norm
        return (
            float(np.round(unit_direction[0], WALL_RAYCAST_CACHE_KEY_DECIMALS)),
            float(np.round(unit_direction[1], WALL_RAYCAST_CACHE_KEY_DECIMALS)),
        )

    def _set_distracted_target_state(
        self,
        target_yaw: float,
        target_xy,
        *,
        behavior_kind: str = DISTRACTED_BEHAVIOR_FOCUS,
        partner_index=None,
    ):
        target_xy = np.asarray(target_xy, dtype=np.float32)
        self.distracted_target_yaw = float(target_yaw)
        self.distracted_target_xy = target_xy
        self.distracted_stop_reached = False
        self.distracted_behavior_kind = str(behavior_kind)
        self.distracted_partner_index = None if partner_index is None else int(partner_index)
        self.speaking_active = bool(behavior_kind == DISTRACTED_BEHAVIOR_CONVERSATION)

    @staticmethod
    def _compose_action(v_xy, yaw_rate):
        action = np.zeros(3, dtype=np.float32)
        action[:2] = v_xy
        action[2] = np.float32(yaw_rate)
        return action

    @staticmethod
    def _limit_speed(v_xy, speed_limit: float):
        v_xy = np.asarray(v_xy, dtype=np.float32)
        speed = float(np.linalg.norm(v_xy))
        if speed <= MIN_SPEED_EPS or speed <= float(speed_limit):
            return v_xy
        return np.asarray(v_xy / speed * float(speed_limit), dtype=np.float32)

    def assign_target_from_context(self, ctx: dict, mode: Optional[str] = None):
        index = ctx["index"]
        n_humans = ctx["n_humans"]
        target_mode = self.mode if mode is None else mode

        if target_mode == HumanMode.FOLLOWING:
            fan_half_angle = ctx["fan_half_angle"]
            radius = float(ctx.get("follow_radius", DEFAULT_FOLLOW_RADIUS))
            base_angle_offset = np.pi
        elif target_mode == HumanMode.IMPATIENT:
            fan_half_angle = float(ctx.get("impatient_fan_half_angle", ctx["fan_half_angle"]))
            radius = ctx["impatient_front_offset"]
            base_angle_offset = 0.0
        else:
            return

        if n_humans > 1:
            relative_angle = (index / (n_humans - 1)) * (2 * fan_half_angle) - fan_half_angle
        else:
            relative_angle = 0.0

        rx, ry, ryaw = ctx["robot_pose"]
        angle = ryaw + base_angle_offset + relative_angle
        self.current_waypoint = np.array(
            [rx + radius * np.cos(angle), ry + radius * np.sin(angle)],
            dtype=np.float32,
        )

    def step(self, model, data, ctx):
        self._runtime_model = model
        self._runtime_data = data
        step_id = ctx.get("step_id")
        if step_id is not None:
            self._begin_wall_query_step(int(step_id))

        pose = self.get_pose(data)
        return human_behaviors.step_behavior(self, ctx, pose)

    def _move(self, to_target_xy, yaw, ctx, current_xy):
        to_target_xy = np.asarray(to_target_xy, dtype=np.float32)
        dist = np.linalg.norm(to_target_xy)
        if dist > NORM_EPS:
            v_follow = self.max_speed * (to_target_xy / dist)
        else:
            v_follow = np.zeros(2, dtype=np.float32)

        v_total = self._compose_move_velocity(
            current_xy=current_xy,
            guide_xy=to_target_xy,
            goal_v_xy=v_follow,
            speed_limit=self.max_speed,
            repulsion_xy=ctx["repulsion"],
            robot_xy=ctx["robot_xy"],
            hr_distance_min=self.hr_distance_min,
            hr_distance_max=self.hr_distance_max,
        )
        speed = np.linalg.norm(v_total)
        desired_yaw = np.arctan2(v_total[1], v_total[0]) if speed > NORM_EPS else yaw
        action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * self._wrap_to_pi(desired_yaw - yaw))
        return action

    def _compose_move_velocity(
        self,
        *,
        current_xy,
        guide_xy,
        goal_v_xy,
        speed_limit,
        repulsion_xy=None,
        robot_xy=None,
        hr_distance_min=None,
        hr_distance_max=None,
    ) -> np.ndarray:
        current_xy = np.asarray(current_xy, dtype=np.float32)
        guide_xy = np.asarray(guide_xy, dtype=np.float32)
        v_total = np.asarray(goal_v_xy, dtype=np.float32).copy()

        if repulsion_xy is not None:
            v_total += np.asarray(repulsion_xy, dtype=np.float32)

        if robot_xy is not None and (hr_distance_min is not None or hr_distance_max is not None):
            v_total += self._compute_hr_spacing_force(
                current_xy=current_xy,
                robot_xy=robot_xy,
                distance_min=hr_distance_min,
                distance_max=hr_distance_max,
            )

        v_total += self._compute_wall_spacing_force(guide_xy)
        v_total = self._limit_speed(v_total, speed_limit)
        return self._adjust_target_velocity_for_walls(
            guide_xy=guide_xy,
            desired_v_xy=v_total,
        )

    def _adjust_target_velocity_for_walls(self, *, guide_xy, desired_v_xy):
        desired_v_xy = np.asarray(desired_v_xy, dtype=np.float32)
        desired_speed = float(np.linalg.norm(desired_v_xy))
        if desired_speed <= MIN_SPEED_EPS:
            return np.zeros(2, dtype=np.float32)
        if desired_speed <= WALL_RAYCAST_SPEED_SKIP_MPS:
            return desired_v_xy

        guide_xy = np.asarray(guide_xy, dtype=np.float32)
        guide_norm = float(np.linalg.norm(guide_xy))
        if guide_norm <= NORM_EPS:
            return self._constrain_velocity_with_walkable(desired_v_xy)
        if guide_norm <= WALL_RAYCAST_GUIDE_SKIP_METERS:
            return desired_v_xy
        # Detect if there's a wall in the way of the desired velocity.
        hit_distance = self._raycast_hit_distance(desired_v_xy)
        if hit_distance is None or hit_distance >= RAYCAST_SLOWDOWN_DISTANCE_METERS:
            return desired_v_xy

        guide_dir = guide_xy / guide_norm
        best_v_xy = None
        best_progress = 0.0
        best_speed = 0.0

        for cos_ang, sin_ang in WALL_DETOUR_ROTATIONS:
            candidate_v_xy = np.array(
                [
                    cos_ang * desired_v_xy[0] - sin_ang * desired_v_xy[1],
                    sin_ang * desired_v_xy[0] + cos_ang * desired_v_xy[1],
                ],
                dtype=np.float32,
            )
            candidate_v_xy = self._constrain_velocity_with_walkable(candidate_v_xy)
            candidate_v_xy = np.asarray(candidate_v_xy, dtype=np.float32)
            candidate_speed = float(np.linalg.norm(candidate_v_xy))
            if candidate_speed <= MIN_SPEED_EPS:
                continue

            progress = float(np.dot(candidate_v_xy, guide_dir))
            if progress <= MIN_SPEED_EPS:
                continue

            is_better_progress = progress > best_progress + MIN_SPEED_EPS
            is_tie_better_speed = (
                abs(progress - best_progress) <= MIN_SPEED_EPS
                and candidate_speed > best_speed + MIN_SPEED_EPS
            )
            if best_v_xy is None or is_better_progress or is_tie_better_speed:
                best_v_xy = candidate_v_xy
                best_progress = progress
                best_speed = candidate_speed

        if best_v_xy is None:
            return np.zeros(2, dtype=np.float32)
        return np.asarray(best_v_xy, dtype=np.float32)

    def _raycast_hit_distance(self, direction_xy):
        cache_key = self._direction_cache_key(direction_xy)
        if cache_key is None:
            return None
        if cache_key in self._wall_raycast_cache:
            return self._wall_raycast_cache[cache_key]

        hit_distance = raycast_hit_distance(
            self._runtime_model,
            self._runtime_data,
            self.body_id,
            direction_xy,
        )
        self._wall_raycast_cache[cache_key] = hit_distance
        return hit_distance

    def is_within_listening_front_sector(self, point_xy, robot_xy, robot_yaw: float, sector_half_angle: float) -> bool:
        point_xy = np.asarray(point_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        rel_xy = point_xy - robot_xy
        if np.dot(rel_xy, rel_xy) <= (NORM_EPS * NORM_EPS):
            rel_angle = 0.0
        else:
            rel_angle = self._wrap_to_pi(np.arctan2(rel_xy[1], rel_xy[0]) - robot_yaw)
        return bool(abs(rel_angle) <= sector_half_angle + LISTENING_SECTOR_PROJECTION_EPS)

    def _compute_hr_spacing_force(
        self,
        current_xy,
        robot_xy,
        *,
        distance_min: Optional[float],
        distance_max: Optional[float],
    ):
        current_xy = np.asarray(current_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        diff = current_xy - robot_xy
        dist_hr = np.linalg.norm(diff)
        direction = np.array([1.0, 0.0], dtype=np.float32) if dist_hr <= NORM_EPS else diff / dist_hr

        # Inside the preferred band, no force is applied. Too close pushes away;
        # too far can optionally pull the human back toward the robot.
        if distance_min is not None and dist_hr < distance_min:
            if dist_hr <= HR_REPULSION_GAIN_NEAR_DISTANCE:
                repulsion_gain = HR_REPULSION_GAIN * HR_REPULSION_GAIN_NEAR_MULTIPLIER
            elif dist_hr <= HR_REPULSION_GAIN_MID_DISTANCE:
                repulsion_gain = HR_REPULSION_GAIN * HR_REPULSION_GAIN_MID_MULTIPLIER
            else:
                repulsion_gain = HR_REPULSION_GAIN
            return np.asarray(repulsion_gain * (distance_min - dist_hr) * direction, dtype=np.float32)

        if distance_max is not None and dist_hr > distance_max:
            return np.asarray(-HR_ATTRACTION_GAIN * (dist_hr - distance_max) * direction, dtype=np.float32)

        return np.zeros(2, dtype=np.float32)

    def _constrain_velocity_with_walkable(self, v_xy):
        v_xy = np.asarray(v_xy, dtype=np.float32)
        speed = np.linalg.norm(v_xy)
        if speed <= MIN_SPEED_EPS:
            return v_xy

        hit_distance = self._raycast_hit_distance(v_xy)
        if hit_distance is None or hit_distance >= RAYCAST_SLOWDOWN_DISTANCE_METERS:
            return v_xy

        clearance = max(0.0, hit_distance - RAYCAST_CLEARANCE_EPS)
        if clearance <= MIN_SPEED_EPS:
            return np.zeros(2, dtype=np.float32)
        return np.asarray(
            v_xy * min(1.0, clearance / RAYCAST_SLOWDOWN_DISTANCE_METERS),
            dtype=np.float32,
        )

    def _compute_wall_spacing_force(self, guide_xy):
        guide_xy = np.asarray(guide_xy, dtype=np.float32)
        guide_norm = float(np.linalg.norm(guide_xy))
        if guide_norm <= NORM_EPS:
            return np.zeros(2, dtype=np.float32)
        if guide_norm <= WALL_RAYCAST_GUIDE_SKIP_METERS:
            return np.zeros(2, dtype=np.float32)

        guide_dir = guide_xy / guide_norm
        left_dir = np.array([-guide_dir[1], guide_dir[0]], dtype=np.float32)
        right_dir = -left_dir

        wall_force = np.zeros(2, dtype=np.float32)
        left_hit_distance = self._raycast_hit_distance(left_dir)
        if (
            left_hit_distance is not None
            and left_hit_distance < WALL_REPULSION_DISTANCE_METERS
        ):
            wall_force -= np.asarray(
                WALL_REPULSION_GAIN
                * (WALL_REPULSION_DISTANCE_METERS - float(left_hit_distance))
                * left_dir,
                dtype=np.float32,
            )

        right_hit_distance = self._raycast_hit_distance(right_dir)
        if (
            right_hit_distance is not None
            and right_hit_distance < WALL_REPULSION_DISTANCE_METERS
        ):
            wall_force += np.asarray(
                WALL_REPULSION_GAIN
                * (WALL_REPULSION_DISTANCE_METERS - float(right_hit_distance))
                * left_dir,
                dtype=np.float32,
            )
        return wall_force

    def get_pose(self, data):
        return (
            float(data.qpos[self.qpos_idx]),
            float(data.qpos[self.qpos_idx + 1]),
            float(data.qpos[self.qpos_idx + 2]),
        )

    def _random_waypoint(self):
        return self.map_layout.sample_spawn_point(HUMAN_WALL_FOOTPRINT_RADIUS, rng=self.rng)

    def _wrap_to_pi(self, ang):
        return wrap_to_pi(ang)


# Imported after Human is defined so the behavior module can reuse Human
# constants without introducing a top-level circular import.
from . import human_behaviors
