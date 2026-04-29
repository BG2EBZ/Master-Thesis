import logging
from typing import Optional

import mujoco
import numpy as np

from .map_layouts import DEFAULT_MUSEUM_LAYOUT, MapLayout

logger = logging.getLogger(__name__)

DEFAULT_WAYPOINT_THRESHOLD = 0.2
DEFAULT_IMPATIENT_FRONT_OFFSET = 1.2
HUMAN_YAW_RATE_GAIN = 20.0
HUMAN_ROTATION_STOP_DEG = 3.0
LISTENING_RING_GAIN = 4.0
LISTENING_SECTOR_PROJECTION_EPS = 1e-2
DISTRACTED_SPEED_SCALE = 0.5
DISTRACTED_YAW_DEVIATION_MIN_DEG = 45.0
DISTRACTED_YAW_DEVIATION_MAX_DEG = 90.0
DISTRACTED_TARGET_DISTANCE_MIN = 0.5
DISTRACTED_TARGET_DISTANCE_MAX = 1.5
DEFAULT_SIM_TIMESTEP_SECONDS = 0.002
DISTRACTED_DURATION_SECONDS_DEFAULT = 10.0
DISTRACTED_STOP_DURATION_SECONDS = 5.0
DEFAULT_FOLLOW_RADIUS = 1.0
DISTRACTED_EXHIBIT_LOOK_RADIUS = 4.0
DISTRACTED_HUMAN_LOOK_RADIUS = 3.0
DISTRACTED_FALLBACK_DISTANCE = 1.0
LISTENING_IMPATIENT_YAW_DEVIATION_MIN_DEG = 45.0
LISTENING_IMPATIENT_YAW_DEVIATION_MAX_DEG = 90.0
LISTENING_IMPATIENT_TARGET_REACHED_DEG = 5.0
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
HUMAN_WALL_FOOTPRINT_RADIUS = 0.25
MIN_SPEED_EPS = 1e-6
RAYCAST_CLEARANCE_EPS = 1e-3
RAYCAST_SLOWDOWN_DISTANCE_METERS = 0.3
WALL_DETOUR_ANGLES_DEG = (30.0, -30.0, 60.0, -60.0, 90.0, -90.0)
WALL_DETOUR_ROTATIONS = tuple(
    (float(np.cos(np.deg2rad(angle_deg))), float(np.sin(np.deg2rad(angle_deg))))
    for angle_deg in WALL_DETOUR_ANGLES_DEG
)
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
        self.distracted_recovery_mode = HumanMode.FOLLOWING
        self.distracted_source = None

        self.impatient_duration = 800
        self.impatient_timer = 0
        self.impatient_speed_multiplier = 1.3
        self.impatient_front_offset = DEFAULT_IMPATIENT_FRONT_OFFSET
        self.impatient_original_max_speed = None
        self.impatient_recovery_mode = HumanMode.FOLLOWING
        self.listening_impatient_yaw_deviation = 0.0
        self.listening_impatient_turn_sign = 1.0

        self.profile = None
        self.following_steps = 0
        self.listening_steps = 0

        self.overwhelmed_stage = None
        self.overwhelmed_backoff_dist = 0.3
        self.overwhelmed_leave_speed = 1.0
        self.max_overwhelmed_leave_duration_seconds = 2.0
        self.overwhelmed_leave_duration = round(self.max_overwhelmed_leave_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS)
        self.overwhelmed_leave_timer = 0
        self.max_overwhelmed_pause_duration_seconds = 3.0
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
        if prev_mode == HumanMode.FOLLOWING:
            self.reset_following_duration()
        if prev_mode == HumanMode.OVERWHELMED:
            self.reset_overwhelmed_state()
        self.mode = mode
        if mode == HumanMode.DISTRACTED:
            self.distracted_timer = 0
            self.distracted_elapsed_steps = 0
            self._clear_distracted_navigation_state()

    def reset_overwhelmed_state(self):
        self.overwhelmed_stage = None
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_pause_timer = 0
        self.overwhelmed_leave_dir = np.zeros(2, dtype=np.float32)
        self.overwhelmed_backoff_start_xy = None

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
        self.reset_listening_session_state()
        self.reset_overwhelmed_state()

    def set_profile(self, profile: str):
        self.profile = profile
        if profile == HumanProfile.NEURODIVERGENT:
            self.hr_distance_min = float(HR_DISTANCE_MIN_ND_DEFAULT)
            self.hr_distance_max = float(HR_DISTANCE_MAX)
        else:
            self.hr_distance_min = float(HR_DISTANCE_MIN)
            self.hr_distance_max = float(HR_DISTANCE_MAX_NORMAL_DEFAULT)

    def reset_following_duration(self):
        self.following_steps = 0

    def reset_listening_session_state(self):
        self.listening_steps = 0

    def update_following_duration(self, eligible_following: bool):
        if eligible_following:
            self.following_steps += 1
        else:
            self.reset_following_duration()

    def update_listening_session_progress(self, active: bool):
        if active:
            self.listening_steps += 1

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
        self.set_mode(HumanMode.IMPATIENT)

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

    def _set_distracted_target_state(self, target_yaw: float, target_xy):
        target_xy = np.asarray(target_xy, dtype=np.float32)
        self.distracted_target_yaw = float(target_yaw)
        self.distracted_target_xy = target_xy
        self.distracted_stop_reached = False

    def apply_callback_response(self, response: str):
        if response == "rejoin":
            self.set_mode(self.distracted_recovery_mode)

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
        fan_half_angle = ctx["fan_half_angle"]
        target_mode = self.mode if mode is None else mode
        if n_humans > 1:
            relative_angle = (index / (n_humans - 1)) * (2 * fan_half_angle) - fan_half_angle
        else:
            relative_angle = 0.0

        if target_mode == HumanMode.FOLLOWING:
            radius = float(ctx.get("follow_radius", DEFAULT_FOLLOW_RADIUS))
            base_angle_offset = np.pi
        elif target_mode == HumanMode.IMPATIENT:
            radius = ctx["impatient_front_offset"]
            base_angle_offset = 0.0
        else:
            return

        rx, ry, ryaw = ctx["robot_pose"]
        angle = ryaw + base_angle_offset + relative_angle
        self.current_waypoint = np.array(
            [rx + radius * np.cos(angle), ry + radius * np.sin(angle)],
            dtype=np.float32,
        )

    def step(self, model, data, ctx):
        self._runtime_model = model
        self._runtime_data = data

        pose = self.get_pose(data)
        return human_behaviors.step_behavior(self, ctx, pose)

    def _move(self, to_target_xy, yaw, ctx, current_xy):
        robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
        v_repulsion = np.asarray(ctx["repulsion"], dtype=np.float32)
        v_hr = self._compute_hr_spacing_force(
            current_xy=current_xy,
            robot_xy=robot_xy,
            distance_min=self.hr_distance_min,
            distance_max=self.hr_distance_max,
        )

        to_target_xy = np.asarray(to_target_xy, dtype=np.float32)
        dist = np.linalg.norm(to_target_xy)
        if dist > NORM_EPS:
            v_follow = self.max_speed * (to_target_xy / dist)
        else:
            v_follow = np.zeros(2, dtype=np.float32)

        v_total = v_follow + v_repulsion + v_hr
        v_total = self._limit_speed(v_total, self.max_speed)

        v_total = self._adjust_target_velocity_for_walls(
            guide_xy=to_target_xy,
            desired_v_xy=v_total,
        )
        speed = np.linalg.norm(v_total)
        desired_yaw = np.arctan2(v_total[1], v_total[0]) if speed > NORM_EPS else yaw
        action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * self._wrap_to_pi(desired_yaw - yaw))
        return action

    def _adjust_target_velocity_for_walls(self, guide_xy, desired_v_xy):
        desired_v_xy = np.asarray(desired_v_xy, dtype=np.float32)
        desired_speed = float(np.linalg.norm(desired_v_xy))
        if desired_speed <= MIN_SPEED_EPS:
            return np.zeros(2, dtype=np.float32)

        guide_xy = np.asarray(guide_xy, dtype=np.float32)
        guide_norm = float(np.linalg.norm(guide_xy))
        if guide_norm <= NORM_EPS:
            return self._constrain_velocity_with_walkable(desired_v_xy)
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
        direction_xy = np.asarray(direction_xy, dtype=np.float32)
        desired_speed = np.linalg.norm(direction_xy)
        if desired_speed <= MIN_SPEED_EPS:
            return None
        if (
            self._runtime_model is None
            or self._runtime_data is None
            or self.body_id is None
            or not hasattr(self._runtime_data, "xpos")
        ):
            return None

        ray_direction = np.zeros(3, dtype=np.float64)
        ray_direction[:2] = direction_xy / desired_speed
        ray_origin = np.array(self._runtime_data.xpos[self.body_id], dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        geomid[0] = -1
        hit_distance = float(
            mujoco.mj_ray(
                self._runtime_model,
                self._runtime_data,
                ray_origin,
                ray_direction,
                None,
                1,
                int(self.body_id),
                geomid,
            )
        )
        return hit_distance if hit_distance >= 0.0 else None

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

    def get_pose(self, data):
        return (
            float(data.qpos[self.qpos_idx]),
            float(data.qpos[self.qpos_idx + 1]),
            float(data.qpos[self.qpos_idx + 2]),
        )

    def _random_waypoint(self):
        return self.map_layout.sample_spawn_point(HUMAN_WALL_FOOTPRINT_RADIUS, rng=np.random)

    def _wrap_to_pi(self, ang):
        return (ang + np.pi) % (2 * np.pi) - np.pi


# Imported after Human is defined so the behavior module can reuse Human
# constants without introducing a top-level circular import.
from . import human_behaviors
