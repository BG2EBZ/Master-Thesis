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
LISTENING_IMPATIENT_YAW_DEVIATION_MIN_DEG = 45.0
LISTENING_IMPATIENT_YAW_DEVIATION_MAX_DEG = 90.0
LISTENING_IMPATIENT_SWAY_SPEED_METERS_PER_SEC = 0.08
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
        self.max_distracted_duration_seconds = float(DISTRACTED_DURATION_SECONDS_DEFAULT)
        self.distracted_duration = max(
            1,
            int(round(self.max_distracted_duration_seconds / DEFAULT_SIM_TIMESTEP_SECONDS)),
        )
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
        self.overwhelmed_leave_speed = 1.5
        self.overwhelmed_leave_duration = 1000
        self.overwhelmed_leave_timer = 0
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
            self._clear_distracted_navigation_state()

    def reset_overwhelmed_state(self):
        self.overwhelmed_stage = None
        self.overwhelmed_leave_timer = 0
        self.overwhelmed_leave_dir = np.zeros(2, dtype=np.float32)
        self.overwhelmed_backoff_start_xy = None
        self.overwhelmed_recovery_mode = HumanMode.FOLLOWING

    def reset_episode_state(self):
        self.mode = None
        self.set_mode(HumanMode.WANDERING)
        self.current_waypoint = self._random_waypoint()

        self.distracted_timer = 0
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
        self.overwhelmed_leave_dir = np.asarray(leave_dir, dtype=np.float32)
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
        self.current_waypoint = target_xy.copy()
        self.distracted_stop_reached = False

    def _initialize_distracted_target(self, current_xy, current_yaw: float):
        target_yaw, sampled_target_xy = self._sample_distracted_target_candidate(
            current_xy=np.asarray(current_xy, dtype=np.float32),
            current_yaw=current_yaw,
        )
        self._set_distracted_target_state(target_yaw=target_yaw, target_xy=sampled_target_xy)

    def _sample_distracted_target_candidate(self, current_xy, current_yaw: float):
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
        return float(target_yaw), np.asarray(current_xy + target_distance * direction_xy, dtype=np.float32)

    def apply_callback_response(self, response: str):
        if response == "rejoin":
            self.set_mode(self.distracted_recovery_mode)

    @staticmethod
    def _compose_action(v_xy, yaw_rate):
        action = np.zeros(3, dtype=np.float32)
        action[:2] = v_xy
        action[2] = np.float32(yaw_rate)
        return action

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
            radius = ctx["follow_radius"]
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
        if self.mode == HumanMode.WANDERING:
            return self.step_wandering(ctx, pose)
        if self.mode == HumanMode.FOLLOWING:
            self.assign_target_from_context(ctx)
            return self.step_following(ctx, pose)
        if self.mode == HumanMode.LISTENING:
            return self.step_listening(ctx, pose)
        if self.mode == HumanMode.DISTRACTED:
            return self.step_distracted(ctx, pose)
        if self.mode == HumanMode.OVERWHELMED:
            return self.step_overwhelmed(ctx, pose)
        if self.mode == HumanMode.IMPATIENT:
            return self.step_impatient(ctx, pose)
        raise ValueError(f"Unknown human mode {self.mode}")

    def step_wandering(self, ctx, pose):
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        yaw = pose[2]
        to_waypoint = self.current_waypoint - current_xy
        if np.linalg.norm(to_waypoint) < self.waypoint_threshold:
            self.current_waypoint = self._random_waypoint()
            to_waypoint = self.current_waypoint - current_xy
        return self._move(to_waypoint, yaw, ctx, current_xy)

    def step_following(self, ctx, pose):
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        return self._move(self.current_waypoint - current_xy, pose[2], ctx, current_xy)

    def step_listening(self, ctx, pose):
        yaw = pose[2]
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
        robot_yaw = ctx["robot_yaw"]
        to_robot = robot_xy - current_xy
        dist_to_robot = np.linalg.norm(to_robot)
        desired_yaw = np.arctan2(to_robot[1], to_robot[0]) if dist_to_robot > NORM_EPS else robot_yaw
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        target_xy = self._compute_listening_sector_target_point(
            current_xy=current_xy,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
            listen_radius=ctx["listen_radius"],
            sector_half_angle=ctx["listening_sector_half_angle"],
        )
        v_goal = LISTENING_RING_GAIN * (target_xy - current_xy)
        v_total = v_goal + np.asarray(ctx["repulsion"], dtype=np.float32)
        v_total += self._compute_hr_spacing_force(
            current_xy=current_xy,
            robot_xy=robot_xy,
            distance_min=self.hr_distance_min,
            distance_max=None,
        )
        speed = np.linalg.norm(v_total)
        if speed > self.max_speed and speed > NORM_EPS:
            v_total = v_total / speed * self.max_speed

        action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
        return self._apply_wall_constraint_to_action(action, current_xy)

    def step_listening_with_anchor_target_and_live_repulsion(
        self,
        ctx,
        pose,
        *,
        anchor_robot_xy,
        anchor_robot_yaw: float,
        live_robot_xy,
    ):
        yaw = pose[2]
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        anchor_robot_xy = np.asarray(anchor_robot_xy, dtype=np.float32)
        live_robot_xy = np.asarray(live_robot_xy, dtype=np.float32)
        to_anchor_robot = anchor_robot_xy - current_xy
        dist_to_anchor_robot = np.linalg.norm(to_anchor_robot)
        desired_yaw = (
            np.arctan2(to_anchor_robot[1], to_anchor_robot[0])
            if dist_to_anchor_robot > NORM_EPS
            else anchor_robot_yaw
        )
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        target_xy = self._compute_listening_sector_target_point(
            current_xy=current_xy,
            robot_xy=anchor_robot_xy,
            robot_yaw=anchor_robot_yaw,
            listen_radius=ctx["listen_radius"],
            sector_half_angle=ctx["listening_sector_half_angle"],
        )
        v_goal = LISTENING_RING_GAIN * (target_xy - current_xy)
        v_total = v_goal + np.asarray(ctx["repulsion"], dtype=np.float32)
        v_total += self._compute_hr_spacing_force(
            current_xy=current_xy,
            robot_xy=live_robot_xy,
            distance_min=self.hr_distance_min,
            distance_max=None,
        )
        speed = np.linalg.norm(v_total)
        if speed > self.max_speed and speed > NORM_EPS:
            v_total = v_total / speed * self.max_speed

        action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
        return self._apply_wall_constraint_to_action(action, current_xy)

    def step_distracted(self, ctx, pose):
        if self.distracted_source == DISTRACTED_SOURCE_LISTENING:
            return self.step_listening_distracted(ctx, pose)

        yaw = pose[2]
        self.distracted_timer += 1
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        if self.distracted_target_xy is None:
            self._initialize_distracted_target(current_xy=current_xy, current_yaw=yaw)

        target_xy = np.asarray(self.distracted_target_xy, dtype=np.float32)
        to_target = target_xy - current_xy
        dist_to_target = np.linalg.norm(to_target)
        if dist_to_target < self.waypoint_threshold:
            self.distracted_stop_reached = True

        if self.distracted_stop_reached:
            action = np.zeros(3, dtype=np.float32)
        else:
            move_speed_limit = DISTRACTED_SPEED_SCALE * self.max_speed
            if dist_to_target > NORM_EPS:
                v_goal = move_speed_limit * (to_target / dist_to_target)
            else:
                v_goal = np.zeros(2, dtype=np.float32)

            v_total = v_goal + np.asarray(ctx["repulsion"], dtype=np.float32)
            speed = np.linalg.norm(v_total)
            if speed > move_speed_limit and speed > NORM_EPS:
                v_total = v_total / speed * move_speed_limit
                speed = np.linalg.norm(v_total)

            desired_yaw = (
                np.arctan2(v_total[1], v_total[0])
                if speed > NORM_EPS
                else self.distracted_target_yaw
            )
            yaw_err = self._wrap_to_pi(desired_yaw - yaw)
            action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)

        if self.distracted_timer >= self.distracted_duration:
            self.set_mode(self.distracted_recovery_mode)
            if self.enable_event_logs:
                logger.info(f">>> {self.name} recovered -> {self.mode.upper()}")

        return self._apply_wall_constraint_to_action(action, current_xy)

    def _initialize_listening_distracted_target(self, current_xy, robot_xy):
        current_xy = np.asarray(current_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        deviation_deg = np.random.uniform(45.0, 90.0)
        deviation_sign = -1.0 if np.random.rand() < 0.5 else 1.0
        robot_facing_yaw = np.arctan2(robot_xy[1] - current_xy[1], robot_xy[0] - current_xy[0])
        target_yaw = self._wrap_to_pi(robot_facing_yaw + deviation_sign * np.deg2rad(deviation_deg))
        self._set_distracted_target_state(target_yaw=target_yaw, target_xy=current_xy)

    def step_listening_distracted(self, ctx, pose):
        self.distracted_timer += 1
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        desired_yaw = self.distracted_target_yaw
        if desired_yaw is None:
            self._initialize_listening_distracted_target(
                current_xy=current_xy,
                robot_xy=np.asarray(ctx["robot_xy"], dtype=np.float32),
            )
            desired_yaw = self.distracted_target_yaw
        yaw_err = self._wrap_to_pi(desired_yaw - pose[2])

        if abs(yaw_err) >= np.deg2rad(HUMAN_ROTATION_STOP_DEG):
            action = self._compose_action(np.zeros(2, dtype=np.float32), HUMAN_YAW_RATE_GAIN * yaw_err)
            return self._apply_wall_constraint_to_action(action, current_xy)
        return np.zeros(3, dtype=np.float32)

    def step_overwhelmed(self, ctx, pose):
        pos_xy = np.asarray(pose[:2], dtype=np.float32)
        leave_dir = np.asarray(self.overwhelmed_leave_dir, dtype=np.float32)
        leave_dir = leave_dir / np.linalg.norm(leave_dir)
        desired_yaw = np.arctan2(leave_dir[1], leave_dir[0])

        if self.overwhelmed_stage == "backoff":
            backoff_target = self.overwhelmed_backoff_start_xy + self.overwhelmed_backoff_dist * leave_dir
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

            action = self._compose_action(v_xy, HUMAN_YAW_RATE_GAIN * self._wrap_to_pi(desired_yaw - pose[2]))
            return self._apply_wall_constraint_to_action(action, pos_xy)

        self.overwhelmed_leave_timer += 1
        v_xy = min(self.overwhelmed_leave_speed, self.max_speed) * leave_dir
        if self.overwhelmed_leave_timer >= self.overwhelmed_leave_duration:
            self.set_mode(self.overwhelmed_recovery_mode)
            if self.enable_event_logs:
                logger.info(f">>> {self.name} recovered from OVERWHELMED -> {self.mode.upper()}")

        action = self._compose_action(v_xy, HUMAN_YAW_RATE_GAIN * self._wrap_to_pi(desired_yaw - pose[2]))
        return self._apply_wall_constraint_to_action(action, pos_xy)

    def step_impatient(self, ctx, pose):
        self.impatient_timer += 1
        current_xy = np.asarray(pose[:2], dtype=np.float32)
        yaw = pose[2]
        if self.impatient_recovery_mode == HumanMode.LISTENING:
            robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
            to_robot = robot_xy - current_xy
            base_yaw = np.arctan2(to_robot[1], to_robot[0]) if np.linalg.norm(to_robot) > NORM_EPS else yaw
            desired_yaw = self._wrap_to_pi(
                base_yaw + self.listening_impatient_turn_sign * self.listening_impatient_yaw_deviation
            )
            yaw_err = self._wrap_to_pi(desired_yaw - yaw)
            if abs(yaw_err) <= np.deg2rad(LISTENING_IMPATIENT_TARGET_REACHED_DEG):
                self.listening_impatient_turn_sign *= -1.0
                desired_yaw = self._wrap_to_pi(
                    base_yaw + self.listening_impatient_turn_sign * self.listening_impatient_yaw_deviation
                )
                yaw_err = self._wrap_to_pi(desired_yaw - yaw)

            perp = np.array([-np.sin(base_yaw), np.cos(base_yaw)], dtype=np.float32)
            v_total = (
                self.listening_impatient_turn_sign
                * LISTENING_IMPATIENT_SWAY_SPEED_METERS_PER_SEC
                * perp
            )
            v_total += np.asarray(ctx["repulsion"], dtype=np.float32)
            speed = np.linalg.norm(v_total)
            if speed > self.max_speed and speed > NORM_EPS:
                v_total = v_total / speed * self.max_speed
            action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
            action = self._apply_wall_constraint_to_action(action, current_xy)
        else:
            self.assign_target_from_context(ctx)
            action = self.step_following(ctx, pose)

        if self.impatient_timer >= self.impatient_duration:
            self.set_mode(self.impatient_recovery_mode)
            if self.enable_event_logs:
                logger.info(f">>> {self.name} recovered from IMPATIENT -> {self.mode.upper()}")
        return action

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
        speed = np.linalg.norm(v_total)
        if speed > self.max_speed:
            v_total = v_total / speed * self.max_speed

        desired_yaw = np.arctan2(v_total[1], v_total[0]) if speed > NORM_EPS else yaw
        action = self._compose_action(v_total, HUMAN_YAW_RATE_GAIN * self._wrap_to_pi(desired_yaw - yaw))
        return self._apply_wall_constraint_to_action(action, current_xy)

    def _raycast_hit_distance(self, direction_xy):
        direction_xy = np.asarray(direction_xy, dtype=np.float32)
        desired_speed = np.linalg.norm(direction_xy)
        if desired_speed <= MIN_SPEED_EPS:
            return None

        ray_direction = np.zeros(3, dtype=np.float64)
        ray_direction[:2] = direction_xy / desired_speed
        ray_origin = np.array(self._runtime_data.xpos[self.body_id], dtype=np.float64)
        primary_height = float(ray_origin[2])
        secondary_height = min(primary_height, 0.10)
        if abs(primary_height - secondary_height) <= MIN_SPEED_EPS:
            ray_heights = (primary_height,)
        else:
            ray_heights = (primary_height, secondary_height)

        geomid = np.array([-1], dtype=np.int32)
        best_hit_distance = None
        for ray_height in ray_heights:
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
            if hit_distance >= 0.0 and (best_hit_distance is None or hit_distance < best_hit_distance):
                best_hit_distance = hit_distance
        return best_hit_distance

    def is_within_listening_front_sector(self, point_xy, robot_xy, robot_yaw: float, sector_half_angle: float) -> bool:
        point_xy = np.asarray(point_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        rel_xy = point_xy - robot_xy
        if np.dot(rel_xy, rel_xy) <= (NORM_EPS * NORM_EPS):
            rel_angle = 0.0
        else:
            rel_angle = self._wrap_to_pi(np.arctan2(rel_xy[1], rel_xy[0]) - robot_yaw)
        return bool(abs(rel_angle) <= sector_half_angle + LISTENING_SECTOR_PROJECTION_EPS)

    def _compute_listening_sector_target_point(
        self,
        current_xy,
        robot_xy,
        robot_yaw: float,
        listen_radius: float,
        sector_half_angle: float,
    ):
        current_xy = np.asarray(current_xy, dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        rel_xy = current_xy - robot_xy
        if np.dot(rel_xy, rel_xy) <= (NORM_EPS * NORM_EPS):
            absolute_angle = robot_yaw
        else:
            absolute_angle = np.arctan2(rel_xy[1], rel_xy[0])
        relative_angle = self._wrap_to_pi(absolute_angle - robot_yaw)
        half_angle = max(0.0, sector_half_angle - LISTENING_SECTOR_PROJECTION_EPS)
        clamped_angle = self._wrap_to_pi(robot_yaw + np.clip(relative_angle, -half_angle, half_angle))
        return np.array(
            [
                robot_xy[0] + listen_radius * np.cos(clamped_angle),
                robot_xy[1] + listen_radius * np.sin(clamped_angle),
            ],
            dtype=np.float32,
        )

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

    def _constrain_velocity_with_walkable(self, current_xy, v_xy):
        del current_xy
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
        return np.asarray(
            v_xy * min(1.0, clearance / RAYCAST_SLOWDOWN_DISTANCE_METERS),
            dtype=np.float32,
        )

    def _apply_wall_constraint_to_action(self, action, current_xy):
        constrained_action = np.array(action, dtype=np.float32)
        constrained_action[0:2] = self._constrain_velocity_with_walkable(current_xy, constrained_action[0:2])
        return constrained_action

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
