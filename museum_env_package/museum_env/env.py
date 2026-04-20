import logging
from importlib import resources
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from .env_reporting import (
    apply_label_scene_option_to_viewer,
    apply_robot_visual_state,
    build_label_scene_option,
    build_step_info,
    resolve_robot_visual_state,
)
from .env_runtime import (
    DIST_EPS,
    build_human_goals,
    build_world_frame,
    compute_reached_goal_indices,
    resolve_fuzzy_metric_input,
)
from .env_state import (
    FOLLOW_PHASE_PRE_LISTEN_ENGAGE,
    FOLLOW_PHASE_TRANSIT,
    LISTEN_PHASE_INTRO,
    LISTEN_PHASE_WAIT,
    POST_EXPLANATION_ROLE_WAIT,
    POST_EXPLANATION_ROLE_YIELD,
    CallbackState,
    ListeningState,
    PostExplanationState,
    RuntimeCache,
    StepEvents,
    build_fuzzy_debug_states,
)
from .human import (
    DISTRACTED_SOURCE_FOLLOWING,
    DISTRACTED_SOURCE_LISTENING,
    HUMAN_WALL_FOOTPRINT_RADIUS,
    LISTENING_DISTRACTED_MOVE_SECONDS,
    Human,
    HumanMode,
    HumanProfile,
)
from .map_layouts import DEFAULT_MUSEUM_LAYOUT, MapLayout, get_map_layout
from .metrics import VectorizedRollingWindow
from .robot import (
    ROBOT_WAYPOINT_REACHED_DIST,
    Robot,
    RobotCallbackPhase,
    RobotMode,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False

ACTION_LOW = -1.0
ACTION_HIGH = 1.0
MAX_STEPS_DEFAULT = 100000
HUMAN_FOLLOW_DISTANCE_DEFAULT = 1.0
SOCIAL_DISTANCE_DEFAULT = 0.8
REPULSION_GAIN_DEFAULT = 4.0
FOLLOW_FAN_HALF_ANGLE_DEG = 80.0
LISTENING_FRONT_SECTOR_HALF_ANGLE_DEG = 70.0
LISTEN_FAN_RADIUS_DEFAULT = 1.0
LISTEN_STAND_THRESHOLD_DEFAULT = 0.05
LISTENING_REPULSION_SCALE = 1.0
LISTEN_REACHED_MIN_DISTANCE = 0.8
LISTEN_INTRO_DELAY_SECONDS_DEFAULT = 3.0
LISTEN_WAIT_SECONDS_DEFAULT = 20.0
HUMAN_MAX_SPEED_DEFAULT = 1.0
FOLLOW_RADIUS_DEFAULT = 1.0
HUMAN_GOAL_THRESHOLD = 0.1
POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD = 0.5
POST_EXPLANATION_HOLD_RESUME_DISTANCE = 2.5
POST_EXPLANATION_YIELD_CORRIDOR_WIDTH = 0.8
POST_EXPLANATION_YIELD_CLOSE_DISTANCE = 1.1
POST_EXPLANATION_YIELD_DISTANCE = 0.5
HUMAN_HUMAN_DISTANCE_WINDOW_SECONDS = 1.0
MAX_DISTRACTED_DURATION_SECONDS_DEFAULT = 15.0

MAX_HUMANS_CAPACITY = 15
HUMAN_SPAWN_MIN_DISTANCE = (2.0 * HUMAN_WALL_FOOTPRINT_RADIUS) + 0.10
HUMAN_SPAWN_MIN_ROBOT_DISTANCE = SOCIAL_DISTANCE_DEFAULT
HUMAN_SPAWN_MAX_ATTEMPTS_PER_HUMAN = 2000

INACTIVE_HUMAN_PARK_X = 50.0
INACTIVE_HUMAN_PARK_Y_BASE = 50.0

CALLBACK_CUE_SECONDS = 3.0
CALLBACK_RESPONSE_SAMPLE_SECONDS = 2.0
CALLBACK_TRIGGER_DISTANCE_METERS_DEFAULT = 2.0
CALLBACK_REJOIN_PROB_NORMAL_DEFAULT = 0.80
CALLBACK_IGNORE_PROB_NORMAL_DEFAULT = 0.20
CALLBACK_REJOIN_PROB_ND_DEFAULT = 0.40
CALLBACK_IGNORE_PROB_ND_DEFAULT = 0.60
ROBOT_HAPPY_HOLD_SECONDS = 3.0


class MuseumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path=None,
        map_name: str = DEFAULT_MUSEUM_LAYOUT.name,
        map_layout: Optional[MapLayout] = None,
        render_mode=None,
        enable_event_logs: bool = True,
        max_distracted_duration_seconds: float = MAX_DISTRACTED_DURATION_SECONDS_DEFAULT,
        callback_rejoin_prob_normal: float = CALLBACK_REJOIN_PROB_NORMAL_DEFAULT,
        callback_ignore_prob_normal: float = CALLBACK_IGNORE_PROB_NORMAL_DEFAULT,
        callback_rejoin_prob_nd: float = CALLBACK_REJOIN_PROB_ND_DEFAULT,
        callback_ignore_prob_nd: float = CALLBACK_IGNORE_PROB_ND_DEFAULT,
        callback_trigger_distance_meters: float = CALLBACK_TRIGGER_DISTANCE_METERS_DEFAULT,
        observation_update_period_seconds: float = 0.1,
        n_humans: int = 15,
    ):
        super().__init__()
        self.enable_event_logs = enable_event_logs
        logger.setLevel(logging.INFO if self.enable_event_logs else logging.CRITICAL + 1)

        self.map_layout = map_layout if map_layout is not None else get_map_layout(map_name)
        self.map_name = self.map_layout.name
        if xml_path is None:
            with resources.path("museum_env.assets", self.map_layout.default_xml_asset) as xml_file:
                xml_path = str(xml_file)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self.render_width = 1920
        self.render_height = 1080

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_update_period_seconds = float(observation_update_period_seconds)
        self.observation_update_period_steps = max(
            1,
            int(round(self.observation_update_period_seconds / self.dt)),
        )
        self.max_steps = MAX_STEPS_DEFAULT
        self.step_count = 0
        self._callback_cue_steps = max(1, int(round(CALLBACK_CUE_SECONDS / self.dt)))
        self._callback_response_sample_steps = max(
            1,
            int(round(CALLBACK_RESPONSE_SAMPLE_SECONDS / self.dt)),
        )

        self.robot = Robot(
            waypoints=self.map_layout.robot_waypoints,
            v_max=1.0,
            k_v=20.0,
            k_yaw=20.0,
        )
        self.follow_phase = None
        self.robot_start_xy = None
        self.human_follow_distance = HUMAN_FOLLOW_DISTANCE_DEFAULT

        self.social_distance = SOCIAL_DISTANCE_DEFAULT
        self.repulsion_gain = REPULSION_GAIN_DEFAULT
        self.follow_fan_half_angle = np.deg2rad(FOLLOW_FAN_HALF_ANGLE_DEG)
        self.listen_front_sector_half_angle = np.deg2rad(LISTENING_FRONT_SECTOR_HALF_ANGLE_DEG)
        self.listen_fan_radius = LISTEN_FAN_RADIUS_DEFAULT
        self.listen_stand_threshold = LISTEN_STAND_THRESHOLD_DEFAULT
        self.listen_intro_delay_seconds = LISTEN_INTRO_DELAY_SECONDS_DEFAULT
        self.listen_intro_delay_steps = max(1, int(round(self.listen_intro_delay_seconds / self.dt)))
        self.listen_wait_seconds = LISTEN_WAIT_SECONDS_DEFAULT
        self.listen_wait_steps = max(1, int(round(self.listen_wait_seconds / self.dt)))

        self.listening_state = ListeningState()
        self.post_explanation_state = PostExplanationState()
        self.callback_state = CallbackState()
        self.runtime_cache = RuntimeCache()
        self.max_distracted_duration_seconds = float(max_distracted_duration_seconds)
        self.callback_trigger_distance_meters = float(callback_trigger_distance_meters)
        self.callback_response_profile_probs = {
            HumanProfile.NORMAL: {
                "rejoin": float(callback_rejoin_prob_normal),
                "ignore": float(callback_ignore_prob_normal),
            },
            HumanProfile.NEURODIVERGENT: {
                "rejoin": float(callback_rejoin_prob_nd),
                "ignore": float(callback_ignore_prob_nd),
            },
        }
        from .fuzzy import FollowingFuzzyEngine

        self.following_fuzzy_engine = FollowingFuzzyEngine()
        self.perceived_distracted_indices = []
        self._last_robot_visual_signature = None

        self.all_humans = [
            Human(
                f"person{idx}",
                f"person{idx}",
                qpos_idx=3 * idx,
                max_speed=HUMAN_MAX_SPEED_DEFAULT,
                map_layout=self.map_layout,
            )
            for idx in range(1, MAX_HUMANS_CAPACITY + 1)
        ]
        for human in self.all_humans:
            human.set_mode(HumanMode.WANDERING)
            human.set_event_logging(self.enable_event_logs)

        self.humans = []
        self.n_humans = 0
        self._set_active_humans(n_humans)
        self._configure_human_behaviors()

        self.nu = 3 + (3 * len(self.humans))
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(self.nu,),
            dtype=np.float32,
        )

        self.robot_body_id = self.model.body("robot").id
        self.robot_base_geom_id = self.model.geom("robot_base").id
        self.robot_speaking_halo_geom_id = self.model.geom("robot_speaking_halo").id
        self.all_human_body_ids = [self.model.body(human.body_name).id for human in self.all_humans]
        for human, body_id in zip(self.all_humans, self.all_human_body_ids):
            human.body_id = body_id
        self.human_body_ids = self.all_human_body_ids[: len(self.humans)]

        self._label_scene_option = build_label_scene_option()
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state(force=True)

    def _log_event(self, msg: str) -> None:
        if self.enable_event_logs:
            logger.info(msg)

    def _set_active_humans(self, n_humans: int) -> None:
        self.humans = list(self.all_humans[: max(0, int(n_humans))])
        self.n_humans = len(self.humans)
        for human in self.all_humans:
            human.set_profile(HumanProfile.NORMAL)
        if self.humans:
            self.humans[0].set_profile(HumanProfile.NEURODIVERGENT)

        window_steps = max(
            1,
            int(round(HUMAN_HUMAN_DISTANCE_WINDOW_SECONDS / self.observation_update_period_seconds)),
        )
        self.hh_distance_metric = VectorizedRollingWindow(window_steps, self.n_humans)
        self.hr_distance_metric = VectorizedRollingWindow(window_steps, self.n_humans)
        self.fuzzy_debug = build_fuzzy_debug_states(self.n_humans)
        self.callback_state.reset(self.n_humans)
        self.runtime_cache.reset()
        if hasattr(self, "all_human_body_ids"):
            self.human_body_ids = self.all_human_body_ids[: self.n_humans]

    def _configure_human_behaviors(self) -> None:
        for human in self.humans:
            human.max_distracted_duration_seconds = self.max_distracted_duration_seconds
            human.distracted_duration = max(
                1,
                int(round(human.max_distracted_duration_seconds / self.dt)),
            )
            human.listening_distracted_move_steps = max(
                1,
                int(round(LISTENING_DISTRACTED_MOVE_SECONDS / self.dt)),
            )
            human.can_be_overwhelmed = True
            human.can_be_impatient = True
            human.impatient_duration = 2000
            human.impatient_speed_multiplier = 1.6
            human.impatient_front_offset = 1.0

    def _sample_active_human_spawn_states(self, robot_xy) -> list[np.ndarray]:
        sampled_states: list[np.ndarray] = []
        sampled_positions: list[np.ndarray] = []
        for _human in self.humans:
            for _attempt in range(HUMAN_SPAWN_MAX_ATTEMPTS_PER_HUMAN):
                candidate_xy = self.map_layout.sample_spawn_point(
                    HUMAN_WALL_FOOTPRINT_RADIUS,
                    rng=self.np_random,
                )
                if np.linalg.norm(candidate_xy - robot_xy) < HUMAN_SPAWN_MIN_ROBOT_DISTANCE:
                    continue
                if any(
                    np.linalg.norm(candidate_xy - existing_xy) < HUMAN_SPAWN_MIN_DISTANCE
                    for existing_xy in sampled_positions
                ):
                    continue
                yaw = self.np_random.uniform(-np.pi, np.pi)
                sampled_states.append(
                    np.array([candidate_xy[0], candidate_xy[1], yaw], dtype=np.float32)
                )
                sampled_positions.append(candidate_xy)
                break
        return sampled_states

    def _reset_human_positions(self, robot_xy) -> None:
        active_spawn_states = self._sample_active_human_spawn_states(robot_xy=np.asarray(robot_xy, dtype=np.float32))
        for human, spawn_state in zip(self.humans, active_spawn_states):
            self.data.qpos[human.qpos_idx : human.qpos_idx + 3] = spawn_state
            self.data.qvel[human.qpos_idx : human.qpos_idx + 3] = 0.0
            human.current_waypoint = self.map_layout.sample_spawn_point(
                HUMAN_WALL_FOOTPRINT_RADIUS,
                rng=self.np_random,
            )

        inactive_humans = self.all_humans[len(self.humans) :]
        for park_idx, human in enumerate(inactive_humans):
            park_pose = np.array(
                [INACTIVE_HUMAN_PARK_X + park_idx, INACTIVE_HUMAN_PARK_Y_BASE, 0.0],
                dtype=np.float32,
            )
            self.data.qpos[human.qpos_idx : human.qpos_idx + 3] = park_pose
            self.data.qvel[human.qpos_idx : human.qpos_idx + 3] = 0.0

    def _sync_robot_speaker_state(self) -> None:
        self.robot.set_speaker_active(
            bool(
                self.listening_state.phase == LISTEN_PHASE_WAIT
                or (
                    self.robot.callback_active
                    and self.robot.callback_phase == RobotCallbackPhase.CUE
                    and self.robot.callback_cue_elapsed_steps < self.robot.callback_cue_total_steps
                )
            )
        )

    def _sync_robot_visual_state(self, force: bool = False) -> None:
        visual_state = resolve_robot_visual_state(
            robot=self.robot,
            callback_visual_active=(
                self.robot.callback_active
                and self.robot.callback_phase == RobotCallbackPhase.CUE
                and self.robot.callback_cue_elapsed_steps < self.robot.callback_cue_total_steps
            ),
        )
        if (not force) and visual_state.signature == self._last_robot_visual_signature:
            return
        apply_robot_visual_state(
            model=self.model,
            robot_base_geom_id=self.robot_base_geom_id,
            robot_speaking_halo_geom_id=self.robot_speaking_halo_geom_id,
            label_scene_option=self._label_scene_option,
            visual_state=visual_state,
        )
        self._last_robot_visual_signature = visual_state.signature

    def _is_robot_in_move_stage(self, robot_pose) -> bool:
        if self.robot.listen_mode or self.listening_state.phase == LISTEN_PHASE_WAIT or self.robot.callback_active:
            return False
        rx, ry, _ = robot_pose
        wx, wy = self.robot.get_current_waypoint()
        dist = float(np.hypot(wx - rx, wy - ry) + DIST_EPS)
        return dist >= ROBOT_WAYPOINT_REACHED_DIST

    def _analyze_humans(self, world_frame) -> dict:
        if len(self.humans) == 0:
            return {
                "perceived_distracted_indices": [],
                "callback_target_idx": None,
                "emotion_modes": [],
            }

        human_modes = [human.mode for human in self.humans]
        emotion_modes = [mode for mode in human_modes if mode != HumanMode.DISTRACTED]
        mode_array = np.asarray(human_modes, dtype=object)
        dist = np.linalg.norm(world_frame.human_xy - world_frame.robot_xy[None, :], axis=1)

        distracted_mask = mode_array == HumanMode.DISTRACTED
        far_distracted_mask = distracted_mask & (dist > self.callback_trigger_distance_meters)
        perceived_distracted_indices = [int(idx) for idx in np.flatnonzero(far_distracted_mask)]

        distracted_source = np.asarray(
            [human.distracted_source for human in self.humans],
            dtype=object,
        )
        listening_distracted_mask = distracted_mask & (
            distracted_source == DISTRACTED_SOURCE_LISTENING
        )
        rearm_eligible = ~np.asarray(self.callback_state.triggered_for_distracted, dtype=bool)

        callback_target_idx = None
        if self.listening_state.controller_active and not self.robot.callback_active:
            candidate_mask = listening_distracted_mask & rearm_eligible
            if np.any(candidate_mask):
                callback_target_idx = int(np.flatnonzero(candidate_mask)[0])
        elif self._is_robot_in_move_stage(world_frame.robot_pose):
            candidate_mask = far_distracted_mask & rearm_eligible
            if np.any(candidate_mask):
                candidate_indices = np.flatnonzero(candidate_mask)
                farthest_local_idx = int(np.argmax(dist[candidate_mask]))
                callback_target_idx = int(candidate_indices[farthest_local_idx])

        return {
            "perceived_distracted_indices": perceived_distracted_indices,
            "callback_target_idx": callback_target_idx,
            "emotion_modes": emotion_modes,
        }

    def _sample_callback_response(self, profile: str):
        u = self.np_random.random()
        profile_probs = self.callback_response_profile_probs.get(
            profile,
            self.callback_response_profile_probs[HumanProfile.NORMAL],
        )
        rejoin_threshold = profile_probs["rejoin"]
        ignore_threshold = rejoin_threshold + profile_probs["ignore"]
        if u < rejoin_threshold:
            return "rejoin"
        if u < ignore_threshold:
            return "ignore"
        return "ignore"

    def _maybe_sample_active_callback_response(self, events: StepEvents) -> None:
        if not self.robot.callback_active or self.robot.callback_phase != RobotCallbackPhase.CUE:
            return
        if self.robot.callback_response_sampled:
            return
        if int(self.robot.callback_cue_elapsed_steps) < int(self._callback_response_sample_steps):
            return

        target_idx = self.callback_state.active_target_idx
        self.robot.callback_response_sampled = True
        if target_idx is None or not (0 <= target_idx < len(self.humans)):
            return

        target_human = self.humans[target_idx]
        if target_human.mode != HumanMode.DISTRACTED:
            return

        callback_response = self._sample_callback_response(profile=target_human.profile)
        target_human.apply_callback_response(response=callback_response, stay_steps=0)
        self.callback_state.last_response = str(callback_response)
        self.callback_state.last_response_target_idx = int(target_idx)
        self._log_event(
            f">>> person{target_idx + 1} callback response: {callback_response} "
            f"(attempt {int(self.robot.callback_attempt_index)})."
        )

    def _resolve_completed_callback(self, events: StepEvents) -> None:
        if not self.robot.callback_cue_completed_this_step:
            return

        target_idx = self.callback_state.active_target_idx
        target_human = None
        if target_idx is not None and 0 <= target_idx < len(self.humans):
            target_human = self.humans[target_idx]

        success = target_human is not None and target_human.mode == self.callback_state.success_mode
        if success:
            events.callback_completed = True
            events.callback_success = True
            hold_steps = max(1, int(round(ROBOT_HAPPY_HOLD_SECONDS / self.dt)))
            self.robot.trigger_happy(hold_steps)
            events.happy_triggered = True
            if target_idx is not None:
                self._log_event(
                    f">>> Robot CALLBACK succeeded for person{target_idx + 1} "
                    f"on attempt {int(self.robot.callback_attempt_index)}."
                )
            self.robot._finish_callback()
            self.callback_state.active_target_idx = None
            self.callback_state.success_mode = HumanMode.FOLLOWING
            if self.listening_state.interrupted:
                self.listening_state.resume()
                self.robot.listen_mode = self.listening_state.active
            return

        if int(self.robot.callback_attempt_index) < 2 and self.robot.start_next_callback_attempt():
            if target_idx is not None:
                self._log_event(f">>> Robot CALLBACK retry started for person{target_idx + 1}.")
            return

        events.callback_completed = True
        if target_idx is not None:
            self._log_event(
                f">>> Robot CALLBACK ended after second attempt for person{target_idx + 1}."
            )
        self.robot._finish_callback()
        self.callback_state.active_target_idx = None
        self.callback_state.success_mode = HumanMode.FOLLOWING
        if self.listening_state.interrupted:
            self.listening_state.resume()
            self.robot.listen_mode = self.listening_state.active

    def _should_evaluate_fuzzy(self, idx: int, context: str) -> bool:
        if context == "following" and self.follow_phase != FOLLOW_PHASE_TRANSIT:
            return False
        if context == "listening" and not self.listening_state.fuzzy_active:
            return False
        debug_state = self.fuzzy_debug[idx]
        if debug_state.dominant_state is None:
            return True
        if debug_state.context != context:
            return True
        return self.runtime_cache.refresh_counter > debug_state.refresh_counter

    def _compute_human_fuzzy_debug(self, idx: int, context: str, session_steps: int, observations):
        inputs = self.following_fuzzy_engine.clip_inputs(
            following_time=float(session_steps) * float(self.dt),
            hhd=resolve_fuzzy_metric_input(
                rolling_mean_value=float(observations.nearest_human_distance_mean_1s[idx]),
                current_value=float(observations.nearest_human_distance[idx]),
            ),
            hrd=resolve_fuzzy_metric_input(
                rolling_mean_value=float(observations.human_robot_distance_mean_1s[idx]),
                current_value=float(observations.human_robot_distance[idx]),
            ),
            density=float(observations.local_crowding_count_1m[idx]),
        )
        result = self.following_fuzzy_engine.compute(**inputs, context=context)
        return {"inputs": inputs, "result": result}

    def _record_fuzzy_debug(self, idx: int, context: str, fuzzy_debug: dict) -> None:
        debug_state = self.fuzzy_debug[idx]
        debug_state.context = str(context)
        debug_state.inputs = dict(fuzzy_debug["inputs"])
        debug_state.scores = {
            state: float(fuzzy_debug["result"][state])
            for state in ("overwhelmed", "distracted", "impatient", "engaged")
        }
        debug_state.dominant_state = str(fuzzy_debug["result"]["dominant_state"])
        debug_state.refresh_counter = int(self.runtime_cache.refresh_counter)

    def _apply_fuzzy_transition(self, human, idx: int, context: str, fuzzy_result: dict, world_frame) -> None:
        dominant_state = fuzzy_result["dominant_state"]
        if dominant_state == "engaged":
            return

        if context == "listening":
            recovery_mode = HumanMode.LISTENING
            distracted_source = DISTRACTED_SOURCE_LISTENING
            distracted_reason = "fuzzy_listening_distracted"
        else:
            recovery_mode = HumanMode.FOLLOWING
            distracted_source = DISTRACTED_SOURCE_FOLLOWING
            distracted_reason = "fuzzy_following_distracted"

        if dominant_state == "distracted":
            human.distracted_source = distracted_source
            human.distracted_recovery_mode = recovery_mode
            human.set_mode(HumanMode.DISTRACTED, reason=distracted_reason)
            if context == "listening":
                self._log_event(f">>> {human.name} became DISTRACTED while listening!")
                if not self.listening_state.interrupted:
                    self.listening_state.pause()
                    self.robot.listen_mode = False
            else:
                self._log_event(f">>> {human.name} became DISTRACTED!")
            return

        if dominant_state == "impatient":
            if recovery_mode == HumanMode.LISTENING:
                human.start_impatient(recovery_mode=HumanMode.LISTENING)
            else:
                human.start_impatient(
                    robot_pose=world_frame.robot_pose,
                    index=idx,
                    n_humans=self.n_humans,
                    recovery_mode=HumanMode.FOLLOWING,
                )
            return

        if dominant_state == "overwhelmed":
            current_xy = np.array(human.get_pose(self.data)[:2], dtype=np.float32)
            human.start_overwhelmed(
                robot_xy=np.array(world_frame.robot_xy, dtype=np.float32),
                current_xy=current_xy,
                recovery_mode=recovery_mode,
            )

    @staticmethod
    def _scalar_cross_2d(a_xy, b_xy) -> float:
        return float(a_xy[0] * b_xy[1] - a_xy[1] * b_xy[0])

    def _build_post_explanation_yield_target(self, current_xy, robot_xy, outbound_dir):
        current_xy = np.array(current_xy, dtype=np.float32)
        robot_xy = np.array(robot_xy, dtype=np.float32)
        outbound_dir = np.array(outbound_dir, dtype=np.float32)
        diff = current_xy - robot_xy
        dist_to_robot = float(np.linalg.norm(diff))
        if dist_to_robot > 1e-6:
            away_dir = diff / dist_to_robot
        else:
            away_dir = -outbound_dir
            away_norm = float(np.linalg.norm(away_dir))
            away_dir = np.array([1.0, 0.0], dtype=np.float32) if away_norm <= 1e-6 else away_dir / away_norm

        left_perp = np.array([-outbound_dir[1], outbound_dir[0]], dtype=np.float32)
        left_norm = float(np.linalg.norm(left_perp))
        left_perp = np.array([0.0, 1.0], dtype=np.float32) if left_norm <= 1e-6 else left_perp / left_norm
        right_perp = -left_perp

        side_sign = self._scalar_cross_2d(diff, outbound_dir)
        preferred_lateral = left_perp if side_sign >= 0.0 else right_perp
        fallback_lateral = right_perp if side_sign >= 0.0 else left_perp
        candidate_dirs = (away_dir, preferred_lateral, fallback_lateral)

        current_clearance = abs(self._scalar_cross_2d(diff, outbound_dir))
        best_target = current_xy.copy()
        best_score = 0.0
        for direction in candidate_dirs:
            candidate_xy = current_xy + float(POST_EXPLANATION_YIELD_DISTANCE) * np.asarray(direction, dtype=np.float32)
            move_vec = candidate_xy - current_xy
            move_dist = float(np.linalg.norm(move_vec))
            if move_dist <= 0.02:
                continue

            new_diff = candidate_xy - robot_xy
            new_dist = float(np.linalg.norm(new_diff))
            new_clearance = abs(self._scalar_cross_2d(new_diff, outbound_dir))
            score = (new_dist - dist_to_robot) + 0.5 * (new_clearance - current_clearance)
            if score > best_score:
                best_score = score
                best_target = np.array(candidate_xy, dtype=np.float32)

        return best_target

    def _start_post_explanation_hold(self, robot_xy, robot_yaw: float, human_xy) -> None:
        robot_xy = np.array(robot_xy, dtype=np.float32)
        goal_xy = np.array(self.robot.get_current_waypoint(), dtype=np.float32)
        outbound_vec = goal_xy - robot_xy
        outbound_norm = float(np.linalg.norm(outbound_vec))
        if outbound_norm <= 1e-6:
            outbound_dir = np.array([np.cos(robot_yaw), np.sin(robot_yaw)], dtype=np.float32)
            fallback_norm = float(np.linalg.norm(outbound_dir))
            outbound_dir = np.array([1.0, 0.0], dtype=np.float32) if fallback_norm <= 1e-6 else outbound_dir / fallback_norm
        else:
            outbound_dir = outbound_vec / outbound_norm

        self.post_explanation_state.active = True
        self.post_explanation_state.robot_start_xy = robot_xy.copy()
        self.post_explanation_state.anchor_robot_xy = robot_xy.copy()
        self.post_explanation_state.anchor_robot_yaw = float(robot_yaw)

        human_xy = np.asarray(human_xy, dtype=np.float32)
        n_humans = len(self.humans)
        targets = np.zeros((n_humans, 2), dtype=np.float32)
        listen_radii = np.zeros((n_humans,), dtype=np.float32)
        roles = [POST_EXPLANATION_ROLE_WAIT] * n_humans
        half_width = 0.5 * float(POST_EXPLANATION_YIELD_CORRIDOR_WIDTH)
        for idx, human in enumerate(self.humans):
            current_xy = (
                np.array(human_xy[idx], dtype=np.float32)
                if idx < human_xy.shape[0]
                else np.array(self.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
            )
            diff = current_xy - robot_xy
            dist_to_robot = float(np.linalg.norm(diff))
            forward = float(np.dot(diff, outbound_dir))
            lateral = abs(self._scalar_cross_2d(diff, outbound_dir))
            should_yield = bool(
                dist_to_robot <= float(POST_EXPLANATION_YIELD_CLOSE_DISTANCE)
                or (forward >= 0.0 and lateral <= half_width)
            )

            listen_radii[idx] = max(float(np.linalg.norm(diff)), self.listen_stand_threshold)
            if should_yield:
                roles[idx] = POST_EXPLANATION_ROLE_YIELD
                targets[idx] = self._build_post_explanation_yield_target(
                    current_xy=current_xy,
                    robot_xy=robot_xy,
                    outbound_dir=outbound_dir,
                )
            else:
                targets[idx] = current_xy

        self.post_explanation_state.roles = roles
        self.post_explanation_state.targets = targets
        self.post_explanation_state.listen_radii = listen_radii

    def _maybe_finish_post_explanation_hold(self, robot_xy, robot_speed: float) -> None:
        if (not self.post_explanation_state.active) or self.post_explanation_state.robot_start_xy is None:
            return
        moved_dist = float(
            np.linalg.norm(np.asarray(robot_xy, dtype=np.float32) - self.post_explanation_state.robot_start_xy)
        )
        if (
            robot_speed >= float(POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD)
            and moved_dist >= float(POST_EXPLANATION_HOLD_RESUME_DISTANCE)
        ):
            self.post_explanation_state.reset()
            self.follow_phase = FOLLOW_PHASE_TRANSIT

    def _maybe_activate_follow_phase_from_robot_progress(self, robot_xy) -> None:
        if (
            self.post_explanation_state.active
            or self.robot.listen_mode
            or self.follow_phase is not None
            or self.robot.callback_active
            or self.listening_state.interrupted
        ):
            return
        if self.robot_start_xy is None:
            return

        moved_dist = float(
            np.linalg.norm(
                np.asarray(robot_xy, dtype=np.float32) - np.asarray(self.robot_start_xy, dtype=np.float32)
            )
        )
        if moved_dist < self.human_follow_distance:
            return

        next_phase = FOLLOW_PHASE_TRANSIT if self.robot.listen_done else FOLLOW_PHASE_PRE_LISTEN_ENGAGE
        self.follow_phase = next_phase

    def _update_human_listening_session_progress(self) -> None:
        if not self.listening_state.fuzzy_active:
            return
        for human in self.humans:
            human.update_listening_session_progress(active=(human.mode == HumanMode.LISTENING))

    def _update_robot_emotion(self, events: StepEvents, analysis: dict) -> None:
        self.perceived_distracted_indices = list(analysis["perceived_distracted_indices"])
        emotion_modes = list(analysis["emotion_modes"])
        if (
            self.robot.callback_active
            and self.robot.callback_phase == RobotCallbackPhase.CUE
            and self.robot.callback_cue_elapsed_steps < self.robot.callback_cue_total_steps
        ):
            emotion_modes.append(HumanMode.DISTRACTED)

        happy_before = int(self.robot.happy_hold_steps_remaining)
        sad_now = any(mode in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED) for mode in emotion_modes)
        self.robot.update_emotion(emotion_modes)
        if happy_before > 0 and self.robot.happy_hold_steps_remaining == 0 and not sad_now:
            events.happy_completed = True

        self._sync_robot_speaker_state()
        self._sync_robot_visual_state()

    def _apply_general_phase_strategy(self, human, idx: int, world_frame, robot_action) -> np.ndarray:
        repulsion_vec = world_frame.repulsion_vectors[idx] if idx < len(world_frame.repulsion_vectors) else np.zeros(2, dtype=np.float32)
        if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
            human.set_mode(HumanMode.FOLLOWING if self.follow_phase is not None else HumanMode.WANDERING)

        human.update_following_duration(
            eligible_following=self.follow_phase is not None and human.mode == HumanMode.FOLLOWING
        )
        if human.mode == HumanMode.FOLLOWING and self._should_evaluate_fuzzy(idx, context="following"):
            fuzzy_debug = self._compute_human_fuzzy_debug(
                idx=idx,
                context="following",
                session_steps=int(human.following_steps),
                observations=world_frame.observations,
            )
            self._record_fuzzy_debug(idx, context="following", fuzzy_debug=fuzzy_debug)
            self._apply_fuzzy_transition(
                human,
                idx=idx,
                context="following",
                fuzzy_result=fuzzy_debug["result"],
                world_frame=world_frame,
            )

        ctx = {
            "index": idx,
            "n_humans": len(self.humans),
            "robot_pose": world_frame.robot_pose,
            "robot_xy": world_frame.robot_xy,
            "robot_yaw": world_frame.robot_pose[2],
            "robot_speed": float(np.hypot(robot_action[0], robot_action[1])),
            "repulsion": repulsion_vec,
            "follow_radius": FOLLOW_RADIUS_DEFAULT,
            "fan_half_angle": self.follow_fan_half_angle,
            "impatient_front_offset": human.impatient_front_offset,
            "listen_radius": self.listen_fan_radius,
            "stand_threshold": self.listen_stand_threshold,
            "listening_sector_half_angle": self.listen_front_sector_half_angle,
            "dt": self.dt,
        }
        return human.step(self.model, self.data, ctx)

    def _apply_listening_phase_strategy(self, human, idx: int, world_frame) -> np.ndarray:
        repulsion_vec = world_frame.repulsion_vectors[idx] if idx < len(world_frame.repulsion_vectors) else np.zeros(2, dtype=np.float32)
        if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
            human.set_mode(HumanMode.LISTENING)

        if human.mode == HumanMode.LISTENING and self._should_evaluate_fuzzy(idx, context="listening"):
            fuzzy_debug = self._compute_human_fuzzy_debug(
                idx=idx,
                context="listening",
                session_steps=int(human.listening_steps),
                observations=world_frame.observations,
            )
            self._record_fuzzy_debug(idx, context="listening", fuzzy_debug=fuzzy_debug)
            self._apply_fuzzy_transition(
                human,
                idx=idx,
                context="listening",
                fuzzy_result=fuzzy_debug["result"],
                world_frame=world_frame,
            )

        ctx = {
            "robot_xy": world_frame.robot_xy,
            "robot_yaw": world_frame.robot_pose[2],
            "robot_speed": 0.0,
            "repulsion": LISTENING_REPULSION_SCALE * repulsion_vec,
            "listen_radius": self.listen_fan_radius,
            "stand_threshold": self.listen_stand_threshold,
            "listening_sector_half_angle": self.listen_front_sector_half_angle,
            "dt": self.dt,
        }
        return human.step(self.model, self.data, ctx)

    def _apply_post_explanation_phase_strategy(self, human, idx: int, world_frame, robot_action) -> np.ndarray:
        repulsion_vec = world_frame.repulsion_vectors[idx] if idx < len(world_frame.repulsion_vectors) else np.zeros(2, dtype=np.float32)
        role = (
            self.post_explanation_state.roles[idx]
            if idx < len(self.post_explanation_state.roles)
            else POST_EXPLANATION_ROLE_WAIT
        )
        target_xy = (
            np.array(self.post_explanation_state.targets[idx], dtype=np.float32)
            if idx < len(self.post_explanation_state.targets)
            else np.array(human.get_pose(self.data)[:2], dtype=np.float32)
        )
        anchor_robot_xy = (
            np.array(self.post_explanation_state.anchor_robot_xy, dtype=np.float32)
            if self.post_explanation_state.anchor_robot_xy is not None
            else np.array(world_frame.robot_xy, dtype=np.float32)
        )
        anchor_robot_yaw = float(self.post_explanation_state.anchor_robot_yaw)

        move_ctx = {
            "robot_xy": world_frame.robot_xy,
            "robot_yaw": world_frame.robot_pose[2],
            "robot_speed": float(np.hypot(robot_action[0], robot_action[1])),
            "repulsion": repulsion_vec,
            "dt": self.dt,
        }
        if human.mode in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
            return human.step(self.model, self.data, move_ctx)

        if role == POST_EXPLANATION_ROLE_YIELD:
            human.set_mode(HumanMode.FOLLOWING)
            human.current_waypoint = target_xy.copy()
            return human.step_following(self.data, move_ctx, human.get_pose(self.data))

        human.set_mode(HumanMode.LISTENING)
        listen_ctx = {
            "robot_xy": world_frame.robot_xy,
            "robot_yaw": world_frame.robot_pose[2],
            "robot_speed": 0.0,
            "repulsion": LISTENING_REPULSION_SCALE * repulsion_vec,
            "listen_radius": (
                self.post_explanation_state.listen_radii[idx]
                if idx < len(self.post_explanation_state.listen_radii)
                else self.listen_fan_radius
            ),
            "stand_threshold": self.listen_stand_threshold,
            "listening_sector_half_angle": self.listen_front_sector_half_angle,
            "dt": self.dt,
        }
        return human.step_listening_with_anchor_target_and_live_repulsion(
            self.data,
            listen_ctx,
            human.get_pose(self.data),
            anchor_robot_xy=anchor_robot_xy,
            anchor_robot_yaw=anchor_robot_yaw,
            live_robot_xy=world_frame.robot_xy,
        )

    def _apply_human_controls(self, world_frame, robot_action) -> np.ndarray:
        human_actions = np.zeros((len(self.humans), 3), dtype=np.float32)
        for idx, human in enumerate(self.humans):
            if self.post_explanation_state.active:
                human_action = self._apply_post_explanation_phase_strategy(
                    human,
                    idx,
                    world_frame,
                    robot_action,
                )
            elif self.listening_state.controller_active:
                human_action = self._apply_listening_phase_strategy(human, idx, world_frame)
            else:
                human_action = self._apply_general_phase_strategy(human, idx, world_frame, robot_action)

            human_actions[idx] = human_action
            ctrl_idx = 3 + idx * 3
            self.data.ctrl[ctrl_idx : ctrl_idx + 3] = human_action

        return human_actions

    def _progress_listening_phase(self, events: StepEvents, world_frame) -> None:
        if self.listening_state.phase == LISTEN_PHASE_INTRO:
            self.listening_state.counter += 1
            if self.listening_state.counter >= self.listen_intro_delay_steps:
                is_final = self.listening_state.is_final
                self.listening_state.enter_wait(is_final=is_final)
                events.started_listen_wait = True
                self._log_event(
                    f">>> Listening explanation started after {self.listen_intro_delay_seconds:.1f}s delay."
                )
            return

        if self.listening_state.phase != LISTEN_PHASE_WAIT:
            return

        self.listening_state.counter += 1
        if self.listening_state.counter < self.listen_wait_steps:
            return

        events.completed_listen_wait = True
        if self.listening_state.is_final:
            events.final_listen_ready = True
            self._log_event(">>> Listening wait complete at final display.")
            self.listening_state.enter_idle()
            for human in self.humans:
                human.reset_listening_session_state()
            return

        self.robot.on_listening_complete()
        self.follow_phase = None
        self.robot_start_xy = np.array(world_frame.robot_xy, dtype=np.float32)
        self._start_post_explanation_hold(
            robot_xy=world_frame.robot_xy,
            robot_yaw=world_frame.robot_pose[2],
            human_xy=world_frame.human_xy,
        )
        self.listening_state.enter_idle()
        for human in self.humans:
            human.reset_listening_session_state()
        self._log_event(">>> Listening wait complete. Resume MOVE to Room B.")

    def reset(self, seed=None, options=None):
        del options
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.robot.reset()
        self.follow_phase = None
        self.listening_state.reset()
        self.post_explanation_state.reset()
        self.callback_state.reset(len(self.humans))
        self.runtime_cache.reset()
        self.fuzzy_debug = build_fuzzy_debug_states(len(self.humans))
        self.perceived_distracted_indices = []
        self.hh_distance_metric.reset()
        self.hr_distance_metric.reset()

        for human in self.humans:
            human.reset_episode_state()
        self._configure_human_behaviors()

        initial_robot_xy = np.array([float(self.data.xpos[self.robot_body_id, 0]), float(self.data.xpos[self.robot_body_id, 1])], dtype=np.float32)
        self._reset_human_positions(initial_robot_xy)
        mujoco.mj_forward(self.model, self.data)

        self.robot_start_xy = np.array(
            [float(self.data.xpos[self.robot_body_id, 0]), float(self.data.xpos[self.robot_body_id, 1])],
            dtype=np.float32,
        )
        for human in self.humans:
            human.reset_listening_session_state()

        world_frame = build_world_frame(
            data=self.data,
            robot_body_id=self.robot_body_id,
            humans=self.humans,
            human_body_ids=self.human_body_ids,
            cache=self.runtime_cache,
            hh_distance_metric=self.hh_distance_metric,
            hr_distance_metric=self.hr_distance_metric,
            observation_update_period_steps=self.observation_update_period_steps,
            social_distance=self.social_distance,
            repulsion_gain=self.repulsion_gain,
            force_observations=True,
        )
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state(force=True)
        gx, gy = self.robot.get_current_waypoint()
        x, y = world_frame.robot_xy
        return np.array([x, y, gx - x, gy - y], dtype=np.float32), {}

    def step(self, action=None):
        del action
        self.step_count += 1
        events = StepEvents()

        pre_frame = build_world_frame(
            data=self.data,
            robot_body_id=self.robot_body_id,
            humans=self.humans,
            human_body_ids=self.human_body_ids,
            cache=self.runtime_cache,
            hh_distance_metric=self.hh_distance_metric,
            hr_distance_metric=self.hr_distance_metric,
            observation_update_period_steps=self.observation_update_period_steps,
            social_distance=self.social_distance,
            repulsion_gain=self.repulsion_gain,
        )
        for idx, human in enumerate(self.humans):
            if human.mode != HumanMode.DISTRACTED:
                self.callback_state.triggered_for_distracted[idx] = False

        waiting_listen_hold = self.listening_state.phase == LISTEN_PHASE_WAIT and not self.robot.callback_active
        robot_action = np.zeros(3, dtype=np.float32)
        if waiting_listen_hold:
            self.robot.mode = RobotMode.STOP
        else:
            analysis = self._analyze_humans(pre_frame)
            target_idx = analysis["callback_target_idx"]
            if target_idx is not None:
                if self.listening_state.controller_active and not self.listening_state.interrupted:
                    self.listening_state.pause()
                    self.robot.listen_mode = False
                self.robot.start_callback(
                    target_idx=target_idx,
                    target_xy=pre_frame.human_xy[target_idx],
                    cue_steps=self._callback_cue_steps,
                )
                events.callback_triggered = True
                self.callback_state.success_mode = (
                    HumanMode.LISTENING if self.listening_state.controller_active else HumanMode.FOLLOWING
                )
                self.callback_state.active_target_idx = target_idx
                self.callback_state.triggered_for_distracted[target_idx] = True
                self._log_event(f">>> Robot CALLBACK triggered for person{target_idx + 1}.")

            robot_out = self.robot.step(
                robot_pose=pre_frame.robot_pose,
                human_xyz=pre_frame.human_xyz,
            )
            robot_action = np.array(robot_out["action"], dtype=np.float32)

            self._maybe_sample_active_callback_response(events)
            self._resolve_completed_callback(events)

            if robot_out["enter_listen"]:
                events.entered_listen = True
                self.follow_phase = None
                self.listening_state.enter_intro(
                    is_final=self.robot.is_final_reached(float(robot_out["dist"]))
                )
                for human in self.humans:
                    human.reset_listening_session_state()
                self.post_explanation_state.reset()
                rx, ry, ryaw = pre_frame.robot_pose
                self._log_event(
                    f">>> Robot entering LISTEN mode. robot=({rx:.2f}, {ry:.2f}, yaw={ryaw:.2f}); "
                    f"silent 3s preparation started while humans regulate to a "
                    f"{self.listen_fan_radius:.2f}m ring inside the front 160 deg sector."
                )

        self._maybe_finish_post_explanation_hold(
            pre_frame.robot_xy,
            float(np.hypot(robot_action[0], robot_action[1])),
        )
        self._maybe_activate_follow_phase_from_robot_progress(pre_frame.robot_xy)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = robot_action
        self._apply_human_controls(pre_frame, robot_action)

        mujoco.mj_step(self.model, self.data)
        post_frame = build_world_frame(
            data=self.data,
            robot_body_id=self.robot_body_id,
            humans=self.humans,
            human_body_ids=self.human_body_ids,
            cache=self.runtime_cache,
            hh_distance_metric=self.hh_distance_metric,
            hr_distance_metric=self.hr_distance_metric,
            observation_update_period_steps=self.observation_update_period_steps,
            social_distance=self.social_distance,
            repulsion_gain=self.repulsion_gain,
            tick_age_before_refresh=True,
        )

        self._update_human_listening_session_progress()
        self._progress_listening_phase(events, post_frame)

        human_goals = build_human_goals(
            self.humans,
            post_frame.robot_xy,
            self.post_explanation_state,
        )
        reached_goal_indices = compute_reached_goal_indices(
            humans=self.humans,
            human_xy=post_frame.human_xy,
            human_goals=human_goals,
            post_explanation_state=self.post_explanation_state,
            robot_xy=post_frame.robot_xy,
            robot_yaw=post_frame.robot_pose[2],
            listen_reached_min_distance=LISTEN_REACHED_MIN_DISTANCE,
            human_goal_threshold=HUMAN_GOAL_THRESHOLD,
            listening_sector_half_angle=self.listen_front_sector_half_angle,
        )

        analysis_after = self._analyze_humans(post_frame)
        self._update_robot_emotion(events, analysis_after)

        terminated = bool(events.final_listen_ready)
        truncated = bool(self.step_count >= self.max_steps)
        terminated_reason = None
        if terminated:
            terminated_reason = "final_listen_ready"
        elif truncated:
            terminated_reason = "max_steps"

        info = build_step_info(
            events=events,
            step_count=self.step_count,
            follow_phase=self.follow_phase,
            listen_phase=self.listening_state.phase,
            robot=self.robot,
            terminated_reason=terminated_reason,
            world_frame=post_frame,
            robot_action=robot_action,
            human_goals=human_goals,
            humans=self.humans,
            reached_goal_indices=reached_goal_indices,
            perceived_distracted_indices=self.perceived_distracted_indices,
        )
        reward = -float(info["robot"]["dist_to_goal"])
        gx, gy = self.robot.get_current_waypoint()
        x, y = post_frame.robot_xy
        return np.array([x, y, gx - x, gy - y], dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            apply_label_scene_option_to_viewer(viewer=self.viewer, label_scene_option=self._label_scene_option)
            self.viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self.renderer is None:
                vis = getattr(self.model, "vis", None)
                global_vis = getattr(vis, "global_", None) if vis is not None else None
                offwidth = int(getattr(global_vis, "offwidth", 640))
                offheight = int(getattr(global_vis, "offheight", 480))
                width = max(1, min(self.render_width, offwidth))
                height = max(1, min(self.render_height, offheight))
                try:
                    self.renderer = mujoco.Renderer(self.model, height=height, width=width)
                except ValueError:
                    self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, scene_option=self._label_scene_option)
            return self.renderer.render()

        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
