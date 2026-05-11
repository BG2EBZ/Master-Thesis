import logging
from importlib import resources
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from .env_reporting import (
    HUMAN_SPEAKING_HALO_RGBA_OFF,
    HUMAN_SPEAKING_HALO_RGBA_ON,
    apply_label_scene_option_to_viewer,
    apply_robot_visual_state,
    build_label_scene_option,
    build_step_info,
    resolve_robot_visual_state,
)
from .env_runtime import (
    build_human_goals,
    build_world_frame,
    compute_reached_goal_indices,
    resolve_fuzzy_metric_input,
)
from .env_state import (
    FOLLOW_PHASE_PRE_LISTEN_ENGAGE,
    FOLLOW_PHASE_TRANSIT,
    LISTEN_PHASE_INTRO,
    LISTEN_PHASE_PAUSED,
    LISTEN_PHASE_WAIT,
    LISTEN_QUESTION_COMPLETION_FINISH_WAIT,
    LISTEN_QUESTION_COMPLETION_RESUME_WAIT,
    LISTEN_QUESTION_PHASE_ANSWER,
    LISTEN_QUESTION_PHASE_NONE,
    LISTEN_QUESTION_PHASE_TURN_BACK,
    LISTEN_QUESTION_PHASE_TURN_TO_HUMAN,
    LISTEN_QUESTION_TIMING_MID_RANDOM,
    LISTEN_QUESTION_TIMING_POST_WAIT,
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
    Human,
    HumanMode,
    HumanProfile,
)
from .map_layouts import DEFAULT_MUSEUM_LAYOUT, MapLayout, get_map_layout
from .metrics import VectorizedRollingWindow
from .robot import (
    Robot,
    RobotMode,
    RobotSpeechMode,
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
IMPATIENT_FAN_HALF_ANGLE_DEG = 30.0
LISTENING_FRONT_SECTOR_HALF_ANGLE_DEG = 70.0
LISTEN_FAN_RADIUS_DEFAULT = 1.0
LISTEN_STAND_THRESHOLD_DEFAULT = 0.05
LISTENING_REPULSION_SCALE = 1.0
LISTEN_REACHED_MIN_DISTANCE = 0.8
LISTEN_INTRO_DELAY_SECONDS_DEFAULT = 3.0
LISTEN_WAIT_SECONDS_DEFAULT = 25.0
LISTEN_QUESTION_PROBABILITY_DEFAULT = 1.0
LISTEN_QUESTION_AFTER_EXPLANATION_PROBABILITY_DEFAULT = 0.50
LISTEN_QUESTION_PAUSE_SECONDS_DEFAULT = 5.0
LISTEN_QUESTION_TURN_YAW_RATE = 1.0
LISTEN_QUESTION_TURN_DONE_YAW_ERR = 0.02
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
FOLLOWING_SLOWDOWN_DISTANCE_THRESHOLD_METERS = 2.5
FOLLOWING_CALLBACK_DISTANCE_THRESHOLD_METERS = 3.5
FOLLOWING_CALLBACK_WAIT_SECONDS = 3.0
FOLLOWING_CALLBACK_CUE_SECONDS = 2.0
FOLLOWING_SLOWDOWN_SPEED_SCALE = 0.7

MAX_HUMANS_CAPACITY = 15
HUMAN_SPAWN_MIN_DISTANCE = (2.0 * HUMAN_WALL_FOOTPRINT_RADIUS) + 0.10
HUMAN_SPAWN_MIN_ROBOT_DISTANCE = SOCIAL_DISTANCE_DEFAULT
HUMAN_SPAWN_MAX_ATTEMPTS_PER_HUMAN = 2000

INACTIVE_HUMAN_PARK_X = 50.0
INACTIVE_HUMAN_PARK_Y_BASE = 50.0

CALLBACK_TRIGGER_DISTANCE_METERS_DEFAULT = 2.0
CALLBACK_REJOIN_PROB_NORMAL_DEFAULT = 0.80
CALLBACK_IGNORE_PROB_NORMAL_DEFAULT = 0.20
CALLBACK_REJOIN_PROB_ND_DEFAULT = 0.40
CALLBACK_IGNORE_PROB_ND_DEFAULT = 0.60


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
        listen_question_probability: float = LISTEN_QUESTION_PROBABILITY_DEFAULT,
        listen_question_after_explanation_probability: float = (
            LISTEN_QUESTION_AFTER_EXPLANATION_PROBABILITY_DEFAULT
        ),
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
        self.impatient_fan_half_angle = np.deg2rad(IMPATIENT_FAN_HALF_ANGLE_DEG)
        self.listen_front_sector_half_angle = np.deg2rad(LISTENING_FRONT_SECTOR_HALF_ANGLE_DEG)
        self.listen_fan_radius = LISTEN_FAN_RADIUS_DEFAULT
        self.listen_stand_threshold = LISTEN_STAND_THRESHOLD_DEFAULT
        self.listen_intro_delay_seconds = LISTEN_INTRO_DELAY_SECONDS_DEFAULT
        self.listen_intro_delay_steps = max(1, int(round(self.listen_intro_delay_seconds / self.dt)))
        self.listen_wait_seconds = LISTEN_WAIT_SECONDS_DEFAULT
        self.listen_wait_steps = max(1, int(round(self.listen_wait_seconds / self.dt)))
        self.listen_question_probability = float(listen_question_probability)
        self.listen_question_after_explanation_probability = float(listen_question_after_explanation_probability)
        self.listen_question_pause_seconds = float(LISTEN_QUESTION_PAUSE_SECONDS_DEFAULT)
        self.listen_question_pause_steps = max(1, int(round(self.listen_question_pause_seconds / self.dt)))
        self.following_callback_wait_steps = max(
            1,
            int(round(FOLLOWING_CALLBACK_WAIT_SECONDS / self.dt)),
        )
        self.following_callback_cue_steps = max(
            1,
            int(round(FOLLOWING_CALLBACK_CUE_SECONDS / self.dt)),
        )

        self.listening_state = ListeningState()
        self.post_explanation_state = PostExplanationState()
        self.callback_state = CallbackState()
        self.runtime_cache = RuntimeCache()
        self.max_distracted_duration_seconds = float(max_distracted_duration_seconds)
        self.callback_trigger_distance_meters = float(callback_trigger_distance_meters)
        self._last_following_slowdown_active = False
        self._last_following_wait_active = False
        self._last_following_callback_active = False
        self._last_following_max_hr_distance = 0.0
        self._following_wait_elapsed_steps = 0
        self._following_wait_callback_triggered = False
        self._following_wait_callback_target_idx = None
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
            human.enable_event_logs = self.enable_event_logs

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
        self.all_human_speaking_halo_geom_ids = [
            self.model.geom(f"{human.body_name}_speaking_halo").id for human in self.all_humans
        ]

        self._label_scene_option = build_label_scene_option()
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state(force=True)
        self._sync_human_visual_state()

    def _log_event(self, msg: str) -> None:
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

    def _configure_human_behaviors(self) -> None:
        for human in self.humans:
            human.max_distracted_duration_seconds = self.max_distracted_duration_seconds
            human.distracted_duration = round(human.max_distracted_duration_seconds / self.dt)
            human.distracted_stop_duration = round(human.distracted_stop_duration_seconds / self.dt)
            human.nd_distracted_stop_and_go_stop_steps = round(
                human.nd_distracted_stop_and_go_stop_seconds / self.dt
            )
            human.nd_distracted_stop_and_go_move_steps = round(
                human.nd_distracted_stop_and_go_move_seconds / self.dt
            )
            human.impatient_duration = round(6.0 / self.dt)
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
        if self.listening_state.phase == LISTEN_PHASE_WAIT:
            self.robot.set_speech_mode(RobotSpeechMode.EXPLANATION)
            return
        if (
            self.listening_state.phase == LISTEN_PHASE_PAUSED
            and self.listening_state.question_phase == LISTEN_QUESTION_PHASE_ANSWER
        ):
            self.robot.set_speech_mode(RobotSpeechMode.ANSWER)
            return
        self.robot.set_speech_mode(RobotSpeechMode.NONE)

    def _build_question_turn_action(self, current_yaw: float, target_yaw: float) -> np.ndarray:
        yaw_err = self.robot._wrap_to_pi(float(target_yaw) - float(current_yaw))
        action = np.zeros(3, dtype=np.float32)
        if abs(yaw_err) < float(LISTEN_QUESTION_TURN_DONE_YAW_ERR):
            return action

        max_yaw_delta = float(LISTEN_QUESTION_TURN_YAW_RATE) * float(self.dt)
        if abs(yaw_err) <= max_yaw_delta:
            action[2] = float(yaw_err / float(self.dt))
        else:
            action[2] = float(np.sign(yaw_err) * float(LISTEN_QUESTION_TURN_YAW_RATE))
        return action

    def _set_listening_question_human_speaking(self, active: bool) -> None:
        idx = self.listening_state.question_human_idx
        if idx is None:
            return
        if 0 <= int(idx) < len(self.humans):
            self.humans[int(idx)].speaking_active = bool(active)

    def _get_listening_hold_robot_action(self, world_frame) -> np.ndarray:
        action = np.zeros(3, dtype=np.float32)
        self.robot.mode = RobotMode.STOP
        if self.listening_state.phase != LISTEN_PHASE_PAUSED:
            return action

        question_phase = self.listening_state.question_phase
        if question_phase == LISTEN_QUESTION_PHASE_TURN_TO_HUMAN:
            idx = self.listening_state.question_human_idx
            if idx is None or not (0 <= int(idx) < len(self.humans)):
                return action
            target_xy = np.asarray(world_frame.human_xy[int(idx)], dtype=np.float32)
            robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
            desired_yaw = float(np.arctan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0]))
            return self._build_question_turn_action(world_frame.robot_pose[2], desired_yaw)

        if question_phase == LISTEN_QUESTION_PHASE_TURN_BACK:
            target_yaw = self.listening_state.question_return_yaw
            if target_yaw is None:
                return action
            return self._build_question_turn_action(world_frame.robot_pose[2], float(target_yaw))

        return action

    def _sync_robot_visual_state(self, force: bool = False) -> None:
        visual_state = resolve_robot_visual_state(
            robot=self.robot,
            callback_visual_active=bool(self._last_following_callback_active),
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

    def _sync_human_visual_state(self) -> None:
        active_count = len(self.humans)
        for idx, geom_id in enumerate(self.all_human_speaking_halo_geom_ids):
            human = self.all_humans[idx]
            halo_rgba = (
                HUMAN_SPEAKING_HALO_RGBA_ON
                if idx < active_count and bool(human.speaking_active)
                else HUMAN_SPEAKING_HALO_RGBA_OFF
            )
            self.model.geom_rgba[geom_id] = halo_rgba

    def _get_current_human_modes(self) -> list[str]:
        return [human.mode for human in self.humans]

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
        human = self.humans[idx]
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
        result = self.following_fuzzy_engine.compute(
            **inputs,
            context=context,
            profile=human.profile,
        )
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

    def _apply_fuzzy_transition(
        self,
        human,
        idx: int,
        context: str,
        fuzzy_result: dict,
        fuzzy_inputs: dict,
        world_frame,
    ) -> None:
        dominant_state = fuzzy_result["dominant_state"]
        if dominant_state == "engaged":
            return

        if context == "listening":
            recovery_mode = HumanMode.LISTENING
            distracted_source = DISTRACTED_SOURCE_LISTENING
        else:
            recovery_mode = HumanMode.FOLLOWING
            distracted_source = DISTRACTED_SOURCE_FOLLOWING
        fuzzy_inputs_log = (
            f"\n    >>> fuzzy_inputs: "
            f"following_time={float(fuzzy_inputs['following_time']):.1f}, "
            f"hhd={float(fuzzy_inputs['hhd']):.2f}, "
            f"hrd={float(fuzzy_inputs['hrd']):.2f}, "
            f"density={float(fuzzy_inputs['density']):.1f}"
        )

        if dominant_state == "distracted":
            human.distracted_source = distracted_source
            human.distracted_recovery_mode = recovery_mode
            human.set_mode(HumanMode.DISTRACTED)
            if context == "listening":
                self._log_event(
                    f">>> {human.name} became DISTRACTED while listening!"
                    f"{fuzzy_inputs_log}"
                )
            else:
                self._log_event(
                    f">>> {human.name} became DISTRACTED!"
                    f"{fuzzy_inputs_log}"
                )
            return

        if dominant_state == "impatient":
            if recovery_mode == HumanMode.LISTENING:
                human.start_impatient(recovery_mode=HumanMode.LISTENING)
                self._log_event(
                    f">>> {human.name} became IMPATIENT while listening!"
                    f"{fuzzy_inputs_log}"
                )
            else:
                human.start_impatient(recovery_mode=HumanMode.FOLLOWING)
                self._log_event(
                    f">>> {human.name} became IMPATIENT!"
                    f"{fuzzy_inputs_log}"
                )
            return

        if dominant_state == "overwhelmed":
            current_xy = np.array(human.get_pose(self.data)[:2], dtype=np.float32)
            human.start_overwhelmed(
                robot_xy=np.array(world_frame.robot_xy, dtype=np.float32),
                current_xy=current_xy,
                recovery_mode=recovery_mode,
            )
            if context == "listening":
                self._log_event(
                    f">>> {human.name} became OVERWHELMED while listening!"
                    f"{fuzzy_inputs_log}"
                )
            else:
                self._log_event(
                    f">>> {human.name} became OVERWHELMED!"
                    f"{fuzzy_inputs_log}"
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

    def _reset_following_crowd_regulation_debug_state(self) -> None:
        self._last_following_slowdown_active = False
        self._last_following_wait_active = False
        self._last_following_callback_active = False
        self._last_following_max_hr_distance = 0.0

    def _reset_following_wait_episode(self) -> None:
        self._following_wait_elapsed_steps = 0
        self._following_wait_callback_triggered = False
        self._following_wait_callback_target_idx = None

    def _reset_following_crowd_regulation_runtime_state(self) -> None:
        self._reset_following_wait_episode()

    def _record_following_max_hr_distance(self, world_frame) -> np.ndarray:
        distances = np.asarray(world_frame.observations.human_robot_distance, dtype=np.float32)
        self._last_following_max_hr_distance = float(np.max(distances)) if distances.size != 0 else 0.0
        return distances

    def _select_following_callback_target(self, world_frame, distances) -> tuple[int | None, np.ndarray | None]:
        distances = np.asarray(distances, dtype=np.float32)
        human_xy = np.asarray(world_frame.human_xy, dtype=np.float32)
        if distances.size == 0 or human_xy.shape[0] == 0:
            return None, None
        target_idx = int(np.argmax(distances))
        if not (0 <= target_idx < human_xy.shape[0]):
            return None, None
        return target_idx, np.asarray(human_xy[target_idx, :2], dtype=np.float32)

    def _start_following_callback(self, world_frame) -> bool:
        distances = self._record_following_max_hr_distance(world_frame)
        target_idx, target_xy = self._select_following_callback_target(world_frame, distances)
        if target_idx is None or target_xy is None:
            return False
        self.robot.start_callback(
            target_idx=target_idx,
            target_xy=target_xy,
            cue_steps=int(self.following_callback_cue_steps),
        )
        self._following_wait_callback_triggered = True
        self._following_wait_callback_target_idx = int(target_idx)
        return True

    def _finish_following_callback(self) -> None:
        self.robot.finish_callback()
        self._following_wait_callback_target_idx = None

    def _apply_following_crowd_regulation_if_needed(
        self,
        robot_action,
        robot_mode: str,
        world_frame,
    ) -> tuple[np.ndarray, bool]:
        adjusted_action = np.array(robot_action, dtype=np.float32, copy=True)
        distances = self._record_following_max_hr_distance(world_frame)
        max_hr_distance = self._last_following_max_hr_distance
        should_start_callback = False
        if self.follow_phase != FOLLOW_PHASE_TRANSIT:
            self._reset_following_wait_episode()
            return adjusted_action, should_start_callback
        if str(robot_mode) != RobotMode.MOVE:
            return adjusted_action, should_start_callback
        if distances.size == 0:
            self._reset_following_wait_episode()
            return adjusted_action, should_start_callback
        if max_hr_distance > float(FOLLOWING_CALLBACK_DISTANCE_THRESHOLD_METERS):
            self._last_following_wait_active = True
            self.robot.mode = RobotMode.STOP
            self._following_wait_elapsed_steps += 1
            if (
                (not self._following_wait_callback_triggered)
                and self._following_wait_elapsed_steps >= int(self.following_callback_wait_steps)
            ):
                should_start_callback = True
            return np.zeros(3, dtype=np.float32), should_start_callback
        self._reset_following_wait_episode()
        if max_hr_distance <= float(FOLLOWING_SLOWDOWN_DISTANCE_THRESHOLD_METERS):
            return adjusted_action, should_start_callback

        self._last_following_slowdown_active = True
        adjusted_action[:2] *= float(FOLLOWING_SLOWDOWN_SPEED_SCALE)
        return adjusted_action, should_start_callback

    def _update_human_listening_session_progress(self) -> None:
        if not self.listening_state.fuzzy_active:
            return
        for human in self.humans:
            human.update_listening_session_progress(active=(human.mode == HumanMode.LISTENING))

    def _prepare_listening_question_plan(self) -> None:
        if self.listening_state.phase != LISTEN_PHASE_WAIT:
            return

        state = self.listening_state
        if state.session_has_question or state.question_timing_mode is not None or state.question_fired:
            return

        state.session_has_question = (float(self.np_random.random()) < float(self.listen_question_probability))
        if not state.session_has_question:
            state.question_fired = True
            return

        if (float(self.np_random.random()) < float(self.listen_question_after_explanation_probability)):
            state.question_timing_mode = LISTEN_QUESTION_TIMING_POST_WAIT
            return

        start_step = max(1, int(self.listen_wait_steps) // 2)
        end_step = int(self.listen_wait_steps) - 1
        if start_step > end_step:
            state.session_has_question = False
            return

        state.question_timing_mode = LISTEN_QUESTION_TIMING_MID_RANDOM
        state.question_trigger_step = int(self.np_random.integers(start_step, end_step + 1))

    # Recover speaking human to normal listening behavior
    def _clear_listening_question_humans(self) -> None:
        self._set_listening_question_human_speaking(False)
        self.listening_state.clear_active_question()

    # Attempt to start a listening question
    def _maybe_start_listening_question(self, events: StepEvents, timing_mode: str, world_frame) -> bool:
        if self.listening_state.phase != LISTEN_PHASE_WAIT:
            return False
        if not self.listening_state.session_has_question or self.listening_state.question_fired:
            return False
        if self.listening_state.question_timing_mode != timing_mode:
            return False

        if timing_mode == LISTEN_QUESTION_TIMING_MID_RANDOM:
            trigger_step = self.listening_state.question_trigger_step
            if trigger_step is None or self.listening_state.counter < int(trigger_step):
                return False

        candidate_indices = [
            idx for idx, human in enumerate(self.humans) if human.mode == HumanMode.LISTENING
        ]
        if not candidate_indices:
            self.listening_state.question_fired = True
            self._log_event(">>> Listening question skipped: no LISTENING human available.")
            return False

        question_human_idx = int(
            candidate_indices[int(self.np_random.integers(0, len(candidate_indices)))]
        )
        self.listening_state.pause()
        self.listening_state.question_human_idx = question_human_idx
        self.listening_state.question_phase = LISTEN_QUESTION_PHASE_TURN_TO_HUMAN
        self.listening_state.question_ask_steps_remaining = int(self.listen_question_pause_steps)
        self.listening_state.question_return_yaw = float(world_frame.robot_pose[2])
        self.listening_state.question_completion_mode = (
            LISTEN_QUESTION_COMPLETION_FINISH_WAIT
            if timing_mode == LISTEN_QUESTION_TIMING_POST_WAIT
            else LISTEN_QUESTION_COMPLETION_RESUME_WAIT
        )
        self.listening_state.question_fired = True
        self._set_listening_question_human_speaking(True)
        events.question_started = True
        self._log_event(
            f">>> Listening question started ({timing_mode}) by person{question_human_idx + 1}."
        )
        return True

    def _finish_listening_wait(self, events: StepEvents, world_frame) -> None:
        events.completed_listen_wait = True
        if self.listening_state.is_final:
            events.final_listen_ready = True
            self._log_event(">>> Listening wait complete at final display.")
            self._clear_listening_question_humans()
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
        self._clear_listening_question_humans()
        self.listening_state.enter_idle()
        for human in self.humans:
            human.reset_listening_session_state()
        self._log_event(">>> Listening wait complete. Resume MOVE to Room B.")

    def _progress_listening_question_pause(self, events: StepEvents, world_frame) -> bool:
        if self.listening_state.phase != LISTEN_PHASE_PAUSED:
            return False
        if not self.listening_state.question_active:
            return False

        question_phase = self.listening_state.question_phase
        if question_phase == LISTEN_QUESTION_PHASE_TURN_TO_HUMAN:
            idx = self.listening_state.question_human_idx
            if idx is None or not (0 <= int(idx) < len(self.humans)):
                self._clear_listening_question_humans()
                self.listening_state.resume()
                return True

            self.listening_state.question_ask_steps_remaining = max(
                0,
                int(self.listening_state.question_ask_steps_remaining) - 1,
            )
            target_xy = np.asarray(world_frame.human_xy[int(idx)], dtype=np.float32)
            robot_xy = np.asarray(world_frame.robot_xy, dtype=np.float32)
            desired_yaw = float(np.arctan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0]))
            yaw_err = self.robot._wrap_to_pi(desired_yaw - float(world_frame.robot_pose[2]))
            if (
                self.listening_state.question_ask_steps_remaining > 0
                or abs(yaw_err) >= float(LISTEN_QUESTION_TURN_DONE_YAW_ERR)
            ):
                return True

            self._set_listening_question_human_speaking(False)
            self.listening_state.question_phase = LISTEN_QUESTION_PHASE_ANSWER
            self.listening_state.question_answer_steps_remaining = int(self.listen_question_pause_steps)
            return True

        if question_phase == LISTEN_QUESTION_PHASE_ANSWER:
            self.listening_state.question_answer_steps_remaining = max(
                0,
                int(self.listening_state.question_answer_steps_remaining) - 1,
            )
            if self.listening_state.question_answer_steps_remaining > 0:
                return True

            if (
                self.listening_state.question_completion_mode
                == LISTEN_QUESTION_COMPLETION_FINISH_WAIT
            ):
                self._clear_listening_question_humans()
                self.listening_state.is_final = bool(self.listening_state.paused_is_final)
                self.listening_state.counter = int(self.listening_state.paused_counter)
                events.question_completed = True
                self._log_event(">>> Listening question completed.")
                self._finish_listening_wait(events, world_frame)
                return True

            self.listening_state.question_phase = LISTEN_QUESTION_PHASE_TURN_BACK
            return True

        if question_phase == LISTEN_QUESTION_PHASE_TURN_BACK:
            target_yaw = self.listening_state.question_return_yaw
            if target_yaw is None:
                self._clear_listening_question_humans()
                self.listening_state.resume()
                events.question_completed = True
                self._log_event(">>> Listening question completed.")
                return True

            yaw_err = self.robot._wrap_to_pi(float(target_yaw) - float(world_frame.robot_pose[2]))
            if abs(yaw_err) >= float(LISTEN_QUESTION_TURN_DONE_YAW_ERR):
                return True

            self._clear_listening_question_humans()
            self.listening_state.resume()
            events.question_completed = True
            self._log_event(">>> Listening question completed.")
            if (
                self.listening_state.phase == LISTEN_PHASE_WAIT
                and self.listening_state.counter >= int(self.listen_wait_steps)
            ):
                self._finish_listening_wait(events, world_frame)
            return True

        if question_phase == LISTEN_QUESTION_PHASE_NONE:
            return False

        return True

    def _apply_general_phase_strategy(self, human, idx: int, world_frame) -> np.ndarray:
        repulsion_vec = world_frame.repulsion_vectors[idx] if idx < len(world_frame.repulsion_vectors) else np.zeros(2, dtype=np.float32)
        if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
            human.set_mode(HumanMode.FOLLOWING if self.follow_phase is not None else HumanMode.WANDERING)
        current_human_modes = self._get_current_human_modes()

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
                fuzzy_inputs=fuzzy_debug["inputs"],
                world_frame=world_frame,
            )

        ctx = {
            "index": idx,
            "n_humans": len(self.humans),
            "robot_pose": world_frame.robot_pose,
            "robot_xy": world_frame.robot_xy,
            "human_xy": world_frame.human_xy,
            "human_modes": current_human_modes,
            "repulsion": repulsion_vec,
            "follow_radius": FOLLOW_RADIUS_DEFAULT,
            "fan_half_angle": self.follow_fan_half_angle,
            "impatient_fan_half_angle": self.impatient_fan_half_angle,
            "impatient_front_offset": human.impatient_front_offset,
            "impatient_recovery_mode": (
                HumanMode.FOLLOWING if self.follow_phase is not None else HumanMode.WANDERING
            ),
        }
        return human.step(self.model, self.data, ctx)

    def _apply_listening_phase_strategy(self, human, idx: int, world_frame) -> np.ndarray:
        repulsion_vec = world_frame.repulsion_vectors[idx] if idx < len(world_frame.repulsion_vectors) else np.zeros(2, dtype=np.float32)
        if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
            human.set_mode(HumanMode.LISTENING)
        current_human_modes = self._get_current_human_modes()
        effective_robot_yaw = float(world_frame.robot_pose[2])
        if self.listening_state.question_active and self.listening_state.question_return_yaw is not None:
            effective_robot_yaw = float(self.listening_state.question_return_yaw)

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
                fuzzy_inputs=fuzzy_debug["inputs"],
                world_frame=world_frame,
            )

        ctx = {
            "index": idx,
            "n_humans": len(self.humans),
            "robot_pose": world_frame.robot_pose,
            "robot_xy": world_frame.robot_xy,
            "robot_yaw": effective_robot_yaw,
            "human_xy": world_frame.human_xy,
            "human_modes": current_human_modes,
            "repulsion": LISTENING_REPULSION_SCALE * repulsion_vec,
            "fan_half_angle": self.follow_fan_half_angle,
            "impatient_fan_half_angle": self.impatient_fan_half_angle,
            "impatient_front_offset": human.impatient_front_offset,
            "impatient_recovery_mode": HumanMode.LISTENING,
            "listen_radius": self.listen_fan_radius,
            "listening_sector_half_angle": self.listen_front_sector_half_angle,
        }
        return human.step(self.model, self.data, ctx)

    def _apply_post_explanation_phase_strategy(self, human, idx: int, world_frame) -> np.ndarray:
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
        current_human_modes = self._get_current_human_modes()

        move_ctx = {
            "index": idx,
            "n_humans": len(self.humans),
            "robot_pose": world_frame.robot_pose,
            "robot_xy": world_frame.robot_xy,
            "human_xy": world_frame.human_xy,
            "human_modes": current_human_modes,
            "repulsion": repulsion_vec,
            "fan_half_angle": self.follow_fan_half_angle,
            "impatient_fan_half_angle": self.impatient_fan_half_angle,
            "impatient_front_offset": human.impatient_front_offset,
            "impatient_recovery_mode": (
                HumanMode.FOLLOWING if role == POST_EXPLANATION_ROLE_YIELD else HumanMode.LISTENING
            ),
        }
        if human.mode in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
            return human.step(self.model, self.data, move_ctx)

        if role == POST_EXPLANATION_ROLE_YIELD:
            human.set_mode(HumanMode.FOLLOWING)
            yield_ctx = {
                **move_ctx,
                "behavior_kind": "post_explanation_yield",
                "target_xy": target_xy.copy(),
            }
            return human.step(self.model, self.data, yield_ctx)

        human.set_mode(HumanMode.LISTENING)
        listen_ctx = {
            "behavior_kind": "post_explanation_listening_anchor",
            "robot_xy": world_frame.robot_xy,
            "robot_yaw": world_frame.robot_pose[2],
            "repulsion": LISTENING_REPULSION_SCALE * repulsion_vec,
            "listen_radius": (
                self.post_explanation_state.listen_radii[idx]
                if idx < len(self.post_explanation_state.listen_radii)
                else self.listen_fan_radius
            ),
            "listening_sector_half_angle": self.listen_front_sector_half_angle,
            "anchor_robot_xy": anchor_robot_xy,
            "anchor_robot_yaw": anchor_robot_yaw,
            "live_robot_xy": world_frame.robot_xy,
        }
        return human.step(self.model, self.data, listen_ctx)

    def _apply_human_controls(self, world_frame) -> np.ndarray:
        human_actions = np.zeros((len(self.humans), 3), dtype=np.float32)
        for idx, human in enumerate(self.humans):
            if self.post_explanation_state.active:
                human_action = self._apply_post_explanation_phase_strategy(
                    human,
                    idx,
                    world_frame,
                )
            elif self.listening_state.controller_active:
                human_action = self._apply_listening_phase_strategy(human, idx, world_frame)
            else:
                human_action = self._apply_general_phase_strategy(human, idx, world_frame)

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
                self._prepare_listening_question_plan()
                events.started_listen_wait = True
                self._log_event(
                    f">>> Listening explanation started after {self.listen_intro_delay_seconds:.1f}s delay."
                )
            return

        if self._progress_listening_question_pause(events, world_frame):
            return

        if self.listening_state.phase != LISTEN_PHASE_WAIT:
            return

        self.listening_state.counter += 1
        if self._maybe_start_listening_question(events, LISTEN_QUESTION_TIMING_MID_RANDOM, world_frame):
            return

        if self.listening_state.counter < self.listen_wait_steps:
            return

        if self._maybe_start_listening_question(events, LISTEN_QUESTION_TIMING_POST_WAIT, world_frame):
            return

        self._finish_listening_wait(events, world_frame)

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
        self.hh_distance_metric.reset()
        self.hr_distance_metric.reset()
        self._reset_following_crowd_regulation_debug_state()
        self._reset_following_crowd_regulation_runtime_state()

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
        self._sync_human_visual_state()
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
        waiting_listen_hold = self.listening_state.phase in (LISTEN_PHASE_WAIT, LISTEN_PHASE_PAUSED)
        self._reset_following_crowd_regulation_debug_state()
        robot_action = np.zeros(3, dtype=np.float32)
        if waiting_listen_hold:
            robot_action = self._get_listening_hold_robot_action(pre_frame)
        else:
            was_listening = bool(self.robot.listen_mode)
            robot_action = np.array(
                self.robot.step(
                    robot_pose=pre_frame.robot_pose,
                    human_xyz=pre_frame.human_xyz,
                ),
                dtype=np.float32,
            )
            if str(self.robot.mode) == RobotMode.CALLBACK:
                self._record_following_max_hr_distance(pre_frame)
                self._last_following_callback_active = True
                if bool(self.robot.callback_cue_completed_this_step):
                    events.callback_completed = True
                    target_idx = self.robot.callback_target_idx
                    person_label = "none" if target_idx is None else f"person{int(target_idx) + 1}"
                    self._log_event(f">>> Robot callback completed for {person_label}.")
                    self._finish_following_callback()
            else:
                robot_action, should_start_callback = self._apply_following_crowd_regulation_if_needed(
                    robot_action,
                    robot_mode=self.robot.mode,
                    world_frame=pre_frame,
                )
                if should_start_callback and self._start_following_callback(pre_frame):
                    events.callback_triggered = True
                    self._last_following_callback_active = True
                    target_idx = self.robot.callback_target_idx
                    person_label = "none" if target_idx is None else f"person{int(target_idx) + 1}"
                    self._log_event(
                        f">>> Robot callback triggered for {person_label} after "
                        f"{self.following_callback_wait_steps * self.dt:.1f}s wait."
                    )
                    robot_action = np.array(
                        self.robot.step(
                            robot_pose=pre_frame.robot_pose,
                            human_xyz=pre_frame.human_xyz,
                        ),
                        dtype=np.float32,
                    )
                    self._last_following_callback_active = (str(self.robot.mode) == RobotMode.CALLBACK)

            if (not was_listening) and bool(self.robot.listen_mode):
                events.entered_listen = True
                self.follow_phase = None
                self._clear_listening_question_humans()
                self.listening_state.enter_intro(
                    is_final=self.robot.is_final_reached(pre_frame.robot_pose)
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
        self._apply_human_controls(pre_frame)

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
        self._sync_human_visual_state()

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

        human_modes = [human.mode for human in self.humans]
        mode_array = np.asarray(human_modes, dtype=object)
        perceived_distracted_indices = [int(idx) for idx in np.flatnonzero(mode_array == HumanMode.DISTRACTED)]
        self.robot.update_emotion(human_modes)
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state()

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
            perceived_distracted_indices=perceived_distracted_indices,
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
