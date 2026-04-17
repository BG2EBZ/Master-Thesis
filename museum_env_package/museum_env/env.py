import logging
from dataclasses import dataclass, field
from importlib import resources
from typing import List, Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

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
    RobotEmotion,
    RobotMode,
)

# MuJoCo + Gym wrapper coordinating robot policy, human behaviors, and event bookkeeping.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


@dataclass
class ListenPhaseState:
    """Track listen intro-delay and wait-phase runtime state."""

    intro_active: bool = False
    intro_counter: int = 0
    intro_is_final: bool = False
    wait_active: bool = False
    wait_counter: int = 0
    wait_is_final: bool = False

    def reset(self):
        self.intro_active = False
        self.intro_counter = 0
        self.intro_is_final = False
        self.wait_active = False
        self.wait_counter = 0
        self.wait_is_final = False


@dataclass
class PostExplanationState:
    """Track the short hold/yield window after a non-final explanation."""

    active: bool = False
    robot_start_xy: Optional[np.ndarray] = None
    anchor_robot_xy: Optional[np.ndarray] = None
    anchor_robot_yaw: float = 0.0
    roles: List[str] = field(default_factory=list)
    targets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    listen_radii: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )

    def reset(self):
        self.active = False
        self.robot_start_xy = None
        self.anchor_robot_xy = None
        self.anchor_robot_yaw = 0.0
        self.roles.clear()
        self.targets = np.zeros((0, 2), dtype=np.float32)
        self.listen_radii = np.zeros((0,), dtype=np.float32)


@dataclass
class EnvCallbackState:
    """Track callback bookkeeping and listening interruption state."""

    triggered_for_distracted: List[bool] = field(default_factory=list)
    active_target_idx: Optional[int] = None
    last_response: Optional[str] = None
    last_response_target_idx: Optional[int] = None
    pending_listening_request: Optional[dict] = None
    paused_listening_state: Optional[dict] = None
    success_mode: str = HumanMode.FOLLOWING

    def reset(self, n_humans: int):
        self.triggered_for_distracted = [False] * int(n_humans)
        self.active_target_idx = None
        self.last_response = None
        self.last_response_target_idx = None
        self.pending_listening_request = None
        self.paused_listening_state = None
        self.success_mode = HumanMode.FOLLOWING

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
HUMAN_MAX_SPEED_DEFAULT = 1.00
FOLLOW_RADIUS_DEFAULT = 1.0
HUMAN_GOAL_THRESHOLD = 0.1
POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD = 0.5
POST_EXPLANATION_HOLD_RESUME_DISTANCE = 2.5
POST_EXPLANATION_YIELD_CORRIDOR_WIDTH = 0.8
POST_EXPLANATION_YIELD_CLOSE_DISTANCE = 1.1
POST_EXPLANATION_YIELD_DISTANCE = 0.5
POST_EXPLANATION_YIELD_ROLE_WAIT = "wait"
POST_EXPLANATION_YIELD_ROLE_YIELD = "yield"
DIST_EPS = 1e-8
HUMAN_HUMAN_DISTANCE_WINDOW_SECONDS = 1.0
LOCAL_CROWDING_RADIUS_METERS = 1.0
MAX_DISTRACTED_DURATION_SECONDS_DEFAULT = 15.0

MAX_HUMANS_CAPACITY = 15
HUMAN_SPAWN_MIN_DISTANCE = (2.0 * HUMAN_WALL_FOOTPRINT_RADIUS) + 0.10
HUMAN_SPAWN_MIN_ROBOT_DISTANCE = SOCIAL_DISTANCE_DEFAULT
HUMAN_SPAWN_MAX_ATTEMPTS_PER_HUMAN = 2000

INACTIVE_HUMAN_PARK_X = 50.0
INACTIVE_HUMAN_PARK_Y_BASE = 50.0
HUMAN_LABEL_SITE_GROUP = 2
ROBOT_EXPLANATION_LABEL_GROUP = 3
ROBOT_FOLLOWME_LABEL_GROUP = 4
HUMAN_LABEL_MODE = mujoco.mjtLabel.mjLABEL_SITE
CALLBACK_CUE_SECONDS = 3.0
CALLBACK_RESPONSE_SAMPLE_SECONDS = 2.0
CALLBACK_TRIGGER_DISTANCE_METERS_DEFAULT = 2.0
CALLBACK_REJOIN_PROB_NORMAL_DEFAULT = 0.80
CALLBACK_IGNORE_PROB_NORMAL_DEFAULT = 0.20
CALLBACK_REJOIN_PROB_ND_DEFAULT = 0.40
CALLBACK_IGNORE_PROB_ND_DEFAULT = 0.60
FOLLOW_PHASE_PRE_LISTEN_ENGAGE = "pre_listen_engage"
FOLLOW_PHASE_TRANSIT = "transit_follow"
ROBOT_HAPPY_HOLD_SECONDS = 3.0
ROBOT_COLOR_NATURAL = np.array([0.85, 0.85, 0.85, 1.0], dtype=np.float32)
ROBOT_COLOR_SAD = np.array([0.20, 0.45, 0.95, 1.0], dtype=np.float32)
ROBOT_COLOR_HAPPY = np.array([0.95, 0.85, 0.20, 1.0], dtype=np.float32)
SPEAKING_HALO_RGBA_ON = np.array([1.0, 0.9, 0.2, 0.35], dtype=np.float32)
SPEAKING_HALO_RGBA_OFF = np.array([1.0, 0.9, 0.2, 0.0], dtype=np.float32)


class MuseumEnv(gym.Env):
    """
    Gymnasium environment for the museum scenario.
    It owns global orchestration: robot decision, human updates, and rich debug/info outputs.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path=None,
        map_name: str = DEFAULT_MUSEUM_LAYOUT.name,
        map_layout: Optional[MapLayout] = None,
        render_mode=None,
        enable_event_logs: bool = True,
        strict_action_validation: bool = True,
        max_distracted_duration_seconds: float = MAX_DISTRACTED_DURATION_SECONDS_DEFAULT,
        callback_rejoin_prob_normal: float = CALLBACK_REJOIN_PROB_NORMAL_DEFAULT,
        callback_ignore_prob_normal: float = CALLBACK_IGNORE_PROB_NORMAL_DEFAULT,
        callback_rejoin_prob_nd: float = CALLBACK_REJOIN_PROB_ND_DEFAULT,
        callback_ignore_prob_nd: float = CALLBACK_IGNORE_PROB_ND_DEFAULT,
        callback_trigger_distance_meters: float = CALLBACK_TRIGGER_DISTANCE_METERS_DEFAULT,
        observation_update_period_seconds: float = 0.1,
        n_humans: int = 15, # number of active humans
    ):
        """Initialize MuJoCo scene, agents, behavior parameters and runtime state."""
        super().__init__()
        self.enable_event_logs = enable_event_logs
        self.strict_action_validation = strict_action_validation
        logger.setLevel(logging.INFO if self.enable_event_logs else logging.CRITICAL + 1)

        if map_layout is not None:
            self.map_layout = map_layout
        else:
            self.map_layout = get_map_layout(map_name)
        self.map_name = self.map_layout.name

        if xml_path is None:
            with resources.path("museum_env.assets", self.map_layout.default_xml_asset) as xml_file:
                xml_path = str(xml_file)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self._viewer_key_callback = None
        self.render_width = 1920
        self.render_height = 1080

        # --- Observation space ---
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.timestep = self.model.opt.timestep
        self.timestep_float = self.timestep
        self.observation_update_period_seconds = observation_update_period_seconds
        if (not np.isfinite(self.observation_update_period_seconds)) or (
            self.observation_update_period_seconds <= 0.0
        ):
            raise ValueError(
                "observation_update_period_seconds must be a finite positive value, "
                f"got {observation_update_period_seconds}"
            )
        self.observation_update_period_steps = max(
            1,
            int(round(self.observation_update_period_seconds / self.timestep_float)),
        )
        self.following_fuzzy_eval_period_seconds = self.observation_update_period_seconds
        self.following_fuzzy_eval_period_steps = int(self.observation_update_period_steps)
        self.listening_fuzzy_eval_period_seconds = self.observation_update_period_seconds
        self.listening_fuzzy_eval_period_steps = int(self.observation_update_period_steps)
        self.max_steps = MAX_STEPS_DEFAULT
        self.step_count = 0
        self._callback_cue_steps = max(1, int(round(CALLBACK_CUE_SECONDS / self.timestep_float)))
        self._callback_response_sample_steps = max(
            1,
            int(round(CALLBACK_RESPONSE_SAMPLE_SECONDS / self.timestep_float)),
        )

        # Robot agent
        self.robot = Robot(
            waypoints=self.map_layout.robot_waypoints,
            v_max=1.0,
            k_v=20.0,
            k_yaw=20.0,
        )

        # Human follow switch (start with random walking)
        self.follow_humans = False
        self.follow_phase = None
        self.robot_start_xy = None
        self.human_follow_distance = HUMAN_FOLLOW_DISTANCE_DEFAULT

        # Social distance (repulsion) parameters
        self.social_distance = SOCIAL_DISTANCE_DEFAULT
        self.repulsion_gain = REPULSION_GAIN_DEFAULT

        # Listening ring in front of the robot after it stops
        self.follow_fan_half_angle = np.deg2rad(FOLLOW_FAN_HALF_ANGLE_DEG)
        self.listen_front_sector_half_angle = np.deg2rad(LISTENING_FRONT_SECTOR_HALF_ANGLE_DEG)
        self.listen_fan_radius = LISTEN_FAN_RADIUS_DEFAULT
        self.listen_stand_threshold = LISTEN_STAND_THRESHOLD_DEFAULT

        # Listening intro delay + wait window
        self.listen_intro_delay_seconds = LISTEN_INTRO_DELAY_SECONDS_DEFAULT
        self.listen_intro_delay_steps = max(
            1,
            int(round(self.listen_intro_delay_seconds / self.timestep)),
        )
        self.listen_wait_seconds = LISTEN_WAIT_SECONDS_DEFAULT
        self.listen_wait_steps = max(1, int(round(self.listen_wait_seconds / self.timestep)))
        self.listen_state = ListenPhaseState()
        self.post_explanation_state = PostExplanationState()
        self.env_callback_state = EnvCallbackState()
        self.max_distracted_duration_seconds = max_distracted_duration_seconds
        if self.max_distracted_duration_seconds <= 0.0:
            raise ValueError(
                "max_distracted_duration_seconds must be > 0, "
                f"got {max_distracted_duration_seconds}"
            )
        self.callback_rejoin_prob_normal = callback_rejoin_prob_normal
        self.callback_ignore_prob_normal = callback_ignore_prob_normal
        self.callback_rejoin_prob_nd = callback_rejoin_prob_nd
        self.callback_ignore_prob_nd = callback_ignore_prob_nd
        self.callback_trigger_distance_meters = callback_trigger_distance_meters
        try:
            from .fuzzy import FollowingFuzzyEngine
        except ImportError as exc:
            raise ImportError(
                "MuseumEnv now requires the optional 'scikit-fuzzy' dependency "
                "because following-state triggers are fuzzy-only."
            ) from exc
        self.following_fuzzy_engine = FollowingFuzzyEngine()
        self.n_humans = int(n_humans)
        if self.n_humans < 1 or self.n_humans > MAX_HUMANS_CAPACITY:
            raise ValueError(
                f"n_humans must be in [1, {MAX_HUMANS_CAPACITY}], got {n_humans}"
            )
        if self.callback_trigger_distance_meters <= 0.0:
            raise ValueError(
                "callback_trigger_distance_meters must be > 0, "
                f"got {callback_trigger_distance_meters}"
            )
        self.callback_response_profile_probs = {
            HumanProfile.NORMAL: {
                "rejoin": self.callback_rejoin_prob_normal,
                "ignore": self.callback_ignore_prob_normal,
            },
            HumanProfile.NEURODIVERGENT: {
                "rejoin": self.callback_rejoin_prob_nd,
                "ignore": self.callback_ignore_prob_nd,
            },
        }
        self._cached_obs = None
        self._observation_sample_age_steps = 0
        self._observation_refresh_counter = 0
        self._following_fuzzy_evaluated_this_step = False
        self._listening_fuzzy_evaluated_this_step = False
        self.perceived_distracted_indices = []
        self._last_robot_base_visual_emotion = None
        self._last_robot_speaking_halo_active = None
        self._last_robot_label_visibility_state = None

        # --- Initialize humans ---
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
        self.humans = []
        for human in self.all_humans:
            human.set_mode(HumanMode.WANDERING)
            human.set_event_logging(self.enable_event_logs)
        self._set_active_humans(self.n_humans)
        self.nu = 3 + (3 * len(self.humans))
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(self.nu,),
            dtype=np.float32,
        )
        self.env_callback_state.reset(len(self.humans))
        self._reset_following_fuzzy_debug_state()

        self._configure_human_distracted_duration()
        self._configure_human_following_variants()

        # Set human parameters related to behaviors
        for human in self.humans:
            human.can_be_overwhelmed = True
            human.can_be_impatient = True
            human.impatient_duration = 2000
            human.impatient_speed_multiplier = 1.6
            human.impatient_front_offset = 1.0

        # Cache MuJoCo body ids (static across episodes).
        self.robot_body_id = self.model.body("robot").id
        self.robot_base_geom_id = self.model.geom("robot_base").id
        self.robot_speaking_halo_geom_id = self.model.geom("robot_speaking_halo").id
        self.all_human_body_ids = [self.model.body(human.body_name).id for human in self.all_humans]
        for human, body_id in zip(self.all_humans, self.all_human_body_ids):
            human.body_id = body_id
        self.human_body_ids = self.all_human_body_ids[: len(self.humans)]
        self._label_scene_option = self._build_label_scene_option()
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state(force=True)

    def _log_event(self, msg: str):
        """Emit environment-level log message when logging is enabled."""
        if self.enable_event_logs:
            logger.info(msg)

    def _reset_following_fuzzy_debug_state(self):
        n_humans = len(self.humans)
        scores = [
            {state: 0.0 for state in ("overwhelmed", "distracted", "impatient", "engaged")}
            for _ in range(n_humans)
        ]
        dominant_state = [None] * n_humans
        inputs = [
            {
                "following_time": 0.0,
                "hhd": 0.0,
                "hrd": 0.0,
                "density": 0.0,
            }
            for _ in range(n_humans)
        ]
        context = ["none"] * n_humans
        self.last_human_fuzzy_scores = scores
        self.last_human_fuzzy_dominant_state = dominant_state
        self.last_human_fuzzy_inputs = inputs
        self.last_human_fuzzy_context = context
        self.last_following_fuzzy_scores = scores
        self.last_following_fuzzy_dominant_state = dominant_state
        self.last_following_fuzzy_inputs = inputs
        self._following_fuzzy_last_eval_refresh_counter = [-1] * n_humans
        self._following_fuzzy_evaluated_this_step = False
        self._listening_fuzzy_evaluated_this_step = False

    def _reset_environment_observation_cache_state(self):
        self._cached_obs = None
        self._observation_sample_age_steps = 0
        self._observation_refresh_counter = 0

    def _tick_observation_sample_age(self):
        self._observation_sample_age_steps += 1

    def _refresh_environment_observations(self, human_xy, robot_xy, force: bool = False):
        should_refresh = force or self._cached_obs is None
        if (not should_refresh) and (
            self._observation_sample_age_steps >= self.observation_update_period_steps
        ):
            should_refresh = True
        if not should_refresh:
            return self._cached_obs

        pairwise_dist = self._compute_human_pairwise_distances(human_xy)
        nearest_human_distance = self._compute_nearest_human_distances_from_pairwise(pairwise_dist)
        local_crowding_count_1m = self._compute_local_crowding_count_1m_from_pairwise(pairwise_dist)
        human_robot_distance = self._compute_human_robot_distances(human_xy, robot_xy)
        nearest_human_distance_mean_1s = self.hh_distance_metric.update(nearest_human_distance)
        human_robot_distance_mean_1s = self.hr_distance_metric.update(human_robot_distance)

        self._cached_obs = {
            "nearest_human_distance": np.array(nearest_human_distance, dtype=np.float32),
            "local_crowding_count_1m": np.array(local_crowding_count_1m, dtype=np.int32),
            "human_robot_distance": np.array(human_robot_distance, dtype=np.float32),
            "nearest_human_distance_mean_1s": np.array(
                nearest_human_distance_mean_1s,
                dtype=np.float32,
            ),
            "human_robot_distance_mean_1s": np.array(
                human_robot_distance_mean_1s,
                dtype=np.float32,
            ),
        }
        self._observation_refresh_counter += 1
        self._observation_sample_age_steps = 0
        return self._cached_obs

    def _set_active_humans(self, n_humans: int):
        """Select the active prefix of humans and assign per-profile defaults."""
        self.humans = list(self.all_humans[: int(n_humans)])
        for human in self.all_humans:
            human.set_profile(HumanProfile.NORMAL)
        if self.humans:
            self.humans[0].set_profile(HumanProfile.NEURODIVERGENT)
        self._reset_following_fuzzy_debug_state()
        if hasattr(self, "all_human_body_ids"):
            self.human_body_ids = self.all_human_body_ids[: len(self.humans)]
        if hasattr(self, "timestep_float"):
            window_steps = max(
                1,
                int(
                    round(
                        HUMAN_HUMAN_DISTANCE_WINDOW_SECONDS
                        / self.observation_update_period_seconds
                    )
                ),
            )
            self.hh_distance_metric = VectorizedRollingWindow(window_steps, len(self.humans))
            self.hr_distance_metric = VectorizedRollingWindow(window_steps, len(self.humans))
        if hasattr(self, "env_callback_state"):
            self.env_callback_state.reset(len(self.humans))
        self._reset_environment_observation_cache_state()

    def _sample_active_human_spawn_states(self, robot_xy):
        """Sample collision-free initial poses for the active humans."""
        sampled_states = []
        sampled_positions = []
        for _ in self.humans:
            for _attempt in range(HUMAN_SPAWN_MAX_ATTEMPTS_PER_HUMAN):
                candidate_xy = self.map_layout.sample_spawn_point(HUMAN_WALL_FOOTPRINT_RADIUS, rng=self.np_random)
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
            else:
                raise RuntimeError(
                    f"Unable to sample non-overlapping spawn positions for {len(self.humans)} humans."
                )
        return sampled_states

    def _reset_human_positions(self, robot_xy):
        """Place active humans in spawn space and park inactive humans outside the map."""
        active_spawn_states = self._sample_active_human_spawn_states(robot_xy=robot_xy)
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

    def _configure_human_following_variants(self):
        """Apply profile-specific spacing parameters and distracted-motion timing to all humans."""
        for human in self.humans:
            human.set_profile(human.profile)
            human.listening_distracted_move_steps = max(
                1,
                int(round(LISTENING_DISTRACTED_MOVE_SECONDS / self.timestep)),
            )

    def _configure_human_distracted_duration(self):
        """Apply common distracted duration config to all humans."""
        for human in self.humans:
            human.max_distracted_duration_seconds = self.max_distracted_duration_seconds
            human.distracted_duration = max(
                1,
                int(round(human.max_distracted_duration_seconds / self.timestep)),
            )

    def _is_human_in_listening_front_sector(self, human, pos_xy, robot_xy, robot_yaw: float) -> bool:
        """Return whether one human position lies inside the robot-front listening sector."""
        return human.is_within_listening_front_sector(
            point_xy=np.array(pos_xy, dtype=np.float32),
            robot_xy=np.array(robot_xy, dtype=np.float32),
            robot_yaw=robot_yaw,
            sector_half_angle=self.listen_front_sector_half_angle,
        )

    def _reset_human_listening_session_states(self):
        """Clear per-session listening counters and flags."""
        for human in self.humans:
            human.reset_listening_session_state()

    def _set_follow_phase(self, phase):
        """Set high-level follow phase and keep follow_humans in sync."""
        if phase not in (None, FOLLOW_PHASE_PRE_LISTEN_ENGAGE, FOLLOW_PHASE_TRANSIT):
            raise ValueError(f"Unknown follow phase: {phase}")
        self.follow_phase = phase
        self.follow_humans = phase is not None

    def _clear_follow_phase(self):
        """Disable follow-like motion until the next engage/transit phase starts."""
        self._set_follow_phase(None)

    def _is_following_fuzzy_active(self) -> bool:
        """Return whether following fuzzy transitions are enabled this step."""
        return self.follow_phase == FOLLOW_PHASE_TRANSIT

    def _is_listening_session_active(self) -> bool:
        return (
            self.robot.listen_mode
            or self.listen_state.intro_active
            or self.listen_state.wait_active
        )

    def _is_listening_interrupted_for_callback(self) -> bool:
        return self.env_callback_state.paused_listening_state is not None

    def _is_listening_controller_active(self) -> bool:
        return self._is_listening_session_active() or self._is_listening_interrupted_for_callback()

    def _is_listening_fuzzy_active(self) -> bool:
        return self._is_listening_session_active() and (not self._is_listening_interrupted_for_callback())

    def _is_listening_callback_interruptible(self) -> bool:
        return (
            self._is_listening_controller_active()
            and (not self.robot.callback_active)
        )

    def _queue_listening_callback_request(self, target_idx: int, target_xy) -> None:
        if (
            self.env_callback_state.pending_listening_request is not None
            or self.robot.callback_active
            or (not self._is_listening_callback_interruptible())
        ):
            return
        self.env_callback_state.pending_listening_request = {
            "target_idx": int(target_idx),
            "target_xy": np.array(target_xy, dtype=np.float32),
            "cue_steps": int(self._get_callback_cue_steps()),
            "success_mode": HumanMode.LISTENING,
            "interrupts_listening": True,
        }

    def _pause_listening_for_callback(self) -> None:
        if self.env_callback_state.paused_listening_state is not None:
            return
        self.env_callback_state.paused_listening_state = {
            "listen_mode": self.robot.listen_mode,
            "listen_intro_delay_active": self.listen_state.intro_active,
            "listen_intro_delay_counter": self.listen_state.intro_counter,
            "listen_intro_delay_is_final": self.listen_state.intro_is_final,
            "listen_wait_active": self.listen_state.wait_active,
            "listen_wait_counter": self.listen_state.wait_counter,
            "listen_wait_is_final": self.listen_state.wait_is_final,
        }
        self.robot.listen_mode = False
        self.listen_state.intro_active = False
        self.listen_state.wait_active = False

    def _resume_listening_after_callback(self) -> None:
        if self.env_callback_state.paused_listening_state is None:
            return
        paused = dict(self.env_callback_state.paused_listening_state)
        self.robot.listen_mode = paused["listen_mode"]
        self.listen_state.intro_active = paused["listen_intro_delay_active"]
        self.listen_state.intro_counter = paused["listen_intro_delay_counter"]
        self.listen_state.intro_is_final = paused["listen_intro_delay_is_final"]
        self.listen_state.wait_active = paused["listen_wait_active"]
        self.listen_state.wait_counter = paused["listen_wait_counter"]
        self.listen_state.wait_is_final = paused["listen_wait_is_final"]
        self.env_callback_state.paused_listening_state = None

    def _should_evaluate_human_fuzzy(self, idx: int, phase: str) -> bool:
        """Evaluate one phase-specific fuzzy result once per refreshed observation sample."""
        if self.following_fuzzy_engine is None:
            return False
        if phase == "following":
            if not self._is_following_fuzzy_active():
                return False
        elif phase == "listening":
            if not self._is_listening_fuzzy_active():
                return False
        else:
            raise ValueError(f"Unknown fuzzy phase: {phase}")
        if not (0 <= idx < len(self.humans)):
            return False
        if idx >= len(self._following_fuzzy_last_eval_refresh_counter):
            return True
        if self.last_human_fuzzy_dominant_state[idx] is None:
            return True
        if self.last_human_fuzzy_context[idx] != phase:
            return True
        return self._observation_refresh_counter > self._following_fuzzy_last_eval_refresh_counter[idx]

    def _maybe_activate_follow_phase_from_robot_progress(self, robot_xy):
        """Start pre-listen engage or transit follow once robot moved far enough."""
        if (
            self.post_explanation_state.active
            or self.robot.listen_mode
            or self.follow_humans
            or self.robot.callback_active
            or self._is_listening_interrupted_for_callback()
        ):
            return
        if self.robot_start_xy is None:
            return

        moved_dist = np.linalg.norm(
            np.asarray(robot_xy, dtype=np.float32) - np.asarray(self.robot_start_xy, dtype=np.float32)
        )
        if moved_dist < self.human_follow_distance:
            return

        next_phase = FOLLOW_PHASE_TRANSIT if self.robot.listen_done else FOLLOW_PHASE_PRE_LISTEN_ENGAGE
        self._set_follow_phase(next_phase)

    @staticmethod
    def _scalar_cross_2d(a_xy, b_xy) -> float:
        """Return the scalar z-component of the 2D cross product."""
        return float(a_xy[0] * b_xy[1] - a_xy[1] * b_xy[0])

    def _build_post_explanation_yield_target(self, human, current_xy, robot_xy, outbound_dir):
        """Build one small yield target using away-first, side-fallback search."""
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
            if away_norm <= 1e-6:
                away_dir = np.array([1.0, 0.0], dtype=np.float32)
            else:
                away_dir = away_dir / away_norm

        left_perp = np.array([-outbound_dir[1], outbound_dir[0]], dtype=np.float32)
        left_norm = float(np.linalg.norm(left_perp))
        if left_norm <= 1e-6:
            left_perp = np.array([0.0, 1.0], dtype=np.float32)
        else:
            left_perp = left_perp / left_norm
        right_perp = -left_perp

        side_sign = self._scalar_cross_2d(diff, outbound_dir)
        preferred_lateral = left_perp if side_sign >= 0.0 else right_perp
        fallback_lateral = right_perp if side_sign >= 0.0 else left_perp
        candidate_dirs = (away_dir, preferred_lateral, fallback_lateral)

        current_clearance = abs(self._scalar_cross_2d(diff, outbound_dir))
        best_target = current_xy.copy()
        best_score = 0.0
        for direction in candidate_dirs:
            cand_xy = current_xy + float(POST_EXPLANATION_YIELD_DISTANCE) * np.array(direction, dtype=np.float32)
            safe_xy = np.array(cand_xy, dtype=np.float32)
            move_vec = safe_xy - current_xy
            move_dist = float(np.linalg.norm(move_vec))
            if move_dist <= 0.02:
                continue

            new_diff = safe_xy - robot_xy
            new_dist = float(np.linalg.norm(new_diff))
            new_clearance = abs(self._scalar_cross_2d(new_diff, outbound_dir))
            score = (new_dist - dist_to_robot) + 0.5 * (new_clearance - current_clearance)
            if score > best_score:
                best_score = score
                best_target = np.array(safe_xy, dtype=np.float32)

        return best_target

    def _start_post_explanation_hold(self, robot_xy, robot_yaw: float, human_xy):
        """Start one short transition window instead of falling back to wandering."""
        robot_xy = np.array(robot_xy, dtype=np.float32)
        goal_xy = np.array(self.robot.get_current_waypoint(), dtype=np.float32)
        outbound_vec = goal_xy - robot_xy
        outbound_norm = float(np.linalg.norm(outbound_vec))
        if outbound_norm <= 1e-6:
            outbound_dir = np.array([np.cos(robot_yaw), np.sin(robot_yaw)], dtype=np.float32)
            fallback_norm = float(np.linalg.norm(outbound_dir))
            if fallback_norm <= 1e-6:
                outbound_dir = np.array([1.0, 0.0], dtype=np.float32)
            else:
                outbound_dir = outbound_dir / fallback_norm
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
        roles = [POST_EXPLANATION_YIELD_ROLE_WAIT] * n_humans
        half_width = 0.5 * float(POST_EXPLANATION_YIELD_CORRIDOR_WIDTH)
        for idx, human in enumerate(self.humans):
            if idx < human_xy.shape[0]:
                current_xy = np.array(human_xy[idx], dtype=np.float32)
            else:
                current_xy = np.array(self.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
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
                roles[idx] = POST_EXPLANATION_YIELD_ROLE_YIELD
                targets[idx] = self._build_post_explanation_yield_target(
                    human=human,
                    current_xy=current_xy,
                    robot_xy=robot_xy,
                    outbound_dir=outbound_dir,
                )
            else:
                targets[idx] = current_xy

        self.post_explanation_state.roles = roles
        self.post_explanation_state.targets = targets
        self.post_explanation_state.listen_radii = listen_radii

    def _maybe_finish_post_explanation_hold(self, robot_xy, robot_speed: float):
        """Restore following once the robot has clearly resumed moving."""
        if (
            (not self.post_explanation_state.active)
            or self.post_explanation_state.robot_start_xy is None
        ):
            return
        moved_dist = float(
            np.linalg.norm(
                np.asarray(robot_xy, dtype=np.float32)
                - self.post_explanation_state.robot_start_xy
            )
        )
        if (
            robot_speed >= float(POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD)
            and moved_dist >= float(POST_EXPLANATION_HOLD_RESUME_DISTANCE)
        ):
            self.post_explanation_state.reset()
            self._set_follow_phase(FOLLOW_PHASE_TRANSIT)

    def _update_human_listening_session_progress(self):
        """Advance per-human listening counters while the listening session is active."""
        if not self._is_listening_fuzzy_active():
            return

        for human in self.humans:
            human.update_listening_session_progress(active=(human.mode == HumanMode.LISTENING))

    def _build_label_scene_option(self):
        """Create MuJoCo scene option object used to toggle site text labels."""
        opt = mujoco.MjvOption()
        opt.label = HUMAN_LABEL_MODE
        opt.sitegroup[:] = 0
        opt.sitegroup[HUMAN_LABEL_SITE_GROUP] = 1
        opt.sitegroup[ROBOT_EXPLANATION_LABEL_GROUP] = 0
        opt.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP] = 0
        return opt

    def _apply_label_options_to_viewer(self):
        """Push current label visibility options into live viewer."""
        if self.viewer is None:
            return
        self.viewer.opt.label = self._label_scene_option.label
        self.viewer.opt.sitegroup[:] = self._label_scene_option.sitegroup

    def set_viewer_key_callback(self, callback):
        """Register optional viewer keyboard callback."""
        if callback is not None and not callable(callback):
            raise ValueError("viewer key callback must be callable or None.")
        self._viewer_key_callback = callback

    @staticmethod
    def _default_events():
        """Create empty event flag dictionary for one environment step."""
        return {
            "entered_listen": False,
            "started_listen_wait": False,
            "completed_listen_wait": False,
            "final_listen_ready": False,
            "overwhelmed_triggered": False,
            "callback_triggered": False,
            "callback_completed": False,
            "callback_forced_recovery": False,
            "callback_response_rejoin": False,
            "callback_response_ignore": False,
            "callback_attempt_1_started": False,
            "callback_attempt_2_started": False,
            "callback_first_attempt_failed": False,
            "callback_success": False,
            "happy_triggered": False,
            "happy_completed": False,
        }

    def _validate_external_action(self, action):
        """Validate optional external action shape/type; return whether provided."""
        if action is None:
            return False
        if not self.strict_action_validation:
            return True
        if not isinstance(action, np.ndarray):
            raise ValueError("Expected action to be a numpy.ndarray when provided.")
        if action.shape != (self.nu,):
            raise ValueError(f"Expected action shape {(self.nu,)}, got {action.shape}.")
        if not np.issubdtype(action.dtype, np.number):
            raise ValueError(f"Expected numeric action dtype, got {action.dtype}.")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or Inf.")
        return True

    def _finalize_step_output(
        self,
        snapshot,
        events,
        dist,
        terminated,
        external_action_received,
        external_action_used=False,
    ):
        """Build Gym step tuple (obs, reward, terminated, truncated, info)."""
        obs = self._get_obs()
        reward = -float(dist)
        truncated = self.step_count >= self.max_steps
        info = self._build_info(
            snapshot=snapshot,
            events=events,
            truncated=truncated,
            external_action_received=external_action_received,
            external_action_used=external_action_used,
        )
        return obs, reward, terminated, truncated, info

    def _get_robot_pose(self):
        """Read robot pose (x, y, yaw) from MuJoCo data."""
        x = float(self.data.xpos[self.robot_body_id, 0])
        y = float(self.data.xpos[self.robot_body_id, 1])
        yaw = float(self.data.qpos[2])
        return x, y, yaw

    def _get_human_poses(self):
        """Read all human poses as array shaped [n_humans, 3]."""
        n_humans = len(self.humans)
        humans_xyz = np.empty((n_humans, 3), dtype=np.float32)
        for idx, (human, human_body_id) in enumerate(zip(self.humans, self.human_body_ids)):
            humans_xyz[idx, 0] = float(self.data.xpos[human_body_id, 0])
            humans_xyz[idx, 1] = float(self.data.xpos[human_body_id, 1])
            humans_xyz[idx, 2] = float(self.data.qpos[human.qpos_idx + 2])
        return humans_xyz

    def _get_goal_xy(self):
        """Return current robot waypoint coordinates."""
        goal_xy = self.robot.get_current_waypoint()
        return float(goal_xy[0]), float(goal_xy[1])

    def _is_robot_in_move_stage(self, robot_pose):
        """Return True only when robot is in nominal moving phase."""
        if self.robot.listen_mode or self.listen_state.wait_active or self.robot.callback_active:
            return False
        rx, ry, _ = robot_pose
        wx, wy = self.robot.get_current_waypoint()
        dist = float(np.hypot(wx - rx, wy - ry) + DIST_EPS)
        return dist >= ROBOT_WAYPOINT_REACHED_DIST

    def _refresh_callback_rearm_flags(self):
        """Re-arm callback eligibility once a human leaves distracted mode."""
        for idx, human in enumerate(self.humans):
            if (
                idx < len(self.env_callback_state.triggered_for_distracted)
                and human.mode != HumanMode.DISTRACTED
            ):
                self.env_callback_state.triggered_for_distracted[idx] = False

    def _analyze_human_state(self, robot_xy, human_xy):
        """Collect per-step human aggregates used by robot decision and diagnostics."""
        perceived_distracted_indices = []
        callback_target_idx = None
        threshold = self.callback_trigger_distance_meters

        if human_xy.size == 0:
            return {
                "perceived_distracted_indices": perceived_distracted_indices,
                "callback_target_idx": callback_target_idx,
                "emotion_modes": [],
            }

        n_humans = min(len(self.humans), int(human_xy.shape[0]))
        if n_humans <= 0:
            return {
                "perceived_distracted_indices": perceived_distracted_indices,
                "callback_target_idx": callback_target_idx,
                "emotion_modes": [],
            }

        human_xy = np.asarray(human_xy[:n_humans], dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        human_modes = [self.humans[idx].mode for idx in range(n_humans)]
        emotion_modes = [mode for mode in human_modes if mode != HumanMode.DISTRACTED]
        mode_array = np.asarray(human_modes, dtype=object)
        dist = np.linalg.norm(human_xy - robot_xy, axis=1)

        distracted_mask = mode_array == HumanMode.DISTRACTED
        far_distracted_mask = distracted_mask & (dist > threshold)
        perceived_distracted_indices = [int(idx) for idx in np.flatnonzero(far_distracted_mask)]
        distracted_source = np.asarray(
            [self.humans[idx].distracted_source for idx in range(n_humans)],
            dtype=object,
        )
        listening_distracted_mask = distracted_mask & (
            distracted_source == DISTRACTED_SOURCE_LISTENING
        )

        rearm_eligible_mask = np.zeros(n_humans, dtype=bool)
        eligible_count = min(n_humans, len(self.env_callback_state.triggered_for_distracted))
        if eligible_count > 0:
            rearm_eligible_mask[:eligible_count] = ~np.asarray(
                self.env_callback_state.triggered_for_distracted[:eligible_count],
                dtype=bool,
            )

        if self._is_listening_callback_interruptible():
            callback_candidate_mask = listening_distracted_mask & rearm_eligible_mask
        else:
            callback_candidate_mask = far_distracted_mask & rearm_eligible_mask
        if np.any(callback_candidate_mask):
            callback_candidate_indices = np.flatnonzero(callback_candidate_mask)
            if self._is_listening_callback_interruptible():
                callback_target_idx = int(callback_candidate_indices[0])
            else:
                farthest_candidate_local_idx = int(np.argmax(dist[callback_candidate_mask]))
                callback_target_idx = int(callback_candidate_indices[farthest_candidate_local_idx])

        return {
            "perceived_distracted_indices": perceived_distracted_indices,
            "callback_target_idx": callback_target_idx,
            "emotion_modes": emotion_modes,
        }

    def _build_callback_request(self, human_xy, robot_pose, human_analysis=None):
        """Build callback request targeting the farthest eligible distracted human."""
        if human_analysis is None:
            rx, ry, _ = robot_pose
            human_analysis = self._analyze_human_state(
                robot_xy=np.array([rx, ry], dtype=np.float32),
                human_xy=human_xy,
            )
        target_idx = human_analysis["callback_target_idx"]
        if target_idx is None:
            return None
        if not (0 <= target_idx < len(self.humans)):
            return None

        target_human = self.humans[target_idx]
        if self._is_listening_callback_interruptible():
            if target_human.distracted_source != DISTRACTED_SOURCE_LISTENING:
                return None
            return {
                "target_idx": int(target_idx),
                "target_xy": np.array(human_xy[target_idx], dtype=np.float32),
                "cue_steps": int(self._get_callback_cue_steps()),
                "success_mode": HumanMode.LISTENING,
                "interrupts_listening": True,
            }

        if not self._is_robot_in_move_stage(robot_pose):
            return None

        return {
            "target_idx": int(target_idx),
            "target_xy": np.array(human_xy[target_idx], dtype=np.float32),
            "cue_steps": int(self._get_callback_cue_steps()),
            "success_mode": HumanMode.FOLLOWING,
            "interrupts_listening": False,
        }

    def _sample_callback_response(self, profile: str):
        """Sample callback response from profile-specific probability distribution."""
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

    def _get_callback_cue_steps(self) -> int:
        """Return callback cue duration in simulation steps."""
        return int(self._callback_cue_steps)

    def _get_callback_response_sample_steps(self) -> int:
        """Return step offset inside cue when callback response should be sampled."""
        return int(self._callback_response_sample_steps)

    def _is_callback_cue_active(self) -> bool:
        """Return True only during callback cue phase after turning completes."""
        return (
            self.robot.callback_active
            and self.robot.callback_phase == RobotCallbackPhase.CUE
            and self.robot.callback_cue_elapsed_steps < self.robot.callback_cue_total_steps
        )

    def _maybe_sample_active_callback_response(self, events):
        """Sample callback response once at the 2-second mark of the active cue."""
        if not self.robot.callback_active:
            return
        if self.robot.callback_phase != RobotCallbackPhase.CUE:
            return
        if self.robot.callback_response_sampled:
            return
        if int(self.robot.callback_cue_elapsed_steps) < self._get_callback_response_sample_steps():
            return

        target_idx = self.env_callback_state.active_target_idx
        self.robot.callback_response_sampled = True
        if target_idx is None or not (0 <= target_idx < len(self.humans)):
            return

        recover_human = self.humans[target_idx]
        if recover_human.mode != HumanMode.DISTRACTED:
            return

        callback_response = self._sample_callback_response(profile=recover_human.profile)
        # Apply callback response to target human
        recovered = recover_human.apply_callback_response(
            response=callback_response,
            stay_steps=0,
        )
        # Record callback response and log event
        self.env_callback_state.last_response = str(callback_response)
        self.env_callback_state.last_response_target_idx = int(target_idx)
        events[f"callback_response_{callback_response}"] = True
        self._log_event(
            f">>> person{target_idx + 1} callback response: {callback_response} "
            f"(attempt {int(self.robot.callback_attempt_index)})."
        )
        if recovered and callback_response == "rejoin":
            events["callback_forced_recovery"] = True

    def _resolve_completed_callback_cue(self, events):
        """Resolve one callback attempt when its 4-second cue window finishes."""
        if not self.robot.callback_cue_completed_this_step:
            return

        target_idx = self.env_callback_state.active_target_idx
        target_human = None
        success = False
        success_mode = self.env_callback_state.success_mode
        if target_idx is not None and 0 <= target_idx < len(self.humans):
            target_human = self.humans[target_idx]
            success = target_human.mode == success_mode

        # Resolve callback attempt outcome and log events
        if success:
            events["callback_completed"] = True
            events["callback_success"] = True
            hold_steps = max(1, int(round(ROBOT_HAPPY_HOLD_SECONDS / self.timestep)))
            self.robot.trigger_happy(hold_steps)
            events["happy_triggered"] = True
            if target_idx is not None:
                self._log_event(
                    f">>> Robot CALLBACK succeeded for person{target_idx + 1} "
                    f"on attempt {int(self.robot.callback_attempt_index)}."
                )
            self.robot._finish_callback()
            self.env_callback_state.active_target_idx = None
            self.env_callback_state.success_mode = HumanMode.FOLLOWING
            self._resume_listening_after_callback()
            return
        # If first attempt failed, start second attempt
        if int(self.robot.callback_attempt_index) < 2:
            events["callback_first_attempt_failed"] = True
            if self.robot.start_next_callback_attempt():
                events["callback_attempt_2_started"] = True
                if target_idx is not None:
                    self._log_event(
                        f">>> Robot CALLBACK retry started for person{target_idx + 1}."
                    )
            else:
                events["callback_completed"] = True
                self.robot._finish_callback()
                self.env_callback_state.active_target_idx = None
                self.env_callback_state.success_mode = HumanMode.FOLLOWING
                self._resume_listening_after_callback()
            return

        events["callback_completed"] = True
        if target_idx is not None:
            self._log_event(
                f">>> Robot CALLBACK ended after second attempt for person{target_idx + 1}."
            )
        self.robot._finish_callback()
        self.env_callback_state.active_target_idx = None
        self.env_callback_state.success_mode = HumanMode.FOLLOWING
        self._resume_listening_after_callback()

    def _robot_base_rgba_for_emotion(self):
        """Return target robot base RGBA for the current emotion."""
        if self.robot.emotion == RobotEmotion.SAD:
            return ROBOT_COLOR_SAD
        if self.robot.emotion == RobotEmotion.HAPPY:
            return ROBOT_COLOR_HAPPY
        return ROBOT_COLOR_NATURAL

    def _apply_robot_base_color_from_robot_emotion(self, force: bool = False) -> bool:
        """Sync robot base color with current emotion only when visual state changes."""
        current_emotion = str(self.robot.emotion)
        if (not force) and self._last_robot_base_visual_emotion == current_emotion:
            return False
        self.model.geom_rgba[self.robot_base_geom_id] = self._robot_base_rgba_for_emotion()
        self._last_robot_base_visual_emotion = current_emotion
        return True

    def _sync_robot_speaker_state(self):
        """Update speaker on/off state from listening wait status."""
        self.robot.set_speaker_active(
            bool(self.listen_state.wait_active or self._is_callback_cue_active())
        )

    def _get_perceived_distracted_indices(self, robot_xy, human_xy):
        """Return distracted human indices that are farther than perception threshold."""
        analysis = self._analyze_human_state(np.asarray(robot_xy, dtype=np.float32), human_xy)
        return list(analysis["perceived_distracted_indices"])

    def _is_callback_visual_active(self):
        """Return True only during callback cue phase."""
        return self._is_callback_cue_active()

    def _robot_text_label_visibility_state(self):
        """Return desired visibility tuple for robot text labels."""
        show_follow_me = self._is_callback_visual_active()
        show_explanation = (not show_follow_me) and bool(self.robot.speaker_active)
        return (show_follow_me, show_explanation)

    def _sync_robot_text_label_visibility(self, force: bool = False) -> bool:
        """Toggle robot text labels with priority: follow-me > explanation."""
        label_state = self._robot_text_label_visibility_state()
        if (not force) and self._last_robot_label_visibility_state == label_state:
            return False

        show_follow_me, show_explanation = label_state
        self._label_scene_option.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP] = 1 if show_follow_me else 0
        self._label_scene_option.sitegroup[ROBOT_EXPLANATION_LABEL_GROUP] = 1 if show_explanation else 0
        self._last_robot_label_visibility_state = label_state
        return True

    def _apply_robot_speaking_halo_visual(self, force: bool = False) -> bool:
        """Show/hide speaking halo geometry only when speaker state changes."""
        speaker_active = bool(self.robot.speaker_active)
        if (not force) and self._last_robot_speaking_halo_active == speaker_active:
            return False
        if speaker_active:
            self.model.geom_rgba[self.robot_speaking_halo_geom_id] = SPEAKING_HALO_RGBA_ON
        else:
            self.model.geom_rgba[self.robot_speaking_halo_geom_id] = SPEAKING_HALO_RGBA_OFF
        self._last_robot_speaking_halo_active = speaker_active
        return True

    def _sync_robot_visual_state(self, force: bool = False):
        """Apply robot visuals only when the rendered state changed."""
        self._apply_robot_base_color_from_robot_emotion(force=force)
        self._sync_robot_text_label_visibility(force=force)
        self._apply_robot_speaking_halo_visual(force=force)

    def _get_robot_text_label(self):
        """Return semantic name of currently active robot text cue."""
        if self._is_callback_visual_active():
            return "Please_follow_me"
        if self.robot.speaker_active:
            return "explanation"
        return "none"

    def _update_robot_emotion_and_visual(self, events, robot_xy, human_xy, human_analysis=None):
        """Update robot emotion and perceived distracted diagnostics from one human snapshot."""
        if human_analysis is None:
            human_analysis = self._analyze_human_state(robot_xy=robot_xy, human_xy=human_xy)

        self.perceived_distracted_indices = list(human_analysis["perceived_distracted_indices"])
        callback_visual_active = self._is_callback_visual_active()
        emotion_modes = list(human_analysis["emotion_modes"])
        if callback_visual_active:
            # Callback cue should explicitly drive SAD even if the target mode changes mid-window.
            emotion_modes.append(HumanMode.DISTRACTED)

        happy_before = int(self.robot.happy_hold_steps_remaining)
        sad_now = any(mode in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED) for mode in emotion_modes)
        self.robot.update_emotion(emotion_modes)
        happy_after = int(self.robot.happy_hold_steps_remaining)
        if happy_before > 0 and happy_after == 0 and (not sad_now):
            events["happy_completed"] = True

    @staticmethod
    def _compute_nearest_human_distances(human_xy):
        """Return each human's nearest human-human Euclidean distance."""
        pairwise_dist = MuseumEnv._compute_human_pairwise_distances(human_xy)
        return MuseumEnv._compute_nearest_human_distances_from_pairwise(pairwise_dist)

    @staticmethod
    def _compute_human_pairwise_distances(human_xy):
        """Return dense pairwise human-human Euclidean distances with self-distance masked."""
        human_xy = np.asarray(human_xy, dtype=np.float32)
        n_humans = int(human_xy.shape[0])
        if n_humans == 0:
            return np.zeros((0, 0), dtype=np.float32)
        pairwise_diff = human_xy[:, None, :] - human_xy[None, :, :]
        pairwise_dist = np.linalg.norm(pairwise_diff, axis=2).astype(np.float32)
        np.fill_diagonal(pairwise_dist, np.inf)
        return pairwise_dist

    @staticmethod
    def _compute_nearest_human_distances_from_pairwise(pairwise_dist):
        """Return each human's nearest human-human Euclidean distance from a dense matrix."""
        pairwise_dist = np.asarray(pairwise_dist, dtype=np.float32)
        if pairwise_dist.shape == (0, 0):
            return np.zeros((0,), dtype=np.float32)

        nearest_human_distance = np.min(pairwise_dist, axis=1).astype(np.float32)
        nearest_human_distance[~np.isfinite(nearest_human_distance)] = np.nan
        return nearest_human_distance

    @staticmethod
    def _compute_local_crowding_count_1m(human_xy):
        """Return each human's count of other humans strictly within 1 meter."""
        pairwise_dist = MuseumEnv._compute_human_pairwise_distances(human_xy)
        return MuseumEnv._compute_local_crowding_count_1m_from_pairwise(pairwise_dist)

    @staticmethod
    def _compute_local_crowding_count_1m_from_pairwise(pairwise_dist):
        """Return each human's count of other humans strictly within the local crowding radius."""
        pairwise_dist = np.asarray(pairwise_dist, dtype=np.float32)
        if pairwise_dist.shape == (0, 0):
            return np.zeros((0,), dtype=np.int32)

        return np.count_nonzero(pairwise_dist < LOCAL_CROWDING_RADIUS_METERS, axis=1).astype(
            np.int32
        )

    @staticmethod
    def _compute_human_robot_distances(human_xy, robot_xy):
        """Return each human's Euclidean distance to the robot."""
        human_xy = np.asarray(human_xy, dtype=np.float32)
        if human_xy.size == 0:
            return np.zeros((0,), dtype=np.float32)
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        return np.linalg.norm(human_xy - robot_xy[None, :], axis=1).astype(np.float32)

    @staticmethod
    def _resolve_fuzzy_metric_input(rolling_mean_value: float, current_value: float) -> float:
        if np.isfinite(rolling_mean_value):
            return float(rolling_mean_value)
        if np.isfinite(current_value):
            return float(current_value)
        return 0.0

    def _compute_human_fuzzy_diagnostics(
        self,
        human,
        idx: int,
        phase: str,
        session_steps: int,
        nearest_human_distance,
        nearest_human_distance_mean_1s,
        human_robot_distance,
        human_robot_distance_mean_1s,
        local_crowding_count_1m,
    ):
        if self.following_fuzzy_engine is None:
            return None

        inputs = self.following_fuzzy_engine.clip_inputs(
            following_time=float(session_steps) * float(self.timestep_float),
            hhd=self._resolve_fuzzy_metric_input(
                rolling_mean_value=float(nearest_human_distance_mean_1s[idx]),
                current_value=float(nearest_human_distance[idx]),
            ),
            hrd=self._resolve_fuzzy_metric_input(
                rolling_mean_value=float(human_robot_distance_mean_1s[idx]),
                current_value=float(human_robot_distance[idx]),
            ),
            density=float(local_crowding_count_1m[idx]),
        )
        result = self.following_fuzzy_engine.compute(**inputs, context=phase)
        return {"inputs": inputs, "result": result}

    def _record_human_fuzzy_result(self, idx: int, phase: str, fuzzy_debug: dict) -> None:
        self.last_human_fuzzy_inputs[idx] = dict(fuzzy_debug["inputs"])
        self.last_human_fuzzy_scores[idx] = {
            state: float(fuzzy_debug["result"][state])
            for state in ("overwhelmed", "distracted", "impatient", "engaged")
        }
        self.last_human_fuzzy_dominant_state[idx] = str(fuzzy_debug["result"]["dominant_state"])
        self.last_human_fuzzy_context[idx] = str(phase)
        self._following_fuzzy_last_eval_refresh_counter[idx] = int(self._observation_refresh_counter)
        if phase == "following":
            self._following_fuzzy_evaluated_this_step = True
        elif phase == "listening":
            self._listening_fuzzy_evaluated_this_step = True
        else:
            raise ValueError(f"Unknown fuzzy phase: {phase}")

    def _apply_following_fuzzy_transition(self, human, idx: int, fuzzy_result: dict, robot_xy, robot_yaw: float):
        dominant_state = fuzzy_result["dominant_state"]
        if dominant_state == "engaged":
            return

        if dominant_state == "distracted":
            human.distracted_source = DISTRACTED_SOURCE_FOLLOWING
            human.distracted_recovery_mode = HumanMode.FOLLOWING
            human.set_mode(HumanMode.DISTRACTED, reason="fuzzy_following_distracted")
            human._log_event(f">>> {human.name} became DISTRACTED!")
            return

        if dominant_state == "impatient":
            robot_pose = (float(robot_xy[0]), float(robot_xy[1]), float(robot_yaw))
            human.start_impatient(
                robot_pose=robot_pose,
                index=idx,
                n_humans=self.n_humans,
                recovery_mode=HumanMode.FOLLOWING,
            )
            return

        if dominant_state == "overwhelmed":
            current_xy = np.array(self.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
            human.start_overwhelmed(
                robot_xy=np.array(robot_xy, dtype=np.float32),
                current_xy=current_xy,
                recovery_mode=HumanMode.FOLLOWING,
            )
            return

    def _apply_listening_fuzzy_transition(self, human, idx: int, fuzzy_result: dict, robot_xy, robot_yaw: float):
        del robot_yaw
        dominant_state = fuzzy_result["dominant_state"]
        if dominant_state == "engaged":
            return

        if dominant_state == "distracted":
            human.distracted_source = DISTRACTED_SOURCE_LISTENING
            human.distracted_recovery_mode = HumanMode.LISTENING
            human.set_mode(HumanMode.DISTRACTED, reason="fuzzy_listening_distracted")
            human._log_event(f">>> {human.name} became DISTRACTED while listening!")
            current_xy = np.array(self.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
            self._queue_listening_callback_request(idx, current_xy)
            return

        if dominant_state == "impatient":
            human.start_impatient(recovery_mode=HumanMode.LISTENING)
            return

        if dominant_state == "overwhelmed":
            current_xy = np.array(self.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
            human.start_overwhelmed(
                robot_xy=np.array(robot_xy, dtype=np.float32),
                current_xy=current_xy,
                recovery_mode=HumanMode.LISTENING,
            )
            return

    def reset(self, seed=None, options=None):
        """Reset MuJoCo state and all episode-level state machines."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0

        # Reset robot agent
        self.robot.reset()

        # Store robot start position (for follow trigger)
        rx, ry, _ = self._get_robot_pose()
        self.robot_start_xy = np.array([rx, ry], dtype=np.float32)
        self._clear_follow_phase()

        # Reset listening state
        self.listen_state.reset()
        self.post_explanation_state.reset()
        self.env_callback_state.reset(len(self.humans))
        self.perceived_distracted_indices = []
        self.hh_distance_metric.reset()
        self.hr_distance_metric.reset()
        self._reset_environment_observation_cache_state()
        self._reset_following_fuzzy_debug_state()

        # Reset humans
        for human in self.humans:
            human.reset_episode_state()
        rx, ry, _ = self._get_robot_pose()
        self._reset_human_positions(robot_xy=np.array([rx, ry], dtype=np.float32))
        mujoco.mj_forward(self.model, self.data)
        self._reset_human_listening_session_states()
        self._configure_human_distracted_duration()
        self._configure_human_following_variants()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        robot_xy = np.array(self._get_robot_pose()[:2], dtype=np.float32)
        self._refresh_environment_observations(human_xy=human_xy, robot_xy=robot_xy, force=True)
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state(force=True)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action=None):
        """
        Run one environment step with robot policy + human state machines.
        """
        external_action_received = self._validate_external_action(action)
        self.step_count += 1
        self._following_fuzzy_evaluated_this_step = False
        self._listening_fuzzy_evaluated_this_step = False
        self.env_callback_state.pending_listening_request = None

        if self.listen_state.wait_active:
            return self._step_waiting_branch(external_action_received=external_action_received)

        return self._step_active_branch(external_action_received=external_action_received)

    def _step_waiting_branch(self, external_action_received=False):
        """Step branch used during listening wait window (explanation phase)."""
        events = self._default_events()

        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        repulsion_vectors = self._compute_social_repulsion(human_xy)
        robot_xy = np.array([rx, ry], dtype=np.float32)
        self.data.ctrl[:] = 0.0
        rb_action = np.zeros(3, dtype=np.float32)
        human_actions = self._update_listening_humans_and_apply_ctrl(
            robot_xy=robot_xy,
            ryaw=ryaw,
            repulsion_vectors=repulsion_vectors,
        )
        if self.env_callback_state.pending_listening_request is not None:
            self._pause_listening_for_callback()
        self.robot.mode = RobotMode.STOP

        self.data.ctrl[0:3] = rb_action

        mujoco.mj_step(self.model, self.data)
        if self.listen_state.wait_active:
            self.listen_state.wait_counter += 1

        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)
        robot_xy = np.array([rx, ry], dtype=np.float32)
        self._tick_observation_sample_age()
        observation_snapshot = self._refresh_environment_observations(
            human_xy=human_xy,
            robot_xy=robot_xy,
        )
        nearest_human_distance = observation_snapshot["nearest_human_distance"]
        local_crowding_count_1m = observation_snapshot["local_crowding_count_1m"]
        nearest_human_distance_mean_1s = observation_snapshot["nearest_human_distance_mean_1s"]
        human_robot_distance = observation_snapshot["human_robot_distance"]
        human_robot_distance_mean_1s = observation_snapshot["human_robot_distance_mean_1s"]

        wx, wy = self.robot.get_current_waypoint()
        dist = float(np.hypot(wx - rx, wy - ry) + DIST_EPS)
        desired_yaw = float(np.arctan2(wy - ry, wx - rx))
        actual_yaw = float(ryaw)

        human_goals = self._build_human_goals(human_xy=human_xy, robot_xy=robot_xy)
        human_reached_goal, human_in_listening_front_sector = self._check_human_goals(
            human_xy,
            human_goals,
            robot_xy=robot_xy,
            robot_yaw=ryaw,
        )
        self._update_human_listening_session_progress()

        final_waypoint_reached = self.robot.is_final_reached(dist)
        all_humans_reached = len(self.humans) > 0 and len(human_reached_goal) == len(self.humans)

        if self.listen_state.wait_counter >= self.listen_wait_steps:
            events["completed_listen_wait"] = True
            if self.listen_state.wait_is_final:
                events["final_listen_ready"] = True
                self._log_event(">>> Listening wait complete at final display.")
            else:
                self.robot.on_listening_complete()
                self._clear_follow_phase()
                self.robot_start_xy = np.array([rx, ry], dtype=np.float32)
                self._start_post_explanation_hold(
                    robot_xy=np.array([rx, ry], dtype=np.float32),
                    robot_yaw=ryaw,
                    human_xy=human_xy,
                )
                self._log_event(">>> Listening wait complete. Resume MOVE to Room B.")

            self.listen_state.intro_active = False
            self.listen_state.intro_counter = 0
            self.listen_state.intro_is_final = False
            self._reset_human_listening_session_states()
            self.listen_state.wait_active = False
            self.listen_state.wait_counter = 0
            self.listen_state.wait_is_final = False
            human_goals = self._build_human_goals(human_xy=human_xy, robot_xy=robot_xy)

        human_analysis = self._analyze_human_state(robot_xy=robot_xy, human_xy=human_xy)
        self._update_robot_emotion_and_visual(
            events=events,
            robot_xy=robot_xy,
            human_xy=human_xy,
            human_analysis=human_analysis,
        )
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state()

        human_state_snapshot = self._collect_human_state_snapshot()

        snapshot = self._collect_step_snapshot(
            robot_pose=(rx, ry, ryaw),
            dist=dist,
            desired_yaw=desired_yaw,
            actual_yaw=actual_yaw,
            robot_mode=str(self.robot.mode),
            robot_action=rb_action,
            human_xy=human_xy,
            human_actual_yaw=human_actual_yaw,
            human_goals=human_goals,
            human_actions=human_actions,
            nearest_human_distance=nearest_human_distance,
            local_crowding_count_1m=local_crowding_count_1m,
            nearest_human_distance_mean_1s=nearest_human_distance_mean_1s,
            human_robot_distance=human_robot_distance,
            human_robot_distance_mean_1s=human_robot_distance_mean_1s,
            human_state_snapshot=human_state_snapshot,
            human_in_listening_front_sector=human_in_listening_front_sector,
            human_reached_goal=human_reached_goal,
            final_waypoint_reached=final_waypoint_reached,
            all_humans_reached=all_humans_reached,
        )

        return self._finalize_step_output(
            snapshot=snapshot,
            events=events,
            dist=dist,
            terminated=events["final_listen_ready"],
            external_action_received=external_action_received,
            external_action_used=False,
        )

    def _step_active_branch(self, external_action_received=False):
        """Main simulation branch outside wait window."""
        # Main runtime branch: robot decision, human updates, MuJoCo step, and info assembly.
        events = self._default_events()

        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        # --- Robot decision ---
        # External action is validated for API compatibility, but this env uses rule-based control.
        robot_pose = self._get_robot_pose()
        robot_xy = np.array([robot_pose[0], robot_pose[1]], dtype=np.float32)
        self._refresh_callback_rearm_flags()
        human_analysis_before = self._analyze_human_state(robot_xy=robot_xy, human_xy=human_xy)
        callback_request = self._build_callback_request(
            human_xy=human_xy,
            robot_pose=robot_pose,
            human_analysis=human_analysis_before,
        )
        if (
            callback_request is not None
            and callback_request.get("interrupts_listening", False)
            and (not self._is_listening_interrupted_for_callback())
        ):
            self._pause_listening_for_callback()
        callback_active_before_step = self.robot.callback_active
        robot_out = self.robot.step(
            robot_pose=robot_pose,
            human_xyz=human_xyz,
            callback_request=callback_request,
        )

        rb_action = robot_out["action"]
        dist = robot_out["dist"]
        desired_yaw = robot_out["desired_yaw"]
        actual_yaw = robot_out["actual_yaw"]
        robot_mode = robot_out["mode"]
        enter_listen = robot_out["enter_listen"]
        events["entered_listen"] = enter_listen
        if callback_request is not None and (not callback_active_before_step) and robot_mode == RobotMode.CALLBACK:
            events["callback_triggered"] = True
            events["callback_attempt_1_started"] = True
            target_idx = int(callback_request["target_idx"])
            self.env_callback_state.success_mode = str(
                callback_request.get("success_mode", HumanMode.FOLLOWING)
            )
            self.env_callback_state.active_target_idx = target_idx
            if 0 <= target_idx < len(self.env_callback_state.triggered_for_distracted):
                # Mark this human as having triggered callback
                self.env_callback_state.triggered_for_distracted[target_idx] = True
                self._log_event(f">>> Robot CALLBACK triggered for person{target_idx + 1}.")
        self._maybe_sample_active_callback_response(events)
        self._resolve_completed_callback_cue(events)

        # If robot just entered listen, start a silent preparation window first.
        if enter_listen:
            rx, ry, ryaw = robot_pose
            final_waypoint_reached_at_entry = self.robot.is_final_reached(dist)
            self._clear_follow_phase()
            self.listen_state.intro_active = True
            self.listen_state.intro_counter = 0
            self.listen_state.intro_is_final = final_waypoint_reached_at_entry
            self.listen_state.wait_active = False
            self.listen_state.wait_counter = 0
            self.listen_state.wait_is_final = False
            self._reset_human_listening_session_states()

            self._log_event(
                f">>> Robot entering LISTEN mode. robot=({rx:.2f}, {ry:.2f}, yaw={ryaw:.2f}); "
                f"silent 3s preparation started while humans regulate to a "
                f"{self.listen_fan_radius:.2f}m ring inside the front 160 deg sector."
            )
            self.post_explanation_state.reset()

        # Apply robot action
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = rb_action

        rx, ry, ryaw = self._get_robot_pose()
        robot_xy = np.array([rx, ry], dtype=np.float32)
        self._maybe_finish_post_explanation_hold(
            robot_xy=robot_xy,
            robot_speed=np.hypot(rb_action[0], rb_action[1]),
        )

        # Start pre-listen engage on the first leg and transit-follow after the first explanation.
        self._maybe_activate_follow_phase_from_robot_progress(robot_xy=robot_xy)

        repulsion_vectors = self._compute_social_repulsion(human_xy)
        human_actions = self._update_humans_and_apply_ctrl(
            human_xy=human_xy,
            robot_xy=robot_xy,
            ryaw=ryaw,
            repulsion_vectors=repulsion_vectors,
        )
        if self.env_callback_state.pending_listening_request is not None:
            self._pause_listening_for_callback()
        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Refresh poses after the step for reporting
        rx, ry, ryaw = self._get_robot_pose()
        robot_xy = np.array([rx, ry], dtype=np.float32)
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)
        self._tick_observation_sample_age()
        observation_snapshot = self._refresh_environment_observations(
            human_xy=human_xy,
            robot_xy=robot_xy,
        )
        nearest_human_distance = observation_snapshot["nearest_human_distance"]
        local_crowding_count_1m = observation_snapshot["local_crowding_count_1m"]
        nearest_human_distance_mean_1s = observation_snapshot["nearest_human_distance_mean_1s"]
        human_robot_distance = observation_snapshot["human_robot_distance"]
        human_robot_distance_mean_1s = observation_snapshot["human_robot_distance_mean_1s"]

        human_goals = self._build_human_goals(
            human_xy=human_xy,
            robot_xy=robot_xy,
        )
        human_reached_goal, human_in_listening_front_sector = self._check_human_goals(
            human_xy,
            human_goals,
            robot_xy=robot_xy,
            robot_yaw=ryaw,
        )
        self._update_human_listening_session_progress()

        if self.listen_state.intro_active:
            self.listen_state.intro_counter += 1
            if self.listen_state.intro_counter >= self.listen_intro_delay_steps:
                self.listen_state.intro_active = False
                self.listen_state.wait_active = True
                self.listen_state.wait_counter = 0
                self.listen_state.wait_is_final = self.listen_state.intro_is_final
                self.listen_state.intro_is_final = False
                events["started_listen_wait"] = True
                self._log_event(
                    f">>> Listening explanation started after {self.listen_intro_delay_seconds:.1f}s delay."
                )

        final_waypoint_reached = self.robot.is_final_reached(dist)
        all_humans_reached = len(self.humans) > 0 and len(human_reached_goal) == len(self.humans)
        human_analysis_after = self._analyze_human_state(robot_xy=robot_xy, human_xy=human_xy)
        self._update_robot_emotion_and_visual(
            events=events,
            robot_xy=robot_xy,
            human_xy=human_xy,
            human_analysis=human_analysis_after,
        )
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state()
        human_state_snapshot = self._collect_human_state_snapshot()

        snapshot = self._collect_step_snapshot(
            robot_pose=(rx, ry, ryaw),
            dist=dist,
            desired_yaw=desired_yaw,
            actual_yaw=actual_yaw,
            robot_mode=str(robot_mode),
            robot_action=np.array(rb_action, dtype=np.float32),
            human_xy=human_xy,
            human_actual_yaw=human_actual_yaw,
            human_goals=human_goals,
            human_actions=human_actions,
            nearest_human_distance=nearest_human_distance,
            local_crowding_count_1m=local_crowding_count_1m,
            nearest_human_distance_mean_1s=nearest_human_distance_mean_1s,
            human_robot_distance=human_robot_distance,
            human_robot_distance_mean_1s=human_robot_distance_mean_1s,
            human_state_snapshot=human_state_snapshot,
            human_in_listening_front_sector=human_in_listening_front_sector,
            human_reached_goal=human_reached_goal,
            final_waypoint_reached=final_waypoint_reached,
            all_humans_reached=all_humans_reached,
        )

        return self._finalize_step_output(
            snapshot=snapshot,
            events=events,
            dist=dist,
            terminated=False,
            external_action_received=external_action_received,
            external_action_used=False,
        )

    def _compute_social_repulsion(self, human_xy):
        """Compute pairwise short-range repulsion vectors for every human."""
        # Pairwise short-range repulsion to prevent humans from collapsing into each other.
        human_xy = np.asarray(human_xy, dtype=np.float32)
        n_humans = int(human_xy.shape[0])
        if n_humans == 0 or self.social_distance <= 1e-6:
            return np.zeros((n_humans, 2), dtype=np.float32)

        pairwise_dist = self._compute_human_pairwise_distances(human_xy)
        mask = (pairwise_dist > 1e-6) & (pairwise_dist < self.social_distance)
        repulsion_vectors = np.zeros((n_humans, 2), dtype=np.float32)
        for i in range(n_humans):
            neighbors = mask[i]
            if not np.any(neighbors):
                continue

            diff = human_xy[i] - human_xy[neighbors]
            dist = pairwise_dist[i, neighbors]
            directions = diff / dist[:, None]
            strengths = (self.social_distance - dist) / self.social_distance
            repulsion_vectors[i] = self.repulsion_gain * np.sum(
                directions * strengths[:, None],
                axis=0,
            )

        return repulsion_vectors

    def _update_listening_humans_and_apply_ctrl(self, robot_xy, ryaw, repulsion_vectors):
        """Apply the minimal listening-force controller to every active human."""
        n_humans = len(self.humans)
        human_actions = np.zeros((n_humans, 3), dtype=np.float32)
        if self._cached_obs is None:
            human_xyz = self._get_human_poses()
            human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
            observation_snapshot = self._refresh_environment_observations(
                human_xy=human_xy,
                robot_xy=robot_xy,
                force=True,
            )
        else:
            observation_snapshot = self._cached_obs
        nearest_human_distance = observation_snapshot["nearest_human_distance"]
        local_crowding_count_1m = observation_snapshot["local_crowding_count_1m"]
        nearest_human_distance_mean_1s = observation_snapshot["nearest_human_distance_mean_1s"]
        human_robot_distance = observation_snapshot["human_robot_distance"]
        human_robot_distance_mean_1s = observation_snapshot["human_robot_distance_mean_1s"]
        ctx = {
            "robot_xy": robot_xy,
            "robot_yaw": ryaw,
            "robot_speed": 0.0,
            "repulsion": np.zeros(2, dtype=np.float32),
            "listen_radius": self.listen_fan_radius,
            "stand_threshold": self.listen_stand_threshold,
            "listening_sector_half_angle": self.listen_front_sector_half_angle,
            "dt": self.timestep_float,
        }

        for i, human in enumerate(self.humans):
            if human.mode not in (
                HumanMode.DISTRACTED,
                HumanMode.OVERWHELMED,
                HumanMode.IMPATIENT,
            ):
                human.set_mode(HumanMode.LISTENING)
            repulsion_vec = repulsion_vectors[i] if i < repulsion_vectors.shape[0] else np.zeros(2, dtype=np.float32)
            ctx["repulsion"] = LISTENING_REPULSION_SCALE * repulsion_vec
            context_dict = {
                "index": i,
                "n_humans": n_humans,
                "robot_pose": (float(robot_xy[0]), float(robot_xy[1]), float(ryaw)),
                "listen_radius": self.listen_fan_radius,
                "listening_sector_half_angle": self.listen_front_sector_half_angle,
                "robot_xy": robot_xy,
                "robot_yaw": ryaw,
            }
            human._assign_target_from_context(context_dict)
            if human.mode == HumanMode.LISTENING and self._should_evaluate_human_fuzzy(i, phase="listening"):
                fuzzy_debug = self._compute_human_fuzzy_diagnostics(
                    human=human,
                    idx=i,
                    phase="listening",
                    session_steps=int(human.listening_steps),
                    nearest_human_distance=nearest_human_distance,
                    nearest_human_distance_mean_1s=nearest_human_distance_mean_1s,
                    human_robot_distance=human_robot_distance,
                    human_robot_distance_mean_1s=human_robot_distance_mean_1s,
                    local_crowding_count_1m=local_crowding_count_1m,
                )
                if fuzzy_debug is not None:
                    self._record_human_fuzzy_result(idx=i, phase="listening", fuzzy_debug=fuzzy_debug)
                    self._apply_listening_fuzzy_transition(
                        human=human,
                        idx=i,
                        fuzzy_result=fuzzy_debug["result"],
                        robot_xy=robot_xy,
                        robot_yaw=ryaw,
                    )
            human_action = human.step(self.model, self.data, ctx)
            human_actions[i] = human_action

            ctrl_idx = 3 + i * 3
            self.data.ctrl[ctrl_idx:ctrl_idx + 3] = human_action

        return human_actions

    def _update_post_explanation_humans_and_apply_ctrl(self, robot_xy, ryaw, repulsion_vectors):
        """Run the reuse-first post-explanation transition without falling back to wandering."""
        n_humans = len(self.humans)
        human_actions = np.zeros((n_humans, 3), dtype=np.float32)
        anchor_robot_xy = (
            np.array(self.post_explanation_state.anchor_robot_xy, dtype=np.float32)
            if self.post_explanation_state.anchor_robot_xy is not None
            else np.array(robot_xy, dtype=np.float32)
        )
        anchor_robot_yaw = float(self.post_explanation_state.anchor_robot_yaw)
        move_ctx = {
            "robot_xy": robot_xy,
            "robot_yaw": ryaw,
            "robot_speed": np.hypot(self.data.ctrl[0], self.data.ctrl[1]),
            "repulsion": np.zeros(2, dtype=np.float32),
            "dt": self.timestep_float,
        }

        for i, human in enumerate(self.humans):
            repulsion_vec = repulsion_vectors[i] if i < repulsion_vectors.shape[0] else np.zeros(2, dtype=np.float32)
            role = (
                self.post_explanation_state.roles[i]
                if i < len(self.post_explanation_state.roles)
                else POST_EXPLANATION_YIELD_ROLE_WAIT
            )
            target_xy = (
                np.array(self.post_explanation_state.targets[i], dtype=np.float32)
                if i < len(self.post_explanation_state.targets)
                else np.array(self.data.qpos[human.qpos_idx : human.qpos_idx + 2], dtype=np.float32)
            )

            if human.mode in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
                move_ctx["repulsion"] = repulsion_vec
                human_action = human.step(self.model, self.data, move_ctx)
            elif role == POST_EXPLANATION_YIELD_ROLE_YIELD:
                human.set_mode(HumanMode.FOLLOWING)
                human.current_waypoint = target_xy.copy()
                move_ctx["repulsion"] = repulsion_vec
                human_action = human._step_following(self.data, move_ctx, human._get_pose(self.data))
            else:
                human.set_mode(HumanMode.LISTENING)
                listen_ctx = {
                    "robot_xy": robot_xy,
                    "robot_yaw": ryaw,
                    "robot_speed": 0.0,
                    "repulsion": LISTENING_REPULSION_SCALE * repulsion_vec,
                    "listen_radius": (
                        self.post_explanation_state.listen_radii[i]
                        if i < len(self.post_explanation_state.listen_radii)
                        else self.listen_fan_radius
                    ),
                    "stand_threshold": self.listen_stand_threshold,
                    "listening_sector_half_angle": self.listen_front_sector_half_angle,
                    "dt": self.timestep_float,
                }
                human_action = human._step_listening_with_anchor_target_and_live_repulsion(
                    self.data,
                    listen_ctx,
                    human._get_pose(self.data),
                    anchor_robot_xy=anchor_robot_xy,
                    anchor_robot_yaw=anchor_robot_yaw,
                    live_robot_xy=robot_xy,
                )

            human_actions[i] = human_action
            ctrl_idx = 3 + i * 3
            self.data.ctrl[ctrl_idx:ctrl_idx + 3] = human_action

        return human_actions

    def _update_humans_and_apply_ctrl(self, human_xy, robot_xy, ryaw, repulsion_vectors):
        """Update all humans and write their commands to control buffer."""
        if self.post_explanation_state.active:
            return self._update_post_explanation_humans_and_apply_ctrl(
                robot_xy=robot_xy,
                ryaw=ryaw,
                repulsion_vectors=repulsion_vectors,
            )
        if self._is_listening_controller_active():
            return self._update_listening_humans_and_apply_ctrl(
                robot_xy=robot_xy,
                ryaw=ryaw,
                repulsion_vectors=repulsion_vectors,
            )

        human_actions = np.zeros((len(self.humans), 3), dtype=np.float32)
        n_humans = len(self.humans)
        follow_radius = FOLLOW_RADIUS_DEFAULT
        robot_speed = np.hypot(self.data.ctrl[0], self.data.ctrl[1])
        if self._cached_obs is None:
            observation_snapshot = self._refresh_environment_observations(
                human_xy=human_xy,
                robot_xy=robot_xy,
                force=True,
            )
        else:
            observation_snapshot = self._cached_obs
        nearest_human_distance = observation_snapshot["nearest_human_distance"]
        local_crowding_count_1m = observation_snapshot["local_crowding_count_1m"]
        nearest_human_distance_mean_1s = observation_snapshot["nearest_human_distance_mean_1s"]
        human_robot_distance = observation_snapshot["human_robot_distance"]
        human_robot_distance_mean_1s = observation_snapshot["human_robot_distance_mean_1s"]
        robot_pose = (float(robot_xy[0]), float(robot_xy[1]), float(ryaw))
        ctx = {
            "robot_xy": robot_xy,
            "robot_yaw": ryaw,
            "robot_speed": robot_speed,
            "repulsion": np.zeros(2, dtype=np.float32),
            "listen_radius": self.listen_fan_radius,
            "stand_threshold": self.listen_stand_threshold,
            "listening_sector_half_angle": self.listen_front_sector_half_angle,
            "dt": self.timestep_float,
        }

        for i, human in enumerate(self.humans):
            repulsion_vec = repulsion_vectors[i] if i < repulsion_vectors.shape[0] else np.zeros(2, dtype=np.float32)
            ctx["repulsion"] = repulsion_vec

            if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
                human.set_mode(HumanMode.FOLLOWING if self.follow_humans else HumanMode.WANDERING)

            if self.follow_humans and human.mode in (HumanMode.FOLLOWING, HumanMode.IMPATIENT):
                context_dict = {
                    "index": i,
                    "n_humans": n_humans,
                    "robot_pose": robot_pose,
                    "follow_radius": follow_radius,
                    "fan_half_angle": self.follow_fan_half_angle,
                    "impatient_front_offset": human.impatient_front_offset,
                    "robot_xy": robot_xy,
                    "robot_yaw": ryaw,
                }
                human._assign_target_from_context(context_dict)

            human.update_following_duration(
                eligible_following=self.follow_humans and human.mode == HumanMode.FOLLOWING
            )
            if human.mode == HumanMode.FOLLOWING and self._should_evaluate_human_fuzzy(i, phase="following"):
                fuzzy_debug = self._compute_human_fuzzy_diagnostics(
                    human=human,
                    idx=i,
                    phase="following",
                    session_steps=int(human.following_steps),
                    nearest_human_distance=nearest_human_distance,
                    nearest_human_distance_mean_1s=nearest_human_distance_mean_1s,
                    human_robot_distance=human_robot_distance,
                    human_robot_distance_mean_1s=human_robot_distance_mean_1s,
                    local_crowding_count_1m=local_crowding_count_1m,
                )
                if fuzzy_debug is not None:
                    self._record_human_fuzzy_result(idx=i, phase="following", fuzzy_debug=fuzzy_debug)
                    self._apply_following_fuzzy_transition(
                        human=human,
                        idx=i,
                        fuzzy_result=fuzzy_debug["result"],
                        robot_xy=robot_xy,
                        robot_yaw=ryaw,
                    )

            human_action = human.step(self.model, self.data, ctx)
            human_actions[i] = human_action

            ctrl_idx = 3 + i * 3
            self.data.ctrl[ctrl_idx:ctrl_idx + 3] = human_action

        return human_actions

    def _collect_human_state_snapshot(self):
        """Collect per-human mode/profile/timer diagnostics in one pass."""
        n_humans = len(self.humans)
        human_modes = [None] * n_humans
        human_profiles = [None] * n_humans
        human_distracted_source = [None] * n_humans
        human_overwhelmed_stage = [None] * n_humans
        human_fuzzy_scores = [None] * n_humans
        human_fuzzy_dominant_state = [None] * n_humans
        human_fuzzy_inputs = [None] * n_humans
        human_fuzzy_context = [None] * n_humans
        human_distracted_timer = np.empty(n_humans, dtype=np.int32)
        human_following_steps = np.empty(n_humans, dtype=np.int32)
        human_listening_steps = np.empty(n_humans, dtype=np.int32)
        human_overwhelmed_leave_timer = np.empty(n_humans, dtype=np.int32)
        human_impatient_timer = np.empty(n_humans, dtype=np.int32)
        active_overwhelmed_indices = []

        for idx, human in enumerate(self.humans):
            mode = human.mode
            human_modes[idx] = mode
            human_profiles[idx] = human.profile
            human_distracted_source[idx] = human.distracted_source
            human_overwhelmed_stage[idx] = human.overwhelmed_stage
            human_fuzzy_scores[idx] = dict(self.last_human_fuzzy_scores[idx])
            human_fuzzy_dominant_state[idx] = self.last_human_fuzzy_dominant_state[idx]
            human_fuzzy_inputs[idx] = dict(self.last_human_fuzzy_inputs[idx])
            human_fuzzy_context[idx] = self.last_human_fuzzy_context[idx]
            human_distracted_timer[idx] = int(human.distracted_timer)
            human_following_steps[idx] = int(human.following_steps)
            human_listening_steps[idx] = int(human.listening_steps)
            human_overwhelmed_leave_timer[idx] = int(human.overwhelmed_leave_timer)
            human_impatient_timer[idx] = int(human.impatient_timer)
            if mode == HumanMode.OVERWHELMED:
                active_overwhelmed_indices.append(int(idx))

        return {
            "human_mode": human_modes,
            "human_profile": human_profiles,
            "human_distracted_source": human_distracted_source,
            "human_overwhelmed_stage": human_overwhelmed_stage,
            "human_fuzzy_scores": human_fuzzy_scores,
            "human_fuzzy_dominant_state": human_fuzzy_dominant_state,
            "human_fuzzy_inputs": human_fuzzy_inputs,
            "human_fuzzy_context": human_fuzzy_context,
            "human_distracted_timer": human_distracted_timer,
            "human_following_steps": human_following_steps,
            "human_listening_steps": human_listening_steps,
            "human_overwhelmed_leave_timer": human_overwhelmed_leave_timer,
            "human_impatient_timer": human_impatient_timer,
            "active_overwhelmed_indices": active_overwhelmed_indices,
        }

    def _build_human_goals(self, human_xy, robot_xy):
        """Build goal-like diagnostics array while letting listening humans point at robot."""
        n_humans = len(self.humans)
        if n_humans == 0:
            return np.zeros((0, 2), dtype=np.float32)

        goals = np.empty((n_humans, 2), dtype=np.float32)
        for idx, human in enumerate(self.humans):
            if self.post_explanation_state.active and idx < len(self.post_explanation_state.targets):
                goals[idx] = np.array(self.post_explanation_state.targets[idx], dtype=np.float32)
            elif human.mode == HumanMode.LISTENING:
                goals[idx] = robot_xy
            else:
                goals[idx] = human.current_waypoint
        return goals

    def _check_human_goals(self, human_xy, human_goals, robot_xy=None, robot_yaw=None):
        """Return indices of humans that satisfy the active goal/reached criterion."""
        human_reached_goal = []
        human_in_listening_front_sector = np.zeros(len(self.humans), dtype=bool)
        robot_xy_arr = None if robot_xy is None else np.asarray(robot_xy, dtype=np.float32)
        for i, (human, pos, goal) in enumerate(zip(self.humans, human_xy, human_goals)):
            if self.post_explanation_state.active and i < len(self.post_explanation_state.targets):
                dist_to_goal = float(np.linalg.norm(pos - goal))
                reached = dist_to_goal < HUMAN_GOAL_THRESHOLD
            elif human.mode == HumanMode.LISTENING and robot_xy_arr is not None and robot_yaw is not None:
                dist_to_robot = np.linalg.norm(pos - robot_xy_arr)
                in_sector = self._is_human_in_listening_front_sector(
                    human=human,
                    pos_xy=pos,
                    robot_xy=robot_xy_arr,
                    robot_yaw=robot_yaw,
                )
                human_in_listening_front_sector[i] = in_sector
                reached = (dist_to_robot > LISTEN_REACHED_MIN_DISTANCE) and in_sector
            else:
                dist_to_goal = float(np.linalg.norm(pos - goal))
                reached = dist_to_goal < HUMAN_GOAL_THRESHOLD

            if reached:
                human_reached_goal.append(i)
        return human_reached_goal, human_in_listening_front_sector

    def _collect_step_snapshot(
        self,
        robot_pose,
        dist,
        desired_yaw,
        actual_yaw,
        robot_mode,
        robot_action,
        human_xy,
        human_actual_yaw,
        human_goals,
        human_actions,
        nearest_human_distance,
        local_crowding_count_1m,
        nearest_human_distance_mean_1s,
        human_robot_distance,
        human_robot_distance_mean_1s,
        human_state_snapshot,
        human_in_listening_front_sector,
        human_reached_goal,
        final_waypoint_reached,
        all_humans_reached,
    ):
        """Collect one-step snapshot used for building info diagnostics."""
        # Single snapshot object consumed by _build_info and returned via info dict.
        rx, ry, _ = robot_pose
        gx, gy = self._get_goal_xy()
        robot_xy = np.array([rx, ry], dtype=np.float32)

        human_desired_yaw = np.arctan2(
            human_goals[:, 1] - human_xy[:, 1],
            human_goals[:, 0] - human_xy[:, 0],
        ).astype(np.float32)

        return {
            "robot_xy": robot_xy,
            "robot_goal_xy": np.array([gx, gy], dtype=np.float32),
            "dist_to_goal": float(dist),
            "robot_yaw": float(actual_yaw),
            "robot_desired_yaw": float(desired_yaw),
            "robot_mode": str(robot_mode),
            "robot_action": np.array(robot_action, dtype=np.float32),
            "human_xy": human_xy,
            "human_goals": human_goals,
            "human_actual_yaw": human_actual_yaw,
            "human_desired_yaw": human_desired_yaw,
            "human_actions": human_actions,
            "nearest_human_distance": np.array(nearest_human_distance, dtype=np.float32),
            "local_crowding_count_1m": np.array(local_crowding_count_1m, dtype=np.int32),
            "nearest_human_distance_mean_1s": np.array(nearest_human_distance_mean_1s, dtype=np.float32),
            "human_robot_distance": np.array(human_robot_distance, dtype=np.float32),
            "human_robot_distance_mean_1s": np.array(human_robot_distance_mean_1s, dtype=np.float32),
            "human_mode": human_state_snapshot["human_mode"],
            "human_profile": human_state_snapshot["human_profile"],
            "human_fuzzy_scores": human_state_snapshot["human_fuzzy_scores"],
            "human_fuzzy_dominant_state": human_state_snapshot["human_fuzzy_dominant_state"],
            "human_fuzzy_inputs": human_state_snapshot["human_fuzzy_inputs"],
            "human_fuzzy_context": human_state_snapshot["human_fuzzy_context"],
            "human_distracted_timer": human_state_snapshot["human_distracted_timer"],
            "human_following_steps": human_state_snapshot["human_following_steps"],
            "human_listening_steps": human_state_snapshot["human_listening_steps"],
            "human_in_listening_front_sector": human_in_listening_front_sector,
            "human_distracted_source": human_state_snapshot["human_distracted_source"],
            "human_overwhelmed_stage": human_state_snapshot["human_overwhelmed_stage"],
            "human_overwhelmed_leave_timer": human_state_snapshot["human_overwhelmed_leave_timer"],
            "human_impatient_timer": human_state_snapshot["human_impatient_timer"],
            "human_reached_goal": human_reached_goal,
            "final_waypoint_reached": final_waypoint_reached,
            "all_humans_reached": all_humans_reached,
            "active_overwhelmed_indices": human_state_snapshot["active_overwhelmed_indices"],
            "post_explanation_hold_active": self.post_explanation_state.active,
            "human_yield_role": list(self.post_explanation_state.roles),
            "human_yield_target_xy": np.array(self.post_explanation_state.targets, dtype=np.float32),
        }

    def _build_info(
        self,
        snapshot,
        events,
        truncated,
        external_action_received=False,
        external_action_used=False,
    ):
        """Build structured info dict for debugging/training analysis."""
        # Structured diagnostics for training/debugging dashboards.
        listen_wait_remaining = (
            max(0, self.listen_wait_steps - self.listen_state.wait_counter)
            if self.listen_state.wait_active
            else 0
        )
        listen_intro_delay_remaining = (
            max(0, self.listen_intro_delay_steps - self.listen_state.intro_counter)
            if self.listen_state.intro_active
            else 0
        )

        terminated_reason = None
        if events["final_listen_ready"]:
            terminated_reason = "final_listen_ready"
        elif truncated:
            terminated_reason = "max_steps"

        robot_action = snapshot["robot_action"]
        human_actions = snapshot["human_actions"]

        return {
            "events": {
                "entered_listen": bool(events["entered_listen"]),
                "started_listen_wait": bool(events["started_listen_wait"]),
                "completed_listen_wait": bool(events["completed_listen_wait"]),
                "final_listen_ready": bool(events["final_listen_ready"]),
                "overwhelmed_triggered": bool(events["overwhelmed_triggered"]),
                "callback_triggered": bool(events["callback_triggered"]),
                "callback_completed": bool(events["callback_completed"]),
                "callback_forced_recovery": bool(events["callback_forced_recovery"]),
                "callback_response_rejoin": bool(events["callback_response_rejoin"]),
                "callback_response_ignore": bool(events["callback_response_ignore"]),
                "callback_attempt_1_started": bool(events["callback_attempt_1_started"]),
                "callback_attempt_2_started": bool(events["callback_attempt_2_started"]),
                "callback_first_attempt_failed": bool(events["callback_first_attempt_failed"]),
                "callback_success": bool(events["callback_success"]),
                "happy_triggered": bool(events["happy_triggered"]),
                "happy_completed": bool(events["happy_completed"]),
            },
            "status": {
                "step_count": int(self.step_count),
                "follow_phase": self.follow_phase,
                "listen_mode": bool(self.robot.listen_mode),
                "listening_interrupted_for_callback": bool(self._is_listening_interrupted_for_callback()),
                "following_fuzzy_eval_period_seconds": float(self.following_fuzzy_eval_period_seconds),
                "following_fuzzy_eval_period_steps": int(self.following_fuzzy_eval_period_steps),
                "following_fuzzy_evaluated_this_step": bool(self._following_fuzzy_evaluated_this_step),
                "listening_fuzzy_eval_period_seconds": float(self.listening_fuzzy_eval_period_seconds),
                "listening_fuzzy_eval_period_steps": int(self.listening_fuzzy_eval_period_steps),
                "listening_fuzzy_evaluated_this_step": bool(self._listening_fuzzy_evaluated_this_step),
                "listen_intro_delay": {
                    "active": bool(self.listen_state.intro_active),
                    "counter": int(self.listen_state.intro_counter),
                    "steps": int(self.listen_intro_delay_steps),
                    "remaining": int(listen_intro_delay_remaining),
                    "is_final": bool(self.listen_state.intro_is_final),
                },
                "listen_wait": {
                    "active": bool(self.listen_state.wait_active),
                    "counter": int(self.listen_state.wait_counter),
                    "steps": int(self.listen_wait_steps),
                    "remaining": int(listen_wait_remaining),
                    "is_final": bool(self.listen_state.wait_is_final),
                },
                "post_explanation_hold": {
                    "active": bool(snapshot["post_explanation_hold_active"]),
                },
                "callback_active": bool(self.robot.callback_active),
                "callback_target_idx": (
                    int(self.robot.callback_target_idx)
                    if self.robot.callback_target_idx is not None
                    else None
                ),
                "callback_attempt_index": int(self.robot.callback_attempt_index),
                "callback_phase": (
                    str(self.robot.callback_phase)
                    if self.robot.callback_phase is not None
                    else None
                ),
                "callback_cue_elapsed_steps": int(self.robot.callback_cue_elapsed_steps),
                "callback_cue_total_steps": int(self.robot.callback_cue_total_steps),
                "callback_cue_remaining_steps": max(
                    0,
                    int(self.robot.callback_cue_total_steps) - int(self.robot.callback_cue_elapsed_steps),
                ),
                "callback_response_sampled": bool(self.robot.callback_response_sampled),
                "callback_last_response": (
                    str(self.env_callback_state.last_response)
                    if self.env_callback_state.last_response is not None
                    else None
                ),
                "callback_last_response_target_idx": (
                    int(self.env_callback_state.last_response_target_idx)
                    if self.env_callback_state.last_response_target_idx is not None
                    else None
                ),
                "robot_emotion": str(self.robot.emotion),
                "happy_remaining_steps": int(self.robot.happy_hold_steps_remaining),
                "happy_hold_seconds": float(ROBOT_HAPPY_HOLD_SECONDS),
                "speaker_active": bool(self.robot.speaker_active),
                "robot_text_label": self._get_robot_text_label(),
                "listening_distracted_window_active": [
                    bool(human.listening_distracted_window_active) for human in self.humans
                ],
                "callback_visual_active": bool(self._is_callback_visual_active()),
                "callback_trigger_distance_meters": float(self.callback_trigger_distance_meters),
                "perceived_distracted_indices": [int(idx) for idx in self.perceived_distracted_indices],
                "active_overwhelmed_indices": snapshot["active_overwhelmed_indices"],
                "external_action_received": bool(external_action_received),
                "external_action_used": bool(external_action_used),
                "terminated_reason": terminated_reason,
            },
            "metrics": {
                "humans": {
                    "nearest_human_distance": snapshot["nearest_human_distance"],
                    "local_crowding_count_1m": snapshot["local_crowding_count_1m"],
                    "nearest_human_distance_mean_1s": snapshot["nearest_human_distance_mean_1s"],
                    "human_robot_distance": snapshot["human_robot_distance"],
                    "human_robot_distance_mean_1s": snapshot["human_robot_distance_mean_1s"],
                    "window_seconds": float(HUMAN_HUMAN_DISTANCE_WINDOW_SECONDS),
                    "window_steps": int(self.hh_distance_metric.window_steps),
                    "observation_update_period_seconds": float(self.observation_update_period_seconds),
                    "observation_update_period_steps": int(self.observation_update_period_steps),
                    "observation_sample_age_steps": int(self._observation_sample_age_steps),
                    "observation_sample_age_seconds": float(
                        self._observation_sample_age_steps * self.timestep_float
                    ),
                },
            },
            "robot": {
                "pose_xy": snapshot["robot_xy"],
                "goal_xy": snapshot["robot_goal_xy"],
                "dist_to_goal": float(snapshot["dist_to_goal"]),
                "yaw": float(snapshot["robot_yaw"]),
                "desired_yaw": float(snapshot["robot_desired_yaw"]),
                "mode": str(snapshot["robot_mode"]),
                "action": {
                    "vx": float(robot_action[0]),
                    "vy": float(robot_action[1]),
                    "yaw_rate": float(robot_action[2]),
                },
                "emotion": str(self.robot.emotion),
                "final_waypoint_reached": bool(snapshot["final_waypoint_reached"]),
            },
            "humans": {
                "pose_xy": snapshot["human_xy"],
                "goal_xy": snapshot["human_goals"],
                "actual_yaw": snapshot["human_actual_yaw"],
                "desired_yaw": snapshot["human_desired_yaw"],
                "mode": snapshot["human_mode"],
                "profile": snapshot["human_profile"],
                "fuzzy_scores": snapshot["human_fuzzy_scores"],
                "fuzzy_dominant_state": snapshot["human_fuzzy_dominant_state"],
                "fuzzy_inputs": snapshot["human_fuzzy_inputs"],
                "fuzzy_context": snapshot["human_fuzzy_context"],
                "distracted_timer": snapshot["human_distracted_timer"],
                "following_steps": snapshot["human_following_steps"],
                "listening_steps": snapshot["human_listening_steps"],
                "in_listening_front_sector": snapshot["human_in_listening_front_sector"],
                "distracted_source": snapshot["human_distracted_source"],
                "overwhelmed_stage": snapshot["human_overwhelmed_stage"],
                "overwhelmed_leave_timer": snapshot["human_overwhelmed_leave_timer"],
                "impatient_timer": snapshot["human_impatient_timer"],
                "yield_role": snapshot["human_yield_role"],
                "yield_target_xy": snapshot["human_yield_target_xy"],
                "reached_goal_indices": snapshot["human_reached_goal"],
                "all_reached": bool(snapshot["all_humans_reached"]),
                "action": {
                    "vx": human_actions[:, 0],
                    "vy": human_actions[:, 1],
                    "yaw_rate": human_actions[:, 2],
                },
            },
        }

    def _get_obs(self):
        """Build compact observation [robot_x, robot_y, goal_dx, goal_dy]."""
        x, y, yaw = self._get_robot_pose()
        gx, gy = self._get_goal_xy()
        return np.array([x, y, gx - x, gy - y], dtype=np.float32)

    def render(self):
        """Render environment in human viewer or rgb array mode."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback=self._viewer_key_callback,
                )
            self._apply_label_options_to_viewer()
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
                    # Last-resort fallback to MuJoCo defaults when XML/driver limits are tighter.
                    self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, scene_option=self._label_scene_option)
            return self.renderer.render()
        return None

    def close(self):
        """Release MuJoCo viewer/renderer resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
