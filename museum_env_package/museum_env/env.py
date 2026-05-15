import logging
from importlib import resources
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from . import env_control, env_flow
from .env_constants import (
    ACTION_HIGH,
    ACTION_LOW,
    CALLBACK_IMPATIENT_IGNORE_PROB_ND_DEFAULT,
    CALLBACK_IMPATIENT_IGNORE_PROB_NORMAL_DEFAULT,
    CALLBACK_IMPATIENT_REJOIN_PROB_ND_DEFAULT,
    CALLBACK_IMPATIENT_REJOIN_PROB_NORMAL_DEFAULT,
    CALLBACK_IGNORE_PROB_ND_DEFAULT,
    CALLBACK_IGNORE_PROB_NORMAL_DEFAULT,
    CALLBACK_OVERWHELMED_IGNORE_PROB_ND_DEFAULT,
    CALLBACK_OVERWHELMED_IGNORE_PROB_NORMAL_DEFAULT,
    CALLBACK_OVERWHELMED_REJOIN_PROB_ND_DEFAULT,
    CALLBACK_OVERWHELMED_REJOIN_PROB_NORMAL_DEFAULT,
    CALLBACK_REJOIN_PROB_ND_DEFAULT,
    CALLBACK_REJOIN_PROB_NORMAL_DEFAULT,
    CALLBACK_TRIGGER_DISTANCE_METERS_DEFAULT,
    DEFAULT_MAP_NAME,
    FOLLOWING_CALLBACK_CUE_SECONDS,
    FOLLOWING_CALLBACK_FRONT_SECTOR_HALF_ANGLE_DEG,
    FOLLOWING_CALLBACK_WAIT_SECONDS,
    FOLLOW_FAN_HALF_ANGLE_DEG,
    HUMAN_WALL_FOOTPRINT_RADIUS,
    HUMAN_FOLLOW_DISTANCE_DEFAULT,
    HUMAN_GOAL_THRESHOLD,
    HUMAN_HUMAN_DISTANCE_WINDOW_SECONDS,
    HUMAN_MAX_SPEED_DEFAULT,
    HUMAN_SPAWN_MAX_ATTEMPTS_PER_HUMAN,
    HUMAN_SPAWN_MIN_DISTANCE,
    HUMAN_SPAWN_MIN_ROBOT_DISTANCE,
    IMPATIENT_FAN_HALF_ANGLE_DEG,
    INACTIVE_HUMAN_PARK_X,
    INACTIVE_HUMAN_PARK_Y_BASE,
    LISTEN_DISTANCE_SHORTEN_SECONDS_PER_HUMAN,
    LISTEN_FAN_RADIUS_DEFAULT,
    LISTEN_INTRO_DELAY_SECONDS_DEFAULT,
    LISTEN_QUESTION_AFTER_EXPLANATION_PROBABILITY_DEFAULT,
    LISTEN_QUESTION_PAUSE_SECONDS_DEFAULT,
    LISTEN_QUESTION_PROBABILITY_DEFAULT,
    LISTEN_REACHED_MIN_DISTANCE,
    LISTEN_STAND_THRESHOLD_DEFAULT,
    LISTEN_WAIT_SECONDS_DEFAULT,
    LISTENING_FRONT_SECTOR_HALF_ANGLE_DEG,
    MAX_DISTRACTED_DURATION_SECONDS_DEFAULT,
    MAX_HUMANS_CAPACITY,
    MAX_STEPS_DEFAULT,
    POST_EXPLANATION_HOLD_RESUME_DISTANCE,
    POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD,
    REPULSION_GAIN_DEFAULT,
    SOCIAL_DISTANCE_DEFAULT,
)
from .env_reporting import (
    HUMAN_SPEAKING_HALO_RGBA_OFF,
    HUMAN_SPEAKING_HALO_RGBA_ON,
    apply_label_scene_option_to_viewer,
    apply_robot_visual_state,
    build_label_scene_option,
    build_step_info,
    resolve_robot_visual_state,
)
from .env_runtime import build_human_goals, build_world_frame, compute_reached_goal_indices
from .env_state import (
    ListeningState,
    PersonalSpaceBackoffState,
    PostExplanationState,
    RuntimeCache,
    StepEvents,
    build_fuzzy_debug_states,
)
from .human import Human, HumanMode, HumanProfile, LISTENING_IMPATIENT_GLANCE_SECONDS_DEFAULT
from .map_layouts import MapLayout, get_map_layout
from .metrics import VectorizedRollingWindow
from .robot import Robot

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


class MuseumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path=None,
        map_name: str = DEFAULT_MAP_NAME,
        map_layout: Optional[MapLayout] = None,
        render_mode=None,
        enable_event_logs: bool = True,
        max_distracted_duration_seconds: float = MAX_DISTRACTED_DURATION_SECONDS_DEFAULT,
        callback_rejoin_prob_normal: float = CALLBACK_REJOIN_PROB_NORMAL_DEFAULT,
        callback_ignore_prob_normal: float = CALLBACK_IGNORE_PROB_NORMAL_DEFAULT,
        callback_rejoin_prob_nd: float = CALLBACK_REJOIN_PROB_ND_DEFAULT,
        callback_ignore_prob_nd: float = CALLBACK_IGNORE_PROB_ND_DEFAULT,
        callback_overwhelmed_rejoin_prob_normal: float = CALLBACK_OVERWHELMED_REJOIN_PROB_NORMAL_DEFAULT,
        callback_overwhelmed_ignore_prob_normal: float = CALLBACK_OVERWHELMED_IGNORE_PROB_NORMAL_DEFAULT,
        callback_overwhelmed_rejoin_prob_nd: float = CALLBACK_OVERWHELMED_REJOIN_PROB_ND_DEFAULT,
        callback_overwhelmed_ignore_prob_nd: float = CALLBACK_OVERWHELMED_IGNORE_PROB_ND_DEFAULT,
        callback_impatient_rejoin_prob_normal: float = CALLBACK_IMPATIENT_REJOIN_PROB_NORMAL_DEFAULT,
        callback_impatient_ignore_prob_normal: float = CALLBACK_IMPATIENT_IGNORE_PROB_NORMAL_DEFAULT,
        callback_impatient_rejoin_prob_nd: float = CALLBACK_IMPATIENT_REJOIN_PROB_ND_DEFAULT,
        callback_impatient_ignore_prob_nd: float = CALLBACK_IMPATIENT_IGNORE_PROB_ND_DEFAULT,
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
        self.observation_update_period_steps = self._steps(self.observation_update_period_seconds)
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
        self.listen_intro_delay_steps = self._steps(self.listen_intro_delay_seconds)
        self.listen_wait_seconds = LISTEN_WAIT_SECONDS_DEFAULT
        self.listen_wait_steps = self._steps(self.listen_wait_seconds)
        self.listen_question_probability = float(listen_question_probability)
        self.listen_question_after_explanation_probability = float(
            listen_question_after_explanation_probability
        )
        self.listen_question_pause_seconds = float(LISTEN_QUESTION_PAUSE_SECONDS_DEFAULT)
        self.listen_question_pause_steps = self._steps(self.listen_question_pause_seconds)
        self.listen_distance_shorten_steps = self._steps(LISTEN_DISTANCE_SHORTEN_SECONDS_PER_HUMAN)
        self.following_callback_wait_steps = self._steps(FOLLOWING_CALLBACK_WAIT_SECONDS)
        self.following_callback_cue_steps = self._steps(FOLLOWING_CALLBACK_CUE_SECONDS)
        self.following_callback_front_sector_half_angle = np.deg2rad(
            FOLLOWING_CALLBACK_FRONT_SECTOR_HALF_ANGLE_DEG
        )
        self.post_explanation_resume_speed_threshold = float(
            POST_EXPLANATION_HOLD_RESUME_SPEED_THRESHOLD
        )
        self.post_explanation_resume_distance = float(POST_EXPLANATION_HOLD_RESUME_DISTANCE)

        self.listening_state = ListeningState()
        self.post_explanation_state = PostExplanationState()
        self.personal_space_backoff_state = PersonalSpaceBackoffState()
        self.runtime_cache = RuntimeCache()
        self.max_distracted_duration_seconds = float(max_distracted_duration_seconds)
        self.callback_trigger_distance_meters = float(callback_trigger_distance_meters)
        env_control.reset_following_wait_episode(self)
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
        self.callback_response_profile_probs_by_mode = {
            HumanMode.DISTRACTED: self.callback_response_profile_probs,
            HumanMode.OVERWHELMED: {
                HumanProfile.NORMAL: {
                    "rejoin": float(callback_overwhelmed_rejoin_prob_normal),
                    "ignore": float(callback_overwhelmed_ignore_prob_normal),
                },
                HumanProfile.NEURODIVERGENT: {
                    "rejoin": float(callback_overwhelmed_rejoin_prob_nd),
                    "ignore": float(callback_overwhelmed_ignore_prob_nd),
                },
            },
            HumanMode.IMPATIENT: {
                HumanProfile.NORMAL: {
                    "rejoin": float(callback_impatient_rejoin_prob_normal),
                    "ignore": float(callback_impatient_ignore_prob_normal),
                },
                HumanProfile.NEURODIVERGENT: {
                    "rejoin": float(callback_impatient_rejoin_prob_nd),
                    "ignore": float(callback_impatient_ignore_prob_nd),
                },
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

    def _steps(self, seconds: float) -> int:
        return max(1, int(round(float(seconds) / self.dt)))

    def _robot_xy_from_data(self) -> np.ndarray:
        return np.array(
            [
                float(self.data.xpos[self.robot_body_id, 0]),
                float(self.data.xpos[self.robot_body_id, 1]),
            ],
            dtype=np.float32,
        )

    def _build_world_frame(self, *, force: bool = False, tick: bool = False):
        return build_world_frame(
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
            force_observations=force,
            tick_age_before_refresh=tick,
        )

    def _build_observation(self, world_frame) -> np.ndarray:
        gx, gy = self.robot.get_current_waypoint()
        x, y = world_frame.robot_xy
        return np.array([x, y, gx - x, gy - y], dtype=np.float32)

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
        self.runtime_cache.reset()
        if hasattr(self, "all_human_body_ids"):
            self.human_body_ids = self.all_human_body_ids[: self.n_humans]

    def _configure_human_behaviors(self) -> None:
        for human in self.humans:
            human.apply_runtime_config(
                dt=self.dt,
                max_distracted_duration_seconds=self.max_distracted_duration_seconds,
                impatient_duration_seconds=6.0,
                impatient_speed_multiplier=1.5,
                impatient_front_offset=1.0,
                listening_impatient_glance_seconds=LISTENING_IMPATIENT_GLANCE_SECONDS_DEFAULT,
            )

    def _sample_active_human_spawn_states(self, robot_xy) -> list[np.ndarray]:
        sampled_states: list[np.ndarray] = []
        sampled_positions: list[np.ndarray] = []
        for human in self.humans:
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
            else:
                raise RuntimeError(
                    f"Failed to sample spawn state for {human.name} "
                    f"after {HUMAN_SPAWN_MAX_ATTEMPTS_PER_HUMAN} attempts."
                )
        return sampled_states

    def _reset_human_positions(self, robot_xy) -> None:
        active_spawn_states = self._sample_active_human_spawn_states(
            robot_xy=np.asarray(robot_xy, dtype=np.float32)
        )
        for human, spawn_state in zip(self.humans, active_spawn_states):
            self.data.qpos[human.qpos_idx : human.qpos_idx + 3] = spawn_state
            self.data.qvel[human.qpos_idx : human.qpos_idx + 3] = 0.0
            human.current_waypoint = self.map_layout.sample_spawn_point(
                HUMAN_WALL_FOOTPRINT_RADIUS,
                rng=self.np_random,
            )

        for park_idx, human in enumerate(self.all_humans[len(self.humans) :]):
            park_pose = np.array(
                [INACTIVE_HUMAN_PARK_X + park_idx, INACTIVE_HUMAN_PARK_Y_BASE, 0.0],
                dtype=np.float32,
            )
            self.data.qpos[human.qpos_idx : human.qpos_idx + 3] = park_pose
            self.data.qvel[human.qpos_idx : human.qpos_idx + 3] = 0.0

    def _sync_robot_speaker_state(self) -> None:
        env_flow.sync_robot_speaker_state(self)

    def _sync_robot_visual_state(self, force: bool = False) -> None:
        visual_state = resolve_robot_visual_state(
            robot=self.robot,
            callback_visual_active=bool(self.robot.callback_active),
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
            halo_rgba = (
                HUMAN_SPEAKING_HALO_RGBA_ON
                if idx < active_count and bool(self.all_humans[idx].speaking_active)
                else HUMAN_SPEAKING_HALO_RGBA_OFF
            )
            self.model.geom_rgba[geom_id] = halo_rgba

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
        self.runtime_cache.reset()
        self.fuzzy_debug = build_fuzzy_debug_states(len(self.humans))
        self.hh_distance_metric.reset()
        self.hr_distance_metric.reset()
        env_control.reset_following_wait_episode(self)
        self.personal_space_backoff_state.reset()

        for human in self.humans:
            human.reset_episode_state()
        self._configure_human_behaviors()

        initial_robot_xy = self._robot_xy_from_data()
        self._reset_human_positions(initial_robot_xy)
        mujoco.mj_forward(self.model, self.data)

        self.robot_start_xy = self._robot_xy_from_data()
        for human in self.humans:
            human.reset_listening_session_state()

        world_frame = self._build_world_frame(force=True)
        self._sync_robot_speaker_state()
        self._sync_robot_visual_state(force=True)
        self._sync_human_visual_state()
        return self._build_observation(world_frame), {}

    def step(self, action=None):
        del action
        self.step_count += 1
        events = StepEvents()

        pre_frame = self._build_world_frame()
        robot_action, _ = env_flow.compute_robot_action(self, pre_frame, events)
        env_flow.maybe_finish_post_explanation_hold(
            self,
            pre_frame.robot_xy,
            float(np.hypot(robot_action[0], robot_action[1])),
        )
        env_flow.maybe_activate_follow_phase_from_robot_progress(self, pre_frame.robot_xy)
        robot_action = env_control.apply_robot_personal_space_backoff_if_needed(
            self,
            robot_action,
            pre_frame,
        )

        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = robot_action
        env_control.apply_human_controls(self, pre_frame)

        mujoco.mj_step(self.model, self.data)
        post_frame = self._build_world_frame(tick=True)

        env_control.update_human_listening_session_progress(self)
        env_flow.progress_listening_phase(self, events, post_frame)
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
        distracted_indices = [
            int(idx) for idx, mode in enumerate(human_modes) if mode == HumanMode.DISTRACTED
        ]
        self.robot.update_emotion()
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
            perceived_distracted_indices=distracted_indices,
        )
        reward = -float(info["robot"]["dist_to_goal"])
        return self._build_observation(post_frame), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            apply_label_scene_option_to_viewer(
                viewer=self.viewer,
                label_scene_option=self._label_scene_option,
            )
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
