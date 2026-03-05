import logging
from importlib import resources

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from .env_info import build_info as build_env_info
from .env_info import collect_step_snapshot as collect_env_step_snapshot
from .env_info import default_events as default_env_events
from .env_runtime import EnvRuntimeState, apply_runtime_state
from .env_stepflows import step_active_branch as run_step_active_branch
from .env_stepflows import step_waiting_branch as run_step_waiting_branch
from .env_transitions import apply_callback_response_via_human_fsm
from .env_transitions import apply_fear_response_via_human_fsm
from .env_transitions import apply_human_mode_transition
from .env_transitions import apply_wait_robot_fsm_effects
from .env_transitions import build_active_human_fsm_ctx
from .env_transitions import build_callback_request
from .env_transitions import build_wait_robot_fsm_ctx
from .env_transitions import compute_move_back_action
from .env_transitions import get_nearest_attack_threat
from .env_transitions import is_robot_in_move_stage
from .env_transitions import refresh_callback_rearm_flags
from .env_transitions import resolve_fear_on_complete_via_human_fsm
from .env_transitions import sample_callback_response
from .env_transitions import sample_fear_response
from .env_visuals import apply_robot_base_color_from_robot_emotion
from .env_visuals import apply_robot_speaking_halo_visual
from .env_visuals import get_robot_text_label
from .env_visuals import sync_robot_speaker_state
from .env_visuals import sync_robot_text_label_visibility
from .env_visuals import update_robot_emotion_and_visual
from .human import Human, HumanMode
from .human_fsm import build_transition_table as build_human_fsm_transition_table
from .human_fsm import decide_mode as decide_human_fsm_mode
from .robot import ROBOT_WAYPOINT_REACHED_DIST, Robot
from .robot_fsm import build_transition_table as build_robot_fsm_transition_table

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
REPULSION_GAIN_DEFAULT = 6.0
FOLLOW_FAN_HALF_ANGLE_DEG = 85.0
LISTEN_FAN_HALF_ANGLE_DEG = 75.0
LISTEN_FAN_RADIUS_DEFAULT = 1.0
LISTEN_STAND_THRESHOLD_DEFAULT = 0.2
LISTEN_WAIT_STEPS_DEFAULT = 1000
OVERWHELMED_TARGET_IDX_DEFAULT = 1
OVERWHELMED_TRIGGER_WAIT_STEP_DEFAULT = 200
ATTACK_TARGET_IDX_DEFAULT = 2
ATTACK_TRIGGER_WAIT_STEP_DEFAULT = OVERWHELMED_TRIGGER_WAIT_STEP_DEFAULT
ATTACK_SPEED_DEFAULT = 1.0
ATTACK_HIT_DISTANCE_DEFAULT = 0.33
HUMAN_MAX_SPEED_DEFAULT = 1.67
FOLLOW_RADIUS_DEFAULT = 1.0
HUMAN_GOAL_THRESHOLD = 0.2
DIST_EPS = 1e-8
HUMAN1_DISTRACTED_PROB = 0.0005
HUMAN5_IMPATIENT_PROB = 0.0005
HUMAN_LABEL_SITE_GROUP = 2
ROBOT_EXPLANATION_LABEL_GROUP = 3
ROBOT_FOLLOWME_LABEL_GROUP = 4
ROBOT_NEED_SPACE_LABEL_GROUP = 5
HUMAN_LABEL_MODE = mujoco.mjtLabel.mjLABEL_SITE
CALLBACK_DISTRACTED_TRIGGER_STEPS = 400
CALLBACK_HOLD_SECONDS = 1.0
CALLBACK_TARGET_WHITELIST_DEFAULT = (0,)
CALLBACK_REJOIN_PROB = 0.4
CALLBACK_STAY_PROB = 0.3
CALLBACK_IGNORE_PROB = 0.3
CALLBACK_STAY_STEPS = 400
MOVE_BACK_SAFE_DISTANCE = SOCIAL_DISTANCE_DEFAULT
MOVE_BACK_SPEED = 0.6
ROBOT_HAPPY_HOLD_SECONDS = 1.0
ROBOT_FEAR_DISTANCE_THRESHOLD = 0.8
FEAR_RESPONSE_MOVE_BACK_PROB = 0.4
FEAR_RESPONSE_STAY_PROB = 0.3
FEAR_RESPONSE_CONTINUE_HIT_PROB = 0.3
ROBOT_COLOR_NATURAL = np.array([0.85, 0.85, 0.85, 1.0], dtype=np.float32)
ROBOT_COLOR_SAD = np.array([0.20, 0.45, 0.95, 1.0], dtype=np.float32)
ROBOT_COLOR_HAPPY = np.array([0.95, 0.85, 0.20, 1.0], dtype=np.float32)
ROBOT_COLOR_FEAR = np.array([0.62, 0.36, 0.88, 1.0], dtype=np.float32)
SPEAKING_HALO_RGBA_ON = np.array([1.0, 0.9, 0.2, 0.35], dtype=np.float32)
SPEAKING_HALO_RGBA_OFF = np.array([1.0, 0.9, 0.2, 0.0], dtype=np.float32)


class MuseumEnv(gym.Env):
    """
    Minimal runnable Gymnasium environment for a MuJoCo museum scene.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path=None,
        render_mode=None,
        enable_event_logs: bool = True,
        strict_action_validation: bool = True,
    ):
        super().__init__()
        self.enable_event_logs = bool(enable_event_logs)
        self.strict_action_validation = bool(strict_action_validation)
        logger.setLevel(logging.INFO if self.enable_event_logs else logging.CRITICAL + 1)

        if xml_path is None:
            with resources.path("museum_env.assets", "museum_scene.xml") as xml_file:
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

        # --- Action space ---
        self.nu = self.model.nu
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(self.nu,),
            dtype=np.float32,
        )

        # --- Observation space ---
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )

        self.timestep = self.model.opt.timestep
        self.max_steps = MAX_STEPS_DEFAULT
        self.step_count = 0

        # Waypoints: room A -> corridor -> room B
        waypoints = [
            (1.0, 5.0),
            (0.6, 4.5),
            (1.0, 2.0),
            (8.5, 2.0),
            (8.5, -10.0),
            (8.5, -12.5),
            (11, -12.5),
        ]

        # Robot agent
        self.robot = Robot(waypoints=waypoints, v_max=1.5, k_v=20.0, k_yaw=20.0)
        self.robot_fsm_table = build_robot_fsm_transition_table()
        self.human_fsm_table = build_human_fsm_transition_table()

        # Human follow switch (start with random walking)
        self.follow_humans = False
        self.robot_start_xy = None
        self.human_follow_distance = HUMAN_FOLLOW_DISTANCE_DEFAULT

        # Social distance (repulsion) parameters
        self.social_distance = SOCIAL_DISTANCE_DEFAULT
        self.repulsion_gain = REPULSION_GAIN_DEFAULT

        # Listening formation (fan around robot after it stops)
        self.follow_fan_half_angle = np.deg2rad(FOLLOW_FAN_HALF_ANGLE_DEG)
        self.listen_fan_half_angle = np.deg2rad(LISTEN_FAN_HALF_ANGLE_DEG)
        self.listen_fan_radius = LISTEN_FAN_RADIUS_DEFAULT
        self.listen_stand_threshold = LISTEN_STAND_THRESHOLD_DEFAULT
        self.listen_reached_logged = set()

        # Listening wait window
        self.listen_wait_steps = LISTEN_WAIT_STEPS_DEFAULT
        self.listen_wait_active = False
        self.listen_wait_counter = 0
        self.listen_wait_is_final = False
        self.listen_session_count = 0
        self.overwhelmed_triggered_once = False
        self.overwhelmed_target_idx = OVERWHELMED_TARGET_IDX_DEFAULT  # person2
        self.overwhelmed_trigger_wait_step = OVERWHELMED_TRIGGER_WAIT_STEP_DEFAULT
        self.attack_target_idx = ATTACK_TARGET_IDX_DEFAULT  # person3
        self.attack_trigger_wait_step = ATTACK_TRIGGER_WAIT_STEP_DEFAULT
        self.attack_triggered_once = False
        self.attack_hit_once = False
        self.callback_target_whitelist = set(CALLBACK_TARGET_WHITELIST_DEFAULT)
        self.callback_triggered_for_current_distracted = []
        self.callback_active_target_idx = None
        self.callback_last_response = None
        self.callback_last_response_target_idx = None
        self.move_back_active = False
        self.move_back_attacker_idx = None
        self.fear_active = False
        self.fear_attacker_idx = None
        self.fear_current_response_mode = None
        self.fear_current_response_target_idx = None
        self.fear_current_freeze_attack = False
        self.fear_last_response = None
        self.fear_last_response_target_idx = None

        # --- Initialize humans ---
        self.humans = [
            Human("person1", "person1", qpos_idx=3, max_speed=HUMAN_MAX_SPEED_DEFAULT),
            Human("person2", "person2", qpos_idx=6, max_speed=HUMAN_MAX_SPEED_DEFAULT),
            Human("person3", "person3", qpos_idx=9, max_speed=HUMAN_MAX_SPEED_DEFAULT),
            Human("person4", "person4", qpos_idx=12, max_speed=HUMAN_MAX_SPEED_DEFAULT),
            Human("person5", "person5", qpos_idx=15, max_speed=HUMAN_MAX_SPEED_DEFAULT),
        ]
        for human in self.humans:
            human.external_waypoint = False
            human.set_mode(HumanMode.WANDERING)
            human.set_event_logging(self.enable_event_logs)
        self.callback_triggered_for_current_distracted = [False] * len(self.humans)

        self._configure_human_following_variants()

        self.humans[1].can_be_overwhelmed = True
        if len(self.humans) > 2:
            self.humans[2].can_attack = True
            self.humans[2].attack_speed = ATTACK_SPEED_DEFAULT
            self.humans[2].attack_hit_distance = ATTACK_HIT_DISTANCE_DEFAULT
        if len(self.humans) > 4:
            self.humans[4].impatient_duration = 2000
            self.humans[4].impatient_speed_multiplier = 1.5
            self.humans[4].impatient_front_offset = 1.0

        # Cache MuJoCo body ids (static across episodes).
        self.robot_body_id = self.model.body("robot").id
        self.robot_base_geom_id = self.model.geom("robot_base").id
        self.robot_speaking_halo_geom_id = self.model.geom("robot_speaking_halo").id
        self.human_body_ids = [self.model.body(human.body_name).id for human in self.humans]
        self._label_scene_option = self._build_label_scene_option()
        self._apply_robot_base_color_from_robot_emotion()
        self._sync_robot_speaker_state()
        self._sync_robot_text_label_visibility()
        self._apply_robot_speaking_halo_visual()

    def _log_event(self, msg: str):
        if self.enable_event_logs:
            logger.info(msg)

    def _configure_human_following_variants(self):
        for human in self.humans:
            human.configure_following_variant(None, 0.0)
        if len(self.humans) > 0:
            self.humans[0].configure_following_variant(HumanMode.DISTRACTED, HUMAN1_DISTRACTED_PROB)
        if len(self.humans) > 4:
            self.humans[4].configure_following_variant(HumanMode.IMPATIENT, HUMAN5_IMPATIENT_PROB)

    def _build_label_scene_option(self):
        opt = mujoco.MjvOption()
        opt.label = HUMAN_LABEL_MODE
        opt.sitegroup[:] = 0
        opt.sitegroup[HUMAN_LABEL_SITE_GROUP] = 1
        opt.sitegroup[ROBOT_EXPLANATION_LABEL_GROUP] = 0
        opt.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP] = 0
        opt.sitegroup[ROBOT_NEED_SPACE_LABEL_GROUP] = 0
        return opt

    def _apply_label_options_to_viewer(self):
        if self.viewer is None:
            return
        self.viewer.opt.label = self._label_scene_option.label
        self.viewer.opt.sitegroup[:] = self._label_scene_option.sitegroup

    def set_viewer_key_callback(self, callback):
        if callback is not None and not callable(callback):
            raise ValueError("viewer key callback must be callable or None.")
        self._viewer_key_callback = callback

    @staticmethod
    def _default_events():
        return default_env_events()

    def _validate_external_action(self, action):
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
        x = float(self.data.xpos[self.robot_body_id, 0])
        y = float(self.data.xpos[self.robot_body_id, 1])
        yaw = float(self.data.qpos[2])
        return x, y, yaw

    def _get_human_poses(self):
        humans_xyz = []
        for human, human_body_id in zip(self.humans, self.human_body_ids):
            x = float(self.data.xpos[human_body_id, 0])
            y = float(self.data.xpos[human_body_id, 1])
            yaw = float(self.data.qpos[human.qpos_idx + 2])
            humans_xyz.append([x, y, yaw])
        return np.array(humans_xyz, dtype=np.float32)

    def _get_goal_xy(self):
        goal_xy = self.robot.get_current_waypoint()
        return float(goal_xy[0]), float(goal_xy[1])

    def _is_robot_in_move_stage(self, robot_pose):
        return is_robot_in_move_stage(
            env=self,
            robot_pose=robot_pose,
            waypoint_reached_dist=ROBOT_WAYPOINT_REACHED_DIST,
            dist_eps=DIST_EPS,
        )

    def _refresh_callback_rearm_flags(self):
        refresh_callback_rearm_flags(self)

    def _build_active_human_fsm_ctx(self):
        return build_active_human_fsm_ctx(self, callback_stay_steps=CALLBACK_STAY_STEPS)

    def _build_wait_robot_fsm_ctx(self, threat_exists: bool, threat_dist):
        return build_wait_robot_fsm_ctx(
            env=self,
            threat_exists=threat_exists,
            threat_dist=threat_dist,
            move_back_safe_distance=MOVE_BACK_SAFE_DISTANCE,
        )

    def _apply_wait_robot_fsm_effects(
        self,
        wait_effects: dict,
        threat,
        move_back_was_active: bool,
        robot_xy: np.ndarray,
        events: dict,
    ):
        return apply_wait_robot_fsm_effects(
            env=self,
            wait_effects=wait_effects,
            threat=threat,
            move_back_was_active=move_back_was_active,
            robot_xy=robot_xy,
            events=events,
            move_back_speed=MOVE_BACK_SPEED,
            dist_eps=DIST_EPS,
        )

    def _apply_human_mode_transition(self, human: Human, human_fsm: dict, reason: str) -> bool:
        return apply_human_mode_transition(
            env=self,
            human=human,
            human_fsm=human_fsm,
            reason=reason,
            callback_stay_steps=CALLBACK_STAY_STEPS,
        )

    def _build_callback_request(self, human_xy, robot_pose):
        return build_callback_request(
            env=self,
            human_xy=human_xy,
            robot_pose=robot_pose,
            waypoint_reached_dist=ROBOT_WAYPOINT_REACHED_DIST,
            dist_eps=DIST_EPS,
            callback_hold_seconds=CALLBACK_HOLD_SECONDS,
            callback_distracted_trigger_steps=CALLBACK_DISTRACTED_TRIGGER_STEPS,
        )

    def _get_nearest_attack_threat(self, robot_xy, human_xy):
        return get_nearest_attack_threat(
            env=self,
            robot_xy=robot_xy,
            human_xy=human_xy,
        )

    @staticmethod
    def _compute_move_back_action(robot_xy, threat_xy):
        return compute_move_back_action(
            robot_xy=robot_xy,
            threat_xy=threat_xy,
            move_back_speed=MOVE_BACK_SPEED,
            dist_eps=DIST_EPS,
        )

    @staticmethod
    def _sample_callback_response():
        return sample_callback_response(
            rejoin_prob=CALLBACK_REJOIN_PROB,
            stay_prob=CALLBACK_STAY_PROB,
            ignore_prob=CALLBACK_IGNORE_PROB,
        )

    @staticmethod
    def _sample_fear_response():
        return sample_fear_response(
            move_back_prob=FEAR_RESPONSE_MOVE_BACK_PROB,
            stay_prob=FEAR_RESPONSE_STAY_PROB,
            continue_hit_prob=FEAR_RESPONSE_CONTINUE_HIT_PROB,
        )

    def _apply_callback_response_via_human_fsm(self, recover_idx: int, events: dict):
        response = self._sample_callback_response()
        return apply_callback_response_via_human_fsm(
            env=self,
            recover_idx=recover_idx,
            callback_response=response,
            callback_stay_steps=CALLBACK_STAY_STEPS,
            happy_hold_seconds=ROBOT_HAPPY_HOLD_SECONDS,
            events=events,
        )

    def _apply_fear_response_on_trigger(self, events):
        if not events.get("fear_triggered", False):
            return
        return apply_fear_response_via_human_fsm(
            env=self,
            response=self._sample_fear_response(),
            callback_stay_steps=CALLBACK_STAY_STEPS,
            events=events,
        )

    def _resolve_fear_response_on_complete(self, events):
        resolve_fear_on_complete_via_human_fsm(self, events)

    def _apply_robot_base_color_from_robot_emotion(self):
        apply_robot_base_color_from_robot_emotion(
            env=self,
            color_fear=ROBOT_COLOR_FEAR,
            color_sad=ROBOT_COLOR_SAD,
            color_happy=ROBOT_COLOR_HAPPY,
            color_natural=ROBOT_COLOR_NATURAL,
        )

    def _sync_robot_speaker_state(self):
        sync_robot_speaker_state(self)

    def _sync_robot_text_label_visibility(self):
        sync_robot_text_label_visibility(
            env=self,
            need_space_group=ROBOT_NEED_SPACE_LABEL_GROUP,
            followme_group=ROBOT_FOLLOWME_LABEL_GROUP,
            explanation_group=ROBOT_EXPLANATION_LABEL_GROUP,
        )

    def _apply_robot_speaking_halo_visual(self):
        apply_robot_speaking_halo_visual(
            env=self,
            halo_rgba_on=SPEAKING_HALO_RGBA_ON,
            halo_rgba_off=SPEAKING_HALO_RGBA_OFF,
        )

    def _get_robot_text_label(self):
        return get_robot_text_label(self)

    def _update_robot_emotion_and_visual(self, events, robot_xy, human_xy):
        update_robot_emotion_and_visual(
            env=self,
            events=events,
            robot_xy=robot_xy,
            human_xy=human_xy,
            fear_distance_threshold=ROBOT_FEAR_DISTANCE_THRESHOLD,
            color_fear=ROBOT_COLOR_FEAR,
            color_sad=ROBOT_COLOR_SAD,
            color_happy=ROBOT_COLOR_HAPPY,
            color_natural=ROBOT_COLOR_NATURAL,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0

        # Reset robot agent
        self.robot.reset()

        # Store robot start position (for follow trigger)
        rx, ry, _ = self._get_robot_pose()
        self.robot_start_xy = np.array([rx, ry], dtype=np.float32)
        self.listen_reached_logged = set()
        apply_runtime_state(
            env=self,
            state=EnvRuntimeState(),
            n_humans=len(self.humans),
        )

        # Reset humans
        for human in self.humans:
            human.reset_episode_state()
        self._configure_human_following_variants()
        self._apply_robot_base_color_from_robot_emotion()
        self._sync_robot_speaker_state()
        self._sync_robot_text_label_visibility()
        self._apply_robot_speaking_halo_visual()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action=None):
        """
        Robot rule-based navigation (via Robot class) + human walking.
        """
        external_action_received = self._validate_external_action(action)
        self.step_count += 1

        if self.listen_wait_active:
            return self._step_waiting_branch(external_action_received=external_action_received)

        return self._step_active_branch(external_action_received=external_action_received)

    def _step_waiting_branch(self, external_action_received=False):
        return run_step_waiting_branch(
            env=self,
            external_action_received=external_action_received,
        )

    def _maybe_trigger_overwhelmed_in_wait(self, robot_xy, human_xy):
        # Trigger on the 200th waiting step of the first listen session only once per episode.
        if self.overwhelmed_triggered_once:
            return False
        if self.listen_session_count != 1:
            return False
        if (self.listen_wait_counter + 1) != self.overwhelmed_trigger_wait_step:
            return False

        idx = self.overwhelmed_target_idx
        if idx < 0 or idx >= len(self.humans) or idx >= human_xy.shape[0]:
            return False

        human = self.humans[idx]
        if not human.can_be_overwhelmed:
            return False

        human.start_overwhelmed(robot_xy=robot_xy, current_xy=human_xy[idx])
        self.overwhelmed_triggered_once = True
        return True

    def _maybe_trigger_attack_in_wait(self, robot_xy, human_xy):
        if self.attack_triggered_once:
            return False
        if self.listen_session_count != 1:
            return False
        if (self.listen_wait_counter + 1) != self.attack_trigger_wait_step:
            return False

        idx = self.attack_target_idx
        if idx < 0 or idx >= len(self.humans) or idx >= human_xy.shape[0]:
            return False

        human = self.humans[idx]
        if not human.can_attack:
            return False

        started = human.start_attack()
        if not started:
            return False

        self.attack_triggered_once = True
        return True

    def _step_active_branch(self, external_action_received=False):
        return run_step_active_branch(
            env=self,
            external_action_received=external_action_received,
        )

    def _compute_social_repulsion(self, human_xy):
        if not human_xy.size:
            return [np.zeros(2, dtype=np.float32) for _ in self.humans]

        repulsion_vectors = []
        for i in range(human_xy.shape[0]):
            pos = human_xy[i]
            diff = pos - human_xy
            neighbor_dist = np.linalg.norm(diff, axis=1)
            mask = (neighbor_dist > 1e-6) & (neighbor_dist < self.social_distance)
            if np.any(mask):
                directions = diff[mask] / neighbor_dist[mask][:, None]
                strengths = (self.social_distance - neighbor_dist[mask]) / self.social_distance
                repulsion = (directions * strengths[:, None]).sum(axis=0)
            else:
                repulsion = np.zeros(2, dtype=np.float32)
            repulsion_vectors.append(self.repulsion_gain * repulsion)

        return repulsion_vectors

    def _update_humans_and_apply_ctrl(self, rx, ry, ryaw, repulsion_vectors):
        human_actions = []
        n_humans = len(self.humans)
        follow_radius = FOLLOW_RADIUS_DEFAULT
        human_fsm_base_ctx = self._build_active_human_fsm_ctx()

        for i, human in enumerate(self.humans):
            repulsion_vec = repulsion_vectors[i] if i < len(repulsion_vectors) else np.zeros(2, dtype=np.float32)

            ctx = {
                "robot_xy": np.array([rx, ry], dtype=np.float32),
                "robot_yaw": ryaw,
                "repulsion": repulsion_vec,
                "stand_threshold": self.listen_stand_threshold,
            }

            mode_before = str(human.mode)
            human_fsm = decide_human_fsm_mode(
                current_mode=mode_before,
                ctx=human_fsm_base_ctx,
                table=self.human_fsm_table,
            )
            self._apply_human_mode_transition(
                human=human,
                human_fsm=human_fsm,
                reason="env_human_fsm",
            )

            if self.follow_humans and human.mode in (HumanMode.FOLLOWING, HumanMode.IMPATIENT):
                human.set_context(
                    index=i,
                    n_humans=n_humans,
                    robot_pose=(rx, ry, ryaw),
                    follow_radius=follow_radius,
                    fan_half_angle=self.follow_fan_half_angle,
                    impatient_front_offset=human.impatient_front_offset,
                    robot_xy=np.array([rx, ry], dtype=np.float32),
                    robot_yaw=ryaw,
                )

            human_action = human.step(self.model, self.data, ctx)
            human_actions.append(human_action)

            ctrl_idx = 3 + i * 3
            self.data.ctrl[ctrl_idx:ctrl_idx + 3] = human_action

        if human_actions:
            return np.array(human_actions, dtype=np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    def _check_human_goals(self, human_xy, human_goals):
        human_reached_goal = []
        for i, (pos, goal) in enumerate(zip(human_xy, human_goals)):
            dist_to_goal = float(np.linalg.norm(pos - goal))
            if dist_to_goal < HUMAN_GOAL_THRESHOLD:
                human_reached_goal.append(i)
                if self.robot.listen_mode and i not in self.listen_reached_logged:
                    self.listen_reached_logged.add(i)
                    self._log_event(f">>> person{i+1} reached their goal at step {self.step_count}!")
        return human_reached_goal

    def _handle_listen_transitions(self, final_waypoint_reached, all_humans_reached):
        events = {
            "started_listen_wait": False,
            "completed_listen_wait": False,
            "final_listen_ready": False,
        }

        # Listening complete condition: all humans reached.
        # Start waiting window instead of switching immediately.
        if self.robot.listen_mode and all_humans_reached and not self.listen_wait_active:
            self.listen_wait_active = True
            self.listen_wait_counter = 0
            self.listen_wait_is_final = bool(final_waypoint_reached)
            events["started_listen_wait"] = True
            self._log_event(f">>> Listening targets reached. Start wait for {self.listen_wait_steps} steps.")

        return events

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
        human_v_follow,
        human_v_repulsion,
        human_v_hr,
        human_reached_goal,
        final_waypoint_reached,
        all_humans_reached,
    ):
        return collect_env_step_snapshot(
            env=self,
            robot_pose=robot_pose,
            dist=dist,
            desired_yaw=desired_yaw,
            actual_yaw=actual_yaw,
            robot_mode=robot_mode,
            robot_action=robot_action,
            human_xy=human_xy,
            human_actual_yaw=human_actual_yaw,
            human_goals=human_goals,
            human_actions=human_actions,
            human_v_follow=human_v_follow,
            human_v_repulsion=human_v_repulsion,
            human_v_hr=human_v_hr,
            human_reached_goal=human_reached_goal,
            final_waypoint_reached=final_waypoint_reached,
            all_humans_reached=all_humans_reached,
        )

    def _build_info(
        self,
        snapshot,
        events,
        truncated,
        external_action_received=False,
        external_action_used=False,
    ):
        return build_env_info(
            env=self,
            snapshot=snapshot,
            events=events,
            truncated=truncated,
            move_back_safe_distance=MOVE_BACK_SAFE_DISTANCE,
            move_back_speed=MOVE_BACK_SPEED,
            happy_hold_seconds=ROBOT_HAPPY_HOLD_SECONDS,
            fear_distance_threshold=ROBOT_FEAR_DISTANCE_THRESHOLD,
            robot_text_label=self._get_robot_text_label(),
            external_action_received=external_action_received,
            external_action_used=external_action_used,
        )

    def _get_obs(self):
        x, y, yaw = self._get_robot_pose()
        gx, gy = self._get_goal_xy()
        return np.array([x, y, gx - x, gy - y], dtype=np.float32)

    def render(self):
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
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
