import logging
from importlib import resources

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from .human import Human, HumanMode
from .robot import ROBOT_WAYPOINT_REACHED_DIST, Robot, RobotEmotion, RobotMode

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
HUMAN_LABEL_MODE = mujoco.mjtLabel.mjLABEL_SITE
CALLBACK_DISTRACTED_TRIGGER_STEPS = 400
CALLBACK_HOLD_SECONDS = 1.0
CALLBACK_TARGET_WHITELIST_DEFAULT = (0,)
MOVE_BACK_SAFE_DISTANCE = SOCIAL_DISTANCE_DEFAULT
MOVE_BACK_SPEED = 0.6
ROBOT_COLOR_NATURAL = np.array([0.85, 0.85, 0.85, 1.0], dtype=np.float32)
ROBOT_COLOR_SAD = np.array([0.20, 0.45, 0.95, 1.0], dtype=np.float32)


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
        self.move_back_active = False
        self.move_back_attacker_idx = None

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
        self.human_body_ids = [self.model.body(human.body_name).id for human in self.humans]
        self._label_scene_option = self._build_label_scene_option()
        self._apply_robot_base_color_from_robot_emotion()

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
        return {
            "entered_listen": False,
            "started_listen_wait": False,
            "completed_listen_wait": False,
            "final_listen_ready": False,
            "overwhelmed_triggered": False,
            "attack_triggered": False,
            "attack_hit": False,
            "callback_triggered": False,
            "callback_completed": False,
            "callback_forced_recovery": False,
            "move_back_triggered": False,
            "move_back_completed": False,
        }

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
        if self.robot.listen_mode or self.listen_wait_active or self.robot.callback_active:
            return False
        rx, ry, _ = robot_pose
        wx, wy = self.robot.get_current_waypoint()
        dist = float(np.hypot(wx - rx, wy - ry) + DIST_EPS)
        return dist >= ROBOT_WAYPOINT_REACHED_DIST

    def _refresh_callback_rearm_flags(self):
        for idx, human in enumerate(self.humans):
            if idx < len(self.callback_triggered_for_current_distracted) and human.mode != HumanMode.DISTRACTED:
                self.callback_triggered_for_current_distracted[idx] = False

    def _build_callback_request(self, human_xy, robot_pose):
        if not self._is_robot_in_move_stage(robot_pose):
            return None
        hold_steps = max(1, int(round(CALLBACK_HOLD_SECONDS / float(self.timestep))))
        for idx in sorted(self.callback_target_whitelist):
            if idx < 0 or idx >= len(self.humans):
                continue
            human = self.humans[idx]
            if human.mode != HumanMode.DISTRACTED:
                continue
            if human.distracted_timer < CALLBACK_DISTRACTED_TRIGGER_STEPS:
                continue
            if idx < len(self.callback_triggered_for_current_distracted):
                if self.callback_triggered_for_current_distracted[idx]:
                    continue
            if idx >= human_xy.shape[0]:
                continue
            return {
                "target_idx": int(idx),
                "target_xy": np.array(human_xy[idx], dtype=np.float32),
                "hold_steps": int(hold_steps),
            }
        return None

    def _get_nearest_attack_threat(self, robot_xy, human_xy):
        if human_xy.size == 0:
            return None

        nearest_idx = None
        nearest_dist = None
        for idx, human in enumerate(self.humans):
            if human.mode != HumanMode.ATTACK:
                continue
            if idx >= human_xy.shape[0]:
                continue
            dist = float(np.linalg.norm(human_xy[idx] - robot_xy))
            if nearest_idx is None or dist < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist

        if nearest_idx is None:
            return None

        return {
            "idx": int(nearest_idx),
            "dist": float(nearest_dist),
            "xy": np.array(human_xy[nearest_idx], dtype=np.float32),
        }

    @staticmethod
    def _compute_move_back_action(robot_xy, threat_xy):
        diff = np.array(robot_xy - threat_xy, dtype=np.float32)
        norm = float(np.linalg.norm(diff))
        if norm < DIST_EPS:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = diff / norm
        v_xy = MOVE_BACK_SPEED * direction
        return np.array([v_xy[0], v_xy[1], 0.0], dtype=np.float32)

    def _apply_robot_base_color_from_robot_emotion(self):
        if self.robot.emotion == RobotEmotion.SAD:
            self.model.geom_rgba[self.robot_base_geom_id] = ROBOT_COLOR_SAD
            return
        self.model.geom_rgba[self.robot_base_geom_id] = ROBOT_COLOR_NATURAL

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
        self.follow_humans = False

        # Reset listening state
        self.listen_reached_logged = set()
        self.listen_wait_active = False
        self.listen_wait_counter = 0
        self.listen_wait_is_final = False
        self.listen_session_count = 0
        self.overwhelmed_triggered_once = False
        self.attack_triggered_once = False
        self.attack_hit_once = False
        self.callback_triggered_for_current_distracted = [False] * len(self.humans)
        self.callback_active_target_idx = None
        self.move_back_active = False
        self.move_back_attacker_idx = None

        # Reset humans
        for human in self.humans:
            human.reset_episode_state()
        self._configure_human_following_variants()
        self._apply_robot_base_color_from_robot_emotion()

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
        events = self._default_events()

        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)

        robot_xy = np.array([rx, ry], dtype=np.float32)
        events["overwhelmed_triggered"] = self._maybe_trigger_overwhelmed_in_wait(
            robot_xy=robot_xy,
            human_xy=human_xy,
        )
        events["attack_triggered"] = self._maybe_trigger_attack_in_wait(
            robot_xy=robot_xy,
            human_xy=human_xy,
        )

        # Freeze everyone during explanation window.
        self.data.ctrl[:] = 0.0
        rb_action = np.zeros(3, dtype=np.float32)
        human_actions = np.zeros((len(self.humans), 3), dtype=np.float32)

        # Let special states move during explanation window.
        tgt_idx = self.overwhelmed_target_idx
        if 0 <= tgt_idx < len(self.humans):
            tgt_human = self.humans[tgt_idx]
            if tgt_human.mode == HumanMode.OVERWHELMED:
                ctx = {
                    "robot_xy": robot_xy,
                    "robot_yaw": ryaw,
                    "repulsion": np.zeros(2, dtype=np.float32),
                    "stand_threshold": self.listen_stand_threshold,
                }
                tgt_action = tgt_human.step(self.model, self.data, ctx)
                human_actions[tgt_idx] = tgt_action
                ctrl_idx = 3 + tgt_idx * 3
                self.data.ctrl[ctrl_idx:ctrl_idx + 3] = tgt_action

        # Allow attack human to move in attack mode during explanation
        attack_idx = self.attack_target_idx
        if 0 <= attack_idx < len(self.humans):
            attack_human = self.humans[attack_idx]
            if attack_human.mode == HumanMode.ATTACK:
                attack_ctx = {
                    "robot_xy": robot_xy,
                    "robot_yaw": ryaw,
                    "repulsion": np.zeros(2, dtype=np.float32),
                    "stand_threshold": self.listen_stand_threshold,
                }
                attack_action = attack_human.step(self.model, self.data, attack_ctx)
                human_actions[attack_idx] = attack_action
                attack_ctrl_idx = 3 + attack_idx * 3
                self.data.ctrl[attack_ctrl_idx:attack_ctrl_idx + 3] = attack_action
                if attack_human.attack_hit_this_step and not self.attack_hit_once:
                    events["attack_hit"] = True
                    self.attack_hit_once = True

        move_back_was_active = bool(self.move_back_active)
        threat = self._get_nearest_attack_threat(robot_xy=robot_xy, human_xy=human_xy)
        if threat is None:
            self.move_back_active = False
            self.move_back_attacker_idx = None
            self.robot.mode = RobotMode.STOP
            if move_back_was_active:
                events["move_back_completed"] = True
                self._log_event(">>> Robot MOVE_BACK completed (attack ended).")
        elif threat["dist"] < MOVE_BACK_SAFE_DISTANCE:
            self.move_back_active = True
            self.move_back_attacker_idx = int(threat["idx"])
            self.robot.mode = RobotMode.MOVE_BACK
            rb_action = self._compute_move_back_action(robot_xy=robot_xy, threat_xy=threat["xy"])
            if not move_back_was_active:
                events["move_back_triggered"] = True
                self._log_event(
                    f">>> Robot MOVE_BACK triggered by person{self.move_back_attacker_idx + 1} "
                    f"(dist={threat['dist']:.3f}m)."
                )
        else:
            # Attack is ongoing but safety distance is satisfied: hold position.
            if move_back_was_active:
                self.move_back_active = True
                self.move_back_attacker_idx = int(threat["idx"])
                self.robot.mode = RobotMode.MOVE_BACK
            else:
                self.move_back_active = False
                self.move_back_attacker_idx = None
                self.robot.mode = RobotMode.STOP

        self.data.ctrl[0:3] = rb_action

        # Mujoco step update
        mujoco.mj_step(self.model, self.data)
        self.listen_wait_counter += 1

        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        wx, wy = self.robot.get_current_waypoint()
        dist = float(np.hypot(wx - rx, wy - ry) + DIST_EPS)
        desired_yaw = float(np.arctan2(wy - ry, wx - rx))
        actual_yaw = float(ryaw)

        human_goals = np.array([h.current_waypoint for h in self.humans], dtype=np.float32)
        human_reached_goal = self._check_human_goals(human_xy, human_goals)

        # Finish conditions
        final_waypoint_reached = self.robot.is_final_reached(dist)
        all_humans_reached = len(self.humans) > 0 and len(human_reached_goal) == len(self.humans)

        if self.listen_wait_counter >= self.listen_wait_steps:
            events["completed_listen_wait"] = True
            if self.listen_wait_is_final:
                events["final_listen_ready"] = True
                self._log_event(">>> Listening wait complete at final display.")
            else:
                self.robot.on_listening_complete()
                self.follow_humans = False
                self.robot_start_xy = np.array([rx, ry], dtype=np.float32)
                self._log_event(">>> Listening wait complete. Resume MOVE to Room B.")

            if self.move_back_active:
                events["move_back_completed"] = True
            self.move_back_active = False
            self.move_back_attacker_idx = None

            self.listen_wait_active = False
            self.listen_wait_counter = 0
            self.listen_wait_is_final = False

        self.robot.update_emotion([h.mode for h in self.humans])
        self._apply_robot_base_color_from_robot_emotion()

        human_v_follow = np.zeros((len(self.humans), 2), dtype=np.float32)
        human_v_repulsion = np.zeros((len(self.humans), 2), dtype=np.float32)
        human_v_hr = np.zeros((len(self.humans), 2), dtype=np.float32)

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
            human_v_follow=human_v_follow,
            human_v_repulsion=human_v_repulsion,
            human_v_hr=human_v_hr,
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
        events = self._default_events()

        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        # --- Robot decision ---
        robot_pose = self._get_robot_pose()
        self._refresh_callback_rearm_flags()
        callback_request = self._build_callback_request(human_xy=human_xy, robot_pose=robot_pose)
        callback_active_before_step = bool(self.robot.callback_active)
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
        events["entered_listen"] = bool(enter_listen)
        if callback_request is not None and (not callback_active_before_step) and robot_mode == RobotMode.CALLBACK:
            events["callback_triggered"] = True
            target_idx = int(callback_request["target_idx"])
            self.callback_active_target_idx = target_idx
            if 0 <= target_idx < len(self.callback_triggered_for_current_distracted):
                self.callback_triggered_for_current_distracted[target_idx] = True
                self._log_event(f">>> Robot CALLBACK triggered for person{target_idx + 1}.")
        if callback_active_before_step and (not self.robot.callback_active):
            events["callback_completed"] = True
            recover_idx = self.callback_active_target_idx
            if recover_idx is not None and 0 <= recover_idx < len(self.humans):
                recover_human = self.humans[recover_idx]
                if recover_human.mode == HumanMode.DISTRACTED:
                    recover_human.set_mode(HumanMode.FOLLOWING)
                    recover_human.distracted_timer = 0
                    events["callback_forced_recovery"] = True
                    self._log_event(f">>> person{recover_idx + 1} forced recovery by CALLBACK -> FOLLOWING.")
            self.callback_active_target_idx = None
            self._log_event(">>> Robot CALLBACK completed.")

        # If robot just entered listen, assign listen targets
        if enter_listen:
            rx, ry, ryaw = robot_pose
            self.listen_reached_logged = set()
            self.listen_wait_active = False
            self.listen_wait_counter = 0
            self.listen_wait_is_final = False
            self.listen_session_count += 1
            n_humans = len(self.humans)

            self._log_event(f">>> Robot entering LISTEN mode. robot=({rx:.2f}, {ry:.2f}, yaw={ryaw:.2f})")
            for i, human in enumerate(self.humans):
                human.assign_listen_target(
                    index=i,
                    n_humans=n_humans,
                    robot_pose=(rx, ry, ryaw),
                    listen_radius=self.listen_fan_radius,
                    fan_half_angle=self.listen_fan_half_angle,
                )
                gx, gy = human.current_waypoint
                self._log_event(f"    person{i+1} listen_goal=({gx:.3f}, {gy:.3f})")

        # Apply robot action
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:3] = rb_action

        rx, ry, ryaw = self._get_robot_pose()

        # Switch to follow once the robot has started moving toward the display
        if not self.robot.listen_mode and not self.follow_humans:
            moved_dist = float(np.hypot(rx - self.robot_start_xy[0], ry - self.robot_start_xy[1]))
            if moved_dist >= self.human_follow_distance:
                self.follow_humans = True

        repulsion_vectors = self._compute_social_repulsion(human_xy)
        human_actions = self._update_humans_and_apply_ctrl(
            rx=rx,
            ry=ry,
            ryaw=ryaw,
            repulsion_vectors=repulsion_vectors,
        )

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Refresh poses after the step for reporting
        rx, ry, ryaw = self._get_robot_pose()
        human_xyz = self._get_human_poses()
        human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
        human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

        human_goals = np.array([h.current_waypoint for h in self.humans], dtype=np.float32)
        human_reached_goal = self._check_human_goals(human_xy, human_goals)

        human_v_follow = np.array([h.last_v_follow for h in self.humans], dtype=np.float32)
        human_v_repulsion = np.array([h.last_v_repulsion for h in self.humans], dtype=np.float32)
        human_v_hr = np.array([h.last_v_hr for h in self.humans], dtype=np.float32)

        final_waypoint_reached = self.robot.is_final_reached(dist)
        all_humans_reached = len(self.humans) > 0 and len(human_reached_goal) == len(self.humans)
        events.update(self._handle_listen_transitions(final_waypoint_reached, all_humans_reached))
        self.robot.update_emotion([h.mode for h in self.humans])
        self._apply_robot_base_color_from_robot_emotion()

        snapshot = self._collect_step_snapshot(
            robot_pose=(rx, ry, ryaw),
            dist=float(dist),
            desired_yaw=float(desired_yaw),
            actual_yaw=float(actual_yaw),
            robot_mode=str(robot_mode),
            robot_action=np.array(rb_action, dtype=np.float32),
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

        return self._finalize_step_output(
            snapshot=snapshot,
            events=events,
            dist=float(dist),
            terminated=False,
            external_action_received=external_action_received,
            external_action_used=False,
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

        for i, human in enumerate(self.humans):
            repulsion_vec = repulsion_vectors[i] if i < len(repulsion_vectors) else np.zeros(2, dtype=np.float32)

            ctx = {
                "robot_xy": np.array([rx, ry], dtype=np.float32),
                "robot_yaw": ryaw,
                "repulsion": repulsion_vec,
                "stand_threshold": self.listen_stand_threshold,
            }

            if self.robot.listen_mode:
                if human.mode != HumanMode.OVERWHELMED:
                    human.set_mode(HumanMode.LISTENING)
            else:
                if human.mode not in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED, HumanMode.IMPATIENT):
                    human.set_mode(HumanMode.FOLLOWING if self.follow_humans else HumanMode.WANDERING)

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
        rx, ry, _ = robot_pose
        gx, gy = self._get_goal_xy()

        human_desired_yaw = np.arctan2(
            human_goals[:, 1] - human_xy[:, 1],
            human_goals[:, 0] - human_xy[:, 0],
        ).astype(np.float32)

        return {
            "robot_xy": np.array([rx, ry], dtype=np.float32),
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
            "human_v_follow": human_v_follow,
            "human_v_repulsion": human_v_repulsion,
            "human_v_hr": human_v_hr,
            "human_v_total": human_v_follow + human_v_repulsion + human_v_hr,
            "human_mode": [h.mode for h in self.humans],
            "human_distracted_timer": np.array([h.distracted_timer for h in self.humans], dtype=np.int32),
            "human_overwhelmed_stage": [h.overwhelmed_stage for h in self.humans],
            "human_overwhelmed_leave_timer": np.array(
                [h.overwhelmed_leave_timer for h in self.humans], dtype=np.int32
            ),
            "human_impatient_timer": np.array([h.impatient_timer for h in self.humans], dtype=np.int32),
            "human_reached_goal": human_reached_goal,
            "final_waypoint_reached": bool(final_waypoint_reached),
            "all_humans_reached": bool(all_humans_reached),
        }

    def _build_info(
        self,
        snapshot,
        events,
        truncated,
        external_action_received=False,
        external_action_used=False,
    ):
        listen_wait_remaining = (
            max(0, self.listen_wait_steps - self.listen_wait_counter) if self.listen_wait_active else 0
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
                "attack_triggered": bool(events["attack_triggered"]),
                "attack_hit": bool(events["attack_hit"]),
                "callback_triggered": bool(events["callback_triggered"]),
                "callback_completed": bool(events["callback_completed"]),
                "callback_forced_recovery": bool(events["callback_forced_recovery"]),
                "move_back_triggered": bool(events["move_back_triggered"]),
                "move_back_completed": bool(events["move_back_completed"]),
            },
            "status": {
                "step_count": int(self.step_count),
                "listen_mode": bool(self.robot.listen_mode),
                "listen_wait": {
                    "active": bool(self.listen_wait_active),
                    "counter": int(self.listen_wait_counter),
                    "steps": int(self.listen_wait_steps),
                    "remaining": int(listen_wait_remaining),
                    "is_final": bool(self.listen_wait_is_final),
                },
                "callback_active": bool(self.robot.callback_active),
                "callback_target_idx": (
                    int(self.robot.callback_target_idx)
                    if self.robot.callback_target_idx is not None
                    else None
                ),
                "callback_hold_remaining": int(self.robot.callback_hold_steps_remaining),
                "move_back_active": bool(self.move_back_active),
                "move_back_attacker_idx": (
                    int(self.move_back_attacker_idx) if self.move_back_attacker_idx is not None else None
                ),
                "move_back_safe_distance": float(MOVE_BACK_SAFE_DISTANCE),
                "move_back_speed": float(MOVE_BACK_SPEED),
                "robot_emotion": str(self.robot.emotion),
                "external_action_received": bool(external_action_received),
                "external_action_used": bool(external_action_used),
                "terminated_reason": terminated_reason,
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
                "distracted_timer": snapshot["human_distracted_timer"],
                "overwhelmed_stage": snapshot["human_overwhelmed_stage"],
                "overwhelmed_leave_timer": snapshot["human_overwhelmed_leave_timer"],
                "impatient_timer": snapshot["human_impatient_timer"],
                "reached_goal_indices": snapshot["human_reached_goal"],
                "all_reached": bool(snapshot["all_humans_reached"]),
                "action": {
                    "vx": human_actions[:, 0],
                    "vy": human_actions[:, 1],
                    "yaw_rate": human_actions[:, 2],
                },
                "velocity_components": {
                    "follow": snapshot["human_v_follow"],
                    "repulsion": snapshot["human_v_repulsion"],
                    "human_robot": snapshot["human_v_hr"],
                    "total": snapshot["human_v_total"],
                },
            },
        }

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
