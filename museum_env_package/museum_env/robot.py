import numpy as np

from .spatial_utils import wrap_to_pi

# Robot-side finite-state controller used by MuseumEnv.
ROBOT_WAYPOINT_REACHED_DIST = 0.2
ROBOT_YAW_RATE_LIMIT = 50.0
ROBOT_TURN_DONE_YAW_ERR = 0.05
ROBOT_TURN_GAIN = 50.0
ROBOT_CALLBACK_TURN_DONE_YAW_ERR = ROBOT_TURN_DONE_YAW_ERR
ROBOT_CALLBACK_TURN_GAIN = ROBOT_TURN_GAIN


class RobotMode:
    MOVE = "move"
    STOP = "stop"
    CALLBACK = "callback"


class RobotEmotion:
    NATURAL = "natural"
    SAD = "sad"
    HAPPY = "happy"


class RobotCallbackPhase:
    TURN = "turn"
    CUE = "cue"


class RobotSpeechMode:
    NONE = "none"
    EXPLANATION = "explanation"
    ANSWER = "answer"
    PASS_REQUEST = "pass_request"


class Robot:
    """
    Robot policy/state wrapper:
    - waypoint following
    - stop at display, turn to face crowd
    - enter listening mode after turning done
    - resume move after listening complete (triggered by env)
    """

    def __init__(self, waypoints, v_max=3.0, k_v=20.0, k_yaw=20.0):
        """Initialize waypoint controller and robot behavior state."""
        self.waypoints = list(waypoints)
        self.v_max = float(v_max)
        self.k_v = float(k_v)
        self.k_yaw = float(k_yaw)

        # state
        self.current_waypoint_idx = 0
        self.listen_mode = False
        self.listen_done = False

        self.turn_target_yaw = None
        self.turn_done = False
        self.mode = RobotMode.MOVE
        self.callback_active = False
        self.callback_target_idx = None
        self.callback_target_xy = None
        self.callback_attempt_index = 0
        self.callback_phase = None
        self.callback_cue_total_steps = 0
        self.callback_cue_elapsed_steps = 0
        self.callback_response_sampled = False
        self.callback_cue_completed_this_step = False
        self.callback_turn_done = False
        self.emotion = RobotEmotion.NATURAL
        self.happy_hold_steps_remaining = 0
        self.speaker_active = False
        self.speech_mode = RobotSpeechMode.NONE

    @staticmethod
    def _wrap_to_pi(ang: float) -> float:
        """Normalize angle to [-pi, pi)."""
        return wrap_to_pi(ang)

    def reset(self):
        """Reset full robot runtime state at episode start."""
        self.current_waypoint_idx = 0
        self.listen_mode = False
        self.listen_done = False
        self.turn_target_yaw = None
        self.turn_done = False
        self.mode = RobotMode.MOVE
        self._reset_callback_state()
        self.emotion = RobotEmotion.NATURAL
        self.happy_hold_steps_remaining = 0
        self.speaker_active = False
        self.speech_mode = RobotSpeechMode.NONE

    def _reset_callback_state(self):
        """Clear callback-related transient state."""
        self.callback_active = False
        self.callback_target_idx = None
        self.callback_target_xy = None
        self.callback_attempt_index = 0
        self.callback_phase = None
        self.callback_cue_total_steps = 0
        self.callback_cue_elapsed_steps = 0
        self.callback_response_sampled = False
        self.callback_cue_completed_this_step = False
        self.callback_turn_done = False

    def get_current_waypoint(self):
        """Return current target waypoint (x, y)."""
        return self.waypoints[self.current_waypoint_idx]

    def _compute_waypoint_metrics(self, robot_pose):
        """Compute distance and heading error reference to current waypoint."""
        x, y, yaw = robot_pose
        wx, wy = self.waypoints[self.current_waypoint_idx]
        dx = wx - x
        dy = wy - y
        dist = float(np.hypot(dx, dy) + 1e-8)
        desired_yaw = float(np.arctan2(dy, dx))
        return dist, desired_yaw, float(yaw)

    def _waypoint_action(self, robot_pose):
        """
        Pure waypoint-following controller (no stop/turn/listen gating).
        Returns:
            action (3,), dist, desired_yaw, actual_yaw
        """
        x, y, yaw = robot_pose
        wx, wy = self.waypoints[self.current_waypoint_idx]

        dx = wx - x
        dy = wy - y
        dist = float(np.hypot(dx, dy) + 1e-8)

        # Switch to next waypoint if close enough.
        # Waypoint 0 is gated until the first listening session is completed.
        if dist < ROBOT_WAYPOINT_REACHED_DIST and self.current_waypoint_idx < len(self.waypoints) - 1:
            
            # don't leave waypoint 0 before the first listening is done.
            if not (self.current_waypoint_idx == 0 and not self.listen_done):
                self.current_waypoint_idx += 1
                wx, wy = self.waypoints[self.current_waypoint_idx]
                dx = wx - x
                dy = wy - y
                dist = float(np.hypot(dx, dy) + 1e-8)

        desired_yaw = float(np.arctan2(dy, dx))
        yaw_err = self._wrap_to_pi(desired_yaw - yaw)

        # (keep your structure)
        # speed scalar
        v = float(np.clip(self.k_v * dist, 0.0, self.v_max))

        # move in heading direction (as your original)
        vx = v * float(np.cos(yaw))
        vy = v * float(np.sin(yaw))
        yaw_rate = float(np.clip(self.k_yaw * yaw_err, -ROBOT_YAW_RATE_LIMIT, ROBOT_YAW_RATE_LIMIT))

        action = np.array([vx, vy, yaw_rate], dtype=np.float32)
        return action, dist, desired_yaw, float(yaw)

    def _turn_to_crowd_action(self, robot_pose, human_xyz):
        """
        When STOP at display: rotate in place by a fixed 180 degrees.
        """
        del human_xyz
        _, _, ryaw = robot_pose

        if self.turn_target_yaw is None:
            self.turn_target_yaw = self._wrap_to_pi(ryaw + np.pi)

        yaw_err = self._wrap_to_pi(self.turn_target_yaw - ryaw)

        action = np.zeros(3, dtype=np.float32)
        if abs(yaw_err) < ROBOT_TURN_DONE_YAW_ERR:
            action[2] = 0.0
            self.turn_done = True
        else:
            action[2] = float(
                np.clip(ROBOT_TURN_GAIN * yaw_err, -ROBOT_YAW_RATE_LIMIT, ROBOT_YAW_RATE_LIMIT)
            )

        return action

    def _start_callback_attempt(self, target_idx, target_xy, cue_steps, attempt_index):
        """Activate one callback attempt toward a distracted target."""
        self.callback_active = True
        self.callback_target_idx = int(target_idx)
        self.callback_target_xy = np.array(target_xy, dtype=np.float32)
        self.callback_attempt_index = max(1, int(attempt_index))
        self.callback_phase = RobotCallbackPhase.TURN
        self.callback_cue_total_steps = max(1, int(cue_steps))
        self.callback_cue_elapsed_steps = 0
        self.callback_response_sampled = False
        self.callback_cue_completed_this_step = False
        self.callback_turn_done = False
        self.mode = RobotMode.CALLBACK

    def start_callback(self, target_idx, target_xy, cue_steps):
        """Start the first callback attempt."""
        self._start_callback_attempt(
            target_idx=target_idx,
            target_xy=target_xy,
            cue_steps=cue_steps,
            attempt_index=1,
        )

    def start_next_callback_attempt(self):
        """Restart callback on the same target using the next attempt index."""
        if self.callback_target_xy is None or self.callback_target_idx is None:
            return False
        next_attempt_index = max(1, int(self.callback_attempt_index) + 1)
        self._start_callback_attempt(
            target_idx=int(self.callback_target_idx),
            target_xy=np.array(self.callback_target_xy, dtype=np.float32),
            cue_steps=int(self.callback_cue_total_steps),
            attempt_index=next_attempt_index,
        )
        return True

    def _finish_callback(self):
        """Exit callback mode and return to MOVE."""
        self._reset_callback_state()
        self.mode = RobotMode.MOVE

    def finish_callback(self):
        """Public env-facing callback cleanup entrypoint."""
        self._finish_callback()

    def _callback_action(self, robot_pose):
        """Generate control while callback mode is active."""
        rx, ry, ryaw = robot_pose
        action = np.zeros(3, dtype=np.float32)
        self.callback_cue_completed_this_step = False
        if self.callback_target_xy is None:
            self._finish_callback()
            return action

        desired_yaw = float(np.arctan2(self.callback_target_xy[1] - ry, self.callback_target_xy[0] - rx))
        yaw_err = self._wrap_to_pi(desired_yaw - ryaw)
        if self.callback_phase == RobotCallbackPhase.TURN:
            if abs(yaw_err) >= ROBOT_CALLBACK_TURN_DONE_YAW_ERR:
                action[2] = float(
                    np.clip(
                        ROBOT_CALLBACK_TURN_GAIN * yaw_err,
                        -ROBOT_YAW_RATE_LIMIT,
                        ROBOT_YAW_RATE_LIMIT,
                    )
                )
                return action
            self.callback_turn_done = True
            self.callback_phase = RobotCallbackPhase.CUE
            self.callback_cue_elapsed_steps = 0
            self.callback_response_sampled = False
            return action

        if self.callback_phase != RobotCallbackPhase.CUE:
            self._finish_callback()
            return action

        self.callback_cue_elapsed_steps = min(
            int(self.callback_cue_total_steps),
            int(self.callback_cue_elapsed_steps) + 1,
        )
        if self.callback_cue_elapsed_steps >= int(self.callback_cue_total_steps):
            self.callback_cue_completed_this_step = True

        return action

    def update_emotion(self):
        """Update robot emotion from robot-local runtime state."""
        if self.callback_active:
            self.emotion = RobotEmotion.SAD
            return self.emotion
        self.emotion = RobotEmotion.NATURAL
        return self.emotion

    def trigger_happy(self, hold_steps: int):
        """Start/refresh HAPPY emotion timer."""
        self.happy_hold_steps_remaining = max(1, int(hold_steps))

    def set_speech_mode(self, mode: str) -> None:
        """Set whether robot is speaking and which speech label should be shown."""
        normalized_mode = str(mode)
        self.speech_mode = normalized_mode
        self.speaker_active = normalized_mode != RobotSpeechMode.NONE

    def step(self, robot_pose, human_xyz):
        """
        Main robot decision step.
        Mutates internal robot state and returns the action for this step.
        """
        if self.callback_active:
            self.mode = RobotMode.CALLBACK
            return self._callback_action(robot_pose)

        # base waypoint action
        base_action, dist, _, _ = self._waypoint_action(robot_pose)
        self.mode = RobotMode.MOVE

        # Stop after reaching the display and turn to face people
        if dist < ROBOT_WAYPOINT_REACHED_DIST:
            self.mode = RobotMode.STOP
            turn_action = self._turn_to_crowd_action(robot_pose, human_xyz)
            base_action[:] = turn_action

        # Enter listening mode after turning is done at the display
        if dist < ROBOT_WAYPOINT_REACHED_DIST and self.turn_done and not self.listen_mode:
            self.listen_mode = True
        return base_action

    def on_listening_complete(self):
        """
        Called by env when all humans reached listen targets.
        Mirrors your original transition.
        """
        self.listen_mode = False
        self.turn_done = False
        self.turn_target_yaw = None

        # Resume move: go to waypoint index 1
        self.current_waypoint_idx = 1
        self.listen_done = True
        self.mode = RobotMode.MOVE
        self._reset_callback_state()

    def is_final_reached(self, robot_pose):
        """Return True when robot reached the last waypoint within threshold."""
        dist, _, _ = self._compute_waypoint_metrics(robot_pose)
        return (
            dist < ROBOT_WAYPOINT_REACHED_DIST
            and self.current_waypoint_idx == len(self.waypoints) - 1
        )
