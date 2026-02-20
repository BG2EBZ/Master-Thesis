import numpy as np

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


class Robot:
    """
    Robot policy/state wrapper:
    - waypoint following
    - stop at display, turn to face crowd
    - enter listening mode after turning done
    - resume move after listening complete (triggered by env)
    """

    def __init__(self, waypoints, v_max=3.0, k_v=20.0, k_yaw=20.0):
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
        self.callback_hold_steps_remaining = 0
        self.callback_turn_done = False

    @staticmethod
    def _wrap_to_pi(ang: float) -> float:
        return (ang + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self.current_waypoint_idx = 0
        self.listen_mode = False
        self.listen_done = False
        self.turn_target_yaw = None
        self.turn_done = False
        self.mode = RobotMode.MOVE
        self._reset_callback_state()

    def _reset_callback_state(self):
        self.callback_active = False
        self.callback_target_idx = None
        self.callback_target_xy = None
        self.callback_hold_steps_remaining = 0
        self.callback_turn_done = False

    def get_current_waypoint(self):
        return self.waypoints[self.current_waypoint_idx]

    def _compute_waypoint_metrics(self, robot_pose):
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

        # Switch to next waypoint if close enough
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
        When STOP at display: turn to face crowd centroid (or fallback).
        """
        rx, ry, ryaw = robot_pose

        if self.turn_target_yaw is None:
            if human_xyz is not None and human_xyz.size:
                mean_hx = float(np.mean(human_xyz[:, 0]))
                mean_hy = float(np.mean(human_xyz[:, 1]))
                dx = mean_hx - rx
                dy = mean_hy - ry
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    self.turn_target_yaw = self._wrap_to_pi(ryaw + np.pi)
                else:
                    self.turn_target_yaw = float(np.arctan2(dy, dx))
            else:
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

    def _start_callback(self, target_idx, target_xy, hold_steps):
        self.callback_active = True
        self.callback_target_idx = int(target_idx)
        self.callback_target_xy = np.array(target_xy, dtype=np.float32)
        self.callback_hold_steps_remaining = max(1, int(hold_steps))
        self.callback_turn_done = False
        self.mode = RobotMode.CALLBACK

    def _finish_callback(self):
        self._reset_callback_state()
        self.mode = RobotMode.MOVE

    def _callback_action(self, robot_pose):
        rx, ry, ryaw = robot_pose
        action = np.zeros(3, dtype=np.float32)
        if self.callback_target_xy is None:
            self._finish_callback()
            return action

        desired_yaw = float(np.arctan2(self.callback_target_xy[1] - ry, self.callback_target_xy[0] - rx))
        yaw_err = self._wrap_to_pi(desired_yaw - ryaw)
        if not self.callback_turn_done:
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

        if self.callback_hold_steps_remaining > 0:
            self.callback_hold_steps_remaining -= 1

        if self.callback_hold_steps_remaining <= 0:
            self._finish_callback()

        return action

    def step(self, robot_pose, human_xyz, callback_request=None):
        """
        Main robot decision step.
        Returns a dict so env can stay clean.

        Outputs:
            {
              "action": np.array(3,),
              "dist": float,
              "desired_yaw": float,
              "actual_yaw": float,
              "mode": str,
              "enter_listen": bool
            }
        """
        if (
            (not self.callback_active)
            and callback_request is not None
            and (not self.listen_mode)
        ):
            self._start_callback(
                target_idx=callback_request["target_idx"],
                target_xy=callback_request["target_xy"],
                hold_steps=callback_request["hold_steps"],
            )

        if self.callback_active:
            dist, desired_yaw, actual_yaw = self._compute_waypoint_metrics(robot_pose)
            callback_action = self._callback_action(robot_pose)
            return {
                "action": callback_action,
                "dist": float(dist),
                "desired_yaw": float(desired_yaw),
                "actual_yaw": float(actual_yaw),
                "mode": RobotMode.CALLBACK,
                "enter_listen": False,
            }

        # base waypoint action
        base_action, dist, desired_yaw, actual_yaw = self._waypoint_action(robot_pose)

        enter_listen = False
        self.mode = RobotMode.MOVE

        # Stop after reaching the display and turn to face people
        if dist < ROBOT_WAYPOINT_REACHED_DIST:
            self.mode = RobotMode.STOP
            turn_action = self._turn_to_crowd_action(robot_pose, human_xyz)
            base_action[:] = turn_action

        # Enter listening mode after turning is done at the display
        if dist < ROBOT_WAYPOINT_REACHED_DIST and self.turn_done and not self.listen_mode:
            self.listen_mode = True
            enter_listen = True

        return {
            "action": base_action,
            "dist": float(dist),
            "desired_yaw": float(desired_yaw),
            "actual_yaw": float(actual_yaw),
            "mode": str(self.mode),
            "enter_listen": bool(enter_listen),
        }

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

    def is_final_reached(self, dist: float):
        return (
            dist < ROBOT_WAYPOINT_REACHED_DIST
            and self.current_waypoint_idx == len(self.waypoints) - 1
        )
