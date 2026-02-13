import numpy as np


class RobotMode:
    MOVE = "move"
    STOP = "stop"


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

    def get_current_waypoint(self):
        return self.waypoints[self.current_waypoint_idx]

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
        if dist < 0.2 and self.current_waypoint_idx < len(self.waypoints) - 1:
            
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
        yaw_rate = float(np.clip(self.k_yaw * yaw_err, -50.0, 50.0))

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
        if abs(yaw_err) < 0.05:
            action[2] = 0.0
            self.turn_done = True
        else:
            k_turn = 50.0
            action[2] = float(np.clip(k_turn * yaw_err, -50.0, 50.0))

        return action

    def step(self, robot_pose, human_xyz):
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
        # base waypoint action
        base_action, dist, desired_yaw, actual_yaw = self._waypoint_action(robot_pose)

        enter_listen = False
        self.mode = RobotMode.MOVE

        # Stop after reaching the display and turn to face people
        if dist < 0.2:
            self.mode = RobotMode.STOP
            turn_action = self._turn_to_crowd_action(robot_pose, human_xyz)
            base_action[:] = turn_action

        # Enter listening mode after turning is done at the display
        if dist < 0.2 and self.turn_done and not self.listen_mode:
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

    def is_final_reached(self, dist: float):
        return (dist < 0.2 and self.current_waypoint_idx == len(self.waypoints) - 1)