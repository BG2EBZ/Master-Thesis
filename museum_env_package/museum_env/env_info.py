import numpy as np


EVENT_KEYS = (
    "entered_listen",
    "started_listen_wait",
    "completed_listen_wait",
    "final_listen_ready",
    "overwhelmed_triggered",
    "attack_triggered",
    "attack_hit",
    "callback_triggered",
    "callback_completed",
    "callback_forced_recovery",
    "callback_response_rejoin",
    "callback_response_stay",
    "callback_response_ignore",
    "happy_triggered",
    "happy_completed",
    "fear_triggered",
    "fear_completed",
    "fear_response_move_back",
    "fear_response_stay",
    "fear_response_continue_hit",
    "move_back_triggered",
    "move_back_completed",
)


def default_events():
    return {k: False for k in EVENT_KEYS}


def collect_step_snapshot(
    env,
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
    gx, gy = env._get_goal_xy()

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
        "human_mode": [h.mode for h in env.humans],
        "human_distracted_timer": np.array([h.distracted_timer for h in env.humans], dtype=np.int32),
        "human_overwhelmed_stage": [h.overwhelmed_stage for h in env.humans],
        "human_overwhelmed_leave_timer": np.array(
            [h.overwhelmed_leave_timer for h in env.humans], dtype=np.int32
        ),
        "human_impatient_timer": np.array([h.impatient_timer for h in env.humans], dtype=np.int32),
        "human_reached_goal": human_reached_goal,
        "final_waypoint_reached": bool(final_waypoint_reached),
        "all_humans_reached": bool(all_humans_reached),
    }


def build_info(
    env,
    snapshot,
    events,
    truncated,
    move_back_safe_distance: float,
    move_back_speed: float,
    happy_hold_seconds: float,
    fear_distance_threshold: float,
    robot_text_label: str,
    external_action_received=False,
    external_action_used=False,
):
    listen_wait_remaining = (
        max(0, env.listen_wait_steps - env.listen_wait_counter) if env.listen_wait_active else 0
    )

    terminated_reason = None
    if events["final_listen_ready"]:
        terminated_reason = "final_listen_ready"
    elif truncated:
        terminated_reason = "max_steps"

    robot_action = snapshot["robot_action"]
    human_actions = snapshot["human_actions"]

    return {
        "events": {k: bool(events[k]) for k in EVENT_KEYS},
        "status": {
            "step_count": int(env.step_count),
            "listen_mode": bool(env.robot.listen_mode),
            "listen_wait": {
                "active": bool(env.listen_wait_active),
                "counter": int(env.listen_wait_counter),
                "steps": int(env.listen_wait_steps),
                "remaining": int(listen_wait_remaining),
                "is_final": bool(env.listen_wait_is_final),
            },
            "callback_active": bool(env.robot.callback_active),
            "callback_target_idx": (
                int(env.robot.callback_target_idx)
                if env.robot.callback_target_idx is not None
                else None
            ),
            "callback_hold_remaining": int(env.robot.callback_hold_steps_remaining),
            "callback_last_response": (
                str(env.callback_last_response)
                if env.callback_last_response is not None
                else None
            ),
            "callback_last_response_target_idx": (
                int(env.callback_last_response_target_idx)
                if env.callback_last_response_target_idx is not None
                else None
            ),
            "move_back_active": bool(env.move_back_active),
            "move_back_attacker_idx": (
                int(env.move_back_attacker_idx) if env.move_back_attacker_idx is not None else None
            ),
            "move_back_safe_distance": float(move_back_safe_distance),
            "move_back_speed": float(move_back_speed),
            "robot_emotion": str(env.robot.emotion),
            "happy_remaining_steps": int(env.robot.happy_hold_steps_remaining),
            "happy_hold_seconds": float(happy_hold_seconds),
            "fear_active": bool(env.fear_active),
            "fear_attacker_idx": int(env.fear_attacker_idx) if env.fear_attacker_idx is not None else None,
            "fear_last_response": (
                str(env.fear_last_response)
                if env.fear_last_response is not None
                else None
            ),
            "fear_last_response_target_idx": (
                int(env.fear_last_response_target_idx)
                if env.fear_last_response_target_idx is not None
                else None
            ),
            "fear_distance_threshold": float(fear_distance_threshold),
            "speaker_active": bool(env.robot.speaker_active),
            "robot_text_label": str(robot_text_label),
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
            "emotion": str(env.robot.emotion),
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
