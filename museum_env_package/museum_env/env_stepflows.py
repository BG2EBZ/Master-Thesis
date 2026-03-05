import mujoco
import numpy as np

from .human import HumanMode
from .robot import RobotMode
from .robot_fsm import ROBOT_STATE_WAIT as ROBOT_FSM_WAIT_STATE
from .robot_fsm import decide_mode as decide_robot_fsm_mode


def _post_step_visual_sync(env, events, rx, ry, human_xy):
    env._update_robot_emotion_and_visual(
        events=events,
        robot_xy=np.array([rx, ry], dtype=np.float32),
        human_xy=human_xy,
    )
    env._apply_fear_response_on_trigger(events)
    env._resolve_fear_response_on_complete(events)
    env._sync_robot_speaker_state()
    env._sync_robot_text_label_visibility()
    env._apply_robot_speaking_halo_visual()


def step_waiting_branch(env, external_action_received=False):
    events = env._default_events()

    rx, ry, ryaw = env._get_robot_pose()
    human_xyz = env._get_human_poses()
    human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)

    robot_xy = np.array([rx, ry], dtype=np.float32)
    events["overwhelmed_triggered"] = env._maybe_trigger_overwhelmed_in_wait(
        robot_xy=robot_xy,
        human_xy=human_xy,
    )
    events["attack_triggered"] = env._maybe_trigger_attack_in_wait(
        robot_xy=robot_xy,
        human_xy=human_xy,
    )

    env.data.ctrl[:] = 0.0
    rb_action = np.zeros(3, dtype=np.float32)
    human_actions = np.zeros((len(env.humans), 3), dtype=np.float32)

    tgt_idx = env.overwhelmed_target_idx
    if 0 <= tgt_idx < len(env.humans):
        tgt_human = env.humans[tgt_idx]
        if tgt_human.mode == HumanMode.OVERWHELMED:
            ctx = {
                "robot_xy": robot_xy,
                "robot_yaw": ryaw,
                "repulsion": np.zeros(2, dtype=np.float32),
                "stand_threshold": env.listen_stand_threshold,
            }
            tgt_action = tgt_human.step(env.model, env.data, ctx)
            human_actions[tgt_idx] = tgt_action
            ctrl_idx = 3 + tgt_idx * 3
            env.data.ctrl[ctrl_idx:ctrl_idx + 3] = tgt_action

    attack_idx = env.attack_target_idx
    if 0 <= attack_idx < len(env.humans):
        attack_human = env.humans[attack_idx]
        attack_ctx = {
            "robot_xy": robot_xy,
            "robot_yaw": ryaw,
            "repulsion": np.zeros(2, dtype=np.float32),
            "stand_threshold": env.listen_stand_threshold,
        }
        if attack_human.mode == HumanMode.ATTACK:
            should_stay_freeze = bool(
                env.fear_active
                and env.fear_current_freeze_attack
                and env.fear_current_response_target_idx == attack_idx
            )
            if should_stay_freeze:
                attack_action = np.zeros(3, dtype=np.float32)
                attack_human.attack_hit_this_step = False
                attack_human.last_v_follow = np.zeros(2, dtype=np.float32)
                attack_human.last_v_repulsion = np.zeros(2, dtype=np.float32)
                attack_human.last_v_hr = np.zeros(2, dtype=np.float32)
            else:
                attack_action = attack_human.step(env.model, env.data, attack_ctx)
                if attack_human.attack_hit_this_step and not env.attack_hit_once:
                    events["attack_hit"] = True
                    env.attack_hit_once = True
            human_actions[attack_idx] = attack_action
            attack_ctrl_idx = 3 + attack_idx * 3
            env.data.ctrl[attack_ctrl_idx:attack_ctrl_idx + 3] = attack_action
        elif attack_human.mode == HumanMode.LISTENING:
            is_move_back_response = bool(
                (env.fear_current_response_mode == "move_back" and env.fear_current_response_target_idx == attack_idx)
                or (env.fear_last_response == "move_back" and env.fear_last_response_target_idx == attack_idx)
            )
            if is_move_back_response:
                attack_action = attack_human.step(env.model, env.data, attack_ctx)
                human_actions[attack_idx] = attack_action
                attack_ctrl_idx = 3 + attack_idx * 3
                env.data.ctrl[attack_ctrl_idx:attack_ctrl_idx + 3] = attack_action

    move_back_was_active = bool(env.move_back_active)
    threat = env._get_nearest_attack_threat(robot_xy=robot_xy, human_xy=human_xy)
    threat_exists = bool(threat is not None)
    threat_dist = float(threat["dist"]) if threat_exists else None

    wait_current_mode = RobotMode.MOVE_BACK if move_back_was_active else ROBOT_FSM_WAIT_STATE
    wait_robot_fsm_ctx = env._build_wait_robot_fsm_ctx(
        threat_exists=threat_exists,
        threat_dist=threat_dist,
    )
    wait_robot_fsm = decide_robot_fsm_mode(
        current_mode=wait_current_mode,
        ctx=wait_robot_fsm_ctx,
        table=env.robot_fsm_table,
    )
    wait_effects = wait_robot_fsm["effects"]
    rb_action = env._apply_wait_robot_fsm_effects(
        wait_effects=wait_effects,
        threat=threat,
        move_back_was_active=move_back_was_active,
        robot_xy=robot_xy,
        events=events,
    )

    env.data.ctrl[0:3] = rb_action

    mujoco.mj_step(env.model, env.data)
    env.listen_wait_counter += 1

    rx, ry, ryaw = env._get_robot_pose()
    human_xyz = env._get_human_poses()
    human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
    human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

    wx, wy = env.robot.get_current_waypoint()
    dist = float(np.hypot(wx - rx, wy - ry) + 1e-8)
    desired_yaw = float(np.arctan2(wy - ry, wx - rx))
    actual_yaw = float(ryaw)

    human_goals = np.array([h.current_waypoint for h in env.humans], dtype=np.float32)
    human_reached_goal = env._check_human_goals(human_xy, human_goals)

    final_waypoint_reached = env.robot.is_final_reached(dist)
    all_humans_reached = len(env.humans) > 0 and len(human_reached_goal) == len(env.humans)

    if env.listen_wait_counter >= env.listen_wait_steps:
        events["completed_listen_wait"] = True
        if env.listen_wait_is_final:
            events["final_listen_ready"] = True
            env._log_event(">>> Listening wait complete at final display.")
        else:
            env.robot.on_listening_complete()
            env.follow_humans = False
            env.robot_start_xy = np.array([rx, ry], dtype=np.float32)
            env._log_event(">>> Listening wait complete. Resume MOVE to Room B.")

        if env.move_back_active:
            events["move_back_completed"] = True
        env.move_back_active = False
        env.move_back_attacker_idx = None

        env.listen_wait_active = False
        env.listen_wait_counter = 0
        env.listen_wait_is_final = False

    _post_step_visual_sync(env, events=events, rx=rx, ry=ry, human_xy=human_xy)

    human_v_follow = np.zeros((len(env.humans), 2), dtype=np.float32)
    human_v_repulsion = np.zeros((len(env.humans), 2), dtype=np.float32)
    human_v_hr = np.zeros((len(env.humans), 2), dtype=np.float32)

    snapshot = env._collect_step_snapshot(
        robot_pose=(rx, ry, ryaw),
        dist=dist,
        desired_yaw=desired_yaw,
        actual_yaw=actual_yaw,
        robot_mode=str(env.robot.mode),
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

    return env._finalize_step_output(
        snapshot=snapshot,
        events=events,
        dist=dist,
        terminated=events["final_listen_ready"],
        external_action_received=external_action_received,
        external_action_used=False,
    )


def step_active_branch(env, external_action_received=False):
    events = env._default_events()

    human_xyz = env._get_human_poses()
    human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
    human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

    robot_pose = env._get_robot_pose()
    env._refresh_callback_rearm_flags()
    callback_request = env._build_callback_request(human_xy=human_xy, robot_pose=robot_pose)
    callback_active_before_step = bool(env.robot.callback_active)
    robot_out = env.robot.step(
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
        env.callback_active_target_idx = target_idx
        if 0 <= target_idx < len(env.callback_triggered_for_current_distracted):
            env.callback_triggered_for_current_distracted[target_idx] = True
            env._log_event(f">>> Robot CALLBACK triggered for person{target_idx + 1}.")
    if callback_active_before_step and (not env.robot.callback_active):
        events["callback_completed"] = True
        recover_idx = env.callback_active_target_idx
        if recover_idx is not None and 0 <= recover_idx < len(env.humans):
            env._apply_callback_response_via_human_fsm(recover_idx=recover_idx, events=events)
        env.callback_active_target_idx = None
        env._log_event(">>> Robot CALLBACK completed.")

    if enter_listen:
        rx, ry, ryaw = robot_pose
        env.listen_reached_logged = set()
        env.listen_wait_active = False
        env.listen_wait_counter = 0
        env.listen_wait_is_final = False
        env.listen_session_count += 1
        n_humans = len(env.humans)

        env._log_event(f">>> Robot entering LISTEN mode. robot=({rx:.2f}, {ry:.2f}, yaw={ryaw:.2f})")
        for i, human in enumerate(env.humans):
            human.assign_listen_target(
                index=i,
                n_humans=n_humans,
                robot_pose=(rx, ry, ryaw),
                listen_radius=env.listen_fan_radius,
                fan_half_angle=env.listen_fan_half_angle,
            )
            gx, gy = human.current_waypoint
            env._log_event(f"    person{i+1} listen_goal=({gx:.3f}, {gy:.3f})")

    env.data.ctrl[:] = 0.0
    env.data.ctrl[0:3] = rb_action

    rx, ry, ryaw = env._get_robot_pose()

    if not env.robot.listen_mode and not env.follow_humans:
        moved_dist = float(np.hypot(rx - env.robot_start_xy[0], ry - env.robot_start_xy[1]))
        if moved_dist >= env.human_follow_distance:
            env.follow_humans = True

    repulsion_vectors = env._compute_social_repulsion(human_xy)
    human_actions = env._update_humans_and_apply_ctrl(
        rx=rx,
        ry=ry,
        ryaw=ryaw,
        repulsion_vectors=repulsion_vectors,
    )

    mujoco.mj_step(env.model, env.data)

    rx, ry, ryaw = env._get_robot_pose()
    human_xyz = env._get_human_poses()
    human_xy = human_xyz[:, :2] if human_xyz.size else np.zeros((0, 2), dtype=np.float32)
    human_actual_yaw = human_xyz[:, 2] if human_xyz.size else np.zeros((0,), dtype=np.float32)

    human_goals = np.array([h.current_waypoint for h in env.humans], dtype=np.float32)
    human_reached_goal = env._check_human_goals(human_xy, human_goals)

    human_v_follow = np.array([h.last_v_follow for h in env.humans], dtype=np.float32)
    human_v_repulsion = np.array([h.last_v_repulsion for h in env.humans], dtype=np.float32)
    human_v_hr = np.array([h.last_v_hr for h in env.humans], dtype=np.float32)

    final_waypoint_reached = env.robot.is_final_reached(dist)
    all_humans_reached = len(env.humans) > 0 and len(human_reached_goal) == len(env.humans)
    events.update(env._handle_listen_transitions(final_waypoint_reached, all_humans_reached))

    _post_step_visual_sync(env, events=events, rx=rx, ry=ry, human_xy=human_xy)

    snapshot = env._collect_step_snapshot(
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

    return env._finalize_step_output(
        snapshot=snapshot,
        events=events,
        dist=float(dist),
        terminated=False,
        external_action_received=external_action_received,
        external_action_used=False,
    )
