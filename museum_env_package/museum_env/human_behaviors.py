"""Mode-specific human behavior implementations.

This module keeps the high-churn step logic out of ``human.py`` so the
``Human`` class can stay focused on state, transitions, and shared motion
utilities.
"""

import numpy as np

from .human import (
    DISTRACTED_BEHAVIOR_CONVERSATION,
    DISTRACTED_CONVERSATION_STOP_DISTANCE,
    DISTRACTED_EXHIBIT_LOOK_RADIUS,
    DISTRACTED_FALLBACK_DISTANCE,
    DISTRACTED_HUMAN_LOOK_RADIUS,
    DISTRACTED_SOURCE_LISTENING,
    DISTRACTED_SPEED_SCALE,
    DISTRACTED_TARGET_DISTANCE_MIN,
    DISTRACTED_YAW_DEVIATION_MAX_DEG,
    DISTRACTED_YAW_DEVIATION_MIN_DEG,
    HUMAN_ROTATION_STOP_DEG,
    HUMAN_YAW_RATE_GAIN,
    LISTENING_IMPATIENT_TARGET_REACHED_DEG,
    LISTENING_RING_GAIN,
    LISTENING_SECTOR_PROJECTION_EPS,
    MIN_SPEED_EPS,
    NORM_EPS,
    OVERWHELMED_STAGE_SWITCH_DIST,
    HumanMode,
    logger,
)


def step_behavior(human, ctx, pose):
    """Dispatch behavior based on explicit variants first, then on mode."""
    behavior_kind = ctx.get("behavior_kind")

    # These variants intentionally override the default mode behavior. They are
    # used for post-explanation coordination where mode alone is not specific
    # enough to describe the intended motion policy.
    if behavior_kind == "post_explanation_yield":
        return _step_post_explanation_yield(human, ctx, pose)
    if behavior_kind == "post_explanation_listening_anchor":
        return _step_post_explanation_listening_anchor(human, ctx, pose)
    if behavior_kind is not None:
        raise ValueError(f"Unknown human behavior kind {behavior_kind!r}")

    if human.mode == HumanMode.WANDERING:
        return _step_wandering(human, ctx, pose)
    if human.mode == HumanMode.FOLLOWING:
        human.assign_target_from_context(ctx)
        return _step_following(human, ctx, pose)
    if human.mode == HumanMode.LISTENING:
        return _step_listening(human, ctx, pose)
    if human.mode == HumanMode.DISTRACTED:
        return _step_distracted(human, ctx, pose)
    if human.mode == HumanMode.OVERWHELMED:
        return _step_overwhelmed(human, ctx, pose)
    if human.mode == HumanMode.IMPATIENT:
        return _step_impatient(human, ctx, pose)
    raise ValueError(f"Unknown human mode {human.mode}")


def _step_wandering(human, ctx, pose):
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    yaw = pose[2]
    to_waypoint = human.current_waypoint - current_xy
    if np.linalg.norm(to_waypoint) < human.waypoint_threshold:
        human.current_waypoint = human._random_waypoint()
        to_waypoint = human.current_waypoint - current_xy
    return human._move(to_waypoint, yaw, ctx, current_xy)


def _step_following(human, ctx, pose):
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    return human._move(human.current_waypoint - current_xy, pose[2], ctx, current_xy)


def _step_listening(human, ctx, pose):
    yaw = pose[2]
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
    robot_yaw = ctx["robot_yaw"]
    to_robot = robot_xy - current_xy
    dist_to_robot = np.linalg.norm(to_robot)
    desired_yaw = np.arctan2(to_robot[1], to_robot[0]) if dist_to_robot > NORM_EPS else robot_yaw
    yaw_err = human._wrap_to_pi(desired_yaw - yaw)

    target_xy = _compute_listening_sector_target_point(
        human,
        current_xy=current_xy,
        robot_xy=robot_xy,
        robot_yaw=robot_yaw,
        listen_radius=ctx["listen_radius"],
        sector_half_angle=ctx["listening_sector_half_angle"],
    )
    guide_xy = target_xy - current_xy
    v_goal = LISTENING_RING_GAIN * guide_xy
    v_total = human._compose_move_velocity(
        current_xy=current_xy,
        guide_xy=guide_xy,
        goal_v_xy=v_goal,
        speed_limit=human.max_speed,
        repulsion_xy=ctx["repulsion"],
        robot_xy=robot_xy,
        hr_distance_min=human.hr_distance_min,
        hr_distance_max=None,
    )

    action = human._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
    return action


def _step_post_explanation_listening_anchor(human, ctx, pose):
    """Listen around a frozen anchor pose while still reacting to live repulsion."""
    yaw = pose[2]
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    anchor_robot_xy = np.asarray(ctx["anchor_robot_xy"], dtype=np.float32)
    live_robot_xy = np.asarray(ctx["live_robot_xy"], dtype=np.float32)
    anchor_robot_yaw = float(ctx["anchor_robot_yaw"])

    to_anchor_robot = anchor_robot_xy - current_xy
    dist_to_anchor_robot = np.linalg.norm(to_anchor_robot)
    desired_yaw = (
        np.arctan2(to_anchor_robot[1], to_anchor_robot[0])
        if dist_to_anchor_robot > NORM_EPS
        else anchor_robot_yaw
    )
    yaw_err = human._wrap_to_pi(desired_yaw - yaw)

    target_xy = _compute_listening_sector_target_point(
        human,
        current_xy=current_xy,
        robot_xy=anchor_robot_xy,
        robot_yaw=anchor_robot_yaw,
        listen_radius=ctx["listen_radius"],
        sector_half_angle=ctx["listening_sector_half_angle"],
    )
    guide_xy = target_xy - current_xy
    v_goal = LISTENING_RING_GAIN * guide_xy
    v_total = human._compose_move_velocity(
        current_xy=current_xy,
        guide_xy=guide_xy,
        goal_v_xy=v_goal,
        speed_limit=human.max_speed,
        repulsion_xy=ctx["repulsion"],
        robot_xy=live_robot_xy,
        hr_distance_min=human.hr_distance_min,
        hr_distance_max=None,
    )

    action = human._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)
    return action


def _step_distracted(human, ctx, pose):
    human.distracted_elapsed_steps += 1
    yaw = pose[2]
    current_xy = np.asarray(pose[:2], dtype=np.float32)

    partner_xy = _sync_distracted_conversation_state(human, ctx, current_xy)
    if partner_xy is not None:
        action = _step_distracted_conversation(
            human,
            ctx,
            current_xy=current_xy,
            current_yaw=yaw,
            partner_xy=partner_xy,
        )

    elif human.distracted_source == DISTRACTED_SOURCE_LISTENING:
        action = _step_listening_distracted(human, ctx, pose)
    else:
        # Check the distracted target state and initialize if needed.
        if human.distracted_target_xy is None:
            _initialize_distracted_target(
                human,
                ctx,
                current_xy=current_xy,
                fallback_reference_yaw=yaw,
            )

        target_xy = np.asarray(human.distracted_target_xy, dtype=np.float32)
        to_target_xy = target_xy - current_xy
        dist_to_target = np.linalg.norm(to_target_xy)
        if (not human.distracted_stop_reached) and dist_to_target <= human.waypoint_threshold:
            human.distracted_stop_reached = True
            human.distracted_timer = 0

        desired_yaw = _resolve_focus_yaw(human, current_xy, fallback_yaw=yaw)
        if human.distracted_stop_reached:
            action = _step_following_distracted_stop(
                human,
                current_yaw=yaw,
                desired_yaw=desired_yaw,
            )
        else:
            move_speed_limit = DISTRACTED_SPEED_SCALE * human.max_speed
            if dist_to_target > NORM_EPS:
                v_goal = move_speed_limit * (to_target_xy / dist_to_target)
            else:
                v_goal = np.zeros(2, dtype=np.float32)

            robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
            v_total = human._compose_move_velocity(
                current_xy=current_xy,
                guide_xy=to_target_xy,
                goal_v_xy=v_goal,
                speed_limit=move_speed_limit,
                repulsion_xy=ctx["repulsion"],
                robot_xy=robot_xy,
                hr_distance_min=human.hr_distance_min,
                hr_distance_max=human.hr_distance_max,
            )
            guide_norm = float(np.linalg.norm(to_target_xy))
            if guide_norm > NORM_EPS:
                guide_dir = to_target_xy / guide_norm
                if float(np.dot(v_total, guide_dir)) <= MIN_SPEED_EPS:
                    fallback_v_xy = human._constrain_velocity_with_walkable(v_goal)
                    if float(np.dot(fallback_v_xy, guide_dir)) > MIN_SPEED_EPS:
                        v_total = np.asarray(fallback_v_xy, dtype=np.float32)
                    else:
                        v_total = np.zeros(2, dtype=np.float32)

            yaw_err = human._wrap_to_pi(desired_yaw - yaw)
            action = human._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)

    if (
        human.mode == HumanMode.DISTRACTED
        and human.distracted_elapsed_steps >= human.distracted_duration
    ):
        human.set_mode(human.distracted_recovery_mode)
        if human.enable_event_logs:
            logger.info(f">>> {human.name} recovered from DISTRACTED timeout -> {human.mode.upper()}")
    return action

def _step_following_distracted_stop(human, *, current_yaw: float, desired_yaw: float):
    human.distracted_timer += 1
    yaw_err = human._wrap_to_pi(desired_yaw - current_yaw)
    if abs(yaw_err) >= np.deg2rad(HUMAN_ROTATION_STOP_DEG):
        action = human._compose_action(np.zeros(2, dtype=np.float32), HUMAN_YAW_RATE_GAIN * yaw_err)
    else:
        action = np.zeros(3, dtype=np.float32)

    if human.distracted_timer >= human.distracted_stop_duration:
        human.set_mode(human.distracted_recovery_mode)
        if human.enable_event_logs:
            logger.info(f">>> {human.name} recovered -> {human.mode.upper()}")
    return action


def _step_listening_distracted(human, ctx, pose):
    human.distracted_timer += 1
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    desired_yaw = human.distracted_target_yaw
    # Generate or update the distracted target if needed.
    if desired_yaw is None:
        robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
        robot_facing_yaw = np.arctan2(
            robot_xy[1] - current_xy[1],
            robot_xy[0] - current_xy[0],
        )
        _initialize_distracted_target(
            human,
            ctx,
            current_xy=current_xy,
            fallback_reference_yaw=robot_facing_yaw,
            fallback_target_xy=current_xy,
        )
        desired_yaw = human.distracted_target_yaw
    yaw_err = human._wrap_to_pi(desired_yaw - pose[2])

    if abs(yaw_err) >= np.deg2rad(HUMAN_ROTATION_STOP_DEG):
        return human._compose_action(np.zeros(2, dtype=np.float32), HUMAN_YAW_RATE_GAIN * yaw_err)
    return np.zeros(3, dtype=np.float32)


def _step_overwhelmed(human, ctx, pose):
    if human.overwhelmed_stage == "pause":
        human.overwhelmed_pause_timer += 1
        if human.overwhelmed_pause_timer >= human.overwhelmed_pause_duration:
            human.set_mode(human.overwhelmed_recovery_mode)
            if human.enable_event_logs:
                logger.info(f">>> {human.name} recovered from OVERWHELMED -> {human.mode.upper()}")
        return np.zeros(3, dtype=np.float32)

    pos_xy = np.asarray(pose[:2], dtype=np.float32)
    leave_dir = human.overwhelmed_leave_dir
    desired_yaw = np.arctan2(leave_dir[1], leave_dir[0])

    if human.overwhelmed_stage == "backoff":
        backoff_target = human.overwhelmed_backoff_start_xy + human.overwhelmed_backoff_dist * leave_dir
        to_target = backoff_target - pos_xy
        dist_to_target = np.linalg.norm(to_target)
        if dist_to_target < OVERWHELMED_STAGE_SWITCH_DIST:
            human.overwhelmed_stage = "leave"
            dist_to_target = 0.0

        move_speed_limit = min(human.overwhelmed_leave_speed, human.max_speed)
        if dist_to_target > NORM_EPS:
            goal_v_xy = move_speed_limit * (to_target / dist_to_target)
        else:
            goal_v_xy = np.zeros(2, dtype=np.float32)
        v_xy = human._compose_move_velocity(
            current_xy=pos_xy,
            guide_xy=to_target,
            goal_v_xy=goal_v_xy,
            speed_limit=move_speed_limit,
        )

        action = human._compose_action(v_xy, HUMAN_YAW_RATE_GAIN * human._wrap_to_pi(desired_yaw - pose[2]))
        return action

    human.overwhelmed_leave_timer += 1
    move_speed_limit = min(human.overwhelmed_leave_speed, human.max_speed)
    goal_v_xy = move_speed_limit * leave_dir
    v_xy = human._compose_move_velocity(
        current_xy=pos_xy,
        guide_xy=leave_dir,
        goal_v_xy=goal_v_xy,
        speed_limit=move_speed_limit,
    )
    if human.overwhelmed_leave_timer >= human.overwhelmed_leave_duration:
        human.overwhelmed_stage = "pause"
        human.overwhelmed_pause_timer = 0

    action = human._compose_action(v_xy, HUMAN_YAW_RATE_GAIN * human._wrap_to_pi(desired_yaw - pose[2]))
    return action


def _step_listening_impatient_glance(human, *, base_yaw: float, yaw: float):
    desired_yaw = human._wrap_to_pi(
        base_yaw + human.listening_impatient_turn_sign * human.listening_impatient_yaw_deviation
    )
    yaw_err = human._wrap_to_pi(desired_yaw - yaw)
    if abs(yaw_err) <= np.deg2rad(LISTENING_IMPATIENT_TARGET_REACHED_DEG):
        human.listening_impatient_turn_sign *= -1.0
        desired_yaw = human._wrap_to_pi(
            base_yaw + human.listening_impatient_turn_sign * human.listening_impatient_yaw_deviation
        )
        yaw_err = human._wrap_to_pi(desired_yaw - yaw)
    return human._compose_action(np.zeros(2, dtype=np.float32), HUMAN_YAW_RATE_GAIN * yaw_err)


def _step_impatient(human, ctx, pose):
    human.impatient_timer += 1
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    yaw = pose[2]
    if human.impatient_recovery_mode == HumanMode.LISTENING:
        robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
        to_robot = robot_xy - current_xy
        base_yaw = np.arctan2(to_robot[1], to_robot[0]) if np.linalg.norm(to_robot) > NORM_EPS else yaw
        look_steps = max(1, int(human.impatient_duration) // 2)
        if human.impatient_timer <= look_steps:
            action = _step_listening_impatient_glance(human, base_yaw=base_yaw, yaw=yaw)
        else:
            room_regions = human.map_layout.metadata.get("room_regions", {})
            transition_targets = human.map_layout.metadata.get("impatient_transition_targets", {})
            source_room = next(
                (
                    str(room_name)
                    for room_name, room_region in room_regions.items()
                    if hasattr(room_region, "contains_point") and room_region.contains_point(robot_xy)
                ),
                None,
            )
            if source_room is None:
                source_room = min(
                    (
                        str(room_name)
                        for room_name, room_region in room_regions.items()
                        if hasattr(room_region, "center")
                    ),
                    key=lambda room_name: float(
                        np.linalg.norm(
                            np.asarray(room_regions[room_name].center(), dtype=np.float32) - robot_xy
                        )
                    ),
                    default=None,
                )

            target_spec = transition_targets.get(source_room, {})
            approach_xy = target_spec.get("approach_xy")
            focus_xy = target_spec.get("focus_xy")
            if approach_xy is None or focus_xy is None:
                action = _step_listening_impatient_glance(human, base_yaw=base_yaw, yaw=yaw)
            else:
                target_xy = np.asarray(approach_xy, dtype=np.float32)
                focus_xy = np.asarray(focus_xy, dtype=np.float32)
                to_target_xy = target_xy - current_xy
                dist_to_target = float(np.linalg.norm(to_target_xy))
                focus_delta = focus_xy - current_xy
                desired_yaw = (
                    float(np.arctan2(focus_delta[1], focus_delta[0]))
                    if np.linalg.norm(focus_delta) > NORM_EPS
                    else float(yaw)
                )

                if dist_to_target <= human.waypoint_threshold:
                    action = human._compose_action(
                        np.zeros(2, dtype=np.float32),
                        HUMAN_YAW_RATE_GAIN * human._wrap_to_pi(desired_yaw - yaw),
                    )
                else:
                    v_goal = human.max_speed * (to_target_xy / dist_to_target)
                    v_total = human._compose_move_velocity(
                        current_xy=current_xy,
                        guide_xy=to_target_xy,
                        goal_v_xy=v_goal,
                        speed_limit=human.max_speed,
                        repulsion_xy=ctx["repulsion"],
                        robot_xy=robot_xy,
                        hr_distance_min=human.hr_distance_min,
                        hr_distance_max=None,
                    )
                    action = human._compose_action(
                        v_total,
                        HUMAN_YAW_RATE_GAIN * human._wrap_to_pi(desired_yaw - yaw),
                    )
    else:
        human.assign_target_from_context(ctx)
        action = _step_following(human, ctx, pose)

    if human.impatient_timer >= human.impatient_duration:
        human.set_mode(human.impatient_recovery_mode)
        if human.enable_event_logs:
            logger.info(f">>> {human.name} recovered from IMPATIENT -> {human.mode.upper()}")
    return action


def _step_post_explanation_yield(human, ctx, pose):
    """Move toward a temporary yield target without exposing a custom public step."""
    target_xy = np.asarray(ctx["target_xy"], dtype=np.float32)
    human.current_waypoint = target_xy.copy()
    current_xy = np.asarray(pose[:2], dtype=np.float32)
    return human._move(target_xy - current_xy, pose[2], ctx, current_xy)


def _step_distracted_conversation(human, ctx, *, current_xy, current_yaw: float, partner_xy):
    partner_xy = np.asarray(partner_xy, dtype=np.float32)
    target_xy = np.asarray(human.distracted_target_xy, dtype=np.float32)
    to_partner_xy = partner_xy - current_xy
    dist_to_partner = float(np.linalg.norm(to_partner_xy))
    desired_yaw = float(
        np.arctan2(to_partner_xy[1], to_partner_xy[0])
        if dist_to_partner > NORM_EPS
        else human.distracted_target_yaw
    )

    if dist_to_partner <= DISTRACTED_CONVERSATION_STOP_DISTANCE + NORM_EPS:
        yaw_err = human._wrap_to_pi(desired_yaw - current_yaw)
        if abs(yaw_err) >= np.deg2rad(HUMAN_ROTATION_STOP_DEG):
            return human._compose_action(
                np.zeros(2, dtype=np.float32),
                HUMAN_YAW_RATE_GAIN * yaw_err,
            )
        return np.zeros(3, dtype=np.float32)

    to_target_xy = target_xy - current_xy
    move_speed_limit = DISTRACTED_SPEED_SCALE * human.max_speed
    if np.linalg.norm(to_target_xy) > NORM_EPS:
        v_goal = move_speed_limit * (to_target_xy / np.linalg.norm(to_target_xy))
    else:
        v_goal = np.zeros(2, dtype=np.float32)

    robot_xy = np.asarray(ctx["robot_xy"], dtype=np.float32)
    v_total = human._compose_move_velocity(
        current_xy=current_xy,
        guide_xy=to_target_xy,
        goal_v_xy=v_goal,
        speed_limit=move_speed_limit,
        repulsion_xy=ctx["repulsion"],
        robot_xy=robot_xy,
        hr_distance_min=human.hr_distance_min,
        hr_distance_max=human.hr_distance_max,
    )

    yaw_err = human._wrap_to_pi(desired_yaw - current_yaw)
    return human._compose_action(v_total, HUMAN_YAW_RATE_GAIN * yaw_err)


def _sync_distracted_conversation_state(human, ctx, current_xy):
    partner = _resolve_distracted_conversation_partner(human, ctx, current_xy)
    if partner is None:
        if human.distracted_behavior_kind == DISTRACTED_BEHAVIOR_CONVERSATION:
            human._clear_distracted_navigation_state()
        return None

    partner_index, partner_xy = partner
    partner_xy = np.asarray(partner_xy, dtype=np.float32)
    to_partner_xy = partner_xy - current_xy
    dist_to_partner = float(np.linalg.norm(to_partner_xy))
    target_yaw = float(
        np.arctan2(to_partner_xy[1], to_partner_xy[0])
        if dist_to_partner > NORM_EPS
        else human.distracted_target_yaw if human.distracted_target_yaw is not None else 0.0
    )
    if dist_to_partner > DISTRACTED_CONVERSATION_STOP_DISTANCE + NORM_EPS:
        target_xy = (
            partner_xy
            - DISTRACTED_CONVERSATION_STOP_DISTANCE * (to_partner_xy / dist_to_partner)
        )
    else:
        target_xy = np.asarray(current_xy, dtype=np.float32).copy()

    human._set_distracted_target_state(
        target_yaw=target_yaw,
        target_xy=target_xy,
        behavior_kind=DISTRACTED_BEHAVIOR_CONVERSATION,
        partner_index=partner_index,
    )
    return partner_xy


def _resolve_distracted_conversation_partner(human, ctx, current_xy):
    human_xy = np.asarray(ctx.get("human_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if human_xy.size == 0:
        return None
    if human_xy.ndim == 1:
        human_xy = human_xy.reshape(1, -1)

    human_modes = tuple(ctx.get("human_modes", ()))
    if len(human_modes) != human_xy.shape[0]:
        human_modes = tuple([None] * human_xy.shape[0])

    current_index = int(ctx.get("index", -1))
    partner_index = human.distracted_partner_index
    if partner_index is not None and 0 <= int(partner_index) < human_xy.shape[0]:
        partner_index = int(partner_index)
        if (
            partner_index != current_index
            and human_modes[partner_index] == HumanMode.DISTRACTED
        ):
            partner_xy = np.asarray(human_xy[partner_index, :2], dtype=np.float32)
            if np.linalg.norm(partner_xy - current_xy) <= DISTRACTED_HUMAN_LOOK_RADIUS:
                return partner_index, partner_xy

    candidate_indices = [
        idx
        for idx, mode in enumerate(human_modes)
        if idx != current_index and mode == HumanMode.DISTRACTED
    ]
    if not candidate_indices:
        return None

    candidate_xy = human_xy[np.asarray(candidate_indices, dtype=np.int32), :2]
    return _select_nearest_candidate_index(
        current_xy,
        candidate_xy,
        DISTRACTED_HUMAN_LOOK_RADIUS,
        candidate_indices=candidate_indices,
    )


def _resolve_focus_yaw(human, current_xy, *, fallback_yaw: float) -> float:
    focus_target_xy = None
    if human.distracted_target_xy is not None:
        focus_target_xy = np.asarray(human.distracted_target_xy, dtype=np.float32)

    if focus_target_xy is not None:
        focus_delta = focus_target_xy - current_xy
        if np.linalg.norm(focus_delta) > NORM_EPS:
            return float(np.arctan2(focus_delta[1], focus_delta[0]))
    if human.distracted_target_yaw is not None:
        return float(human.distracted_target_yaw)
    return float(fallback_yaw)


def _select_nearest_candidate_index(
    current_xy,
    candidates,
    max_distance: float,
    *,
    candidate_indices=None,
    exclude_index=None,
):
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.size == 0:
        return None, None
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    if candidate_indices is None:
        candidate_indices = np.arange(candidates.shape[0], dtype=np.int32)
    else:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int32)

    deltas = candidates[:, :2] - np.asarray(current_xy, dtype=np.float32)[None, :]
    distances = np.linalg.norm(deltas, axis=1)
    if exclude_index is not None:
        distances[candidate_indices == int(exclude_index)] = np.inf

    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    if (not np.isfinite(best_distance)) or best_distance > float(max_distance):
        return None, None
    return int(candidate_indices[best_idx]), np.asarray(candidates[best_idx, :2], dtype=np.float32)


def _select_nearest_candidate(current_xy, candidates, max_distance: float, *, exclude_index=None):
    _candidate_index, candidate_xy = _select_nearest_candidate_index(
        current_xy,
        candidates,
        max_distance,
        exclude_index=exclude_index,
    )
    return candidate_xy


def _get_distractor_exhibit_points(human):
    exhibit_points = human.map_layout.metadata.get("distractor_exhibit_points", ())
    points = np.asarray(exhibit_points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return points.reshape(-1, 2)


def _initialize_distracted_target(
    human,
    ctx,
    current_xy,
    *,
    fallback_reference_yaw: float,
    fallback_target_xy=None,
):
    current_xy = np.asarray(current_xy, dtype=np.float32)

    exhibit_target_xy = _select_nearest_candidate(
        current_xy,
        _get_distractor_exhibit_points(human),
        DISTRACTED_EXHIBIT_LOOK_RADIUS,
    )
    if exhibit_target_xy is not None:
        to_exhibit_xy = exhibit_target_xy - current_xy
        dist_to_exhibit = float(np.linalg.norm(to_exhibit_xy))
        target_yaw = np.arctan2(
            exhibit_target_xy[1] - current_xy[1],
            exhibit_target_xy[0] - current_xy[0],
        )
        if dist_to_exhibit > DISTRACTED_TARGET_DISTANCE_MIN + NORM_EPS:
            target_xy = (
                exhibit_target_xy
                - DISTRACTED_TARGET_DISTANCE_MIN * (to_exhibit_xy / dist_to_exhibit)
            )
        else:
            target_xy = current_xy.copy()
        human._set_distracted_target_state(target_yaw=target_yaw, target_xy=target_xy)
        return

    focus_target_xy = _select_nearest_candidate(
        current_xy,
        ctx.get("human_xy", np.zeros((0, 2), dtype=np.float32)),
        DISTRACTED_HUMAN_LOOK_RADIUS,
        exclude_index=ctx.get("index"),
    )
    if focus_target_xy is not None:
        target_yaw = np.arctan2(
            focus_target_xy[1] - current_xy[1],
            focus_target_xy[0] - current_xy[0],
        )
        human._set_distracted_target_state(target_yaw=target_yaw, target_xy=focus_target_xy)
        return

    # No valid focus target, so pick a random fallback yaw and corresponding target point.
    deviation_deg = np.random.uniform(
        DISTRACTED_YAW_DEVIATION_MIN_DEG,
        DISTRACTED_YAW_DEVIATION_MAX_DEG,
    )
    deviation_sign = -1.0 if np.random.rand() < 0.5 else 1.0
    deviation_rad = np.deg2rad(deviation_deg) * deviation_sign
    fallback_yaw = _wrap_to_pi(float(fallback_reference_yaw) + deviation_rad)
    if fallback_target_xy is None:
        direction_xy = np.array([np.cos(fallback_yaw), np.sin(fallback_yaw)], dtype=np.float32)
        fallback_target_xy = current_xy + DISTRACTED_FALLBACK_DISTANCE * direction_xy

    human._set_distracted_target_state(
        target_yaw=float(fallback_yaw),
        target_xy=np.asarray(fallback_target_xy, dtype=np.float32),
    )


def _compute_listening_sector_target_point(
    human,
    *,
    current_xy,
    robot_xy,
    robot_yaw: float,
    listen_radius: float,
    sector_half_angle: float,
):
    current_xy = np.asarray(current_xy, dtype=np.float32)
    robot_xy = np.asarray(robot_xy, dtype=np.float32)
    rel_xy = current_xy - robot_xy
    if np.dot(rel_xy, rel_xy) <= (NORM_EPS * NORM_EPS):
        absolute_angle = robot_yaw
    else:
        absolute_angle = np.arctan2(rel_xy[1], rel_xy[0])
    relative_angle = human._wrap_to_pi(absolute_angle - robot_yaw)
    half_angle = max(0.0, sector_half_angle - LISTENING_SECTOR_PROJECTION_EPS)
    clamped_angle = human._wrap_to_pi(robot_yaw + np.clip(relative_angle, -half_angle, half_angle))
    return np.array(
        [
            robot_xy[0] + listen_radius * np.cos(clamped_angle),
            robot_xy[1] + listen_radius * np.sin(clamped_angle),
        ],
        dtype=np.float32,
    )


def _wrap_to_pi(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi
