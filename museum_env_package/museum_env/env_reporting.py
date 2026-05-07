from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .robot import RobotEmotion, RobotSpeechMode

HUMAN_LABEL_SITE_GROUP = 2
ROBOT_EXPLANATION_LABEL_GROUP = 3
ROBOT_FOLLOWME_LABEL_GROUP = 4
ROBOT_ANSWER_LABEL_GROUP = 1
HUMAN_LABEL_MODE = mujoco.mjtLabel.mjLABEL_SITE

ROBOT_COLOR_NATURAL = np.array([0.85, 0.85, 0.85, 1.0], dtype=np.float32)
ROBOT_COLOR_SAD = np.array([0.20, 0.45, 0.95, 1.0], dtype=np.float32)
ROBOT_COLOR_HAPPY = np.array([0.95, 0.85, 0.20, 1.0], dtype=np.float32)
SPEAKING_HALO_RGBA_ON = np.array([1.0, 0.9, 0.2, 0.35], dtype=np.float32)
SPEAKING_HALO_RGBA_OFF = np.array([1.0, 0.9, 0.2, 0.0], dtype=np.float32)
HUMAN_SPEAKING_HALO_RGBA_ON = np.array([1.0, 0.9, 0.2, 0.35], dtype=np.float32)
HUMAN_SPEAKING_HALO_RGBA_OFF = np.array([1.0, 0.9, 0.2, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class RobotVisualState:
    base_rgba: np.ndarray
    halo_rgba: np.ndarray
    show_follow_me: bool
    show_explanation: bool
    show_answer: bool
    text_label: str
    signature: tuple[str, str, bool]


def build_label_scene_option():
    opt = mujoco.MjvOption()
    opt.label = HUMAN_LABEL_MODE
    opt.sitegroup[:] = 0
    opt.sitegroup[HUMAN_LABEL_SITE_GROUP] = 1
    opt.sitegroup[ROBOT_EXPLANATION_LABEL_GROUP] = 0
    opt.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP] = 0
    opt.sitegroup[ROBOT_ANSWER_LABEL_GROUP] = 0
    return opt


def resolve_robot_visual_state(*, robot, callback_visual_active: bool) -> RobotVisualState:
    if robot.emotion == RobotEmotion.SAD:
        base_rgba = ROBOT_COLOR_SAD
    elif robot.emotion == RobotEmotion.HAPPY:
        base_rgba = ROBOT_COLOR_HAPPY
    else:
        base_rgba = ROBOT_COLOR_NATURAL

    show_follow_me = bool(callback_visual_active)
    show_explanation = (not show_follow_me) and (robot.speech_mode == RobotSpeechMode.EXPLANATION)
    show_answer = (not show_follow_me) and (robot.speech_mode == RobotSpeechMode.ANSWER)
    if show_follow_me:
        text_label = "Please_follow_me"
    elif show_explanation:
        text_label = "explanation"
    elif show_answer:
        text_label = "answer question"
    else:
        text_label = "none"

    halo_rgba = SPEAKING_HALO_RGBA_ON if robot.speaker_active else SPEAKING_HALO_RGBA_OFF
    return RobotVisualState(
        base_rgba=np.array(base_rgba, dtype=np.float32),
        halo_rgba=np.array(halo_rgba, dtype=np.float32),
        show_follow_me=show_follow_me,
        show_explanation=show_explanation,
        show_answer=show_answer,
        text_label=text_label,
        signature=(str(robot.emotion), str(robot.speech_mode), bool(callback_visual_active)),
    )


def apply_robot_visual_state(
    *,
    model,
    robot_base_geom_id: int,
    robot_speaking_halo_geom_id: int,
    label_scene_option,
    visual_state: RobotVisualState,
) -> None:
    model.geom_rgba[robot_base_geom_id] = visual_state.base_rgba
    model.geom_rgba[robot_speaking_halo_geom_id] = visual_state.halo_rgba
    label_scene_option.sitegroup[ROBOT_FOLLOWME_LABEL_GROUP] = 1 if visual_state.show_follow_me else 0
    label_scene_option.sitegroup[ROBOT_EXPLANATION_LABEL_GROUP] = (
        1 if visual_state.show_explanation else 0
    )
    label_scene_option.sitegroup[ROBOT_ANSWER_LABEL_GROUP] = 1 if visual_state.show_answer else 0


def apply_label_scene_option_to_viewer(*, viewer, label_scene_option) -> None:
    if viewer is None:
        return
    viewer.opt.label = label_scene_option.label
    viewer.opt.sitegroup[:] = label_scene_option.sitegroup


def build_step_info(
    *,
    events,
    step_count: int,
    follow_phase,
    listen_phase: str,
    robot,
    terminated_reason,
    world_frame,
    robot_action,
    human_goals,
    humans,
    reached_goal_indices,
    perceived_distracted_indices,
) -> dict:
    return {
        "events": events.as_dict(),
        "state": {
            "step_count": int(step_count),
            "follow_phase": follow_phase,
            "listen_phase": listen_phase,
            "robot_mode": str(robot.mode),
            "callback_phase": (
                str(robot.callback_phase) if robot.callback_phase is not None else None
            ),
            "robot_emotion": str(robot.emotion),
            "speaker_active": bool(robot.speaker_active),
            "terminated_reason": terminated_reason,
        },
        "robot": {
            "pose_xy": np.array(world_frame.robot_xy, dtype=np.float32),
            "goal_xy": np.array(robot.get_current_waypoint(), dtype=np.float32),
            "dist_to_goal": float(
                np.hypot(
                    float(robot.get_current_waypoint()[0]) - float(world_frame.robot_xy[0]),
                    float(robot.get_current_waypoint()[1]) - float(world_frame.robot_xy[1]),
                )
            ),
            "yaw": float(world_frame.robot_pose[2]),
            "action": {
                "vx": float(robot_action[0]),
                "vy": float(robot_action[1]),
                "yaw_rate": float(robot_action[2]),
            },
        },
        "humans": {
            "pose_xy": np.array(world_frame.human_xy, dtype=np.float32),
            "goal_xy": np.array(human_goals, dtype=np.float32),
            "mode": [human.mode for human in humans],
            "profile": [human.profile for human in humans],
            "reached_goal_indices": [int(idx) for idx in reached_goal_indices],
            "perceived_distracted_indices": [int(idx) for idx in perceived_distracted_indices],
        },
    }
