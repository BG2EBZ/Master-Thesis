import numpy as np

from .human import HumanMode
from .robot import RobotEmotion


def apply_robot_base_color_from_robot_emotion(env, color_fear, color_sad, color_happy, color_natural):
    if env.robot.emotion == RobotEmotion.FEAR:
        env.model.geom_rgba[env.robot_base_geom_id] = color_fear
        return
    if env.robot.emotion == RobotEmotion.SAD:
        env.model.geom_rgba[env.robot_base_geom_id] = color_sad
        return
    if env.robot.emotion == RobotEmotion.HAPPY:
        env.model.geom_rgba[env.robot_base_geom_id] = color_happy
        return
    env.model.geom_rgba[env.robot_base_geom_id] = color_natural


def sync_robot_speaker_state(env):
    env.robot.set_speaker_active(bool(env.listen_wait_active))


def has_any_distracted_human(env):
    return any(h.mode == HumanMode.DISTRACTED for h in env.humans)


def sync_robot_text_label_visibility(env, need_space_group: int, followme_group: int, explanation_group: int):
    show_need_space = bool(env.fear_active)
    show_follow_me = (not show_need_space) and has_any_distracted_human(env)
    show_explanation = (not show_need_space) and (not show_follow_me) and bool(env.robot.speaker_active)
    env._label_scene_option.sitegroup[need_space_group] = 1 if show_need_space else 0
    env._label_scene_option.sitegroup[followme_group] = 1 if show_follow_me else 0
    env._label_scene_option.sitegroup[explanation_group] = 1 if show_explanation else 0


def apply_robot_speaking_halo_visual(env, halo_rgba_on, halo_rgba_off):
    if env.robot.speaker_active:
        env.model.geom_rgba[env.robot_speaking_halo_geom_id] = halo_rgba_on
        return
    env.model.geom_rgba[env.robot_speaking_halo_geom_id] = halo_rgba_off


def get_robot_text_label(env):
    if env.fear_active:
        return "I_need_more_space"
    if has_any_distracted_human(env):
        return "Please_follow_me"
    if env.robot.speaker_active:
        return "explanation"
    return "none"


def update_robot_emotion_and_visual(
    env,
    events,
    robot_xy,
    human_xy,
    fear_distance_threshold: float,
    color_fear,
    color_sad,
    color_happy,
    color_natural,
):
    fear_before = bool(env.fear_active)
    threat = env._get_nearest_attack_threat(robot_xy=robot_xy, human_xy=human_xy)
    fear_now = bool(threat is not None and threat["dist"] < float(fear_distance_threshold))
    env.fear_active = fear_now
    env.fear_attacker_idx = int(threat["idx"]) if fear_now else None
    if (not fear_before) and fear_now:
        events["fear_triggered"] = True
    elif fear_before and (not fear_now):
        events["fear_completed"] = True

    happy_before = int(env.robot.happy_hold_steps_remaining)
    sad_now = any(h.mode in (HumanMode.DISTRACTED, HumanMode.OVERWHELMED) for h in env.humans)
    env.robot.update_emotion([h.mode for h in env.humans], fear_active=env.fear_active)
    happy_after = int(env.robot.happy_hold_steps_remaining)
    if happy_before > 0 and happy_after == 0 and (not sad_now) and (not env.fear_active):
        events["happy_completed"] = True

    apply_robot_base_color_from_robot_emotion(
        env=env,
        color_fear=color_fear,
        color_sad=color_sad,
        color_happy=color_happy,
        color_natural=color_natural,
    )
