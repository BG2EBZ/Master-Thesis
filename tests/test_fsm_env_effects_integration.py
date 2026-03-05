import numpy as np

import museum_env.env as env_module
from museum_env.env import MuseumEnv
from museum_env.human import HumanMode


def _make_env():
    env = MuseumEnv(
        render_mode=None,
        enable_event_logs=False,
    )
    env.reset(seed=0)
    env.follow_humans = True
    for human in env.humans:
        human.configure_following_variant(None, 0.0)
    return env


def _prime_callback_completion(env: MuseumEnv, target_idx: int = 0):
    human = env.humans[target_idx]
    human.transition_to(HumanMode.DISTRACTED, reason="test_prime_distracted", force=True)
    human.distracted_timer = 500
    human.distracted_duration = 1200
    human.callback_response_mode = None
    human.callback_stay_steps_remaining = 0

    env.callback_active_target_idx = int(target_idx)
    env.robot.callback_active = True
    env.robot.callback_target_idx = int(target_idx)
    env.robot.callback_target_xy = None
    env.robot.callback_hold_steps_remaining = 1
    env.robot.callback_turn_done = True


def _prime_wait_attack(env: MuseumEnv):
    env.listen_wait_active = True
    env.listen_wait_counter = 0
    env.listen_wait_is_final = False
    env.attack_triggered_once = True
    env.overwhelmed_triggered_once = True

    attack_idx = int(env.attack_target_idx)
    attacker = env.humans[attack_idx]
    attacker.current_waypoint = np.array([2.5, 1.5], dtype=np.float32)
    attacker.start_attack()
    attacker.attack_hit_distance = 1e-5
    return attack_idx


def _threat_fn_for(env: MuseumEnv, idx: int, *, dist: float = 0.2):
    def _fn(robot_xy, human_xy):
        if idx < 0 or idx >= len(env.humans):
            return None
        if env.humans[idx].mode != HumanMode.ATTACK:
            return None
        xy = np.array(human_xy[idx], dtype=np.float32) if idx < human_xy.shape[0] else np.zeros(2, dtype=np.float32)
        return {"idx": int(idx), "dist": float(dist), "xy": xy}

    return _fn


def test_callback_effects_are_applied_by_env_fsm_path(monkeypatch):
    env = _make_env()
    try:
        _prime_callback_completion(env, target_idx=0)
        monkeypatch.setattr(np.random, "rand", lambda: 0.5)  # stay

        original_decide = env_module.decide_human_fsm_mode

        def _patched_decide(current_mode, ctx, table=None):
            out = original_decide(current_mode=current_mode, ctx=ctx, table=table)
            if ctx.get("callback_response") == "stay":
                out["effects"]["callback_stay_steps"] = 7
            return out

        monkeypatch.setattr(env_module, "decide_human_fsm_mode", _patched_decide)
        _, _, _, _, info = env.step(None)

        assert info["events"]["callback_response_stay"] is True
        assert env.humans[0].callback_response_mode == "stay"
        assert env.humans[0].callback_stay_steps_remaining == 6
    finally:
        env.close()


def test_fear_effects_are_applied_by_env_fsm_path(monkeypatch):
    env = _make_env()
    try:
        attack_idx = _prime_wait_attack(env)
        monkeypatch.setattr(env, "_sample_fear_response", lambda: "stay")
        monkeypatch.setattr(env, "_get_nearest_attack_threat", _threat_fn_for(env, attack_idx, dist=0.2))

        original_decide = env_module.decide_human_fsm_mode

        def _patched_decide(current_mode, ctx, table=None):
            out = original_decide(current_mode=current_mode, ctx=ctx, table=table)
            if ctx.get("fear_response") == "stay":
                out["effects"] = {}
            return out

        monkeypatch.setattr(env_module, "decide_human_fsm_mode", _patched_decide)

        env.step(None)  # trigger fear + stay response (without freeze effect)
        _, _, _, _, info2 = env.step(None)
        attack_action_norm = float(
            np.linalg.norm(
                [
                    info2["humans"]["action"]["vx"][attack_idx],
                    info2["humans"]["action"]["vy"][attack_idx],
                    info2["humans"]["action"]["yaw_rate"][attack_idx],
                ]
            )
        )
        assert attack_action_norm > 0.0
    finally:
        env.close()
