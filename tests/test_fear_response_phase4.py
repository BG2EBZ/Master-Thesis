import numpy as np
import pytest

from museum_env.env import MuseumEnv
from museum_env.human import HumanMode


def _make_env():
    env = MuseumEnv(render_mode=None, enable_event_logs=False)
    env.reset(seed=0)
    env.follow_humans = True
    env.listen_wait_active = True
    env.listen_wait_counter = 0
    env.listen_wait_is_final = False
    env.attack_triggered_once = True
    env.overwhelmed_triggered_once = True
    for human in env.humans:
        human.configure_following_variant(None, 0.0)
    return env


def _prime_attacker(env: MuseumEnv):
    attack_idx = int(env.attack_target_idx)
    attacker = env.humans[attack_idx]
    attacker.current_waypoint = np.array([2.5, 1.5], dtype=np.float32)
    attacker.start_attack()
    attacker.attack_hit_distance = 1e-5
    return attack_idx, attacker


def _threat_fn_for(env: MuseumEnv, idx: int, *, dist: float = 0.2, enabled_ref=None):
    def _fn(robot_xy, human_xy):
        if enabled_ref is not None and not enabled_ref["enabled"]:
            return None
        if idx < 0 or idx >= len(env.humans):
            return None
        if env.humans[idx].mode != HumanMode.ATTACK:
            return None
        xy = (
            np.array(human_xy[idx], dtype=np.float32)
            if idx < human_xy.shape[0]
            else np.array([0.0, 0.0], dtype=np.float32)
        )
        return {"idx": int(idx), "dist": float(dist), "xy": xy}

    return _fn


@pytest.mark.parametrize(
    ("response", "event_key"),
    [
        ("move_back", "fear_response_move_back"),
        ("stay", "fear_response_stay"),
        ("continue_hit", "fear_response_continue_hit"),
    ],
)
def test_fear_response_sampling_event_flags(monkeypatch, response, event_key):
    env = _make_env()
    try:
        attack_idx, _ = _prime_attacker(env)
        monkeypatch.setattr(env, "_sample_fear_response", lambda: response)
        monkeypatch.setattr(env, "_get_nearest_attack_threat", _threat_fn_for(env, attack_idx, dist=0.2))

        _, _, _, _, info = env.step(None)
        events = info["events"]

        assert events["fear_triggered"] is True
        assert events[event_key] is True
        for other_key in {
            "fear_response_move_back",
            "fear_response_stay",
            "fear_response_continue_hit",
        } - {event_key}:
            assert events[other_key] is False

        assert info["status"]["fear_last_response"] == response
        assert info["status"]["fear_last_response_target_idx"] == attack_idx
    finally:
        env.close()


def test_move_back_reaction_transitions_to_listening_and_returns_to_anchor(monkeypatch):
    env = _make_env()
    try:
        attack_idx, attacker = _prime_attacker(env)
        expected_anchor = np.array(attacker.attack_origin_listen_waypoint, dtype=np.float32)
        monkeypatch.setattr(env, "_sample_fear_response", lambda: "move_back")
        monkeypatch.setattr(env, "_get_nearest_attack_threat", _threat_fn_for(env, attack_idx, dist=0.2))

        _, _, _, _, info = env.step(None)

        assert info["events"]["fear_response_move_back"] is True
        assert attacker.mode == HumanMode.LISTENING
        assert np.allclose(attacker.current_waypoint, expected_anchor)
    finally:
        env.close()


def test_stay_reaction_freezes_attacker_until_fear_completed_then_listening(monkeypatch):
    env = _make_env()
    try:
        attack_idx, attacker = _prime_attacker(env)
        monkeypatch.setattr(env, "_sample_fear_response", lambda: "stay")
        threat_switch = {"enabled": True}
        monkeypatch.setattr(
            env,
            "_get_nearest_attack_threat",
            _threat_fn_for(env, attack_idx, dist=0.2, enabled_ref=threat_switch),
        )

        # Step 1: trigger fear + sample stay.
        env.step(None)
        assert attacker.mode == HumanMode.ATTACK

        # Step 2: still fear active -> ATTACK action should be frozen to zero.
        _, _, _, _, info2 = env.step(None)
        assert info2["events"]["attack_hit"] is False
        assert np.isclose(info2["humans"]["action"]["vx"][attack_idx], 0.0)
        assert np.isclose(info2["humans"]["action"]["vy"][attack_idx], 0.0)
        assert np.isclose(info2["humans"]["action"]["yaw_rate"][attack_idx], 0.0)

        # Step 3: fear completed -> resolve stay => LISTENING.
        threat_switch["enabled"] = False
        _, _, _, _, info3 = env.step(None)
        assert info3["events"]["fear_completed"] is True
        assert attacker.mode == HumanMode.LISTENING
    finally:
        env.close()


def test_continue_hit_keeps_attack_behavior(monkeypatch):
    env = _make_env()
    try:
        attack_idx, attacker = _prime_attacker(env)
        monkeypatch.setattr(env, "_sample_fear_response", lambda: "continue_hit")
        monkeypatch.setattr(env, "_get_nearest_attack_threat", _threat_fn_for(env, attack_idx, dist=0.2))

        env.step(None)
        _, _, _, _, info2 = env.step(None)

        assert info2["status"]["fear_last_response"] == "continue_hit"
        assert attacker.mode == HumanMode.ATTACK
        action_norm = float(
            np.linalg.norm(
                [
                    info2["humans"]["action"]["vx"][attack_idx],
                    info2["humans"]["action"]["vy"][attack_idx],
                    info2["humans"]["action"]["yaw_rate"][attack_idx],
                ]
            )
        )
        assert action_norm > 0.0
    finally:
        env.close()


def test_only_attacker_gets_fear_response(monkeypatch):
    env = _make_env()
    try:
        attack_idx, attacker = _prime_attacker(env)
        other_idx = 1 if attack_idx != 1 else 0
        other = env.humans[other_idx]
        other.transition_to(HumanMode.ATTACK, reason="test_other_attack", force=True)

        monkeypatch.setattr(env, "_sample_fear_response", lambda: "move_back")
        monkeypatch.setattr(env, "_get_nearest_attack_threat", _threat_fn_for(env, attack_idx, dist=0.2))

        _, _, _, _, info = env.step(None)

        assert info["events"]["fear_response_move_back"] is True
        assert attacker.mode == HumanMode.LISTENING
        assert other.mode == HumanMode.ATTACK
    finally:
        env.close()


def test_fear_response_info_keys_exist():
    env = _make_env()
    try:
        _, _, _, _, info = env.step(None)
        events = info["events"]
        status = info["status"]

        assert "fear_response_move_back" in events
        assert "fear_response_stay" in events
        assert "fear_response_continue_hit" in events

        assert "fear_last_response" in status
        assert "fear_last_response_target_idx" in status
    finally:
        env.close()
