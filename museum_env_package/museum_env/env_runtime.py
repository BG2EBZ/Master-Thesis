from dataclasses import dataclass


@dataclass
class EnvRuntimeState:
    follow_humans: bool = False
    listen_wait_active: bool = False
    listen_wait_counter: int = 0
    listen_wait_is_final: bool = False
    listen_session_count: int = 0
    overwhelmed_triggered_once: bool = False
    attack_triggered_once: bool = False
    attack_hit_once: bool = False
    callback_active_target_idx: int | None = None
    callback_last_response: str | None = None
    callback_last_response_target_idx: int | None = None
    move_back_active: bool = False
    move_back_attacker_idx: int | None = None
    fear_active: bool = False
    fear_attacker_idx: int | None = None
    fear_current_response_mode: str | None = None
    fear_current_response_target_idx: int | None = None
    fear_current_freeze_attack: bool = False
    fear_last_response: str | None = None
    fear_last_response_target_idx: int | None = None


def apply_runtime_state(env, state: EnvRuntimeState, n_humans: int) -> None:
    env.follow_humans = bool(state.follow_humans)
    env.listen_wait_active = bool(state.listen_wait_active)
    env.listen_wait_counter = int(state.listen_wait_counter)
    env.listen_wait_is_final = bool(state.listen_wait_is_final)
    env.listen_session_count = int(state.listen_session_count)
    env.overwhelmed_triggered_once = bool(state.overwhelmed_triggered_once)
    env.attack_triggered_once = bool(state.attack_triggered_once)
    env.attack_hit_once = bool(state.attack_hit_once)
    env.callback_triggered_for_current_distracted = [False] * int(n_humans)
    env.callback_active_target_idx = state.callback_active_target_idx
    env.callback_last_response = state.callback_last_response
    env.callback_last_response_target_idx = state.callback_last_response_target_idx
    env.move_back_active = bool(state.move_back_active)
    env.move_back_attacker_idx = state.move_back_attacker_idx
    env.fear_active = bool(state.fear_active)
    env.fear_attacker_idx = state.fear_attacker_idx
    env.fear_current_response_mode = state.fear_current_response_mode
    env.fear_current_response_target_idx = state.fear_current_response_target_idx
    env.fear_current_freeze_attack = bool(state.fear_current_freeze_attack)
    env.fear_last_response = state.fear_last_response
    env.fear_last_response_target_idx = state.fear_last_response_target_idx
