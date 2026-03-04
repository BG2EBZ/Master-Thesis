from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence


HUMAN_MODE_WANDERING = "wandering"
HUMAN_MODE_FOLLOWING = "following"
HUMAN_MODE_LISTENING = "listening"
HUMAN_MODE_DISTRACTED = "distracted"
HUMAN_MODE_OVERWHELMED = "overwhelmed"
HUMAN_MODE_IMPATIENT = "impatient"
HUMAN_MODE_ATTACK = "attack"


@dataclass(frozen=True)
class Transition:
    src: str    # source state
    event: str 
    dst: str    # destination state
    guard: Optional[Callable[[dict], bool]] = None  # guard condition (optional)
    effects: Optional[Callable[[dict], dict]] = None # actions


def apply_transitions(
    current_state: str,
    events: Iterable[str],
    ctx: dict,
    table: Sequence[Transition],
) -> tuple[str, dict]:
    event_set = set(events)
    for tr in table:
        if tr.src != current_state:
            continue
        if tr.event not in event_set:
            continue
        if tr.guard is not None and (not bool(tr.guard(ctx))):
            continue
        effects = tr.effects(ctx) if tr.effects is not None else {}
        return tr.dst, dict(effects)
    return str(current_state), {}


def build_transition_table() -> list[Transition]:
    return [
        Transition(src=HUMAN_MODE_ATTACK, event="attack_hit", dst=HUMAN_MODE_LISTENING),
        Transition(
            src=HUMAN_MODE_ATTACK,
            event="fear_move_back",
            dst=HUMAN_MODE_LISTENING,
            effects=lambda c: {"restore_listen_anchor": True},
        ),
        Transition(
            src=HUMAN_MODE_ATTACK,
            event="fear_stay",
            dst=HUMAN_MODE_ATTACK,
            effects=lambda c: {"freeze_attack": True},
        ),
        Transition(src=HUMAN_MODE_ATTACK, event="fear_continue_hit", dst=HUMAN_MODE_ATTACK),
        Transition(
            src=HUMAN_MODE_ATTACK,
            event="fear_completed_stay_active",
            dst=HUMAN_MODE_LISTENING,
        ),
        Transition(src=HUMAN_MODE_WANDERING, event="robot_listen_mode", dst=HUMAN_MODE_LISTENING),
        Transition(src=HUMAN_MODE_FOLLOWING, event="robot_listen_mode", dst=HUMAN_MODE_LISTENING),
        Transition(src=HUMAN_MODE_LISTENING, event="robot_listen_mode", dst=HUMAN_MODE_LISTENING),
        Transition(src=HUMAN_MODE_DISTRACTED, event="robot_listen_mode", dst=HUMAN_MODE_LISTENING),
        Transition(src=HUMAN_MODE_IMPATIENT, event="robot_listen_mode", dst=HUMAN_MODE_LISTENING),
        Transition(src=HUMAN_MODE_ATTACK, event="robot_listen_mode", dst=HUMAN_MODE_LISTENING),
        Transition(src=HUMAN_MODE_DISTRACTED, event="callback_rejoin", dst=HUMAN_MODE_FOLLOWING),
        Transition(
            src=HUMAN_MODE_DISTRACTED,
            event="callback_stay",
            dst=HUMAN_MODE_DISTRACTED,
            effects=lambda c: {
                "set_callback_submode": "stay",
                "callback_stay_steps": int(c.get("callback_stay_steps", 0)),
            },
        ),
        Transition(
            src=HUMAN_MODE_DISTRACTED,
            event="callback_ignore",
            dst=HUMAN_MODE_DISTRACTED,
            effects=lambda c: {"set_callback_submode": "ignore"},
        ),
        Transition(src=HUMAN_MODE_DISTRACTED, event="distracted_timeout", dst=HUMAN_MODE_FOLLOWING),
        Transition(src=HUMAN_MODE_FOLLOWING, event="variant_distracted", dst=HUMAN_MODE_DISTRACTED),
        Transition(src=HUMAN_MODE_FOLLOWING, event="variant_impatient", dst=HUMAN_MODE_IMPATIENT),
        Transition(src=HUMAN_MODE_IMPATIENT, event="impatient_timeout", dst=HUMAN_MODE_FOLLOWING),
        Transition(src=HUMAN_MODE_WANDERING, event="follow_enabled", dst=HUMAN_MODE_FOLLOWING),
        Transition(src=HUMAN_MODE_FOLLOWING, event="follow_disabled", dst=HUMAN_MODE_WANDERING),
        Transition(src=HUMAN_MODE_LISTENING, event="follow_enabled", dst=HUMAN_MODE_FOLLOWING),
        Transition(src=HUMAN_MODE_LISTENING, event="follow_disabled", dst=HUMAN_MODE_WANDERING),
        Transition(src=HUMAN_MODE_ATTACK, event="follow_enabled", dst=HUMAN_MODE_FOLLOWING),
        Transition(src=HUMAN_MODE_ATTACK, event="follow_disabled", dst=HUMAN_MODE_WANDERING),
    ]


def collect_events(ctx: dict) -> list[str]:
    events: list[str] = []

    if ctx.get("robot_listen_mode", False):
        events.append("robot_listen_mode")

    if ctx.get("follow_enabled", False):
        events.append("follow_enabled")
    else:
        events.append("follow_disabled")

    variant_trigger = ctx.get("variant_trigger", None)
    if variant_trigger == "distracted":
        events.append("variant_distracted")
    elif variant_trigger == "impatient":
        events.append("variant_impatient")

    callback_response = ctx.get("callback_response", None)
    if callback_response in ("rejoin", "stay", "ignore"):
        events.append(f"callback_{callback_response}")

    fear_response = ctx.get("fear_response", None)
    if fear_response in ("move_back", "stay", "continue_hit"):
        events.append(f"fear_{fear_response}")

    if ctx.get("distracted_timeout", False):
        events.append("distracted_timeout")
    if ctx.get("impatient_timeout", False):
        events.append("impatient_timeout")
    if ctx.get("attack_hit", False):
        events.append("attack_hit")

    if ctx.get("fear_completed", False) and ctx.get("fear_stay_active", False):
        events.append("fear_completed_stay_active")

    return events


def decide_mode(current_mode: str, ctx: dict, table: Optional[Sequence[Transition]] = None) -> dict:
    if table is None:
        table = build_transition_table()

    events = collect_events(ctx)
    next_mode, effects = apply_transitions(
        current_state=str(current_mode),
        events=events,
        ctx=ctx,
        table=table,
    )

    return {
        "current_mode": str(current_mode),
        "events": list(events),
        "next_mode": str(next_mode),
        "effects": dict(effects),
    }
