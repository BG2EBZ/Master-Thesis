from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence


ROBOT_STATE_WAIT = "wait"
ROBOT_STATE_LISTEN = "listen"


@dataclass(frozen=True)
class Transition:
    src: str
    event: str
    dst: str
    guard: Optional[Callable[[dict], bool]] = None
    effects: Optional[Callable[[dict], dict]] = None


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
        Transition(
            src="move",
            event="callback_request",
            dst="callback",
            guard=lambda c: (not c.get("listen_mode", False)) and (not c.get("callback_active", False)),
            effects=lambda c: {"start_callback": True},
        ),
        Transition(
            src="callback",
            event="callback_done",
            dst="move",
            effects=lambda c: {"finish_callback": True},
        ),
        Transition(src="move", event="reached_display", dst="stop"),
        Transition(
            src="stop",
            event="turn_done",
            dst="stop",
            guard=lambda c: not c.get("listen_mode", False),
            effects=lambda c: {"enter_listen": True},
        ),
        Transition(
            src=ROBOT_STATE_WAIT,
            event="threat_too_close",
            dst="move_back",
            effects=lambda c: {"set_move_back": True},
        ),
        Transition(
            src=ROBOT_STATE_LISTEN,
            event="threat_too_close",
            dst="move_back",
            effects=lambda c: {"set_move_back": True},
        ),
        Transition(
            src="move_back",
            event="threat_too_close",
            dst="move_back",
            effects=lambda c: {"set_move_back": True},
        ),
        Transition(
            src="move_back",
            event="threat_exists",
            dst="move_back",
            effects=lambda c: {"hold_move_back": True},
        ),
        Transition(
            src=ROBOT_STATE_WAIT,
            event="threat_exists",
            dst="stop",
            effects=lambda c: {"clear_move_back": True},
        ),
        Transition(
            src=ROBOT_STATE_LISTEN,
            event="threat_exists",
            dst="stop",
            effects=lambda c: {"clear_move_back": True},
        ),
        Transition(
            src="move_back",
            event="threat_cleared",
            dst="stop",
            effects=lambda c: {"clear_move_back": True},
        ),
        Transition(
            src=ROBOT_STATE_WAIT,
            event="threat_cleared",
            dst="stop",
            effects=lambda c: {"clear_move_back": True},
        ),
        Transition(
            src=ROBOT_STATE_LISTEN,
            event="threat_cleared",
            dst="stop",
            effects=lambda c: {"clear_move_back": True},
        ),
    ]


def collect_events(ctx: dict) -> list[str]:
    events: list[str] = []

    if ctx.get("callback_request_exists", False):
        events.append("callback_request")
    if ctx.get("callback_done", False):
        events.append("callback_done")
    if ctx.get("reached_display", False):
        events.append("reached_display")
    if ctx.get("turn_done", False):
        events.append("turn_done")
    if ctx.get("listen_wait_done", False):
        events.append("listen_wait_done")
    if ctx.get("final_listen_ready", False):
        events.append("final_listen_ready")

    threat_exists = bool(ctx.get("threat_exists", False))
    threat_dist = ctx.get("threat_dist", None)
    safe_dist = float(ctx.get("move_back_safe_distance", 0.0))

    if threat_exists:
        if threat_dist is not None and float(threat_dist) < safe_dist:
            events.append("threat_too_close")
        else:
            events.append("threat_exists")
    else:
        events.append("threat_cleared")

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
