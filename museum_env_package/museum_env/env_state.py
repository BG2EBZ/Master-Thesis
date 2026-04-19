from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .human import HumanMode

FOLLOW_PHASE_PRE_LISTEN_ENGAGE = "pre_listen_engage"
FOLLOW_PHASE_TRANSIT = "transit_follow"

LISTEN_PHASE_IDLE = "idle"
LISTEN_PHASE_INTRO = "intro"
LISTEN_PHASE_WAIT = "wait"
LISTEN_PHASE_PAUSED = "paused"

POST_EXPLANATION_ROLE_WAIT = "wait"
POST_EXPLANATION_ROLE_YIELD = "yield"


def _zero_targets() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


def _zero_radii() -> np.ndarray:
    return np.zeros((0,), dtype=np.float32)


def _default_fuzzy_inputs() -> dict[str, float]:
    return {
        "following_time": 0.0,
        "hhd": 0.0,
        "hrd": 0.0,
        "density": 0.0,
    }


def _default_fuzzy_scores() -> dict[str, float]:
    return {
        "overwhelmed": 0.0,
        "distracted": 0.0,
        "impatient": 0.0,
        "engaged": 0.0,
    }


@dataclass
class ListeningState:
    phase: str = LISTEN_PHASE_IDLE
    counter: int = 0
    is_final: bool = False
    paused_phase: str = LISTEN_PHASE_IDLE
    paused_counter: int = 0
    paused_is_final: bool = False

    def reset(self) -> None:
        self.phase = LISTEN_PHASE_IDLE
        self.counter = 0
        self.is_final = False
        self.paused_phase = LISTEN_PHASE_IDLE
        self.paused_counter = 0
        self.paused_is_final = False

    def enter_intro(self, is_final: bool) -> None:
        self.phase = LISTEN_PHASE_INTRO
        self.counter = 0
        self.is_final = bool(is_final)

    def enter_wait(self, is_final: bool) -> None:
        self.phase = LISTEN_PHASE_WAIT
        self.counter = 0
        self.is_final = bool(is_final)

    def enter_idle(self) -> None:
        self.phase = LISTEN_PHASE_IDLE
        self.counter = 0
        self.is_final = False

    def pause(self) -> None:
        self.paused_phase = self.phase
        self.paused_counter = int(self.counter)
        self.paused_is_final = bool(self.is_final)
        self.phase = LISTEN_PHASE_PAUSED
        self.counter = 0
        self.is_final = False

    def resume(self) -> None:
        self.phase = self.paused_phase
        self.counter = int(self.paused_counter)
        self.is_final = bool(self.paused_is_final)
        self.paused_phase = LISTEN_PHASE_IDLE
        self.paused_counter = 0
        self.paused_is_final = False

    @property
    def active(self) -> bool:
        return self.phase in (LISTEN_PHASE_INTRO, LISTEN_PHASE_WAIT)

    @property
    def interrupted(self) -> bool:
        return self.phase == LISTEN_PHASE_PAUSED

    @property
    def controller_active(self) -> bool:
        return self.phase in (LISTEN_PHASE_INTRO, LISTEN_PHASE_WAIT, LISTEN_PHASE_PAUSED)

    @property
    def fuzzy_active(self) -> bool:
        return self.phase in (LISTEN_PHASE_INTRO, LISTEN_PHASE_WAIT)


@dataclass
class PostExplanationState:
    active: bool = False
    robot_start_xy: Optional[np.ndarray] = None
    anchor_robot_xy: Optional[np.ndarray] = None
    anchor_robot_yaw: float = 0.0
    roles: list[str] = field(default_factory=list)
    targets: np.ndarray = field(default_factory=_zero_targets)
    listen_radii: np.ndarray = field(default_factory=_zero_radii)

    def reset(self) -> None:
        self.active = False
        self.robot_start_xy = None
        self.anchor_robot_xy = None
        self.anchor_robot_yaw = 0.0
        self.roles.clear()
        self.targets = _zero_targets()
        self.listen_radii = _zero_radii()


@dataclass
class CallbackRequest:
    target_idx: int
    target_xy: np.ndarray
    cue_steps: int
    success_mode: str = HumanMode.FOLLOWING
    interrupts_listening: bool = False


@dataclass
class CallbackState:
    triggered_for_distracted: list[bool] = field(default_factory=list)
    active_target_idx: Optional[int] = None
    last_response: Optional[str] = None
    last_response_target_idx: Optional[int] = None
    pending_request: Optional[CallbackRequest] = None
    success_mode: str = HumanMode.FOLLOWING

    def reset(self, n_humans: int) -> None:
        self.triggered_for_distracted = [False] * int(n_humans)
        self.active_target_idx = None
        self.last_response = None
        self.last_response_target_idx = None
        self.pending_request = None
        self.success_mode = HumanMode.FOLLOWING


@dataclass
class ObservationSnapshot:
    nearest_human_distance: np.ndarray
    local_crowding_count_1m: np.ndarray
    human_robot_distance: np.ndarray
    nearest_human_distance_mean_1s: np.ndarray
    human_robot_distance_mean_1s: np.ndarray


@dataclass
class FuzzyDebugState:
    context: str = "none"
    inputs: dict[str, float] = field(default_factory=_default_fuzzy_inputs)
    scores: dict[str, float] = field(default_factory=_default_fuzzy_scores)
    dominant_state: Optional[str] = None
    refresh_counter: int = -1


def build_fuzzy_debug_states(n_humans: int) -> list[FuzzyDebugState]:
    return [FuzzyDebugState() for _ in range(int(n_humans))]


@dataclass
class RuntimeCache:
    observations: Optional[ObservationSnapshot] = None
    sample_age_steps: int = 0
    refresh_counter: int = 0

    def reset(self) -> None:
        self.observations = None
        self.sample_age_steps = 0
        self.refresh_counter = 0


@dataclass
class StepEvents:
    entered_listen: bool = False
    started_listen_wait: bool = False
    completed_listen_wait: bool = False
    final_listen_ready: bool = False
    callback_triggered: bool = False
    callback_completed: bool = False
    callback_success: bool = False
    happy_triggered: bool = False
    happy_completed: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "entered_listen": bool(self.entered_listen),
            "started_listen_wait": bool(self.started_listen_wait),
            "completed_listen_wait": bool(self.completed_listen_wait),
            "final_listen_ready": bool(self.final_listen_ready),
            "callback_triggered": bool(self.callback_triggered),
            "callback_completed": bool(self.callback_completed),
            "callback_success": bool(self.callback_success),
            "happy_triggered": bool(self.happy_triggered),
            "happy_completed": bool(self.happy_completed),
        }


@dataclass
class WorldFrame:
    robot_pose: tuple[float, float, float]
    robot_xy: np.ndarray
    human_xyz: np.ndarray
    human_xy: np.ndarray
    human_yaw: np.ndarray
    pairwise_distances: np.ndarray
    repulsion_vectors: np.ndarray
    observations: ObservationSnapshot
