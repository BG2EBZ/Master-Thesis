from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FOLLOW_PHASE_PRE_LISTEN_ENGAGE = "pre_listen_engage"
FOLLOW_PHASE_TRANSIT = "transit_follow"

LISTEN_PHASE_IDLE = "idle"
LISTEN_PHASE_INTRO = "intro"
LISTEN_PHASE_WAIT = "wait"
LISTEN_PHASE_PAUSED = "paused"

LISTEN_QUESTION_TIMING_MID_RANDOM = "mid_random"
LISTEN_QUESTION_TIMING_POST_WAIT = "post_wait"
LISTEN_QUESTION_PHASE_NONE = "none"
LISTEN_QUESTION_PHASE_TURN_TO_HUMAN = "turn_to_human"
LISTEN_QUESTION_PHASE_ANSWER = "answer"
LISTEN_QUESTION_PHASE_TURN_BACK = "turn_back"
LISTEN_QUESTION_COMPLETION_RESUME_WAIT = "resume_wait"
LISTEN_QUESTION_COMPLETION_FINISH_WAIT = "finish_wait"

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
    wait_target_steps: int = 0
    distance_shorten_triggered_indices: set[int] = field(default_factory=set)
    paused_phase: str = LISTEN_PHASE_IDLE
    paused_counter: int = 0
    paused_is_final: bool = False
    session_has_question: bool = False
    question_timing_mode: Optional[str] = None
    question_trigger_step: Optional[int] = None
    question_fired: bool = False
    question_human_idx: Optional[int] = None
    question_phase: str = LISTEN_QUESTION_PHASE_NONE
    question_ask_steps_remaining: int = 0
    question_answer_steps_remaining: int = 0
    question_return_yaw: Optional[float] = None
    question_completion_mode: Optional[str] = None

    def _clear_paused_state(self) -> None:
        self.paused_phase = LISTEN_PHASE_IDLE
        self.paused_counter = 0
        self.paused_is_final = False

    def _clear_active_question_fields(self) -> None:
        self.question_human_idx = None
        self.question_phase = LISTEN_QUESTION_PHASE_NONE
        self.question_ask_steps_remaining = 0
        self.question_answer_steps_remaining = 0
        self.question_return_yaw = None
        self.question_completion_mode = None

    def _reset_question_state(self) -> None:
        self.session_has_question = False
        self.question_timing_mode = None
        self.question_trigger_step = None
        self.question_fired = False
        self._clear_active_question_fields()

    def clear_active_question(self) -> None:
        self._clear_active_question_fields()

    def reset_wait_runtime(self) -> None:
        self.wait_target_steps = 0
        self.distance_shorten_triggered_indices.clear()

    def initialize_wait_runtime(self, default_wait_steps: int) -> int:
        self.reset_wait_runtime()
        self.wait_target_steps = max(1, int(default_wait_steps))
        return self.wait_target_steps

    def ensure_wait_target_steps(self, default_wait_steps: int) -> int:
        if self.wait_target_steps <= 0:
            self.wait_target_steps = max(1, int(default_wait_steps))
        return int(self.wait_target_steps)

    def _enter_phase(self, phase: str, *, is_final: bool) -> None:
        self.phase = phase
        self.counter = 0
        self.is_final = bool(is_final)
        self.reset_wait_runtime()
        self._reset_question_state()

    def reset(self) -> None:
        self._enter_phase(LISTEN_PHASE_IDLE, is_final=False)
        self._clear_paused_state()

    def enter_intro(self, is_final: bool) -> None:
        self._enter_phase(LISTEN_PHASE_INTRO, is_final=is_final)

    def enter_wait(self, is_final: bool) -> None:
        self._enter_phase(LISTEN_PHASE_WAIT, is_final=is_final)

    def enter_idle(self) -> None:
        self._enter_phase(LISTEN_PHASE_IDLE, is_final=False)

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
        self._clear_paused_state()

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

    @property
    def question_active(self) -> bool:
        return self.question_phase != LISTEN_QUESTION_PHASE_NONE


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
    question_started: bool = False
    question_completed: bool = False
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
            "question_started": bool(self.question_started),
            "question_completed": bool(self.question_completed),
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
