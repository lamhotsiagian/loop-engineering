"""Loop state as an explicit state machine (Chapter 2).

Modelling the loop as a small set of named phases with explicit transitions
makes two classes of bug impossible to hide: a phase with no exit, and a
terminal phase that is never reachable. Every transition is logged so a run can
be replayed and audited (Chapter 13).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from ..llm.base import Message


class LoopPhase(str, Enum):
    """The states an agent loop can occupy."""

    PLANNING = "planning"
    GATHER_CONTEXT = "gather_context"
    REASONING = "reasoning"
    ACTING = "acting"               # a tool call is in flight
    OBSERVING = "observing"         # processing a tool result
    VERIFYING = "verifying"
    # Terminal phases -- few and explicit.
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"             # needs human input
    FAILED = "failed"              # exhausted retries or budget

    @property
    def is_terminal(self) -> bool:
        return self in {LoopPhase.SUCCEEDED, LoopPhase.BLOCKED, LoopPhase.FAILED}


@dataclass
class Transition:
    """A single recorded state change, for tracing and replay."""

    iteration: int
    from_phase: LoopPhase
    to_phase: LoopPhase
    reason: str
    at: float = field(default_factory=time.time)


@dataclass
class LoopState:
    """The serialisable working memory of one loop run.

    Everything needed to resume the loop after a crash (Chapter 3,
    resumability) lives here and nowhere else.
    """

    goal: str
    phase: LoopPhase = LoopPhase.PLANNING
    iteration: int = 0
    messages: list[Message] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    tokens_used: int = 0
    tool_calls_made: int = 0
    started_at: float = field(default_factory=time.time)
    scratchpad: dict = field(default_factory=dict)

    def transition_to(self, phase: LoopPhase, reason: str) -> None:
        self.transitions.append(
            Transition(self.iteration, self.phase, phase, reason)
        )
        self.phase = phase

    def add_message(self, role: str, content: str, tool_call_id: str | None = None) -> None:
        self.messages.append(Message(role=role, content=content, tool_call_id=tool_call_id))

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at
