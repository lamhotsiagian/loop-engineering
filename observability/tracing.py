"""Structured tracing (Chapter 13).

A loop you cannot replay is a loop you cannot debug. The tracer emits one
structured event per significant moment (phase change, model call, tool call,
verification) so a run can be reconstructed offline and turned into an eval
fixture.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO


@dataclass
class TraceEvent:
    iteration: int
    kind: str           # phase | model | tool | verify | terminal
    detail: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


class Tracer:
    """Collects events in memory and optionally streams them as JSON lines."""

    def __init__(self, stream: TextIO | None = None, echo: bool = False) -> None:
        self.events: list[TraceEvent] = []
        self._stream = stream if stream is not None else sys.stdout
        self._echo = echo

    def emit(self, iteration: int, kind: str, **detail: Any) -> None:
        event = TraceEvent(iteration=iteration, kind=kind, detail=detail)
        self.events.append(event)
        if self._echo:
            self._stream.write(json.dumps(asdict(event), default=str) + "\n")
            self._stream.flush()

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return counts
