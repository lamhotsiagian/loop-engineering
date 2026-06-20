"""Heartbeat / event-driven loop (Chapter 11).

An event-driven loop runs as a persistent background component: an event fires,
the loop runs, and findings land in a triage inbox for asynchronous human
review. Two infrastructure concerns are load-bearing the moment a loop runs
unattended: idempotent event handling (the same webhook firing twice must not
double-act) and a durable record of what was processed.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    id: str                 # stable id used for idempotent de-duplication
    kind: str               # "webhook" | "cron" | "message"
    payload: dict = field(default_factory=dict)


@dataclass
class Finding:
    event_id: str
    summary: str
    at: float = field(default_factory=time.time)


class Heartbeat:
    """Processes events exactly once and routes findings to an inbox.

    ``handler`` inspects an event and returns an optional ``Finding`` -- None
    means "nothing worth surfacing", and the run is archived silently; a Finding
    is appended to the triage inbox for a human to review on their own schedule.
    """

    def __init__(self, handler: Callable[[Event], Finding | None]) -> None:
        self._handler = handler
        self._seen: set[str] = set()      # idempotency ledger
        self.inbox: list[Finding] = []
        self.processed = 0
        self.skipped_duplicates = 0

    def deliver(self, event: Event) -> Finding | None:
        # Idempotent handling: a redelivered event is a no-op (common with
        # at-least-once webhook delivery).
        if event.id in self._seen:
            self.skipped_duplicates += 1
            return None
        self._seen.add(event.id)
        self.processed += 1
        finding = self._handler(event)
        if finding is not None:
            self.inbox.append(finding)
        return finding

    def tick(self, events: list[Event]) -> list[Finding]:
        """Process a batch (e.g. one cron wake-up's worth of events)."""
        return [f for e in events if (f := self.deliver(e)) is not None]
