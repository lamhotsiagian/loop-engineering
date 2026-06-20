"""Robustness primitives (Chapter 7).

Failure is a normal loop event, not an exception. These small, testable objects
encode the standard defenses: a circuit breaker that halts a whole *category* of
action after repeated failures (broader than a per-action retry cap), and a
stuck-state detector that fires on absence of progress even when individual
actions differ.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    """Trips after ``threshold`` failures for a given action category, blocking
    further attempts of that category until a cooldown elapses.

    Differs from a retry cap: a retry cap governs one specific action; the
    breaker governs an entire class (e.g. "any database write") even when the
    specific failing call differs each time.
    """

    threshold: int = 3
    cooldown_seconds: float = 30.0
    _failures: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _opened_at: dict[str, float] = field(default_factory=dict)

    def record_failure(self, category: str) -> None:
        self._failures[category] += 1
        if self._failures[category] >= self.threshold:
            self._opened_at[category] = time.time()

    def record_success(self, category: str) -> None:
        self._failures[category] = 0
        self._opened_at.pop(category, None)

    def is_open(self, category: str) -> bool:
        """True if the breaker is tripped (the category should not be attempted)."""
        opened = self._opened_at.get(category)
        if opened is None:
            return False
        if time.time() - opened >= self.cooldown_seconds:
            # Cooldown elapsed -> half-open: allow one trial attempt.
            self._failures[category] = self.threshold - 1
            self._opened_at.pop(category, None)
            return False
        return True


@dataclass
class StuckDetector:
    """Fires when a monotonic progress metric has not improved for ``patience``
    consecutive observations -- catches non-converging activity that exact
    repetition detection (Chapter 4) misses because the actions differ."""

    patience: int = 3
    _best: float | None = None
    _stale: int = 0

    def observe(self, progress: float) -> bool:
        """Feed the current progress metric; return True if the loop is stuck."""
        if self._best is None or progress > self._best:
            self._best = progress
            self._stale = 0
        else:
            self._stale += 1
        return self._stale >= self.patience
