"""Safety controls (Chapter 14).

Two controls that interactive loops rarely need but unattended ones always do:
an external **kill switch** (a human halts the loop regardless of its internal
logic) and a **step guard** that re-evaluates permission against the *current*
state at every iteration -- because a guardrail checked once at the start has
gone stale by iteration five.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class KillSwitchTripped(Exception):
    """Raised when an external halt has been requested."""


@dataclass
class KillSwitch:
    """An external halt independent of the loop's own logic.

    A circuit breaker (Chapter 7) is the loop reacting to its *own* observed
    failures; a kill switch is a human stopping it for a reason the loop's logic
    never detected. Checked at the top of every iteration.
    """

    _tripped: bool = False
    reason: str = ""

    def trip(self, reason: str = "halted by operator") -> None:
        self._tripped = True
        self.reason = reason

    @property
    def tripped(self) -> bool:
        return self._tripped

    def ensure_live(self) -> None:
        if self._tripped:
            raise KillSwitchTripped(self.reason)


@dataclass
class StepGuard:
    """Re-evaluates a permission decision against the current state every step.

    ``sensitive`` actions are blocked unless explicitly approved, and the check
    is parameterised by a *fresh* read of the environment each call so a stale
    initial approval cannot authorise an action the current state forbids.
    """

    sensitive: set[str] = field(default_factory=lambda: {"deploy", "delete", "transfer"})
    approved: set[str] = field(default_factory=set)

    def allow(self, action: str, *, state_is_safe: Callable[[], bool]) -> tuple[bool, str]:
        if action not in self.sensitive:
            return True, "non-sensitive"
        if action not in self.approved:
            return False, f"'{action}' requires explicit approval"
        # Re-evaluate against current state, not the state at approval time.
        if not state_is_safe():
            return False, f"'{action}' approved earlier but current state is unsafe"
        return True, "approved and state safe"
