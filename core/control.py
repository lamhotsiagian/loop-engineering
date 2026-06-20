"""Loop control structures (Chapter 3).

Budgets, termination conditions and stopping rules are first-class objects, not
``if`` statements buried in the loop body. Separating them makes the policy
auditable and testable in isolation: you can unit-test "does this budget stop
the loop at 5 tool calls?" without ever invoking a model.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .state import LoopPhase, LoopState


@dataclass
class Budget:
    """A hard resource ceiling, independent of whether the loop thinks it is
    making progress. Exhausting a budget is a *signal*, not merely a stop
    (Chapter 3)."""

    max_iterations: int = 12
    max_tokens: int = 100_000
    max_tool_calls: int = 40
    max_seconds: float = 300.0
    max_cost_usd: float = 5.0

    def exceeded(self, state: LoopState) -> str | None:
        """Return a human-readable reason if any ceiling is breached, else None."""
        if state.iteration >= self.max_iterations:
            return f"iteration budget exhausted ({self.max_iterations})"
        if state.tokens_used >= self.max_tokens:
            return f"token budget exhausted ({self.max_tokens})"
        if state.tool_calls_made >= self.max_tool_calls:
            return f"tool-call budget exhausted ({self.max_tool_calls})"
        if state.elapsed_seconds >= self.max_seconds:
            return f"time budget exhausted ({self.max_seconds}s)"
        return None


# A stopping rule inspects the proposed next action and may veto it. Returning a
# string blocks the action with that reason; returning None allows it.
StoppingRule = Callable[[LoopState, str, dict], str | None]


def destructive_action_guard(dangerous: set[str] | None = None) -> StoppingRule:
    """Block irreversible tools unless a human has approved them in scratchpad."""
    dangerous = dangerous or {"drop_table", "force_push", "rotate_credentials", "delete_bucket"}

    def rule(state: LoopState, tool_name: str, args: dict) -> str | None:
        if tool_name in dangerous and not state.scratchpad.get("human_approved", set()):
            return f"destructive action '{tool_name}' requires human approval"
        approved = state.scratchpad.get("human_approved", set())
        if tool_name in dangerous and tool_name not in approved:
            return f"destructive action '{tool_name}' not in approved set"
        return None

    return rule


def scope_guard(allowed_paths: tuple[str, ...]) -> StoppingRule:
    """Block file writes outside the task's declared boundary."""

    def rule(state: LoopState, tool_name: str, args: dict) -> str | None:
        if tool_name in {"write_file", "edit_file"}:
            path = str(args.get("path", ""))
            if not path.startswith(allowed_paths):
                return f"path '{path}' is outside the task scope {allowed_paths}"
        return None

    return rule


@dataclass
class TerminationPolicy:
    """Bundles a budget with a success predicate and stopping rules.

    The loop consults this object -- it never hard-codes "done" itself.
    """

    budget: Budget
    success_when: Callable[[LoopState], bool] = lambda s: False
    stopping_rules: tuple[StoppingRule, ...] = ()
    no_progress_limit: int = 3  # identical-error iterations before giving up

    def check_terminal(self, state: LoopState) -> tuple[LoopPhase, str] | None:
        """Decide whether the loop must stop *now*, before the next action."""
        if self.success_when(state):
            return LoopPhase.SUCCEEDED, "success condition met"
        reason = self.budget.exceeded(state)
        if reason:
            return LoopPhase.FAILED, reason
        stalls = state.scratchpad.get("consecutive_no_progress", 0)
        if stalls >= self.no_progress_limit:
            return LoopPhase.BLOCKED, f"no progress for {stalls} iterations"
        return None

    def veto_action(self, state: LoopState, tool_name: str, args: dict) -> str | None:
        """Run every stopping rule; the first veto wins."""
        for rule in self.stopping_rules:
            blocked = rule(state, tool_name, args)
            if blocked:
                return blocked
        return None
