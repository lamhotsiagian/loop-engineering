"""Run-level metrics from a trace (Chapter 13).

Observability is only useful if you compute the right metrics off it. The most
commonly missed one is the *runaway* rate: a budget-exhausted run looks almost
identical to a deliberately bounded one in aggregate success stats, so it must
be tracked as its own explicit signal rather than hidden inside "not succeeded".
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.agent_loop import LoopResult
from ..core.state import LoopPhase


@dataclass
class RunMetrics:
    succeeded: bool
    budget_exhausted: bool      # the runaway signal -- tracked separately
    blocked: bool
    iterations: int
    tool_calls: int
    tokens: int
    tool_failures: int          # step-level inefficiency signal

    def as_row(self) -> dict:
        return self.__dict__.copy()


def run_metrics(result: LoopResult) -> RunMetrics:
    tracer = result.tracer
    tool_failures = sum(
        1 for e in tracer.events if e.kind == "tool" and e.detail.get("ok") is False
    )
    # "failed" terminal whose transition reason mentions budget == a runaway,
    # not a clean stop. The reason lives in the transition log, not the message.
    budget_exhausted = result.phase is LoopPhase.FAILED and any(
        "budget" in t.reason.lower()
        for t in result.state.transitions
        if t.to_phase is LoopPhase.FAILED
    )
    return RunMetrics(
        succeeded=result.succeeded,
        budget_exhausted=budget_exhausted,
        blocked=result.phase is LoopPhase.BLOCKED,
        iterations=result.state.iteration,
        tool_calls=result.state.tool_calls_made,
        tokens=result.state.tokens_used,
        tool_failures=tool_failures,
    )


def fleet_summary(results: list[LoopResult]) -> dict[str, float]:
    """Aggregate metrics across many runs -- the input the hill-climbing loop
    (Chapter 12) and A/B tests consume."""
    n = len(results) or 1
    rows = [run_metrics(r) for r in results]
    return {
        "n": len(results),
        "success_rate": sum(r.succeeded for r in rows) / n,
        "runaway_rate": sum(r.budget_exhausted for r in rows) / n,   # tracked separately!
        "blocked_rate": sum(r.blocked for r in rows) / n,
        "avg_tool_calls": sum(r.tool_calls for r in rows) / n,
        "avg_tokens": sum(r.tokens for r in rows) / n,
    }
