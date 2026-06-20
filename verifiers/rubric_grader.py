"""Rubric-based grading for the verification loop (Chapter 8).

A single pass/fail signal is poor feedback. A rubric decomposes success into
named dimensions -- correctness, scope, style -- each scored independently, so
the next agent attempt sees *which* dimension fell short rather than a bare
"failed". Dimensions can be deterministic (a code check, preferred wherever the
criterion is expressible as code) or agentic (an LLM-as-judge call for criteria
that genuinely need judgment).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.state import LoopState
from ..llm.base import LLMClient, Message
from .base import Verdict, Verifier


@dataclass
class DimensionResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RubricResult:
    dimensions: list[DimensionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(d.passed for d in self.dimensions)

    @property
    def score(self) -> float:
        if not self.dimensions:
            return 1.0
        return sum(d.passed for d in self.dimensions) / len(self.dimensions)

    def feedback(self) -> str:
        """Actionable, per-dimension feedback for the next attempt."""
        failed = [d for d in self.dimensions if not d.passed]
        if not failed:
            return "All rubric dimensions passed."
        return "Rubric failures:\n" + "\n".join(f"  - {d.name}: {d.detail}" for d in failed)


# A deterministic dimension is just a predicate over the final output text.
DeterministicCheck = Callable[[str], DimensionResult]


def deterministic(name: str, predicate: Callable[[str], bool], on_fail: str) -> DeterministicCheck:
    def check(output: str) -> DimensionResult:
        return DimensionResult(name, passed=predicate(output), detail="" if predicate(output) else on_fail)
    return check


class RubricGrader(Verifier):
    """A maker/checker grader: deterministic dimensions plus optional agentic ones.

    Plugs into ``AgentLoop`` as a ``Verifier`` -- it grades the agent's final
    assistant message and returns a ``Verdict`` whose reason is the rubric
    feedback, so a failure becomes actionable context for the next attempt.
    """

    name = "rubric"

    def __init__(
        self,
        deterministic_checks: list[DeterministicCheck] | None = None,
        judge_llm: LLMClient | None = None,
        judge_criteria: dict[str, str] | None = None,
    ) -> None:
        self._checks = deterministic_checks or []
        self._judge = judge_llm
        self._criteria = judge_criteria or {}

    def grade(self, output: str) -> RubricResult:
        result = RubricResult()
        # Deterministic first -- cheap, unambiguous.
        for check in self._checks:
            result.dimensions.append(check(output))
        # Agentic dimensions only for criteria that need judgment.
        for name, criterion in self._criteria.items():
            result.dimensions.append(self._judge_dimension(name, criterion, output))
        return result

    def _judge_dimension(self, name: str, criterion: str, output: str) -> DimensionResult:
        if self._judge is None:
            return DimensionResult(name, passed=True, detail="no judge configured")
        messages = [
            Message("system", "You are a strict grader. Reply 'PASS' or 'FAIL: <reason>'."),
            Message("user", f"CRITERION: {criterion}\n\nOUTPUT:\n{output}"),
        ]
        verdict = self._judge.complete(messages).text.strip()
        passed = verdict.upper().startswith("PASS")
        detail = "" if passed else (verdict.split(":", 1)[1].strip() if ":" in verdict else verdict)
        return DimensionResult(name, passed=passed, detail=detail)

    def check(self, state: LoopState) -> Verdict:
        output = next((m.content for m in reversed(state.messages) if m.role == "assistant"), "")
        result = self.grade(output)
        return Verdict(passed=result.passed, reason=result.feedback())
