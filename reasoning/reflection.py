"""Reflection / self-critique (Chapter 5).

A reflection loop adds an explicit generate-then-critique cycle. The critical
design point: the critique must be framed *differently* from the generation and
ideally run as a separate call -- asking the same call 'are you sure?' yields
confirmation, not correction. Framing it as 'find the flaw' yields measurably
better self-correction.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..llm.base import LLMClient, Message

CRITIQUE_FRAMING = (
    "You are a strict reviewer. Find the single most important flaw in the draft "
    "below relative to the stated goal. If it is correct and complete, reply "
    "exactly 'APPROVED'. Otherwise reply 'REVISE: <the flaw>'."
)


@dataclass
class Critique:
    approved: bool
    issue: str = ""


class Reflector:
    """Runs a draft through a separate critique call until approved or budget.

    ``accept`` lets callers short-circuit with a deterministic check (e.g. a
    real verifier) instead of trusting the model's self-judgment.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_rounds: int = 2,
        accept: Callable[[str], bool] | None = None,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        self.accept = accept

    def critique(self, goal: str, draft: str) -> Critique:
        messages = [
            Message("system", CRITIQUE_FRAMING),
            Message("user", f"GOAL:\n{goal}\n\nDRAFT:\n{draft}"),
        ]
        verdict = self.llm.complete(messages).text.strip()
        if verdict.upper().startswith("APPROVED"):
            return Critique(approved=True)
        issue = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
        return Critique(approved=False, issue=issue)

    def refine(self, goal: str, draft: str, revise: Callable[[str, str], str]) -> str:
        """Iterate generate->critique->revise up to ``max_rounds`` times."""
        current = draft
        for _ in range(self.max_rounds):
            if self.accept is not None and self.accept(current):
                return current
            verdict = self.critique(goal, current)
            if verdict.approved:
                return current
            current = revise(current, verdict.issue)
        return current
