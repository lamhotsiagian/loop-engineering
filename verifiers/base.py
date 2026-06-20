"""Verifiers (Chapter 8).

A verifier is the loop's source of ground truth. The model proposes; the
verifier disposes. Crucially, a verifier returns a *reason* on failure so that
failure becomes new context fed back into the next reasoning step rather than a
silent rejection.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.state import LoopState


@dataclass
class Verdict:
    passed: bool
    reason: str = ""


class Verifier:
    """Protocol-style base: name + a ``check`` returning a Verdict."""

    name: str = "verifier"

    def check(self, state: LoopState) -> Verdict:  # pragma: no cover - overridden
        raise NotImplementedError


class PredicateVerifier(Verifier):
    """Wrap a plain predicate over the final assistant message."""

    def __init__(self, name: str, predicate: Callable[[str], bool], reason: str) -> None:
        self.name = name
        self._predicate = predicate
        self._reason = reason

    def check(self, state: LoopState) -> Verdict:
        final = next(
            (m.content for m in reversed(state.messages) if m.role == "assistant"),
            "",
        )
        if self._predicate(final):
            return Verdict(passed=True)
        return Verdict(passed=False, reason=self._reason)


class KeywordVerifier(PredicateVerifier):
    """Succeeds only when the final answer contains a required marker.

    A stand-in for "tests pass" in environments where we cannot run a real
    suite; the production analogue is ``TestsPassVerifier`` keyed off the
    structured ``run_tests`` result.
    """

    def __init__(self, keyword: str) -> None:
        super().__init__(
            name=f"contains:{keyword}",
            predicate=lambda text: keyword.lower() in text.lower(),
            reason=f"final answer must contain '{keyword}'",
        )
