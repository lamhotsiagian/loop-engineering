"""Scratchpad with attempt-tracking and rolling-summary compaction (Chapter 4).

The raw message history is a fragile memory: it is bounded, costly, and its
signal degrades as it grows. The ``Scratchpad`` is the structured working
memory that sits beside the transcript -- it records the plan, every attempt
and its outcome, and the still-open items, and it can compact older attempts
into a rolling summary so the context stays small without losing the substance
of what was tried.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Attempt:
    """One recorded action and what came of it."""

    action: str
    outcome: str           # "ok" | "failed" | "partial"
    detail: str = ""       # e.g. the error class, or what partially worked

    def signature(self) -> str:
        """A stable key used to detect 'we already tried exactly this'."""
        return f"{self.action}|{self.detail}".strip().lower()


@dataclass
class Scratchpad:
    """Structured, compactable working memory for one loop run."""

    plan: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    open_items: list[str] = field(default_factory=list)
    summary: str = ""                 # rolling summary of compacted-away attempts
    keep_verbatim: int = 6            # how many recent attempts stay un-summarised

    # --- recording --------------------------------------------------------

    def record_attempt(self, action: str, outcome: str, detail: str = "") -> None:
        self.attempts.append(Attempt(action=action, outcome=outcome, detail=detail))

    def has_tried(self, action: str, detail: str = "") -> bool:
        """True if this exact action+detail was already attempted -- the single
        most effective guard against a loop spinning on a repeated failure."""
        sig = Attempt(action, "", detail).signature()
        return any(a.signature() == sig for a in self.attempts)

    def consecutive_failures(self) -> int:
        n = 0
        for a in reversed(self.attempts):
            if a.outcome == "failed":
                n += 1
            else:
                break
        return n

    # --- compaction -------------------------------------------------------

    def compact(self) -> None:
        """Fold all but the most recent ``keep_verbatim`` attempts into the
        rolling summary. A summary like 'fix A (failed: TypeError), fix B
        (failed: same error)' is far more useful than the full transcript."""
        if len(self.attempts) <= self.keep_verbatim:
            return
        old, self.attempts = self.attempts[:-self.keep_verbatim], self.attempts[-self.keep_verbatim:]
        folded = "; ".join(
            f"{a.action} ({a.outcome}{': ' + a.detail if a.detail else ''})" for a in old
        )
        self.summary = (self.summary + " " + folded).strip() if self.summary else folded

    # --- context rendering ------------------------------------------------

    def as_context(self) -> str:
        """Render the compact working memory the model actually reasons over."""
        lines: list[str] = []
        if self.plan:
            lines.append("Plan: " + " -> ".join(self.plan))
        if self.summary:
            lines.append("Earlier attempts (summarised): " + self.summary)
        if self.attempts:
            lines.append("Recent attempts:")
            lines += [
                f"  - {a.action}: {a.outcome}{' (' + a.detail + ')' if a.detail else ''}"
                for a in self.attempts
            ]
        if self.open_items:
            lines.append("Open: " + ", ".join(self.open_items))
        return "\n".join(lines)
