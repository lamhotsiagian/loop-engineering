"""Cost accounting and difficulty-based model routing (Chapter 15).

Two of the highest-leverage cost controls: a per-iteration cost meter (so you
know which run/pattern is expensive) and a router that escalates to an
expensive model only when the loop is actually struggling -- routing by
difficulty, not by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostMeter:
    """Accumulates spend across a run from per-model token prices (USD / 1k)."""

    prices_per_1k: dict[str, float]
    total_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)

    def charge(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.prices_per_1k.get(model, 0.0)
        cost = (prompt_tokens + completion_tokens) / 1000.0 * price
        self.total_usd += cost
        self.by_model[model] = self.by_model.get(model, 0.0) + cost
        return cost


@dataclass
class ModelRouter:
    """Routes each iteration to a cheap or strong model by observed difficulty.

    Use the cheap model for routine steps; escalate to the strong model only
    when the loop is stuck or verification keeps failing. This matches model
    cost to actual difficulty instead of paying frontier prices for every step.
    """

    cheap: str
    strong: str
    escalate_after_failures: int = 2

    def route(self, *, stuck: bool = False, verifier_failures: int = 0) -> str:
        if stuck or verifier_failures >= self.escalate_after_failures:
            return self.strong
        return self.cheap
