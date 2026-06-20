"""LLM abstraction layer (Chapter 2).

The loop never talks to a concrete provider directly. It depends only on the
``LLMClient`` protocol, so the same orchestration code runs against a
deterministic mock in tests, a local model in development, and a frontier API
in production. This is the seam that makes the loop testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    """A single chat message. ``role`` is one of system|user|assistant|tool."""

    role: str
    content: str
    # For tool result messages we keep the originating call id so the model can
    # correlate a result with the request that produced it.
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    """A structured request from the model to invoke a named tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalised response. Either ``text`` (final answer) or ``tool_calls``."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Token accounting is surfaced so budgets (Chapter 3) can be enforced.
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)

    def as_json(self) -> str:
        return json.dumps(
            {
                "text": self.text,
                "tool_calls": [tc.__dict__ for tc in self.tool_calls],
                "tokens": self.total_tokens,
            },
            indent=2,
        )


@runtime_checkable
class LLMClient(Protocol):
    """The only contract the agent loop depends on.

    Implementations must be deterministic given identical inputs *if* they want
    reproducible tests; production adapters obviously are not, which is exactly
    why the loop is built around verification rather than trust.
    """

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> LLMResponse:
        ...
