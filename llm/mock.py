"""A deterministic mock LLM (Chapter 2).

The mock lets the whole library run and be tested with no network and no API
key. It is *scripted*: you give it a queue of responses, or a policy function
that inspects the conversation and decides what to "say" next. This is the same
technique you use in production to write fast, hermetic tests around a loop
whose real model is expensive and non-deterministic.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolCall


class MockLLM(LLMClient):
    """A scripted LLM client.

    Two modes:

    * **Scripted queue** -- pass ``responses=[LLMResponse, ...]`` and each call
      pops the next one. Perfect for deterministic unit tests.
    * **Policy** -- pass ``policy=fn`` where ``fn(messages) -> LLMResponse``.
      Useful for simulating an agent that reacts to tool output, e.g. "if the
      last tool result contains 'FAILED', try again, else finish".
    """

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        policy: Callable[[list[Message]], LLMResponse] | None = None,
    ) -> None:
        if responses is None and policy is None:
            raise ValueError("MockLLM needs either `responses` or `policy`")
        self._responses = list(responses or [])
        self._policy = policy
        self.call_count = 0

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> LLMResponse:
        self.call_count += 1
        if self._policy is not None:
            return self._policy(messages)
        if not self._responses:
            # A real model never runs out of things to say; a mock can, and that
            # usually means the loop iterated more than the test expected.
            return LLMResponse(text="DONE (mock exhausted)", completion_tokens=4)
        return self._responses.pop(0)


# --- Convenience builders -------------------------------------------------

def say(text: str, *, tokens: int = 16) -> LLMResponse:
    """Shorthand for a final text answer."""
    return LLMResponse(text=text, prompt_tokens=tokens, completion_tokens=tokens)


def call_tool(name: str, *, call_id: str = "c1", **arguments: Any) -> LLMResponse:
    """Shorthand for a single tool call."""
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=dict(arguments))],
        prompt_tokens=16,
        completion_tokens=8,
    )
