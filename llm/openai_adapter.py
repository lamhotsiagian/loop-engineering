"""Optional real provider adapter (Chapter 2).

This is intentionally *not* imported by the package by default -- the library
runs fully on the mock. Supply an API key and install the provider SDK to use
a real model:

    pip install openai
    export OPENAI_API_KEY=sk-...

The adapter normalises the provider response into the same ``LLMResponse`` the
mock produces, so the agent loop cannot tell the difference. That symmetry is
the entire point of the ``LLMClient`` seam.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolCall


class OpenAIAdapter(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI  # imported lazily so the package has no hard dep
        except ImportError as exc:  # pragma: no cover - exercised only with the SDK
            raise ImportError(
                "OpenAIAdapter requires the `openai` package: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> LLMResponse:  # pragma: no cover
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[self._encode(m) for m in messages],
            tools=tools or None,
        )
        choice = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (choice.tool_calls or [])
        ]
        usage = resp.usage
        return LLMResponse(
            text=choice.content or "",
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )

    @staticmethod
    def _encode(m: Message) -> dict[str, Any]:  # pragma: no cover
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_call_id:
            out["tool_call_id"] = m.tool_call_id
        return out
