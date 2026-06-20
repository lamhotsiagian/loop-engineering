"""Tool abstraction and registry (Chapter 6).

A tool is a typed, named, side-effecting capability the agent can invoke. The
registry exposes JSON schemas to the model and dispatches validated calls.
Every result is captured as an ``Observation`` -- the loop's job is to feed
that observation back into the next reasoning step, never to discard it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """The normalised outcome of one tool invocation."""

    ok: bool
    content: str
    # Structured payload for verifiers / downstream tools (e.g. test counts).
    data: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        status = "OK" if self.ok else "FAILED"
        return f"[{status}] {self.content}"


@dataclass
class Tool:
    """A callable capability with a JSON schema for the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
    # Tools that mutate the world should declare it so guards can reason about
    # idempotency and approval (Chapters 3 & 6).
    side_effecting: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds the tools available to a loop and dispatches calls safely."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def is_side_effecting(self, name: str) -> bool:
        return self._tools[name].side_effecting if name in self._tools else False

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(ok=False, content=f"unknown tool '{name}'")
        try:
            return self._tools[name].handler(**arguments)
        except TypeError as exc:
            # A schema/argument mismatch is an observation the model can fix,
            # not a crash that should kill the loop.
            return ToolResult(ok=False, content=f"bad arguments for '{name}': {exc}")
        except Exception as exc:  # noqa: BLE001 - tools are untrusted code
            return ToolResult(ok=False, content=f"tool '{name}' raised: {exc}")
