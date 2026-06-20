"""loop_engineering_lab -- a minimal, runnable reference implementation that
accompanies *Loop Engineering: A Practitioner's Guide*.

The package is deliberately small and readable rather than a production
framework. Every module maps to one chapter of the book so a reader can open
the chapter and the corresponding source side by side.

Design goals
------------
* Runs with **no API key** by default: the LLM is a deterministic mock so the
  demos and the test-suite are reproducible in CI.
* A single, explicit agent loop modelled as a state machine (Chapter 2).
* Pluggable termination, budgets and stopping rules (Chapter 3).
* Pluggable tools, memory and verifiers so later chapters can extend the loop
  without rewriting the core.

Quick start
-----------
    python -m loop_engineering_lab demo
    pytest loop_engineering_lab/tests
"""
from __future__ import annotations

from .core.agent_loop import AgentLoop, LoopResult
from .core.state import LoopPhase, LoopState
from .core.control import Budget, TerminationPolicy
from .llm.base import LLMClient, Message, LLMResponse
from .llm.mock import MockLLM
from .tools.base import Tool, ToolRegistry, ToolResult

__version__ = "0.1.0"

__all__ = [
    "AgentLoop",
    "LoopResult",
    "LoopPhase",
    "LoopState",
    "Budget",
    "TerminationPolicy",
    "LLMClient",
    "Message",
    "LLMResponse",
    "MockLLM",
    "Tool",
    "ToolRegistry",
    "ToolResult",
]
