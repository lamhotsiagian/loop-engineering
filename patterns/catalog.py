"""Pattern catalog as configuration, not new code (Chapter 9).

The named loop patterns are not nine different engines -- they are the *same*
``AgentLoop`` configured with a different verifier, termination policy, and tool
set. These builders make that explicit: each returns a ready-to-run loop whose
"pattern" is entirely a matter of which feedback signal it is wired to.
"""
from __future__ import annotations

from ..core.agent_loop import AgentLoop
from ..core.control import Budget, TerminationPolicy
from ..llm.base import LLMClient
from ..tools.base import ToolRegistry
from ..verifiers.base import KeywordVerifier, Verifier


def retry_loop(llm: LLMClient, tools: ToolRegistry, *, success_marker: str, max_iters: int = 5) -> AgentLoop:
    """Simplest pattern: attempt, check a pass/fail marker, retry on failure."""
    policy = TerminationPolicy(budget=Budget(max_iterations=max_iters))
    return AgentLoop(llm, tools, policy, verifiers=[KeywordVerifier(success_marker)])


def test_driven_loop(llm: LLMClient, tools: ToolRegistry, verifier: Verifier, *, max_iters: int = 8) -> AgentLoop:
    """Failure-encoded-as-test pattern: loop until the supplied verifier passes."""
    policy = TerminationPolicy(budget=Budget(max_iterations=max_iters, max_tool_calls=24))
    return AgentLoop(llm, tools, policy, verifiers=[verifier])


def plan_execute_verify_loop(
    llm: LLMClient, tools: ToolRegistry, verifier: Verifier, *, max_iters: int = 12
) -> AgentLoop:
    """Order-dependent pattern: a larger budget and a verifier gate per step."""
    policy = TerminationPolicy(
        budget=Budget(max_iterations=max_iters, max_tool_calls=40),
        no_progress_limit=3,
    )
    return AgentLoop(llm, tools, policy, verifiers=[verifier])


PATTERNS = {
    "retry": retry_loop,
    "test_driven": test_driven_loop,
    "plan_execute_verify": plan_execute_verify_loop,
}


def register_pattern(name: str, builder) -> None:
    """Capture a proven loop design as a named, reusable organizational asset
    (Chapter 16) -- register it once so the whole team reuses it instead of
    reconstructing prompts, validation commands, and stopping rules."""
    if name in PATTERNS:
        raise ValueError(f"pattern '{name}' already captured")
    PATTERNS[name] = builder
