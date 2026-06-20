"""Runnable demos: ``python -m loop_engineering_lab demo``.

Every demo runs on the deterministic MockLLM so it works with no API key. Pass
``--real`` to route through the OpenAI adapter instead (requires the SDK and an
API key).
"""
from __future__ import annotations

import argparse
import sys

from .core.agent_loop import AgentLoop
from .core.control import Budget, TerminationPolicy, scope_guard
from .core.state import LoopPhase
from .llm.base import LLMResponse, Message
from .llm.mock import MockLLM, call_tool, say
from .observability.tracing import Tracer
from .tools.base import ToolRegistry
from .tools.builtins import make_python_exec
from .verifiers.base import KeywordVerifier


def _react_policy(messages: list[Message]) -> LLMResponse:
    """A tiny scripted 'agent': run code once, read the result, then finish.

    This simulates the ReAct cycle (reason -> act -> observe -> reason) without
    a real model, so the demo is fully deterministic and offline.
    """
    last = messages[-1]
    if last.role == "tool":
        # We have observed the tool output -- now produce the verified answer.
        return say("The sum is 42. tests-pass", tokens=12)
    # First turn: decide to act.
    return call_tool("python_exec", code="print(sum(range(9)) + 6)")


def demo_react(real: bool = False) -> int:
    """A minimal verified ReAct loop end to end."""
    llm = MockLLM(policy=_react_policy)

    tools = ToolRegistry()
    tools.register(make_python_exec())

    policy = TerminationPolicy(
        budget=Budget(max_iterations=6, max_tool_calls=4),
        success_when=lambda s: False,  # success is decided by the verifier
        stopping_rules=(scope_guard(allowed_paths=("src/",)),),
    )
    verifier = KeywordVerifier("tests-pass")
    tracer = Tracer(echo=True)

    loop = AgentLoop(llm, tools, policy, verifiers=[verifier], tracer=tracer)
    result = loop.run("Compute a number and confirm with a passing check.")

    print("\n=== RESULT ===")
    print(result.report())
    print("final:", result.final_message)
    print("phase:", result.phase.value)
    return 0 if result.phase is LoopPhase.SUCCEEDED else 1


def demo_budget() -> int:
    """Show a loop that never finishes being stopped by its budget."""
    # The model always asks to run another tool; it never finishes.
    forever = MockLLM(policy=lambda msgs: call_tool("python_exec", code="pass"))
    tools = ToolRegistry()
    tools.register(make_python_exec())
    policy = TerminationPolicy(budget=Budget(max_iterations=3, max_tool_calls=3))
    loop = AgentLoop(forever, tools, policy, tracer=Tracer())
    result = loop.run("Loop forever (the budget should stop me).")
    print(result.report())
    print("phase:", result.phase.value, "->", "budget enforced correctly"
          if result.phase is LoopPhase.FAILED else "BUDGET NOT ENFORCED")
    return 0 if result.phase is LoopPhase.FAILED else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loop_engineering_lab")
    parser.add_argument("command", choices=["demo", "budget"], help="which demo to run")
    parser.add_argument("--real", action="store_true", help="use a real LLM provider")
    args = parser.parse_args(argv)

    if args.command == "demo":
        return demo_react(real=args.real)
    if args.command == "budget":
        return demo_budget()
    return 2


if __name__ == "__main__":
    sys.exit(main())
