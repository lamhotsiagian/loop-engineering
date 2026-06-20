"""Tests for the core agent loop (Chapter 2)."""
from __future__ import annotations

from loop_engineering_lab.core.agent_loop import AgentLoop
from loop_engineering_lab.core.control import Budget, TerminationPolicy
from loop_engineering_lab.core.state import LoopPhase
from loop_engineering_lab.llm.base import LLMResponse, Message
from loop_engineering_lab.llm.mock import MockLLM, call_tool, say
from loop_engineering_lab.tools.base import ToolRegistry
from loop_engineering_lab.tools.builtins import make_python_exec
from loop_engineering_lab.verifiers.base import KeywordVerifier


def _tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(make_python_exec())
    return reg


def test_loop_finishes_on_final_answer():
    llm = MockLLM(responses=[say("all good")])
    policy = TerminationPolicy(budget=Budget(max_iterations=3))
    loop = AgentLoop(llm, _tools(), policy)
    result = loop.run("say hi")
    assert result.phase is LoopPhase.SUCCEEDED
    assert result.final_message == "all good"


def test_loop_runs_tool_then_finishes():
    def policy(messages: list[Message]) -> LLMResponse:
        if messages[-1].role == "tool":
            return say("done")
        return call_tool("python_exec", code="print(1+1)")

    llm = MockLLM(policy=policy)
    term = TerminationPolicy(budget=Budget(max_iterations=5))
    loop = AgentLoop(llm, _tools(), term)
    result = loop.run("compute")
    assert result.succeeded
    assert result.state.tool_calls_made == 1


def test_verifier_keeps_loop_running_until_satisfied():
    # First final answer lacks the keyword; second contains it.
    llm = MockLLM(responses=[say("not yet"), say("now with PASS marker")])
    term = TerminationPolicy(budget=Budget(max_iterations=5))
    loop = AgentLoop(llm, _tools(), term, verifiers=[KeywordVerifier("PASS")])
    result = loop.run("produce PASS")
    assert result.succeeded
    assert result.state.iteration == 2


def test_budget_stops_runaway_loop():
    forever = MockLLM(policy=lambda msgs: call_tool("python_exec", code="pass"))
    term = TerminationPolicy(budget=Budget(max_iterations=3, max_tool_calls=3))
    loop = AgentLoop(forever, _tools(), term)
    result = loop.run("never finish")
    assert result.phase is LoopPhase.FAILED
    assert "budget" in result.final_message or result.state.phase is LoopPhase.FAILED


def test_transitions_are_recorded():
    llm = MockLLM(responses=[say("ok")])
    loop = AgentLoop(llm, _tools(), TerminationPolicy(budget=Budget()))
    result = loop.run("trace me")
    assert len(result.state.transitions) >= 1
    assert result.tracer.summary()  # non-empty event counts
