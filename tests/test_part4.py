"""Tests for multi-agent, event-driven, and hill-climbing modules (Ch 10-12)."""
from __future__ import annotations

from loop_engineering_lab.automation.heartbeat import Event, Finding, Heartbeat
from loop_engineering_lab.core.agent_loop import AgentLoop
from loop_engineering_lab.core.control import Budget, TerminationPolicy
from loop_engineering_lab.hillclimbing.trace_analyzer import analyze_traces
from loop_engineering_lab.llm.mock import MockLLM, say
from loop_engineering_lab.multiagent.orchestrator import Orchestrator
from loop_engineering_lab.observability.tracing import Tracer
from loop_engineering_lab.tools.base import ToolRegistry
from loop_engineering_lab.tools.builtins import make_python_exec


def _tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(make_python_exec())
    return reg


def test_orchestrator_runs_isolated_workers():
    def build_worker() -> AgentLoop:
        # A fresh MockLLM per worker => genuinely isolated context.
        return AgentLoop(MockLLM(responses=[say("done")]), _tools(),
                         TerminationPolicy(budget=Budget(max_iterations=3)))

    orch = Orchestrator(build_worker)
    results = orch.run(["subtask A", "subtask B", "subtask C"])
    assert len(results) == 3
    assert Orchestrator.all_succeeded(results)
    assert results[0].subtask == "subtask A"


def test_orchestrator_reports_failures():
    def build_worker() -> AgentLoop:
        # Never finishes -> budget fails the worker.
        from loop_engineering_lab.llm.mock import call_tool
        llm = MockLLM(policy=lambda msgs: call_tool("python_exec", code="pass"))
        return AgentLoop(llm, _tools(), TerminationPolicy(budget=Budget(max_iterations=2)))

    orch = Orchestrator(build_worker)
    results = orch.run(["x"])
    assert not Orchestrator.all_succeeded(results)
    assert len(Orchestrator.failures(results)) == 1


def test_heartbeat_is_idempotent():
    hb = Heartbeat(handler=lambda e: Finding(e.id, f"handled {e.kind}"))
    e = Event(id="evt-1", kind="webhook")
    hb.deliver(e)
    hb.deliver(e)  # redelivery -> no-op
    assert hb.processed == 1
    assert hb.skipped_duplicates == 1
    assert len(hb.inbox) == 1


def test_heartbeat_silent_when_nothing_found():
    hb = Heartbeat(handler=lambda e: None)  # nothing worth surfacing
    findings = hb.tick([Event(id="a", kind="cron"), Event(id="b", kind="cron")])
    assert findings == []
    assert hb.processed == 2
    assert hb.inbox == []


def test_trace_analyzer_ranks_recurring_failures():
    tracers = []
    for _ in range(4):
        t = Tracer()
        t.emit(1, "tool", name="run_tests", ok=False)
        t.emit(1, "verify", name="rubric", passed=False)
        tracers.append(t)
    # One trace without the tool failure, to exercise the share calculation.
    t2 = Tracer()
    t2.emit(1, "verify", name="rubric", passed=False)
    tracers.append(t2)

    report = analyze_traces(tracers)
    assert report.n_traces == 5
    signals = {f.signal for f in report.findings}
    assert "verifier:rubric rejected" in signals
    assert "tool:run_tests failed" in signals
    # The verifier failure appears in all 5 traces -> share 1.0.
    rubric = next(f for f in report.findings if f.signal == "verifier:rubric rejected")
    assert rubric.share == 1.0
    assert "PROPOSED HARNESS CHANGE" in report.draft_harness_change()
