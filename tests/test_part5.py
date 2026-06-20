"""Tests for observability metrics, safety, and economics (Ch 13-15)."""
from __future__ import annotations

import pytest

from loop_engineering_lab.core.agent_loop import AgentLoop
from loop_engineering_lab.core.control import Budget, TerminationPolicy
from loop_engineering_lab.core.safety import KillSwitch, KillSwitchTripped, StepGuard
from loop_engineering_lab.economics.router import CostMeter, ModelRouter
from loop_engineering_lab.llm.mock import MockLLM, call_tool, say
from loop_engineering_lab.observability.metrics import fleet_summary, run_metrics
from loop_engineering_lab.tools.base import ToolRegistry
from loop_engineering_lab.tools.builtins import make_python_exec


def _tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(make_python_exec())
    return reg


def _run(llm, budget) -> "AgentLoop":
    return AgentLoop(llm, _tools(), TerminationPolicy(budget=budget))


def test_run_metrics_flags_runaway_separately():
    forever = MockLLM(policy=lambda msgs: call_tool("python_exec", code="pass"))
    result = _run(forever, Budget(max_iterations=3, max_tool_calls=3)).run("never ends")
    m = run_metrics(result)
    assert not m.succeeded
    assert m.budget_exhausted          # runaway tracked explicitly
    assert m.tool_calls >= 1


def test_fleet_summary_separates_success_and_runaway():
    good = _run(MockLLM(responses=[say("done")]), Budget()).run("ok")
    forever = MockLLM(policy=lambda msgs: call_tool("python_exec", code="pass"))
    bad = _run(forever, Budget(max_iterations=2, max_tool_calls=2)).run("loop")
    summ = fleet_summary([good, bad])
    assert summ["n"] == 2
    assert summ["success_rate"] == 0.5
    assert summ["runaway_rate"] == 0.5


def test_kill_switch_halts_independently():
    ks = KillSwitch()
    ks.ensure_live()                   # no-op while live
    ks.trip("operator stop")
    assert ks.tripped
    with pytest.raises(KillSwitchTripped):
        ks.ensure_live()


def test_step_guard_reevaluates_current_state():
    guard = StepGuard(sensitive={"deploy"}, approved={"deploy"})
    ok, _ = guard.allow("deploy", state_is_safe=lambda: True)
    assert ok
    # Same approval, but current state is now unsafe -> blocked.
    blocked, reason = guard.allow("deploy", state_is_safe=lambda: False)
    assert not blocked and "unsafe" in reason
    # Unapproved sensitive action is always blocked.
    guard2 = StepGuard(sensitive={"delete"})
    nope, _ = guard2.allow("delete", state_is_safe=lambda: True)
    assert not nope


def test_cost_meter_accumulates_by_model():
    meter = CostMeter(prices_per_1k={"cheap": 0.5, "strong": 10.0})
    meter.charge("cheap", 1000, 1000)   # 2k tokens * 0.5/1k = 1.0
    meter.charge("strong", 500, 500)    # 1k tokens * 10/1k = 10.0
    assert round(meter.total_usd, 2) == 11.0
    assert round(meter.by_model["strong"], 2) == 10.0


def test_model_router_routes_by_difficulty():
    router = ModelRouter(cheap="haiku", strong="opus", escalate_after_failures=2)
    assert router.route() == "haiku"                       # routine
    assert router.route(stuck=True) == "opus"              # stuck -> escalate
    assert router.route(verifier_failures=2) == "opus"     # repeated failure -> escalate
    assert router.route(verifier_failures=1) == "haiku"    # one failure -> still cheap
