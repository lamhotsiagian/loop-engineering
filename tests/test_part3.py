"""Tests for robustness, rubric grading, and pattern builders (Ch 7-9)."""
from __future__ import annotations

from loop_engineering_lab.core.agent_loop import AgentLoop
from loop_engineering_lab.core.robustness import CircuitBreaker, StuckDetector
from loop_engineering_lab.core.state import LoopPhase
from loop_engineering_lab.llm.mock import MockLLM, say
from loop_engineering_lab.patterns.catalog import (
    retry_loop as build_retry_loop,
    test_driven_loop as build_test_driven_loop,
)
from loop_engineering_lab.tools.base import ToolRegistry
from loop_engineering_lab.tools.builtins import make_python_exec
from loop_engineering_lab.verifiers.base import KeywordVerifier
from loop_engineering_lab.verifiers.rubric_grader import RubricGrader, deterministic


def _tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(make_python_exec())
    return reg


def test_circuit_breaker_trips_on_category():
    cb = CircuitBreaker(threshold=3, cooldown_seconds=999)
    assert not cb.is_open("db_write")
    for _ in range(3):
        cb.record_failure("db_write")
    assert cb.is_open("db_write")           # whole category blocked
    assert not cb.is_open("file_read")      # unrelated category unaffected
    cb.record_success("db_write")
    assert not cb.is_open("db_write")


def test_stuck_detector_fires_without_progress():
    sd = StuckDetector(patience=2)
    assert not sd.observe(0.1)   # improvement
    assert not sd.observe(0.1)   # stale 1
    assert sd.observe(0.1)       # stale 2 -> stuck


def test_stuck_detector_resets_on_progress():
    sd = StuckDetector(patience=2)
    sd.observe(0.1)
    sd.observe(0.1)              # stale 1
    assert not sd.observe(0.5)  # improvement resets


def test_rubric_grader_deterministic_pass_and_fail():
    grader = RubricGrader(deterministic_checks=[
        deterministic("mentions_tests", lambda o: "tests" in o.lower(), "must mention tests"),
        deterministic("nonempty", lambda o: len(o) > 0, "empty output"),
    ])
    good = grader.grade("All tests pass.")
    assert good.passed and good.score == 1.0
    bad = grader.grade("done")
    assert not bad.passed
    assert "must mention tests" in bad.feedback()


def test_rubric_grader_as_verifier_in_loop():
    # First answer fails the rubric, second passes -> loop continues then succeeds.
    llm = MockLLM(responses=[say("done"), say("all tests pass")])
    grader = RubricGrader(deterministic_checks=[
        deterministic("tests", lambda o: "tests" in o.lower(), "mention tests"),
    ])
    from loop_engineering_lab.core.control import Budget, TerminationPolicy
    al = AgentLoop(llm, _tools(), TerminationPolicy(budget=Budget(max_iterations=5)), verifiers=[grader])
    result = al.run("produce output mentioning tests")
    assert result.phase is LoopPhase.SUCCEEDED
    assert result.state.iteration == 2


def test_pattern_builders_return_runnable_loops():
    llm = MockLLM(responses=[say("PASS marker")])
    loop = build_retry_loop(llm, _tools(), success_marker="PASS")
    assert loop.run("go").succeeded

    llm2 = MockLLM(responses=[say("contains DONE")])
    loop2 = build_test_driven_loop(llm2, _tools(), KeywordVerifier("DONE"))
    assert loop2.run("go").succeeded
