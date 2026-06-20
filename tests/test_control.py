"""Tests for control structures (Chapter 3)."""
from __future__ import annotations

from loop_engineering_lab.core.control import (
    Budget,
    TerminationPolicy,
    destructive_action_guard,
    scope_guard,
)
from loop_engineering_lab.core.state import LoopPhase, LoopState


def test_budget_iteration_ceiling():
    budget = Budget(max_iterations=2)
    state = LoopState(goal="x")
    state.iteration = 2
    assert budget.exceeded(state) is not None


def test_scope_guard_blocks_outside_paths():
    guard = scope_guard(allowed_paths=("src/",))
    state = LoopState(goal="x")
    assert guard(state, "write_file", {"path": "etc/passwd"}) is not None
    assert guard(state, "write_file", {"path": "src/app.py"}) is None


def test_destructive_guard_requires_approval():
    guard = destructive_action_guard()
    state = LoopState(goal="x")
    assert guard(state, "drop_table", {}) is not None
    state.scratchpad["human_approved"] = {"drop_table"}
    assert guard(state, "drop_table", {}) is None


def test_success_predicate_short_circuits():
    policy = TerminationPolicy(budget=Budget(), success_when=lambda s: True)
    state = LoopState(goal="x")
    terminal = policy.check_terminal(state)
    assert terminal is not None
    assert terminal[0] is LoopPhase.SUCCEEDED


def test_no_progress_blocks():
    policy = TerminationPolicy(budget=Budget(), no_progress_limit=2)
    state = LoopState(goal="x")
    state.scratchpad["consecutive_no_progress"] = 2
    terminal = policy.check_terminal(state)
    assert terminal is not None
    assert terminal[0] is LoopPhase.BLOCKED
