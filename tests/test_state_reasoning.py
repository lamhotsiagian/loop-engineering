"""Tests for scratchpad, structured feedback, and reflection (Ch 4-6)."""
from __future__ import annotations

from loop_engineering_lab.llm.mock import MockLLM, say
from loop_engineering_lab.reasoning.reflection import Reflector
from loop_engineering_lab.state.scratchpad import Scratchpad
from loop_engineering_lab.tools.base import ToolResult
from loop_engineering_lab.tools.structured_feedback import error_fingerprint, structured_feedback


def test_scratchpad_detects_repeated_attempt():
    sp = Scratchpad()
    sp.record_attempt("patch query", "failed", "TypeError")
    assert sp.has_tried("patch query", "TypeError")
    assert not sp.has_tried("patch query", "ValueError")
    assert sp.consecutive_failures() == 1


def test_scratchpad_compaction_preserves_substance():
    sp = Scratchpad(keep_verbatim=2)
    for i in range(6):
        sp.record_attempt(f"fix-{i}", "failed", "Err")
    sp.compact()
    assert len(sp.attempts) == 2          # only recent kept verbatim
    assert "fix-0" in sp.summary           # older folded into summary
    assert "Recent attempts" in sp.as_context()


def test_error_fingerprint_is_stable():
    a = error_fingerprint("Traceback...\nTypeError: bad operand")
    b = error_fingerprint("Traceback (other)...\nTypeError: bad operand")
    assert a == b == "TypeError:TypeError: bad operand"


def test_structured_feedback_flags_repeat():
    sp = Scratchpad()
    bad = ToolResult(ok=False, content="TypeError: bad operand")
    first = structured_feedback(bad, goal="fix bug", scratchpad=sp)
    second = structured_feedback(bad, goal="fix bug", scratchpad=sp)
    assert "NEW error" in first
    assert "REPEAT" in second


def test_reflector_stops_on_approval():
    # Critique call returns APPROVED immediately.
    llm = MockLLM(responses=[say("APPROVED")])
    reflector = Reflector(llm, max_rounds=3)
    out = reflector.refine("goal", "draft", revise=lambda d, issue: d + "!")
    assert out == "draft"  # unchanged because approved on first critique


def test_reflector_revises_then_accepts():
    llm = MockLLM(responses=[say("REVISE: missing edge case"), say("APPROVED")])
    reflector = Reflector(llm, max_rounds=3)
    out = reflector.refine("goal", "draft", revise=lambda d, issue: d + " [fixed]")
    assert "[fixed]" in out
