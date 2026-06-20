"""Structured tool-result feedback (Chapter 6).

Raw tool output -- a bare stack trace, a verbose log -- is rarely the most
useful form for the next reasoning step. This module pre-processes a failed
``ToolResult`` into structured feedback: it attaches the causal context, a
one-line reminder of the goal, and -- most importantly -- a flag for whether
this error is a *repeat* of one already seen this run, which measurably reduces
repeated failed attempts.
"""
from __future__ import annotations

import re

from ..state.scratchpad import Scratchpad
from .base import ToolResult

# A crude but effective error fingerprint: the exception class + first line.
_ERR_CLASS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*Error|Exception)\b")


def error_fingerprint(text: str) -> str:
    """Reduce a stack trace to a stable signature for repeat-detection."""
    m = _ERR_CLASS.search(text)
    cls = m.group(1) if m else "Error"
    first = text.strip().splitlines()[-1] if text.strip() else ""
    return f"{cls}:{first[:80]}"


def structured_feedback(
    result: ToolResult,
    *,
    goal: str,
    code_context: str = "",
    scratchpad: Scratchpad | None = None,
) -> str:
    """Turn a raw tool result into structured feedback for the next step."""
    if result.ok:
        return f"[OK] (goal: {goal})\n{result.content.strip()}"

    fp = error_fingerprint(result.content)
    repeat = False
    if scratchpad is not None:
        repeat = scratchpad.has_tried("tool_error", fp)
        scratchpad.record_attempt("tool_error", "failed", fp)

    parts = [
        f"[FAILED] while pursuing: {goal}",
        f"error: {fp}",
        "NOTE: this is a REPEAT of an error already seen this run -- change "
        "approach rather than retrying the same fix."
        if repeat
        else "this is a NEW error.",
    ]
    if code_context:
        parts.append("relevant code:\n" + code_context.strip())
    parts.append("raw:\n" + result.content.strip()[:1200])
    return "\n".join(parts)
