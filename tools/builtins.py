"""A handful of realistic built-in tools (Chapter 6).

These are deliberately simple but not toys: a sandboxed Python runner, file
read/write confined to a working directory, and a pytest runner that returns
structured pass/fail counts a verifier can consume.
"""
from __future__ import annotations

import io
import contextlib
import subprocess
import sys
from pathlib import Path

from .base import Tool, ToolResult


def make_python_exec() -> Tool:
    """Run a short Python snippet and capture stdout/stderr.

    In production this would run inside a network-isolated container with a CPU
    and memory ceiling. Here we capture output in-process to keep the demo
    dependency-free, while keeping the *interface* identical to a sandboxed
    runner.
    """

    def handler(code: str) -> ToolResult:
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(code, "<agent_snippet>", "exec"), {"__name__": "__agent__"})
        except Exception as exc:  # noqa: BLE001 - capture as observation
            return ToolResult(
                ok=False,
                content=f"{err.getvalue()}{type(exc).__name__}: {exc}",
                data={"stdout": out.getvalue()},
            )
        return ToolResult(ok=True, content=out.getvalue() or "(no output)", data={"stdout": out.getvalue()})

    return Tool(
        name="python_exec",
        description="Execute a Python snippet and return stdout or the error.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to run"}},
            "required": ["code"],
        },
        handler=handler,
        side_effecting=True,
    )


def make_file_tools(workdir: Path) -> tuple[Tool, Tool]:
    """Read/write tools confined to ``workdir`` (path-traversal is rejected)."""
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    def _safe(path: str) -> Path:
        target = (workdir / path).resolve()
        if not str(target).startswith(str(workdir)):
            raise ValueError(f"path '{path}' escapes the working directory")
        return target

    def read_handler(path: str) -> ToolResult:
        try:
            target = _safe(path)
        except ValueError as exc:
            return ToolResult(ok=False, content=str(exc))
        if not target.exists():
            return ToolResult(ok=False, content=f"no such file: {path}")
        return ToolResult(ok=True, content=target.read_text())

    def write_handler(path: str, content: str) -> ToolResult:
        try:
            target = _safe(path)
        except ValueError as exc:
            return ToolResult(ok=False, content=str(exc))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return ToolResult(ok=True, content=f"wrote {len(content)} bytes to {path}")

    read_tool = Tool(
        name="read_file",
        description="Read a UTF-8 text file from the working directory.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        handler=read_handler,
    )
    write_tool = Tool(
        name="write_file",
        description="Write a UTF-8 text file into the working directory.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        handler=write_handler,
        side_effecting=True,
    )
    return read_tool, write_tool


def make_pytest_runner(workdir: Path) -> Tool:
    """Run pytest in ``workdir`` and return structured results for a verifier."""

    def handler(target: str = ".") -> ToolResult:
        proc = subprocess.run(  # noqa: S603 - trusted local invocation
            [sys.executable, "-m", "pytest", "-q", target],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        passed = proc.returncode == 0
        return ToolResult(
            ok=passed,
            content=proc.stdout[-2000:] + proc.stderr[-500:],
            data={"returncode": proc.returncode, "passed": passed},
        )

    return Tool(
        name="run_tests",
        description="Run the project's pytest suite and report pass/fail.",
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string", "default": "."}},
        },
        handler=handler,
        side_effecting=False,
    )
