"""Tests for tools and memory (Chapters 6 & 4)."""
from __future__ import annotations

from pathlib import Path

from loop_engineering_lab.memory.working import Episode, EpisodicMemory, WorkingMemory
from loop_engineering_lab.tools.base import ToolRegistry
from loop_engineering_lab.tools.builtins import make_file_tools, make_python_exec


def test_python_exec_captures_output():
    tool = make_python_exec()
    result = tool.handler(code="print('hello')")
    assert result.ok
    assert "hello" in result.content


def test_python_exec_captures_error_as_observation():
    tool = make_python_exec()
    result = tool.handler(code="raise ValueError('boom')")
    assert not result.ok
    assert "ValueError" in result.content


def test_file_tools_confined_to_workdir(tmp_path: Path):
    read_tool, write_tool = make_file_tools(tmp_path)
    write_tool.handler(path="a.txt", content="data")
    assert read_tool.handler(path="a.txt").content == "data"
    # Path traversal is rejected, surfaced as a failed observation.
    escaped = write_tool.handler(path="../escape.txt", content="x")
    assert not escaped.ok


def test_registry_unknown_tool_is_observation_not_crash():
    reg = ToolRegistry()
    result = reg.invoke("nope", {})
    assert not result.ok


def test_working_memory_is_bounded():
    mem = WorkingMemory(max_facts=2)
    mem.remember("a")
    mem.remember("b")
    mem.remember("c")
    assert len(mem.facts) == 2
    assert "a" not in mem.facts


def test_episodic_recall(tmp_path: Path):
    store = EpisodicMemory(tmp_path / "episodes.jsonl")
    store.record(Episode(goal="fix flaky test", succeeded=True, lesson="seed RNG"))
    hits = store.recall("flaky")
    assert hits and hits[0].lesson == "seed RNG"
