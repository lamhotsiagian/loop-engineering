"""Orchestrator--worker coordination (Chapter 10).

The orchestrator holds the overall plan; each worker runs its own loop over a
single sub-task with its own fresh context and budget. The isolation is the
point: a worker that only ever sees its one sub-task reasons more reliably than
one large agent juggling the whole task at once. Building a *fresh* worker per
sub-task is how this implementation guarantees that isolation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.agent_loop import AgentLoop, LoopResult


@dataclass
class SubtaskResult:
    subtask: str
    result: LoopResult

    @property
    def succeeded(self) -> bool:
        return self.result.succeeded


class Orchestrator:
    """Dispatches sub-tasks to freshly built worker loops and collects results.

    ``build_worker`` is a factory so every worker starts with isolated state --
    no shared context window, no cross-talk. Concurrency and merge happen at
    explicit synchronization points (here, the end of ``run``).
    """

    def __init__(self, build_worker: Callable[[], AgentLoop]) -> None:
        self._build_worker = build_worker

    def run(self, subtasks: list[str]) -> list[SubtaskResult]:
        results: list[SubtaskResult] = []
        for subtask in subtasks:
            worker = self._build_worker()          # fresh worker => isolated context
            results.append(SubtaskResult(subtask, worker.run(subtask)))
        return results

    @staticmethod
    def all_succeeded(results: list[SubtaskResult]) -> bool:
        """A synchronization barrier: the orchestrator only proceeds when every
        parallel worker has completed successfully."""
        return all(r.succeeded for r in results)

    @staticmethod
    def failures(results: list[SubtaskResult]) -> list[SubtaskResult]:
        return [r for r in results if not r.succeeded]
