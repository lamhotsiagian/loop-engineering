"""Working and episodic memory (Chapter 4).

The agent loop carries its short-term state in ``LoopState.messages``. That
window is finite, so long or repeated runs need two extra stores:

* **WorkingMemory** -- a compacted, structured summary of the current run that
  survives context-window truncation.
* **EpisodicMemory** -- a durable record of past runs the agent can retrieve
  from, so it does not re-learn the same lesson every time.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorkingMemory:
    """A bounded scratchpad of facts learned during a run."""

    facts: list[str] = field(default_factory=list)
    max_facts: int = 50

    def remember(self, fact: str) -> None:
        if fact not in self.facts:
            self.facts.append(fact)
        # Bound the store; drop the oldest facts first (a crude but effective
        # context-compaction policy -- Chapter 4 covers smarter strategies).
        if len(self.facts) > self.max_facts:
            self.facts = self.facts[-self.max_facts :]

    def as_context(self) -> str:
        if not self.facts:
            return ""
        return "Known facts:\n" + "\n".join(f"- {f}" for f in self.facts)


@dataclass
class Episode:
    goal: str
    succeeded: bool
    lesson: str
    at: float = field(default_factory=time.time)


class EpisodicMemory:
    """A JSON-lines durable store of past runs, queryable by keyword."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, episode: Episode) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(episode.__dict__) + "\n")

    def recall(self, query: str, limit: int = 3) -> list[Episode]:
        if not self.path.exists():
            return []
        hits: list[Episode] = []
        for line in self.path.read_text().splitlines():
            data = json.loads(line)
            if query.lower() in data["goal"].lower() or query.lower() in data["lesson"].lower():
                hits.append(Episode(**data))
        return hits[-limit:]
