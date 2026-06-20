"""Hill-climbing: batch trace analysis (Chapter 12).

A single trace is debugging data; aggregated across many runs, traces are
evidence about which parts of the harness systematically underperform. This
analyzer scans a batch of traces, ranks recurring failure signals (a tool that
fails often, a verifier that rejects often), and drafts a proposed harness
change for *human review* -- the loop never edits the harness unattended.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..observability.tracing import Tracer


@dataclass
class TraceFinding:
    signal: str            # e.g. "tool:run_tests failed"
    count: int
    share: float           # fraction of analysed traces exhibiting it


@dataclass
class AnalysisReport:
    n_traces: int
    findings: list[TraceFinding] = field(default_factory=list)

    def top(self, k: int = 3) -> list[TraceFinding]:
        return self.findings[:k]

    def draft_harness_change(self) -> str:
        """A proposed, human-reviewable change -- never auto-applied."""
        if not self.findings:
            return "No systematic failure pattern found; no harness change proposed."
        worst = self.findings[0]
        return (
            f"PROPOSED HARNESS CHANGE (requires human review)\n"
            f"Dominant failure signal: {worst.signal} "
            f"({worst.count} occurrences across {self.n_traces} traces, "
            f"{worst.share:.0%}).\n"
            f"Suggested action: review the prompt/tool/grader responsible for "
            f"'{worst.signal}' and adjust, then re-evaluate on a held-out set."
        )


def analyze_traces(tracers: list[Tracer]) -> AnalysisReport:
    """Aggregate failure signals across a batch of run traces."""
    n = len(tracers)
    signal_counts: Counter[str] = Counter()
    traces_with_signal: Counter[str] = Counter()

    for tracer in tracers:
        seen_here: set[str] = set()
        for ev in tracer.events:
            signal = None
            if ev.kind == "tool" and ev.detail.get("ok") is False:
                signal = f"tool:{ev.detail.get('name', '?')} failed"
            elif ev.kind == "verify" and ev.detail.get("passed") is False:
                signal = f"verifier:{ev.detail.get('name', '?')} rejected"
            elif ev.kind == "terminal" and ev.detail.get("phase") == "failed":
                signal = "loop terminated: failed"
            if signal:
                signal_counts[signal] += 1
                seen_here.add(signal)
        for s in seen_here:
            traces_with_signal[s] += 1

    findings = [
        TraceFinding(signal=s, count=c, share=(traces_with_signal[s] / n if n else 0.0))
        for s, c in signal_counts.most_common()
    ]
    return AnalysisReport(n_traces=n, findings=findings)
