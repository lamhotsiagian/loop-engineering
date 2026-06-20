"""The core agent loop (Chapter 2).

This is the load-bearing primitive of the whole book: a model calling tools in
a loop until a task is complete. It is implemented as an explicit state machine
so every transition is observable and every stop is a deliberate, named
decision rather than a fall-through.

The loop depends only on abstractions:

* an ``LLMClient`` (mock or real),
* a ``ToolRegistry`` (what the agent can do),
* a ``TerminationPolicy`` (budgets, success predicate, stopping rules),
* an optional list of ``Verifier`` objects (Chapter 8).

That dependency-inversion is what lets the same 120 lines run a hermetic unit
test and a production coding agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import LLMClient, Message
from ..observability.tracing import Tracer
from ..tools.base import ToolRegistry
from .control import TerminationPolicy
from .state import LoopPhase, LoopState

SYSTEM_PROMPT = (
    "You are an autonomous engineering agent operating inside a verification "
    "loop. Work toward the goal one step at a time. Call a tool when you need "
    "to act on the world; reply with a final answer only when the goal is "
    "demonstrably met."
)


@dataclass
class LoopResult:
    """The outcome of a finished run -- success/failure plus full trace."""

    phase: LoopPhase
    final_message: str
    state: LoopState
    tracer: Tracer

    @property
    def succeeded(self) -> bool:
        return self.phase is LoopPhase.SUCCEEDED

    def report(self) -> str:
        return (
            f"phase={self.phase.value} iterations={self.state.iteration} "
            f"tokens={self.state.tokens_used} tools={self.state.tool_calls_made} "
            f"events={self.tracer.summary()}"
        )


class AgentLoop:
    """A single-agent observe--reason--act loop with explicit control."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        policy: TerminationPolicy,
        verifiers: list | None = None,
        tracer: Tracer | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.policy = policy
        self.verifiers = verifiers or []
        self.tracer = tracer or Tracer()
        self.system_prompt = system_prompt

    def run(self, goal: str) -> LoopResult:
        state = LoopState(goal=goal)
        state.add_message("system", self.system_prompt)
        state.add_message("user", goal)

        while not state.phase.is_terminal:
            # 1) Control gate -- may we take another iteration at all?
            terminal = self.policy.check_terminal(state)
            if terminal is not None:
                phase, reason = terminal
                state.transition_to(phase, reason)
                self.tracer.emit(state.iteration, "terminal", phase=phase.value, reason=reason)
                break

            state.iteration += 1

            # 2) Reason -- ask the model for the next move.
            state.transition_to(LoopPhase.REASONING, "request next action")
            response = self.llm.complete(state.messages, tools=self.tools.schemas())
            state.tokens_used += response.total_tokens
            self.tracer.emit(
                state.iteration, "model",
                wants_tool=response.wants_tool, tokens=response.total_tokens,
            )

            # 3a) Final answer path -> verify before declaring success.
            if not response.wants_tool:
                state.add_message("assistant", response.text)
                if self._verify(state):
                    state.transition_to(LoopPhase.SUCCEEDED, "verified final answer")
                    self.tracer.emit(state.iteration, "terminal", phase="succeeded")
                    return LoopResult(LoopPhase.SUCCEEDED, response.text, state, self.tracer)
                # Verification failed -- feed that back and keep looping.
                self._record_no_progress(state, made_progress=False)
                continue

            # 3b) Action path -- run the requested tool calls.
            self._record_no_progress(state, made_progress=True)
            for call in response.tool_calls:
                veto = self.policy.veto_action(state, call.name, call.arguments)
                if veto is not None:
                    state.transition_to(LoopPhase.BLOCKED, veto)
                    self.tracer.emit(state.iteration, "terminal", phase="blocked", reason=veto)
                    return LoopResult(LoopPhase.BLOCKED, veto, state, self.tracer)

                state.transition_to(LoopPhase.ACTING, f"invoke {call.name}")
                result = self.tools.invoke(call.name, call.arguments)
                state.tool_calls_made += 1
                self.tracer.emit(state.iteration, "tool", name=call.name, ok=result.ok)

                state.transition_to(LoopPhase.OBSERVING, f"observe {call.name}")
                state.add_message("tool", result.render(), tool_call_id=call.id)

        # Loop exited without an in-band success return.
        final = state.messages[-1].content if state.messages else ""
        return LoopResult(state.phase, final, state, self.tracer)

    # --- helpers ----------------------------------------------------------

    def _verify(self, state: LoopState) -> bool:
        if not self.verifiers:
            return True
        state.transition_to(LoopPhase.VERIFYING, "run verifiers")
        for verifier in self.verifiers:
            verdict = verifier.check(state)
            self.tracer.emit(state.iteration, "verify", name=verifier.name, passed=verdict.passed)
            if not verdict.passed:
                state.add_message("user", f"Verification failed: {verdict.reason}")
                return False
        return True

    @staticmethod
    def _record_no_progress(state: LoopState, made_progress: bool) -> None:
        key = "consecutive_no_progress"
        state.scratchpad[key] = 0 if made_progress else state.scratchpad.get(key, 0) + 1
