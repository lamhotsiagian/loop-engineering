# loop_engineering_lab

A small, **runnable** reference implementation that accompanies *Loop
Engineering: A Practitioner's Guide*. Each module maps to a chapter so you can
read the book and the source side by side.

It runs with **no API key** by default — the LLM is a deterministic mock, so
the demos and tests are reproducible offline and in CI.

## Layout

| Path | Chapter | Purpose |
|------|---------|---------|
| `llm/base.py` | 2 | `LLMClient` protocol, `Message`, `LLMResponse` |
| `llm/mock.py` | 2 | Deterministic, scripted mock model |
| `llm/openai_adapter.py` | 2 | Optional real provider (no hard dependency) |
| `core/state.py` | 2 | Loop state machine (`LoopPhase`, `LoopState`) |
| `core/control.py` | 3 | `Budget`, `TerminationPolicy`, stopping rules |
| `core/agent_loop.py` | 2 | The observe–reason–act loop |
| `tools/` | 6 | Tool registry + sandboxed built-ins |
| `memory/` | 4 | Working + episodic memory |
| `verifiers/` | 8 | Verdict-returning verifiers |
| `observability/` | 13 | Structured tracer |
| `tests/` | — | Hermetic unit tests |

## Run

```bash
# from the repository root
python -m loop_engineering_lab demo      # a verified ReAct loop, end to end
python -m loop_engineering_lab budget     # a runaway loop stopped by its budget

# tests
pip install pytest
pytest loop_engineering_lab/tests
```

## Use a real model (optional)

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python -m loop_engineering_lab demo --real
```

The agent loop depends only on the `LLMClient` protocol, so swapping the mock
for a real provider changes no orchestration code — that seam is the point.
# loop-engineering
