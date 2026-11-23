# loop-budget-agent

Runtime guardrails for tool-calling LLM agents built with **LangGraph** and **SQLite**. Prevents infinite tool-call loops, enforces hard token and USD spending limits before every model request, and supports stateful Human-in-the-Loop (HITL) interruption and resumption without duplicating tool side effects.

## Problem

Tool-calling agents often get stuck in retry loops when upstream tools return empty payloads, transient errors, or ambiguous statuses. Without execution-level circuit breakers, an unattended agent burns through token quotas and API credits before timing out.

`loop-budget-agent` wraps a LangGraph `StateGraph` with deterministic runtime checks:
- **Duplicate & Sequence Loop Detection:** Serializes tool arguments (`json.dumps(..., sort_keys=True)`) to catch identical retries (`A -> A`) and alternating tool cycles (`A -> B -> A -> B`) across a configurable history window.
- **Preflight & Post-Call Budget Caps:** Tracks cumulative input/output tokens and estimated USD cost against hard limits (`MAX_TOKEN_BUDGET`, `MAX_BUDGET_USD`). Hard limits cannot be bypassed even after operator approval.
- **Persistent HITL Checkpointing:** Suspends execution via LangGraph `interrupt()` backed by `SqliteSaver`, allowing an operator to inspect the state and resume (`APPROVE` or `STOP`) across process restarts using an execution ledger to prevent duplicate tool execution.

## Architecture

```text
  [Start / Resume Thread]
             |
             v
  +-----------------------------------------+
  | Agent Node                              |
  | (Preflight Token, USD & Step Check)     |
  +-----------------------------------------+
       |               |                |
       | Budget/Steps  | Final Answer   | Tool Calls Requested
       | Exceeded      |                v
       |               |     +-------------------------------------------+
       |               |     | Guardrails & Tool Executor                |
       |               |     | (Signature + A-B-A-B Cycle Detection)     |
       |               |     +-------------------------------------------+
       |               |          |                            |
       |               |          | Loop / Threshold           | Permitted &
       |               |          | Triggered                  | Not in Ledger
       |               |          v                            v
       |               |     +-----------------------+    +----------------------+
       |               |     | HITL Interrupt        |    | Execute Tool &       |
       |               |     | (SqliteSaver Persist) |    | Record Ledger        |
       |               |     +-----------------------+    +----------------------+
       |               |          |            |               |
       |               |          | STOP       | APPROVE       |
       v               v          v            +---------------+
  +-----------------------------------------+                  |
  | Finalize & Persist Terminal State       |                  |
  +-----------------------------------------+                  |
       ^                                                       |
       +-------------------------------------------------------+ (Loop back to Agent)
```

## Project Structure

```text
loop-budget-agent/
├── src/loop_budget_agent/
│   ├── state.py          # AgentState TypedDict and Pydantic v2 runtime settings
│   ├── tools.py          # Deterministic order/warehouse tools with simulated failure modes
│   ├── guardrails.py    # Exact-duplicate and alternating-sequence loop detection engine
│   ├── budget.py         # Token accounting, USD cost estimation, and hard budget enforcement
│   ├── persistence.py    # SQLite checkpointer lifecycle and thread recovery helpers
│   ├── agent.py          # LangGraph StateGraph wiring, execution ledger, and HITL routing
│   ├── cli.py            # Rich CLI interface for live, mock, resume, and benchmark modes
│   └── evaluate.py       # 50-task deterministic reliability and Stuck Rate benchmark
├── tests/                # Offline pytest suite (34 unit & integration tests)
├── results/              # Generated benchmark JSON/Markdown reports
├── .env.example
└── pyproject.toml
```

## Setup & Usage

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone [https://github.com/Sujalgarg84/loop-budget-agent.git](https://github.com/Sujalgarg84/loop-budget-agent.git)
cd loop-budget-agent
uv sync --extra dev
```

### Offline Mode (No API Key Required)

Runs with a deterministic mock model to exercise tool calls, loop detection, and budget cutoffs:

```bash
uv run loop-agent --mock run "What is 17 * 24?"
```

### Live Gemini Mode

Copy `.env.example` to `.env` and set your Google AI Studio key:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
MAX_ITERATIONS=10
MAX_TOKEN_BUDGET=10000
MAX_BUDGET_USD=0.05
```

Run a task against the live model:

```bash
uv run loop-agent run "Check whether order 1024 has been shipped"
```

## Evaluation Benchmark

`evaluate.py` runs 50 deterministic tasks covering normal execution, exact duplicate retries, alternating `A-B-A-B` cycles, flaky tool responses, budget exhaustion, and HITL escalations:

```bash
uv run loop-agent --mock eval
```

```text
Passed 50/50 (100.0%) -> results/benchmark.json
```

- **Stuck Rate:** `(unhandled_or_unbounded_tasks / total_tasks) * 100`. Tasks terminated cleanly by budget caps, loop circuit breakers, or operator `STOP` decisions are counted as safely handled (`0.0%` stuck rate).

## Running Tests

All 34 unit and integration tests run offline without network access or API credentials:

```bash
uv run pytest -v
```

## Limitations & Trade-offs

- **Exact vs. Semantic Loop Detection:** Tool signatures match normalized JSON arguments; semantically equivalent calls with varied phrasing or dynamic timestamps require embedding-based similarity checks.
- **Simulated Business Tools:** Included order and warehouse tools are deterministic local functions designed to reproduce flaky API edge cases safely.
- **Single-Node SQLite Checkpointer:** `SqliteSaver` fits local CLI usage and single-process recovery; multi-worker deployments should use `PostgresSaver`.
