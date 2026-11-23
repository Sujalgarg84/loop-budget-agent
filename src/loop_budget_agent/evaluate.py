"""50-task offline benchmark. Runs entirely in mock mode (no API key)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .agent import MockBudgetLLM, resume_task, run_task
from .persistence import make_checkpointer
from .state import AgentConfig

RESULTS_DIR = Path("results")


@dataclass
class BenchmarkTask:
    id: str
    prompt: str
    category: str
    expected_answer: str
    expected_status: str
    plan: list[dict[str, Any]] = field(default_factory=list)
    mock_behavior: str = "auto"
    max_steps: Optional[int] = None
    auto_approve: bool = False
    grader: str = "contains"


def build_benchmark_tasks() -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []

    arithmetic = [
        (12, 8, "*"),
        (17, 24, "*"),
        (81, 9, "/"),
        (144, 12, "/"),
        (250, 75, "+"),
        (1000, 337, "-"),
        (19, 21, "*"),
        (64, 8, "/"),
        (7, 13, "+"),
        (90, 45, "-"),
    ]
    for i, (a, b, op) in enumerate(arithmetic, 1):
        expr = f"{a} {op} {b}"
        value = str(int(eval(expr)))  # noqa: S307 — fixed numeric literals only
        tasks.append(
            BenchmarkTask(
                id=f"arith_{i:02d}",
                prompt=f"What is {expr}?",
                category="arithmetic",
                expected_answer=value,
                expected_status="completed",
                plan=[
                    {"tool": "calculator", "args": {"expression": expr}},
                    {"answer": value},
                ],
            )
        )

    multi = [
        ("(3 + 5) * 4", "32"),
        ("(10 - 2) * 9", "72"),
        ("(18 / 3) + 7", "13"),
        ("(6 * 7) - 5", "37"),
        ("(100 / 4) * 3", "75"),
        ("(2 ** 5) + 10", "42"),
        ("(15 + 5) / 4", "5"),
        ("(9 * 9) + 19", "100"),
    ]
    for i, (expr, value) in enumerate(multi, 1):
        tasks.append(
            BenchmarkTask(
                id=f"multi_{i:02d}",
                prompt=f"Compute {expr}.",
                category="multi_step_math",
                expected_answer=value,
                expected_status="completed",
                plan=[
                    {"tool": "calculator", "args": {"expression": expr}},
                    {"answer": value},
                ],
            )
        )

    lookups = [
        ("What is the capital of France?", "Paris", "paris"),
        ("What is the population of Tokyo?", "13960000", "tokyo"),
        ("What is the capital of Germany?", "Berlin", "berlin"),
        ("How long is the Nile in km?", "6650", "nile"),
        ("How tall is Mount Everest in meters?", "8849", "everest"),
        ("Who created Python?", "Guido van Rossum", "python"),
        ("What is the chemical formula of water?", "H2O", "water"),
        ("What is the largest ocean on Earth?", "Pacific", "pacific"),
    ]
    for i, (prompt, expected, key) in enumerate(lookups, 1):
        tasks.append(
            BenchmarkTask(
                id=f"lookup_{i:02d}",
                prompt=prompt,
                category="knowledge",
                expected_answer=expected,
                expected_status="completed",
                plan=[
                    {"tool": "knowledge_lookup", "args": {"query": key}},
                    {"answer": expected},
                ],
            )
        )

    mixed = [
        (
            "Add 1000 to the population of Paris.",
            "2162000",
            [
                {"tool": "knowledge_lookup", "args": {"query": "paris"}},
                {"tool": "calculator", "args": {"expression": "2161000 + 1000"}},
                {"answer": "2162000"},
            ],
        ),
        (
            "How many meters is Everest plus 151?",
            "9000",
            [
                {"tool": "knowledge_lookup", "args": {"query": "everest"}},
                {"tool": "calculator", "args": {"expression": "8849 + 151"}},
                {"answer": "9000"},
            ],
        ),
        (
            "Nile length minus Amazon length in km?",
            "250",
            [
                {"tool": "knowledge_lookup", "args": {"query": "nile"}},
                {"tool": "knowledge_lookup", "args": {"query": "amazon"}},
                {"tool": "calculator", "args": {"expression": "6650 - 6400"}},
                {"answer": "250"},
            ],
        ),
        (
            "Berlin population divided by 3?",
            "1223000",
            [
                {"tool": "knowledge_lookup", "args": {"query": "berlin"}},
                {"tool": "calculator", "args": {"expression": "3669000 / 3"}},
                {"answer": "1223000"},
            ],
        ),
    ]
    for i, (prompt, expected, plan) in enumerate(mixed, 1):
        tasks.append(
            BenchmarkTask(
                id=f"mixed_{i:02d}",
                prompt=prompt,
                category="mixed",
                expected_answer=expected,
                expected_status="completed",
                plan=plan,
            )
        )

    unknown = [
        "What is the capital of Narnia?",
        "Who won the 2140 World Cup?",
        "What is SKU-9999's price?",
        "Translate this into Klingon: hello",
        "What is Alice's private home address?",
    ]
    for i, prompt in enumerate(unknown, 1):
        tasks.append(
            BenchmarkTask(
                id=f"unknown_{i:02d}",
                prompt=prompt,
                category="unanswerable",
                expected_answer="I don't know",
                expected_status="completed",
                plan=[{"answer": "I don't know."}],
                grader="contains",
            )
        )

    for i in range(1, 6):
        tasks.append(
            BenchmarkTask(
                id=f"loop_{i:02d}",
                prompt=(
                    f"Keep searching the notes for the hidden passphrase "
                    f"ALPHA-NEVER-FOUND until you find it (variant {i})."
                ),
                category="loop_trap",
                expected_answer="loop detected",
                expected_status="loop_halted",
                mock_behavior="loop",
                grader="status",
            )
        )

    recipients = [
        ("ops@example.com", "deploy complete"),
        ("security@example.com", "rotate keys"),
        ("finance@example.com", "invoice ready"),
        ("sre@example.com", "pager silenced"),
        ("legal@example.com", "contract signed"),
    ]
    for i, (to, message) in enumerate(recipients, 1):
        tasks.append(
            BenchmarkTask(
                id=f"hitl_{i:02d}",
                prompt=f"Send a notification to {to} saying '{message}'.",
                category="hitl",
                expected_answer=f"Notification sent to {to}",
                expected_status="completed",
                mock_behavior="hitl",
                auto_approve=True,
                plan=[],
            )
        )

    for i in range(1, 6):
        tasks.append(
            BenchmarkTask(
                id=f"budget_{i:02d}",
                prompt=f"Call the calculator on a new expression every turn, forever (case {i}).",
                category="budget_trap",
                expected_answer="step limit exceeded",
                expected_status="budget_halted",
                mock_behavior="burn",
                max_steps=3,
                grader="status",
            )
        )

    if len(tasks) != 50:
        raise RuntimeError(f"expected 50 benchmark tasks, built {len(tasks)}")
    return tasks


BENCHMARK_TASKS = build_benchmark_tasks()


def grade(task: BenchmarkTask, result: dict[str, Any]) -> bool:
    if task.grader == "status" or task.expected_status != "completed":
        return result.get("status") == task.expected_status
    if result.get("status") != task.expected_status:
        return False
    answer = (result.get("final_answer") or "").lower()
    return task.expected_answer.lower() in answer


def run_benchmark(
    *,
    output_path: Optional[Path] = None,
    config: Optional[AgentConfig] = None,
) -> dict[str, Any]:
    base = config or AgentConfig(mode="mock", max_steps=10)
    plans = {t.prompt: t.plan for t in BENCHMARK_TASKS if t.plan}
    rows: list[dict[str, Any]] = []
    passed = 0

    for task in BENCHMARK_TASKS:
        cfg = base.model_copy(
            update={
                "mode": "mock",
                "max_steps": task.max_steps or base.max_steps,
            }
        )
        llm = MockBudgetLLM(
            plans=plans,
            behavior=task.mock_behavior,  # type: ignore[arg-type]
            hitl_recipient=_email_from(task.prompt),
            hitl_message=_quoted_message(task.prompt),
        )
        saver = make_checkpointer(None)
        result = run_task(task.prompt, config=cfg, llm=llm, checkpointer=saver)
        if result["status"] == "hitl_pending" and task.auto_approve:
            result = resume_task(
                result["thread_id"],
                approved=True,
                config=cfg,
                llm=llm,
                checkpointer=saver,
            )
        ok = grade(task, result)
        passed += int(ok)
        rows.append(
            {
                "id": task.id,
                "category": task.category,
                "passed": ok,
                "expected_status": task.expected_status,
                "status": result["status"],
                "expected_answer": task.expected_answer,
                "final_answer": result["final_answer"],
                "step_count": result["step_count"],
                "token_count": result["token_count"],
                "cost_usd": result["cost_usd"],
                "loop_detected": result["loop_detected"],
                "budget_exceeded": result["budget_exceeded"],
            }
        )

    summary = _summarize(rows, passed)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock",
        "n_tasks": len(BENCHMARK_TASKS),
        "passed": passed,
        "accuracy": summary["accuracy"],
        "by_category": summary["by_category"],
        "results": rows,
    }

    path = output_path or (RESULTS_DIR / "benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = path.with_suffix(".md")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    payload["output_path"] = str(path)
    payload["markdown_path"] = str(md_path)
    return payload


def _email_from(prompt: str) -> str:
    import re

    match = re.search(r"[\w.]+@[\w.]+", prompt)
    return match.group(0) if match else "ops@example.com"


def _quoted_message(prompt: str) -> str:
    import re

    match = re.search(r"'([^']+)'", prompt)
    return match.group(1) if match else "deploy complete"


def _summarize(rows: list[dict[str, Any]], passed: int) -> dict[str, Any]:
    cats: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = cats.setdefault(row["category"], {"n": 0, "passed": 0})
        bucket["n"] += 1
        bucket["passed"] += int(row["passed"])
    return {
        "accuracy": passed / len(rows) if rows else 0.0,
        "by_category": {
            name: {
                **vals,
                "accuracy": vals["passed"] / vals["n"] if vals["n"] else 0.0,
            }
            for name, vals in cats.items()
        },
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Loop Budget Agent — 50-task benchmark",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Mode: `{payload['mode']}` (offline, no API key)",
        f"- Passed: **{payload['passed']}/{payload['n_tasks']}** "
        f"({payload['accuracy'] * 100:.1f}%)",
        "",
        "## By category",
        "",
        "| Category | Passed | N | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, vals in payload["by_category"].items():
        lines.append(
            f"| {name} | {vals['passed']} | {vals['n']} | {vals['accuracy'] * 100:.1f}% |"
        )
    lines += ["", "## Tasks", "", "| ID | Category | Status | Pass | Steps |", "|---|---|---|---|---:|"]
    for row in payload["results"]:
        mark = "yes" if row["passed"] else "no"
        lines.append(
            f"| {row['id']} | {row['category']} | {row['status']} | {mark} | {row['step_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run the 50-task mock benchmark.")
    parser.add_argument("--out", default=str(RESULTS_DIR / "benchmark.json"))
    args = parser.parse_args(argv)
    payload = run_benchmark(output_path=Path(args.out))
    print(
        f"Passed {payload['passed']}/{payload['n_tasks']} "
        f"({payload['accuracy'] * 100:.1f}%) -> {payload['output_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
