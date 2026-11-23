import json
from pathlib import Path

from loop_budget_agent.evaluate import BENCHMARK_TASKS, grade, run_benchmark
from loop_budget_agent.evaluate import BenchmarkTask
from loop_budget_agent.state import AgentConfig


def test_benchmark_has_fifty_tasks() -> None:
    assert len(BENCHMARK_TASKS) == 50
    ids = [t.id for t in BENCHMARK_TASKS]
    assert len(ids) == len(set(ids))


def test_categories_cover_guardrails() -> None:
    cats = {t.category for t in BENCHMARK_TASKS}
    assert {"arithmetic", "loop_trap", "budget_trap", "hitl", "unanswerable"} <= cats


def test_grade_contains_and_status() -> None:
    task = BenchmarkTask(
        id="t",
        prompt="x",
        category="arithmetic",
        expected_answer="408",
        expected_status="completed",
    )
    assert grade(task, {"status": "completed", "final_answer": "The answer is 408."})
    assert not grade(task, {"status": "completed", "final_answer": "nope"})

    loop_task = BenchmarkTask(
        id="l",
        prompt="x",
        category="loop_trap",
        expected_answer="loop",
        expected_status="loop_halted",
        grader="status",
    )
    assert grade(loop_task, {"status": "loop_halted", "final_answer": "Stopped"})


def test_run_benchmark_offline(tmp_path: Path) -> None:
    out = tmp_path / "benchmark.json"
    payload = run_benchmark(
        output_path=out,
        config=AgentConfig(mode="mock", google_api_key=""),
    )
    assert out.exists()
    assert payload["n_tasks"] == 50
    assert payload["passed"] == 50
    assert payload["accuracy"] == 1.0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_tasks"] == 50
    assert (tmp_path / "benchmark.md").exists()
