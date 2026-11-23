from loop_budget_agent.agent import MockBudgetLLM, run_task
from loop_budget_agent.guardrails import call_signature, detect_loop
from loop_budget_agent.state import AgentConfig


def test_identical_trailing_calls_trip_loop() -> None:
    history = [{"name": "search_notes", "args": {"query": "x"}}] * 3
    verdict = detect_loop(history, threshold=3)
    assert verdict.detected
    assert "identical" in verdict.reason


def test_under_threshold_is_not_a_loop() -> None:
    history = [{"name": "search_notes", "args": {"query": "x"}}] * 2
    assert detect_loop(history, threshold=3).detected is False


def test_same_signature_scattered_trips_loop() -> None:
    history = [
        {"name": "calculator", "args": {"expression": "1+1"}},
        {"name": "knowledge_lookup", "args": {"query": "paris"}},
        {"name": "calculator", "args": {"expression": "1+1"}},
        {"name": "calculator", "args": {"expression": "1+1"}},
    ]
    assert detect_loop(history, threshold=3).detected


def test_abab_cycle_detected() -> None:
    history = [
        {"name": "search_notes", "args": {"query": "a"}},
        {"name": "knowledge_lookup", "args": {"query": "b"}},
        {"name": "search_notes", "args": {"query": "a"}},
        {"name": "knowledge_lookup", "args": {"query": "b"}},
    ]
    verdict = detect_loop(history, threshold=3)
    assert verdict.detected
    assert "cycle" in verdict.reason


def test_signatures_are_order_insensitive_for_args() -> None:
    a = call_signature({"name": "calculator", "args": {"b": 1, "a": 2}})
    b = call_signature({"name": "calculator", "args": {"a": 2, "b": 1}})
    assert a == b


def test_looping_mock_halts_the_graph(agent_config: AgentConfig, memory_checkpointer) -> None:
    cfg = agent_config.model_copy(update={"loop_repeat_threshold": 3, "max_steps": 10})
    result = run_task(
        "Keep searching until you find ALPHA-NEVER-FOUND",
        config=cfg,
        llm=MockBudgetLLM(behavior="loop"),
        checkpointer=memory_checkpointer,
    )
    assert result["status"] == "loop_halted"
    assert result["loop_detected"] is True
    assert "loop detected" in result["final_answer"].lower()
    assert result["step_count"] <= cfg.max_steps
