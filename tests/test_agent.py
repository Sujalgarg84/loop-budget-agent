from loop_budget_agent.agent import MockBudgetLLM, create_llm, run_task
from loop_budget_agent.cli import main as cli_main
from loop_budget_agent.guardrails import is_transient_provider_error, provider_error_message
from loop_budget_agent.state import AgentConfig, Settings
from loop_budget_agent.tools import calculator


def test_calculator_tool() -> None:
    assert calculator.invoke({"expression": "17 * 24"}) == "408"
    assert "error" in calculator.invoke({"expression": "import os"}).lower()


def test_mock_arithmetic_run(agent_config: AgentConfig, memory_checkpointer) -> None:
    prompt = "What is 17 * 24?"
    llm = MockBudgetLLM(
        plans={
            prompt: [
                {"tool": "calculator", "args": {"expression": "17 * 24"}},
                {"answer": "408"},
            ]
        }
    )
    result = run_task(prompt, config=agent_config, llm=llm, checkpointer=memory_checkpointer)
    assert result["status"] == "completed"
    assert "408" in result["final_answer"]
    assert result["step_count"] >= 1
    assert result["cost_usd"] >= 0


def test_create_llm_mock_does_not_need_key() -> None:
    llm = create_llm(AgentConfig(mode="mock", google_api_key=""))
    assert llm._llm_type == "mock-budget-llm"


def test_settings_default_model() -> None:
    settings = Settings(_env_file=None)
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.google_api_key == ""


def test_429_is_transient() -> None:
    exc = RuntimeError("429 ResourceExhausted: quota exceeded")
    assert is_transient_provider_error(exc)
    assert "temporarily unavailable" in provider_error_message(exc).lower()


def test_503_is_transient() -> None:
    exc = RuntimeError("503 ServiceUnavailable")
    assert is_transient_provider_error(exc)


def test_graph_handles_429(agent_config: AgentConfig, memory_checkpointer) -> None:
    result = run_task(
        "Anything",
        config=agent_config,
        llm=MockBudgetLLM(behavior="fail_429"),
        checkpointer=memory_checkpointer,
    )
    assert result["status"] == "provider_error"
    assert "temporarily unavailable" in result["final_answer"].lower()


def test_graph_handles_503(agent_config: AgentConfig, memory_checkpointer) -> None:
    result = run_task(
        "Anything",
        config=agent_config,
        llm=MockBudgetLLM(behavior="fail_503"),
        checkpointer=memory_checkpointer,
    )
    assert result["status"] == "provider_error"


def test_cli_mock_run(capsys) -> None:
    code = cli_main(["--mock", "run", "What is 2 + 2?"])
    assert code == 0
    out = capsys.readouterr().out
    assert "completed" in out.lower() or "status" in out.lower()


def test_no_hardcoded_api_keys_in_source() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "AIza" not in text
        assert "GOOGLE_API_KEY=" not in text.replace('google_api_key: str = ""', "")
