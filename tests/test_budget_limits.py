from loop_budget_agent.agent import MockBudgetLLM, run_task
from loop_budget_agent.budget import check_budget, estimate_cost_usd, estimate_tokens
from loop_budget_agent.state import AgentConfig


def test_step_cap() -> None:
    cfg = AgentConfig(max_steps=2)
    verdict = check_budget(step_count=3, token_count=10, cost_usd=0.0, config=cfg)
    assert verdict.exceeded
    assert "step" in verdict.reason


def test_token_cap() -> None:
    cfg = AgentConfig(max_tokens=100)
    verdict = check_budget(step_count=1, token_count=101, cost_usd=0.0, config=cfg)
    assert verdict.exceeded
    assert "token" in verdict.reason


def test_cost_cap() -> None:
    cfg = AgentConfig(max_cost_usd=0.01)
    verdict = check_budget(step_count=1, token_count=10, cost_usd=0.02, config=cfg)
    assert verdict.exceeded
    assert "cost" in verdict.reason


def test_within_budget() -> None:
    cfg = AgentConfig(max_steps=5, max_tokens=1000, max_cost_usd=1.0)
    verdict = check_budget(step_count=2, token_count=20, cost_usd=0.0001, config=cfg)
    assert verdict.exceeded is False


def test_token_estimate_is_positive() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("") == 0
    assert estimate_cost_usd(1_000_000, 1_000_000) > 0


def test_burn_mock_hits_step_budget(agent_config: AgentConfig, memory_checkpointer) -> None:
    cfg = agent_config.model_copy(update={"max_steps": 3, "loop_repeat_threshold": 99})
    result = run_task(
        "Call calculator forever",
        config=cfg,
        llm=MockBudgetLLM(behavior="burn"),
        checkpointer=memory_checkpointer,
    )
    assert result["status"] == "budget_halted"
    assert result["budget_exceeded"] is True
    assert result["step_count"] >= cfg.max_steps
    assert result["loop_detected"] is False
