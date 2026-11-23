from loop_budget_agent.agent import MockBudgetLLM, resume_task, run_task
from loop_budget_agent.guardrails import SENSITIVE_TOOLS, has_sensitive_tool
from loop_budget_agent.state import AgentConfig


def test_send_notification_is_sensitive() -> None:
    assert "send_notification" in SENSITIVE_TOOLS
    assert has_sensitive_tool([{"name": "send_notification", "args": {}}])
    assert not has_sensitive_tool([{"name": "calculator", "args": {"expression": "1+1"}}])


def test_sensitive_tool_pauses_for_human(agent_config: AgentConfig, memory_checkpointer) -> None:
    result = run_task(
        "Notify ops that deploy finished",
        config=agent_config,
        llm=MockBudgetLLM(behavior="hitl"),
        checkpointer=memory_checkpointer,
        thread_id="hitl-thread-1",
    )
    assert result["status"] == "hitl_pending"
    assert result["needs_approval"] is True
    assert result["interrupt"] is not None


def test_approve_resumes_and_sends(agent_config: AgentConfig, memory_checkpointer) -> None:
    llm = MockBudgetLLM(behavior="hitl", hitl_recipient="ops@example.com")
    paused = run_task(
        "Send a notification to ops@example.com saying 'deploy complete'.",
        config=agent_config,
        llm=llm,
        checkpointer=memory_checkpointer,
        thread_id="hitl-thread-2",
    )
    assert paused["status"] == "hitl_pending"

    resumed = resume_task(
        "hitl-thread-2",
        approved=True,
        config=agent_config,
        llm=llm,
        checkpointer=memory_checkpointer,
    )
    assert resumed["status"] == "completed"
    assert "notification sent" in resumed["final_answer"].lower()
    assert "ops@example.com" in resumed["final_answer"]


def test_reject_does_not_send(agent_config: AgentConfig, memory_checkpointer) -> None:
    llm = MockBudgetLLM(behavior="hitl")
    paused = run_task(
        "Send a notification",
        config=agent_config,
        llm=llm,
        checkpointer=memory_checkpointer,
        thread_id="hitl-thread-3",
    )
    assert paused["status"] == "hitl_pending"

    resumed = resume_task(
        "hitl-thread-3",
        approved=False,
        config=agent_config,
        llm=llm,
        checkpointer=memory_checkpointer,
    )
    assert resumed["status"] == "rejected"
    assert "rejected" in resumed["final_answer"].lower()
    history_names = [h["name"] for h in resumed["tool_history"]]
    assert "send_notification" not in history_names
