from loop_budget_agent.agent import MockBudgetLLM, resume_task, run_task
from loop_budget_agent.persistence import make_checkpointer, thread_config
from loop_budget_agent.state import AgentConfig


def test_memory_checkpointer_is_not_sqlite() -> None:
    saver = make_checkpointer(None)
    assert type(saver).__name__ in {"MemorySaver", "InMemorySaver"}


def test_sqlite_checkpointer_creates_file(tmp_path) -> None:
    db = tmp_path / "ckpts.sqlite"
    saver = make_checkpointer(db)
    assert db.exists()
    assert type(saver).__name__ == "SqliteSaver"


def test_sqlite_roundtrip_resumes_hitl(tmp_path, agent_config: AgentConfig) -> None:
    db = tmp_path / "hitl.sqlite"
    saver = make_checkpointer(db)
    llm = MockBudgetLLM(behavior="hitl", hitl_recipient="ops@example.com")
    paused = run_task(
        "Notify ops",
        config=agent_config,
        llm=llm,
        checkpointer=saver,
        thread_id="persist-1",
    )
    assert paused["status"] == "hitl_pending"

    # Simulate a new process: new saver + new graph, same sqlite file and thread id.
    saver2 = make_checkpointer(db)
    llm2 = MockBudgetLLM(behavior="hitl", hitl_recipient="ops@example.com")
    resumed = resume_task(
        "persist-1",
        approved=True,
        config=agent_config,
        llm=llm2,
        checkpointer=saver2,
    )
    assert resumed["status"] == "completed"
    assert "notification sent" in resumed["final_answer"].lower()


def test_thread_config_shape() -> None:
    cfg = thread_config("abc")
    assert cfg["configurable"]["thread_id"] == "abc"
