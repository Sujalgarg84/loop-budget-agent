"""Shared fixtures. Tests stay offline: no GOOGLE_API_KEY, no network."""

from __future__ import annotations

import os

import pytest

from loop_budget_agent.agent import MockBudgetLLM, build_graph
from loop_budget_agent.persistence import make_checkpointer
from loop_budget_agent.state import AgentConfig


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    # Guard against accidental live calls if a key leaked into the parent env.
    os.environ.pop("GOOGLE_API_KEY", None)


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(
        mode="mock",
        google_api_key="",
        max_steps=8,
        max_tokens=8_000,
        max_cost_usd=0.05,
        loop_repeat_threshold=3,
    )


@pytest.fixture
def memory_checkpointer():
    return make_checkpointer(None)


@pytest.fixture
def mock_llm() -> MockBudgetLLM:
    return MockBudgetLLM()


@pytest.fixture
def graph(agent_config: AgentConfig, mock_llm: MockBudgetLLM, memory_checkpointer):
    return build_graph(agent_config, llm=mock_llm, checkpointer=memory_checkpointer)
