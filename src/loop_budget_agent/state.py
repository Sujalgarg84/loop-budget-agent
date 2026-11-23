"""Agent state, runtime settings, and shared constants."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import TypedDict

Status = Literal[
    "running",
    "completed",
    "loop_halted",
    "budget_halted",
    "hitl_pending",
    "rejected",
    "provider_error",
]

HALT_STATUSES = frozenset(
    {"completed", "loop_halted", "budget_halted", "rejected", "provider_error"}
)


class Settings(BaseSettings):
    """Environment-backed settings. Never required for mock/offline runs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    google_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    max_steps: int = 10
    max_tokens: int = 8_000
    max_cost_usd: float = 0.05
    loop_repeat_threshold: int = 3
    checkpoint_path: str = "checkpoints.sqlite"


class AgentConfig(BaseSettings):
    """Per-run limits. Can be constructed from Settings or overridden in tests."""

    model_config = SettingsConfigDict(extra="ignore")

    google_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    max_steps: int = 10
    max_tokens: int = 8_000
    max_cost_usd: float = 0.05
    loop_repeat_threshold: int = 3
    checkpoint_path: str = "checkpoints.sqlite"
    mode: Literal["live", "mock"] = "mock"

    @classmethod
    def from_settings(
        cls, settings: Optional[Settings] = None, **overrides: Any
    ) -> AgentConfig:
        src = settings or Settings()
        data = {
            "google_api_key": src.google_api_key,
            "gemini_model": src.gemini_model,
            "max_steps": src.max_steps,
            "max_tokens": src.max_tokens,
            "max_cost_usd": src.max_cost_usd,
            "loop_repeat_threshold": src.loop_repeat_threshold,
            "checkpoint_path": src.checkpoint_path,
        }
        data.update(overrides)
        return cls(**data)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    step_count: int
    token_count: int
    cost_usd: float
    tool_history: list[dict[str, Any]]
    loop_detected: bool
    budget_exceeded: bool
    status: Status
    final_answer: str
    needs_approval: bool
    pending_action: Optional[dict[str, Any]]
    error: str


def initial_state(task: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "step_count": 0,
        "token_count": 0,
        "cost_usd": 0.0,
        "tool_history": [],
        "loop_detected": False,
        "budget_exceeded": False,
        "status": "running",
        "final_answer": "",
        "needs_approval": False,
        "pending_action": None,
        "error": "",
    }
