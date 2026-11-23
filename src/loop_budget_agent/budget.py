"""Hard step / token / dollar budgets. Enforced in code, not in the prompt."""

from __future__ import annotations

from dataclasses import dataclass

from .state import AgentConfig

# Approximate Gemini 2.5 Flash list prices (USD / million tokens).
INPUT_USD_PER_MILLION = 0.15
OUTPUT_USD_PER_MILLION = 0.60


def estimate_tokens(text: str) -> int:
    """Cheap local token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    )


def usage_from_message(message: object) -> tuple[int, int]:
    """Read provider usage metadata when present; otherwise estimate from text."""
    usage = getattr(message, "usage_metadata", None) or {}
    if isinstance(usage, dict) and (usage.get("input_tokens") or usage.get("output_tokens")):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    content = getattr(message, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    out_tokens = estimate_tokens(content)
    return 0, out_tokens


@dataclass(frozen=True)
class BudgetVerdict:
    exceeded: bool
    reason: str
    step_count: int
    token_count: int
    cost_usd: float


def check_budget(
    *,
    step_count: int,
    token_count: int,
    cost_usd: float,
    config: AgentConfig,
) -> BudgetVerdict:
    """Return a halt verdict if any hard cap is crossed or would be by this step."""
    if step_count > config.max_steps:
        return BudgetVerdict(
            True,
            f"step limit exceeded ({step_count} > {config.max_steps})",
            step_count,
            token_count,
            cost_usd,
        )
    if token_count > config.max_tokens:
        return BudgetVerdict(
            True,
            f"token limit exceeded ({token_count} > {config.max_tokens})",
            step_count,
            token_count,
            cost_usd,
        )
    if cost_usd > config.max_cost_usd:
        return BudgetVerdict(
            True,
            f"cost limit exceeded (${cost_usd:.6f} > ${config.max_cost_usd:.6f})",
            step_count,
            token_count,
            cost_usd,
        )
    return BudgetVerdict(False, "", step_count, token_count, cost_usd)


def would_exceed_next_step(step_count: int, config: AgentConfig) -> bool:
    return (step_count + 1) > config.max_steps
