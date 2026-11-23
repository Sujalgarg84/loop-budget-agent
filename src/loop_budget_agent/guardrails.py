"""Deterministic loop detection, sensitive-tool policy, and provider-error handling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

SENSITIVE_TOOLS = frozenset({"send_notification"})

_TRANSIENT_MARKERS = (
    "429",
    "503",
    "resourceexhausted",
    "serviceunavailable",
    "rate limit",
    "too many requests",
    "unavailable",
    "temporarily",
    "quota",
)


@dataclass(frozen=True)
class LoopVerdict:
    detected: bool
    reason: str
    signature: str


def call_signature(entry: dict[str, Any]) -> str:
    payload = {"name": entry.get("name"), "args": entry.get("args") or {}}
    return json.dumps(payload, sort_keys=True, default=str)


def detect_loop(history: list[dict[str, Any]], threshold: int = 3) -> LoopVerdict:
    """Halt on identical trailing calls, repeated signatures, or a 2-cycle."""
    if threshold < 2:
        threshold = 2
    if len(history) < threshold:
        return LoopVerdict(False, "", "")

    sigs = [call_signature(item) for item in history]
    last = sigs[-1]

    trailing = sigs[-threshold:]
    if all(sig == last for sig in trailing):
        return LoopVerdict(
            True,
            f"identical tool call repeated {threshold} times in a row",
            last,
        )

    if sigs.count(last) >= threshold:
        return LoopVerdict(
            True,
            f"same tool signature appeared {sigs.count(last)} times (threshold {threshold})",
            last,
        )

    if (
        len(sigs) >= 4
        and sigs[-4] == sigs[-2]
        and sigs[-3] == sigs[-1]
        and sigs[-4] != sigs[-3]
    ):
        return LoopVerdict(True, "alternating two-call cycle detected", last)

    return LoopVerdict(False, "", last)


def has_sensitive_tool(tool_calls: Iterable[dict[str, Any]]) -> bool:
    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        if name in SENSITIVE_TOOLS:
            return True
    return False


def is_transient_provider_error(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (429, 503):
        return True
    return any(marker in blob for marker in _TRANSIENT_MARKERS)


def provider_error_message(exc: BaseException) -> str:
    if is_transient_provider_error(exc):
        return (
            "The language model is temporarily unavailable (rate limit or outage). "
            "No further tool calls were made. Retry later or use --mock."
        )
    return f"The language model failed: {exc}"
