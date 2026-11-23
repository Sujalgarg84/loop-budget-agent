"""Deterministic tools. Sensitive actions are flagged for HITL, not auto-executed."""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool

from .guardrails import SENSITIVE_TOOLS

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

FACTS: dict[str, str] = {
    "paris": "Paris is the capital of France. Population: 2161000.",
    "france": "France is a country in Europe. Capital: Paris. Population: 68000000.",
    "tokyo": "Tokyo is the capital of Japan. Population: 13960000.",
    "japan": "Japan is an island country in East Asia. Capital: Tokyo.",
    "berlin": "Berlin is the capital of Germany. Population: 3669000.",
    "germany": "Germany is a country in Europe. Capital: Berlin. Currency: euro.",
    "nile": "The Nile is a river in Africa. Length: 6650 km.",
    "amazon": "The Amazon is a river in South America. Length: 6400 km.",
    "everest": "Mount Everest is the highest mountain on Earth. Height: 8849 m.",
    "pacific": "The Pacific Ocean is the largest ocean on Earth.",
    "python": "Python is a programming language created by Guido van Rossum in 1991.",
    "langgraph": "LangGraph is a library for building stateful multi-actor LLM applications.",
    "gemini": "Gemini is a family of multimodal models from Google.",
    "water": "The chemical formula of water is H2O. Boiling point at 1 atm: 100 C.",
    "earth": "Earth is the third planet from the Sun. One year is about 365.25 days.",
}

NOTES: dict[str, str] = {
    "oncall": "On-call rotation this week: Priya (primary), Alex (secondary).",
    "deploy": "Production deploys require a human-approved notification to ops@example.com.",
    "budget": "Default run budget is 10 steps, 8000 tokens, and $0.05.",
    "office": "The London office address is 10 Example Street. Hours: 09:00-17:00.",
    "sku": "SKU-4412 is a USB-C hub. Unit cost: 24.50 USD. Stock: 180.",
}


def _eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_math(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_math(node.left)
        right = _eval_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression using + - * / // % ** and parentheses."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval_math(tree)
    except Exception as exc:  # noqa: BLE001 — tool errors are returned to the model
        return f"Calculator error: {exc}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _lookup(store: dict[str, str], query: str, empty: str) -> str:
    q = (query or "").lower()
    hits = [text for key, text in store.items() if key in q or q in key]
    if not hits:
        for key, text in store.items():
            if any(word in text.lower() for word in q.split() if len(word) > 3):
                hits.append(text)
    if not hits:
        return empty
    return " ".join(dict.fromkeys(hits))


@tool
def knowledge_lookup(query: str) -> str:
    """Look up a short fact from the local knowledge base. Not the public internet."""
    return _lookup(FACTS, query, "No facts found.")


@tool
def search_notes(query: str) -> str:
    """Search internal notes. Returns 'No notes found.' when nothing matches."""
    return _lookup(NOTES, query, "No notes found.")


@tool
def send_notification(recipient: str, message: str) -> str:
    """Send a notification to a recipient. This is a sensitive action and needs approval."""
    return f"Notification sent to {recipient}: {message}"


ALL_TOOLS = [calculator, knowledge_lookup, search_notes, send_notification]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    raw = getattr(message, "tool_calls", None) or []
    normalized: list[dict[str, Any]] = []
    for call in raw:
        if isinstance(call, dict):
            normalized.append(
                {
                    "name": call.get("name", ""),
                    "args": call.get("args") or {},
                    "id": call.get("id", ""),
                }
            )
        else:
            normalized.append(
                {
                    "name": getattr(call, "name", ""),
                    "args": getattr(call, "args", {}) or {},
                    "id": getattr(call, "id", ""),
                }
            )
    return normalized


def is_sensitive_name(name: str) -> bool:
    return name in SENSITIVE_TOOLS
