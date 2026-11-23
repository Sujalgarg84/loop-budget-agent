"""LangGraph ReAct agent with budget, loop, HITL, and mock/live LLM backends."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import Field

from .budget import (
    check_budget,
    estimate_tokens,
    estimate_cost_usd,
    usage_from_message,
    would_exceed_next_step,
)
from .guardrails import detect_loop, has_sensitive_tool, provider_error_message
from .persistence import make_checkpointer, thread_config
from .state import AgentConfig, AgentState, HALT_STATUSES, initial_state
from .tools import ALL_TOOLS, tool_calls_from_message

SYSTEM_PROMPT = """You are a careful operations assistant with a hard step/token/cost budget.

Rules:
- Use tools when they are needed. Prefer calculator for arithmetic.
- Do not repeat the same tool with the same arguments.
- If the knowledge base or notes do not contain the answer, say you do not know.
- send_notification is sensitive and will pause for a human.
- When you have the answer, respond with a concise final answer and no tool call.
- Stay within the remaining step budget given below.
"""


class MockBudgetLLM(BaseChatModel):
    """Deterministic tool-calling model. Used by pytest and --mock. Zero network."""

    plans: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    behavior: Literal["auto", "loop", "burn", "hitl", "fail_429", "fail_503"] = "auto"
    loop_tool: str = "search_notes"
    loop_args: dict[str, Any] = Field(
        default_factory=lambda: {"query": "ALPHA-NEVER-FOUND"}
    )
    hitl_recipient: str = "ops@example.com"
    hitl_message: str = "deploy complete"
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "mock-budget-llm"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> MockBudgetLLM:
        return self.model_copy(update={"bound_tools": list(tools)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.behavior == "fail_429":
            raise RuntimeError("429 ResourceExhausted: quota exceeded")
        if self.behavior == "fail_503":
            raise RuntimeError("503 ServiceUnavailable: backend unavailable")

        task = _first_human_text(messages)
        tool_results = [m for m in messages if getattr(m, "type", "") == "tool"]
        n_results = len(tool_results)

        if self.behavior == "loop":
            return _tool_result(
                self.loop_tool,
                dict(self.loop_args),
                call_id=f"loop-{n_results}",
            )
        if self.behavior == "burn":
            expr = f"{n_results + 1} + {n_results + 2}"
            return _tool_result("calculator", {"expression": expr}, call_id=f"burn-{n_results}")
        if self.behavior == "hitl":
            if n_results == 0:
                return _tool_result(
                    "send_notification",
                    {"recipient": self.hitl_recipient, "message": self.hitl_message},
                    call_id="hitl-0",
                )
            last = getattr(tool_results[-1], "content", "") or ""
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=str(last)))]
            )

        plan = self.plans.get(task) or _heuristic_plan(task)
        return _follow_plan(plan, tool_results)


def _tool_result(name: str, args: dict[str, Any], call_id: str) -> ChatResult:
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )
    return ChatResult(generations=[ChatGeneration(message=msg)])


def _follow_plan(plan: list[dict[str, Any]], tool_results: list[Any]) -> ChatResult:
    remaining = plan[len(tool_results) :]
    if remaining and "tool" in remaining[0]:
        step = remaining[0]
        return _tool_result(
            step["tool"],
            dict(step.get("args") or {}),
            call_id=step.get("id") or f"plan-{len(tool_results)}",
        )

    if remaining and remaining[0].get("answer") is not None:
        answer = str(remaining[0]["answer"])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    if tool_results:
        last = getattr(tool_results[-1], "content", "") or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(last)))])

    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="I don't know."))]
    )


def _first_human_text(messages: list[BaseMessage]) -> str:
    for msg in messages:
        if getattr(msg, "type", "") == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _heuristic_plan(task: str) -> list[dict[str, Any]]:
    lower = task.lower()
    if any(word in lower for word in ("notify", "notification", "email", "send a message")):
        recipient = "ops@example.com"
        match = re.search(r"[\w.]+@[\w.]+", task)
        if match:
            recipient = match.group(0)
        return [
            {
                "tool": "send_notification",
                "args": {"recipient": recipient, "message": task},
            },
            {"answer": f"Notification sent to {recipient}"},
        ]
    if any(word in lower for word in ("keep searching", "until you find", "hidden passphrase")):
        return [
            {"tool": "search_notes", "args": {"query": "ALPHA-NEVER-FOUND"}},
            {"tool": "search_notes", "args": {"query": "ALPHA-NEVER-FOUND"}},
            {"tool": "search_notes", "args": {"query": "ALPHA-NEVER-FOUND"}},
        ]
    math = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", task)
    if math:
        expr = f"{math.group(1)} {math.group(2)} {math.group(3)}"
        return [{"tool": "calculator", "args": {"expression": expr}}]
    if any(word in lower for word in ("what is", "who", "capital", "population", "length")):
        return [{"tool": "knowledge_lookup", "args": {"query": task}}]
    return [{"answer": "I don't know."}]


def create_llm(config: AgentConfig, mock: Optional[MockBudgetLLM] = None) -> BaseChatModel:
    if config.mode == "mock" or mock is not None:
        return mock or MockBudgetLLM()

    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {
        "model": config.gemini_model,
        "temperature": 0,
    }
    if config.google_api_key:
        kwargs["google_api_key"] = config.google_api_key
    return ChatGoogleGenerativeAI(**kwargs)


def build_graph(
    config: AgentConfig,
    *,
    llm: Optional[BaseChatModel] = None,
    tools: Optional[list[BaseTool]] = None,
    checkpointer: Any = None,
):
    tools = tools or list(ALL_TOOLS)
    model = llm or create_llm(config)
    if hasattr(model, "bind_tools"):
        bound = model.bind_tools(tools)
    else:
        bound = model
    tool_node = ToolNode(tools)

    def llm_node(state: AgentState) -> dict[str, Any]:
        if would_exceed_next_step(state["step_count"], config):
            verdict = check_budget(
                step_count=state["step_count"] + 1,
                token_count=state["token_count"],
                cost_usd=state["cost_usd"],
                config=config,
            )
            return {
                "budget_exceeded": True,
                "status": "budget_halted",
                "final_answer": f"Stopped: {verdict.reason}.",
                "step_count": state["step_count"] + 1,
            }

        remaining = max(0, config.max_steps - state["step_count"])
        sys = SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\nRemaining steps: {remaining}. "
                f"Tokens used: {state['token_count']}/{config.max_tokens}. "
                f"Cost used: ${state['cost_usd']:.6f}/${config.max_cost_usd:.6f}."
            )
        )
        try:
            response = bound.invoke([sys, *state["messages"]])
        except Exception as exc:  # noqa: BLE001 — fail closed, never crash the CLI
            return {
                "status": "provider_error",
                "error": str(exc),
                "final_answer": provider_error_message(exc),
                "step_count": state["step_count"] + 1,
            }

        in_tok, out_tok = usage_from_message(response)
        if in_tok == 0:
            in_tok = estimate_tokens(state["task"]) + estimate_tokens(sys.content)
        tokens = state["token_count"] + in_tok + out_tok
        cost = state["cost_usd"] + estimate_cost_usd(in_tok, out_tok)
        calls = tool_calls_from_message(response)
        pending = {"tool_calls": calls} if has_sensitive_tool(calls) else None
        budget = check_budget(
            step_count=state["step_count"] + 1,
            token_count=tokens,
            cost_usd=cost,
            config=config,
        )
        update: dict[str, Any] = {
            "messages": [response],
            "step_count": state["step_count"] + 1,
            "token_count": tokens,
            "cost_usd": cost,
            "needs_approval": pending is not None,
            "pending_action": pending,
        }
        if budget.exceeded:
            update.update(
                {
                    "budget_exceeded": True,
                    "status": "budget_halted",
                    "final_answer": f"Stopped: {budget.reason}.",
                }
            )
        return update

    def tools_with_history(state: AgentState) -> dict[str, Any]:
        result = tool_node.invoke(state)
        last_ai = _last_ai(state["messages"])
        calls = tool_calls_from_message(last_ai) if last_ai else []
        history = list(state.get("tool_history") or []) + [
            {"name": c["name"], "args": c["args"]} for c in calls
        ]
        loop = detect_loop(history, config.loop_repeat_threshold)
        update: dict[str, Any] = {**result, "tool_history": history, "pending_action": None}
        if loop.detected:
            update.update(
                {
                    "loop_detected": True,
                    "status": "loop_halted",
                    "final_answer": f"Stopped: loop detected ({loop.reason}).",
                }
            )
        return update

    def review_node(state: AgentState) -> dict[str, Any]:
        payload = state.get("pending_action") or {}
        decision = interrupt(
            {
                "type": "approval",
                "action": payload,
                "task": state.get("task"),
            }
        )
        approved = _is_approved(decision)
        if approved:
            return {
                "needs_approval": False,
                "status": "running",
            }
        return {
            "needs_approval": False,
            "status": "rejected",
            "pending_action": None,
            "final_answer": "Action rejected by human reviewer.",
            "messages": [AIMessage(content="Action rejected by human reviewer.")],
        }

    def finalize_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1] if state["messages"] else None
        content = getattr(last, "content", "") if last else ""
        if not isinstance(content, str):
            content = str(content)
        return {"status": "completed", "final_answer": content.strip()}

    def halt_node(state: AgentState) -> dict[str, Any]:
        if state.get("final_answer"):
            return {}
        return {"final_answer": "Stopped by guardrail."}

    def route_after_llm(state: AgentState) -> str:
        if state.get("status") in HALT_STATUSES:
            return "halt"
        last = _last_ai(state["messages"])
        calls = tool_calls_from_message(last) if last else []
        if not calls:
            return "finalize"
        if state.get("needs_approval") or has_sensitive_tool(calls):
            return "review"
        return "tools"

    def route_after_tools(state: AgentState) -> str:
        if state.get("status") in HALT_STATUSES:
            return "halt"
        return "llm"

    def route_after_review(state: AgentState) -> str:
        if state.get("status") == "rejected":
            return "halt"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_with_history)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("halt", halt_node)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {"tools": "tools", "review": "review", "finalize": "finalize", "halt": "halt"},
    )
    graph.add_conditional_edges("tools", route_after_tools, {"llm": "llm", "halt": "halt"})
    graph.add_conditional_edges(
        "review", route_after_review, {"tools": "tools", "halt": "halt"}
    )
    graph.add_edge("finalize", END)
    graph.add_edge("halt", END)

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[],
    )
    return compiled


def _last_ai(messages: list[Any]) -> Any:
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            return msg
    return None


def _is_approved(decision: Any) -> bool:
    if decision is True:
        return True
    if decision is False or decision is None:
        return False
    if isinstance(decision, str):
        return decision.strip().lower() in {"yes", "y", "approve", "approved", "true"}
    if isinstance(decision, dict):
        if "approved" in decision:
            return bool(decision["approved"])
        if "decision" in decision:
            return _is_approved(decision["decision"])
    return False


def _extract_interrupt(result: Any, graph: Any, config: dict[str, Any]) -> Optional[Any]:
    if isinstance(result, dict) and result.get("__interrupt__"):
        return result["__interrupt__"]
    snapshot = graph.get_state(config)
    if snapshot and getattr(snapshot, "tasks", None):
        for task in snapshot.tasks:
            interrupts = getattr(task, "interrupts", None) or ()
            if interrupts:
                return interrupts
    next_nodes = getattr(snapshot, "next", ()) if snapshot else ()
    if next_nodes:
        return next_nodes
    return None


def run_task(
    task: str,
    *,
    config: Optional[AgentConfig] = None,
    llm: Optional[BaseChatModel] = None,
    checkpointer: Any = None,
    thread_id: Optional[str] = None,
    recursion_limit: Optional[int] = None,
) -> dict[str, Any]:
    cfg = config or AgentConfig(mode="mock")
    saver = checkpointer if checkpointer is not None else make_checkpointer(None)
    graph = build_graph(cfg, llm=llm, checkpointer=saver)
    tid = thread_id or str(uuid4())
    invoke_cfg: dict[str, Any] = {
        **thread_config(tid),
        "recursion_limit": recursion_limit or max(25, cfg.max_steps * 4),
    }
    result = graph.invoke(initial_state(task), invoke_cfg)
    return _package_result(result, graph, invoke_cfg, tid, cfg)


def resume_task(
    thread_id: str,
    *,
    approved: bool,
    config: Optional[AgentConfig] = None,
    llm: Optional[BaseChatModel] = None,
    checkpointer: Any,
    recursion_limit: Optional[int] = None,
) -> dict[str, Any]:
    cfg = config or AgentConfig(mode="mock")
    graph = build_graph(cfg, llm=llm, checkpointer=checkpointer)
    invoke_cfg: dict[str, Any] = {
        **thread_config(thread_id),
        "recursion_limit": recursion_limit or max(25, cfg.max_steps * 4),
    }
    result = graph.invoke(Command(resume={"approved": approved}), invoke_cfg)
    return _package_result(result, graph, invoke_cfg, thread_id, cfg)


def _package_result(
    result: dict[str, Any],
    graph: Any,
    invoke_cfg: dict[str, Any],
    thread_id: str,
    cfg: AgentConfig,
) -> dict[str, Any]:
    pending = _extract_interrupt(result, graph, invoke_cfg)
    status = result.get("status") or "running"
    if pending and status not in HALT_STATUSES:
        status = "hitl_pending"
    return {
        "thread_id": thread_id,
        "status": status,
        "final_answer": result.get("final_answer") or "",
        "step_count": result.get("step_count") or 0,
        "token_count": result.get("token_count") or 0,
        "cost_usd": float(result.get("cost_usd") or 0.0),
        "loop_detected": bool(result.get("loop_detected")),
        "budget_exceeded": bool(result.get("budget_exceeded")),
        "needs_approval": status == "hitl_pending",
        "pending_action": result.get("pending_action"),
        "tool_history": result.get("tool_history") or [],
        "error": result.get("error") or "",
        "config": {
            "max_steps": cfg.max_steps,
            "max_tokens": cfg.max_tokens,
            "max_cost_usd": cfg.max_cost_usd,
            "mode": cfg.mode,
        },
        "interrupt": _jsonable(pending) if pending else None,
    }


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
