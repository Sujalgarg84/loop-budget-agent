"""Command-line interface for loop-budget-agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .agent import MockBudgetLLM, resume_task, run_task
from .evaluate import run_benchmark
from .persistence import make_checkpointer
from .state import AgentConfig, Settings

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loop-agent",
        description="Loop-budget LangGraph agent (mock-first, Gemini-backed).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run fully offline with the deterministic mock model (no API key).",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-cost", type=float, default=None, dest="max_cost_usd")
    parser.add_argument("--checkpoint", default=None, help="SQLite checkpoint path.")

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a single task.")
    run_p.add_argument("task", help="User task / question.")
    run_p.add_argument("--thread-id", default=None)

    resume_p = sub.add_parser("resume", help="Resume a HITL-paused thread.")
    resume_p.add_argument("thread_id")
    group = resume_p.add_mutually_exclusive_group(required=True)
    group.add_argument("--approve", action="store_true")
    group.add_argument("--reject", action="store_true")

    eval_p = sub.add_parser("eval", help="Run the 50-task offline benchmark.")
    eval_p.add_argument("--out", default="results/benchmark.json")

    return parser


def _config_from_args(args: argparse.Namespace) -> AgentConfig:
    settings = Settings()
    overrides: dict = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens
    if args.max_cost_usd is not None:
        overrides["max_cost_usd"] = args.max_cost_usd
    if args.checkpoint:
        overrides["checkpoint_path"] = args.checkpoint
    mode = "mock" if args.mock else "live"
    if mode == "live" and not (overrides.get("google_api_key") or settings.google_api_key):
        console.print(
            "[yellow]No GOOGLE_API_KEY found; falling back to --mock so the CLI stays usable.[/yellow]"
        )
        mode = "mock"
    overrides["mode"] = mode
    return AgentConfig.from_settings(settings, **overrides)


def _print_result(result: dict) -> None:
    table = Table(title="Run result")
    table.add_column("Field")
    table.add_column("Value")
    for key in (
        "thread_id",
        "status",
        "final_answer",
        "step_count",
        "token_count",
        "cost_usd",
        "loop_detected",
        "budget_exceeded",
        "needs_approval",
    ):
        table.add_row(key, str(result.get(key)))
    console.print(table)
    if result.get("interrupt"):
        console.print("[cyan]HITL interrupt — resume with:[/cyan]")
        console.print(f"  loop-agent resume {result['thread_id']} --approve")


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = _config_from_args(args)

    if args.command == "eval":
        payload = run_benchmark(output_path=Path(args.out), config=cfg.model_copy(update={"mode": "mock"}))
        console.print(
            f"[green]Passed {payload['passed']}/{payload['n_tasks']} "
            f"({payload['accuracy'] * 100:.1f}%)[/green] -> {payload['output_path']}"
        )
        return 0

    checkpointer = make_checkpointer(cfg.checkpoint_path)
    llm = MockBudgetLLM() if cfg.mode == "mock" else None

    if args.command == "run":
        result = run_task(
            args.task,
            config=cfg,
            llm=llm,
            checkpointer=checkpointer,
            thread_id=args.thread_id,
        )
        _print_result(result)
        return 0 if result["status"] != "provider_error" else 2

    if args.command == "resume":
        result = resume_task(
            args.thread_id,
            approved=bool(args.approve),
            config=cfg,
            llm=llm,
            checkpointer=checkpointer,
        )
        _print_result(result)
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
