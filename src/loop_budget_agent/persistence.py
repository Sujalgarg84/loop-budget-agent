"""SQLite checkpoints so a run can pause for HITL and resume in another process."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional, Union

try:
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
except ImportError:  # pragma: no cover - older langgraph
    from langgraph.checkpoint.memory import MemorySaver

Checkpointer = Any


def make_checkpointer(path: Optional[Union[str, Path]] = None) -> Checkpointer:
    """Return a MemorySaver when path is None/empty, otherwise a SqliteSaver."""
    if not path:
        return MemorySaver()

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:  # pragma: no cover - package name varies by version
        from langgraph_checkpoint_sqlite import SqliteSaver  # type: ignore

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    setup = getattr(saver, "setup", None)
    if callable(setup):
        setup()
    return saver


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}
