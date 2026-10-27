"""build_graph(): the FireBinder StateGraph, its checkpointer, and the per-run config.

resolve -> gather -> reconcile -> draft -> audit -> route_after_audit
    -> "draft":        bump_revision -> draft   (the audit loop, revision + 1)
    -> "human_review": human_review -> finalize -> END
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # LANGSMITH_* must be in the environment before LangChain/LangGraph read it

import sqlite3  # noqa: E402
import subprocess  # noqa: E402
from collections.abc import Callable  # noqa: E402
from importlib import import_module  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Literal  # noqa: E402

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

from firebinder.nodes.finalize import finalize  # noqa: E402
from firebinder.nodes.reconcile import reconcile  # noqa: E402
from firebinder.nodes.resolve import resolve  # noqa: E402
from firebinder.nodes.review import MAX_REVISIONS, audit_failing, human_review  # noqa: E402
from firebinder.state import FireBinderState  # noqa: E402

DEFAULT_CHECKPOINT_PATH = "out/checkpoints.sqlite"
LLM_NODES = ("gather", "draft", "audit")  # live in firebinder.nodes.<name>.<name>


def route_after_audit(state: FireBinderState) -> Literal["draft", "human_review"]:
    """Failing audit (any contradicted, or >1 unsupported) and revision < 3 loops back to draft."""
    if audit_failing(state.get("audit", [])) and state.get("revision", 0) < MAX_REVISIONS:
        return "draft"
    return "human_review"


def bump_revision(state: FireBinderState) -> dict[str, int]:
    """Runs on the loop edge only, so draft sees revision + 1 alongside the audit that sent it back."""
    return {"revision": state.get("revision", 0) + 1}


def _llm_node(name: str) -> Callable[..., dict]:
    return getattr(import_module(f"firebinder.nodes.{name}"), name)


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    nodes_override: dict[str, Callable[..., dict]] | None = None,
) -> CompiledStateGraph:
    """Compile the graph. `nodes_override` maps node name -> function (tests stub the LLM nodes this way)."""
    overrides = nodes_override or {}
    nodes: dict[str, Callable[..., Any]] = {
        "resolve": resolve,
        "reconcile": reconcile,
        "bump_revision": bump_revision,
        "human_review": human_review,
        "finalize": finalize,
    }
    for name in LLM_NODES:
        if name not in overrides:
            nodes[name] = _llm_node(name)
    nodes.update(overrides)

    graph = StateGraph(FireBinderState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "resolve")
    graph.add_edge("resolve", "gather")
    graph.add_edge("gather", "reconcile")
    graph.add_edge("reconcile", "draft")
    graph.add_edge("draft", "audit")
    graph.add_conditional_edges(
        "audit", route_after_audit, {"draft": "bump_revision", "human_review": "human_review"}
    )
    graph.add_edge("bump_revision", "draft")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def get_checkpointer(path: str | Path = DEFAULT_CHECKPOINT_PATH) -> SqliteSaver:
    """A SqliteSaver on a long-lived connection, so a run paused at human_review can be resumed later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_config(
    incident: str,
    date: str,
    mode: str,
    thread_id: str,
    **configurable: Any,
) -> RunnableConfig:
    """Thread id for the checkpointer plus tags/metadata so LangSmith runs are comparable across versions.

    Extra keyword args (e.g. state="OR", skip_review=True) go into `configurable`.
    """
    sha = git_sha()
    return {
        "configurable": {"thread_id": thread_id, **configurable},
        "tags": [f"incident:{incident}", f"date:{date}", f"mode:{mode}", f"sha:{sha}"],
        "metadata": {"incident": incident, "date": date, "mode": mode, "sha": sha, "thread_id": thread_id},
        "run_name": f"firebinder {incident} {date}",
    }
