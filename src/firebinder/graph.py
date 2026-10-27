"""build_graph(): the FireBinder StateGraph, its checkpointer, and the per-run config.

START -> resolve -> gather -> reconcile -> draft -> audit -> route_after_audit
    -> "draft":        draft (the audit loop; draft itself bumps revision when audit is non-empty)
    -> "human_review": human_review -> finalize -> END
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # LANGSMITH_* must be in the environment before LangChain/LangGraph read it

import sqlite3  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Literal  # noqa: E402

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

# Imported by name so tests can monkeypatch firebinder.graph.<node>; build_graph reads them at call time.
from firebinder.nodes.audit import audit  # noqa: E402
from firebinder.nodes.draft import draft  # noqa: E402
from firebinder.nodes.finalize import finalize  # noqa: E402
from firebinder.nodes.gather import gather  # noqa: E402
from firebinder.nodes.reconcile import reconcile  # noqa: E402  (already @traceable(name="reconcile"))
from firebinder.nodes.resolve import resolve  # noqa: E402
from firebinder.nodes.review import MAX_REVISIONS, audit_failing, human_review  # noqa: E402
from firebinder.state import FireBinderState  # noqa: E402

DEFAULT_CHECKPOINT_PATH = "out/checkpoints.sqlite"


def route_after_audit(state: FireBinderState) -> Literal["draft", "human_review"]:
    """Any contradicted, or more than one unsupported, and revision < 3 -> draft; otherwise human_review."""
    if audit_failing(state.get("audit", [])) and state.get("revision", 0) < MAX_REVISIONS:
        return "draft"
    return "human_review"


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(FireBinderState)
    graph.add_node("resolve", resolve)
    graph.add_node("gather", gather)
    graph.add_node("reconcile", reconcile)
    graph.add_node("draft", draft)
    graph.add_node("audit", audit)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "resolve")
    graph.add_edge("resolve", "gather")
    graph.add_edge("gather", "reconcile")
    graph.add_edge("reconcile", "draft")
    graph.add_edge("draft", "audit")
    graph.add_conditional_edges("audit", route_after_audit, ["draft", "human_review"])
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def get_checkpointer(path: str | None = None) -> SqliteSaver:
    """A SqliteSaver on a long-lived connection, so a run paused at human_review can be resumed later."""
    db = Path(path or DEFAULT_CHECKPOINT_PATH)
    db.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(db, check_same_thread=False))


def initial_state(incident_name: str, report_date: str) -> FireBinderState:
    return FireBinderState(
        incident_name=incident_name,
        report_date=report_date,
        irwin_id=None,
        centroid=None,
        mode="live",  # resolve sets the real mode
        evidence=[],
        conflicts=[],
        draft=None,
        audit=[],
        revision=0,
        approved=None,
        messages=[],
    )


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
    skip_review: bool = False,
    state: str | None = None,
) -> RunnableConfig:
    """Checkpointer thread plus LangSmith tags/metadata (incident, date, mode, git SHA) so runs compare across versions."""
    sha = git_sha()
    return {
        "configurable": {"thread_id": thread_id, "skip_review": skip_review, "state": state},
        "tags": [f"incident:{incident}", f"date:{date}", f"mode:{mode}", f"git_sha:{sha}"],
        "metadata": {"incident": incident, "date": date, "mode": mode, "git_sha": sha},
        "run_name": f"firebinder {incident} {date}",
    }
