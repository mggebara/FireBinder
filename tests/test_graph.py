"""Graph wiring tests: offline, no LLM. gather/draft/audit are deterministic stubs; resolve_incident is patched."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typer.testing import CliRunner

import firebinder.graph as graph_module
from firebinder.cli import app
from firebinder.graph import build_graph, get_checkpointer, route_after_audit, run_config
from firebinder.nodes import resolve as resolve_module
from firebinder.nodes.resolve import ResolvedIncident, resolve
from firebinder.state import AuditFinding

IRWIN = "{TEST-IRWIN}"
EVIDENCE = [
    {"field": "acres", "value": 1200, "source": "get_incident_record", "url": "https://wfigs", "fetched_at": "2026-10-26T00:00:00Z"},
    {"field": "acres", "value": 5000, "source": "get_inciweb_updates", "url": "https://inciweb", "fetched_at": "2026-10-26T00:00:00Z"},
]


def finding(status: str) -> AuditFinding:
    return {"claim": f"a {status} claim", "status": status, "evidence": []}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    out = tmp_path / "out"
    monkeypatch.setenv("FIREBINDER_OUT", str(out))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(
        resolve_module,
        "resolve_incident",
        lambda name, date, state=None: ResolvedIncident(
            irwin_id=IRWIN, incident_name=name, state=state, centroid=(44.0, -121.0), discovery_date="2026-10-01"
        ),
    )
    return out


class Stubs:
    """Stub LLM nodes. `audit_by_revision(revision) -> findings` drives the loop."""

    def __init__(self, audit_by_revision: Callable[[int], list[AuditFinding]] = lambda r: [finding("supported")]):
        self.audit_by_revision = audit_by_revision
        self.draft_revisions: list[int] = []

    def gather(self, state: dict) -> dict:
        return {"evidence": EVIDENCE}

    def draft(self, state: dict) -> dict:
        self.draft_revisions.append(state["revision"])
        return {
            "draft": {
                "narrative": f"Draft revision {state['revision']}. Acres disagree between sources.",
                "fields": {"acres": 1200},
                "citations": {"acres": "https://wfigs"},
            }
        }

    def audit(self, state: dict) -> dict:
        return {"audit": self.audit_by_revision(state["revision"])}

    def overrides(self) -> dict[str, Callable[..., dict]]:
        return {"gather": self.gather, "draft": self.draft, "audit": self.audit}


def initial(date: str = "2026-10-26") -> dict[str, Any]:
    return {
        "incident_name": "TESTFIRE", "report_date": date, "irwin_id": None, "centroid": None, "mode": "live",
        "evidence": [], "conflicts": [], "draft": None, "audit": [], "revision": 0, "approved": None, "messages": [],
    }


def run(stubs: Stubs, *, skip_review: bool = True, thread: str = "t1", checkpointer: Any = None) -> tuple[Any, dict, dict]:
    graph = build_graph(checkpointer or MemorySaver(), nodes_override=stubs.overrides())
    config = run_config("TESTFIRE", "2026-10-26", "live", thread, skip_review=skip_review)
    return graph, config, graph.invoke(initial(), config)


def test_happy_path_writes_four_files(offline: Path) -> None:
    stubs = Stubs()
    _, _, result = run(stubs)
    out = offline / IRWIN / "2026-10-26"
    assert sorted(p.name for p in out.iterdir()) == ["209.md", "audit.json", "conflicts.json", "evidence.json"]
    assert result["approved"] is True and result["irwin_id"] == IRWIN
    assert stubs.draft_revisions == [0]
    md = (out / "209.md").read_text()
    assert "Draft revision 0" in md and "| acres | 1200 | https://wfigs |" in md
    assert "Flag:" not in md
    assert json.loads((out / "conflicts.json").read_text())[0]["field"] == "acres"  # reconcile ran for real


def test_contradiction_loops_once() -> None:
    stubs = Stubs(lambda r: [finding("contradicted")] if r == 0 else [finding("supported")])
    _, _, result = run(stubs)
    assert stubs.draft_revisions == [0, 1]
    assert result["revision"] == 1


def test_revision_cap_sends_flagged_draft_to_review(offline: Path) -> None:
    stubs = Stubs(lambda r: [finding("contradicted")])
    _, _, result = run(stubs)
    assert stubs.draft_revisions == [0, 1, 2, 3]
    assert result["revision"] == 3 and result["approved"] is True
    assert any(getattr(m, "name", None) == "audit_gate" for m in result["messages"])
    assert "Flag:" in (offline / IRWIN / "2026-10-26" / "209.md").read_text()


@pytest.mark.parametrize(("unsupported", "drafts"), [(2, [0, 1]), (1, [0])])
def test_unsupported_threshold(unsupported: int, drafts: list[int]) -> None:
    stubs = Stubs(lambda r: [finding("unsupported")] * unsupported if r == 0 else [])
    run(stubs)
    assert stubs.draft_revisions == drafts


def test_route_after_audit_rules() -> None:
    assert route_after_audit({"audit": [finding("contradicted")], "revision": 2}) == "draft"
    assert route_after_audit({"audit": [finding("contradicted")], "revision": 3}) == "human_review"
    assert route_after_audit({"audit": [finding("unsupported")], "revision": 0}) == "human_review"
    assert route_after_audit({"audit": [], "revision": 0}) == "human_review"


def test_interrupt_then_approve(offline: Path) -> None:
    graph, config, result = run(Stubs(), skip_review=False)
    assert "__interrupt__" in result
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert snapshot.values["draft"]["fields"] == {"acres": 1200}
    assert result["__interrupt__"][0].value["draft"]["fields"] == {"acres": 1200}
    assert not (offline / IRWIN).exists()

    final = graph.invoke(Command(resume={"action": "approve"}), config)
    assert final["approved"] is True
    assert (offline / IRWIN / "2026-10-26" / "209.md").exists()


def test_interrupt_then_edit_or_reject() -> None:
    edited = {"narrative": "Human edit.", "fields": {"acres": 1300}, "citations": {"acres": "https://wfigs"}}
    graph, config, _ = run(Stubs(), skip_review=False, thread="edit")
    final = graph.invoke(Command(resume={"action": "edit", "draft": edited}), config)
    assert final["draft"] == edited and final["approved"] is True

    graph, config, _ = run(Stubs(), skip_review=False, thread="reject")
    final = graph.invoke(Command(resume={"action": "reject"}), config)
    assert final["approved"] is False


def test_sqlite_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ckpt" / "checkpoints.sqlite"
    stubs = Stubs()
    _, config, _ = run(stubs, skip_review=False, checkpointer=get_checkpointer(path))
    assert path.exists()

    reopened = build_graph(get_checkpointer(path), nodes_override=stubs.overrides())
    assert reopened.get_state(config).next == ("human_review",)
    final = reopened.invoke(Command(resume={"action": "approve"}), config)
    assert final["approved"] is True


def test_resolve_sets_mode_and_reads_state_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(name: str, date: str, state: str | None = None) -> ResolvedIncident:
        seen["state"] = state
        return ResolvedIncident(irwin_id=IRWIN, incident_name=name, state=state, centroid=(1.0, 2.0), discovery_date=None)

    monkeypatch.setattr(resolve_module, "resolve_incident", fake)
    out = resolve(initial("2014-08-01"), {"configurable": {"state": "OR"}})
    assert out == {"irwin_id": IRWIN, "centroid": (1.0, 2.0), "mode": "historical"}
    assert seen["state"] == "OR"
    assert resolve(initial("2999-01-01"), {"configurable": {}})["mode"] == "live"


def test_run_config_tags() -> None:
    config = run_config("GRASSHOPPER", "2026-10-26", "live", "grasshopper-2026-10-26", skip_review=True)
    assert config["configurable"] == {"thread_id": "grasshopper-2026-10-26", "skip_review": True}
    assert config["tags"][:3] == ["incident:GRASSHOPPER", "date:2026-10-26", "mode:live"]
    assert config["tags"][3].startswith("sha:")
    assert config["metadata"]["incident"] == "GRASSHOPPER"


def test_cli_draft_then_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline: Path) -> None:
    stubs = Stubs()
    real_build = graph_module.build_graph
    ckpt = tmp_path / "cli.sqlite"
    monkeypatch.setattr(graph_module, "build_graph", lambda cp=None: real_build(cp, nodes_override=stubs.overrides()))
    monkeypatch.setattr(graph_module, "get_checkpointer", lambda path=None: get_checkpointer(ckpt))
    runner = CliRunner()

    result = runner.invoke(app, ["draft", "--incident", "Test Fire", "--date", "2026-10-26"])
    assert result.exit_code == 0, result.output
    assert "firebinder resume --thread-id test-fire-2026-10-26" in result.output

    edit = tmp_path / "edit.json"
    edit.write_text(json.dumps({"narrative": "Edited.", "fields": {"acres": 1}, "citations": {"acres": "x"}}))
    result = runner.invoke(app, ["resume", "--thread-id", "test-fire-2026-10-26", "--edit", str(edit)])
    assert result.exit_code == 0, result.output
    assert "Finalized (approved)" in result.output
    assert "Edited." in (offline / IRWIN / "2026-10-26" / "209.md").read_text()

    result = runner.invoke(app, ["draft", "--incident", "Test Fire", "--date", "2026-10-26", "--skip-review"])
    assert result.exit_code == 0, result.output
    assert "Thread test-fire-2026-10-26-2" in result.output
