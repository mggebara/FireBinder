"""Graph wiring tests: fake resolve/gather/draft/audit, real reconcile/human_review/finalize. No network, no model."""

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typer.testing import CliRunner

import firebinder.graph as graph_module
from firebinder.cli import app
from firebinder.graph import build_graph, initial_state, route_after_audit, run_config
from firebinder.nodes.review import FAILING_FLAG

IRWIN = "{FAKE-IRWIN-0001}"
DATE = "2024-08-01"
GEOMETRY = {"type": "MultiPolygon", "coordinates": [[[[0, 0], [0, 0.01], [0.01, 0.01], [0, 0]]]] * 3}


def _ev(field: str, value: object, source: str, url: str) -> dict:
    return {"field": field, "value": value, "source": source, "url": url, "fetched_at": "2024-08-01T12:00:00+00:00"}


def fake_resolve(state: dict) -> dict:
    return {"irwin_id": IRWIN, "centroid": (44.0, -120.0), "mode": "historical"}


def fake_gather(state: dict) -> dict:
    return {
        "evidence": [
            _ev("acres", 1000, "get_incident_record", "https://wfigs.example/1"),
            _ev("acres", 3000, "get_inciweb_updates", "https://inciweb.example/1"),
            _ev("perimeter_geometry", GEOMETRY, "get_incident_record", "https://wfigs.example/perim"),
        ],
        "messages": [],
    }


def fake_draft(state: dict) -> dict:
    out: dict = {
        "draft": {
            "narrative": "The fire is 1,000 acres; InciWeb reports 3,000 acres.",
            "fields": {"acres": 1000},
            "citations": {"acres": "https://wfigs.example/1"},
        }
    }
    if state.get("audit"):
        out["revision"] = state["revision"] + 1
    return out


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, int]:
    calls = {"audit": 0}

    def fake_audit(state: dict) -> dict:
        calls["audit"] += 1
        status = "contradicted" if calls["audit"] == 1 else "supported"
        return {"audit": [{"claim": "1,000 acres", "status": status, "evidence": []}]}

    monkeypatch.setattr(graph_module, "resolve", fake_resolve)
    monkeypatch.setattr(graph_module, "gather", fake_gather)
    monkeypatch.setattr(graph_module, "draft", fake_draft)
    monkeypatch.setattr(graph_module, "audit", fake_audit)
    monkeypatch.setenv("FIREBINDER_OUT_DIR", str(tmp_path))
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    return calls


def _config(thread: str, skip_review: bool = False) -> dict:
    return run_config("FAKE", DATE, "historical", thread, skip_review=skip_review)


def test_run_interrupts_then_approve_finalizes(patched: dict[str, int], tmp_path: Path) -> None:
    graph = build_graph(MemorySaver())
    config = _config("t1")
    result = graph.invoke(initial_state("FAKE", DATE), config)

    payload = result["__interrupt__"][0].value
    assert set(payload) == {"draft", "conflicts", "audit", "revision", "flag"}
    assert payload["revision"] == 1  # the loop fired exactly once
    assert payload["flag"] is None
    assert patched["audit"] == 2
    assert [c["field"] for c in payload["conflicts"]][0] == "acres"
    assert graph.get_state(config).next == ("human_review",)

    graph.invoke(Command(resume={"action": "approve"}), config)
    out = tmp_path / IRWIN / DATE
    assert sorted(p.name for p in out.iterdir()) == ["209.md", "audit.json", "conflicts.json", "evidence.json"]
    md = (out / "209.md").read_text()
    assert "approved" in md and "https://wfigs.example/1" in md and "## Conflicts" in md and "## Audit" in md
    geometry = next(e["value"] for e in json.loads((out / "evidence.json").read_text()) if e["field"] == "perimeter_geometry")
    assert geometry == {"type": "MultiPolygon", "parts": 3}
    assert graph.get_state(config).values["approved"] is True


def test_edit_resume_replaces_draft_and_rejects_bad_edits(patched: dict[str, int]) -> None:
    graph = build_graph(MemorySaver())
    config = _config("t2")
    graph.invoke(initial_state("FAKE", DATE), config)
    with pytest.raises(ValueError):
        graph.invoke(Command(resume={"action": "edit", "draft": {"narrative": "x"}}), config)

    graph = build_graph(MemorySaver())
    graph.invoke(initial_state("FAKE", DATE), config)
    edited = {"narrative": "Edited.", "fields": {}, "citations": {}}
    graph.invoke(Command(resume={"action": "edit", "draft": edited}), config)
    values = graph.get_state(config).values
    assert values["draft"] == edited and values["approved"] is True


def test_skip_review_bypasses_interrupt(patched: dict[str, int], tmp_path: Path) -> None:
    graph = build_graph(MemorySaver())
    config = _config("t3", skip_review=True)
    result = graph.invoke(initial_state("FAKE", DATE), config)
    assert "__interrupt__" not in result
    assert result["approved"] is None
    assert "bypassed" in (tmp_path / IRWIN / DATE / "209.md").read_text()


def _state(statuses: list[str], revision: int = 0) -> dict:
    return {"audit": [{"claim": "c", "status": s, "evidence": []} for s in statuses], "revision": revision}


@pytest.mark.parametrize(
    ("statuses", "revision", "expected"),
    [
        (["contradicted"], 0, "draft"),
        (["unsupported"], 0, "human_review"),
        (["unsupported", "unsupported"], 0, "draft"),
        (["supported"], 0, "human_review"),
        (["contradicted"], 2, "draft"),
        (["contradicted"], 3, "human_review"),
        (["unsupported", "unsupported"], 3, "human_review"),
    ],
)
def test_route_after_audit(statuses: list[str], revision: int, expected: str) -> None:
    assert route_after_audit(_state(statuses, revision)) == expected


def test_flag_set_when_revision_cap_hit(patched: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph_module, "audit", lambda state: {"audit": [{"claim": "c", "status": "contradicted", "evidence": []}]}
    )
    graph = build_graph(MemorySaver())
    result = graph.invoke(initial_state("FAKE", DATE), _config("t4"))
    payload = result["__interrupt__"][0].value
    assert payload["revision"] == 3 and payload["flag"] == FAILING_FLAG


def test_cli_draft_skip_review_prints_output_path(patched: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(graph_module, "get_checkpointer", lambda path=None: MemorySaver())
    result = CliRunner().invoke(
        app, ["draft", "--incident", "FAKE", "--date", DATE, "--skip-review"], env={"COLUMNS": "250"}
    )
    assert result.exit_code == 0, result.output
    assert str(tmp_path / IRWIN / DATE / "209.md") in result.output
    assert "Finalized (bypassed)" in result.output


def test_cli_draft_then_resume_over_sqlite(patched: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pause in one process-like invocation, resume in another: the SqliteSaver file carries the thread."""
    db = str(tmp_path / "checkpoints.sqlite")
    real_get_checkpointer = graph_module.get_checkpointer
    monkeypatch.setattr(graph_module, "get_checkpointer", lambda path=None: real_get_checkpointer(db))
    runner = CliRunner()
    env = {"COLUMNS": "250"}

    paused = runner.invoke(app, ["draft", "--incident", "FAKE", "--date", DATE, "--thread-id", "cli-1"], env=env)
    assert paused.exit_code == 0, paused.output
    assert "Paused for review" in paused.output and "firebinder resume --thread-id cli-1 --approve" in paused.output
    assert "Audit" in paused.output and "Conflicts" in paused.output

    resumed = runner.invoke(app, ["resume", "--thread-id", "cli-1", "--reject"], env=env)
    assert resumed.exit_code == 0, resumed.output
    assert "Finalized (rejected)" in resumed.output
    assert "rejected" in (tmp_path / IRWIN / DATE / "209.md").read_text()
