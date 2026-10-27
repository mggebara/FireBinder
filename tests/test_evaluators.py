"""Offline tests for evals/evaluators.py and evals/run_eval.py. The judge is a fake; nothing hits LangSmith."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from evals import evaluators as ev
from evals import run_eval
from firebinder import llm
from firebinder.state import Evidence
from tests.test_nodes import FakeModel


def _e(field: str, value: Any, source: str = "get_incident_record") -> Evidence:
    return Evidence(field=field, value=value, source=source, url="https://example.test/a", fetched_at="2014-07-20T18:00:00+00:00")


def _conflict(field: str, severity: str = "warn") -> dict:
    return {"field": field, "values": [], "severity": severity, "note": ""}


def _by_key(result: dict) -> dict[str, dict]:
    return {r["key"]: r for r in result["results"]}


@pytest.fixture
def judge(monkeypatch):
    def install(*responses: Any) -> FakeModel:
        model = FakeModel(*responses)
        roles: list[str] = []
        monkeypatch.setattr(llm, "get_model", lambda role: roles.append(role) or model)
        model.roles = roles  # type: ignore[attr-defined]
        return model

    return install


# ---------- judge model ----------


def test_judge_defaults_to_a_model_other_than_the_drafter(monkeypatch):
    monkeypatch.delenv("FIREBINDER_MODEL_JUDGE", raising=False)
    monkeypatch.delenv("FIREBINDER_MODEL_FAST", raising=False)
    assert llm.model_name("judge") == llm.DEFAULT_JUDGE_MODEL != llm.model_name("draft")


def test_judge_equal_to_drafter_is_refused(monkeypatch):
    monkeypatch.setenv("FIREBINDER_MODEL_FAST", "claude-x")
    monkeypatch.setenv("FIREBINDER_MODEL_JUDGE", "claude-x")
    with pytest.raises(ValueError, match="drafting model"):
        llm.model_name("judge")


# ---------- conflict_recall ----------


def test_conflict_recall_full():
    known = [{"field": "acres", "severity": "warn"}, {"field": "personnel", "severity": "warn"}]
    out = _by_key(ev.conflict_recall({}, {"conflicts": [_conflict("acres", "block"), _conflict("personnel")]}, {"known_conflicts": known}))
    assert out["conflict_recall"]["score"] == 1.0
    assert out["conflict_false_positives"]["score"] == 0
    assert "1/2 with matching severity" in out["conflict_recall"]["comment"]
    assert "acres (known warn, flagged block)" in out["conflict_recall"]["comment"]


def test_conflict_recall_partial_with_false_positive():
    known = [{"field": "acres", "severity": "warn"}, {"field": "pct_contained", "severity": "block"}]
    flagged = [_conflict("acres"), _conflict("polygon_area_vs_acres", "info")]
    out = _by_key(ev.conflict_recall({}, {"conflicts": flagged}, {"known_conflicts": known}))
    assert out["conflict_recall"]["score"] == 0.5
    assert "Missed: pct_contained (block)" in out["conflict_recall"]["comment"]
    assert out["conflict_false_positives"]["score"] == 1
    assert "polygon_area_vs_acres (info)" in out["conflict_false_positives"]["comment"]


def test_conflict_recall_empty_known_no_flags():
    out = _by_key(ev.conflict_recall({}, {"conflicts": []}, {"known_conflicts": []}))
    assert out["conflict_recall"]["score"] == 1.0
    assert out["conflict_false_positives"]["score"] == 0


def test_conflict_recall_empty_known_with_flags_counts_false_positives():
    out = _by_key(ev.conflict_recall({}, {"conflicts": [_conflict("acres"), _conflict("personnel")]}, {"known_conflicts": []}))
    assert out["conflict_recall"]["score"] == 1.0
    assert out["conflict_false_positives"]["score"] == 2


# ---------- faithfulness ----------


def test_faithfulness_recomputes_score_from_claims(judge):
    verdict = ev.FaithfulnessVerdict(
        claims=[
            ev.Claim(claim="acres = 12000", supported=True, why="[0]"),
            ev.Claim(claim="pct_contained = 35", supported=True, why="[1]"),
            ev.Claim(claim="Highway 20 closed", supported=False, why="no evidence mentions a closure"),
        ],
        score=0.9,  # the judge's arithmetic is wrong; ours wins
    )
    model = judge(verdict)
    outputs = {"draft": {"narrative": "Highway 20 closed.", "fields": {"acres": 12000}}, "evidence": [_e("acres", 12000)]}
    res = ev.faithfulness({}, outputs, {"narrative": "SECRET REFERENCE"})
    assert res["key"] == "faithfulness"
    assert res["score"] == pytest.approx(2 / 3)
    assert "Highway 20 closed (no evidence mentions a closure)" in res["comment"]
    assert model.roles == ["judge"]
    prompt = model.prompts[0][0].content
    assert "[0] acres=12000" in prompt  # the audit's evidence rendering
    assert "SECRET REFERENCE" not in prompt


def test_faithfulness_no_claims_and_no_draft(judge):
    judge(ev.FaithfulnessVerdict(claims=[], score=0.0))
    assert ev.faithfulness({}, {"draft": {"narrative": "", "fields": {}}, "evidence": []}, {})["score"] == 1.0
    assert ev.faithfulness({}, {"draft": None}, {})["score"] is None


# ---------- vs_published_209: tolerance and field table ----------


@pytest.mark.parametrize(
    ("field", "draft", "ref", "expected"),
    [
        ("acres", 1500, 1000, True),  # 500 ac floor
        ("acres", 1501, 1000, False),
        ("acres", 11000, 10000, True),  # 10% of the larger value = 1100
        ("acres", 8800, 10000, False),  # 10% of 10000 = 1000 < 1200
        ("pct_contained", 55, 50, True),
        ("pct_contained", 55.5, 50, False),
        ("personnel", 100.0, 100, True),
        ("personnel", 101, 100, False),
        ("structures_destroyed", 0, 0, True),
        ("structures_threatened", None, 5, False),
        ("structures_threatened", 5, None, None),
        ("structures_threatened", None, None, None),
    ],
)
def test_within_tolerance(field, draft, ref, expected):
    assert ev.within_tolerance(field, draft, ref) is expected


def test_field_table_and_fields_within_tolerance():
    draft = {"fields": {"acres": 12000, "pct_contained": 35, "personnel": 850, "structures_threatened": None, "structures_destroyed": 4}}
    reference = {"fields": {"acres": 12500, "pct_contained": 45, "personnel": 850, "structures_threatened": None, "structures_destroyed": 3}}
    table = ev.field_table(draft["fields"], reference["fields"])
    assert [c["within_tolerance"] for c in table] == [True, False, True, None, False]
    res = ev.fields_within_tolerance({}, {"draft": draft}, reference)
    assert res["key"] == "fields_within_tolerance"
    assert res["score"] == pytest.approx(2 / 4)
    assert "| pct_contained | 35 | 45 | NO |" in res["comment"]
    assert ev.fields_within_tolerance({}, {"draft": draft}, {"fields": {}})["score"] is None


def test_vs_published_209_keeps_raw_score_and_code_table(judge):
    verdict = ev.Vs209Verdict(
        field_checks=[],
        missed_events=["evacuations lifted for Sisters"],
        invented_events=[],
        score=4,
        rationale="Covers the main runs.",
    )
    model = judge(verdict)
    outputs = {"draft": {"narrative": "Fire grew east.", "fields": {"acres": 12000}}}
    reference = {"fields": {"acres": 20000}, "narrative": "Fire grew east. Evacuations lifted for Sisters."}
    res = ev.vs_published_209({"incident_name": "X", "report_date": "2014-07-20"}, outputs, reference)
    assert res["key"] == "vs_published_209" and res["score"] == 4
    assert "| acres | 12000 | 20000 | NO |" in res["comment"]
    assert "evacuations lifted for Sisters" in res["comment"]
    prompt = model.prompts[0][0].content
    assert "Evacuations lifted for Sisters." in prompt and "| acres | 12000 | 20000 | NO |" in prompt


def test_cost_metrics_read_outputs():
    assert ev.total_tokens({}, {"total_tokens": 1234}, {}) == {"key": "total_tokens", "score": 1234}
    assert ev.wall_seconds({}, {"wall_seconds": 3.5}, {}) == {"key": "wall_seconds", "score": 3.5}


# ---------- run_eval ----------


def test_planted_evidence_lands_in_initial_state():
    planted = [_e("acres", 14500, source="get_inciweb_updates")]
    state = run_eval.build_initial_state({"incident_name": "EXAMPLE", "report_date": "2014-07-20", "state": "OR"}, planted)
    assert state["evidence"] == planted and state["incident_name"] == "EXAMPLE" and state["revision"] == 0
    assert run_eval.build_initial_state({"incident_name": "E", "report_date": "2014-07-20"}, None)["evidence"] == []


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

    def invoke(self, state: dict, config: dict) -> dict:
        self.calls.append((state, config))
        geometry = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        return {
            **state,
            "evidence": state["evidence"] + [_e("perimeter_geometry", geometry, source="get_perimeter")],
            "conflicts": [_conflict("acres")],
            "draft": {"narrative": "n", "fields": {}, "citations": {}},
            "revision": 1,
        }


def test_target_injects_planted_evidence_by_example_id(monkeypatch):
    from langsmith import traceable
    from langsmith.run_helpers import tracing_context

    import firebinder.nodes.resolve as resolve_mod

    monkeypatch.setattr(resolve_mod, "run_mode", lambda date: "historical")
    example_id = uuid.uuid4()
    planted = [_e("acres", 14500, source="get_inciweb_updates")]
    graph = FakeGraph()
    target = traceable(run_eval.make_target(graph, {str(example_id): planted}))
    inputs = {"incident_name": "EXAMPLE", "report_date": "2014-07-20", "state": "OR"}
    with tracing_context(enabled="local"):
        out = target(inputs, langsmith_extra={"reference_example_id": example_id})

    state, config = graph.calls[0]
    assert state["evidence"] == planted
    assert config["configurable"]["skip_review"] is True and config["configurable"]["state"] == "OR"
    assert uuid.UUID(config["configurable"]["thread_id"])
    assert set(out) == {"draft", "conflicts", "audit", "evidence", "revision", "total_tokens", "wall_seconds"}
    assert out["evidence"][-1]["value"] == "<Polygon perimeter polygon omitted>"
    assert out["total_tokens"] == 0 and out["wall_seconds"] >= 0 and out["revision"] == 1


def test_feedback_means_and_row_formatting(tmp_path):
    class Res:
        def __init__(self, key: str, score: float | None):
            self.key, self.score = key, score

    rows = [
        {"evaluation_results": {"results": [Res("faithfulness", 1.0), Res("conflict_recall", 1.0), Res("total_tokens", 1000)]}},
        {"evaluation_results": {"results": [Res("faithfulness", 0.5), Res("conflict_recall", None), Res("total_tokens", 3000)]}},
    ]
    n, means = run_eval.feedback_means(rows)
    assert n == 2 and means == {"faithfulness": 0.75, "conflict_recall": 1.0, "total_tokens": 2000}
    row = run_eval.format_row("2026-10-26", "abc1234", n, means, "audit | v2")
    assert row == "| 2026-10-26 | abc1234 | 2 | 0.75 | 1.00 | n/a | n/a | n/a | 2,000 | n/a | audit \\| v2 |"
    path = tmp_path / "results.md"
    run_eval.append_row(row, path)
    run_eval.append_row(row, path)
    lines = path.read_text().splitlines()
    assert lines[2].startswith("| date | git sha | n |") and lines[-1] == lines[-2] == row
    assert lines[2].count("|") == row.count("|") - 1  # the escaped pipe in the note is not a column
