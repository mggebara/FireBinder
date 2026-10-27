"""Offline tests for the LLM nodes (gather, draft, audit). get_model is monkeypatched with a fake."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from firebinder import llm
from firebinder.nodes import audit as audit_node
from firebinder.nodes import draft as draft_node
from firebinder.nodes.audit import AuditOutput, Finding, audit
from firebinder.nodes.draft import DraftOutput, UncitedNumberError, check_citations, draft
from firebinder.nodes.fields import REQUIRED_FIELDS
from firebinder.nodes.gather import gather, missing_required_fields
from firebinder.state import Evidence
from firebinder.tools import firms, inciweb, nws, wfigs
from tests.test_tools import GRASSHOPPER_CENTROID, GRASSHOPPER_IRWIN, serve  # noqa: F401 (fixture)


class FakeModel:
    """Returns canned responses in order; records every prompt. Stands in for bound/structured models."""

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.prompts: list[list] = []
        self.bound: list[str] = []

    def bind_tools(self, tools: list, **_: Any) -> FakeModel:
        self.bound = [t.name for t in tools]
        return self

    def with_structured_output(self, schema: type, **_: Any) -> FakeModel:
        return self

    def invoke(self, messages: list) -> Any:
        self.prompts.append(list(messages))
        return self.responses.pop(0)


@pytest.fixture
def fake(monkeypatch):
    def install(*responses: Any) -> FakeModel:
        model = FakeModel(*responses)
        monkeypatch.setattr(llm, "get_model", lambda role: model)
        return model

    return install


def _ev(field: str, value: Any, url: str = "https://example.test/a", source: str = "get_incident_record") -> Evidence:
    return Evidence(field=field, value=value, source=source, url=url, fetched_at="2026-10-26T00:00:00+00:00")


def _state(**overrides: Any) -> dict:
    state = {
        "incident_name": "GRASSHOPPER",
        "report_date": "2026-10-26",
        "irwin_id": GRASSHOPPER_IRWIN,
        "centroid": GRASSHOPPER_CENTROID,
        "mode": "live",
        "evidence": [],
        "conflicts": [],
        "draft": None,
        "audit": [],
        "revision": 0,
    }
    return state | overrides


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# --- gather -----------------------------------------------------------------------------------


def test_missing_required_fields() -> None:
    assert missing_required_fields([]) == REQUIRED_FIELDS
    evidence = [_ev("gis_acres", 10), _ev("forecast_periods", []), _ev("personnel", None)]
    missing = missing_required_fields(evidence)
    assert "acres" not in missing and "outlook_24h" not in missing
    assert "personnel" in missing  # a null value is not evidence


def test_gather_appends_evidence_from_tool_artifacts(serve, fake) -> None:  # noqa: F811
    serve()
    model = fake(
        AIMessage(content="", tool_calls=[_call("get_incident_record", {"irwin_id": GRASSHOPPER_IRWIN}, "c1")]),
        AIMessage(content="Done."),
    )
    out = gather(_state())

    assert "get_published_209" not in model.bound and len(model.bound) == 4
    by_field = {e["field"]: e for e in out["evidence"]}
    assert by_field["acres"]["value"] == 93972
    assert by_field["perimeter_geometry"]["value"]["type"] == "MultiPolygon"
    assert [type(m) for m in out["messages"]] == [AIMessage, ToolMessage, AIMessage]
    tool_msg = out["messages"][1]
    assert tool_msg.status == "success" and len(tool_msg.content) <= 1500
    assert "93972" in tool_msg.content and "coordinates" not in tool_msg.content


class _Stub:
    def to_evidence(self) -> list[Evidence]:
        return []


def test_gather_stops_when_every_tool_tried(fake, monkeypatch) -> None:
    for module, name in [(wfigs, "fetch_incident_record"), (firms, "fetch_hotspots"),
                         (inciweb, "fetch_inciweb_updates"), (nws, "fetch_weather")]:
        monkeypatch.setattr(module, name, lambda *a, **k: _Stub())
    lat, lon = GRASSHOPPER_CENTROID
    model = fake(
        AIMessage(
            content="",
            tool_calls=[
                _call("get_incident_record", {"irwin_id": GRASSHOPPER_IRWIN}, "c1"),
                _call("get_hotspots", {"lat": lat, "lon": lon, "date": "2026-10-26"}, "c2"),
                _call("get_inciweb_updates", {"incident_name": "GRASSHOPPER"}, "c3"),
                _call("get_weather", {"lat": lat, "lon": lon}, "c4"),
            ],
        )
    )  # a second model call would pop from an empty list and fail the test
    out = gather(_state())
    assert len(model.prompts) == 1
    assert out["evidence"] == []
    assert [m.status for m in out["messages"][1:]] == ["success"] * 4


def test_gather_tool_exception_becomes_error_message(fake, monkeypatch) -> None:
    monkeypatch.setenv("FIRMS_MAP_KEY", "SECRETKEY123")

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("FIRMS returned 403 for https://firms/api/area/csv/SECRETKEY123/VIIRS")

    monkeypatch.setattr(firms, "fetch_hotspots", boom)
    lat, lon = GRASSHOPPER_CENTROID
    model = fake(
        AIMessage(content="", tool_calls=[_call("get_hotspots", {"lat": lat, "lon": lon, "date": "2026-10-26"}, "c1")]),
        AIMessage(content="Nothing else to call."),
    )
    out = gather(_state())
    tool_msg = out["messages"][1]
    assert tool_msg.status == "error"
    assert "SECRETKEY123" not in tool_msg.content and "<FIRMS_MAP_KEY>" in tool_msg.content
    assert len(model.prompts) == 2  # the loop continued after the error
    assert "get_hotspots" not in model.prompts[1][-1].content.split("not yet tried:")[1]


# --- draft ------------------------------------------------------------------------------------

EVIDENCE = [_ev("acres", 93972, "https://wfigs.test/loc"), _ev("pct_contained", 98, "https://wfigs.test/loc")]


def _draft_out(fields: dict, citations: dict) -> DraftOutput:
    return DraftOutput(narrative="The fire is 93,972 acres.", fields=fields, citations=citations)


def test_check_citations() -> None:
    good = {"fields": {"acres": 93972, "evacuation_status": "None"}, "citations": {"acres": "https://wfigs.test/loc"}}
    assert check_citations(good, EVIDENCE) == []
    uncited = {"fields": {"acres": 93972, "personnel": "120"}, "citations": {"acres": "https://wfigs.test/loc"}}
    assert check_citations(uncited, EVIDENCE) == ["personnel=120 has no citation"]
    bad_url = {"fields": {"acres": 93972}, "citations": {"acres": "https://made.up/url"}}
    [violation] = check_citations(bad_url, EVIDENCE)
    assert "not an evidence url" in violation


def test_draft_reprompts_once_then_raises(fake) -> None:
    bad = _draft_out({"acres": 93972}, {})
    model = fake(bad, bad)
    with pytest.raises(UncitedNumberError) as err:
        draft(_state(evidence=EVIDENCE))
    assert err.value.violations == ["acres=93972 has no citation"]
    assert len(model.prompts) == 2
    assert "Citation check failed" in model.prompts[1][0].content


def test_draft_reprompt_fixes_and_normalises_fields(fake) -> None:
    good = _draft_out({"acres": 93972, "bogus": 1}, {"acres": "https://wfigs.test/loc"})
    fake(_draft_out({"acres": 93972}, {}), good)
    out = draft(_state(evidence=EVIDENCE))["draft"]
    assert list(out["fields"]) == REQUIRED_FIELDS  # exactly the required keys, missing ones null
    assert out["fields"]["acres"] == 93972 and out["fields"]["personnel"] is None
    assert out["citations"] == {"acres": "https://wfigs.test/loc"}


def test_draft_revision_includes_audit_and_prior_draft(fake) -> None:
    model = fake(_draft_out({"acres": 93972}, {"acres": "https://wfigs.test/loc"}))
    prior = {"narrative": "PRIOR NARRATIVE", "fields": {}, "citations": {}}
    findings = [{"claim": "1,200 personnel", "status": "unsupported", "evidence": []}]
    draft(_state(evidence=EVIDENCE, revision=1, draft=prior, audit=findings))
    prompt = model.prompts[0][0].content
    assert "PRIOR NARRATIVE" in prompt and "[unsupported] 1,200 personnel" in prompt
    assert "[1] acres=93972 (get_incident_record, https://wfigs.test/loc" in prompt

    first = fake(_draft_out({}, {}))
    draft(_state(evidence=EVIDENCE))
    assert "## Revision" not in first.prompts[0][0].content


# --- audit ------------------------------------------------------------------------------------


def test_audit_maps_indices_and_drops_invalid(fake) -> None:
    model = fake(
        AuditOutput(
            findings=[
                Finding(claim="93,972 acres", status="supported", evidence_indices=[1, 1]),
                Finding(claim="40% contained", status="contradicted", evidence_indices=[2, 0, 99]),
            ]
        )
    )
    conflicts = [{"field": "acres", "values": [], "severity": "warn", "note": "CONFLICT-NOTE"}]
    draft_ = {"narrative": "93,972 acres, 40% contained.", "fields": {"acres": 93972}, "citations": {}}
    out = audit(_state(evidence=EVIDENCE, draft=draft_, conflicts=conflicts))["audit"]

    assert out[0]["evidence"] == [EVIDENCE[0]]
    assert out[1] == {"claim": "40% contained", "status": "contradicted", "evidence": [EVIDENCE[1]]}
    prompt = model.prompts[0][0].content
    assert "CONFLICT-NOTE" not in prompt and "40% contained" in prompt


def test_prompts_load_from_package() -> None:
    for prompt in (draft_node.build_prompt(_state()), audit_node.build_prompt(_state(draft={"narrative": "x"}))):
        assert "{evidence}" not in prompt and "(no evidence was gathered)" in prompt
