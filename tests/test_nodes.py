"""Offline tests for the LLM nodes (gather, draft, audit). get_model is monkeypatched with a fake."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from firebinder import llm
from firebinder.nodes import audit as audit_node
from firebinder.nodes import draft as draft_node
from firebinder.nodes.audit import AuditOutput, Finding, audit, to_audit_findings
from firebinder.nodes.draft import Draft, check_citations, draft
from firebinder.nodes.fields import REQUIRED_FIELDS
from firebinder.nodes.gather import REQUIRED_EVIDENCE, gather, missing_fields
from firebinder.state import Evidence
from firebinder.tools import LIVE_TOOLS, firms, inciweb, nws, wfigs
from firebinder.tools.wfigs import IncidentRecord
from tests.test_tools import GRASSHOPPER_CENTROID, GRASSHOPPER_IRWIN, serve  # noqa: F401 (fixture)


class FakeModel:
    """Returns canned responses in order and records every prompt. Stands in for bound/structured models."""

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.prompts: list[list] = []
        self.bound: list[str] = []
        self.structured_kwargs: dict[str, Any] = {}

    def bind_tools(self, tools: list, **_: Any) -> FakeModel:
        self.bound = [t.name for t in tools]
        return self

    def with_structured_output(self, schema: type, **kwargs: Any) -> FakeModel:
        self.structured_kwargs = kwargs
        return self

    def invoke(self, messages: list, *_: Any, **__: Any) -> Any:
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


LAT, LON = GRASSHOPPER_CENTROID
ALL_FOUR = [
    _call("get_incident_record", {"irwin_id": GRASSHOPPER_IRWIN}, "c1"),
    _call("get_hotspots", {"lat": LAT, "lon": LON, "date": "2026-10-26"}, "c2"),
    _call("get_inciweb_updates", {"incident_name": "GRASSHOPPER", "state": "OR"}, "c3"),
    _call("get_weather", {"lat": LAT, "lon": LON}, "c4"),
]


# --- tools: content_and_artifact ---------------------------------------------------------------


def test_tool_wrappers_return_summary_and_artifact(monkeypatch) -> None:
    record = IncidentRecord(
        source="get_incident_record", url="https://wfigs.test/loc", fetched_at="t", irwin_id=GRASSHOPPER_IRWIN,
        incident_name="GRASSHOPPER", irwin_daily_acres=93972, gis_acres=93971.25,
        perimeter_geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        locations_url="https://wfigs.test/loc", perimeters_url="https://wfigs.test/perim",
    )
    monkeypatch.setattr(wfigs, "fetch_incident_record", lambda irwin_id: record)
    tool = LIVE_TOOLS[0]
    message = tool.invoke(ALL_FOUR[0])
    assert isinstance(message, ToolMessage) and message.artifact is record
    assert "93,972" in message.content and "coordinates" not in message.content
    assert tool.invoke({"irwin_id": GRASSHOPPER_IRWIN}) == message.content  # plain args -> summary only


# --- gather -----------------------------------------------------------------------------------


def test_missing_fields() -> None:
    assert missing_fields([]) == REQUIRED_EVIDENCE
    missing = missing_fields([_ev("acres", 10), _ev("forecast_periods", []), _ev("personnel", None)])
    assert "acres" not in missing and "forecast_periods" not in missing
    assert "personnel" in missing  # a null value is not evidence


def test_gather_appends_evidence_from_tool_artifacts(serve, fake) -> None:  # noqa: F811
    serve()
    model = fake(AIMessage(content="", tool_calls=[ALL_FOUR[0]]), AIMessage(content="Done."), AIMessage(content="Done."))
    out = gather(_state())

    assert model.bound == [t.name for t in LIVE_TOOLS] and "get_published_209" not in model.bound
    system = model.prompts[0][0].content
    assert GRASSHOPPER_IRWIN in system and str(LAT) in system and "GRASSHOPPER" in system
    by_field = {e["field"]: e for e in out["evidence"]}
    assert by_field["acres"]["value"] == 93972
    assert by_field["perimeter_geometry"]["value"]["type"] == "MultiPolygon"
    tool_msg = out["messages"][1]
    assert isinstance(tool_msg, ToolMessage) and tool_msg.status == "success"
    assert "93,972" in tool_msg.content and "coordinates" not in tool_msg.content
    assert tool_msg.artifact is None  # artifacts are not stored in state
    # the model declined twice (once after a nudge), so the loop ended
    assert [type(m) for m in out["messages"]] == [AIMessage, ToolMessage, AIMessage, AIMessage]
    assert "Tools not yet tried: get_hotspots" in model.prompts[2][-1].content


class _Stub:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def to_evidence(self) -> list[Evidence]:
        return []

    def summary(self) -> str:
        return "stub"


def _stub_tools(monkeypatch) -> None:
    for module, name in [(wfigs, "fetch_incident_record"), (firms, "fetch_hotspots"),
                         (inciweb, "fetch_inciweb_updates"), (nws, "fetch_weather")]:
        monkeypatch.setattr(module, name, _Stub)


def test_gather_stops_when_every_tool_tried(fake, monkeypatch) -> None:
    _stub_tools(monkeypatch)
    model = fake(AIMessage(content="", tool_calls=ALL_FOUR))  # a second model call would fail (empty list)
    out = gather(_state())
    assert len(model.prompts) == 1
    assert out["evidence"] == []
    assert [m.status for m in out["messages"][1:]] == ["success"] * 4


def test_gather_stops_after_max_iterations(fake, monkeypatch) -> None:
    _stub_tools(monkeypatch)
    model = fake(*[AIMessage(content="", tool_calls=[ALL_FOUR[3] | {"id": f"w{i}"}]) for i in range(10)])
    gather(_state())
    assert len(model.prompts) == 8


def test_gather_tool_exception_becomes_error_message(fake, monkeypatch) -> None:
    monkeypatch.setenv("FIRMS_MAP_KEY", "SECRETKEY123")

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("FIRMS returned 403 for https://firms/api/area/csv/SECRETKEY123/VIIRS")

    monkeypatch.setattr(firms, "fetch_hotspots", boom)
    model = fake(AIMessage(content="", tool_calls=[ALL_FOUR[1]]), AIMessage(content="ok"), AIMessage(content="ok"))
    out = gather(_state())
    tool_msg = out["messages"][1]
    assert tool_msg.status == "error" and "403" in tool_msg.content
    assert "SECRETKEY123" not in tool_msg.content
    assert len(model.prompts) == 3  # the loop continued after the error
    assert "get_hotspots" not in model.prompts[2][-1].content.split("not yet tried:")[1]  # errored = tried


# --- draft ------------------------------------------------------------------------------------

EVIDENCE = [_ev("acres", 93972, "https://wfigs.test/loc"), _ev("pct_contained", 98, "https://wfigs.test/loc")]


def _draft_out(fields: dict, citations: dict) -> Draft:
    return Draft.model_validate(
        {"narrative": "The fire is 93,972 acres.", "fields": dict.fromkeys(REQUIRED_FIELDS) | fields, "citations": citations}
    )


def test_check_citations_numbers_only() -> None:
    fields = {"acres": 93972, "pct_contained": 98.5, "personnel": None, "structures_destroyed": 0,
              "evacuation_status": "Level 3", "red_flag": True, "significant_events": "12 homes"}
    assert check_citations({"fields": fields, "citations": {"acres": "u"}}) == ["pct_contained", "structures_destroyed"]
    assert check_citations({"fields": {"acres": 1}, "citations": {"acres": "u"}}) == []
    assert check_citations({"fields": {}, "citations": {}}) == []


def test_draft_returns_draft_keys_and_keeps_revision_on_first_draft(fake) -> None:
    model = fake(_draft_out({"acres": 93972}, {"acres": "https://wfigs.test/loc", "bogus": "x"}))
    out = draft(_state(evidence=EVIDENCE))
    assert set(out) == {"draft"}
    assert set(out["draft"]) == {"narrative", "fields", "citations"}
    assert list(out["draft"]["fields"]) == REQUIRED_FIELDS
    assert out["draft"]["citations"] == {"acres": "https://wfigs.test/loc"}
    prompt = model.prompts[0][0].content
    assert "[0] acres=93972 (get_incident_record, https://wfigs.test/loc" in prompt
    assert "## Revision" not in prompt


def test_redraft_increments_revision_and_sees_audit(fake) -> None:
    model = fake(_draft_out({"acres": 93972}, {"acres": "https://wfigs.test/loc"}))
    prior = {"narrative": "PRIOR NARRATIVE", "fields": {}, "citations": {}}
    findings = [{"claim": "1,200 personnel", "status": "unsupported", "evidence": []},
                {"claim": "93,972 acres", "status": "supported", "evidence": [EVIDENCE[0]]}]
    out = draft(_state(evidence=EVIDENCE, revision=1, draft=prior, audit=findings))
    assert out["revision"] == 2
    prompt = model.prompts[0][0].content
    assert "PRIOR NARRATIVE" in prompt and "[unsupported] 1,200 personnel" in prompt
    assert "[supported]" not in prompt


def test_draft_reprompt_fixes_uncited(fake) -> None:
    model = fake(_draft_out({"acres": 93972}, {}), _draft_out({"acres": 93972}, {"acres": "https://wfigs.test/loc"}))
    out = draft(_state(evidence=EVIDENCE))
    assert len(model.prompts) == 2 and "Citation check failed" in model.prompts[1][0].content
    assert "acres" in model.prompts[1][0].content.split("no citation:")[1]
    assert out["draft"]["fields"]["acres"] == 93972 and "messages" not in out


def test_draft_drops_still_uncited_numbers(fake) -> None:
    bad = _draft_out({"acres": 93972, "personnel": 120}, {"acres": "https://wfigs.test/loc"})
    fake(bad, bad)
    out = draft(_state(evidence=EVIDENCE))
    assert out["draft"]["fields"]["personnel"] is None and out["draft"]["fields"]["acres"] == 93972
    assert check_citations(out["draft"]) == []
    [message] = out["messages"]
    assert isinstance(message, AIMessage) and "personnel" in message.content


# --- audit ------------------------------------------------------------------------------------


def test_audit_maps_indexes_back_to_evidence(fake) -> None:
    model = fake(
        AuditOutput(
            findings=[
                Finding(claim="93,972 acres", status="supported", evidence_indexes=[0, 0], reason="[0] says so"),
                Finding(claim="40% contained", status="contradicted", evidence_indexes=[1, 99, -1], reason="98%"),
            ]
        )
    )
    conflicts = [{"field": "acres", "values": [EVIDENCE[0]], "severity": "warn", "note": "CONFLICT-NOTE"}]
    draft_ = {"narrative": "93,972 acres, 40% contained.", "fields": {"acres": 93972}, "citations": {}}
    out = audit(_state(evidence=EVIDENCE, draft=draft_, conflicts=conflicts))
    assert set(out) == {"audit"}
    assert out["audit"][0] == {"claim": "93,972 acres", "status": "supported", "evidence": [EVIDENCE[0]]}
    assert out["audit"][1] == {"claim": "40% contained", "status": "contradicted", "evidence": [EVIDENCE[1]]}
    assert model.structured_kwargs.get("method") == "json_schema"
    prompt = model.prompts[0][0].content
    assert "CONFLICT-NOTE" in prompt and "40% contained" in prompt and "[1] pct_contained=98" in prompt


def test_audit_rejects_malformed_status() -> None:
    with pytest.raises(ValidationError):
        Finding(claim="x", status="maybe", evidence_indexes=[], reason="r")
    with pytest.raises(ValidationError):
        to_audit_findings({"findings": [{"claim": "x", "status": "probably", "evidence_indexes": [], "reason": "r"}]}, [])


def test_prompts_load_from_package() -> None:
    for prompt in (draft_node.build_prompt(_state()), audit_node.build_prompt(_state(draft={"narrative": "x"}))):
        assert "{evidence}" not in prompt and "(no evidence was gathered)" in prompt


def test_model_selection(monkeypatch) -> None:
    monkeypatch.delenv("FIREBINDER_MODEL_FAST", raising=False)
    monkeypatch.delenv("FIREBINDER_MODEL_AUDIT", raising=False)
    assert llm.model_name("gather") == llm.model_name("draft") == "claude-haiku-5-5"
    assert llm.model_name("audit") == "claude-opus-5-5"
    monkeypatch.setenv("FIREBINDER_MODEL_AUDIT", "audit-x")
    monkeypatch.setenv("FIREBINDER_MODEL_FAST", "fast-x")
    assert (llm.model_name("draft"), llm.model_name("audit")) == ("fast-x", "audit-x")
