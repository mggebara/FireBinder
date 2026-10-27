"""CLI tests for `firebinder conflicts`, served from recorded responses (no live network)."""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

from rich.console import Console

from firebinder.cli import _render, app, format_value
from firebinder.nodes.conflicts import ConflictsRun, ToolError, _short_message
from firebinder.nodes.resolve import ResolvedIncident
from firebinder.tools import firms
from tests.test_tools import GRASSHOPPER_CENTROID, RECORDED, serve, serve_firms  # noqa: F401  (serve is a fixture)

runner = CliRunner()
ARGS = ["conflicts", "--incident", "GRASSHOPPER", "--date", "2026-10-26", "--state", "OR"]


def _resolve_from_location_record(request: httpx.Request) -> httpx.Response | None:
    """resolve_incident's name search was not recorded; answer it with the recorded GRASSHOPPER location,
    with the point moved to the centroid the NWS recordings were captured at."""
    if "LIKE" in str(request.url):
        body = json.loads((RECORDED / "wfigs_grasshopper_01.json").read_text())
        lat, lon = GRASSHOPPER_CENTROID
        body["features"][0]["geometry"] = {"x": lon, "y": lat}
        return httpx.Response(200, json=body)
    return None


def _no_match(request: httpx.Request) -> httpx.Response | None:
    if "LIKE" in str(request.url):
        return httpx.Response(200, json={"features": []})
    return None


def test_conflicts_runs_with_firms_error(serve, monkeypatch) -> None:  # noqa: F811
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    serve(_resolve_from_location_record)
    result = runner.invoke(app, ARGS, env={"COLUMNS": "250"})  # wide enough that cells do not wrap
    assert result.exit_code == 0, result.output
    out = result.output
    assert "GRASSHOPPER" in out and "{6AA4C355-AD18-472F-AAAB-78BAA9DE31B1}" in out
    assert "Tool errors" in out
    assert "get_hotspots" in out and "FIRMS_MAP_KEY" in out and ".env" in out
    assert "Evidence" in out
    assert "MultiPolygon (" in out  # geometry is summarised, not dumped
    # the recorded GRASSHOPPER sources disagree on personnel: WFIGS 85 vs InciWeb 218
    assert "No conflicts found." not in out
    assert "personnel" in out and "85" in out and "218" in out


def test_conflicts_json_output(serve, monkeypatch) -> None:  # noqa: F811
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    serve(_resolve_from_location_record)
    result = runner.invoke(app, [*ARGS, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload) == {"resolved", "evidence", "conflicts", "errors"}
    assert payload["resolved"]["irwin_id"] == "{6AA4C355-AD18-472F-AAAB-78BAA9DE31B1}"
    assert [e["tool"] for e in payload["errors"]] == ["get_hotspots"]
    geometry = next(e["value"] for e in payload["evidence"] if e["field"] == "perimeter_geometry")
    assert geometry["type"] == "MultiPolygon" and geometry["coordinates"]


def test_conflicts_historical_day(serve, monkeypatch) -> None:  # noqa: F811
    """A 2013 day: GeoMAC record + FIRMS SP archive; InciWeb and NWS refuse the old date."""
    monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
    monkeypatch.setattr(firms, "_availability", None)
    serve(serve_firms)
    result = runner.invoke(app, ["conflicts", "--incident", "Deluz", "--date", "2013-10-06", "--state", "CA", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["resolved"]["irwin_id"] == "2013-CAMCP-003197"
    assert payload["resolved"]["record_source"] == "geomac"
    errors = {e["tool"]: e["message"] for e in payload["errors"]}
    assert errors == {
        "get_inciweb_updates": "InciWeb only serves current incidents; no data for 2013-10-06",
        "get_weather": "NWS only serves current forecasts and alerts; no data for 2013-10-06",
    }
    fields = [e["field"] for e in payload["evidence"]]
    assert fields.count("gis_acres") == 2 and "acres" in fields and "hotspot_count" in fields


def test_conflicts_resolution_failure_exits_1(serve) -> None:  # noqa: F811
    serve(_no_match)
    result = runner.invoke(app, ARGS)
    assert result.exit_code == 1
    assert "No WFIGS incident matches name 'GRASSHOPPER'" in result.output


def test_format_value_truncates() -> None:
    assert format_value({"type": "Polygon", "coordinates": [[[0, 0]]]}) == "Polygon (1 part)"
    assert len(format_value("x" * 500)) == 80
    assert len(format_value([{"a": "b" * 200}])) == 80


def _ev(field: str, value: object, source: str) -> dict:
    return {"field": field, "value": value, "source": source, "url": f"https://example.test/{source}", "fetched_at": "2026-10-26T00:00:00+00:00"}


def test_render_conflicts_table() -> None:
    run = ConflictsRun(
        resolved=ResolvedIncident(irwin_id="{X}", incident_name="TEST", state="US-OR", centroid=(45.0, -121.0), discovery_date=None),
        evidence=[_ev("acres", 1000, "get_incident_record"), _ev("acres", 5000, "get_inciweb_updates")],
        conflicts=[
            {
                "field": "acres",
                "values": [_ev("acres", 1000, "get_incident_record"), _ev("acres", 5000, "get_inciweb_updates")],
                "severity": "block",
                "note": "IRWIN daily 1000 vs InciWeb 5000",
            }
        ],
        errors=[ToolError(tool="get_weather", message="boom")],
        tools_called=4,
    )
    console = Console(record=True, width=200)
    _render(console, run)
    out = console.export_text()
    assert "Conflicts" in out and "block" in out
    assert "get_incident_record=1000" in out and "get_inciweb_updates=5000" in out
    assert "Tool errors" in out and "get_weather: boom" in out


def test_tool_error_message_hides_firms_key(monkeypatch) -> None:
    monkeypatch.setenv("FIRMS_MAP_KEY", "sekrit123")
    message = _short_message(RuntimeError("404 for https://firms/api/area/csv/sekrit123/VIIRS"))
    assert "sekrit123" not in message and "<FIRMS_MAP_KEY>" in message
