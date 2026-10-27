"""Tool tests against recorded HTTP responses. No live network calls.

Recordings in tests/recorded/ were captured live on 2026-10-26 (GRASSHOPPER fire, OR) except
`firms_handmade.csv` and `nws_alerts_redflag_handmade.json`, which were written by hand
(no FIRMS key was available, and no Red Flag Warning was active anywhere at record time).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from shapely.geometry import Polygon, box, shape

from firebinder.nodes.resolve import IncidentResolutionError, resolve_incident
from firebinder.tools import LIVE_TOOLS, _http
from firebinder.tools._geo import area_acres
from firebinder.tools.firms import fetch_hotspots, parse_firms_csv
from firebinder.tools.inciweb import fetch_inciweb_updates
from firebinder.tools.nws import fetch_weather
from firebinder.tools.wfigs import esri_rings_to_geometry, fetch_incident_record

RECORDED = Path(__file__).parent / "recorded"
INDEX: dict[str, str] = json.loads((RECORDED / "index.json").read_text())

GRASSHOPPER_IRWIN = "{6AA4C355-AD18-472F-AAAB-78BAA9DE31B1}"
GRASSHOPPER_CENTROID = (45.32482051723143, -121.41757017985736)

CONTENT_TYPES = {".json": "application/json", ".xml": "application/xml", ".html": "text/html", ".csv": "text/csv"}

Override = Callable[[httpx.Request], httpx.Response | None]


def _file_response(name: str, status: int = 200) -> httpx.Response:
    path = RECORDED / name
    return httpx.Response(status, content=path.read_bytes(), headers={"content-type": CONTENT_TYPES[path.suffix]})


@pytest.fixture
def serve() -> Iterator[Callable[..., list[str]]]:
    """serve(override=None) installs a MockTransport that answers from index.json; returns the list of requested URLs."""

    def install(override: Override | None = None) -> list[str]:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            seen.append(url)
            if override is not None:
                response = override(request)
                if response is not None:
                    return response
            if url not in INDEX:
                raise AssertionError(f"No recording for {url}")
            return _file_response(INDEX[url])

        _http.configure(transport=httpx.MockTransport(handler))
        return seen

    yield install
    _http.configure(transport=None)


def _check_evidence(evidence: list[dict], expected_fields: set[str], source: str) -> dict[str, dict]:
    by_field = {e["field"]: e for e in evidence}
    assert set(by_field) == expected_fields
    for e in evidence:
        assert e["source"] == source
        assert e["url"].startswith("https://")
        assert e["fetched_at"].endswith("+00:00")
    return by_field


# --- shared plumbing ---------------------------------------------------------------------------


def test_live_tools_registered() -> None:
    assert [t.name for t in LIVE_TOOLS] == ["get_incident_record", "get_hotspots", "get_inciweb_updates", "get_weather"]


def test_client_sends_user_agent_and_timeout(serve) -> None:
    serve()
    client = _http.get_client()
    assert client.headers["User-Agent"]
    assert client.timeout.read == 15.0


# --- geometry ----------------------------------------------------------------------------------


def test_area_acres_on_one_square_km() -> None:
    lat0 = 45.0
    dlat = 1000 / 110_574
    dlon = 1000 / (111_320 * math.cos(math.radians(lat0)))
    square = box(-121.0, lat0 - dlat / 2, -121.0 + dlon, lat0 + dlat / 2)
    assert area_acres(square) == pytest.approx(247.105, rel=1e-3)


def test_esri_rings_with_hole_become_geojson_polygon() -> None:
    exterior_cw = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]  # clockwise = exterior in ESRI
    hole_ccw = [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]  # counter-clockwise = hole
    second_cw = [[20, 0], [20, 5], [25, 5], [25, 0], [20, 0]]

    single = esri_rings_to_geometry([exterior_cw, hole_ccw])
    assert isinstance(single, Polygon)
    assert len(single.interiors) == 1
    assert single.area == pytest.approx(100 - 4)

    multi = esri_rings_to_geometry([exterior_cw, hole_ccw, second_cw])
    assert multi.geom_type == "MultiPolygon"
    assert multi.area == pytest.approx(96 + 25)


# --- WFIGS -------------------------------------------------------------------------------------


def test_fetch_incident_record_parses_grasshopper(serve) -> None:
    serve()
    record = fetch_incident_record(GRASSHOPPER_IRWIN)
    assert record.incident_name == "GRASSHOPPER"
    assert record.state == "US-OR"
    assert record.county == "Wasco"
    assert record.irwin_daily_acres == 93972
    assert record.gis_acres == pytest.approx(93971.25, abs=0.01)
    assert record.pct_contained == 98
    assert record.discovery_date == "2026-07-24T02:41:00+00:00"
    assert record.perimeter_geometry is not None
    assert record.perimeter_geometry["type"] == "MultiPolygon"
    assert record.centroid == pytest.approx(GRASSHOPPER_CENTROID)
    # Hand-projected polygon area lands within 1% of the service's GIS acres.
    assert area_acres(shape(record.perimeter_geometry)) == pytest.approx(record.gis_acres, rel=0.01)


def test_incident_record_evidence(serve) -> None:
    serve()
    record = fetch_incident_record(GRASSHOPPER_IRWIN)
    by_field = _check_evidence(
        record.to_evidence(),
        {"acres", "gis_acres", "pct_contained", "personnel", "discovery_date", "perimeter_geometry", "perimeter_date"},
        "get_incident_record",
    )
    assert by_field["acres"]["value"] == 93972
    assert "WFIGS_Incident_Locations_Current" in by_field["acres"]["url"]
    assert "WFIGS_Interagency_Perimeters_Current" in by_field["gis_acres"]["url"]
    assert by_field["perimeter_geometry"]["value"]["type"] == "MultiPolygon"


# --- resolve -----------------------------------------------------------------------------------


def test_resolve_picks_state_match_from_multiple_candidates(serve) -> None:
    serve()
    resolved = resolve_incident("Turkey Creek", "2026-10-25", "MS")
    assert resolved.irwin_id == "{0CC3152B-3492-4CC2-8441-51CAE6E7EA95}"
    assert resolved.state == "US-MS"


def test_resolve_uses_discovery_date_when_no_state(serve) -> None:
    serve(lambda r: _file_response("resolve_turkey_creek_03.json") if "LIKE" in str(r.url) else None)
    # Only the Louisiana fire (discovered 2026-10-08) existed on 2026-10-10; Mississippi's started 10-20.
    resolved = resolve_incident("Turkey Creek", "2026-10-10")
    assert resolved.state == "US-LA"
    assert resolved.irwin_id == "{291CAB79-6CE0-4E88-94A4-6AE8DBE887DD}"


def test_resolve_fails_loudly_when_ambiguous(serve) -> None:
    serve(lambda r: _file_response("resolve_turkey_creek_03.json") if "LIKE" in str(r.url) else None)
    with pytest.raises(IncidentResolutionError, match="2 incidents match") as excinfo:
        resolve_incident("Turkey Creek", "2026-10-25")
    assert "US-LA" in str(excinfo.value) and "US-MS" in str(excinfo.value)


# --- FIRMS (hand-made CSV) ---------------------------------------------------------------------


def test_parse_firms_csv() -> None:
    detections = parse_firms_csv((RECORDED / "firms_handmade.csv").read_text())
    assert len(detections) == 5
    assert detections[0].acq_time == "0942"
    assert detections[2].frp == 48.9
    assert detections[4].satellite == "N20"


def test_fetch_hotspots_dedupes_and_computes_hull(serve, monkeypatch) -> None:
    monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")

    def firms(request: httpx.Request) -> httpx.Response | None:
        url = str(request.url)
        if "firms.modaps" not in url:
            return None
        if "VIIRS_NOAA21_NRT" in url:
            return httpx.Response(400, text="Invalid source")
        # SNPP and NOAA-20 both return the same rows, so dedupe must collapse them.
        return _file_response("firms_handmade.csv")

    seen = serve(firms)
    summary = fetch_hotspots(*GRASSHOPPER_CENTROID, "2026-08-12")
    assert len(seen) == 3
    assert "/2026-08-10" in seen[0]  # 3-day window ending on the report date
    assert summary.count == 5
    assert summary.sources == ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
    assert summary.max_frp == 48.9
    assert summary.latest_detection == "2026-08-12T02:18:00+00:00"
    # Hull is the 0.01 x 0.01 degree square of the four corner detections.
    expected = box(-121.42, 45.32, -121.41, 45.33)
    assert summary.hull_acres == pytest.approx(area_acres(expected))
    assert summary.hull_acres == pytest.approx(0.01 * 111_320 * math.cos(math.radians(45.325)) * 0.01 * 110_574 / 4046.8564224, rel=1e-6)
    assert all("testkey" not in u for u in summary.urls)

    by_field = _check_evidence(
        summary.to_evidence(), {"hotspot_count", "hotspot_hull_acres", "max_frp", "latest_detection"}, "get_hotspots"
    )
    assert by_field["hotspot_count"]["value"] == 5


def test_fetch_hotspots_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIRMS_MAP_KEY"):
        fetch_hotspots(45.0, -121.0, "2026-08-12")


# --- InciWeb -----------------------------------------------------------------------------------


def test_fetch_inciweb_updates_parses_grasshopper(serve) -> None:
    serve()
    update = fetch_inciweb_updates("GRASSHOPPER", "OR")
    assert update.page_url == "https://inciweb.wildfire.gov/incident-information/ormhf-grasshopper-fire"
    assert update.incident_name == "Grasshopper Fire"
    assert update.size_acres == 93972
    assert update.pct_contained == 98
    assert update.personnel == 218
    assert update.current_as_of == "2026-10-06T15:02:00-07:00"
    assert update.evacuation_text and "evacuation" in update.evacuation_text.lower()
    assert update.updates[0].published == "2026-09-30T13:11:25+00:00"
    assert update.narrative and "Acres Burned: 93,971 acres" in update.narrative
    assert "final daily update" in update.narrative

    _check_evidence(
        update.to_evidence(),
        {"acres", "pct_contained", "evacuation_status", "narrative", "inciweb_updated"},
        "get_inciweb_updates",
    )


# --- NWS ---------------------------------------------------------------------------------------


def test_fetch_weather_parses_forecast(serve) -> None:
    serve()
    weather = fetch_weather(*GRASSHOPPER_CENTROID)
    assert weather.forecast_office == "PDT"
    assert len(weather.periods) == 6
    first = weather.periods[0]
    assert first.name and first.temperature is not None and first.wind_speed
    assert first.relative_humidity_min is not None  # filled from the hourly forecast
    assert weather.alerts == []
    assert weather.red_flag_warning is False

    by_field = _check_evidence(
        weather.to_evidence(), {"forecast_periods", "red_flag_warning", "active_alerts"}, "get_weather"
    )
    assert len(by_field["forecast_periods"]["value"]) == 6
    assert by_field["active_alerts"]["value"] == []


def test_fetch_weather_flags_red_flag_warning(serve) -> None:
    serve(lambda r: _file_response("nws_alerts_redflag_handmade.json") if "/alerts/active" in str(r.url) else None)
    weather = fetch_weather(*GRASSHOPPER_CENTROID)
    assert weather.red_flag_warning is True
    assert weather.fire_weather_watch is False
    events = {e["field"]: e["value"] for e in weather.to_evidence()}
    assert events["red_flag_warning"] is True
    assert events["active_alerts"] == ["Red Flag Warning", "Air Quality Alert"]
