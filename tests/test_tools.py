"""Tool tests against recorded HTTP responses. No live network calls.

Recordings in tests/recorded/ were captured live on 2026-10-26 (GRASSHOPPER fire, OR) except
`firms_handmade.csv` and `nws_alerts_redflag_handmade.json`, which were written by hand
(no FIRMS key was available, and no Red Flag Warning was active anywhere at record time).

Historical mode: `resolve_deluz_2013_12.json`, `geomac_deluz_*` (Historic GeoMAC 2013 layer,
Deluz fire, CA, 2013-10-06: three real perimeters that day), `firms_deluz_*.csv` (VIIRS_SNPP_SP
and MODIS_SP for 2013-10-04..06) and `firms_availability_2026-10-26.csv` were captured live on
2026-10-26.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from shapely.geometry import Polygon, box, shape

from firebinder.nodes.resolve import IncidentResolutionError, resolve_incident
from firebinder.tools import LIVE_TOOLS, _http, firms
from firebinder.tools._base import HistoricalNotAvailable
from firebinder.tools._geo import area_acres
from firebinder.tools.firms import FirmsError, choose_sources, fetch_hotspots, parse_availability_csv, parse_firms_csv
from firebinder.tools.inciweb import fetch_inciweb_updates
from firebinder.tools.nws import fetch_weather
from firebinder.tools.wfigs import (
    WfigsError,
    esri_rings_to_geometry,
    fetch_incident_record,
    geomac_query_url,
    is_geomac_id,
)

RECORDED = Path(__file__).parent / "recorded"
INDEX: dict[str, str] = json.loads((RECORDED / "index.json").read_text())

GRASSHOPPER_IRWIN = "{6AA4C355-AD18-472F-AAAB-78BAA9DE31B1}"
GRASSHOPPER_CENTROID = (45.32482051723143, -121.41757017985736)
DELUZ_ID = "2013-CAMCP-003197"
DELUZ_CENTROID = (33.36324983004942, -117.32180386032918)  # service centroid of the 22:54 perimeter

CONTENT_TYPES = {".json": "application/json", ".xml": "application/xml", ".html": "text/html", ".csv": "text/csv"}

Override = Callable[[httpx.Request], httpx.Response | None]


def _file_response(name: str, status: int = 200) -> httpx.Response:
    path = RECORDED / name
    return httpx.Response(status, content=path.read_bytes(), headers={"content-type": CONTENT_TYPES[path.suffix]})


@pytest.fixture(autouse=True)
def _fresh_firms_availability(monkeypatch) -> None:
    """The availability list is cached per process; start every test without it."""
    monkeypatch.setattr(firms, "_availability", None)


def serve_firms(request: httpx.Request) -> httpx.Response | None:
    """Override for serve(): FIRMS availability plus the recorded Deluz SP CSVs; 400 for anything else."""
    url = str(request.url)
    if "firms.modaps" not in url:
        return None
    if "/data_availability/" in url:
        return _file_response("firms_availability_2026-10-26.csv")
    for source in ("VIIRS_SNPP_SP", "MODIS_SP"):
        if f"/{source}/" in url:
            return _file_response(f"firms_deluz_{source}.csv")
    return httpx.Response(400, text="Invalid source")


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
        if "/data_availability/" in url:
            return _file_response("firms_availability_2026-10-26.csv")
        if "VIIRS_NOAA21_NRT" in url:
            return httpx.Response(400, text="Invalid source")
        # SNPP, NOAA-20 and MODIS all return the same rows, so dedupe must collapse them.
        return _file_response("firms_handmade.csv")

    seen = serve(firms)
    summary = fetch_hotspots(*GRASSHOPPER_CENTROID, "2026-08-12")
    assert len(seen) == 5  # availability + four sources
    assert "/2026-08-10" in seen[1]  # 3-day window ending on the report date
    assert summary.count == 5
    assert summary.sources == ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]  # NOAA-21 failed, skipped
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


def test_choose_sources_by_date() -> None:
    availability = parse_availability_csv((RECORDED / "firms_availability_2026-10-26.csv").read_text())
    assert choose_sources(availability, "2013-10-04", "2013-10-06") == ["VIIRS_SNPP_SP", "MODIS_SP"]
    # Live: NRT only (SP ends months back); SNPP_NRT ends a day before the others but still overlaps.
    assert choose_sources(availability, "2026-10-25", "2026-10-27") == [
        "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT",
    ]
    # Both cover the whole window: SP wins.
    assert choose_sources(availability, "2026-07-10", "2026-07-12") == [
        "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_SP",
    ]
    assert choose_sources(availability, "1999-08-01", "1999-08-03") == []


def test_fetch_hotspots_historical_uses_sp_archive(serve, monkeypatch) -> None:
    monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
    seen = serve(serve_firms)
    summary = fetch_hotspots(*DELUZ_CENTROID, "2013-10-06")
    assert summary.sources == ["VIIRS_SNPP_SP", "MODIS_SP"]
    assert summary.count == 50  # 43 VIIRS + 7 MODIS rows, no duplicates across instruments
    assert summary.latest_detection == "2013-10-06T09:26:00+00:00"
    assert all("testkey" not in u for u in summary.urls)

    fetch_hotspots(*DELUZ_CENTROID, "2013-10-06")
    assert sum("/data_availability/" in u for u in seen) == 1  # fetched once per process


def test_fetch_hotspots_errors_when_no_source_covers(serve, monkeypatch) -> None:
    monkeypatch.setenv("FIRMS_MAP_KEY", "testkey")
    serve(serve_firms)
    with pytest.raises(FirmsError, match="No FIRMS VIIRS or MODIS source covers 1999-07-30 to 1999-08-01"):
        fetch_hotspots(*DELUZ_CENTROID, "1999-08-01")


# --- historical (GeoMAC) -----------------------------------------------------------------------


def test_geomac_ids_and_layer_names() -> None:
    assert is_geomac_id(DELUZ_ID) and is_geomac_id("2013-CASTF-002857")
    assert not is_geomac_id(GRASSHOPPER_IRWIN)
    assert "/Historic_Geomac_Perimeters_2013/" in geomac_query_url(2013)
    assert "/Historic_GeoMAC_Perimeters_2019/" in geomac_query_url(2019)
    with pytest.raises(WfigsError, match="2000 to 2019"):
        geomac_query_url(2021)


def test_resolve_falls_back_to_geomac_for_old_dates(serve) -> None:
    seen = serve()
    resolved = resolve_incident("Deluz", "2013-10-06", "CA")
    assert resolved.irwin_id == DELUZ_ID
    assert resolved.record_source == "geomac"
    assert resolved.incident_name == "Deluz" and resolved.state == "CA"
    assert resolved.centroid == pytest.approx(DELUZ_CENTROID)
    assert resolved.discovery_date == "2013-10-06T16:04:00+00:00"  # earliest perimeter
    assert len(seen) == 1 and "Historic_Geomac_Perimeters_2013" in seen[0]


def test_resolve_geomac_requires_a_perimeter_by_the_date(serve) -> None:
    serve()
    with pytest.raises(IncidentResolutionError, match="on or before 2013-10-05") as excinfo:
        resolve_incident("Deluz", "2013-10-05", "CA")
    assert DELUZ_ID in str(excinfo.value)


def test_historical_record_evidence_includes_same_day_perimeters(serve) -> None:
    serve()
    record = fetch_incident_record(DELUZ_ID, "2013-10-06")
    assert record.record_source == "geomac"
    assert record.incident_name == "Deluz" and record.state == "CA"
    assert record.perimeter_date == "2013-10-06T22:54:00+00:00"  # the latest of the day's three
    assert record.irwin_daily_acres is None and record.pct_contained is None and record.personnel is None

    evidence = record.to_evidence()
    assert [e["field"] for e in evidence] == ["acres", "perimeter_geometry", "perimeter_date", "gis_acres", "gis_acres"]
    for e in evidence:
        assert e["source"] == "get_incident_record" and e["url"].startswith("https://")
    acres, geometry, _date, *others = evidence
    assert acres["value"] == pytest.approx(2708.69, abs=0.01)
    assert geometry["value"]["type"] == "Polygon"
    assert area_acres(shape(geometry["value"])) == pytest.approx(acres["value"], rel=0.01)
    assert [round(e["value"], 2) for e in others] == [2752.51, 2757.5]
    assert [e["url"].split("objectIds=")[1].split("&")[0] for e in others] == ["154", "155"]
    assert len({e["url"] for e in evidence if e["field"] in ("acres", "gis_acres")}) == 3
    assert "2,708.69 GIS acres" in record.summary()


def test_historical_record_requires_a_date() -> None:
    with pytest.raises(WfigsError, match="report date is required"):
        fetch_incident_record(DELUZ_ID)


def test_current_only_tools_refuse_old_dates() -> None:
    # No transport installed: these must fail before any HTTP call.
    with pytest.raises(HistoricalNotAvailable, match="^InciWeb only serves current incidents; no data for 2013-08-22$"):
        fetch_inciweb_updates("Rim", "CA", date="2013-08-22")
    with pytest.raises(HistoricalNotAvailable, match="^NWS only serves current forecasts and alerts; no data for 2013-08-22$"):
        fetch_weather(37.9, -120.0, date="2013-08-22")


def test_current_only_tools_accept_todays_date(serve) -> None:
    today = datetime.now(UTC).date().isoformat()
    serve()
    assert fetch_inciweb_updates("GRASSHOPPER", "OR", date=today).size_acres == 93972
    assert fetch_weather(*GRASSHOPPER_CENTROID, date=today).forecast_office == "PDT"


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
        {"acres", "pct_contained", "personnel", "evacuation_status", "narrative", "inciweb_updated"},
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
