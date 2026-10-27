"""Unit tests for the reconcile node's tolerance rules and polygon-vs-reported-acres check."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from firebinder.nodes.reconcile import find_conflicts, polygon_area_acres, reconcile
from firebinder.state import Evidence

FIXTURES = sorted((Path(__file__).parent.parent / "evals" / "fixtures").glob("*.json"))

IRWIN = ("get_incident_record", "https://irwin.example/incident")
PERIM = ("get_incident_record", "https://wfigs.example/perimeter")
INCIWEB = ("get_inciweb_updates", "https://inciweb.example/fire")


def ev(field: str, value: Any, origin: tuple[str, str] = IRWIN) -> Evidence:
    return Evidence(field=field, value=value, source=origin[0], url=origin[1], fetched_at="2014-07-20T00:00:00Z")


def pair(field: str, a: Any, b: Any) -> list[Evidence]:
    return [ev(field, a, IRWIN), ev(field, b, INCIWEB)]


def found(evidence: list[Evidence]) -> set[tuple[str, str]]:
    return {(c["field"], c["severity"]) for c in find_conflicts(evidence)}


def square(lon: float, lat: float, side_m: float) -> list[list[float]]:
    """Closed ring of a side_m x side_m square centred on (lon, lat), using the reconcile projection."""
    dlat = side_m / 110_574 / 2
    dlon = side_m / (111_320 * math.cos(math.radians(lat))) / 2
    return [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat], [lon + dlon, lat + dlat],
            [lon - dlon, lat + dlat], [lon - dlon, lat - dlat]]


# ---------- acreage ----------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (9_000, 10_000, set()),  # spread 1,000 == 10% of 10,000: at tolerance, no conflict
        (9_000, 10_001, {("acres", "warn")}),  # just beyond 10%
        (1_000, 1_500, set()),  # spread 500 == 500-acre floor
        (1_000, 1_501, {("acres", "warn")}),  # just beyond the floor
        (2_000, 2_400, set()),  # small fire: 10% (240) would flag, 500 floor wins
        (50_000, 54_000, set()),  # large fire: floor (500) would flag, 10% (5,400) wins
        (50_000, 55_600, {("acres", "warn")}),  # large fire beyond 10% (5,560)
    ],
)
def test_acreage_tolerance_boundaries(a: float, b: float, expected: set[tuple[str, str]]) -> None:
    assert found(pair("acres", a, b)) == expected


@pytest.mark.parametrize(
    ("a", "b", "severity"),
    [
        (1_000, 2_500, "warn"),  # spread 1,500 == 3x tol 500: not beyond
        (1_000, 2_501, "block"),
        (10_000, 13_300, "warn"),  # tol 1,330, 3x = 3,990
        (10_000, 25_000, "block"),  # tol 2,500, spread 15,000 > 7,500
    ],
)
def test_acreage_block_threshold(a: float, b: float, severity: str) -> None:
    assert found(pair("acres", a, b)) == {("acres", severity)}


def test_acres_and_gis_acres_are_one_group() -> None:
    evidence = [ev("acres", 10_000, IRWIN), ev("gis_acres", 12_500, PERIM)]
    conflicts = find_conflicts(evidence)
    assert [(c["field"], c["severity"]) for c in conflicts] == [("acres", "warn")]
    assert conflicts[0]["values"] == evidence


def test_acreage_emits_one_conflict_with_all_values_and_correct_note() -> None:
    evidence = [ev("acres", 94_120, IRWIN), ev("gis_acres", 95_000, PERIM), ev("acres", 110_000, INCIWEB)]
    conflicts = find_conflicts(evidence)
    assert len(conflicts) == 1
    assert conflicts[0]["values"] == evidence
    assert conflicts[0]["note"] == (
        "IRWIN daily 94,120 ac vs InciWeb 110,000 ac: spread 15,880 ac exceeds tolerance 11,000 ac"
    )


# ---------- containment ----------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (30, 35, set()),
        (30, 36, {("pct_contained", "warn")}),
        (30, 50, {("pct_contained", "warn")}),  # exactly 20 points: warn, not block
        (30, 51, {("pct_contained", "block")}),
    ],
)
def test_containment(a: float, b: float, expected: set[tuple[str, str]]) -> None:
    assert found(pair("pct_contained", a, b)) == expected


# ---------- counts and text ----------


@pytest.mark.parametrize("field", ["personnel", "structures_threatened", "structures_destroyed"])
def test_counts_must_match_exactly(field: str) -> None:
    assert found(pair(field, 850, 850)) == set()
    assert found(pair(field, 850, 851)) == {(field, "warn")}


def test_evacuation_text_normalised_then_info() -> None:
    assert found(pair("evacuation_status", "Level 2  (Be Set)", "  level 2 (be set) ")) == set()
    assert found(pair("evacuation_status", "Level 2 (Be Set)", "Level 3 (Go Now)")) == {("evacuation_status", "info")}


def test_narrative_and_unknown_fields_ignored() -> None:
    assert found(pair("narrative", "a", "b") + pair("hotspot_count", 3, 400) + pair("perimeter_date", "x", "y")) == set()


# ---------- origins and bad values ----------


def test_single_origin_never_conflicts() -> None:
    evidence = [
        ev("acres", 1_000), ev("acres", 90_000), ev("pct_contained", 0), ev("pct_contained", 100),
        ev("personnel", 1), ev("personnel", 2), ev("evacuation_status", "none"), ev("evacuation_status", "Go now"),
    ]
    assert find_conflicts(evidence) == []


def test_none_and_non_numeric_values_skipped() -> None:
    evidence = [
        ev("acres", 10_000, IRWIN), ev("acres", None, INCIWEB), ev("acres", "lots", PERIM), ev("acres", True, PERIM),
        ev("pct_contained", float("nan"), INCIWEB), ev("pct_contained", 10, IRWIN),
        ev("personnel", None, INCIWEB), ev("personnel", 5, IRWIN), ev("evacuation_status", None, INCIWEB),
        ev("evacuation_status", "Level 1", IRWIN),
    ]
    assert find_conflicts(evidence) == []


def test_empty_evidence() -> None:
    assert find_conflicts([]) == []


# ---------- polygon area ----------


def test_polygon_area_of_one_square_km() -> None:
    area = polygon_area_acres({"type": "Polygon", "coordinates": [square(-120.0, 44.0, 1_000)]})
    assert area == pytest.approx(247.105, rel=0.01)


def test_multipolygon_with_hole() -> None:
    outer = square(-120.0, 44.0, 1_000)
    hole = square(-120.0, 44.0, 500)[::-1]
    other = square(-119.9, 44.0, 1_000)
    geom = {"type": "MultiPolygon", "coordinates": [[outer, hole], [other]]}
    # 1 km^2 - 0.25 km^2 + 1 km^2 = 1.75 km^2
    assert polygon_area_acres(geom) == pytest.approx(1.75 * 247.105, rel=0.01)


def polygon_evidence(reported: list[tuple[str, float, tuple[str, str]]], side_m: float = 2_011.7) -> list[Evidence]:
    """A ~1,000-acre square perimeter (2,011.7 m side) plus the given (field, value, origin) claims."""
    geom = {"type": "Polygon", "coordinates": [square(-120.0, 44.0, side_m)]}
    return [ev("perimeter_geometry", geom, PERIM)] + [ev(f, v, o) for f, v, o in reported]


def test_polygon_vs_acres_within_tolerance() -> None:
    assert found(polygon_evidence([("acres", 1_400, IRWIN)])) == set()


def test_polygon_vs_acres_warn_and_synthetic_evidence() -> None:
    conflicts = find_conflicts(polygon_evidence([("acres", 1_800, IRWIN)]))
    assert [(c["field"], c["severity"]) for c in conflicts] == [("polygon_area_vs_acres", "warn")]
    values = conflicts[0]["values"]
    assert values[0]["value"] == 1_800
    synthetic = values[-1]
    assert synthetic["field"] == "polygon_area_acres"
    assert synthetic["source"] == "reconcile"
    assert synthetic["url"] == PERIM[1]
    assert synthetic["value"] == pytest.approx(1_000, rel=0.01)
    assert "1,800 ac" in conflicts[0]["note"] and "tolerance 500 ac" in conflicts[0]["note"]


def test_polygon_vs_acres_block() -> None:
    assert found(polygon_evidence([("acres", 3_000, IRWIN)])) == {("polygon_area_vs_acres", "block")}


def test_polygon_checks_inciweb_acres_but_not_gis_acres() -> None:
    # gis_acres is the same polygon's own area, so it is never compared to the polygon
    assert found(polygon_evidence([("gis_acres", 3_000, PERIM)])) == set()
    assert found(polygon_evidence([("acres", 3_000, INCIWEB)])) == {("polygon_area_vs_acres", "block")}


def test_no_polygon_conflict_without_geometry_or_acres() -> None:
    assert found(polygon_evidence([])) == set()
    assert found([ev("acres", 3_000, IRWIN)]) == set()


# ---------- node contract and fixtures ----------


def test_reconcile_returns_only_conflicts_key() -> None:
    state: Any = {"incident_name": "X", "evidence": pair("acres", 1_000, 5_000), "revision": 0}
    result = reconcile(state)
    assert set(result) == {"conflicts"}
    assert [(c["field"], c["severity"]) for c in result["conflicts"]] == [("acres", "block")]


def test_fixtures_exist() -> None:
    assert len(FIXTURES) >= 7


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_fixture_expected_conflicts(path: Path) -> None:
    case = json.loads(path.read_text())
    expected = {(c["field"], c["severity"]) for c in case["expected_conflicts"]}
    conflicts = find_conflicts(case["evidence"])
    assert len(conflicts) == len(expected)
    assert found(case["evidence"]) == expected
