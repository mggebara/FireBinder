"""reconcile: pure-Python conflict detection over gathered evidence. No LLM, no HTTP.

Rules (SPEC.md, "Node contracts: reconcile"):
- acres / gis_acres: one group; tolerance max(10% of the larger value, 500 ac);
  warn outside tolerance, block beyond 3x tolerance.
- pct_contained: tolerance 5 points; warn outside, block beyond 20 points.
- personnel, structures_threatened, structures_destroyed: exact match, else warn.
- evacuation_status: differing text (whitespace/case normalised) -> info.
- perimeter_geometry area vs each "acres" claim (not gis_acres): acreage tolerance.
- Only evidence from distinct (source, url) origins is compared; one origin never conflicts.
- None / non-numeric values in numeric fields are skipped.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Literal

from shapely.geometry import shape

from firebinder.state import Conflict, Evidence, FireBinderState
from firebinder.tools._geo import area_acres

Severity = Literal["info", "warn", "block"]

ACREAGE_FIELDS = ("acres", "gis_acres")
ACREAGE_MIN_TOLERANCE = 500.0
ACREAGE_PCT_TOLERANCE = 0.10
ACREAGE_BLOCK_MULTIPLE = 3.0

CONTAINMENT_TOLERANCE = 5.0
CONTAINMENT_BLOCK = 20.0

COUNT_FIELDS = ("personnel", "structures_threatened", "structures_destroyed")
TEXT_FIELDS = ("evacuation_status",)

SOURCE_LABELS = {
    ("get_incident_record", "acres"): "IRWIN daily",
    ("get_incident_record", "gis_acres"): "WFIGS GIS",
    ("get_inciweb_updates", "acres"): "InciWeb",
    ("get_incident_record", None): "IRWIN",
    ("get_inciweb_updates", None): "InciWeb",
    ("reconcile", "polygon_area_acres"): "perimeter polygon",
}


# ---------- small helpers ----------


def _origin(ev: Evidence) -> tuple[str, str]:
    return (ev["source"], ev["url"])


def _label(ev: Evidence) -> str:
    key = (ev["source"], ev["field"])
    if key in SOURCE_LABELS:
        return SOURCE_LABELS[key]
    return SOURCE_LABELS.get((ev["source"], None), ev["source"])


def _number(value: Any) -> float | None:
    """Return value as float if it is a real finite number, else None (skipped)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _numeric(evidence: list[Evidence], fields: tuple[str, ...]) -> list[tuple[Evidence, float]]:
    out: list[tuple[Evidence, float]] = []
    for ev in evidence:
        if ev["field"] in fields:
            n = _number(ev["value"])
            if n is not None:
                out.append((ev, n))
    return out


def _cross_origin_pairs(
    items: list[tuple[Evidence, float]],
) -> list[tuple[tuple[Evidence, float], tuple[Evidence, float]]]:
    return [(a, b) for a, b in combinations(items, 2) if _origin(a[0]) != _origin(b[0])]


def _fmt(n: float) -> str:
    return f"{n:,.0f}" if abs(n - round(n)) < 0.05 else f"{n:,.1f}"


def acreage_tolerance(a: float, b: float) -> float:
    """max(10% of the larger value, 500 acres)."""
    return max(ACREAGE_PCT_TOLERANCE * max(a, b), ACREAGE_MIN_TOLERANCE)


def acreage_severity(spread: float, tolerance: float) -> Severity | None:
    if spread > ACREAGE_BLOCK_MULTIPLE * tolerance:
        return "block"
    if spread > tolerance:
        return "warn"
    return None


# ---------- rules ----------


def _acreage_conflict(evidence: list[Evidence]) -> Conflict | None:
    items = _numeric(evidence, ACREAGE_FIELDS)
    worst: tuple[float, float, float, Evidence, Evidence, float, float] | None = None
    for (ea, a), (eb, b) in _cross_origin_pairs(items):
        spread = abs(a - b)
        tol = acreage_tolerance(a, b)
        ratio = spread / tol
        if worst is None or ratio > worst[0]:
            worst = (ratio, spread, tol, ea, eb, a, b)
    if worst is None:
        return None
    _, spread, tol, ea, eb, a, b = worst
    severity = acreage_severity(spread, tol)
    if severity is None:
        return None
    note = (
        f"{_label(ea)} {_fmt(a)} ac vs {_label(eb)} {_fmt(b)} ac: "
        f"spread {_fmt(spread)} ac exceeds tolerance {_fmt(tol)} ac"
    )
    if severity == "block":
        note += f" by more than {ACREAGE_BLOCK_MULTIPLE:g}x"
    return Conflict(field="acres", values=[ev for ev, _ in items], severity=severity, note=note)


def _containment_conflict(evidence: list[Evidence]) -> Conflict | None:
    items = _numeric(evidence, ("pct_contained",))
    worst: tuple[float, Evidence, Evidence, float, float] | None = None
    for (ea, a), (eb, b) in _cross_origin_pairs(items):
        spread = abs(a - b)
        if worst is None or spread > worst[0]:
            worst = (spread, ea, eb, a, b)
    if worst is None or worst[0] <= CONTAINMENT_TOLERANCE:
        return None
    spread, ea, eb, a, b = worst
    severity: Severity = "block" if spread > CONTAINMENT_BLOCK else "warn"
    limit = CONTAINMENT_BLOCK if severity == "block" else CONTAINMENT_TOLERANCE
    note = (
        f"{_label(ea)} {_fmt(a)}% vs {_label(eb)} {_fmt(b)}% contained: "
        f"spread {_fmt(spread)} points exceeds {_fmt(limit)} points"
    )
    return Conflict(
        field="pct_contained", values=[ev for ev, _ in items], severity=severity, note=note
    )


def _count_conflict(evidence: list[Evidence], field: str) -> Conflict | None:
    items = _numeric(evidence, (field,))
    mismatches = [(x, y) for x, y in _cross_origin_pairs(items) if x[1] != y[1]]
    if not mismatches:
        return None
    (ea, a), (eb, b) = mismatches[0]
    note = f"{field}: {_label(ea)} reports {_fmt(a)} but {_label(eb)} reports {_fmt(b)}"
    return Conflict(field=field, values=[ev for ev, _ in items], severity="warn", note=note)


def _normalise_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).lower()
    return text or None


def _text_conflict(evidence: list[Evidence], field: str) -> Conflict | None:
    items = [
        (ev, t) for ev in evidence if ev["field"] == field
        if (t := _normalise_text(ev["value"])) is not None
    ]
    for (ea, a), (eb, b) in combinations(items, 2):
        if _origin(ea) != _origin(eb) and a != b:
            note = f"{field}: {_label(ea)} says {ea['value']!r} but {_label(eb)} says {eb['value']!r}"
            return Conflict(field=field, values=[ev for ev, _ in items], severity="info", note=note)
    return None


def polygon_area_acres(geojson: dict[str, Any]) -> float:
    """Area in acres of a GeoJSON Polygon/MultiPolygon dict in lon/lat."""
    return area_acres(shape(geojson))


def _polygon_conflicts(evidence: list[Evidence]) -> list[Conflict]:
    acres = _numeric(evidence, ("acres",))
    geoms = [ev for ev in evidence if ev["field"] == "perimeter_geometry" and isinstance(ev["value"], dict)]
    if not acres or not geoms:
        return []
    conflicts: list[Conflict] = []
    now = datetime.now(timezone.utc).isoformat()
    for geom_ev in geoms:
        area = polygon_area_acres(geom_ev["value"])
        synthetic = Evidence(
            field="polygon_area_acres", value=area, source="reconcile", url=geom_ev["url"], fetched_at=now
        )
        disagreeing: list[tuple[Evidence, float, float, float]] = []  # (ev, value, spread, tol)
        for ev, value in acres:
            spread = abs(area - value)
            tol = acreage_tolerance(area, value)
            if acreage_severity(spread, tol) is not None:
                disagreeing.append((ev, value, spread, tol))
        if not disagreeing:
            continue
        ev, value, spread, tol = max(disagreeing, key=lambda d: d[2] / d[3])
        severity = acreage_severity(spread, tol)
        assert severity is not None
        note = (
            f"Perimeter polygon area {_fmt(area)} ac vs {_label(ev)} {_fmt(value)} ac: "
            f"spread {_fmt(spread)} ac exceeds tolerance {_fmt(tol)} ac"
        )
        if severity == "block":
            note += f" by more than {ACREAGE_BLOCK_MULTIPLE:g}x"
        conflicts.append(
            Conflict(
                field="polygon_area_vs_acres",
                values=[d[0] for d in disagreeing] + [synthetic],
                severity=severity,
                note=note,
            )
        )
    return conflicts


# ---------- public API ----------


def find_conflicts(evidence: list[Evidence]) -> list[Conflict]:
    """Apply every tolerance rule to the evidence and return the conflicts found."""
    conflicts: list[Conflict] = []
    for found in (_acreage_conflict(evidence), _containment_conflict(evidence)):
        if found:
            conflicts.append(found)
    for field in COUNT_FIELDS:
        if found := _count_conflict(evidence, field):
            conflicts.append(found)
    for field in TEXT_FIELDS:
        if found := _text_conflict(evidence, field):
            conflicts.append(found)
    conflicts.extend(_polygon_conflicts(evidence))
    return conflicts


def reconcile(state: FireBinderState) -> dict[str, list[Conflict]]:
    """Graph node: returns only the key it changes."""
    return {"conflicts": find_conflicts(state.get("evidence", []))}
