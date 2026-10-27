"""Resolve a user-supplied incident name + date to one WFIGS incident (IRWIN ID and centroid).

Plain function, not a tool. Fails loudly if nothing matches or the match is still ambiguous.
Report dates before WFIGS_FIRST_YEAR resolve against the Historic GeoMAC perimeter layer for
that year instead; the "IRWIN ID" is then the GeoMAC unique fire identifier.
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from firebinder.state import FireBinderState
from firebinder.tools.wfigs import (
    LOCATIONS_QUERY_URL,
    arcgis_query,
    end_of_day_ms,
    epoch_ms_to_iso,
    geomac_query_url,
)

RESOLVE_FIELDS = ["IncidentName", "IrwinID", "POOState", "POOCounty", "FireDiscoveryDateTime", "IncidentSize"]
GEOMAC_RESOLVE_FIELDS = ["OBJECTID", "uniquefireidentifier", "incidentname", "state", "gisacres", "perimeterdatetime"]
WFIGS_FIRST_YEAR = 2020  # the WFIGS current layers hold nothing older; earlier dates go to GeoMAC


class IncidentResolutionError(LookupError):
    """No incident, or more than one, matched the name/date/state."""


class ResolvedIncident(BaseModel):
    irwin_id: str
    incident_name: str
    state: str | None
    centroid: tuple[float, float]  # (lat, lon)
    discovery_date: str | None
    record_source: Literal["wfigs", "geomac"] = "wfigs"


def _normalize_state(state: str | None) -> str | None:
    """'OR', 'or', 'US-OR' -> 'OR'. WFIGS stores POOState as 'US-OR'."""
    if not state:
        return None
    state = state.strip().upper()
    return state.removeprefix("US-")


def _describe(feature: dict) -> str:
    a = feature["attributes"]
    return (
        f"{a.get('IncidentName')} ({a.get('POOState')}, {a.get('POOCounty')} County, "
        f"discovered {epoch_ms_to_iso(a.get('FireDiscoveryDateTime'))}, "
        f"{a.get('IncidentSize')} ac, IRWIN {a.get('IrwinID')})"
    )


def _fail(message: str, candidates: list[dict]) -> IncidentResolutionError:
    listing = "\n".join(f"  - {_describe(f)}" for f in candidates) or "  (none)"
    return IncidentResolutionError(f"{message}\nCandidates:\n{listing}")


def resolve_incident(name: str, date: str, state: str | None = None) -> ResolvedIncident:
    """Find the one WFIGS incident called `name` that was burning on `date` (YYYY-MM-DD).

    Dates before WFIGS_FIRST_YEAR resolve against the Historic GeoMAC layer for that year.
    """
    if int(date[:4]) < WFIGS_FIRST_YEAR:
        return resolve_geomac_incident(name, date, state)
    safe_name = name.strip().replace("'", "''")
    body, _url = arcgis_query(
        LOCATIONS_QUERY_URL,
        {
            "where": f"UPPER(IncidentName) LIKE UPPER('%{safe_name}%')",
            "outFields": ",".join(RESOLVE_FIELDS),
            "returnGeometry": "true",
            "outSR": "4326",
        },
    )
    all_features: list[dict] = body.get("features", [])
    if not all_features:
        raise _fail(f"No WFIGS incident matches name {name!r}.", [])

    candidates = all_features
    exact = [f for f in candidates if (f["attributes"].get("IncidentName") or "").strip().upper() == name.strip().upper()]
    if exact:
        candidates = exact

    wanted_state = _normalize_state(state)
    if wanted_state:
        candidates = [f for f in candidates if _normalize_state(f["attributes"].get("POOState")) == wanted_state]
        if not candidates:
            raise _fail(f"No incident named {name!r} in state {state!r}.", all_features)

    # Must have been discovered on or before the end of the report date.
    end_of_day_ms = (datetime.fromisoformat(date).replace(tzinfo=UTC) + timedelta(days=1)).timestamp() * 1000
    before = [f for f in candidates if (f["attributes"].get("FireDiscoveryDateTime") or 0) < end_of_day_ms]
    before.sort(key=lambda f: f["attributes"].get("FireDiscoveryDateTime") or 0, reverse=True)
    if not before:
        raise _fail(f"No incident named {name!r} was discovered on or before {date}.", candidates)
    if len(before) > 1:
        raise _fail(
            f"{len(before)} incidents match {name!r} on {date}; pass a state or a more exact name.",
            before,
        )

    chosen = before[0]
    attrs = chosen["attributes"]
    geometry = chosen.get("geometry") or {}
    if "x" not in geometry or "y" not in geometry:
        raise _fail(f"Incident {name!r} has no point geometry.", [chosen])
    return ResolvedIncident(
        irwin_id=attrs["IrwinID"],
        incident_name=attrs["IncidentName"],
        state=attrs.get("POOState"),
        centroid=(geometry["y"], geometry["x"]),
        discovery_date=epoch_ms_to_iso(attrs.get("FireDiscoveryDateTime")),
    )


def _describe_geomac(fire_id: str, perimeters: list[dict]) -> str:
    times = [a["perimeterdatetime"] for a in perimeters if a.get("perimeterdatetime")]
    acres = max((a.get("gisacres") or 0 for a in perimeters), default=0)
    first = perimeters[0]
    return (
        f"{first.get('incidentname')} ({first.get('state')}, fire ID {fire_id}, {len(perimeters)} perimeters "
        f"from {epoch_ms_to_iso(min(times, default=None))} to {epoch_ms_to_iso(max(times, default=None))}, "
        f"up to {acres:,.0f} ac)"
    )


def _fail_geomac(message: str, fires: dict[str, list[dict]]) -> IncidentResolutionError:
    listing = "\n".join(f"  - {_describe_geomac(k, v)}" for k, v in fires.items()) or "  (none)"
    return IncidentResolutionError(f"{message}\nCandidates:\n{listing}")


def resolve_geomac_incident(name: str, date: str, state: str | None = None) -> ResolvedIncident:
    """Find the one fire called `name` in the Historic GeoMAC layer for `date`'s year with a perimeter
    dated on or before the end of `date`. Centroid comes from the latest such perimeter."""
    query_url = geomac_query_url(int(date[:4]))
    safe_name = name.strip().replace("'", "''")
    params = {
        "where": f"UPPER(incidentname) LIKE UPPER('%{safe_name}%')",
        "outFields": ",".join(GEOMAC_RESOLVE_FIELDS),
        "returnGeometry": "false",
        "returnCentroid": "true",
        "outSR": "4326",
    }
    body, _url = arcgis_query(query_url, params)
    features: list[dict] = body.get("features", [])
    while body.get("exceededTransferLimit") and body.get("features"):  # a broad name spans pages
        body, _url = arcgis_query(query_url, {**params, "resultOffset": str(len(features))})
        features += body.get("features", [])

    by_fire: dict[str, list[dict]] = defaultdict(list)
    centroids: dict[int, dict] = {}
    for feature in features:
        attrs = feature["attributes"]
        if attrs.get("uniquefireidentifier"):
            by_fire[attrs["uniquefireidentifier"]].append(attrs)
            centroids[attrs["OBJECTID"]] = feature.get("centroid") or {}
    all_fires = dict(by_fire)
    if not all_fires:
        raise _fail_geomac(f"No historic GeoMAC fire matches name {name!r} in {date[:4]}.", {})

    def names(perimeters: list[dict]) -> set[str]:
        return {(a.get("incidentname") or "").strip().upper() for a in perimeters}

    fires = all_fires
    exact = {k: v for k, v in fires.items() if name.strip().upper() in names(v)}
    if exact:
        fires = exact

    wanted_state = _normalize_state(state)
    if wanted_state:
        fires = {k: v for k, v in fires.items() if any(_normalize_state(a.get("state")) == wanted_state for a in v)}
        if not fires:
            raise _fail_geomac(f"No historic fire named {name!r} in state {state!r}.", all_fires)

    end_ms = end_of_day_ms(date)
    before = {
        k: [a for a in v if 0 < (a.get("perimeterdatetime") or 0) < end_ms] for k, v in fires.items()
    }
    before = {k: v for k, v in before.items() if v}
    if not before:
        raise _fail_geomac(f"No historic fire named {name!r} has a perimeter on or before {date}.", fires)
    if len(before) > 1:
        raise _fail_geomac(
            f"{len(before)} historic fires match {name!r} on {date}; pass a state or a more exact name.", before
        )

    fire_id, perimeters = next(iter(before.items()))
    latest = max(perimeters, key=lambda a: (a["perimeterdatetime"], a["OBJECTID"]))
    centroid = centroids.get(latest["OBJECTID"]) or {}
    if "x" not in centroid or "y" not in centroid:
        raise _fail_geomac(f"Historic fire {name!r} has no perimeter centroid.", before)
    return ResolvedIncident(
        irwin_id=fire_id,
        incident_name=latest.get("incidentname") or name,
        state=latest.get("state"),
        centroid=(centroid["y"], centroid["x"]),
        discovery_date=epoch_ms_to_iso(min(a["perimeterdatetime"] for a in all_fires[fire_id] if a.get("perimeterdatetime"))),
        record_source="geomac",
    )


def run_mode(report_date: str, today: str | None = None) -> str:
    """'historical' if the report date is before today (UTC), else 'live'."""
    today = today or datetime.now(UTC).date().isoformat()
    return "historical" if report_date < today else "live"


def resolve(state: FireBinderState, config: RunnableConfig | None = None) -> dict:
    """Graph node. Optional `configurable.state` disambiguates like `firebinder conflicts --state`.

    Sets mode from the report date (spec is silent on how): before today (UTC) is historical, else live.
    Lets IncidentResolutionError propagate: the run should stop if the incident can't be pinned down.
    """
    configurable = (config or {}).get("configurable", {})
    resolved = resolve_incident(state["incident_name"], state["report_date"], state=configurable.get("state"))
    return {
        "irwin_id": resolved.irwin_id,
        "centroid": resolved.centroid,
        "mode": run_mode(state["report_date"]),
    }
