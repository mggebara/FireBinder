"""Resolve a user-supplied incident name + date to one WFIGS incident (IRWIN ID and centroid).

Plain function, not a tool. Fails loudly if nothing matches or the match is still ambiguous.
"""

from datetime import UTC, datetime, timedelta

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from firebinder.state import FireBinderState
from firebinder.tools.wfigs import LOCATIONS_QUERY_URL, arcgis_query, epoch_ms_to_iso

RESOLVE_FIELDS = ["IncidentName", "IrwinID", "POOState", "POOCounty", "FireDiscoveryDateTime", "IncidentSize"]


class IncidentResolutionError(LookupError):
    """No incident, or more than one, matched the name/date/state."""


class ResolvedIncident(BaseModel):
    irwin_id: str
    incident_name: str
    state: str | None
    centroid: tuple[float, float]  # (lat, lon)
    discovery_date: str | None


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
    """Find the one WFIGS incident called `name` that was burning on `date` (YYYY-MM-DD)."""
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
