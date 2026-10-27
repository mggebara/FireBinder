"""Phase-1 orchestration for `firebinder conflicts`: resolve, call the four live tools, reconcile.

Plain function (no graph yet). A failing tool is recorded and the run continues with
whatever evidence the other tools produced.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from pydantic import BaseModel

from firebinder.nodes.reconcile import find_conflicts
from firebinder.nodes.resolve import ResolvedIncident, resolve_incident
from firebinder.state import Conflict, Evidence
from firebinder.tools._base import SourceStamped
from firebinder.tools.firms import fetch_hotspots
from firebinder.tools.inciweb import fetch_inciweb_updates
from firebinder.tools.nws import fetch_weather
from firebinder.tools.wfigs import fetch_incident_record


class ToolError(BaseModel):
    tool: str
    message: str


class ConflictsRun(BaseModel):
    resolved: ResolvedIncident
    evidence: list[Evidence]
    conflicts: list[Conflict]
    errors: list[ToolError]
    tools_called: int


def _short_message(exc: Exception, limit: int = 300) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    if not isinstance(exc, RuntimeError | LookupError | ValueError):
        text = f"{type(exc).__name__}: {text}"
    secret = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if secret:
        text = text.replace(secret, "<FIRMS_MAP_KEY>")  # FIRMS puts the key in the URL path
    if "FIRMS_MAP_KEY" in text and ".env" not in text:
        text += " Set FIRMS_MAP_KEY in .env."
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_conflicts(
    incident: str,
    date: str,
    state: str | None = None,
    radius_km: float = 25,
    day_range: int = 3,
) -> ConflictsRun:
    """Resolve the incident (raises IncidentResolutionError on failure), gather evidence, reconcile."""
    resolved = resolve_incident(incident, date, state)
    lat, lon = resolved.centroid
    calls: dict[str, Callable[[], SourceStamped]] = {
        "get_incident_record": lambda: fetch_incident_record(resolved.irwin_id),
        "get_hotspots": lambda: fetch_hotspots(lat, lon, date, radius_km=radius_km, day_range=day_range),
        "get_inciweb_updates": lambda: fetch_inciweb_updates(resolved.incident_name, state or resolved.state),
        "get_weather": lambda: fetch_weather(lat, lon),
    }

    evidence: list[Evidence] = []
    errors: list[ToolError] = []
    for tool, call in calls.items():
        try:
            evidence.extend(call().to_evidence())
        except Exception as exc:  # one tool failing must not sink the run
            errors.append(ToolError(tool=tool, message=_short_message(exc)))

    return ConflictsRun(
        resolved=resolved,
        evidence=evidence,
        conflicts=find_conflicts(evidence),
        errors=errors,
        tools_called=len(calls),
    )
