"""The fixed subset of 209 fields the draft must fill, and which evidence fields satisfy each."""

from __future__ import annotations

REQUIRED_FIELDS: list[str] = [
    "acres",
    "pct_contained",
    "personnel",
    "structures_threatened",
    "structures_destroyed",
    "evacuation_status",
    "significant_events",
    "outlook_12h",
    "outlook_24h",
    "outlook_72h",
]

_OUTLOOK_EVIDENCE = ["forecast_periods", "red_flag_warning", "active_alerts"]

FIELD_TO_EVIDENCE: dict[str, list[str]] = {
    "acres": ["acres", "gis_acres"],
    "pct_contained": ["pct_contained"],
    "personnel": ["personnel"],
    "structures_threatened": ["structures_threatened"],
    "structures_destroyed": ["structures_destroyed"],
    "evacuation_status": ["evacuation_status"],
    "significant_events": ["narrative"],
    "outlook_12h": _OUTLOOK_EVIDENCE,
    "outlook_24h": _OUTLOOK_EVIDENCE,
    "outlook_72h": _OUTLOOK_EVIDENCE,
}
