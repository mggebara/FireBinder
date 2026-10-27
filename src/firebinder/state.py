"""Graph state and shared record types."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class Evidence(TypedDict):
    field: str  # e.g. "acres", "pct_contained"
    value: Any
    source: str  # tool name
    url: str
    fetched_at: str  # ISO


class Conflict(TypedDict):
    field: str
    values: list[Evidence]  # the disagreeing evidence
    severity: Literal["info", "warn", "block"]
    note: str


class AuditFinding(TypedDict):
    claim: str  # a sentence or number from the draft
    status: Literal["supported", "unsupported", "contradicted"]
    evidence: list[Evidence]


class FireBinderState(TypedDict):
    incident_name: str
    report_date: str
    irwin_id: str | None
    centroid: tuple[float, float] | None
    mode: Literal["historical", "live"]
    evidence: Annotated[list[Evidence], operator.add]
    conflicts: list[Conflict]
    draft: dict | None  # {narrative: str, fields: dict[str, Any], citations: dict[str, str]}
    audit: list[AuditFinding]
    revision: int
    approved: bool | None
    messages: Annotated[list, add_messages]
