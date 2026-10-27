"""draft: LLM node that writes the 209 narrative and field table from evidence, citing every number.

After the model answers, a cheap Python check (SPEC "Rules the nodes share") looks for numeric
fields without a citation. One re-prompt listing them; if any are still uncited, those fields are
set to None and dropped, and a message says so. The audit never sees an uncited number.
"""

import json
from importlib import resources
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from firebinder import llm
from firebinder.nodes.fields import REQUIRED_FIELDS
from firebinder.state import AuditFinding, Conflict, Evidence, FireBinderState

VALUE_LIMIT = 1200
FORECAST_PERIODS_SHOWN = 7  # ~3.5 days of day/night periods covers the 72h outlook


class DraftFields(BaseModel):
    acres: float | None = Field(description="Fire size in acres, or null")
    pct_contained: float | None = Field(description="Percent contained, 0-100, or null")
    personnel: int | None = Field(description="Total personnel assigned, or null")
    structures_threatened: int | None = Field(description="Structures threatened, or null")
    structures_destroyed: int | None = Field(description="Structures destroyed, or null")
    evacuation_status: str | None = Field(description="Short text, or null")
    significant_events: str | None = Field(description="Short text, or null")
    outlook_12h: str | None = Field(description="Forecast summary for the next 12 hours, or null")
    outlook_24h: str | None = Field(description="Forecast summary for the next 24 hours, or null")
    outlook_72h: str | None = Field(description="Forecast summary for the next 72 hours, or null")


class Draft(BaseModel):
    narrative: str = Field(description="The 209 narrative. Mentions every warn/block conflict explicitly.")
    fields: DraftFields
    citations: dict[str, str] = Field(
        description="Field name -> url of the evidence item it comes from, copied exactly from the evidence list."
    )


assert list(DraftFields.model_fields) == REQUIRED_FIELDS


# ---------- rendering (audit reuses render_evidence and render_conflicts) ----------


def _truncate(text: str, limit: int = VALUE_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _period(p: dict[str, Any]) -> str:
    parts = [str(p.get("name"))]
    if p.get("temperature") is not None:
        parts.append(f"{p['temperature']:g}{p.get('temperature_unit') or ''}")
    if p.get("wind_speed"):
        parts.append(f"wind {p.get('wind_direction') or ''} {p['wind_speed']}".replace("  ", " "))
    if p.get("relative_humidity_min") is not None:
        parts.append(f"min RH {p['relative_humidity_min']:g}%")
    if p.get("short_forecast"):
        parts.append(str(p["short_forecast"]))
    return ", ".join(parts)


def render_value(field: str, value: Any) -> str:
    if field == "perimeter_geometry" and isinstance(value, dict):
        return f"<{value.get('type', 'GeoJSON')} perimeter polygon omitted>"
    if field == "forecast_periods" and isinstance(value, list):
        shown = [_period(p) for p in value[:FORECAST_PERIODS_SHOWN] if isinstance(p, dict)]
        return _truncate(" | ".join(shown), 2 * VALUE_LIMIT)
    if isinstance(value, str):
        return _truncate(" ".join(value.split()))
    return _truncate(json.dumps(value, default=str))


def render_evidence(evidence: list[Evidence]) -> str:
    """Numbered evidence list. [n] is the index into state["evidence"] (0-based)."""
    if not evidence:
        return "(no evidence was gathered)"
    return "\n".join(
        f"[{n}] {ev['field']}={render_value(ev['field'], ev['value'])} ({ev['source']}, {ev['url']}, {ev['fetched_at']})"
        for n, ev in enumerate(evidence)
    )


def render_conflicts(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "(none found)"
    lines = []
    for c in conflicts:
        values = "; ".join(f"{v['source']}={render_value(v['field'], v['value'])}" for v in c["values"])
        lines.append(f"- [{c['severity']}] {c['field']}: {c['note']} ({values})")
    return "\n".join(lines)


def render_revision(prior: dict | None, findings: list[AuditFinding]) -> str:
    flagged = [f for f in findings if f["status"] != "supported"]
    lines = [
        "",
        "## Revision",
        "",
        "An auditor checked your previous draft against the evidence. Rewrite the draft and fix each",
        "finding below: correct every contradicted claim to what the evidence says, and remove every",
        "unsupported claim or replace it with what the evidence actually states. Keep supported",
        "content and the conflict mentions. Return the complete draft.",
        "",
        "Previous draft:",
        json.dumps(prior or {}, default=str, indent=1),
        "",
        "Audit findings to fix:",
    ]
    for f in flagged:
        cited = "; ".join(f"{ev['field']}={render_value(ev['field'], ev['value'])} ({ev['source']})" for ev in f["evidence"])
        lines.append(f"- [{f['status']}] {f['claim']}" + (f" (evidence: {cited})" if cited else ""))
    if not flagged:
        lines.append("- (none flagged)")
    return "\n".join(lines)


def build_prompt(state: FireBinderState) -> str:
    template = resources.files("firebinder").joinpath("prompts/draft.md").read_text(encoding="utf-8")
    revision = ""
    if (state.get("revision") or 0) > 0 or state.get("audit"):
        revision = render_revision(state.get("draft"), state.get("audit") or [])
    return template.format(
        incident_name=state.get("incident_name"),
        report_date=state.get("report_date"),
        irwin_id=state.get("irwin_id"),
        evidence=render_evidence(state.get("evidence") or []),
        conflicts=render_conflicts(state.get("conflicts") or []),
        required_fields=", ".join(REQUIRED_FIELDS),
        revision=revision,
    )


# ---------- the citation check ----------


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def check_citations(draft: dict) -> list[str]:
    """Return the names of numeric fields in draft["fields"] that have no key in draft["citations"]."""
    citations = draft.get("citations") or {}
    return [name for name, value in (draft.get("fields") or {}).items() if _is_number(value) and name not in citations]


def to_draft_dict(output: Draft | dict) -> dict:
    """Validated model output -> {narrative, fields (exactly REQUIRED_FIELDS), citations}."""
    out = output if isinstance(output, Draft) else Draft.model_validate(output)
    fields = out.fields.model_dump()
    citations = {k: v for k, v in out.citations.items() if k in fields and fields[k] is not None and v}
    return {"narrative": out.narrative, "fields": fields, "citations": citations}


def _retry_prompt(prompt: str, rejected: dict, uncited: list[str]) -> str:
    return (
        f"{prompt}\n\n## Citation check failed\n\n"
        f"Your draft was rejected because these numeric fields have no citation: {', '.join(uncited)}.\n\n"
        f"Rejected draft:\n{json.dumps(rejected, default=str, indent=1)}\n\n"
        "Return the complete draft again. Cite each numeric field with the exact url of its evidence "
        "item, or set the field to null if no evidence states it."
    )


def draft(state: FireBinderState, config: RunnableConfig | None = None) -> dict:
    model = llm.get_model("draft").with_structured_output(Draft)
    prompt = build_prompt(state)

    result = to_draft_dict(model.invoke([HumanMessage(prompt)]))
    uncited = check_citations(result)
    if uncited:
        result = to_draft_dict(model.invoke([HumanMessage(_retry_prompt(prompt, result, uncited))]))
        uncited = check_citations(result)

    update: dict[str, Any] = {"draft": result}
    if uncited:
        for name in uncited:
            result["fields"][name] = None
            result["citations"].pop(name, None)
        update["messages"] = [AIMessage(f"draft: dropped uncited numeric fields after one retry: {', '.join(uncited)}")]
    if state.get("audit"):  # a redraft after an audit; the conditional edge cannot write state
        update["revision"] = (state.get("revision") or 0) + 1
    return update
