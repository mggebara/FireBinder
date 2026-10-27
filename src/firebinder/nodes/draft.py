"""draft: LLM node that writes the 209 narrative and field table from evidence, citing every number.

After the model answers, a cheap Python check (SPEC "Rules the nodes share") rejects any numeric
field without a citation to a url present in the evidence. One re-prompt, then UncitedNumberError.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from firebinder import llm
from firebinder.nodes.fields import REQUIRED_FIELDS
from firebinder.state import AuditFinding, Conflict, Evidence, FireBinderState

VALUE_LIMIT = 1200
FORECAST_PERIODS_SHOWN = 7  # ~3.5 days of day/night periods covers the 72h outlook


class DraftOutput(BaseModel):
    narrative: str = Field(description="The 209 narrative. Mentions every warn/block conflict explicitly.")
    fields: dict[str, Any] = Field(
        description=f"Exactly these keys: {', '.join(REQUIRED_FIELDS)}. Null when there is no evidence."
    )
    citations: dict[str, str] = Field(
        description="Field name -> url of the evidence item it comes from, copied exactly from the evidence list."
    )


class UncitedNumberError(ValueError):
    """The draft still had numeric fields without a valid citation after one re-prompt."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("Draft has uncited numbers: " + "; ".join(violations))


# ---------- rendering ----------


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
    """Numbered evidence list, 1-based. audit uses the same numbering for evidence_indices."""
    if not evidence:
        return "(no evidence was gathered)"
    return "\n".join(
        f"[{n}] {ev['field']}={render_value(ev['field'], ev['value'])} ({ev['source']}, {ev['url']}, {ev['fetched_at']})"
        for n, ev in enumerate(evidence, start=1)
    )


def render_conflicts(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "(none found)"
    return "\n".join(f"- [{c['severity']}] {c['field']}: {c['note']}" for c in conflicts)


def render_revision(prior: dict | None, findings: list[AuditFinding]) -> str:
    flagged = [f for f in findings if f["status"] != "supported"]
    lines = [
        "",
        "## Revision",
        "",
        "An auditor checked your previous draft against the evidence. Rewrite the draft: remove or",
        "correct every contradicted claim, and remove every unsupported claim or replace it with what",
        "the evidence actually says. Keep supported content. Return the complete draft.",
        "",
        "Previous draft:",
        json.dumps(prior or {}, default=str, indent=1),
        "",
        "Audit findings to fix:",
    ]
    lines += [f"- [{f['status']}] {f['claim']}" for f in flagged] or ["- (none flagged)"]
    return "\n".join(lines)


def build_prompt(state: FireBinderState) -> str:
    template = resources.files("firebinder").joinpath("prompts/draft.md").read_text(encoding="utf-8")
    revision = ""
    if (state.get("revision") or 0) > 0:
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
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
    if isinstance(value, str):
        try:
            float(value.replace(",", "").strip())
        except ValueError:
            return False
        return True
    return False


def check_citations(draft: dict, evidence: list[Evidence]) -> list[str]:
    """Violations: numeric fields with no citation, or citing a url that is not in the evidence."""
    urls = {ev["url"] for ev in evidence}
    citations = draft.get("citations") or {}
    violations: list[str] = []
    for field, value in (draft.get("fields") or {}).items():
        if not _is_number(value):
            continue
        url = citations.get(field)
        if not url:
            violations.append(f"{field}={value} has no citation")
        elif url not in urls:
            violations.append(f"{field}={value} cites {url}, which is not an evidence url")
    return violations


def _to_draft(out: DraftOutput) -> dict:
    fields = {name: out.fields.get(name) for name in REQUIRED_FIELDS}
    citations = {k: v for k, v in out.citations.items() if k in fields and fields[k] is not None}
    return {"narrative": out.narrative, "fields": fields, "citations": citations}


def draft(state: FireBinderState) -> dict:
    evidence = state.get("evidence") or []
    model = llm.get_model("draft").with_structured_output(DraftOutput)
    prompt = build_prompt(state)

    result = _to_draft(model.invoke([HumanMessage(prompt)]))
    violations = check_citations(result, evidence)
    if violations:
        retry = (
            f"{prompt}\n\n## Citation check failed\n\nYour draft was rejected:\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\n\nRejected draft:\n"
            + json.dumps(result, default=str, indent=1)
            + "\n\nReturn the complete draft again. Cite each numeric field with an exact evidence url, "
            "or set the field to null if no evidence supports it."
        )
        result = _to_draft(model.invoke([HumanMessage(retry)]))
        violations = check_citations(result, evidence)
        if violations:
            raise UncitedNumberError(violations)
    return {"draft": result}
