"""gather: ReAct-style tool-calling node over the four live tools (never get_published_209).

The loop is written by hand rather than with ToolNode so each tool's Pydantic result can be
turned into Evidence directly. The model only ever sees a compact JSON summary of each result.

Stops when every required field has evidence, every tool has been tried once, or after
MAX_TURNS model calls, whichever comes first.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from firebinder import llm
from firebinder.nodes.conflicts import _short_message
from firebinder.nodes.fields import FIELD_TO_EVIDENCE, REQUIRED_FIELDS
from firebinder.state import Evidence, FireBinderState
from firebinder.tools import LIVE_TOOLS

MAX_TURNS = 8
SUMMARY_LIMIT = 1500
VALUE_LIMIT = 300

SYSTEM_PROMPT = """You gather public evidence for a wildfire ICS-209 daily status summary.

Incident: {incident_name}
Report date: {report_date}
IRWIN ID: {irwin_id}
Centroid (lat, lon): {centroid}
Mode: {mode}

Call the tools to collect evidence. Use the IRWIN ID for get_incident_record, the centroid for
get_hotspots and get_weather (with the report date for get_hotspots), and the incident name for
get_inciweb_updates. Call each tool at most once; you may call several tools in one turn.
Required fields still lacking evidence: {missing}.
When there is nothing more worth calling, reply with a one-line summary and no tool calls."""


def missing_required_fields(evidence: list[Evidence]) -> list[str]:
    """Required draft fields for which no evidence item (with a non-null value) exists yet."""
    present = {ev["field"] for ev in evidence if ev.get("value") is not None}
    return [f for f in REQUIRED_FIELDS if not any(name in present for name in FIELD_TO_EVIDENCE[f])]


def _compact(field: str, value: Any) -> Any:
    if field == "perimeter_geometry" and isinstance(value, dict):
        return f"<{value.get('type', 'geometry')} omitted>"
    if isinstance(value, list):
        head = [_compact(field, v) for v in value[:2]]
        return {"count": len(value), "first": head} if len(value) > 2 else head
    if isinstance(value, dict):
        text = json.dumps(value, default=str)
        return value if len(text) <= VALUE_LIMIT else text[: VALUE_LIMIT - 1] + "…"
    if isinstance(value, str) and len(value) > VALUE_LIMIT:
        return value[: VALUE_LIMIT - 1] + "…"
    return value


def summarize_evidence(evidence: list[Evidence]) -> str:
    """Compact JSON for the model: field -> value, without full geometry or forecast lists."""
    summary = {
        "source": evidence[0]["source"] if evidence else None,
        "fields": {ev["field"]: _compact(ev["field"], ev["value"]) for ev in evidence},
    }
    text = json.dumps(summary, default=str)
    return text if len(text) <= SUMMARY_LIMIT else text[: SUMMARY_LIMIT - 1] + "…"


def _system_prompt(state: FireBinderState, missing: list[str]) -> str:
    return SYSTEM_PROMPT.format(
        incident_name=state.get("incident_name"),
        report_date=state.get("report_date"),
        irwin_id=state.get("irwin_id"),
        centroid=state.get("centroid"),
        mode=state.get("mode"),
        missing=", ".join(missing) or "none",
    )


def gather(state: FireBinderState) -> dict:
    tools = {t.name: t for t in LIVE_TOOLS}
    model = llm.get_model("gather").bind_tools(LIVE_TOOLS)

    collected: list[Evidence] = []
    produced: list[BaseMessage] = []
    tried: set[str] = set()
    conversation: list[BaseMessage] = [
        SystemMessage(_system_prompt(state, missing_required_fields(state.get("evidence") or []))),
        HumanMessage("Gather the evidence now."),
    ]

    for _ in range(MAX_TURNS):
        missing = missing_required_fields([*(state.get("evidence") or []), *collected])
        if not missing or tried >= set(tools):
            break
        if produced:  # nudge after the first round with what is still open
            untried = sorted(set(tools) - tried)
            conversation.append(
                HumanMessage(f"Still missing: {', '.join(missing)}. Tools not yet tried: {', '.join(untried)}.")
            )

        ai = model.invoke(conversation)
        conversation.append(ai)
        produced.append(ai)
        if not isinstance(ai, AIMessage) or not ai.tool_calls:
            break

        for call in ai.tool_calls:
            name = call["name"]
            tool = tools.get(name)
            if tool is None:
                content, status = f"Error: unknown tool {name!r}. Available: {', '.join(tools)}.", "error"
            else:
                tried.add(name)
                try:
                    result = tool.invoke(call["args"])
                    evidence = result.to_evidence()
                    collected.extend(evidence)
                    content, status = summarize_evidence(evidence), "success"
                except Exception as exc:  # a failing tool is reported to the model and counts as tried
                    content, status = f"Error: {_short_message(exc)}", "error"
            message = ToolMessage(content=content, tool_call_id=call["id"], name=name, status=status)
            conversation.append(message)
            produced.append(message)

    return {"evidence": collected, "messages": produced}
