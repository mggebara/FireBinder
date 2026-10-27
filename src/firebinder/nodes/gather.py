"""gather: ReAct-style tool-calling node over the four live tools (never get_published_209).

The loop is written by hand rather than with ToolNode so each tool's Pydantic artifact can be
turned into Evidence directly. The tools use response_format="content_and_artifact": the model
only sees the short text summary, the node keeps the artifact.

Stops when every required field has evidence, every tool has been tried once, or after
MAX_ITERATIONS model calls, whichever comes first.
"""

from importlib import resources

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from firebinder import llm
from firebinder.nodes.conflicts import _short_message  # redacts FIRMS_MAP_KEY from errors
from firebinder.state import Evidence, FireBinderState
from firebinder.tools import LIVE_TOOLS
from firebinder.tools._base import SourceStamped

MAX_ITERATIONS = 8

# Evidence field names (see nodes/reconcile.py) that the draft needs at least one source for.
REQUIRED_EVIDENCE: list[str] = [
    "acres",
    "pct_contained",
    "personnel",
    "structures_threatened",
    "structures_destroyed",
    "evacuation_status",
    "narrative",  # significant events
    "forecast_periods",  # 12/24/72-hour outlook
]


def missing_fields(evidence: list[Evidence]) -> list[str]:
    """Required evidence fields with no non-null value yet."""
    present = {ev["field"] for ev in evidence if ev.get("value") is not None}
    return [field for field in REQUIRED_EVIDENCE if field not in present]


def system_prompt(state: FireBinderState, config: RunnableConfig | None) -> str:
    template = resources.files("firebinder").joinpath("prompts/gather.md").read_text(encoding="utf-8")
    lat, lon = state.get("centroid") or (None, None)
    us_state = ((config or {}).get("configurable") or {}).get("state")
    return template.format(
        incident_name=state.get("incident_name"),
        us_state=us_state or "unknown (omit the state argument)",
        irwin_id=state.get("irwin_id"),
        lat=lat,
        lon=lon,
        report_date=state.get("report_date"),
        mode=state.get("mode"),
        required_fields=", ".join(REQUIRED_EVIDENCE),
        missing=", ".join(missing_fields(state.get("evidence") or [])) or "none",
    )


def run_tool_call(
    tools: dict[str, BaseTool], call: ToolCall, report_date: str | None = None
) -> tuple[ToolMessage, list[Evidence]]:
    """Execute one tool call. Errors become a status="error" ToolMessage; the loop carries on.

    If the model left out `report_date` on a tool that takes one, the run's report date is filled
    in: a historical run must never silently get today's data.
    """
    name, call_id = call["name"], call["id"] or ""
    tool = tools.get(name)
    if tool is None:
        content = f"Error: unknown tool {name!r}. Available tools: {', '.join(tools)}."
        return ToolMessage(content=content, tool_call_id=call_id, name=name, status="error"), []
    if report_date and "report_date" in tool.args and not call["args"].get("report_date"):
        call = {**call, "args": {**call["args"], "report_date": report_date}}
    try:
        message = tool.invoke(call)
    except Exception as exc:  # a failing tool is reported to the model and counts as tried
        return ToolMessage(content=f"Error: {_short_message(exc)}", tool_call_id=call_id, name=name, status="error"), []
    artifact = getattr(message, "artifact", None)
    evidence = artifact.to_evidence() if isinstance(artifact, SourceStamped) else []
    # Keep state and checkpoints small and plain: the evidence is stored separately.
    stored = ToolMessage(content=message.content, tool_call_id=call_id, name=name, status=message.status)
    return stored, evidence


def gather(state: FireBinderState, config: RunnableConfig | None = None) -> dict:
    tools = {t.name: t for t in LIVE_TOOLS}
    model = llm.get_model("gather").bind_tools(LIVE_TOOLS)

    conversation: list[BaseMessage] = [
        SystemMessage(system_prompt(state, config)),
        HumanMessage("Gather the evidence now."),
    ]
    produced: list[BaseMessage] = []
    collected: list[Evidence] = []
    tried: set[str] = set()
    nudged = False

    for _ in range(MAX_ITERATIONS):
        if not missing_fields(collected) or tried >= set(tools):
            break
        ai = model.invoke(conversation)
        conversation.append(ai)
        produced.append(ai)

        if not isinstance(ai, AIMessage) or not ai.tool_calls:
            if nudged:  # the model declined twice; stop rather than loop
                break
            nudged = True
            untried = ", ".join(name for name in tools if name not in tried)
            missing = ", ".join(missing_fields(collected))
            conversation.append(HumanMessage(f"Still missing: {missing}. Tools not yet tried: {untried}."))
            continue

        nudged = False
        for call in ai.tool_calls:
            message, evidence = run_tool_call(tools, call, state.get("report_date"))
            if call["name"] in tools:
                tried.add(call["name"])
            collected.extend(evidence)
            conversation.append(message)
            produced.append(message)

    return {"evidence": collected, "messages": produced}
