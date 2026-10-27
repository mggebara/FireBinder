"""human_review: pause the run with interrupt() so a person can approve, reject, or edit the draft."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from firebinder.state import AuditFinding, FireBinderState

MAX_REVISIONS = 3
AUDIT_GATE = "audit_gate"  # name on the message that flags a draft sent to review still failing audit


def audit_failing(audit: list[AuditFinding]) -> bool:
    """Spec rule: any contradicted claim, or more than one unsupported claim, fails the audit."""
    statuses = [finding["status"] for finding in audit]
    return "contradicted" in statuses or statuses.count("unsupported") > 1


def _gate_flag(state: FireBinderState) -> list[AIMessage]:
    if not audit_failing(state.get("audit", [])):
        return []
    note = f"Audit still failing after {state.get('revision', 0)} revision(s); sent to review flagged."
    return [AIMessage(content=note, name=AUDIT_GATE)]


def human_review(state: FireBinderState, config: RunnableConfig) -> dict[str, Any]:
    """Resume payloads: {"action": "approve"} | {"action": "reject"} | {"action": "edit", "draft": {...}}.

    `configurable.skip_review` (historical eval runs) approves without interrupting.
    """
    flag = _gate_flag(state)
    if config.get("configurable", {}).get("skip_review"):
        return {"approved": True, "messages": flag}

    decision = interrupt(
        {
            "draft": state.get("draft"),
            "conflicts": state.get("conflicts", []),
            "audit": state.get("audit", []),
            "revision": state.get("revision", 0),
            "flagged": bool(flag),
        }
    )
    action = (decision or {}).get("action")
    if action == "approve":
        return {"approved": True, "messages": flag}
    if action == "reject":
        return {"approved": False, "messages": flag}
    if action == "edit":
        draft = decision.get("draft")
        if not isinstance(draft, dict):
            raise ValueError("edit resume needs a 'draft' dict")
        return {"draft": draft, "approved": True, "messages": flag}
    raise ValueError(f"unknown review action {action!r}; expected approve, reject, or edit")
