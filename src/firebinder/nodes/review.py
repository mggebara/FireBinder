"""human_review: pause the run with interrupt() so a person can approve, reject, or edit the draft."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from firebinder.state import AuditFinding, FireBinderState

MAX_REVISIONS = 3
FAILING_FLAG = "audit still failing after 3 revisions"
DRAFT_KEYS = ("narrative", "fields", "citations")


def audit_failing(audit: list[AuditFinding]) -> bool:
    """Spec rule: any contradicted claim, or more than one unsupported claim, fails the audit."""
    statuses = [finding["status"] for finding in audit]
    return "contradicted" in statuses or statuses.count("unsupported") > 1


def review_flag(state: FireBinderState) -> str | None:
    """Set when the revision cap was hit and the audit still fails."""
    if state.get("revision", 0) >= MAX_REVISIONS and audit_failing(state.get("audit", [])):
        return FAILING_FLAG
    return None


def _validate_draft(draft: Any) -> dict:
    if not isinstance(draft, dict) or any(key not in draft for key in DRAFT_KEYS):
        raise ValueError(f"edited draft must be a dict with keys {', '.join(DRAFT_KEYS)}")
    return draft


def human_review(state: FireBinderState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Resume values: {"action": "approve"} | {"action": "reject"} | {"action": "edit", "draft": {...}}.

    `configurable.skip_review` (historical eval runs) bypasses the interrupt and leaves approved = None.
    """
    if (config or {}).get("configurable", {}).get("skip_review"):
        return {"approved": None}

    decision = interrupt(
        {
            "draft": state.get("draft"),
            "conflicts": state.get("conflicts", []),
            "audit": state.get("audit", []),
            "revision": state.get("revision", 0),
            "flag": review_flag(state),
        }
    )
    action = (decision or {}).get("action")
    if action == "approve":
        return {"approved": True}
    if action == "reject":
        return {"approved": False}
    if action == "edit":
        return {"approved": True, "draft": _validate_draft(decision.get("draft"))}
    raise ValueError(f"unknown review action {action!r}; expected approve, reject, or edit")
