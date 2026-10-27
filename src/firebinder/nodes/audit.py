"""audit: the strongest model labels every factual claim in the draft against the evidence.

No tools. It sees only the draft and the evidence: never conflicts, never the published 209.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from firebinder import llm
from firebinder.nodes.draft import render_evidence
from firebinder.state import AuditFinding, Evidence, FireBinderState


class Finding(BaseModel):
    claim: str = Field(description="The specific number or short phrase from the draft.")
    status: Literal["supported", "unsupported", "contradicted"]
    evidence_indices: list[int] = Field(description="[n] numbers of the evidence items that support or contradict it.")


class AuditOutput(BaseModel):
    findings: list[Finding]


def build_prompt(state: FireBinderState) -> str:
    draft = state.get("draft") or {}
    template = resources.files("firebinder").joinpath("prompts/audit.md").read_text(encoding="utf-8")
    return template.format(
        evidence=render_evidence(state.get("evidence") or []),
        narrative=draft.get("narrative") or "",
        fields=json.dumps(draft.get("fields") or {}, default=str, indent=1),
        citations=json.dumps(draft.get("citations") or {}, indent=1),
    )


def to_audit_findings(output: AuditOutput, evidence: list[Evidence]) -> list[AuditFinding]:
    """Map 1-based evidence indices back to Evidence; out-of-range indices are dropped."""
    return [
        AuditFinding(
            claim=f.claim,
            status=f.status,
            evidence=[evidence[i - 1] for i in dict.fromkeys(f.evidence_indices) if 1 <= i <= len(evidence)],
        )
        for f in output.findings
    ]


def audit(state: FireBinderState) -> dict:
    model = llm.get_model("audit").with_structured_output(AuditOutput, method="json_schema")
    output = model.invoke([HumanMessage(build_prompt(state))])
    return {"audit": to_audit_findings(output, state.get("evidence") or [])}
