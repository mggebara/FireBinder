"""audit: the strongest model labels every factual claim in the draft against the evidence.

No tools and no outside knowledge. It sees the draft, the evidence and the conflicts, nothing
else (never the published 209). Evidence is numbered [n] in the prompt, n being the index into
state["evidence"]; the node maps the model's indexes back to Evidence dicts.
"""

import json
from importlib import resources
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from firebinder import llm
from firebinder.nodes.draft import render_conflicts, render_evidence
from firebinder.state import AuditFinding, Evidence, FireBinderState


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="The specific number or short phrase quoted from the draft.")
    status: Literal["supported", "unsupported", "contradicted"]
    evidence_indexes: list[int] = Field(description="[n] numbers of the evidence items that support or contradict it.")
    reason: str = Field(description="One sentence: why this label; for unsupported/contradicted, what is missing or what the evidence says instead.")


class AuditOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding]


def build_prompt(state: FireBinderState) -> str:
    draft = state.get("draft") or {}
    template = resources.files("firebinder").joinpath("prompts/audit.md").read_text(encoding="utf-8")
    return template.format(
        evidence=render_evidence(state.get("evidence") or []),
        conflicts=render_conflicts(state.get("conflicts") or []),
        narrative=draft.get("narrative") or "",
        fields=json.dumps(draft.get("fields") or {}, default=str, indent=1),
        citations=json.dumps(draft.get("citations") or {}, indent=1),
    )


def to_audit_findings(output: AuditOutput | dict, evidence: list[Evidence]) -> list[AuditFinding]:
    """Validate the model output and map evidence indexes back to Evidence; out-of-range ones are dropped."""
    out = output if isinstance(output, AuditOutput) else AuditOutput.model_validate(output)
    return [
        AuditFinding(
            claim=f.claim,
            status=f.status,
            evidence=[evidence[i] for i in dict.fromkeys(f.evidence_indexes) if 0 <= i < len(evidence)],
        )
        for f in out.findings
    ]


def audit(state: FireBinderState, config: RunnableConfig | None = None) -> dict:
    model = llm.get_model("audit").with_structured_output(AuditOutput, method="json_schema")  # strict, constrained decoding
    output = model.invoke([HumanMessage(build_prompt(state))])
    return {"audit": to_audit_findings(output, state.get("evidence") or [])}
