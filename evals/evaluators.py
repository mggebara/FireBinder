"""LangSmith evaluators for FireBinder (SPEC "LangSmith: tracing, datasets, evaluators").

Each evaluator takes (inputs, outputs, reference_outputs) and returns a feedback dict
({"key", "score", "comment"}) or {"results": [...]} for several keys, as langsmith.evaluate() wants.

- faithfulness        LLM judge: fraction of draft claims supported by the run's own evidence.
- conflict_recall     code: known conflicts flagged by reconcile, plus a false-positive count.
- vs_published_209    LLM judge on the narrative only (1-5); the numeric fields are checked in code.
- fields_within_tolerance, total_tokens, wall_seconds: code only, no judge.

The judge never does arithmetic we rely on: faithfulness is recomputed from its claim list, and
the field tolerances are computed here with reconcile's own rules.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from firebinder import llm
from firebinder.nodes.draft import render_evidence
from firebinder.nodes.reconcile import CONTAINMENT_TOLERANCE, acreage_tolerance

# The numeric 209 fields the published reference has; compared by code, not by the judge.
REFERENCE_FIELDS = ["acres", "pct_contained", "personnel", "structures_threatened", "structures_destroyed"]


def _prompt(name: str) -> str:
    return resources.files("firebinder").joinpath(f"prompts/{name}").read_text(encoding="utf-8")


def _draft(outputs: dict | None) -> dict | None:
    return (outputs or {}).get("draft") or None


# ---------- faithfulness (LLM judge) ----------


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="The claim, quoted or closely paraphrased from the draft.")
    supported: bool
    why: str = Field(description="One sentence naming the [n] evidence items relied on, or what is missing.")


class FaithfulnessVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(description="Every factual claim in the draft, listed before the score.")
    score: float = Field(description="Supported claims / total claims, 0 to 1.")


def faithfulness_score(claims: list[Claim]) -> float:
    """Fraction supported, recomputed from the claim list; 1.0 for a draft with no claims."""
    if not claims:
        return 1.0
    return sum(c.supported for c in claims) / len(claims)


def build_faithfulness_prompt(outputs: dict) -> str:
    draft = _draft(outputs) or {}
    return _prompt("judge_faithfulness.md").format(
        evidence=render_evidence(outputs.get("evidence") or []),
        narrative=draft.get("narrative") or "",
        fields=json.dumps(draft.get("fields") or {}, default=str, indent=1),
    )


def faithfulness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Judge sees the draft and the run's evidence, never the reference."""
    if _draft(outputs) is None:
        return {"key": "faithfulness", "score": None, "comment": "no draft produced"}
    model = llm.get_model("judge").with_structured_output(FaithfulnessVerdict, method="json_schema")
    raw = model.invoke([HumanMessage(build_faithfulness_prompt(outputs))])
    verdict = raw if isinstance(raw, FaithfulnessVerdict) else FaithfulnessVerdict.model_validate(raw)
    score = faithfulness_score(verdict.claims)
    unsupported = [c for c in verdict.claims if not c.supported]
    lines = [f"{len(verdict.claims) - len(unsupported)}/{len(verdict.claims)} claims supported."]
    if unsupported:
        lines.append("Unsupported:")
        lines += [f"- {c.claim} ({c.why})" for c in unsupported]
    return {"key": "faithfulness", "score": score, "comment": "\n".join(lines)}


# ---------- conflict_recall (code) ----------


def conflict_recall(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Recall over the example's known conflicts (matched by field), plus flagged conflicts on other fields.

    Empty case: no known conflicts means recall is 1.0 (nothing was missed); any flagged conflicts
    then show up only in conflict_false_positives.
    """
    known = (reference_outputs or {}).get("known_conflicts") or []
    flagged = (outputs or {}).get("conflicts") or []
    flagged_severities: dict[str, set[str]] = {}
    for c in flagged:
        flagged_severities.setdefault(c["field"], set()).add(c["severity"])
    known_fields = {k["field"] for k in known}

    hits = [k for k in known if k["field"] in flagged_severities]
    missed = [k for k in known if k["field"] not in flagged_severities]
    severity_matches = [k for k in hits if k["severity"] in flagged_severities[k["field"]]]
    false_positives = [c for c in flagged if c["field"] not in known_fields]

    recall = len(hits) / len(known) if known else 1.0
    recall_comment = (
        f"{len(hits)}/{len(known)} known conflicts flagged; {len(severity_matches)}/{len(hits)} with matching severity."
        if known
        else "no known conflicts"
    )
    if missed:
        recall_comment += " Missed: " + ", ".join(f"{k['field']} ({k['severity']})" for k in missed) + "."
    mismatched = [k for k in hits if k not in severity_matches]
    if mismatched:
        recall_comment += " Severity differs: " + ", ".join(
            f"{k['field']} (known {k['severity']}, flagged {'/'.join(sorted(flagged_severities[k['field']]))})" for k in mismatched
        ) + "."
    fp_comment = (
        "False positives: " + ", ".join(f"{c['field']} ({c['severity']})" for c in false_positives)
        if false_positives
        else "no false positives"
    )
    return {
        "results": [
            {"key": "conflict_recall", "score": recall, "comment": recall_comment},
            {"key": "conflict_false_positives", "score": len(false_positives), "comment": fp_comment},
        ]
    }


# ---------- vs_published_209: field table (code) + narrative (LLM judge) ----------


class FieldCheck(TypedDict):
    field: str
    draft_value: float | None
    reference_value: float | None
    within_tolerance: bool | None  # None when the reference has no value for the field


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def within_tolerance(field: str, draft_value: Any, reference_value: Any) -> bool | None:
    """reconcile's rules: acres max(10% of the larger, 500 ac); pct_contained 5 points; counts exact.

    None when the reference value is missing (nothing to compare); False when only the draft's is.
    """
    ref = _number(reference_value)
    if ref is None:
        return None
    got = _number(draft_value)
    if got is None:
        return False
    if field == "acres":
        return abs(got - ref) <= acreage_tolerance(got, ref)
    if field == "pct_contained":
        return abs(got - ref) <= CONTAINMENT_TOLERANCE
    return got == ref


def field_table(draft_fields: dict | None, reference_fields: dict | None) -> list[FieldCheck]:
    draft_fields = draft_fields or {}
    reference_fields = reference_fields or {}
    return [
        FieldCheck(
            field=name,
            draft_value=_number(draft_fields.get(name)),
            reference_value=_number(reference_fields.get(name)),
            within_tolerance=within_tolerance(name, draft_fields.get(name), reference_fields.get(name)),
        )
        for name in REFERENCE_FIELDS
    ]


def _fmt(value: float | None) -> str:
    return "null" if value is None else f"{value:g}"


def render_field_table(table: list[FieldCheck]) -> str:
    marks = {True: "yes", False: "NO", None: "n/a (no reference)"}
    lines = ["| field | draft | published | within tolerance |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {c['field']} | {_fmt(c['draft_value'])} | {_fmt(c['reference_value'])} | {marks[c['within_tolerance']]} |"
        for c in table
    ]
    return "\n".join(lines)


def fields_within_tolerance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Fraction of non-null reference fields the draft matched within tolerance; None if the reference has none."""
    draft = _draft(outputs) or {}
    table = field_table(draft.get("fields"), (reference_outputs or {}).get("fields"))
    compared = [c for c in table if c["within_tolerance"] is not None]
    score = sum(bool(c["within_tolerance"]) for c in compared) / len(compared) if compared else None
    return {"key": "fields_within_tolerance", "score": score, "comment": render_field_table(table)}


class JudgedFieldCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    draft_value: float | None
    reference_value: float | None
    within_tolerance: bool | None


class Vs209Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_checks: list[JudgedFieldCheck] = Field(description="The code-computed field table, restated unchanged.")
    missed_events: list[str] = Field(description="Significant events in the published 209 the draft does not mention.")
    invented_events: list[str] = Field(description="Events in the draft that the published 209 does not mention or contradicts.")
    score: int = Field(description="1-5 per the rubric, given after the lists.")
    rationale: str


def build_vs_209_prompt(inputs: dict, outputs: dict, reference_outputs: dict, table: list[FieldCheck]) -> str:
    draft = _draft(outputs) or {}
    return _prompt("judge_vs_209.md").format(
        incident_name=(inputs or {}).get("incident_name"),
        report_date=(inputs or {}).get("report_date"),
        field_table=render_field_table(table),
        reference_narrative=(reference_outputs or {}).get("narrative") or "(the published 209 has no narrative)",
        draft_narrative=draft.get("narrative") or "",
        draft_fields=json.dumps(draft.get("fields") or {}, default=str, indent=1),
    )


def vs_published_209(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Raw 1-5 rubric score for narrative coverage; the comment carries the code field table and the judge's lists."""
    draft = _draft(outputs)
    if draft is None:
        return {"key": "vs_published_209", "score": None, "comment": "no draft produced"}
    table = field_table(draft.get("fields"), (reference_outputs or {}).get("fields"))
    model = llm.get_model("judge").with_structured_output(Vs209Verdict, method="json_schema")
    raw = model.invoke([HumanMessage(build_vs_209_prompt(inputs, outputs, reference_outputs, table))])
    verdict = raw if isinstance(raw, Vs209Verdict) else Vs209Verdict.model_validate(raw)
    score = max(1, min(5, int(verdict.score)))
    lines = [
        render_field_table(table),  # ours, not the judge's restatement
        "",
        "Missed events: " + ("; ".join(verdict.missed_events) or "none"),
        "Invented events: " + ("; ".join(verdict.invented_events) or "none"),
        f"Rationale: {verdict.rationale}",
    ]
    return {"key": "vs_published_209", "score": score, "comment": "\n".join(lines)}


# ---------- cost metrics, read straight from the target's outputs ----------


def total_tokens(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "total_tokens", "score": (outputs or {}).get("total_tokens")}


def wall_seconds(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "wall_seconds", "score": (outputs or {}).get("wall_seconds")}


CODE_EVALUATORS = [conflict_recall, fields_within_tolerance, total_tokens, wall_seconds]
JUDGE_EVALUATORS = [faithfulness, vs_published_209]
