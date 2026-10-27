"""finalize: write the 209 draft and its supporting JSON to <out>/<irwin_id>/<date>/."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from firebinder.nodes.review import review_flag
from firebinder.state import Evidence, FireBinderState

OUTPUT_FILES = ("209.md", "evidence.json", "conflicts.json", "audit.json")


def output_dir(irwin_id: str | None, report_date: str) -> Path:
    """<root>/<irwin_id>/<date>; the root comes from FIREBINDER_OUT_DIR (default "out")."""
    root = Path(os.environ.get("FIREBINDER_OUT_DIR", "out"))
    return root / (irwin_id or "unresolved") / report_date


def approval_status(approved: bool | None) -> str:
    return {True: "approved", False: "rejected", None: "bypassed"}[approved]


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return " ".join(text.split()).replace("|", "\\|")


def render_209(state: FireBinderState) -> str:
    draft = state.get("draft") or {}
    fields: dict[str, Any] = draft.get("fields", {})
    citations: dict[str, str] = draft.get("citations", {})
    audit = state.get("audit", [])
    conflicts = state.get("conflicts", [])

    lines = [
        f"# ICS-209 summary: {state['incident_name']} ({state['report_date']})",
        "",
        f"**Status:** {approval_status(state.get('approved'))}",
        "",
        f"IRWIN {state.get('irwin_id')} · mode {state.get('mode')} · revision {state.get('revision', 0)}",
        "",
    ]
    if flag := review_flag(state):
        lines += [f"> **Flag:** {flag}.", ""]

    lines += ["## Fields", "", "| field | value | citation |", "| --- | --- | --- |"]
    for name, value in fields.items():
        lines.append(f"| {name} | {_cell(value)} | {_cell(citations.get(name, '(uncited)'))} |")

    lines += ["", "## Narrative", "", draft.get("narrative") or "(no narrative)", "", "## Conflicts", ""]
    if not conflicts:
        lines.append("None.")
    for c in conflicts:
        lines.append(f"- **{c['severity']}** {c['field']}: {c['note']}")

    counts = {s: sum(1 for f in audit if f["status"] == s) for s in ("supported", "unsupported", "contradicted")}
    lines += ["", "## Audit", "", ", ".join(f"{n} {s}" for s, n in counts.items())]
    for f in audit:
        if f["status"] != "supported":
            lines.append(f"- **{f['status']}**: {f['claim']}")
    return "\n".join(lines) + "\n"


def slim_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Replace each perimeter_geometry value with {type, parts}; the full polygon stays out of the file."""
    slim: list[Evidence] = []
    for ev in evidence:
        value = ev["value"]
        if ev["field"] == "perimeter_geometry" and isinstance(value, dict):
            geom_type = value.get("type")
            coords = value.get("coordinates") or []
            parts = len(coords) if str(geom_type).startswith("Multi") else 1
            ev = {**ev, "value": {"type": geom_type, "parts": parts}}
        slim.append(ev)
    return slim


def _dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def finalize(state: FireBinderState, config: RunnableConfig | None = None) -> dict[str, Any]:
    out = output_dir(state.get("irwin_id"), state["report_date"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "209.md").write_text(render_209(state))
    _dump(out / "evidence.json", slim_evidence(state.get("evidence", [])))
    _dump(out / "conflicts.json", state.get("conflicts", []))
    _dump(out / "audit.json", state.get("audit", []))
    return {"messages": [AIMessage(content=f"Wrote {out / '209.md'}", name="finalize")]}
