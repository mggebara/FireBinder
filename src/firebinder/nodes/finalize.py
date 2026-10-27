"""finalize: write the 209 draft and its supporting JSON to out/<irwin_id>/<date>/."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from firebinder.nodes.review import audit_failing
from firebinder.state import FireBinderState


def output_dir(irwin_id: str | None, report_date: str) -> Path:
    """out/<irwin_id>/<date>; the root comes from FIREBINDER_OUT (default "out")."""
    root = Path(os.environ.get("FIREBINDER_OUT", "out"))
    return root / (irwin_id or "unresolved") / report_date


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return " ".join(text.split()).replace("|", "\\|")


def render_209(state: FireBinderState) -> str:
    draft = state.get("draft") or {}
    fields: dict[str, Any] = draft.get("fields", {})
    citations: dict[str, str] = draft.get("citations", {})
    audit = state.get("audit", [])
    conflicts = state.get("conflicts", [])
    approved = state.get("approved")
    status = {True: "approved", False: "REJECTED", None: "not reviewed"}[approved]

    lines = [
        f"# ICS-209 summary: {state['incident_name']} ({state['report_date']})",
        "",
        f"IRWIN {state.get('irwin_id')} · mode {state.get('mode')} · revision {state.get('revision', 0)} · {status}",
        "",
    ]
    if audit_failing(audit):
        lines += ["> **Flag:** this draft still failed the audit when it was sent to review.", ""]
    lines += ["## Narrative", "", draft.get("narrative", "(no narrative)"), "", "## Fields", ""]
    lines += ["| field | value | source |", "| --- | --- | --- |"]
    for name, value in fields.items():
        lines.append(f"| {name} | {_cell(value)} | {_cell(citations.get(name, '(uncited)'))} |")

    lines += ["", "## Conflicts", ""]
    if not conflicts:
        lines.append("None.")
    for c in conflicts:
        sources = ", ".join(f"{ev['source']}={_cell(ev['value'])}" for ev in c["values"])
        lines.append(f"- **{c['severity']}** {c['field']}: {c['note']} ({sources})")

    counts = {s: sum(1 for f in audit if f["status"] == s) for s in ("supported", "unsupported", "contradicted")}
    lines += ["", "## Audit", "", ", ".join(f"{n} {s}" for s, n in counts.items())]
    for f in audit:
        if f["status"] != "supported":
            lines.append(f"- **{f['status']}**: {f['claim']}")
    return "\n".join(lines) + "\n"


def _dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def finalize(state: FireBinderState) -> dict[str, Any]:
    out = output_dir(state.get("irwin_id"), state["report_date"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "209.md").write_text(render_209(state))
    _dump(out / "evidence.json", state.get("evidence", []))
    _dump(out / "conflicts.json", state.get("conflicts", []))
    _dump(out / "audit.json", state.get("audit", []))
    return {"messages": [AIMessage(content=f"Wrote {out}", name="finalize")]}
