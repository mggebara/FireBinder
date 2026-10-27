"""FireBinder command-line interface. Thin: orchestration lives in firebinder.nodes.conflicts."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # LANGSMITH_* before anything LangChain is imported

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from firebinder.nodes.conflicts import ConflictsRun, run_conflicts  # noqa: E402
from firebinder.nodes.resolve import IncidentResolutionError, run_mode  # noqa: E402
from firebinder.tools.wfigs import WfigsError  # noqa: E402

app = typer.Typer(help="FireBinder: draft and audit wildfire ICS-209 summaries from public data.")

MAX_VALUE_CHARS = 80
MAX_URL_CHARS = 60  # the full URL is in --json


@app.callback()
def main() -> None:
    """FireBinder CLI."""


def _truncate(text: str, limit: int = MAX_VALUE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_value(value: Any) -> str:
    """Short display form: geometries become 'MultiPolygon (107 parts)', long text/lists are cut to ~80 chars."""
    if isinstance(value, dict) and "type" in value and "coordinates" in value:
        geom_type = value["type"]
        parts = len(value["coordinates"]) if geom_type.startswith("Multi") else 1
        return f"{geom_type} ({parts} part{'s' if parts != 1 else ''})"
    if isinstance(value, list | dict):
        return _truncate(json.dumps(value, default=str))
    return _truncate(" ".join(str(value).split()))


def _resolved_line(run: ConflictsRun) -> str:
    r = run.resolved
    lat, lon = r.centroid
    return (
        f"[bold]{r.incident_name}[/bold]  IRWIN {r.irwin_id}  state {r.state}  "
        f"centroid ({lat:.4f}, {lon:.4f})  discovered {r.discovery_date}"
    )


def _render(console: Console, run: ConflictsRun) -> None:
    console.print(_resolved_line(run), highlight=False)

    if run.errors:
        console.print("\n[bold red]Tool errors[/bold red]")
        for err in run.errors:
            console.print(f"  - {err.tool}: {err.message}", highlight=False, markup=False)

    evidence_table = Table(title="Evidence", show_lines=False)
    for column in ("field", "value", "source", "fetched_at"):
        evidence_table.add_column(column)
    evidence_table.add_column("url", no_wrap=True)
    for ev in run.evidence:
        evidence_table.add_row(ev["field"], format_value(ev["value"]), ev["source"], ev["fetched_at"], _truncate(ev["url"], MAX_URL_CHARS))
    console.print(evidence_table)

    if not run.conflicts:
        console.print("No conflicts found.")
        return
    conflict_table = Table(title="Conflicts", show_lines=True)
    for column in ("severity", "field", "note", "values"):
        conflict_table.add_column(column)
    for c in run.conflicts:
        values = "\n".join(f"{ev['source']}={format_value(ev['value'])}" for ev in c["values"])
        conflict_table.add_row(c["severity"], c["field"], c["note"], values)
    console.print(conflict_table)


@app.command()
def conflicts(
    incident: str = typer.Option(..., "--incident", help="Incident name, e.g. GRASSHOPPER."),
    date: str = typer.Option(..., "--date", help="Report date, YYYY-MM-DD."),
    state: str | None = typer.Option(None, "--state", help="Two-letter state to disambiguate, e.g. OR."),
    radius_km: float = typer.Option(25, "--radius-km", help="FIRMS search radius around the centroid."),
    day_range: int = typer.Option(3, "--day-range", help="FIRMS days ending on --date."),
    as_json: bool = typer.Option(False, "--json", help="Dump {resolved, evidence, conflicts, errors} as JSON."),
) -> None:
    """Resolve an incident, gather evidence from the four live tools, and print the conflicts."""
    console = Console()
    try:
        run = run_conflicts(incident, date, state=state, radius_km=radius_km, day_range=day_range)
    except (IncidentResolutionError, WfigsError, httpx.HTTPError) as exc:
        Console(stderr=True).print(f"Could not resolve incident: {exc}", markup=False, highlight=False)
        raise typer.Exit(1) from exc

    if as_json:
        payload = run.model_dump(include={"resolved", "evidence", "conflicts", "errors"})
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        _render(console, run)

    if len(run.errors) == run.tools_called:
        raise typer.Exit(1)



# --- draft / resume -------------------------------------------------------------------------------


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "incident"


def _tracing_on() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}


def _invoke_traced(console: Console, graph: Any, graph_input: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Invoke the graph; with LANGSMITH_TRACING on, collect the root run and print its LangSmith URL."""
    if not _tracing_on():
        return graph.invoke(graph_input, config)
    from langchain_core.tracers.context import tracing_v2_enabled

    with tracing_v2_enabled(project_name=os.environ.get("LANGSMITH_PROJECT")) as cb:
        result = graph.invoke(graph_input, config)
    try:
        console.print(f"LangSmith run: {cb.get_run_url()}", markup=False, highlight=False)
    except Exception:  # noqa: BLE001 - the URL is best-effort; the run itself already finished
        pass
    return result


def _render_review(console: Console, payload: dict[str, Any], thread_id: str) -> None:
    draft = payload.get("draft") or {}
    console.print(f"\n[bold]Draft (revision {payload.get('revision', 0)})[/bold]")
    if payload.get("flag"):
        console.print(f"Flag: {payload['flag']}", style="bold red", markup=False, highlight=False)
    console.print(draft.get("narrative", "(no narrative)"), markup=False, highlight=False)

    fields = Table(title="Fields")
    for column in ("field", "value", "citation"):
        fields.add_column(column)
    citations = draft.get("citations", {})
    for name, value in draft.get("fields", {}).items():
        fields.add_row(name, format_value(value), _truncate(str(citations.get(name, "(uncited)")), MAX_URL_CHARS))
    console.print(fields)

    conflicts = payload.get("conflicts", [])
    if conflicts:
        table = Table(title="Conflicts", show_lines=True)
        for column in ("severity", "field", "note"):
            table.add_column(column)
        for c in conflicts:
            table.add_row(c["severity"], c["field"], c["note"])
        console.print(table)
    else:
        console.print("No conflicts found.")

    audit = payload.get("audit", [])
    counts = {s: sum(1 for f in audit if f["status"] == s) for s in ("supported", "unsupported", "contradicted")}
    console.print("Audit: " + ", ".join(f"{n} {s}" for s, n in counts.items()), highlight=False)
    table = Table(title="Audit")
    table.add_column("status")
    table.add_column("claim")
    for finding in audit:
        table.add_row(finding["status"], _truncate(finding["claim"], 120))
    console.print(table)

    console.print(
        f"\nPaused for review. Thread {thread_id}. Continue with one of:\n"
        f"  firebinder resume --thread-id {thread_id} --approve\n"
        f"  firebinder resume --thread-id {thread_id} --reject\n"
        f"  firebinder resume --thread-id {thread_id} --edit DRAFT.json   "
        "(JSON with narrative, fields, citations)",
        markup=False,
        highlight=False,
    )


def _report(console: Console, graph: Any, result: dict[str, Any], config: dict[str, Any], thread_id: str) -> None:
    """After invoke/resume: show the review prompt if paused, else where finalize wrote."""
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        _render_review(console, interrupts[0].value, thread_id)
        return
    from firebinder.nodes.finalize import approval_status, output_dir

    state = graph.get_state(config).values
    out = output_dir(state.get("irwin_id"), state["report_date"])
    console.print(f"Finalized ({approval_status(state.get('approved'))}).", highlight=False)
    console.print(f"Wrote {out / '209.md'} (+ evidence.json, conflicts.json, audit.json)", markup=False, highlight=False)


def _thread_exists(graph: Any, thread_id: str) -> bool:
    return bool(graph.get_state({"configurable": {"thread_id": thread_id}}).values)


def _fresh_thread_id(graph: Any, base: str) -> str:
    """base, or base-2, base-3, ... so a re-run never appends to an old run's evidence."""
    candidate, n = base, 1
    while _thread_exists(graph, candidate):
        n += 1
        candidate = f"{base}-{n}"
    return candidate


@app.command()
def draft(
    incident: str = typer.Option(..., "--incident", help="Incident name, e.g. GRASSHOPPER."),
    date: str = typer.Option(..., "--date", help="Report date, YYYY-MM-DD."),
    state: str | None = typer.Option(None, "--state", help="Two-letter state to disambiguate, e.g. OR."),
    skip_review: bool = typer.Option(False, "--skip-review", help="Approve without pausing (historical eval runs)."),
    thread_id: str | None = typer.Option(None, "--thread-id", help="Checkpoint thread; default <slug>-<date>."),
) -> None:
    """Run the full graph: resolve, gather, reconcile, draft, audit (with revisions), review, finalize."""
    from firebinder.graph import build_graph, get_checkpointer, initial_state, run_config

    console = Console()
    graph = build_graph(get_checkpointer())
    if thread_id is None:
        thread_id = _fresh_thread_id(graph, f"{slug(incident)}-{date}")
    elif _thread_exists(graph, thread_id):
        Console(stderr=True).print(
            f"Thread {thread_id} already has a run; use `firebinder resume` or another --thread-id.",
            markup=False,
            highlight=False,
        )
        raise typer.Exit(1)
    config = run_config(incident, date, run_mode(date), thread_id, skip_review=skip_review, state=state)
    console.print(f"Thread {thread_id}", highlight=False)
    try:
        result = _invoke_traced(console, graph, initial_state(incident, date), config)
    except (IncidentResolutionError, WfigsError, httpx.HTTPError) as exc:
        Console(stderr=True).print(f"Could not resolve incident: {exc}", markup=False, highlight=False)
        raise typer.Exit(1) from exc
    _report(console, graph, result, config, thread_id)


def _resume_payload(approve: bool, reject: bool, edit: Path | None) -> dict[str, Any]:
    if sum([approve, reject, edit is not None]) != 1:
        raise typer.BadParameter("pass exactly one of --approve, --reject, --edit FILE")
    if approve:
        return {"action": "approve"}
    if reject:
        return {"action": "reject"}
    data = json.loads(edit.read_text())  # type: ignore[union-attr]
    return {"action": "edit", "draft": data.get("draft", data)}  # accept the draft or {"draft": ...}


@app.command()
def resume(
    thread_id: str = typer.Option(..., "--thread-id", help="Thread printed by `firebinder draft`."),
    approve: bool = typer.Option(False, "--approve", help="Approve the draft as is."),
    reject: bool = typer.Option(False, "--reject", help="Reject the draft (still written, marked rejected)."),
    edit: Path | None = typer.Option(None, "--edit", exists=True, dir_okay=False, help="JSON file with the edited draft."),
) -> None:
    """Resume a run paused at human review."""
    from langgraph.types import Command

    from firebinder.graph import build_graph, get_checkpointer, run_config

    payload = _resume_payload(approve, reject, edit)
    console = Console()
    graph = build_graph(get_checkpointer())
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.next:
        Console(stderr=True).print(f"Thread {thread_id} is not paused at review.", markup=False, highlight=False)
        raise typer.Exit(1)
    values = snapshot.values
    config = run_config(values["incident_name"], values["report_date"], values.get("mode", "live"), thread_id)
    result = _invoke_traced(console, graph, Command(resume=payload), config)
    _report(console, graph, result, config, thread_id)


if __name__ == "__main__":
    app()
