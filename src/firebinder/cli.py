"""FireBinder command-line interface. Thin: orchestration lives in firebinder.nodes.conflicts."""

from __future__ import annotations

import json
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from firebinder.nodes.conflicts import ConflictsRun, run_conflicts
from firebinder.nodes.resolve import IncidentResolutionError
from firebinder.tools.wfigs import WfigsError

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


if __name__ == "__main__":
    app()
