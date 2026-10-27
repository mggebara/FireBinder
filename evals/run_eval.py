"""Run the FireBinder graph over the LangSmith dataset and score it (SPEC "LangSmith").

    uv run python -m evals.run_eval --limit 2          # smoke run on the first two examples
    uv run python -m evals.run_eval --note "audit prompt v2"

Each example runs the full graph with review skipped, on an in-memory checkpointer and a fresh
thread. Planted evidence from the example's reference outputs is injected into the initial state
(evidence is an append reducer, so it survives gather). The experiment is named by git SHA, and a
row of mean scores is appended to evals/results.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # `python evals/run_eval.py`: make `evals` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import datetime as dt  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections.abc import Callable, Iterable  # noqa: E402
from typing import Any  # noqa: E402

import typer  # noqa: E402
from langchain_core.callbacks import UsageMetadataCallbackHandler  # noqa: E402

from evals.evaluators import CODE_EVALUATORS, JUDGE_EVALUATORS  # noqa: E402
from firebinder.graph import git_sha, initial_state, run_config  # noqa: E402
from firebinder.nodes.draft import render_value  # noqa: E402
from firebinder.state import Evidence, FireBinderState  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "results.md"
RESULTS_HEADER = (
    "| date | git sha | n | faithfulness | conflict_recall | false positives (mean) | vs_209 (1-5) "
    "| fields_within_tolerance | mean tokens | mean seconds | note |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)
# (feedback key, format) in results.md column order
RESULT_COLUMNS = [
    ("faithfulness", "{:.2f}"),
    ("conflict_recall", "{:.2f}"),
    ("conflict_false_positives", "{:.2f}"),
    ("vs_published_209", "{:.2f}"),
    ("fields_within_tolerance", "{:.2f}"),
    ("total_tokens", "{:,.0f}"),
    ("wall_seconds", "{:.1f}"),
]

app = typer.Typer(add_completion=False)


# ---------- target ----------


def build_initial_state(inputs: dict, planted_evidence: list[Evidence] | None) -> FireBinderState:
    state = initial_state(inputs["incident_name"], inputs["report_date"])
    state["evidence"] = list(planted_evidence or [])
    return state


def _strip_geometry(ev: Evidence) -> Evidence:
    if ev.get("field") == "perimeter_geometry":
        return {**ev, "value": render_value("perimeter_geometry", ev["value"])}
    return ev


def summarize_outputs(result: dict, total_tokens: int, wall_seconds: float) -> dict:
    """The graph's final state -> the outputs the evaluators read (no multi-MB perimeter polygons)."""
    conflicts = [{**c, "values": [_strip_geometry(v) for v in c.get("values", [])]} for c in result.get("conflicts") or []]
    return {
        "draft": result.get("draft"),
        "conflicts": conflicts,
        "audit": result.get("audit") or [],
        "evidence": [_strip_geometry(ev) for ev in result.get("evidence") or []],
        "revision": result.get("revision", 0),
        "total_tokens": total_tokens,
        "wall_seconds": round(wall_seconds, 2),
    }


def sum_tokens(handler: UsageMetadataCallbackHandler) -> int:
    """Input + output tokens over every model the run called."""
    return sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in handler.usage_metadata.values())


def _current_example_id() -> str | None:
    """evaluate() sets reference_example_id on the target's root run; the target only receives inputs."""
    from langsmith.run_helpers import get_current_run_tree

    run = get_current_run_tree()
    example_id = getattr(run, "reference_example_id", None) if run else None
    return str(example_id) if example_id else None


def make_target(graph: Any, planted_by_example: dict[str, list[Evidence] | None]) -> Callable[[dict], dict]:
    from firebinder.nodes.resolve import run_mode

    def target(inputs: dict) -> dict:
        planted = planted_by_example.get(_current_example_id() or "")
        config = run_config(
            inputs["incident_name"],
            inputs["report_date"],
            run_mode(inputs["report_date"]),
            thread_id=str(uuid.uuid4()),
            skip_review=True,
            state=inputs.get("state"),
        )
        usage = UsageMetadataCallbackHandler()
        config["callbacks"] = [usage]
        start = time.perf_counter()
        result = graph.invoke(build_initial_state(inputs, planted), config)
        return summarize_outputs(result, sum_tokens(usage), time.perf_counter() - start)

    return target


# ---------- results ----------


def _run_failed(row: dict) -> bool:
    run = row.get("run")
    error = getattr(run, "error", None) if run is not None and not isinstance(run, dict) else (run or {}).get("error")
    return bool(error)


def feedback_means(rows: Iterable[dict]) -> tuple[int, int, dict[str, float]]:
    """(successful examples, failed examples, mean score per feedback key over successful ones).

    A run that raised (API outage, exhausted credit, resolution failure) contributes nothing: its
    code evaluators would otherwise score an empty output and drag the means down silently.
    """
    scores: dict[str, list[float]] = {}
    n = 0
    failed = 0
    for row in rows:
        if _run_failed(row):
            failed += 1
            continue
        n += 1
        for res in (row.get("evaluation_results") or {}).get("results", []):
            key, score = (res.get("key"), res.get("score")) if isinstance(res, dict) else (res.key, res.score)
            if score is not None:
                scores.setdefault(key, []).append(float(score))
    return n, failed, {key: sum(vals) / len(vals) for key, vals in scores.items()}


def format_row(date: str, sha: str, n: int, means: dict[str, float], note: str) -> str:
    cells = [date, sha, str(n)]
    cells += [fmt.format(means[key]) if key in means else "n/a" for key, fmt in RESULT_COLUMNS]
    cells.append(note.replace("|", "\\|").replace("\n", " "))
    return "| " + " | ".join(cells) + " |"


def append_row(row: str, path: Path = RESULTS_PATH) -> None:
    if not path.exists():
        path.write_text("# FireBinder eval results\n\n" + RESULTS_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


# ---------- CLI ----------


@app.command()
def main(
    dataset: str = typer.Option("firebinder-historical", "--dataset", help="LangSmith dataset name."),
    limit: int | None = typer.Option(None, "--limit", help="Smoke run on the first N examples."),
    concurrency: int = typer.Option(2, "--concurrency", help="Examples run in parallel."),
    skip_judges: bool = typer.Option(False, "--skip-judges", help="Code evaluators only (no LLM judge calls)."),
    note: str = typer.Option("", "--note", help="Free text recorded in the experiment metadata and results.md."),
) -> None:
    """Evaluate the current commit against the dataset and append a row to evals/results.md."""
    from langgraph.checkpoint.memory import MemorySaver
    from langsmith import Client, evaluate

    from firebinder.graph import build_graph

    client = Client()
    examples = list(client.list_examples(dataset_name=dataset, limit=limit))
    if not examples:
        typer.echo(f"No examples in dataset {dataset!r}.", err=True)
        raise typer.Exit(1)
    planted = {str(ex.id): (ex.outputs or {}).get("planted_evidence") for ex in examples}

    sha = git_sha()
    date = dt.date.today().isoformat()
    evaluators = CODE_EVALUATORS + ([] if skip_judges else JUDGE_EVALUATORS)
    results = evaluate(
        make_target(build_graph(MemorySaver()), planted),
        data=examples,
        evaluators=evaluators,
        experiment_prefix=f"firebinder-{sha}",
        description=note or None,
        metadata={"git_sha": sha, "note": note, "date": date, "skip_judges": skip_judges},
        max_concurrency=concurrency,
        client=client,
    )

    n, failed, means = feedback_means(results)
    if failed:
        note = f"{note} ({failed} of {n + failed} runs errored)".strip()
        typer.echo(f"WARNING: {failed} of {n + failed} runs errored; means cover the {n} that finished", err=True)
    row = format_row(date, sha, n, means, note)
    if n == 0:
        typer.echo("No run finished; not recording a results row", err=True)
    else:
        append_row(row)
    typer.echo(RESULTS_HEADER.splitlines()[0])
    typer.echo(row)
    typer.echo(json.dumps({"experiment": results.experiment_name, "url": results.url}))


if __name__ == "__main__":
    app()
