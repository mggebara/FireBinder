"""Build the `firebinder-historical` LangSmith dataset from ICS209-PLUS + GeoMAC.

    uv run python evals/build_dataset.py candidates   # local CSVs + GeoMAC -> evals/candidates.json
    uv run python evals/build_dataset.py select --n 20  # -> evals/dataset.json
    uv run python evals/build_dataset.py confirm      # run_conflicts per example -> known_conflicts
    uv run python evals/build_dataset.py plant        # + 3 planted-conflict examples
    uv run python evals/build_dataset.py upload --dataset firebinder-historical

Window is 2012-2014: ICS209-PLUS ends in 2014 and FIRMS VIIRS starts 2012-01-20. GeoMAC
historic perimeters stand in for the historical incident record; a fire's GeoMAC id is derived
from the 209 INCIDENT_NUMBER ("CA-STF-002857" in 2013 -> "2013-CASTF-002857"), with a
name+state fallback in the same query.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from firebinder.tools import ics209plus
from firebinder.tools._http import get_client

ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = ROOT / "evals"
CANDIDATES_PATH = EVALS_DIR / "candidates.json"
DATASET_PATH = EVALS_DIR / "dataset.json"
FIXTURES_DIR = EVALS_DIR / "fixtures"
GEOMAC_CACHE = ROOT / "data" / "geomac"
DATA_DIR = ROOT / "data" / "ics209plus"

YEARS = (2012, 2014)
MIN_FINAL_ACRES = 10_000
MIN_SITREPS = 5
FIRMS_START = "2012-01-20"  # first day of VIIRS active-fire coverage
MAX_DAYS_PER_FIRE = 2
MIN_DAYS_APART = 3
MIN_FIRES = 8
MIN_CONFIRMED = 5
FIELD_NAMES = ("acres", "pct_contained", "personnel", "structures_threatened", "structures_destroyed")
GEOMAC_FIELDS = "uniquefireidentifier,incidentname,state,gisacres,perimeterdatetime,fireyear"

app = typer.Typer(add_completion=False, no_args_is_help=True)


# ---------------------------------------------------------------- GeoMAC


def geomac_layer_url(year: int) -> str:
    name = "Historic_GeoMAC_Perimeters_2019" if year == 2019 else f"Historic_Geomac_Perimeters_{year}"
    return f"https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/{name}/FeatureServer/0/query"


def geomac_id(incident_number: object, year: int) -> str | None:
    """'CA-STF-002857', 2013 -> '2013-CASTF-002857' (GeoMAC uniquefireidentifier format)."""
    if not isinstance(incident_number, str):
        return None
    parts = incident_number.strip().upper().split("-")
    if len(parts) != 3 or not all(parts):
        return None
    return f"{year}-{parts[0]}{parts[1]}-{parts[2]}"


def _sql(text: str) -> str:
    return text.replace("'", "''")


def fetch_geomac_perimeters(name: str, state: str | None, year: int, fire_id: str | None,
                            cache_dir: Path = GEOMAC_CACHE) -> list[dict]:
    """Every GeoMAC perimeter for the fire in that year's layer (attributes only). Disk-cached."""
    key = (fire_id or f"{name}_{state}").replace("/", "_").replace(" ", "_")
    cache = cache_dir / str(year) / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    clauses = []
    if fire_id:
        clauses.append(f"uniquefireidentifier = '{_sql(fire_id)}'")
    names = {name.upper(), f"{name.upper()} FIRE"}
    name_list = ", ".join(f"'{_sql(n)}'" for n in sorted(names))
    by_name = f"UPPER(incidentname) IN ({name_list})"
    clauses.append(f"({by_name} AND state = '{_sql(state)}')" if state else by_name)
    params = {"where": " OR ".join(clauses), "outFields": GEOMAC_FIELDS, "returnGeometry": "false",
              "orderByFields": "perimeterdatetime", "f": "json"}
    response = get_client().get(geomac_layer_url(year), params=params)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"GeoMAC query failed: {body['error']}")
    perimeters = [_perimeter(f["attributes"]) for f in body.get("features", [])]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(perimeters))
    return perimeters


def _perimeter(attrs: dict) -> dict:
    stamp = attrs.get("perimeterdatetime")
    date = datetime.fromtimestamp(stamp / 1000, UTC).strftime("%Y-%m-%d") if stamp else None
    return {"id": attrs.get("uniquefireidentifier"), "name": attrs.get("incidentname"),
            "state": attrs.get("state"), "acres": attrs.get("gisacres"), "date": date}


# ---------------------------------------------------------------- candidates


def _pct_ok(df: pd.DataFrame) -> pd.Series:
    unit = df["PCT_CONT_COMPL_UOM_ABBREV"].fillna("PCT").astype(str).str.strip().str.upper()
    return df["PCT_CONTAINED_COMPLETED"].notna() & (unit == "PCT")


def _has_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip() != ""


def candidate_days(sitreps: pd.DataFrame, incidents: pd.DataFrame, years: tuple[int, int] = YEARS,
                   min_final_acres: float = MIN_FINAL_ACRES, min_sitreps: int = MIN_SITREPS) -> pd.DataFrame:
    """One row per eligible incident-day (the day's sitrep chosen as get_published_209 would)."""
    fires = incidents[incidents["START_YEAR"].between(*years) & (incidents["FINAL_ACRES"] >= min_final_acres)]
    reps = sitreps[sitreps["INCIDENT_ID"].isin(fires["INCIDENT_ID"])]
    counts = reps["INCIDENT_ID"].value_counts()
    reps = reps[reps["INCIDENT_ID"].map(counts) >= min_sitreps]
    first_day = reps.groupby("INCIDENT_ID")["REPORT_DATE"].transform("min")
    days = (reps.sort_values(["ACRES", "REPORT_TO_DATE"], ascending=False, na_position="last")
                .drop_duplicates(["INCIDENT_ID", "REPORT_DATE"]))
    first_day = first_day.loc[days.index]
    in_year = days["REPORT_DATE"].str[:4] == days["START_YEAR"].astype(int).astype(str)
    keep = ((days["REPORT_DATE"] != first_day) & in_year & (days["REPORT_DATE"] >= FIRMS_START) & days["ACRES"].notna() & _pct_ok(days)
            & (_has_text(days["REMARKS"]) | _has_text(days["CURRENT_THREAT_NARR"])))
    days = days[keep].merge(fires[["INCIDENT_ID", "FINAL_ACRES"]], on="INCIDENT_ID")
    return days.sort_values(["INCIDENT_ID", "REPORT_DATE"]).reset_index(drop=True)


PerimeterLookup = Callable[[str, str | None, int, str | None], list[dict]]


def attach_geomac(days: pd.DataFrame, lookup: PerimeterLookup = fetch_geomac_perimeters) -> list[dict]:
    """Candidate dicts for days with a GeoMAC perimeter on or before the date (one lookup per fire)."""
    candidates: list[dict] = []
    for incident_id, group in days.groupby("INCIDENT_ID", sort=True):
        first = group.iloc[0]
        year = int(first["START_YEAR"])
        state = first["POO_STATE"] if isinstance(first["POO_STATE"], str) else None
        derived = geomac_id(first["INCIDENT_NUMBER"], year)
        try:
            perimeters = lookup(ics209plus.normalize_name(first["INCIDENT_NAME"]), state, year, derived)
        except Exception as exc:  # one bad fire should not sink the run
            print(f"GeoMAC lookup failed for {incident_id}: {exc}")
            continue
        ids = Counter(p["id"] for p in perimeters if p.get("id"))
        fire_id = derived if derived in ids else (ids.most_common(1)[0][0] if ids else None)
        dates = [p["date"] for p in perimeters if p.get("date") and (p.get("id") == fire_id or fire_id is None)]
        for _, row in group.iterrows():
            day = row["REPORT_DATE"]
            to_date = sum(d <= day for d in dates)
            if to_date == 0:
                continue
            candidates.append(_candidate(row, fire_id, sum(d == day for d in dates), to_date))
    return candidates


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _candidate(row: pd.Series, fire_id: str | None, that_day: int, to_date: int) -> dict:
    return {
        "incident_id": row["INCIDENT_ID"],
        "incident_name": row["INCIDENT_NAME"],
        "state": _clean(row["POO_STATE"]),
        "year": int(row["START_YEAR"]),
        "report_date": row["REPORT_DATE"],
        "final_acres": _clean(row["FINAL_ACRES"]),
        "acres": _clean(row["ACRES"]),
        "new_acres": _clean(row["NEW_ACRES"]),
        "pct_contained": _clean(row["PCT_CONTAINED_COMPLETED"]),
        "personnel": _clean(row["TOTAL_PERSONNEL"]),
        "geomac_fire_id": fire_id,
        "n_perimeters_that_day": that_day,
        "n_perimeters_to_date": to_date,
    }


@app.command()
def candidates(out: Path = CANDIDATES_PATH, show: int = 40) -> None:
    """List eligible incident-days (2012-2014, >= 10k ac, >= 5 sitreps, GeoMAC perimeter by then)."""
    days = candidate_days(ics209plus.load_sitreps(DATA_DIR), ics209plus.load_incidents(DATA_DIR))
    print(f"{days['INCIDENT_ID'].nunique()} fires, {len(days)} eligible days before the GeoMAC check")
    found = attach_geomac(days)
    out.write_text("[\n" + ",\n".join(json.dumps(c) for c in found) + "\n]\n")
    fires = {c["incident_id"] for c in found}
    print(f"{len(found)} candidate days across {len(fires)} fires -> {out}")
    _print_table(sorted(found, key=_score, reverse=True)[:show])


def _print_table(rows: list[dict]) -> None:
    print(f"{'fire':<24} {'st':<3} {'date':<10} {'acres':>9} {'new':>8} {'pct':>4} {'perims':>6}")
    for c in rows:
        print(f"{c['incident_name'][:24]:<24} {c['state'] or '':<3} {c['report_date']} "
              f"{c['acres'] or 0:>9,.0f} {c['new_acres'] or 0:>8,.0f} {c['pct_contained'] or 0:>4.0f} "
              f"{c['n_perimeters_that_day']:>6}")


# ---------------------------------------------------------------- select


def _score(c: dict) -> tuple:
    multi = c["n_perimeters_that_day"] >= 2
    growth = (c["new_acres"] or 0) > 0
    return (2 * multi + growth, c["n_perimeters_that_day"], c["new_acres"] or 0)


def select_candidates(cands: list[dict], n: int = 20, max_per_fire: int = MAX_DAYS_PER_FIRE) -> list[dict]:
    """Greedy pick: best day of the best fire in each year first, then by score, <= 2 days per fire."""
    ranked = sorted(cands, key=lambda c: (_score(c), c["incident_id"], c["report_date"]), reverse=True)
    chosen: list[dict] = []

    def fits(c: dict) -> bool:
        same = [x for x in chosen if x["incident_id"] == c["incident_id"]]
        if len(same) >= max_per_fire or c in chosen:
            return False
        day = datetime.fromisoformat(c["report_date"])
        return all(abs((day - datetime.fromisoformat(x["report_date"])).days) >= MIN_DAYS_APART for x in same)

    for year in sorted({c["year"] for c in ranked}):
        best = next(c for c in ranked if c["year"] == year)
        chosen.append(best)
    # breadth first: one day per new fire, then second days
    for limit in range(1, max_per_fire + 1):
        for c in ranked:
            if len(chosen) >= n:
                break
            if sum(x["incident_id"] == c["incident_id"] for x in chosen) < limit and fits(c):
                chosen.append(c)
    return sorted(chosen, key=lambda c: (c["report_date"], c["incident_name"]))


def slug_for(incident_name: str, state: str | None, report_date: str) -> str:
    name = "-".join(ics209plus.normalize_name(incident_name).lower().split())
    return f"{name}-{(state or 'xx').lower()}-{report_date}"


def to_example(published: ics209plus.Published209, cand: dict) -> dict:
    return {
        "inputs": {"incident_name": cand["incident_name"].title(), "report_date": cand["report_date"],
                   "state": cand["state"]},
        "outputs": {"fields": published.fields(), "narrative": published.narrative,
                    "known_conflicts": [], "source": "ics209plus", "planted_evidence": None},
        "metadata": {"incident_id": cand["incident_id"], "year": cand["year"],
                     "final_acres": cand["final_acres"], "geomac_fire_id": cand["geomac_fire_id"],
                     "new_acres": cand["new_acres"], "n_perimeters_that_day": cand["n_perimeters_that_day"],
                     "tool_errors": [],
                     "slug": slug_for(cand["incident_name"], cand["state"], cand["report_date"])},
    }


def build_examples(chosen: list[dict],
                   lookup: Callable[..., ics209plus.Published209 | None] = ics209plus.get_published_209) -> list[dict]:
    """Reference outputs straight from get_published_209, so evaluators and dataset agree."""
    examples = []
    for cand in chosen:
        published = lookup(cand["incident_name"], cand["report_date"], cand["state"])
        if published is None or published.incident_id != cand["incident_id"]:
            print(f"skip {cand['incident_name']} {cand['report_date']}: get_published_209 picks another row")
            continue
        examples.append(to_example(published, cand))
    return examples


@app.command()
def select(n: int = typer.Option(20, min=15, max=25), candidates_path: Path = CANDIDATES_PATH,
           out: Path = DATASET_PATH) -> None:
    """Choose 15-25 incident-days across >= 8 fires and all three years -> evals/dataset.json."""
    cands = json.loads(candidates_path.read_text())
    chosen = select_candidates(cands, n)
    examples = build_examples(chosen)
    fires = {e["metadata"]["incident_id"] for e in examples}
    years = {e["metadata"]["year"] for e in examples}
    if len(fires) < MIN_FIRES or len(years) < YEARS[1] - YEARS[0] + 1:
        print(f"WARNING: only {len(fires)} fires / years {sorted(years)}")
    _write(out, examples)
    _print_table([c for c in chosen if any(e["metadata"]["incident_id"] == c["incident_id"]
                                           and e["inputs"]["report_date"] == c["report_date"] for e in examples)])
    print(f"{len(examples)} examples, {len(fires)} fires, years {sorted(years)} -> {out}")


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------- confirm

SEVERITY_RANK = {"info": 0, "warn": 1, "block": 2}


def summarize_conflicts(conflicts: list[dict]) -> list[dict]:
    """One {field, severity} per field, keeping the worst severity."""
    worst: dict[str, str] = {}
    for c in conflicts:
        field, severity = c["field"], c["severity"]
        if field not in worst or SEVERITY_RANK[severity] > SEVERITY_RANK[worst[field]]:
            worst[field] = severity
    return [{"field": f, "severity": s} for f, s in sorted(worst.items())]


def confirm_examples(examples: list[dict], run: Callable[..., Any]) -> int:
    """Fill known_conflicts / tool_errors from a live conflicts run. Returns days with a conflict."""
    confirmed = 0
    for ex in examples:
        if ex["outputs"]["source"] == "planted":
            continue
        inputs = ex["inputs"]
        try:
            result = run(inputs["incident_name"], inputs["report_date"], inputs["state"])
        except Exception as exc:
            ex["outputs"]["known_conflicts"] = []
            ex["metadata"]["tool_errors"] = [{"tool": "resolve", "message": " ".join(str(exc).split())[:300]}]
            ex["metadata"]["confirmed"] = False
            continue
        ex["outputs"]["known_conflicts"] = summarize_conflicts(list(result.conflicts))
        ex["metadata"]["tool_errors"] = [
            e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in result.errors
        ]
        ex["metadata"]["confirmed"] = True
        confirmed += bool(ex["outputs"]["known_conflicts"])
    return confirmed


@app.command()
def confirm(path: Path = DATASET_PATH) -> None:
    """Run run_conflicts on every real example and record what it finds."""
    from firebinder.nodes.conflicts import run_conflicts

    examples = json.loads(path.read_text())
    confirmed = confirm_examples(examples, run_conflicts)
    _write(path, examples)
    real = sum(e["outputs"]["source"] != "planted" for e in examples)
    print(f"{confirmed} of {real} real examples have at least one conflict")
    if confirmed < MIN_CONFIRMED:
        print(f"WARNING: the spec wants at least {MIN_CONFIRMED} confirmed-disagreement days")


# ---------------------------------------------------------------- plant


def _planted_evidence(field: str, value: float, slug: str, report_date: str) -> dict:
    return {"field": field, "value": value, "source": "get_inciweb_updates",
            "url": f"planted://inciweb/{slug}", "fetched_at": f"{report_date}T18:00:00+00:00"}


def plant_one(example: dict, kind: str) -> dict:
    """Copy an example and plant one InciWeb-style conflict: 'acres', 'pct_contained' or 'personnel'."""
    from firebinder.nodes.reconcile import acreage_severity, acreage_tolerance

    ex = json.loads(json.dumps(example))
    fields = ex["outputs"]["fields"]
    date = ex["inputs"]["report_date"]
    if kind == "acres":
        real = float(fields["acres"])
        planted = float(round(real * 1.6))
        severity = acreage_severity(abs(planted - real), acreage_tolerance(real, planted))
    elif kind == "pct_contained":
        real = float(fields["pct_contained"])
        planted = real + 15 if real <= 85 else real - 15
        severity = "warn"
    elif kind == "personnel":
        real = int(fields["personnel"])
        planted = real + max(25, round(real * 0.2))
        severity = "warn"
    else:
        raise ValueError(f"unknown plant kind {kind!r}")
    slug = f"{ex['metadata']['slug']}-{kind.replace('_', '-')}"
    ex["metadata"].update(slug=slug, planted_from=example["metadata"]["slug"], tool_errors=[])
    ex["outputs"].update(source="planted", known_conflicts=[{"field": kind, "severity": severity}],
                         planted_evidence=[_planted_evidence(kind, planted, slug, date)])
    return ex


PLANT_KINDS = ("acres", "pct_contained", "personnel")


def plant_examples(examples: list[dict]) -> list[dict]:
    """Three planted examples from three different real fires that have the needed field."""
    real = [e for e in examples if e["outputs"]["source"] != "planted"]
    used: set[str] = set()
    planted = []
    for kind in PLANT_KINDS:
        donor = next((e for e in real if e["metadata"]["incident_id"] not in used
                      and e["outputs"]["fields"].get(kind) is not None), None)
        if donor is None:
            raise ValueError(f"no example with {kind} to plant from")
        used.add(donor["metadata"]["incident_id"])
        planted.append(plant_one(donor, kind))
    return planted


@app.command()
def plant(path: Path = DATASET_PATH, fixtures_dir: Path = FIXTURES_DIR) -> None:
    """Add 3 planted-conflict examples (replacing earlier planted ones)."""
    examples = [e for e in json.loads(path.read_text()) if e["outputs"]["source"] != "planted"]
    planted = plant_examples(examples)
    for ex in planted:
        _write(fixtures_dir / f"planted_{ex['metadata']['slug']}.json", ex)
        print(f"planted {ex['metadata']['slug']}: {ex['outputs']['known_conflicts']}")
    _write(path, examples + planted)


# ---------------------------------------------------------------- upload


@app.command()
def upload(dataset: str = "firebinder-historical", path: Path = DATASET_PATH) -> None:
    """Replace the LangSmith dataset's examples with evals/dataset.json."""
    from langsmith import Client

    client = Client()
    examples = json.loads(path.read_text())
    if client.has_dataset(dataset_name=dataset):
        ds = client.read_dataset(dataset_name=dataset)
        old = [e.id for e in client.list_examples(dataset_id=ds.id)]
        if old:
            client.delete_examples(example_ids=old)
    else:
        ds = client.create_dataset(dataset, description="FireBinder historical incident-days (ICS209-PLUS 2012-2014)")
    client.create_examples(
        dataset_id=ds.id,
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        metadata=[e["metadata"] for e in examples],
    )
    print(f"uploaded {len(examples)} examples to {dataset}: {getattr(ds, 'url', None) or ds.id}")


if __name__ == "__main__":
    app()
