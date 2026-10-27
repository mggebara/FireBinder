"""Offline tests for the ICS209-PLUS tool and the dataset builder (12-row CSV fixture)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from evals import build_dataset as bd
from firebinder.tools import LIVE_TOOLS, ics209plus

FIXTURES = Path(__file__).parent / "fixtures_ics209plus"


@pytest.fixture
def data_dir(tmp_path):
    shutil.copy(FIXTURES / "sitreps_trimmed.csv", tmp_path / ics209plus.TRIMMED_FILE)
    ics209plus.reset_cache()
    yield tmp_path
    ics209plus.reset_cache()


def g(name, date, state=None, data_dir=None):
    return ics209plus.get_published_209(name, date, state, data_dir=data_dir)


def test_exact_date_match_and_narrative(data_dir):
    p = g("Alpha Fire", "2013-08-02", "ca", data_dir)
    assert p is not None
    assert (p.incident_id, p.state, p.report_date) == ("2013_CA-XXX-000001_ALPHA", "CA", "2013-08-02")
    assert p.fields() == {"acres": 3000.0, "pct_contained": 10.0, "personnel": 300,
                          "structures_threatened": 40, "structures_destroyed": 2}
    assert p.evacuation_in_progress is True and p.num_evacuated == 120
    assert p.narrative == (
        "Remarks:\nfire crossed the ridge\n\nSignificant events:\nspotting half a mile\n\n"
        "Current threat:\nhomes on the east flank\n\nProjected activity:\ncontinued growth east"
    )
    assert p.source == "ics209plus" and p.url == ics209plus.ARTICLE_URL
    assert {e["field"] for e in p.to_evidence()} == set(p.fields())


def test_no_match(data_dir):
    assert g("Alpha", "2013-07-30", data_dir=data_dir) is None
    assert g("Alpha", "2013-08-02", "OR", data_dir) is None
    assert g("Nonexistent", "2013-08-02", data_dir=data_dir) is None


def test_same_day_reports_pick_largest_acres(data_dir):
    p = g("Alpha", "2013-08-03", data_dir=data_dir)
    assert p.acres == 5000 and p.remarks == "evening report"
    assert p.evacuation_in_progress is False
    assert p.narrative == "Remarks:\nevening report"


def test_complex_tie_break(data_dir):
    assert g("Bravo", "2013-08-03", "CA", data_dir).incident_name == "BRAVO"  # exact beats bigger complex
    assert g("Bravo", "2013-08-03", "NV", data_dir).acres == 700
    assert g("Charlie", "2014-07-10", data_dir=data_dir).incident_name == "CHARLIE WEST"  # largest acres


def test_state_fill_pct_units_and_bools(data_dir):
    p = g("Delta", "2012-08-15", data_dir=data_dir)
    assert p.state == "MT"  # from INCIDENT_NUMBER
    assert p.pct_contained is None  # MMA, not a percentage
    assert p.evacuation_in_progress is True
    assert p.narrative is None
    assert g("Charlie West", "2014-07-10", data_dir=data_dir).evacuation_in_progress is None
    assert ics209plus.parse_bool("N") is False and ics209plus.parse_bool("maybe") is None


def test_tool_not_live():
    assert ics209plus.get_published_209_tool.name == "get_published_209"
    assert "get_published_209" not in {t.name for t in LIVE_TOOLS}


def _days(data_dir):
    return bd.candidate_days(ics209plus.load_sitreps(data_dir), pd.read_csv(FIXTURES / "incidents.csv"))


def test_candidate_filtering(data_dir):
    days = _days(data_dir)
    # ALPHA only: first day, no-pct day and no-narrative day dropped; BRAVO COMPLEX has 1 sitrep
    assert list(zip(days["INCIDENT_ID"], days["REPORT_DATE"])) == [
        ("2013_CA-XXX-000001_ALPHA", "2013-08-02"), ("2013_CA-XXX-000001_ALPHA", "2013-08-03")]
    assert days.loc[1, "ACRES"] == 5000


def test_geomac_id():
    assert bd.geomac_id("CA-STF-002857", 2013) == "2013-CASTF-002857"
    assert bd.geomac_id(None, 2013) is None


def test_attach_geomac_counts_perimeters(data_dir):
    calls = []

    def lookup(name, state, year, fire_id):
        calls.append((name, state, year, fire_id))
        return [{"id": fire_id, "date": "2013-08-02"}, {"id": fire_id, "date": "2013-08-03"},
                {"id": fire_id, "date": "2013-08-03"}, {"id": "other", "date": "2013-08-01"}]

    cands = bd.attach_geomac(_days(data_dir), lookup)
    assert calls == [("ALPHA", "CA", 2013, "2013-CAXXX-000001")]
    assert [(c["report_date"], c["n_perimeters_that_day"], c["n_perimeters_to_date"]) for c in cands] == [
        ("2013-08-02", 1, 1), ("2013-08-03", 2, 3)]
    assert cands[0]["geomac_fire_id"] == "2013-CAXXX-000001"


def test_attach_geomac_drops_days_before_first_perimeter(data_dir):
    cands = bd.attach_geomac(_days(data_dir), lambda *a: [{"id": "X", "date": "2013-08-03"}])
    assert [c["report_date"] for c in cands] == ["2013-08-03"]
    assert cands[0]["geomac_fire_id"] == "X"


def _cand(i, year, day, perims=1, new=0.0):
    return {"incident_id": f"F{i}", "incident_name": f"FIRE{i}", "state": "CA", "year": year,
            "report_date": f"{year}-08-{day:02d}", "final_acres": 20000, "acres": 1000, "new_acres": new,
            "pct_contained": 10, "personnel": 100, "geomac_fire_id": None,
            "n_perimeters_that_day": perims, "n_perimeters_to_date": perims}


def test_select_spreads_fires_years_and_favours_multi_perimeter_growth():
    cands = [_cand(i, 2012 + i % 3, d, perims=2 if d == 10 else 1, new=100.0 if d != 12 else 0.0)
             for i in range(10) for d in (10, 11, 12, 20)]
    chosen = bd.select_candidates(cands, n=15)
    assert len(chosen) == 15
    assert len({c["incident_id"] for c in chosen}) == 10
    assert {c["year"] for c in chosen} == {2012, 2013, 2014}
    assert all(sum(c["incident_id"] == f"F{i}" for c in chosen) <= 2 for i in range(10))
    assert all(c["report_date"].endswith(("-10", "-20")) for c in chosen)  # day 11 is < 3 days from 10


def test_build_examples_contract(data_dir):
    cands = bd.attach_geomac(_days(data_dir), lambda *a: [{"id": "G", "date": "2013-08-01"}])
    ex = bd.build_examples(cands, lambda n, d, s: g(n, d, s, data_dir))
    assert len(ex) == 2
    e = ex[0]
    assert e["inputs"] == {"incident_name": "Alpha", "report_date": "2013-08-02", "state": "CA"}
    assert set(e["outputs"]) == {"fields", "narrative", "known_conflicts", "source", "planted_evidence"}
    assert set(e["outputs"]["fields"]) == set(bd.FIELD_NAMES)
    assert e["outputs"]["source"] == "ics209plus" and e["outputs"]["planted_evidence"] is None
    assert {"incident_id", "year", "final_acres", "geomac_fire_id", "new_acres",
            "n_perimeters_that_day", "tool_errors"} <= set(e["metadata"])
    json.dumps(ex)


def _example(i, acres=10000.0, pct=30.0, personnel=400):
    return {"inputs": {"incident_name": f"Fire{i}", "report_date": "2013-08-02", "state": "CA"},
            "outputs": {"fields": {"acres": acres, "pct_contained": pct, "personnel": personnel,
                                   "structures_threatened": 0, "structures_destroyed": 0},
                        "narrative": "x", "known_conflicts": [], "source": "ics209plus",
                        "planted_evidence": None},
            "metadata": {"incident_id": f"I{i}", "slug": f"fire{i}-ca-2013-08-02", "tool_errors": []}}


def test_plant_examples_known_conflicts():
    planted = bd.plant_examples([_example(1), _example(2, pct=90.0), _example(3)])
    got = [(p["outputs"]["known_conflicts"], p["outputs"]["planted_evidence"][0]) for p in planted]
    assert got[0][0] == [{"field": "acres", "severity": "block"}] and got[0][1]["value"] == 16000.0
    assert got[1][0] == [{"field": "pct_contained", "severity": "warn"}] and got[1][1]["value"] == 75.0
    assert got[2][0] == [{"field": "personnel", "severity": "warn"}] and got[2][1]["value"] == 480
    for p, (_, ev) in zip(planted, got):
        assert p["outputs"]["source"] == "planted"
        assert ev["source"] == "get_inciweb_updates" and ev["url"].startswith("planted://inciweb/")
        assert set(ev) == {"field", "value", "source", "url", "fetched_at"}
    assert len({p["metadata"]["incident_id"] for p in planted}) == 3


def test_confirm_merges_mocked_run_conflicts():
    examples = [_example(1), _example(2), bd.plant_one(_example(3), "personnel")]

    def run(name, date, state):
        if name == "Fire2":
            raise RuntimeError("no incident record found")
        return SimpleNamespace(
            conflicts=[{"field": "acres", "severity": "warn"}, {"field": "acres", "severity": "block"},
                       {"field": "pct_contained", "severity": "info"}],
            errors=[SimpleNamespace(model_dump=lambda: {"tool": "get_weather", "message": "historical"})],
        )

    assert bd.confirm_examples(examples, run) == 1
    assert examples[0]["outputs"]["known_conflicts"] == [
        {"field": "acres", "severity": "block"}, {"field": "pct_contained", "severity": "info"}]
    assert examples[0]["metadata"]["tool_errors"] == [{"tool": "get_weather", "message": "historical"}]
    assert examples[1]["outputs"]["known_conflicts"] == []
    assert examples[1]["metadata"]["tool_errors"][0]["tool"] == "resolve"
    assert examples[2]["outputs"]["known_conflicts"] == [{"field": "personnel", "severity": "warn"}]
