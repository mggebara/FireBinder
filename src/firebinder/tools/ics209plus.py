"""get_published_209: the real daily ICS-209 for a past incident-day, from ICS209-PLUS.

Historical ground truth for the `vs_published_209` judge evaluator and for building the
`firebinder-historical` dataset. It is deliberately NOT in `LIVE_TOOLS` and must never be bound
to the gather node: the drafter would simply copy the answer.

Source: ICS209-PLUS cleaned databases, 1999-2014 (figshare article 8048252). The 26.7 MB zip is
downloaded once into `data/ics209plus/` (gitignored); the 152 MB sitrep table is then trimmed to
the columns used here and cached as `sitreps_trimmed.csv` so later loads take about a second.

Data notes (checked against the 2014 release):
- One row per daily 209; several rows can share a calendar day. REPORT_FROM_DATE is mostly empty,
  so the report day is the calendar date of REPORT_TO_DATE (local incident time, as published).
- POO_STATE is blank on some rows (e.g. the 2013 Rim fire); it is filled from the INCIDENT_NUMBER
  unit prefix ("CA-STF-002857" -> "CA").
- There is no COMPLEX_NAME column; COMPLEX is a boolean. SIGNIF_EVENTS_SUMMARY exists and is
  included in the narrative.
- PCT_CONTAINED_COMPLETED is a percentage only when its unit is PCT (or blank); MMA (miles) rows
  give pct_contained None.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from firebinder.state import Evidence
from firebinder.tools._base import SourceStamped, clip, fmt, utc_now_iso
from firebinder.tools._http import get_client

TOOL_NAME = "get_published_209"
SOURCE = "ics209plus"
ARTICLE_URL = "https://figshare.com/articles/dataset/ICS209-PLUS_Cleaned_databases/8048252"
ZIP_URL = "https://ndownloader.figshare.com/files/21491724"
DEFAULT_DATA_DIR = Path("data/ics209plus")
BUNDLE_DIR = "ics209-plus-wildfire"
SITREPS_FILE = "ics209-plus-wf_sitreps_1999to2014.csv"
INCIDENTS_FILE = "ics209-plus-wf_incidents_1999to2014.csv"
TRIMMED_FILE = "sitreps_trimmed.csv"
ENCODING = "latin-1"

SITREP_COLUMNS = [
    "INCIDENT_ID", "INCIDENT_NUMBER", "INC_IDENTIFIER", "INCIDENT_NAME", "COMPLEX", "START_YEAR",
    "POO_STATE", "POO_LATITUDE", "POO_LONGITUDE", "DISCOVERY_DATE",
    "REPORT_FROM_DATE", "REPORT_TO_DATE", "REPORT_DOY",
    "ACRES", "NEW_ACRES", "PCT_CONTAINED_COMPLETED", "PCT_CONT_COMPL_UOM_ABBREV", "TOTAL_PERSONNEL",
    "STR_THREATENED", "STR_DAMAGED", "STR_DESTROYED", "EVACUATION_IN_PROGRESS", "NUM_EVACUATED",
    "REMARKS", "SIGNIF_EVENTS_SUMMARY", "CURRENT_THREAT_NARR", "PROJECTED_ACTIVITY_NARR",
    "WEATHER_CONCERNS_NARR", "STRATEGIC_NARR", "CRIT_RES_NEEDS_NARR", "LL_UPDATE",
]
DATE_COLUMNS = ["DISCOVERY_DATE", "REPORT_FROM_DATE", "REPORT_TO_DATE"]

_sitreps: pd.DataFrame | None = None


def ensure_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and unzip the ICS209-PLUS bundle if the CSVs are missing; return the bundle folder."""
    bundle = data_dir / BUNDLE_DIR
    if (bundle / SITREPS_FILE).exists() and (bundle / INCIDENTS_FILE).exists():
        return bundle
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ics209-plus-wildfire.zip"
    if not zip_path.exists():
        with get_client().stream("GET", ZIP_URL, timeout=300) as response:
            response.raise_for_status()
            size = int(response.headers.get("content-length", 0))
            print(f"Downloading ICS209-PLUS ({size / 1e6:.1f} MB) from {ZIP_URL}")
            with zip_path.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(BUNDLE_DIR + "/") and ".." not in m]
        zf.extractall(data_dir, members=members)
    return bundle


def _state_from_incident_number(number: object) -> str | None:
    if isinstance(number, str) and len(number) >= 2 and number[:2].isalpha():
        return number[:2].upper()
    return None


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    missing = df["POO_STATE"].isna()
    df.loc[missing, "POO_STATE"] = df.loc[missing, "INCIDENT_NUMBER"].map(_state_from_incident_number)
    df["REPORT_DATE"] = df["REPORT_TO_DATE"].dt.strftime("%Y-%m-%d")
    df["NAME_KEY"] = df["INCIDENT_NAME"].map(normalize_name)
    return df


def load_sitreps(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """All wildfire sitreps (trimmed columns, parsed dates). Cached in memory and on disk."""
    global _sitreps
    if _sitreps is not None:
        return _sitreps
    trimmed = data_dir / TRIMMED_FILE
    if trimmed.exists():
        df = pd.read_csv(trimmed, low_memory=False)
    else:
        bundle = ensure_dataset(data_dir)
        df = pd.read_csv(bundle / SITREPS_FILE, usecols=SITREP_COLUMNS, encoding=ENCODING, low_memory=False)
        df = df[SITREP_COLUMNS]
        df.to_csv(trimmed, index=False)
    _sitreps = _prepare(df)
    return _sitreps


def load_incidents(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """One row per incident (only the columns the dataset builder needs)."""
    bundle = ensure_dataset(data_dir)
    columns = ["INCIDENT_ID", "INCIDENT_NUMBER", "INCIDENT_NAME", "FINAL_ACRES", "DISCOVERY_DATE",
               "START_YEAR", "POO_STATE", "POO_LATITUDE", "POO_LONGITUDE", "WF_PEAK_PERSONNEL",
               "WF_MAX_GROWTH_DATE", "STR_DESTROYED_TOTAL"]
    return pd.read_csv(bundle / INCIDENTS_FILE, usecols=columns, encoding=ENCODING, low_memory=False)


def reset_cache() -> None:
    global _sitreps
    _sitreps = None


def normalize_name(name: object) -> str:
    """Upper-case, collapse whitespace, drop a trailing ' FIRE' ("Rim Fire" -> "RIM")."""
    if not isinstance(name, str):
        return ""
    text = " ".join(name.upper().split())
    return text[: -len(" FIRE")] if text.endswith(" FIRE") else text


class Published209(SourceStamped):
    incident_id: str
    incident_name: str
    state: str | None
    report_from: str | None
    report_to: str
    report_date: str
    acres: float | None
    pct_contained: float | None
    personnel: int | None
    structures_threatened: int | None
    structures_destroyed: int | None
    structures_damaged: int | None
    evacuation_in_progress: bool | None
    num_evacuated: int | None
    remarks: str | None
    significant_events: str | None
    current_threat: str | None
    projected_activity: str | None
    weather_concerns: str | None
    narrative: str | None
    poo_lat: float | None
    poo_lon: float | None
    discovery_date: str | None
    source: str = SOURCE
    url: str = ARTICLE_URL

    def fields(self) -> dict:
        """The five numeric fields the dataset and evaluators compare."""
        return {
            "acres": self.acres,
            "pct_contained": self.pct_contained,
            "personnel": self.personnel,
            "structures_threatened": self.structures_threatened,
            "structures_destroyed": self.structures_destroyed,
        }

    def to_evidence(self) -> list[Evidence]:
        return [self._evidence(field, value) for field, value in self.fields().items() if value is not None]

    def summary(self) -> str:
        return (
            f"Published ICS-209 for {self.incident_name} ({self.state or '??'}) on {self.report_date}: "
            f"{fmt(self.acres, ' ac')}, {fmt(self.pct_contained, '%')} contained, "
            f"{fmt(self.personnel)} personnel, {fmt(self.structures_threatened)} structures threatened, "
            f"{fmt(self.structures_destroyed)} destroyed. Narrative: {clip(self.narrative)}"
        )


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _int(value: object) -> int | None:
    number = _number(value)
    return int(round(number)) if number is not None else None


def parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "Y", "YES", "1", "1.0"}:
        return True
    if text in {"FALSE", "F", "N", "NO", "0", "0.0"}:
        return False
    return None


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def build_narrative(remarks: str | None, significant_events: str | None,
                    current_threat: str | None, projected_activity: str | None) -> str | None:
    sections = [
        ("Remarks", remarks),
        ("Significant events", significant_events),
        ("Current threat", current_threat),
        ("Projected activity", projected_activity),
    ]
    parts = [f"{heading}:\n{text}" for heading, text in sections if text]
    return "\n\n".join(parts) or None


def row_to_published(row: pd.Series) -> Published209:
    pct_unit = row.get("PCT_CONT_COMPL_UOM_ABBREV")
    pct = _number(row.get("PCT_CONTAINED_COMPLETED"))
    if isinstance(pct_unit, str) and pct_unit.strip().upper() != "PCT":
        pct = None
    remarks = _text(row.get("REMARKS"))
    significant = _text(row.get("SIGNIF_EVENTS_SUMMARY"))
    threat = _text(row.get("CURRENT_THREAT_NARR"))
    projected = _text(row.get("PROJECTED_ACTIVITY_NARR"))
    state = row.get("POO_STATE")
    return Published209(
        incident_id=str(row["INCIDENT_ID"]),
        incident_name=str(row["INCIDENT_NAME"]),
        state=state if isinstance(state, str) else None,
        report_from=_iso(row.get("REPORT_FROM_DATE")),
        report_to=_iso(row["REPORT_TO_DATE"]) or "",
        report_date=str(row["REPORT_DATE"]),
        acres=_number(row.get("ACRES")),
        pct_contained=pct,
        personnel=_int(row.get("TOTAL_PERSONNEL")),
        structures_threatened=_int(row.get("STR_THREATENED")),
        structures_destroyed=_int(row.get("STR_DESTROYED")),
        structures_damaged=_int(row.get("STR_DAMAGED")),
        evacuation_in_progress=parse_bool(row.get("EVACUATION_IN_PROGRESS")),
        num_evacuated=_int(row.get("NUM_EVACUATED")),
        remarks=remarks,
        significant_events=significant,
        current_threat=threat,
        projected_activity=projected,
        weather_concerns=_text(row.get("WEATHER_CONCERNS_NARR")),
        narrative=build_narrative(remarks, significant, threat, projected),
        poo_lat=_number(row.get("POO_LATITUDE")),
        poo_lon=_number(row.get("POO_LONGITUDE")),
        discovery_date=_iso(row.get("DISCOVERY_DATE")),
        fetched_at=utc_now_iso(),
    )


def find_sitreps(sitreps: pd.DataFrame, incident_name: str, report_date: str,
                 state: str | None = None) -> pd.DataFrame:
    """Rows reported on that calendar date whose name equals or contains the query name."""
    key = normalize_name(incident_name)
    if not key:
        return sitreps.iloc[0:0]
    day = sitreps[sitreps["REPORT_DATE"] == report_date]
    if state:
        day = day[day["POO_STATE"].fillna("").str.upper() == state.upper()]
    return day[day["NAME_KEY"].str.contains(key, regex=False, na=False)]


def pick_sitrep(matches: pd.DataFrame, incident_name: str) -> pd.Series | None:
    """Exact name first, then the largest ACRES, then the latest report time."""
    if matches.empty:
        return None
    ranked = matches.assign(
        _exact=matches["NAME_KEY"] == normalize_name(incident_name),
        _acres=matches["ACRES"].fillna(-1),
    ).sort_values(["_exact", "_acres", "REPORT_TO_DATE"], ascending=False)
    return ranked.iloc[0]


def get_published_209(incident_name: str, report_date: str, state: str | None = None,
                      data_dir: Path = DEFAULT_DATA_DIR) -> Published209 | None:
    """The published 209 for an incident on a calendar date (YYYY-MM-DD), or None if none was filed."""
    row = pick_sitrep(find_sitreps(load_sitreps(data_dir), incident_name, report_date, state), incident_name)
    return row_to_published(row) if row is not None else None


class Published209Args(BaseModel):
    incident_name: str = Field(description="Incident name, e.g. 'Rim'")
    report_date: str = Field(description="Report date, YYYY-MM-DD (1999-2014)")
    state: str | None = Field(default=None, description="Two-letter state code, e.g. 'CA'")


@tool(TOOL_NAME, args_schema=Published209Args, response_format="content_and_artifact")
def get_published_209_tool(incident_name: str, report_date: str,
                           state: str | None = None) -> tuple[str, Published209 | None]:
    """Get the real published ICS-209 for a past incident-day (1999-2014). Evaluation only."""
    published = get_published_209(incident_name, report_date, state)
    if published is None:
        return f"No published 209 for {incident_name} on {report_date}.", None
    return published.summary(), published
