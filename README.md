# FireBinder

FireBinder drafts a wildfire incident's daily ICS-209 status summary from public data
and then checks its own work. Every number in the draft points at the source it came
from. The sources are compared against each other before a word is written. A second,
stronger model audits the draft claim by claim and sends it back if anything is
unsupported. A person approves, rejects, or edits the result before it is final.

## Who it's for

The ICS-209 is the daily situation report every large US wildfire files: acres,
containment, personnel, structures, evacuations, and a short narrative. Someone on the
incident, usually in the situation unit, assembles it every day from feeds that do not
agree with each other. The perimeter polygon's area rarely matches the reported
acreage. InciWeb runs hours or days behind the incident record. Personnel counts differ
between feeds by a factor of two or three. Finding those gaps by hand, every day, on a
fire with a hundred moving numbers, is slow and easy to get wrong.

FireBinder is for the people who do that reconciliation, and for anyone downstream who
reads a 209 and wants to know where a number came from:

- **Situation unit leaders and PIOs** get a first draft with a citation on every figure
  and an explicit list of where the public feeds disagree, so the review is about
  judgment calls rather than data entry.
- **Analysts and researchers** working with wildfire data get a conflict detector that
  runs without any model and prints exactly which sources disagree and by how much.
- **Anyone building on the same public feeds** gets four typed, tested clients for
  WFIGS, FIRMS, InciWeb, and the NWS API, with the current quirks of each already
  handled.

The agent never finalizes anything on its own. It stops and waits for a person.

## What it does

```
resolve -> gather -> reconcile -> draft -> audit --+-- fails and revision < 3 --> draft (revision + 1)
                                                   |
                                                   +-- passes, or revision == 3 --> human_review -> finalize
```

| Step | What happens | Model |
| --- | --- | --- |
| resolve | Name and date to an IRWIN ID and centroid, via the WFIGS incident layer. Fails loudly if the name is ambiguous. | none |
| gather | A tool-calling loop over the four sources below. Stops when every required field has evidence or every tool has been tried once. | Haiku 5.5 |
| reconcile | Plain Python. Groups evidence by field and applies tolerance rules. Also computes the perimeter polygon's area and compares it to the reported acreage. | none |
| draft | Writes the narrative and fills the field table, citing an evidence URL for every number. Must mention every warn or block conflict. A draft with an uncited number is rejected before the audit ever sees it. | Haiku 5.5 |
| audit | Extracts every factual claim from the draft and labels it supported, unsupported, or contradicted against the gathered evidence. No tools. | Opus 5.5 |
| route | Any contradicted claim, or more than one unsupported, sends the draft back with the findings attached. Three revisions max. | none |
| human_review | `interrupt()`. The run pauses in a SQLite checkpoint until someone approves, rejects, or edits. | none |
| finalize | Writes `out/<irwin_id>/<date>/209.md`, `evidence.json`, `conflicts.json`, `audit.json`. | none |

### Sources

| Tool | Source | Returns |
| --- | --- | --- |
| `get_incident_record` | NIFC WFIGS current incident locations and interagency perimeters (ArcGIS REST, no key) | IRWIN daily acres, GIS acres, percent contained, personnel, discovery date, perimeter polygon |
| `get_hotspots` | NASA FIRMS area API (free MAP_KEY) | VIIRS detections near the centroid for the last few days, count, hull area, max FRP |
| `get_inciweb_updates` | InciWeb incident page and national feeds (no key) | Size, containment, personnel, evacuation text, latest update |
| `get_weather` | NWS gridpoint forecast and active alerts (no key, User-Agent required) | Forecast periods with wind and minimum RH, Red Flag Warnings |

Every tool returns a Pydantic model stamped with the source URL and fetch time, and
exposes it as a list of `(field, value, source, url, fetched_at)` evidence records.

### Conflict rules

| Field | Tolerance | Severity |
| --- | --- | --- |
| acres (IRWIN daily, GIS, InciWeb) | 10% of the larger value or 500 acres, whichever is larger | warn outside tolerance, block beyond 3x |
| polygon area vs reported acres | same as acres | warn, block beyond 3x |
| pct_contained | 5 points | warn, block beyond 20 points |
| personnel, structures threatened, structures destroyed | exact | warn |
| evacuation_status text | normalized string match | info |

A field with evidence from only one source never conflicts. Polygon area uses a local
equirectangular projection around the perimeter's centroid, which is accurate to well
under one percent at the size of a fire.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Fill in `.env`:

- `ANTHROPIC_API_KEY` for the draft and audit models.
- `FIRMS_MAP_KEY`, free from [firms.modaps.eosdis.nasa.gov/api](https://firms.modaps.eosdis.nasa.gov/api/).
- `LANGSMITH_API_KEY` with `LANGSMITH_TRACING=true` if you want traces. Optional.

## Use

Conflict detection only. No model calls, no API key beyond FIRMS.

```bash
uv run firebinder conflicts --incident GRASSHOPPER --date 2026-10-26 --state OR
```

Output from that command on 26 October 2026, trimmed:

```
GRASSHOPPER  IRWIN {6AA4C355-AD18-472F-AAAB-78BAA9DE31B1}  state US-OR  centroid (45.2530, -121.5101)

field               value                       source
acres               93972.0                     get_incident_record
gis_acres           93971.25                    get_incident_record
pct_contained       98.0                        get_incident_record
personnel           85                          get_incident_record
perimeter_geometry  MultiPolygon (107 parts)    get_incident_record
hotspot_count       0                           get_hotspots
acres               93972.0                     get_inciweb_updates
pct_contained       98.0                        get_inciweb_updates
personnel           218                         get_inciweb_updates
red_flag_warning    False                       get_weather

Conflicts
warn  personnel  personnel: IRWIN reports 85 but InciWeb reports 218
```

Add `--json` to get the evidence, conflicts, and tool errors as JSON, geometry included.

The full graph:

```bash
uv run firebinder draft --incident "LITTLE GIANT" --date 2026-10-26 --state WA
```

This prints the draft, the field table with citations, the conflicts, and the audit
findings, then pauses. Pick it up later from the checkpoint:

```bash
uv run firebinder resume --thread-id little-giant-2026-10-26 --approve
uv run firebinder resume --thread-id little-giant-2026-10-26 --reject
uv run firebinder resume --thread-id little-giant-2026-10-26 --edit my_draft.json
```

`--skip-review` approves without pausing, for batch evaluation runs.

## Two runs from 26 October 2026

Both fires were nearly contained, so the interesting disagreements were about people,
not acres.

| Incident | Drafts | Audit result | Time | Tokens | Cost |
| --- | --- | --- | --- | --- | --- |
| GRASSHOPPER (OR, 93,972 ac, 98%) | 1 | 28 supported, 1 unsupported | 38 s | 19.8k | $0.09 |
| LITTLE GIANT (WA, 172,912 ac, 95%) | 2 | revision 0 rejected, revision 1: 27 supported | 72 s | 41.6k | $0.18 |

On GRASSHOPPER the reconcile step flagged personnel (WFIGS 85, InciWeb 218). The draft
named the conflict as required, then added that it was using the InciWeb figure
"because it is the more recent-appearing public update". The audit marked that
justification unsupported, since nothing in the evidence says which feed is fresher.
One unsupported claim is under the gate's threshold, so the draft went to review as is.

On LITTLE GIANT the audit rejected the first draft and the second passed clean. That
loop is the point of using a graph rather than a pipeline.

Traces, for anyone with access to the LangSmith project:
[GRASSHOPPER](https://smith.langchain.com/o/dca6c6e5-e248-4aa9-ba1b-b1494ed61204/projects/p/5e8bb3cd-43ed-4d21-89f4-346043a850c2/r/01a17be1-55c6-77b6-98d7-cbae78e0e2cd)
and
[LITTLE GIANT](https://smith.langchain.com/o/dca6c6e5-e248-4aa9-ba1b-b1494ed61204/projects/p/5e8bb3cd-43ed-4d21-89f4-346043a850c2/r/01a17be2-d30a-75b0-9041-007ab8ebd476).
Each run is tagged with the incident, date, mode, and git SHA.

## Tests

```bash
uv run pytest
```

92 tests, all offline. HTTP responses were recorded once from live calls and are served
through an `httpx.MockTransport`. The LLM nodes are tested with a fake chat model. The
graph tests cover the audit loop, the revision cap, the interrupt and resume cycle, and
the SQLite checkpoint round trip. `evals/fixtures/` holds planted-conflict cases, each
with the conflicts it is expected to raise.

## Layout

```
src/firebinder/
  state.py           graph state: Evidence, Conflict, AuditFinding, FireBinderState
  graph.py           build_graph(), route_after_audit(), checkpointer, run config
  llm.py             model selection per role
  nodes/             resolve, gather, reconcile, draft, audit, review, finalize
  tools/             wfigs, firms, inciweb, nws, plus the shared httpx client
  prompts/           gather.md, draft.md, audit.md
  cli.py             firebinder conflicts | draft | resume
evals/fixtures/      planted-conflict cases for reconcile
tests/               offline suite with recorded responses
out/                 run outputs and checkpoints (gitignored)
```

## Things worth knowing

- The WFIGS perimeter layer that most docs link to now requires a token. The open
  layers are `WFIGS_Interagency_Perimeters_Current` and
  `WFIGS_Incident_Locations_Current` on the same host.
- InciWeb no longer publishes per-incident RSS. The tool finds the incident in the
  national KML and RSS feeds and reads the incident page's Current Situation table.
- NWS dropped relative humidity from the period forecast, so minimum RH comes from the
  hourly forecast.
- A report date before today runs in historical mode, which only changes tagging and
  review for now. The tools still read the current layers. Wiring the full-history
  perimeter service for past incident-days is next.
- Model overrides: `FIREBINDER_MODEL_FAST` for gather and draft, `FIREBINDER_MODEL_AUDIT`
  for the audit.

## What's next

- A LangSmith dataset of 15 to 25 historical incident-days with published 209s as
  ground truth, drawn from the ICS209-PLUS research dataset.
- Three evaluators: faithfulness of the draft to its evidence, conflict recall against
  the planted and confirmed cases, and a judge comparing the draft to the real 209.
- A results table across versions, so changes to prompts and tolerance rules can be
  measured instead of eyeballed.

## License

MIT.
