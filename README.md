# FireBinder

A LangGraph agent that drafts a wildfire incident's daily ICS-209 status summary from
public data, audits every number in its own draft back to a source, flags where the
sources disagree, and stops for a human before anything is final.

See [SPEC.md](SPEC.md) for the full build spec.

## Setup

```bash
uv sync
cp .env.example .env   # fill in keys
```

## Phase 1: conflict detector

```bash
uv run firebinder conflicts --incident "<name>" --date YYYY-MM-DD
```
