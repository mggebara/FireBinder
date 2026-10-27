"""Base model for every tool output: each one is stamped with where and when it came from."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from firebinder.state import Evidence


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SourceStamped(BaseModel):
    source: str  # tool name, e.g. "get_incident_record"
    url: str  # primary URL the data came from
    fetched_at: str  # ISO 8601, UTC

    def to_evidence(self) -> list[Evidence]:
        raise NotImplementedError(f"{type(self).__name__} must implement to_evidence()")

    def summary(self) -> str:
        """Short human-readable paragraph of the key values (no geometry). What the model sees."""
        raise NotImplementedError(f"{type(self).__name__} must implement summary()")

    def _evidence(self, field: str, value: object, url: str | None = None) -> Evidence:
        return Evidence(
            field=field,
            value=value,
            source=self.source,
            url=url or self.url,
            fetched_at=self.fetched_at,
        )


def fmt(value: object, unit: str = "") -> str:
    """Format a value for a tool summary: "unknown" for None, thousands separators for numbers."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
        return text + unit
    return str(value) + unit


def clip(text: str | None, limit: int = 400) -> str:
    """Collapse whitespace and truncate with an ellipsis."""
    if not text:
        return "none"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
