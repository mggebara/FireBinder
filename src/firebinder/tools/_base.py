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

    def _evidence(self, field: str, value: object, url: str | None = None) -> Evidence:
        return Evidence(
            field=field,
            value=value,
            source=self.source,
            url=url or self.url,
            fetched_at=self.fetched_at,
        )
