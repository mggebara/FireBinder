"""draft node. Stub: real implementation lands in the next commits."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from firebinder.state import FireBinderState


def draft(state: FireBinderState, config: RunnableConfig | None = None) -> dict:
    raise NotImplementedError("draft node not implemented yet")


def check_citations(draft: dict) -> list[str]:
    """Return the names of numeric fields in draft["fields"] that have no key in draft["citations"]."""
    raise NotImplementedError
