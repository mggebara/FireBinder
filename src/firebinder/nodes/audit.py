"""audit node. Stub: real implementation lands in the next commits."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from firebinder.state import FireBinderState


def audit(state: FireBinderState, config: RunnableConfig | None = None) -> dict:
    raise NotImplementedError("audit node not implemented yet")
