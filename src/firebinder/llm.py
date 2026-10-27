"""Chat model factory. Nodes call `llm.get_model(role)` at call time so tests can monkeypatch it."""

from __future__ import annotations

import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

Role = Literal["gather", "draft", "audit"]

DEFAULT_FAST_MODEL = "claude-haiku-5-5"
DEFAULT_AUDIT_MODEL = "claude-opus-5-5"


def model_name(role: Role) -> str:
    """gather and draft use the cheap fast model; audit uses the strongest model (it's the gate)."""
    if role == "audit":
        return os.environ.get("FIREBINDER_AUDIT_MODEL", "").strip() or DEFAULT_AUDIT_MODEL
    return os.environ.get("FIREBINDER_FAST_MODEL", "").strip() or DEFAULT_FAST_MODEL


def get_model(role: Role) -> BaseChatModel:
    return ChatAnthropic(model=model_name(role), temperature=0, max_tokens=4096)
