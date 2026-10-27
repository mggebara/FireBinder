"""Model selection. Nodes call `llm.get_model(role)` at call time so tests can monkeypatch it.

gather and draft use the cheap fast model; audit uses the strongest model we can afford,
because it is the gate before a human sees the draft (SPEC "Rules the nodes share").
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

Role = Literal["gather", "draft", "audit"]

DEFAULT_FAST_MODEL = "claude-haiku-5-5"
DEFAULT_AUDIT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 4096

load_dotenv()  # ANTHROPIC_API_KEY, without needing firebinder.tools to be imported first


def model_name(role: Role) -> str:
    """Env overrides: FIREBINDER_MODEL_FAST (gather, draft) and FIREBINDER_MODEL_AUDIT (audit)."""
    if role == "audit":
        return os.environ.get("FIREBINDER_MODEL_AUDIT", "").strip() or DEFAULT_AUDIT_MODEL
    if role in ("gather", "draft"):
        return os.environ.get("FIREBINDER_MODEL_FAST", "").strip() or DEFAULT_FAST_MODEL
    raise ValueError(f"Unknown model role: {role!r}")


def get_model(role: Role) -> BaseChatModel:
    return ChatAnthropic(model=model_name(role), temperature=0, max_tokens=MAX_TOKENS)
