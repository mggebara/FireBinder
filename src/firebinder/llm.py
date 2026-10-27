"""Model selection. Nodes call `llm.get_model(role)` at call time so tests can monkeypatch it.

gather and draft use the cheap fast model; audit uses the strongest model we can afford,
because it is the gate before a human sees the draft (SPEC "Rules the nodes share").
The eval judge (evals/evaluators.py) must be a different model from the drafting model (SPEC
"LangSmith"), so judge_model_name() refuses a judge equal to the fast model.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

Role = Literal["gather", "draft", "audit", "judge"]

DEFAULT_FAST_MODEL = "claude-haiku-5-5"
DEFAULT_AUDIT_MODEL = "claude-opus-5-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4096

load_dotenv()  # ANTHROPIC_API_KEY, without needing firebinder.tools to be imported first


def model_name(role: Role) -> str:
    """Env overrides: FIREBINDER_MODEL_FAST (gather, draft), FIREBINDER_MODEL_AUDIT (audit), FIREBINDER_MODEL_JUDGE (judge)."""
    if role == "judge":
        return judge_model_name()
    if role == "audit":
        return os.environ.get("FIREBINDER_MODEL_AUDIT", "").strip() or DEFAULT_AUDIT_MODEL
    if role in ("gather", "draft"):
        return os.environ.get("FIREBINDER_MODEL_FAST", "").strip() or DEFAULT_FAST_MODEL
    raise ValueError(f"Unknown model role: {role!r}")


def judge_model_name() -> str:
    """The eval judge; raises if it is the same model that drafts, since a model grading itself is not a check."""
    judge = os.environ.get("FIREBINDER_MODEL_JUDGE", "").strip() or DEFAULT_JUDGE_MODEL
    fast = model_name("draft")
    if judge == fast:
        raise ValueError(f"Judge model {judge!r} equals the drafting model; set FIREBINDER_MODEL_JUDGE to a different model.")
    return judge


def get_model(role: Role) -> BaseChatModel:
    # No temperature: the 5.5 models reject it ("`temperature` is deprecated for this model").
    return ChatAnthropic(model=model_name(role), max_tokens=MAX_TOKENS)
