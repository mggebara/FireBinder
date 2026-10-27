"""Model selection. Stub: real implementation lands in the next commits."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel

Role = Literal["gather", "draft", "audit"]


def get_model(role: Role) -> BaseChatModel:
    raise NotImplementedError
