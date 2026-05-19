from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from SimpleLLMFunc.base.types.source import DataFromSelfRef
from SimpleLLMFunc.type.message import NormalizedMessageList


@dataclass(frozen=True)
class ReducedTurnContext:
    transcript: NormalizedMessageList
    selfref_snapshot: Optional[DataFromSelfRef] = None


@dataclass(frozen=True)
class CompiledTurnContext:
    transcript: NormalizedMessageList
    system_prompt: Optional[str]
    llm_messages: NormalizedMessageList
    selfref_snapshot: Optional[DataFromSelfRef] = None


__all__ = ["CompiledTurnContext", "ReducedTurnContext"]
