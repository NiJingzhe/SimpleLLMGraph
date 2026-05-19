from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from SimpleLLMFunc.base.types.mutation import ContextMutation
from SimpleLLMFunc.base.types.source import DataFromSelfRef
from SimpleLLMFunc.type.message import NormalizedMessageList


@dataclass
class ContextState:
    messages: NormalizedMessageList
    data_from_selfref: Optional[DataFromSelfRef] = None
    pending_mutations: List[ContextMutation] = field(default_factory=list)


@dataclass
class CompiledContext:
    messages: NormalizedMessageList
    data_from_selfref: Optional[DataFromSelfRef] = None


__all__ = ["CompiledContext", "ContextState"]
