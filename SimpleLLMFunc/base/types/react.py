from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from SimpleLLMFunc.base.types.context import ContextState
from SimpleLLMFunc.base.types.mutation import ContextMutation


@dataclass
class ReactLoopState:
    context_state: ContextState
    pending_mutations: List[ContextMutation] = field(default_factory=list)
    iteration: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0


__all__ = ["ReactLoopState"]
