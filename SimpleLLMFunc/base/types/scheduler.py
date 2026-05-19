from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from SimpleLLMFunc.base.types.mutation import ContextMutation


@dataclass
class ToolSchedulerResult:
    mutations: List[ContextMutation] = field(default_factory=list)
    total_tool_calls: int = 0
    aborted: bool = False


__all__ = ["ToolSchedulerResult"]
