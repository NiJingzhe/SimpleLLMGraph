from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.base.types.mutation import ContextMutation


@dataclass
class SingleLLMCallResult:
    response: Any = None
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_details: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[CompletionUsage] = None
    execution_time: float = 0.0
    mutations: List[ContextMutation] = field(default_factory=list)
    aborted: bool = False


@dataclass
class SingleLLMPhaseResultYield:
    result: SingleLLMCallResult


__all__ = ["SingleLLMCallResult", "SingleLLMPhaseResultYield"]
