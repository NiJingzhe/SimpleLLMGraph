"""Source-level context inputs for compile-time assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from SimpleLLMFunc.type.message import NormalizedMessageList


@dataclass(frozen=True)
class DataFromAgentConfig:
    base_system_prompt: str
    template_params: Optional[Dict[str, Any]] = None
    tool_prompt_specs: List[Dict[str, Any]] = field(default_factory=list)
    include_must_principles: bool = False


@dataclass(frozen=True)
class DataFromSelfRef:
    base_system_prompt: str
    experiences: List[Dict[str, str]] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    summary_message: Optional[Dict[str, Any]] = None
    working_messages: NormalizedMessageList = field(default_factory=list)


@dataclass(frozen=True)
class CompileSource:
    data_from_agent_config: DataFromAgentConfig
    data_from_selfref: Optional[DataFromSelfRef]
    input_messages: NormalizedMessageList


__all__ = [
    "CompileSource",
    "DataFromAgentConfig",
    "DataFromSelfRef",
]
