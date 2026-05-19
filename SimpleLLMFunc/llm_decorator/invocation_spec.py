"""Invocation-level contracts shared by llm_function and llm_chat.

These dataclasses model Python call semantics. They intentionally do not model
provider-facing requests; conversion to LLM input belongs to the compile
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional

from SimpleLLMFunc.base.types import DataFromSelfRef
from SimpleLLMFunc.type.message import NormalizedMessageList


@dataclass(frozen=True)
class ParameterContract:
    name: str
    type_hint: Any = None
    description: str = ""


@dataclass(frozen=True)
class ReturnContract:
    return_type: Any
    description: str = ""
    structured: bool = False


@dataclass(frozen=True)
class PromptContract:
    base_instruction: str
    parameter_contract: List[ParameterContract] = field(default_factory=list)
    return_contract: Optional[ReturnContract] = None
    tool_prompt_specs: List[Dict[str, Any]] = field(default_factory=list)
    include_must_principles: bool = False
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class TranscriptSeed:
    initial_messages: NormalizedMessageList
    external_history_ref: Optional[List[Dict[str, Any]]] = None
    history_authority: Literal["external", "selfref", "seed"] = "seed"


@dataclass(frozen=True)
class InvocationSpec:
    mode: Literal["function", "chat"]
    func_name: str
    trace_id: str
    docstring: str
    bound_args: Mapping[str, Any]
    type_hints: Mapping[str, Any]
    return_type: Any
    template_params: Optional[Mapping[str, Any]]
    llm_kwargs: Mapping[str, Any]
    stream: bool
    prompt_contract: PromptContract
    transcript_seed: TranscriptSeed
    data_from_selfref: Optional[DataFromSelfRef] = None


__all__ = [
    "InvocationSpec",
    "ParameterContract",
    "PromptContract",
    "ReturnContract",
    "TranscriptSeed",
]
