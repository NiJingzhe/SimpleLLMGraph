from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class AssistantMessageMutation:
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolResultMutation:
    tool_call_id: str
    content: str
    role: Literal["tool"] = "tool"


@dataclass
class MultimodalToolResultMutation:
    tool_call_id: str
    tool_name: str
    arguments: str
    user_messages: List[Dict[str, Any]]


@dataclass
class UserMessageMutation:
    message: Dict[str, Any]


@dataclass
class ContextReplaceMutation:
    messages: List[Dict[str, Any]]


@dataclass
class ContextSummaryMutation:
    summary_message: Dict[str, Any]
    remember: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ExperienceRememberMutation:
    text: str


@dataclass
class ExperienceForgetMutation:
    experience_id: str


@dataclass
class AssistantTruncatedMutation:
    partial_content: str
    abort_reason: str = ""


@dataclass
class ToolCancelledMutation:
    tool_call_id: str
    tool_name: str
    abort_reason: str = ""


ContextMutation = Union[
    AssistantMessageMutation,
    ToolResultMutation,
    MultimodalToolResultMutation,
    UserMessageMutation,
    ContextReplaceMutation,
    ContextSummaryMutation,
    ExperienceRememberMutation,
    ExperienceForgetMutation,
    AssistantTruncatedMutation,
    ToolCancelledMutation,
]


__all__ = [
    "AssistantMessageMutation",
    "AssistantTruncatedMutation",
    "ContextMutation",
    "ContextReplaceMutation",
    "ContextSummaryMutation",
    "ExperienceRememberMutation",
    "ExperienceForgetMutation",
    "MultimodalToolResultMutation",
    "ToolCancelledMutation",
    "ToolResultMutation",
    "UserMessageMutation",
]
