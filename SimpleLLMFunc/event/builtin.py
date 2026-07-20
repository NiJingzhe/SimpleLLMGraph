"""Built-in semantic events used by common conversational Loops."""

from __future__ import annotations

from typing import Literal

from SimpleLLMFunc.context.ir import ContentPart, ToolCall, ToolResult
from SimpleLLMFunc.event.base import BaseEvent, CtxMixin


class SystemPromptEvent(BaseEvent, CtxMixin):
    type: Literal["system_prompt"] = "system_prompt"
    content: str


class UserMessageEvent(BaseEvent, CtxMixin):
    type: Literal["user_message"] = "user_message"
    content: str | list[ContentPart]


class AssistantMessageEvent(BaseEvent, CtxMixin):
    type: Literal["assistant_message"] = "assistant_message"
    content: str | list[ContentPart] | None = None
    refusal: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class ToolCallEvent(BaseEvent, CtxMixin):
    """A model-produced tool request. This event never executes the tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    arguments: str


class ToolResultEvent(BaseEvent, CtxMixin):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool = False


BUILTIN_EVENTS: tuple[type[BaseEvent], ...] = (
    SystemPromptEvent,
    UserMessageEvent,
    AssistantMessageEvent,
    ToolCallEvent,
    ToolResultEvent,
)
