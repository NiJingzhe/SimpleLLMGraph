"""Semantic facts and immutable EventLog views.

Events only describe facts that already happened. Pending external operations
are represented by :mod:`SimpleLLMFunc.loop` Effects instead.
"""

from SimpleLLMFunc.event.base import BaseEvent, CtxMixin, EventView
from SimpleLLMFunc.event.builtin import (
    AssistantMessageEvent,
    BUILTIN_EVENTS,
    SystemPromptEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)

__all__ = [
    "AssistantMessageEvent",
    "BUILTIN_EVENTS",
    "BaseEvent",
    "CtxMixin",
    "EventView",
    "SystemPromptEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "UserMessageEvent",
]
