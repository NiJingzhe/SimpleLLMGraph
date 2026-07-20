"""Conversation entries: messages, tool calls and tool results.

A ``Conversation`` is an ordered list of ``ConversationEntry`` items, a
discriminated union keyed on ``role`` that mirrors the OpenAI Chat
Completions ``messages`` array. Splitting by role gives per-role
type-safety (e.g. only ``ToolMessage`` carries ``tool_call_id``) while
remaining a single flat sequence on the wire.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from SimpleLLMFunc.context.ir._enums import Role, ToolCallType
from SimpleLLMFunc.context.ir.parts import ContentPart


class ToolCallFunction(BaseModel):
    """The function part of a tool call.

    ``arguments`` is the JSON-serialised argument object as a string,
    matching the OpenAI convention. During streaming it is accumulated
    fragment by fragment and may be incomplete on intermediate deltas.
    """

    name: str
    arguments: str


class ToolCall(BaseModel):
    """A complete tool call emitted by the assistant."""

    id: str
    type: ToolCallType = ToolCallType.FUNCTION
    function: ToolCallFunction


class SystemMessage(BaseModel):
    """Top-level system instruction."""

    role: Literal[Role.SYSTEM] = Role.SYSTEM
    content: str


class DeveloperMessage(BaseModel):
    """Developer-scoped instruction (OpenAI ``developer`` role)."""

    role: Literal[Role.DEVELOPER] = Role.DEVELOPER
    content: str


class UserMessage(BaseModel):
    """A user turn. ``content`` is either a plain string or a list of
    multimodal :data:`ContentPart` items (text / image / audio).
    """

    role: Literal[Role.USER] = Role.USER
    content: str | list[ContentPart]


class AssistantMessage(BaseModel):
    """An assistant turn. ``content`` may be ``None`` when the turn only
    carries tool calls. ``refusal`` records a refusal, ``name`` is the
    legacy participant-name field.
    """

    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None
    name: str | None = None


class ToolMessage(BaseModel):
    """The result of a tool call, addressed back by ``tool_call_id``."""

    role: Literal[Role.TOOL] = Role.TOOL
    tool_call_id: str
    content: str


ConversationEntry = Annotated[
    Union[
        SystemMessage,
        DeveloperMessage,
        UserMessage,
        AssistantMessage,
        ToolMessage,
    ],
    Field(discriminator="role"),
]

Conversation = list[ConversationEntry]
