"""Non-streaming completion IR.

The non-streaming counterpart of :mod:`SimpleLLMFunc.context.ir.streaming`.
Mirrors the OpenAI Chat Completions ``chat.completion`` object: a
``Completion`` carries one or more ``CompletionChoice`` items, each holding a
fully-formed assistant :class:`AssistantMessage` plus a ``finish_reason`` and
optional ``usage``.

``message`` reuses :class:`AssistantMessage` from :mod:`.messages` so the IR
has a single assistant representation across the streaming and
non-streaming paths and across providers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from SimpleLLMFunc.context.ir._enums import FinishReason
from SimpleLLMFunc.context.ir.messages import AssistantMessage
from SimpleLLMFunc.context.ir.streaming import Usage


class CompletionChoice(BaseModel):
    """One choice in a non-streaming completion."""

    index: int
    message: AssistantMessage
    finish_reason: FinishReason | None = None
    logprobs: Any | None = None


class Completion(BaseModel):
    """A non-streaming ``chat.completion`` object in the neutral IR."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None
