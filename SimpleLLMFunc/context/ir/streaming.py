"""Streaming deltas and the chunk envelope.

Modelled on the OpenAI Chat Completions streaming spec: each ``Chunk`` is a
``chat.completion.chunk`` object whose ``choices[].delta`` carries
incremental fragments. The minimal extension here is ``Delta.reasoning``,
which carries surfaced thinking fragments for providers that expose them;
providers that do not surface reasoning simply leave it ``None``.

Aggregation rules a consumer is expected to follow:
    * ``Delta.role`` appears only on the first chunk for a choice.
    * ``Delta.content`` is a text fragment to be concatenated.
    * ``Delta.reasoning`` is a thinking fragment to be concatenated.
    * ``Delta.tool_calls`` is a list of partial tool calls keyed by
      ``ToolCallDelta.index``; ``id`` / ``function.name`` appear on the
      first fragment for that index, ``function.arguments`` is appended
      across subsequent fragments.
    * ``Choice.finish_reason`` is ``None`` until the final chunk.
    * ``Chunk.usage`` appears only on the final chunk when
      ``StreamOptions.include_usage`` was requested.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from SimpleLLMFunc.context.ir._enums import (
    FinishReason,
    Role,
    StreamObject,
    ToolCallType,
)


class ToolCallDeltaFunction(BaseModel):
    """The function part of a streaming tool call.

    On the first fragment for an ``index`` this may carry ``name``; on every
    fragment it may carry a piece of ``arguments`` to be appended.
    """

    name: str | None = None
    arguments: str = ""


class ToolCallDelta(BaseModel):
    """A partial tool call. ``index`` identifies which tool call in the
    turn this fragment belongs to and is required so fragments aggregate.
    """

    index: int
    id: str | None = None
    type: ToolCallType | None = None
    function: ToolCallDeltaFunction = Field(default_factory=ToolCallDeltaFunction)


class Delta(BaseModel):
    """Incremental content for a single choice on a single chunk.

    Fields are all optional: a chunk typically populates only one of them.
    ``reasoning`` is the neutral-IR extension for surfaced thinking.
    """

    role: Role | None = None
    content: str | None = None
    reasoning: str | None = None
    reasoning_signature: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    refusal: str | None = None


class Choice(BaseModel):
    """One choice in a chunk. ``delta`` is the incremental payload."""

    index: int
    delta: Delta
    finish_reason: FinishReason | None = None
    logprobs: Any | None = None


class CompletionTokensDetails(BaseModel):
    """Breakdown of completion tokens, including surfaced reasoning tokens."""

    reasoning_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class Usage(BaseModel):
    """Token accounting for the request, present on the final chunk when
    usage was requested.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None


class Chunk(BaseModel):
    """A single streamed ``chat.completion.chunk`` object."""

    id: str
    object: Literal[StreamObject.CHAT_COMPLETION_CHUNK] = (
        StreamObject.CHAT_COMPLETION_CHUNK
    )
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
