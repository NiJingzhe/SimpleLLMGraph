from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function as OpenAIFunction,
)

from SimpleLLMFunc.base.llm_call import execute_single_llm_phase
from SimpleLLMFunc.base.types import AssistantMessageMutation, AssistantTruncatedMutation
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.events import LLMCallEndEvent, LLMCallStartEvent, ReActEventType
from SimpleLLMFunc.hooks.stream import EventYield, ResponseYield


def _completion(content: str) -> ChatCompletion:
    return ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=123456,
        model="test-model",
        object="chat.completion",
    )


def _tool_completion(content: str) -> ChatCompletion:
    return ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_123",
                            type="function",
                            function=OpenAIFunction(name="test_tool", arguments='{"arg":"value"}'),
                        )
                    ],
                ),
            )
        ],
        created=123456,
        model="test-model",
        object="chat.completion",
    )


def _chunk(content: str) -> ChatCompletionChunk:
    delta = ChoiceDelta(content=content, role="assistant")
    choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
    return ChatCompletionChunk(
        id="chunk-id",
        choices=[choice],
        created=123,
        model="test-model",
        object="chat.completion.chunk",
    )


@pytest.mark.asyncio
async def test_execute_single_llm_phase_non_streaming_emits_start_response_end() -> None:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=_completion("done"))

    outputs = []
    async for output in execute_single_llm_phase(
        llm_interface=llm,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        llm_kwargs={},
        trace_id="trace-1",
        func_name="test_func",
        iteration=0,
        stream=False,
    ):
        outputs.append(output)

    assert isinstance(outputs[0], EventYield)
    assert isinstance(outputs[0].event, LLMCallStartEvent)
    assert isinstance(outputs[1], ResponseYield)
    assert outputs[1].response.choices[0].message.content == "done"
    assert isinstance(outputs[2], EventYield)
    assert isinstance(outputs[2].event, LLMCallEndEvent)
    assert outputs[2].event.content == "done"


@pytest.mark.asyncio
async def test_execute_single_llm_phase_abort_builds_truncated_mutation() -> None:
    llm = MagicMock()
    abort_signal = AbortSignal()

    async def chat_stream(**kwargs: Any):
        yield _chunk("partial ")
        abort_signal.abort("user_interrupt")
        await asyncio.sleep(0)
        yield _chunk("ignored")

    llm.chat_stream = chat_stream

    outputs = []
    async for output in execute_single_llm_phase(
        llm_interface=llm,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        llm_kwargs={},
        trace_id="trace-1",
        func_name="test_func",
        iteration=0,
        stream=True,
        abort_signal=abort_signal,
    ):
        outputs.append(output)

    assert isinstance(outputs[-2], EventYield)
    end_event = outputs[-2].event
    assert isinstance(end_event, LLMCallEndEvent)
    assert end_event.content == "partial "

    assert outputs[-1].result.aborted is True
    assert outputs[-1].result.mutations

    result_mutation = AssistantTruncatedMutation(
        partial_content=end_event.content,
        abort_reason=abort_signal.reason,
    )
    assert result_mutation.partial_content == "partial "
    assert result_mutation.abort_reason == "user_interrupt"


@pytest.mark.asyncio
async def test_execute_single_llm_phase_non_streaming_maps_content_to_assistant_mutation() -> None:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=_completion("done"))

    content = None
    async for output in execute_single_llm_phase(
        llm_interface=llm,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        llm_kwargs={},
        trace_id="trace-1",
        func_name="test_func",
        iteration=0,
        stream=False,
    ):
        if isinstance(output, EventYield) and isinstance(output.event, LLMCallEndEvent):
            content = output.event.content

    mutation = AssistantMessageMutation(content=content)
    assert mutation.content == "done"


@pytest.mark.asyncio
async def test_execute_single_llm_phase_non_streaming_keeps_content_with_tool_calls() -> None:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=_tool_completion("Let me check that."))

    result = None
    async for output in execute_single_llm_phase(
        llm_interface=llm,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        llm_kwargs={},
        trace_id="trace-1",
        func_name="test_func",
        iteration=0,
        stream=False,
    ):
        if getattr(output, "result", None) is not None:
            result = output.result

    assert result is not None
    assert len(result.mutations) == 1
    mutation = result.mutations[0]
    assert isinstance(mutation, AssistantMessageMutation)
    assert mutation.content == "Let me check that."
    assert mutation.tool_calls[0]["id"] == "call_123"
