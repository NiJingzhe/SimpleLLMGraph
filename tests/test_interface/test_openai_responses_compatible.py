"""Tests for OpenAIResponsesCompatible adapter behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningContent,
)

from SimpleLLMFunc.base.tool_call.extraction import (
    accumulate_tool_calls_from_chunks,
    extract_tool_calls_from_stream_response,
)
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.interface.openai_responses_compatible import (
    OpenAIResponsesCompatible,
)
from SimpleLLMFunc.logger.context_manager import (
    get_current_context_attribute,
    set_current_context_attribute,
)


def _build_llm() -> OpenAIResponsesCompatible:
    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-openai-responses-compatible",
    )
    llm = OpenAIResponsesCompatible(
        api_key_pool=key_pool,
        model_name="gpt-test",
        base_url="https://example.com/v1",
        max_retries=1,
        retry_delay=0.0,
    )
    llm.token_bucket.acquire = AsyncMock(return_value=True)
    return llm


def _make_usage(input_tokens: int, output_tokens: int) -> ResponseUsage:
    total = input_tokens + output_tokens
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details={"cached_tokens": 0},
        output_tokens=output_tokens,
        output_tokens_details={"reasoning_tokens": 0},
        total_tokens=total,
    )


def _make_text_response(text: str, *, reasoning_text: str | None = None) -> Response:
    output = []
    if reasoning_text is not None:
        output.append(
            ResponseReasoningItem(
                id="rs_1",
                type="reasoning",
                summary=[],
                content=[
                    ResponseReasoningContent(
                        type="reasoning_text",
                        text=reasoning_text,
                    )
                ],
                status="completed",
            )
        )
    output.append(
        ResponseOutputMessage(
            id="msg_1",
            role="assistant",
            status="completed",
            type="message",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )
    )
    return Response(
        id="resp_1",
        created_at=123,
        model="gpt-test",
        object="response",
        output=output,
        parallel_tool_calls=True,
        status="completed",
        text={"format": {"type": "text"}},
        tool_choice="auto",
        tools=[],
        truncation="disabled",
        usage=_make_usage(7, 5),
    )


def _make_tool_call_response() -> Response:
    return Response(
        id="resp_tool",
        created_at=123,
        model="gpt-test",
        object="response",
        output=[
            ResponseFunctionToolCall(
                id="fc_1",
                call_id="call_123",
                type="function_call",
                name="test_tool",
                arguments='{"arg1":"value1"}',
                status="completed",
            )
        ],
        parallel_tool_calls=True,
        status="completed",
        text={"format": {"type": "text"}},
        tool_choice="auto",
        tools=[],
        truncation="disabled",
        usage=_make_usage(3, 2),
    )


class _EventStream:
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for event in self._events:
            yield event

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_chat_adapts_text_and_reasoning() -> None:
    llm = _build_llm()
    response = _make_text_response("hello", reasoning_text="step one")
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=response))
    )
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    result = await llm.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "hello"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.total_tokens == 12
    assert result.choices[0].message.reasoning_details == [
        {
            "id": "rs_1",
            "format": "text",
            "index": 0,
            "type": "reasoning.text",
            "data": "step one",
        }
    ]


@pytest.mark.asyncio
async def test_chat_adapts_function_tool_calls() -> None:
    llm = _build_llm()
    response = _make_tool_call_response()
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    result = await llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    request_kwargs = create_mock.await_args.kwargs
    assert request_kwargs["tools"][0]["type"] == "function"
    assert request_kwargs["tools"][0]["name"] == "test_tool"
    assert request_kwargs["tools"][0]["parameters"]["additionalProperties"] is False
    assert request_kwargs["tools"][0]["parameters"]["required"] == []
    assert result.choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.tool_calls[0].id == "call_123"
    assert result.choices[0].message.tool_calls[0].function.name == "test_tool"
    assert (
        result.choices[0].message.tool_calls[0].function.arguments
        == '{"arg1":"value1"}'
    )


@pytest.mark.asyncio
async def test_chat_preserves_assistant_history_as_assistant_input() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "next question"},
        ]
    )

    request_kwargs = create_mock.await_args.kwargs
    request_input = request_kwargs["input"]
    assert request_kwargs["instructions"] == "sys"
    assert request_input[0]["role"] == "assistant"
    assert request_input[0]["content"][0]["type"] == "output_text"
    assert request_input[0]["content"][0]["text"] == "previous answer"
    assert request_input[1]["role"] == "user"


@pytest.mark.asyncio
async def test_chat_forwards_reasoning_kwargs_to_responses_api() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        reasoning={"effort": "xhigh"},
    )

    request_kwargs = create_mock.await_args.kwargs
    assert request_kwargs["reasoning"] == {"effort": "xhigh"}


@pytest.mark.asyncio
async def test_chat_filters_temperature_from_responses_request() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        temperature=1.0,
        reasoning={"effort": "xhigh"},
    )

    request_kwargs = create_mock.await_args.kwargs
    assert request_kwargs["reasoning"] == {"effort": "xhigh"}
    assert "temperature" not in request_kwargs


@pytest.mark.asyncio
async def test_chat_normalizes_nested_object_schemas_for_responses() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "nested_tool",
                    "description": "Nested schema tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                },
                                "required": ["name"],
                            }
                        },
                        "required": ["payload"],
                    },
                },
            }
        ],
    )

    request_kwargs = create_mock.await_args.kwargs
    tool_schema = request_kwargs["tools"][0]["parameters"]
    assert tool_schema["additionalProperties"] is False
    assert tool_schema["properties"]["payload"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_chat_normalizes_optional_properties_for_responses_tools() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "timeout_seconds": {"type": ["number", "null"]},
                            "event_emitter": {"type": ["string", "null"]},
                        },
                        "required": ["code"],
                    },
                },
            }
        ],
    )

    request_kwargs = create_mock.await_args.kwargs
    tool_schema = request_kwargs["tools"][0]["parameters"]
    assert set(tool_schema["properties"].keys()) == {"code", "timeout_seconds"}
    assert tool_schema["required"] == ["code", "timeout_seconds"]
    assert tool_schema["properties"]["timeout_seconds"]["anyOf"] == [
        {"type": "number"},
        {"type": "null"},
    ]


@pytest.mark.asyncio
async def test_chat_maps_system_messages_to_instructions() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Reply with OK only."},
        ]
    )

    request_kwargs = create_mock.await_args.kwargs
    assert request_kwargs["instructions"] == "You are helpful."
    assert request_kwargs["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Reply with OK only."}],
        }
    ]


@pytest.mark.asyncio
async def test_chat_maps_multimodal_user_content_to_responses_input() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/cat.png",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
    )

    request_input = create_mock.await_args.kwargs["input"]
    assert request_input == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "high",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_chat_maps_assistant_tool_calls_to_function_call_items() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"arg":"value"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "tool-result",
            },
        ]
    )

    request_input = create_mock.await_args.kwargs["input"]
    assert request_input[0] == {
        "type": "function_call",
        "call_id": "call_123",
        "name": "test_tool",
        "arguments": '{"arg":"value"}',
    }
    assert request_input[1] == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": "tool-result",
    }


@pytest.mark.asyncio
async def test_chat_merges_multiple_system_messages_into_instructions() -> None:
    llm = _build_llm()
    response = _make_text_response("ok")
    create_mock = AsyncMock(return_value=response)
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[
            {"role": "system", "content": "Rule one."},
            {"role": "system", "content": "Rule two."},
            {"role": "user", "content": "Reply with OK only."},
        ]
    )

    request_kwargs = create_mock.await_args.kwargs
    assert request_kwargs["instructions"] == "Rule one.\n\nRule two."
    assert len(request_kwargs["input"]) == 1


@pytest.mark.asyncio
async def test_chat_stream_adapts_text_tool_and_reasoning_deltas() -> None:
    llm = _build_llm()
    completed_response = _make_tool_call_response()
    stream = _EventStream(
        [
            ResponseCreatedEvent(
                type="response.created",
                sequence_number=0,
                response=Response(
                    id="resp_stream",
                    created_at=123,
                    model="gpt-test",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    status="in_progress",
                    text={"format": {"type": "text"}},
                    tool_choice="auto",
                    tools=[],
                    truncation="disabled",
                    usage=None,
                ),
            ),
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=1,
                item_id="msg_1",
                output_index=0,
                content_index=0,
                delta="hel",
                logprobs=[],
            ),
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=2,
                output_index=1,
                item=ResponseFunctionToolCall(
                    id="fc_1",
                    call_id="call_456",
                    type="function_call",
                    name="lookup",
                    arguments="",
                    status="in_progress",
                ),
            ),
            SimpleNamespace(
                type="response.reasoning_text.delta",
                sequence_number=3,
                delta="thinking",
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=4,
                item_id="fc_1",
                output_index=1,
                delta='{"city":"',
            ),
            ResponseCompletedEvent(
                type="response.completed",
                sequence_number=5,
                response=completed_response,
            ),
        ]
    )

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=stream))
    )
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    chunks = []
    async for chunk in llm.chat_stream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert stream.closed is True
    assert chunks[0].choices[0].delta.content == "hel"
    assert chunks[1].choices[0].delta.reasoning_details == [
        {
            "id": "",
            "format": "text",
            "index": 0,
            "type": "reasoning.text",
            "data": "thinking",
        }
    ]
    tool_call = chunks[2].choices[0].delta.tool_calls[0]
    assert tool_call.id == "call_456"
    assert tool_call.function.name == "lookup"
    assert tool_call.function.arguments == '{"city":"'
    assert chunks[3].choices[0].finish_reason == "tool_calls"
    assert chunks[3].usage is not None
    assert chunks[3].usage.total_tokens == 5


@pytest.mark.asyncio
async def test_chat_stream_keeps_parallel_tool_call_arguments_separate() -> None:
    llm = _build_llm()
    completed_response = Response(
        id="resp_parallel_tools",
        created_at=123,
        model="gpt-test",
        object="response",
        output=[
            ResponseFunctionToolCall(
                id="fc_exec_1",
                call_id="call_exec_1",
                type="function_call",
                name="execute_code",
                arguments='{"code":"print(1)","timeout_seconds":600}',
                status="completed",
            ),
            ResponseFunctionToolCall(
                id="fc_exec_2",
                call_id="call_exec_2",
                type="function_call",
                name="execute_code",
                arguments='{"code":"import os\\n","timeout_seconds":600}',
                status="completed",
            ),
            ResponseFunctionToolCall(
                id="fc_read_1",
                call_id="call_read_1",
                type="function_call",
                name="read_file",
                arguments='{"path":"pyproject.toml","start_line":1,"end_line":250}',
                status="completed",
            ),
        ],
        parallel_tool_calls=True,
        status="completed",
        text={"format": {"type": "text"}},
        tool_choice="auto",
        tools=[],
        truncation="disabled",
        usage=_make_usage(9, 6),
    )
    stream = _EventStream(
        [
            ResponseCreatedEvent(
                type="response.created",
                sequence_number=0,
                response=Response(
                    id="resp_parallel_tools",
                    created_at=123,
                    model="gpt-test",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    status="in_progress",
                    text={"format": {"type": "text"}},
                    tool_choice="auto",
                    tools=[],
                    truncation="disabled",
                    usage=None,
                ),
            ),
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=1,
                output_index=0,
                item=ResponseFunctionToolCall(
                    id="fc_exec_1",
                    call_id="call_exec_1",
                    type="function_call",
                    name="execute_code",
                    arguments="",
                    status="in_progress",
                ),
            ),
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=2,
                output_index=1,
                item=ResponseFunctionToolCall(
                    id="fc_exec_2",
                    call_id="call_exec_2",
                    type="function_call",
                    name="execute_code",
                    arguments="",
                    status="in_progress",
                ),
            ),
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=3,
                output_index=2,
                item=ResponseFunctionToolCall(
                    id="fc_read_1",
                    call_id="call_read_1",
                    type="function_call",
                    name="read_file",
                    arguments="",
                    status="in_progress",
                ),
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=4,
                item_id="fc_exec_1",
                output_index=0,
                delta='{"code":"print(1)"',
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=5,
                item_id="fc_exec_2",
                output_index=1,
                delta='{"code":"import os\\n"',
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=6,
                item_id="fc_read_1",
                output_index=2,
                delta='{"path":"pyproject.toml","start_line":1,"end_line":250}',
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=7,
                item_id="fc_exec_1",
                output_index=0,
                delta=',"timeout_seconds":600}',
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=8,
                item_id="fc_exec_2",
                output_index=1,
                delta=',"timeout_seconds":600}',
            ),
            ResponseCompletedEvent(
                type="response.completed",
                sequence_number=9,
                response=completed_response,
            ),
        ]
    )

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=stream))
    )
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    chunks = []
    async for chunk in llm.chat_stream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    tool_call_chunks = []
    for chunk in chunks:
        tool_call_chunks.extend(extract_tool_calls_from_stream_response(chunk))

    accumulated_tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
    assert [chunk["index"] for chunk in tool_call_chunks] == [0, 1, 2, 0, 1]
    assert accumulated_tool_calls == [
        {
            "id": "call_exec_1",
            "type": "function",
            "function": {
                "name": "execute_code",
                "arguments": '{"code":"print(1)","timeout_seconds":600}',
            },
        },
        {
            "id": "call_exec_2",
            "type": "function",
            "function": {
                "name": "execute_code",
                "arguments": '{"code":"import os\\n","timeout_seconds":600}',
            },
        },
        {
            "id": "call_read_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"pyproject.toml","start_line":1,"end_line":250}',
            },
        },
    ]


@pytest.mark.asyncio
async def test_chat_and_stream_update_context_token_usage() -> None:
    llm = _build_llm()
    text_response = _make_text_response("hello")
    stream = _EventStream(
        [
            ResponseCompletedEvent(
                type="response.completed",
                sequence_number=0,
                response=_make_text_response("bye"),
            )
        ]
    )
    create_mock = AsyncMock(side_effect=[text_response, stream])
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    before_input = int(get_current_context_attribute("input_tokens") or 0)
    before_output = int(get_current_context_attribute("output_tokens") or 0)
    set_current_context_attribute("input_tokens", before_input)
    set_current_context_attribute("output_tokens", before_output)

    await llm.chat(messages=[{"role": "user", "content": "hi"}])
    async for _ in llm.chat_stream(messages=[{"role": "user", "content": "hi"}]):
        pass

    after_input = int(get_current_context_attribute("input_tokens") or 0)
    after_output = int(get_current_context_attribute("output_tokens") or 0)
    assert after_input == before_input + 14
    assert after_output == before_output + 10
