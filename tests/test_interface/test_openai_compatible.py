"""Tests for OpenAICompatible streaming behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
)
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.interface.llm_interface import DEFAULT_CONTEXT_WINDOW
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible
from SimpleLLMFunc.logger.context_manager import (
    get_current_context_attribute,
    set_current_context_attribute,
)


def _make_chunk(content: str | None, finish_reason: str | None) -> ChatCompletionChunk:
    delta = ChoiceDelta(content=content, role="assistant")
    choice = ChunkChoice(delta=delta, finish_reason=finish_reason, index=0)
    return ChatCompletionChunk(
        id="chunk-id",
        choices=[choice],
        created=123,
        model="test-model",
        object="chat.completion.chunk",
    )


def _make_usage_chunk(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk-id",
        choices=[],
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def _make_completion(content: str = "hello") -> ChatCompletion:
    return ChatCompletion(
        id="completion-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=123,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


class _NeverEndingStream:
    """Async iterable that never finishes after yielding initial chunks."""

    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = chunks
        self.closed = False
        self.close_call_count = 0

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk

        # Simulate providers that keep SSE open after finish_reason.
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.close_call_count += 1
        self.closed = True


@pytest.mark.asyncio
async def test_chat_stream_stops_after_finish_reason() -> None:
    """chat_stream should not block after receiving a terminal chunk."""

    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-openai-compatible-stream-finish",
    )
    llm = OpenAICompatible(
        api_key_pool=key_pool,
        model_name="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        retry_delay=0.0,
    )

    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._stream_finish_grace_timeout = 0.01

    stream_response = _NeverEndingStream(
        chunks=[
            _make_chunk(content="hello", finish_reason=None),
            _make_chunk(content=None, finish_reason="stop"),
        ]
    )
    create_mock = AsyncMock(return_value=stream_response)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    async def _collect_chunks() -> list[ChatCompletionChunk]:
        chunks: list[ChatCompletionChunk] = []
        async for chunk in llm.chat_stream(
            messages=[{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)
        return chunks

    chunks = await asyncio.wait_for(_collect_chunks(), timeout=0.2)

    assert len(chunks) == 2
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert stream_response.closed is True
    assert stream_response.close_call_count == 1
    create_mock.assert_awaited_once()
    create_kwargs = create_mock.await_args_list[0].kwargs
    assert create_kwargs["stream_options"]["include_usage"] is True


@pytest.mark.asyncio
async def test_chat_stream_keeps_post_finish_usage_chunk() -> None:
    """chat_stream should keep one post-finish chunk for usage accounting."""

    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-openai-compatible-stream-usage",
    )
    llm = OpenAICompatible(
        api_key_pool=key_pool,
        model_name="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        retry_delay=0.0,
    )

    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._stream_finish_grace_timeout = 0.05

    stream_response = _NeverEndingStream(
        chunks=[
            _make_chunk(content="hello", finish_reason=None),
            _make_chunk(content=None, finish_reason="stop"),
            _make_usage_chunk(prompt_tokens=8, completion_tokens=3, total_tokens=11),
        ]
    )
    create_mock = AsyncMock(return_value=stream_response)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    async def _collect_chunks() -> list[ChatCompletionChunk]:
        chunks: list[ChatCompletionChunk] = []
        async for chunk in llm.chat_stream(
            messages=[{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)
        return chunks

    chunks = await asyncio.wait_for(_collect_chunks(), timeout=0.3)

    assert len(chunks) == 3
    assert chunks[1].choices[0].finish_reason == "stop"
    assert chunks[2].usage is not None
    assert chunks[2].usage.total_tokens == 11


@pytest.mark.asyncio
async def test_chat_stream_allows_non_usage_chunk_before_usage() -> None:
    """chat_stream should continue within grace window until usage arrives."""

    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-openai-compatible-stream-usage-late",
    )
    llm = OpenAICompatible(
        api_key_pool=key_pool,
        model_name="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        retry_delay=0.0,
    )

    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._stream_finish_grace_timeout = 0.05

    stream_response = _NeverEndingStream(
        chunks=[
            _make_chunk(content="hello", finish_reason=None),
            _make_chunk(content=None, finish_reason="stop"),
            _make_chunk(content=None, finish_reason=None),
            _make_usage_chunk(prompt_tokens=9, completion_tokens=4, total_tokens=13),
        ]
    )
    create_mock = AsyncMock(return_value=stream_response)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    async def _collect_chunks() -> list[ChatCompletionChunk]:
        chunks: list[ChatCompletionChunk] = []
        async for chunk in llm.chat_stream(
            messages=[{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)
        return chunks

    chunks = await asyncio.wait_for(_collect_chunks(), timeout=0.3)

    assert len(chunks) == 4
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 13


@pytest.mark.asyncio
async def test_chat_stream_fallback_without_stream_options() -> None:
    """chat_stream should retry without stream_options if provider rejects it."""

    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-openai-compatible-stream-no-stream-options",
    )
    llm = OpenAICompatible(
        api_key_pool=key_pool,
        model_name="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        retry_delay=0.0,
    )

    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._stream_finish_grace_timeout = 0.01

    stream_response = _NeverEndingStream(
        chunks=[
            _make_chunk(content="hello", finish_reason=None),
            _make_chunk(content=None, finish_reason="stop"),
        ]
    )
    create_mock = AsyncMock(
        side_effect=[
            Exception("Unknown parameter: stream_options"),
            stream_response,
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    async def _collect_chunks() -> list[ChatCompletionChunk]:
        chunks: list[ChatCompletionChunk] = []
        async for chunk in llm.chat_stream(
            messages=[{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)
        return chunks

    chunks = await asyncio.wait_for(_collect_chunks(), timeout=0.3)

    assert len(chunks) == 2
    assert create_mock.await_count == 2
    first_kwargs = create_mock.await_args_list[0].kwargs
    second_kwargs = create_mock.await_args_list[1].kwargs
    assert first_kwargs["stream_options"]["include_usage"] is True
    assert "stream_options" not in second_kwargs


@pytest.mark.asyncio
async def test_chat_stream_usage_chunks_should_not_double_count_tokens() -> None:
    """Later cumulative usage chunk should overwrite earlier usage totals."""

    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-openai-compatible-stream-usage-double-count",
    )
    llm = OpenAICompatible(
        api_key_pool=key_pool,
        model_name="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        retry_delay=0.0,
    )

    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._stream_finish_grace_timeout = 0.05

    stream_response = _NeverEndingStream(
        chunks=[
            _make_usage_chunk(prompt_tokens=3, completion_tokens=1, total_tokens=4),
            _make_chunk(content=None, finish_reason="stop"),
            _make_usage_chunk(prompt_tokens=8, completion_tokens=3, total_tokens=11),
        ]
    )
    create_mock = AsyncMock(return_value=stream_response)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    before_input = int(get_current_context_attribute("input_tokens") or 0)
    before_output = int(get_current_context_attribute("output_tokens") or 0)
    set_current_context_attribute("input_tokens", before_input)
    set_current_context_attribute("output_tokens", before_output)

    async def _consume_stream() -> None:
        async for _chunk in llm.chat_stream(
            messages=[{"role": "user", "content": "hi"}]
        ):
            pass

    await asyncio.wait_for(_consume_stream(), timeout=0.3)

    after_input = int(get_current_context_attribute("input_tokens") or 0)
    after_output = int(get_current_context_attribute("output_tokens") or 0)

    assert after_input - before_input == 8
    assert after_output - before_output == 3


def test_openai_compatible_exposes_context_window_attribute() -> None:
    """OpenAICompatible should keep configured context window metadata."""

    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-openai-compatible-context-window",
    )
    llm = OpenAICompatible(
        api_key_pool=key_pool,
        model_name="test-model",
        base_url="https://example.com/v1",
        context_window=128000,
    )

    assert llm.context_window == 128000


def test_openai_compatible_uses_placeholder_context_window_by_default() -> None:
    """OpenAICompatible should expose a 200K placeholder when unspecified."""

    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-openai-compatible-context-window-default",
    )
    llm = OpenAICompatible(
        api_key_pool=key_pool,
        model_name="test-model",
        base_url="https://example.com/v1",
    )

    assert llm.context_window == DEFAULT_CONTEXT_WINDOW


def test_load_from_json_file_reads_context_window(tmp_path) -> None:
    """Provider JSON metadata should populate context_window on instances."""

    provider_json = tmp_path / "provider.json"
    provider_json.write_text(
        json.dumps(
            {
                "openai": [
                    {
                        "model_name": "gpt-4o-mini",
                        "api_keys": ["test-key"],
                        "base_url": "https://api.openai.com/v1",
                        "context_window": 128000,
                    },
                    {
                        "model_name": "gpt-4.1",
                        "api_keys": ["test-key"],
                        "base_url": "https://api.openai.com/v1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    providers = OpenAICompatible.load_from_json_file(str(provider_json))

    assert providers["openai"]["gpt-4o-mini"].context_window == 128000
    assert providers["openai"]["gpt-4.1"].context_window == DEFAULT_CONTEXT_WINDOW


def test_load_from_json_file_reads_api_params(tmp_path) -> None:
    """Provider JSON api_params should be stored and accessible on instances."""

    provider_json = tmp_path / "provider.json"
    provider_json.write_text(
        json.dumps(
            {
                "openai": [
                    {
                        "model_name": "o3-mini",
                        "api_keys": ["test-key"],
                        "base_url": "https://api.openai.com/v1",
                        "api_params": {"reasoning_effort": "high"},
                    },
                    {
                        "model_name": "gpt-4o",
                        "api_keys": ["test-key"],
                        "base_url": "https://api.openai.com/v1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    providers = OpenAICompatible.load_from_json_file(str(provider_json))

    assert providers["openai"]["o3-mini"]._api_params == {"reasoning_effort": "high"}
    assert providers["openai"]["gpt-4o"]._api_params == {}


@pytest.mark.asyncio
async def test_chat_stream_passes_api_params(tmp_path) -> None:
    """api_params from provider.json should be passed to the API call."""

    provider_json = tmp_path / "provider.json"
    provider_json.write_text(
        json.dumps(
            {
                "test": [
                    {
                        "model_name": "o3-mini",
                        "api_keys": ["test-key"],
                        "base_url": "https://example.com/v1",
                        "api_params": {"reasoning_effort": "high", "temperature": 0.7},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    providers = OpenAICompatible.load_from_json_file(str(provider_json))
    llm = providers["test"]["o3-mini"]

    create_mock = AsyncMock(
        return_value=_NeverEndingStream([
            _make_chunk("hello", "stop"),
            _make_usage_chunk(10, 5, 15),
        ])
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    chunks = []
    async for chunk in llm.chat_stream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert create_mock.await_count == 1
    create_kwargs = create_mock.await_args_list[0].kwargs
    assert create_kwargs.get("reasoning_effort") == "high"
    assert create_kwargs.get("temperature") == 0.7


@pytest.mark.asyncio
async def test_chat_stream_call_kwargs_override_api_params(tmp_path) -> None:
    """Call-level kwargs should override instance api_params."""

    provider_json = tmp_path / "provider.json"
    provider_json.write_text(
        json.dumps(
            {
                "test": [
                    {
                        "model_name": "o3-mini",
                        "api_keys": ["test-key"],
                        "base_url": "https://example.com/v1",
                        "api_params": {"reasoning_effort": "low"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    providers = OpenAICompatible.load_from_json_file(str(provider_json))
    llm = providers["test"]["o3-mini"]

    create_mock = AsyncMock(
        return_value=_NeverEndingStream([
            _make_chunk("hello", "stop"),
            _make_usage_chunk(10, 5, 15),
        ])
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    chunks = []
    async for chunk in llm.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",  # override
    ):
        chunks.append(chunk)

    assert create_mock.await_count == 1
    create_kwargs = create_mock.await_args_list[0].kwargs
    # Call-level override should win
    assert create_kwargs.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_chat_passes_api_params(tmp_path) -> None:
    """api_params should also be passed by non-streaming chat (used by llm_function)."""

    provider_json = tmp_path / "provider.json"
    provider_json.write_text(
        json.dumps(
            {
                "test": [
                    {
                        "model_name": "o3-mini",
                        "api_keys": ["test-key"],
                        "base_url": "https://example.com/v1",
                        "api_params": {"reasoning_effort": "high", "temperature": 0.7},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    providers = OpenAICompatible.load_from_json_file(str(provider_json))
    llm = providers["test"]["o3-mini"]

    create_mock = AsyncMock(return_value=_make_completion())
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(messages=[{"role": "user", "content": "hi"}])

    assert create_mock.await_count == 1
    create_kwargs = create_mock.await_args_list[0].kwargs
    assert create_kwargs.get("reasoning_effort") == "high"
    assert create_kwargs.get("temperature") == 0.7


@pytest.mark.asyncio
async def test_chat_call_kwargs_override_api_params(tmp_path) -> None:
    """Call-level kwargs should override api_params for non-streaming chat."""

    provider_json = tmp_path / "provider.json"
    provider_json.write_text(
        json.dumps(
            {
                "test": [
                    {
                        "model_name": "o3-mini",
                        "api_keys": ["test-key"],
                        "base_url": "https://example.com/v1",
                        "api_params": {"reasoning_effort": "low"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    providers = OpenAICompatible.load_from_json_file(str(provider_json))
    llm = providers["test"]["o3-mini"]

    create_mock = AsyncMock(return_value=_make_completion())
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    llm.token_bucket.acquire = AsyncMock(return_value=True)
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    await llm.chat(
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )

    assert create_mock.await_count == 1
    create_kwargs = create_mock.await_args_list[0].kwargs
    assert create_kwargs.get("reasoning_effort") == "high"
