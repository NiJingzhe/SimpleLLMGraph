import asyncio
from typing import Any, cast

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from SimpleLLMFunc.cancellation import CancellationToken
from SimpleLLMFunc.context.ir import Chunk, Request, UserMessage
from SimpleLLMFunc.interface import (
    APIKeyPool,
    LLM_Interface,
    OpenAICompatible,
    OpenAIResponsesCompatible,
)


def request() -> Request:
    return Request(
        model="test-model",
        messages=[UserMessage(content="hello")],
    )


class BlockingCreate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.calls = 0

    async def __call__(self, **_: object) -> object:
        self.calls += 1
        self.started.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


class BlockingStream:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()

    def __aiter__(self) -> "BlockingStream":
        return self

    async def __anext__(self) -> object:
        self.started.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.closed.set()


class FailingAfterChunkStream:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.yielded = False

    def __aiter__(self) -> "FailingAfterChunkStream":
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        if self.yielded:
            raise RuntimeError("stream failed after output")
        self.yielded = True
        return ChatCompletionChunk.model_validate(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "test-model",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "x"},
                        "finish_reason": None,
                    }
                ],
            }
        )

    async def close(self) -> None:
        self.closed.set()


class OneChunkStream:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.yielded = False

    def __aiter__(self) -> "OneChunkStream":
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        if self.yielded:
            await asyncio.Future[None]()
            raise AssertionError("unreachable")
        self.yielded = True
        return ChatCompletionChunk.model_validate(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "test-model",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "x"},
                        "finish_reason": None,
                    }
                ],
            }
        )

    async def close(self) -> None:
        self.closed.set()


class FakeChatCompletions:
    def __init__(self, create: BlockingCreate) -> None:
        self.create = create


class FakeChat:
    def __init__(self, create: BlockingCreate) -> None:
        self.completions = FakeChatCompletions(create)


class FakeResponses:
    def __init__(self, create: BlockingCreate) -> None:
        self.create = create


class FakeClient:
    def __init__(self, create: BlockingCreate) -> None:
        self.chat = FakeChat(create)
        self.responses = FakeResponses(create)

    async def close(self) -> None:
        return None


class TrackingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__(BlockingCreate())
        self.closed = asyncio.Event()

    async def close(self) -> None:
        self.closed.set()


class BlockingCloseClient(TrackingClient):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def close(self) -> None:
        self.close_started.set()
        await self.release_close.wait()
        await super().close()


def adapter(
    adapter_type: type[LLM_Interface],
    provider_id: str,
) -> tuple[LLM_Interface, APIKeyPool]:
    pool = APIKeyPool(["test-key"], provider_id)
    if adapter_type is OpenAICompatible:
        value = OpenAICompatible(
            api_key_pool=pool,
            model_name="test-model",
            base_url=f"https://{provider_id}.example",
            max_retries=3,
            retry_delay=0.01,
        )
    else:
        value = OpenAIResponsesCompatible(
            api_key_pool=pool,
            model_name="test-model",
            base_url=f"https://{provider_id}.example",
            max_retries=3,
            retry_delay=0.01,
        )
    return value, pool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_type",
    [OpenAICompatible, OpenAIResponsesCompatible],
)
async def test_provider_chat_cancellation_interrupts_http_creation(
    adapter_type: type[LLM_Interface],
) -> None:
    llm, pool = adapter(adapter_type, f"chat-cancel-{adapter_type.__name__}")
    create = BlockingCreate()
    fake_client = FakeClient(create)
    llm.client = cast(Any, fake_client)  # type: ignore[attr-defined]
    setattr(llm, "_current_key", "test-key")
    cancellation = CancellationToken()

    task = asyncio.create_task(
        llm.chat(request(), cancellation=cancellation),
    )
    await create.started.wait()
    cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert create.cancelled.is_set()
    assert create.calls == 1
    assert pool.key_to_task_count["test-key"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_type",
    [OpenAICompatible, OpenAIResponsesCompatible],
)
async def test_provider_stream_cancellation_closes_response(
    adapter_type: type[LLM_Interface],
) -> None:
    llm, pool = adapter(adapter_type, f"stream-cancel-{adapter_type.__name__}")
    create = BlockingCreate()
    stream = BlockingStream()

    async def create_stream(**_: object) -> BlockingStream:
        create.calls += 1
        create.started.set()
        return stream

    fake_client = FakeClient(create)
    if adapter_type is OpenAICompatible:
        fake_client.chat.completions.create = cast(Any, create_stream)
    else:
        fake_client.responses.create = cast(Any, create_stream)
    llm.client = cast(Any, fake_client)  # type: ignore[attr-defined]
    setattr(llm, "_current_key", "test-key")
    cancellation = CancellationToken()

    async def consume() -> None:
        async for _ in llm.chat_stream(request(), cancellation=cancellation):
            pass

    task = asyncio.create_task(consume())
    await create.started.wait()
    await stream.started.wait()
    cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.cancelled.is_set()
    assert stream.closed.is_set()
    assert pool.key_to_task_count["test-key"] == 0


@pytest.mark.asyncio
async def test_chat_stream_does_not_retry_after_visible_output() -> None:
    llm, pool = adapter(OpenAICompatible, "stream-no-retry-after-output")
    create = BlockingCreate()
    stream = FailingAfterChunkStream()

    async def create_stream(**_: object) -> FailingAfterChunkStream:
        create.calls += 1
        return stream

    fake_client = FakeClient(create)
    fake_client.chat.completions.create = cast(Any, create_stream)
    llm.client = cast(Any, fake_client)  # type: ignore[attr-defined]
    setattr(llm, "_current_key", "test-key")

    chunks: list[Chunk] = []
    with pytest.raises(RuntimeError, match="failed after output"):
        async for chunk in llm.chat_stream(request()):
            chunks.append(chunk)

    assert [chunk.choices[0].delta.content for chunk in chunks] == ["x"]
    assert create.calls == 1
    assert stream.closed.is_set()
    assert pool.key_to_task_count["test-key"] == 0


@pytest.mark.asyncio
async def test_closing_partially_consumed_stream_releases_resources() -> None:
    llm, pool = adapter(OpenAICompatible, "stream-explicit-close")
    create = BlockingCreate()
    stream = OneChunkStream()

    async def create_stream(**_: object) -> OneChunkStream:
        create.calls += 1
        return stream

    fake_client = FakeClient(create)
    fake_client.chat.completions.create = cast(Any, create_stream)
    llm.client = cast(Any, fake_client)  # type: ignore[attr-defined]
    setattr(llm, "_current_key", "test-key")

    output = llm.chat_stream(request())
    chunk = await anext(output)
    assert chunk.choices[0].delta.content == "x"
    await output.aclose()

    assert create.calls == 1
    assert stream.closed.is_set()
    assert pool.key_to_task_count["test-key"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_type",
    [OpenAICompatible, OpenAIResponsesCompatible],
)
async def test_provider_clients_are_kept_per_key_and_closed_together(
    adapter_type: type[LLM_Interface],
) -> None:
    provider_id = f"client-lifecycle-{adapter_type.__name__}"
    pool = APIKeyPool(["key-a", "key-b"], provider_id)
    if adapter_type is OpenAICompatible:
        llm: Any = OpenAICompatible(
            api_key_pool=pool,
            model_name="test-model",
            base_url=f"https://{provider_id}.example",
        )
    else:
        llm = OpenAIResponsesCompatible(
            api_key_pool=pool,
            model_name="test-model",
            base_url=f"https://{provider_id}.example",
        )
    client_a = TrackingClient()
    client_b = TrackingClient()
    llm.client = client_a
    llm._current_key = "key-a"
    llm._clients = {"key-a": client_a, "key-b": client_b}

    selected = await llm._get_or_create_client("key-b")

    assert selected is client_b
    assert not client_a.closed.is_set()
    await llm.aclose()
    assert client_a.closed.is_set()
    assert client_b.closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_type",
    [OpenAICompatible, OpenAIResponsesCompatible],
)
async def test_cancelled_provider_shutdown_retains_unclosed_clients(
    adapter_type: type[LLM_Interface],
) -> None:
    provider_id = f"cancel-close-{adapter_type.__name__}"
    pool = APIKeyPool(["key-a", "key-b"], provider_id)
    if adapter_type is OpenAICompatible:
        llm: Any = OpenAICompatible(
            api_key_pool=pool,
            model_name="test-model",
            base_url=f"https://{provider_id}.example",
        )
    else:
        llm = OpenAIResponsesCompatible(
            api_key_pool=pool,
            model_name="test-model",
            base_url=f"https://{provider_id}.example",
        )
    client_a = BlockingCloseClient()
    client_b = TrackingClient()
    llm.client = client_a
    llm._current_key = "key-a"
    llm._clients = {"key-a": client_a, "key-b": client_b}
    closing = asyncio.create_task(llm.aclose())
    await client_a.close_started.wait()

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert set(llm._clients) == {"key-a", "key-b"}
    client_a.release_close.set()
    await llm.aclose()
    assert client_a.closed.is_set()
    assert client_b.closed.is_set()
