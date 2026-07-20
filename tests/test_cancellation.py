import asyncio

import pytest

from SimpleLLMFunc.cancellation import CancellationToken, await_with_cancellation


@pytest.mark.asyncio
async def test_await_with_cancellation_does_not_start_pre_cancelled_operation() -> None:
    started = False

    async def operation() -> int:
        nonlocal started
        started = True
        return 1

    cancellation = CancellationToken()
    cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await await_with_cancellation(operation, cancellation)

    assert not started


@pytest.mark.asyncio
async def test_await_with_cancellation_cancels_in_flight_operation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def operation() -> int:
        started.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    cancellation = CancellationToken()
    task = asyncio.create_task(
        await_with_cancellation(operation, cancellation)
    )
    await started.wait()
    cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_await_with_cancellation_prefers_completed_operation() -> None:
    cancellation = CancellationToken()

    async def operation() -> int:
        cancellation.cancel()
        return 7

    assert await await_with_cancellation(operation, cancellation) == 7


@pytest.mark.asyncio
async def test_await_with_cancellation_without_signal_is_a_plain_await() -> None:
    async def operation() -> int:
        return 3

    assert await await_with_cancellation(operation, None) == 3
