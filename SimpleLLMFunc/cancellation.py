"""Provider-neutral cancellation primitives."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


class CancellationToken:
    """Cooperative abort signal shared across one framework operation."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""

        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation. Repeated calls are harmless."""

        self._event.set()

    async def wait(self) -> None:
        """Wait until cancellation is requested."""

        await self._event.wait()


async def await_with_cancellation(
    operation: Callable[[], Awaitable[T]],
    cancellation: CancellationToken | None,
) -> T:
    """Await an operation while allowing a shared abort signal to interrupt it.

    A completed operation wins if completion and cancellation become observable
    in the same scheduling turn. The operation factory prevents a pre-cancelled
    token from starting the underlying provider request.
    """

    if cancellation is None:
        return await operation()
    if cancellation.cancelled:
        raise asyncio.CancelledError

    operation_task: asyncio.Future[T] = asyncio.ensure_future(operation())
    cancellation_task = asyncio.create_task(cancellation.wait())
    try:
        done, _ = await asyncio.wait(
            {operation_task, cancellation_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return operation_task.result()

        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise asyncio.CancelledError
    finally:
        cancellation_task.cancel()
        operation_task.cancel()
        await asyncio.gather(
            operation_task,
            cancellation_task,
            return_exceptions=True,
        )
