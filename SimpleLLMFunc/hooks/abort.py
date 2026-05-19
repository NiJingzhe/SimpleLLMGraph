"""Abort signal utilities for interrupting agent turns."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

ABORT_SIGNAL_PARAM = "_abort_signal"


@dataclass
class AbortSignal:
    """Lightweight abort signal shared across an agent turn."""

    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _reason: str = ""

    def abort(self, reason: str = "") -> None:
        if reason:
            self._reason = reason
        self._event.set()

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def is_aborted(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()


__all__ = [
    "AbortSignal",
    "ABORT_SIGNAL_PARAM",
]
