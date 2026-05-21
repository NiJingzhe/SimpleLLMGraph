from __future__ import annotations

import contextlib
from typing import Any, Optional

from .pyrepl_worker_client import PyReplWorkerClient


class PyReplWorkerMixin:
    """Facade-compatible worker lifecycle wrappers for PyRepl."""

    @staticmethod
    def _stream_fileno(stream: Any) -> Optional[int]:
        return PyReplWorkerClient.stream_fileno(stream)

    @staticmethod
    def _close_queue_handle(queue_handle: Any) -> None:
        PyReplWorkerClient.close_queue_handle(queue_handle)

    @contextlib.contextmanager
    def _temporary_valid_stderr(self):
        """Temporarily ensure ``sys.stderr`` has a valid POSIX file descriptor."""

        with self._worker_client.temporary_valid_stderr():
            yield

    def _sync_worker_aliases(self) -> None:
        self._command_queue = self._worker_client.command_queue
        self._event_queue = self._worker_client.event_queue
        self._process = self._worker_client.process
        self._prefetched_events = self._worker_client.prefetched_events

    def _sync_worker_client_from_aliases(self) -> None:
        self._worker_client.command_queue = self._command_queue
        self._worker_client.event_queue = self._event_queue
        self._worker_client.process = self._process
        self._worker_client.prefetched_events = self._prefetched_events

    def _ensure_worker_locked(self) -> None:
        self._worker_client.ensure_worker(closed=self._closed)
        self._sync_worker_aliases()

    def _drain_event_queue_locked(self) -> None:
        self._worker_client.drain_event_queue()
        self._sync_worker_aliases()

    def _send_worker_command_locked(self, command: dict[str, Any]) -> None:
        self._worker_client.send_command(command, closed=self._closed)
        self._sync_worker_aliases()

    async def _receive_worker_event(
        self,
        timeout_seconds: float,
    ) -> Optional[dict[str, Any]]:
        event = await self._worker_client.receive_event(timeout_seconds)
        with self._lock:
            self._sync_worker_aliases()
        return event

    def _interrupt_worker_locked(self) -> None:
        self._worker_client.interrupt_worker()
        self._sync_worker_aliases()

    def _shutdown_worker_locked(self) -> None:
        self._sync_worker_client_from_aliases()
        self._worker_client.shutdown_worker()
        self._sync_worker_aliases()

    def _restart_worker_locked(self) -> None:
        self._worker_client.restart_worker(closed=self._closed)
        self._sync_worker_aliases()

    def _start_execute_worker_command(self, command: dict[str, Any]) -> None:
        """Start an execute command without blocking the asyncio event loop."""

        with self._lock:
            self._ensure_worker_locked()
            self._drain_event_queue_locked()
            self._send_worker_command_locked(command)

    def _start_reset_worker_command(self, command: dict[str, Any]) -> None:
        """Start a reset command without blocking the asyncio event loop."""

        with self._lock:
            self.namespace.clear()
            self._ensure_worker_locked()
            self._send_worker_command_locked(command)


__all__ = ["PyReplWorkerMixin"]
