from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

from .pyrepl_worker import COMMAND_SHUTDOWN, EVENT_WORKER_ERROR, EVENT_WORKER_READY, run_pyrepl_worker


class PyReplWorkerClient:
    """Subprocess and queue lifecycle for PyRepl's IPython worker."""

    def __init__(self, working_directory: Optional[Union[str, Path]]) -> None:
        self._working_directory = (
            str(working_directory) if working_directory is not None else None
        )
        self._ctx = mp.get_context("spawn")
        self.command_queue: Any = None
        self.event_queue: Any = None
        self.process: Any = None
        self.prefetched_events: list[dict[str, Any]] = []

    @staticmethod
    def stream_fileno(stream: Any) -> Optional[int]:
        if stream is None:
            return None

        fileno = getattr(stream, "fileno", None)
        if fileno is None:
            return None

        try:
            value = fileno()
        except Exception:
            return None

        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def close_queue_handle(queue_handle: Any) -> None:
        if queue_handle is None:
            return

        close = getattr(queue_handle, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

        join_thread = getattr(queue_handle, "join_thread", None)
        if callable(join_thread):
            try:
                join_thread()
            except Exception:
                pass

    @contextlib.contextmanager
    def temporary_valid_stderr(self):
        current_stderr = sys.stderr
        current_fd = self.stream_fileno(current_stderr)
        if current_fd is not None and current_fd >= 0:
            yield
            return

        temp_stream = None
        replacement = sys.__stderr__
        replacement_fd = self.stream_fileno(replacement)

        if replacement_fd is None or replacement_fd < 0:
            temp_stream = open(os.devnull, "w", encoding="utf-8")
            replacement = temp_stream

        sys.stderr = replacement
        try:
            yield
        finally:
            sys.stderr = current_stderr
            if temp_stream is not None:
                temp_stream.close()

    def ensure_worker(self, *, closed: bool) -> None:
        if closed:
            raise RuntimeError("PyRepl is closed")

        if self.process is not None and self.process.is_alive():
            return

        with self.temporary_valid_stderr():
            self.command_queue = self._ctx.Queue()
            self.event_queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=run_pyrepl_worker,
                args=(
                    self.command_queue,
                    self.event_queue,
                    self._working_directory,
                ),
                daemon=True,
            )
            process.start()
        self.process = process

        assert self.event_queue is not None
        startup_deadline = time.monotonic() + 10.0
        while time.monotonic() < startup_deadline:
            if not process.is_alive():
                raise RuntimeError("PyRepl worker exited before startup")

            try:
                event = self.event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
            if event_type == EVENT_WORKER_READY:
                return

            if event_type == EVENT_WORKER_ERROR:
                message = str(event.get("message", "PyRepl worker error"))
                raise RuntimeError(message)

            if isinstance(event, dict):
                self.prefetched_events.append(event)

        raise RuntimeError("Timed out waiting for PyRepl worker startup")

    def drain_event_queue(self) -> None:
        self.prefetched_events.clear()

        if self.event_queue is None:
            return

        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                return

    def send_command(self, command: dict[str, Any], *, closed: bool) -> None:
        self.ensure_worker(closed=closed)
        assert self.command_queue is not None
        self.command_queue.put(command)

    async def receive_event(self, timeout_seconds: float) -> Optional[dict[str, Any]]:
        if self.prefetched_events:
            return self.prefetched_events.pop(0)
        event_queue = self.event_queue

        if event_queue is None:
            return None

        try:
            return await asyncio.to_thread(
                event_queue.get,
                True,
                timeout_seconds,
            )
        except queue.Empty:
            return None

    def interrupt_worker(self) -> None:
        process = self.process
        if process is None or not process.is_alive():
            return

        pid = process.pid
        if not pid:
            return

        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def shutdown_worker(self) -> None:
        process = self.process
        command_queue = self.command_queue
        event_queue = self.event_queue

        if process is not None:
            if process.is_alive():
                try:
                    self.send_command({"type": COMMAND_SHUTDOWN}, closed=False)
                except Exception:
                    pass

                process.join(timeout=1.0)

            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

            close_process = getattr(process, "close", None)
            if callable(close_process):
                try:
                    close_process()
                except Exception:
                    pass

        self.process = None
        self.command_queue = None
        self.event_queue = None
        self.prefetched_events.clear()

        self.close_queue_handle(command_queue)
        self.close_queue_handle(event_queue)

    def restart_worker(self, *, closed: bool) -> None:
        self.shutdown_worker()
        self.ensure_worker(closed=closed)


__all__ = ["PyReplWorkerClient"]
