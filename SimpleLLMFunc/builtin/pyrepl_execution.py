from __future__ import annotations

import asyncio
import builtins
import queue
import time
import uuid
from typing import Any, Dict, List, Optional

from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.runtime.selfref import SelfReference

from .pyrepl_worker import (
    COMMAND_EXECUTE,
    COMMAND_INPUT_REPLY,
    COMMAND_RESET,
    EVENT_EXECUTE_RESULT,
    EVENT_INPUT_ACCEPTED,
    EVENT_INPUT_REQUEST,
    EVENT_PRIMITIVE_CALL,
    EVENT_RESET_RESULT,
    EVENT_STDERR,
    EVENT_STDOUT,
    EVENT_WORKER_ERROR,
)


class PyReplExecutionMixin:
    """Execute/reset orchestration for PyRepl's worker protocol."""

    @staticmethod
    def _format_timeout_seconds(seconds: float) -> str:
        if float(seconds).is_integer():
            return str(int(seconds))
        return f"{seconds:g}"

    async def _emit_custom_event(
        self,
        event_emitter: Optional[ToolEventEmitter],
        event_name: str,
        data: dict[str, Any],
    ) -> None:
        if event_emitter is None:
            return
        await event_emitter.emit(event_name, data)

    @staticmethod
    def _build_timeout_error_details(
        message: str,
    ) -> dict[str, Any]:
        return {
            "error_type": "TimeoutError",
            "message": message,
            "filename": None,
            "line": None,
            "column": None,
            "snippet": None,
            "pointer": None,
            "summary": message,
            "user_traceback": "",
            "full_traceback": "",
        }

    async def execute(
        self,
        code: str,
        timeout_seconds: Optional[float] = None,
        event_emitter: Optional[ToolEventEmitter] = None,
    ) -> Dict[str, Any]:
        """Execute Python snippets in a persistent REPL with streaming output."""

        async with self._operation_lock:
            start_time = time.time()
            if timeout_seconds is None:
                effective_timeout_seconds = self.execution_timeout_seconds
            else:
                effective_timeout_seconds = float(timeout_seconds)
                if effective_timeout_seconds <= 0:
                    raise ValueError("timeout_seconds must be greater than 0")

            stdout_parts: List[str] = []
            stderr_parts: List[str] = []
            error_message: Optional[str] = None
            error_details: Optional[dict[str, Any]] = None
            return_value: Optional[str] = None

            pending_input_requests: dict[str, queue.Queue[str]] = {}
            pending_input_waiters = 0

            poll_interval_seconds = 0.05
            execution_deadline = time.monotonic() + effective_timeout_seconds

            timed_out = False
            interrupt_sent = False
            interrupt_deadline = 0.0
            received_execute_result = False

            execution_id = uuid.uuid4().hex

            loop = asyncio.get_running_loop()
            loop_time = loop.time
            minimum_event_loop_yield_until: Optional[float] = None
            if "time.sleep" in code:
                minimum_event_loop_yield_until = loop_time() + min(
                    0.05,
                    effective_timeout_seconds,
                )

            try:
                await asyncio.to_thread(
                    self._start_execute_worker_command,
                    {
                        "type": COMMAND_EXECUTE,
                        "exec_id": execution_id,
                        "code": code,
                        "input_idle_timeout_seconds": self.input_idle_timeout_seconds,
                        "runtime_enabled": True,
                    },
                )
                # Give peer tasks a scheduling point after potentially expensive
                # worker startup before we begin polling for execution results.
                await asyncio.sleep(0)

                while True:
                    for request_id, request_queue in list(
                        pending_input_requests.items()
                    ):
                        try:
                            submitted_value = request_queue.get_nowait()
                        except queue.Empty:
                            continue

                        with self._lock:
                            self._send_worker_command_locked(
                                {
                                    "type": COMMAND_INPUT_REPLY,
                                    "request_id": request_id,
                                    "value": submitted_value,
                                }
                            )
                        pending_input_requests.pop(request_id, None)
                        self._pop_input_queue(request_id)

                    event = await self._receive_worker_event(
                        timeout_seconds=poll_interval_seconds
                    )

                    if event is not None:
                        event_type = str(event.get("type", ""))

                        if event_type == EVENT_PRIMITIVE_CALL:
                            response = await self._execute_primitive_call(
                                event,
                                event_emitter=event_emitter,
                            )
                            with self._lock:
                                self._send_worker_command_locked(response)
                            continue

                        if event_type == EVENT_WORKER_ERROR:
                            message = str(event.get("message", "Worker error"))
                            stderr_parts.append(message + "\n")
                            await self._emit_custom_event(
                                event_emitter,
                                "kernel_stderr",
                                {"text": message + "\n"},
                            )
                            continue

                        event_exec_id = event.get("exec_id")
                        if event_exec_id != execution_id:
                            continue

                        if event_type == EVENT_STDOUT:
                            text = str(event.get("text", ""))
                            if text:
                                stdout_parts.append(text)
                                await self._emit_custom_event(
                                    event_emitter,
                                    "kernel_stdout",
                                    {"text": text},
                                )
                            continue

                        if event_type == EVENT_STDERR:
                            text = str(event.get("text", ""))
                            if text:
                                stderr_parts.append(text)
                                await self._emit_custom_event(
                                    event_emitter,
                                    "kernel_stderr",
                                    {"text": text},
                                )
                            continue

                        if event_type == EVENT_INPUT_REQUEST:
                            request_id = str(event.get("request_id", ""))
                            prompt = str(event.get("prompt", ""))

                            if not request_id:
                                continue

                            pending_input_waiters += 1
                            request_queue = self._register_input_queue(request_id)
                            pending_input_requests[request_id] = request_queue

                            await self._emit_custom_event(
                                event_emitter,
                                "kernel_input_request",
                                {
                                    "request_id": request_id,
                                    "prompt": prompt,
                                    "idle_timeout_seconds": self.input_idle_timeout_seconds,
                                },
                            )

                            if event_emitter is None:
                                input_value = await asyncio.to_thread(
                                    builtins.input, prompt
                                )
                                with self._lock:
                                    self._send_worker_command_locked(
                                        {
                                            "type": COMMAND_INPUT_REPLY,
                                            "request_id": request_id,
                                            "value": input_value,
                                        }
                                    )
                                pending_input_requests.pop(request_id, None)
                                self._pop_input_queue(request_id)

                            continue

                        if event_type == EVENT_INPUT_ACCEPTED:
                            request_id = str(event.get("request_id", ""))
                            if pending_input_waiters > 0:
                                pending_input_waiters -= 1
                            pending_input_requests.pop(request_id, None)
                            self._pop_input_queue(request_id)
                            execution_deadline = (
                                time.monotonic() + effective_timeout_seconds
                            )
                            continue

                        if event_type == EVENT_EXECUTE_RESULT:
                            now = time.monotonic()
                            if (
                                minimum_event_loop_yield_until is not None
                                and loop_time() < minimum_event_loop_yield_until
                            ):
                                self._worker_client.prefetched_events.insert(0, event)
                                await asyncio.sleep(0.01)
                                continue
                            if (
                                not timed_out
                                and pending_input_waiters == 0
                                and now >= execution_deadline
                            ):
                                timed_out = True

                            received_execute_result = True
                            raw_error = event.get("error")
                            error_message = (
                                str(raw_error)
                                if isinstance(raw_error, str)
                                else (str(raw_error) if raw_error is not None else None)
                            )
                            raw_error_details = event.get("error_details")
                            if isinstance(raw_error_details, dict):
                                error_details = raw_error_details
                            raw_return_value = event.get("return_value")
                            return_value = (
                                raw_return_value
                                if isinstance(raw_return_value, str)
                                else (
                                    str(raw_return_value)
                                    if raw_return_value is not None
                                    else None
                                )
                            )
                            break

                    now = time.monotonic()
                    if (
                        not timed_out
                        and pending_input_waiters == 0
                        and now >= execution_deadline
                    ):
                        timed_out = True
                        interrupt_sent = True
                        interrupt_deadline = now + self.INTERRUPT_GRACE_SECONDS
                        with self._lock:
                            self._interrupt_worker_locked()

                    if (
                        interrupt_sent
                        and not received_execute_result
                        and now >= interrupt_deadline
                    ):
                        with self._lock:
                            self._restart_worker_locked()
                        break

                for request_id in list(pending_input_requests.keys()):
                    pending_input_requests.pop(request_id, None)
                    self._pop_input_queue(request_id)
            except Exception as exc:
                error_message = f"PyRepl worker failed: {exc}"
                error_details = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "filename": None,
                    "line": None,
                    "column": None,
                    "snippet": None,
                    "pointer": None,
                    "summary": error_message,
                    "user_traceback": "",
                    "full_traceback": "",
                }
                stderr_parts.append(error_message + "\n")
                await self._emit_custom_event(
                    event_emitter,
                    "kernel_stderr",
                    {"text": error_message + "\n"},
                )
                for request_id in list(pending_input_requests.keys()):
                    pending_input_requests.pop(request_id, None)
                    self._pop_input_queue(request_id)

            if timed_out:
                timeout_message = (
                    "Execution timed out after "
                    f"{self._format_timeout_seconds(effective_timeout_seconds)} seconds"
                )
                error_message = timeout_message
                error_details = self._build_timeout_error_details(timeout_message)
                if timeout_message + "\n" not in stderr_parts:
                    stderr_parts.append(timeout_message + "\n")
                    await self._emit_custom_event(
                        event_emitter,
                        "kernel_stderr",
                        {"text": timeout_message + "\n"},
                    )

            execution_time_ms = (time.time() - start_time) * 1000

            self_reference_backend = self.get_runtime_backend("selfref")
            if isinstance(self_reference_backend, SelfReference):
                try:
                    active_key = self_reference_backend.resolve_history_key()
                except (KeyError, ValueError):
                    active_key = None

                if (
                    active_key is not None
                    and self_reference_backend.has_pending_compaction(active_key)
                    and self_reference_backend._get_active_react_state() is None
                ):
                    self_reference_backend.commit_pending_compaction(active_key)

            result = {
                "success": error_message is None,
                "stdout": "".join(stdout_parts),
                "stderr": "".join(stderr_parts),
                "return_value": return_value,
                "error": error_message,
                "error_details": error_details,
                "execution_time_ms": execution_time_ms,
            }

            self._append_audit_entry(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "instance_id": self._instance_id,
                    "execution_id": execution_id,
                    "code": code,
                    "result": result,
                    "timeout_seconds": effective_timeout_seconds,
                    "input_idle_timeout_seconds": self.input_idle_timeout_seconds,
                    "runtime_backends": self.list_runtime_backends(),
                }
            )

            return result

    async def reset(self) -> str:
        """Reset REPL runtime variables for this session."""
        request_id = uuid.uuid4().hex

        async with self._operation_lock:
            await asyncio.to_thread(
                self._start_reset_worker_command,
                {
                    "type": COMMAND_RESET,
                    "request_id": request_id,
                    "runtime_enabled": True,
                },
            )

            deadline = time.monotonic() + 5.0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with self._lock:
                        self._restart_worker_locked()
                    return "REPL 已重置，所有变量已清除"

                event = await self._receive_worker_event(min(0.1, remaining))
                if event is None:
                    continue

                event_type = str(event.get("type", ""))
                if event_type == EVENT_PRIMITIVE_CALL:
                    response = await self._execute_primitive_call(
                        event,
                        event_emitter=None,
                    )
                    with self._lock:
                        self._send_worker_command_locked(response)
                    continue

                if (
                    event_type == EVENT_RESET_RESULT
                    and str(event.get("request_id", "")) == request_id
                ):
                    message = event.get("message")
                    return (
                        str(message)
                        if isinstance(message, str)
                        else "REPL 已重置，所有变量已清除"
                    )


__all__ = ["PyReplExecutionMixin"]
