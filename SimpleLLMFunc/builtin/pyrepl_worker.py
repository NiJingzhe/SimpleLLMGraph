"""Subprocess worker runtime for PyRepl.

This module hosts an IPython ``InteractiveShell`` in a dedicated process.
Parent/worker communication is message-based via multiprocessing queues.
"""

from __future__ import annotations

import ast
import base64
import builtins
import io
import linecache
import os
import queue
import sys
import tempfile
import time
import traceback
import uuid
from typing import Any, Callable, Optional

import IPython.display as ipython_display
from IPython.core.interactiveshell import InteractiveShell
from SimpleLLMFunc.runtime.worker_proxy import WorkerRuntimeProxy
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl


EVENT_STDOUT = "stdout"
EVENT_STDERR = "stderr"
EVENT_INPUT_REQUEST = "input_request"
EVENT_INPUT_ACCEPTED = "input_accepted"
EVENT_PRIMITIVE_CALL = "primitive_call"
EVENT_EXECUTE_RESULT = "execute_result"
EVENT_RESET_RESULT = "reset_result"
EVENT_WORKER_ERROR = "worker_error"
EVENT_WORKER_READY = "worker_ready"

COMMAND_EXECUTE = "execute"
COMMAND_RESET = "reset"
COMMAND_INPUT_REPLY = "input_reply"
COMMAND_PRIMITIVE_RESULT = "primitive_result"
COMMAND_SHUTDOWN = "shutdown"

RUNTIME_GLOBAL_NAME = "runtime"


class _InputRequestTimeoutError(TimeoutError):
    """Raised when a tool input request is not answered in time."""


class _WorkerControlInterruptedError(RuntimeError):
    """Raised when a worker control flow gets interrupted unexpectedly."""


class _LineCapture(io.TextIOBase):
    """Line buffered writer that emits complete lines via callback."""

    def __init__(self, callback: Callable[[str], None]):
        self._callback = callback
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._callback(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._callback(self._buffer)
            self._buffer = ""


class _PyReplWorker:
    """Process-local worker that executes code in a persistent shell."""

    def __init__(self, command_queue: Any, event_queue: Any):
        self._command_queue = command_queue
        self._event_queue = event_queue
        self._pending_commands: list[dict[str, Any]] = []
        self._active_exec_id: Optional[str] = None
        self._input_idle_timeout_seconds = 300.0
        self._cell_counter = 0

        self._shell = InteractiveShell()
        self._namespace = self._shell.user_ns
        self._shell.colors = "NoColor"
        self._baseline_names = set(self._namespace.keys())

        self._runtime_proxy = WorkerRuntimeProxy(self)
        self._runtime_enabled = True

    def run_forever(self) -> None:
        self._emit(EVENT_WORKER_READY)

        while True:
            try:
                command = self._next_command(timeout=None)
            except (EOFError, KeyboardInterrupt):
                break

            if command is None:
                continue

            command_type = str(command.get("type", ""))

            if command_type == COMMAND_SHUTDOWN:
                break

            if command_type == COMMAND_EXECUTE:
                self._handle_execute(command)
                continue

            if command_type == COMMAND_RESET:
                self._handle_reset(command)
                continue

            if command_type in {COMMAND_INPUT_REPLY, COMMAND_PRIMITIVE_RESULT}:
                self._pending_commands.append(command)
                continue

            self._emit(
                EVENT_WORKER_ERROR,
                message=f"Unknown worker command: {command_type}",
            )

    def _emit(self, event_type: str, **payload: Any) -> None:
        message = {"type": event_type, **payload}
        self._event_queue.put(message)

    def _next_command(self, timeout: Optional[float]) -> Optional[dict[str, Any]]:
        if self._pending_commands:
            return self._pending_commands.pop(0)
        if timeout is None:
            return self._command_queue.get()
        try:
            return self._command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _wait_for_command(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        timeout: Optional[float],
    ) -> Optional[dict[str, Any]]:
        deadline = None if timeout is None else (time.monotonic() + timeout)

        while True:
            if deadline is None:
                command = self._next_command(timeout=None)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                command = self._next_command(timeout=min(0.1, remaining))

            if command is None:
                continue

            if predicate(command):
                return command

            self._pending_commands.append(command)

    def _sync_runtime_binding(
        self,
        *,
        runtime_enabled: bool,
    ) -> None:
        self._runtime_enabled = bool(runtime_enabled)

        if self._runtime_enabled:
            self._namespace[RUNTIME_GLOBAL_NAME] = self._runtime_proxy
        else:
            self._namespace.pop(RUNTIME_GLOBAL_NAME, None)

    def _new_cell_filename(self) -> str:
        self._cell_counter += 1
        return f"pyrepl-cell-{self._cell_counter}"

    @staticmethod
    def _cache_source(filename: str, code: str) -> None:
        lines = code.splitlines(keepends=True)
        if not lines:
            lines = ["\n"]
        elif not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        linecache.cache[filename] = (len(code), None, lines, filename)

    @staticmethod
    def _format_timeout_seconds(seconds: float) -> str:
        if float(seconds).is_integer():
            return str(int(seconds))
        return f"{seconds:g}"

    def _build_error_details(
        self,
        exc: Exception,
        code: str,
        filename: str,
    ) -> dict[str, Any]:
        code_lines = code.splitlines()
        error_type = type(exc).__name__

        if isinstance(exc, SyntaxError):
            raw_message = exc.msg or str(exc)
            line_no = int(exc.lineno) if exc.lineno else None
            column_no = int(exc.offset) if exc.offset else None
        else:
            raw_message = str(exc)
            line_no = None
            column_no = None
            stack = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
            for frame in reversed(stack):
                if frame.filename == filename:
                    line_no = frame.lineno
                    break

        snippet: Optional[str] = None
        if line_no is not None and 1 <= line_no <= len(code_lines):
            snippet = code_lines[line_no - 1]
        elif isinstance(exc, SyntaxError) and isinstance(exc.text, str):
            snippet = exc.text.rstrip("\n")

        pointer: Optional[str] = None
        if snippet is not None and column_no is not None and column_no > 0:
            pointer = " " * (column_no - 1) + "^"

        if isinstance(exc, SyntaxError):
            user_traceback = ""
        else:
            user_frames = []
            stack = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
            for frame in stack:
                if frame.filename == filename:
                    user_frames.append(frame)

            if not user_frames and stack:
                user_frames.append(stack[-1])

            user_traceback_lines = traceback.format_list(user_frames)
            exception_line = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            user_traceback = "".join(user_traceback_lines).rstrip()
            if exception_line:
                if user_traceback:
                    user_traceback += "\n"
                user_traceback += exception_line

        full_traceback = ""

        summary_lines = [f"{error_type}: {raw_message}"]
        if line_no is not None:
            location = f"{filename}:{line_no}"
            if column_no is not None and column_no > 0:
                location += f":{column_no}"
            summary_lines.append(f"at {location}")
        if snippet is not None:
            summary_lines.append(snippet)
        if pointer is not None:
            summary_lines.append(pointer)

        hint: Optional[str] = None
        if isinstance(exc, (ModuleNotFoundError, ImportError)):
            normalized_message = raw_message.lower()
            if "runtime" in normalized_message or "import runtime" in code:
                hint = (
                    "Hint: runtime is an injected global and cannot be imported. "
                    "Call primitives as runtime.<namespace>.<name>(...) instead."
                )
        if hint:
            summary_lines.append(hint)

        return {
            "error_type": error_type,
            "message": raw_message,
            "filename": filename,
            "line": line_no,
            "column": column_no,
            "snippet": snippet,
            "pointer": pointer,
            "summary": "\n".join(summary_lines),
            "hint": hint,
            "user_traceback": user_traceback,
            "full_traceback": full_traceback,
        }

    @staticmethod
    def _make_remote_error(error_type: str, message: str) -> Exception:
        mapping = {
            "KeyError": KeyError,
            "ValueError": ValueError,
            "IndexError": IndexError,
            "TypeError": TypeError,
            "RuntimeError": RuntimeError,
        }
        exc_type = mapping.get(error_type, RuntimeError)
        return exc_type(message)

    def call_primitive(
        self,
        name: str,
        args: list[Any],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        call_id = uuid.uuid4().hex
        payload_kwargs = kwargs if isinstance(kwargs, dict) else {}
        self._emit(
            EVENT_PRIMITIVE_CALL,
            exec_id=self._active_exec_id,
            call_id=call_id,
            name=name,
            args=args,
            kwargs=payload_kwargs,
        )

        command = self._wait_for_command(
            predicate=lambda item: (
                item.get("type") == COMMAND_PRIMITIVE_RESULT
                and item.get("call_id") == call_id
            ),
            timeout=None,
        )

        if command is None:
            raise _WorkerControlInterruptedError("primitive response was interrupted")

        ok = bool(command.get("ok"))
        if ok:
            return command.get("result")

        error_type = str(command.get("error_type", "RuntimeError"))
        error_message = str(command.get("error_message", "primitive call failed"))
        raise self._make_remote_error(error_type, error_message)

    @staticmethod
    def _image_suffix_for_mime_type(mime_type: str) -> str:
        return {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/webp": ".webp",
        }.get(mime_type, ".png")

    @staticmethod
    def _image_mime_type_for_repr_method(method_name: str) -> str:
        return {
            "_repr_png_": "image/png",
            "_repr_jpeg_": "image/jpeg",
            "_repr_svg_": "image/svg+xml",
        }.get(method_name, "image/png")

    @staticmethod
    def _coerce_image_data_to_bytes(data: Any) -> Optional[bytes]:
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, str):
            try:
                return base64.b64decode(data, validate=True)
            except Exception:
                return data.encode("utf-8")
        return None

    def _write_image_artifact(
        self,
        *,
        data: Any,
        mime_type: str,
        source: str,
    ) -> Optional[dict[str, Any]]:
        if mime_type == "image/svg+xml":
            return None

        image_bytes = self._coerce_image_data_to_bytes(data)
        if image_bytes is None:
            return None

        suffix = self._image_suffix_for_mime_type(mime_type)
        fd, path = tempfile.mkstemp(prefix="pyrepl_image_", suffix=suffix)
        with os.fdopen(fd, "wb") as handle:
            handle.write(image_bytes)

        return {
            "type": "image",
            "source": source,
            "path": path,
            "mime_type": mime_type,
            "detail": "auto",
        }

    def _artifact_from_mime_bundle(
        self,
        data: Any,
        *,
        source: str,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(data, dict):
            return None

        for mime_type in ("image/png", "image/jpeg", "image/gif", "image/bmp", "image/webp"):
            if mime_type not in data:
                continue
            return self._write_image_artifact(
                data=data[mime_type],
                mime_type=mime_type,
                source=source,
            )
        return None

    @staticmethod
    def _artifact_from_img_url(value: ImgUrl) -> dict[str, Any]:
        return {
            "type": "image",
            "source": "return_value",
            "url": value.url,
            "detail": value.detail,
        }

    @staticmethod
    def _artifact_from_img_path(value: ImgPath) -> dict[str, Any]:
        path = value.path.expanduser().resolve()
        return {
            "type": "image",
            "source": "return_value",
            "path": str(path),
            "mime_type": value.get_mime_type(),
            "detail": value.detail,
        }

    def _capture_result_artifact(
        self,
        result: Any,
        artifacts: list[dict[str, Any]],
    ) -> bool:
        if isinstance(result, ImgUrl):
            artifacts.append(self._artifact_from_img_url(result))
            return True

        if isinstance(result, ImgPath):
            artifacts.append(self._artifact_from_img_path(result))
            return True

        for method_name in ("_repr_png_", "_repr_jpeg_"):
            repr_method = getattr(result, method_name, None)
            if not callable(repr_method):
                continue
            try:
                image_data = repr_method()
            except Exception:
                continue
            if image_data is None:
                continue
            artifact = self._write_image_artifact(
                data=image_data,
                mime_type=self._image_mime_type_for_repr_method(method_name),
                source="return_value",
            )
            if artifact is not None:
                artifacts.append(artifact)
                return True

        try:
            data, _metadata = self._shell.display_formatter.format(result)
        except Exception:
            data = None
        artifact = self._artifact_from_mime_bundle(data, source="return_value")
        if artifact is not None:
            artifacts.append(artifact)
            return True

        return False

    def _execute_python_code(
        self,
        code: str,
        filename: str,
    ) -> tuple[Optional[str], list[dict[str, Any]]]:
        artifacts: list[dict[str, Any]] = []
        display_pub = self._shell.display_pub
        old_publish = display_pub.publish
        old_display = ipython_display.display
        namespace_display_was_present = "display" in self._namespace
        old_namespace_display = self._namespace.get("display")

        def publish_hook(
            data: Any,
            metadata: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            artifact = self._artifact_from_mime_bundle(data, source="display_data")
            if artifact is not None:
                artifacts.append(artifact)
            return old_publish(data, metadata, *args, **kwargs)

        def display_hook(*objs: Any, **kwargs: Any) -> None:
            passthrough_objs = []
            for obj in objs:
                if self._capture_result_artifact(obj, artifacts):
                    continue
                passthrough_objs.append(obj)
            if not passthrough_objs:
                return None
            return old_display(*passthrough_objs, **kwargs)

        transformed_code = self._shell.transform_cell(code)
        module_ast = ast.parse(transformed_code, filename=filename, mode="exec")
        body_nodes = list(module_ast.body)

        last_expression: Optional[ast.expr] = None
        if body_nodes and isinstance(body_nodes[-1], ast.Expr):
            last_expression = body_nodes.pop().value

        display_pub.publish = publish_hook
        ipython_display.display = display_hook
        if self._namespace.get("display") is old_display:
            self._namespace["display"] = display_hook
        try:
            if body_nodes:
                module_for_exec = ast.Module(body=body_nodes, type_ignores=[])
                ast.fix_missing_locations(module_for_exec)
                exec(
                    compile(module_for_exec, filename, "exec"),
                    self._namespace,
                    self._namespace,
                )

            if last_expression is None:
                return None, artifacts

            expression_for_eval = ast.Expression(body=last_expression)
            ast.fix_missing_locations(expression_for_eval)
            result = eval(
                compile(expression_for_eval, filename, "eval"),
                self._namespace,
                self._namespace,
            )
            if result is None:
                return None, artifacts
            if self._capture_result_artifact(result, artifacts):
                return None, artifacts
            return repr(result), artifacts
        finally:
            display_pub.publish = old_publish
            ipython_display.display = old_display
            if self._namespace.get("display") is display_hook:
                if namespace_display_was_present:
                    self._namespace["display"] = old_namespace_display
                else:
                    self._namespace["display"] = old_display


    def _handle_execute(self, command: dict[str, Any]) -> None:
        exec_id = str(command.get("exec_id", ""))
        code = command.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        try:
            input_idle_timeout = float(command.get("input_idle_timeout_seconds", 300.0))
        except (TypeError, ValueError):
            input_idle_timeout = 300.0

        self._input_idle_timeout_seconds = input_idle_timeout
        self._sync_runtime_binding(
            runtime_enabled=bool(command.get("runtime_enabled", True)),
        )

        filename = self._new_cell_filename()
        self._cache_source(filename, code)

        error_message: Optional[str] = None
        error_details: Optional[dict[str, Any]] = None
        return_value: Optional[str] = None
        artifacts: list[dict[str, Any]] = []

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_input = builtins.input

        self._active_exec_id = exec_id

        def on_stdout(text: str) -> None:
            self._emit(EVENT_STDOUT, exec_id=exec_id, text=text)

        def on_stderr(text: str) -> None:
            self._emit(EVENT_STDERR, exec_id=exec_id, text=text)

        def input_hook(prompt: str = "") -> str:
            request_id = uuid.uuid4().hex
            self._emit(
                EVENT_INPUT_REQUEST,
                exec_id=exec_id,
                request_id=request_id,
                prompt=prompt,
                idle_timeout_seconds=self._input_idle_timeout_seconds,
            )

            deadline = time.monotonic() + self._input_idle_timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _InputRequestTimeoutError(
                        "Input request timed out after "
                        f"{self._format_timeout_seconds(self._input_idle_timeout_seconds)} seconds"
                    )

                command_message = self._wait_for_command(
                    predicate=lambda item: (
                        item.get("type") == COMMAND_INPUT_REPLY
                        and item.get("request_id") == request_id
                    ),
                    timeout=min(0.1, remaining),
                )
                if command_message is None:
                    continue

                value = command_message.get("value", "")
                if not isinstance(value, str):
                    value = str(value)
                self._emit(
                    EVENT_INPUT_ACCEPTED,
                    exec_id=exec_id,
                    request_id=request_id,
                )
                return value

        sys.stdout = _LineCapture(on_stdout)
        sys.stderr = _LineCapture(on_stderr)
        builtins.input = input_hook

        try:
            return_value, artifacts = self._execute_python_code(
                code=code,
                filename=filename,
            )
        except _InputRequestTimeoutError as exc:
            error_message = str(exc)
            on_stderr(error_message + "\n")
            error_details = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "filename": filename,
                "line": None,
                "column": None,
                "snippet": None,
                "pointer": None,
                "summary": str(exc),
                "user_traceback": "",
                "full_traceback": "",
            }
        except KeyboardInterrupt:
            error_message = "Execution interrupted"
            on_stderr(error_message + "\n")
            error_details = {
                "error_type": "KeyboardInterrupt",
                "message": "Execution interrupted",
                "filename": filename,
                "line": None,
                "column": None,
                "snippet": None,
                "pointer": None,
                "summary": error_message,
                "user_traceback": "",
                "full_traceback": "",
            }
        except Exception as exc:  # pragma: no cover - exercised through public API
            error_details = self._build_error_details(
                exc=exc, code=code, filename=filename
            )
            error_message = str(error_details["summary"])
            on_stderr(error_message + "\n")
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass

            sys.stdout = old_stdout
            sys.stderr = old_stderr
            builtins.input = old_input
            self._active_exec_id = None

        self._emit(
            EVENT_EXECUTE_RESULT,
            exec_id=exec_id,
            success=error_message is None,
            return_value=return_value,
            artifacts=artifacts,
            error=error_message,
            error_details=error_details,
        )

    def _handle_reset(self, command: dict[str, Any]) -> None:
        request_id = str(command.get("request_id", ""))
        self._sync_runtime_binding(
            runtime_enabled=bool(command.get("runtime_enabled", True)),
        )

        to_remove = []
        for name in list(self._namespace.keys()):
            if name in self._baseline_names:
                continue
            if self._runtime_enabled and name == RUNTIME_GLOBAL_NAME:
                continue
            to_remove.append(name)

        for name in to_remove:
            self._namespace.pop(name, None)

        if self._runtime_enabled:
            self._namespace[RUNTIME_GLOBAL_NAME] = self._runtime_proxy
        else:
            self._namespace.pop(RUNTIME_GLOBAL_NAME, None)

        self._emit(
            EVENT_RESET_RESULT,
            request_id=request_id,
            message="REPL 已重置，所有变量已清除",
        )


def run_pyrepl_worker(
    command_queue: Any,
    event_queue: Any,
    working_directory: Optional[str] = None,
) -> None:
    """Entrypoint executed inside PyRepl subprocess worker."""

    if working_directory:
        try:
            os.chdir(working_directory)
        except Exception as exc:
            event_queue.put(
                {
                    "type": EVENT_WORKER_ERROR,
                    "message": f"Failed to set working_directory: {exc}",
                }
            )
            return

    worker = _PyReplWorker(command_queue=command_queue, event_queue=event_queue)
    worker.run_forever()


__all__ = [
    "EVENT_STDOUT",
    "EVENT_STDERR",
    "EVENT_INPUT_REQUEST",
    "EVENT_INPUT_ACCEPTED",
    "EVENT_PRIMITIVE_CALL",
    "EVENT_EXECUTE_RESULT",
    "EVENT_RESET_RESULT",
    "EVENT_WORKER_ERROR",
    "EVENT_WORKER_READY",
    "COMMAND_EXECUTE",
    "COMMAND_RESET",
    "COMMAND_INPUT_REPLY",
    "COMMAND_PRIMITIVE_RESULT",
    "COMMAND_SHUTDOWN",
    "RUNTIME_GLOBAL_NAME",
    "run_pyrepl_worker",
]
