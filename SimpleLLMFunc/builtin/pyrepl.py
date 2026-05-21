"""PyRepl builtin tool for SimpleLLMFunc.

轻量级 Python REPL，基于 subprocess + IPython InteractiveShell。
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.runtime import RuntimePrimitiveBackend
from SimpleLLMFunc.runtime.selfref.primitives import (
    DEFAULT_SELF_REFERENCE_BACKEND_NAME as RUNTIME_DEFAULT_SELF_REFERENCE_BACKEND_NAME,
)
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.builtin.pyrepl_audit import PyReplAuditLog
from SimpleLLMFunc.builtin.pyrepl_tools import (
    EXECUTE_TOOL_BEST_PRACTICES,
    EXECUTE_TOOL_DESCRIPTION,
    RESET_TOOL_BEST_PRACTICES,
    RESET_TOOL_DESCRIPTION,
    create_pyrepl_tools,
    execute_tool_adapter,
    format_execute_tool_output,
)
from SimpleLLMFunc.builtin.pyrepl_worker_client import PyReplWorkerClient
from SimpleLLMFunc.builtin.pyrepl_primitive_host import PyReplPrimitiveHostMixin
from SimpleLLMFunc.builtin.pyrepl_input_mixin import PyReplInputMixin
from SimpleLLMFunc.builtin.pyrepl_execution import PyReplExecutionMixin
from SimpleLLMFunc.builtin.pyrepl_worker_mixin import PyReplWorkerMixin


class PyRepl(
    PyReplExecutionMixin,
    PyReplWorkerMixin,
    PyReplInputMixin,
    PyReplPrimitiveHostMixin,
):
    """轻量级 Python REPL

    基于 subprocess + IPython InteractiveShell，支持：
    - 实时 stdout/stderr streaming
    - 变量跨调用持久化
    - 独立进程执行，支持更可靠中断

    Usage:
        repl = PyRepl()
        tools = repl.toolset

        @llm_chat(toolkit=tools + [...], ...)
        async def chat(message: str, history=None):
            '''Python 编程助手'''
    """

    DEFAULT_EXECUTION_TIMEOUT_SECONDS = 600.0
    DEFAULT_INPUT_IDLE_TIMEOUT_SECONDS = 300.0
    INTERRUPT_GRACE_SECONDS = 1.0

    EXECUTE_TOOL_DESCRIPTION = EXECUTE_TOOL_DESCRIPTION
    RESET_TOOL_DESCRIPTION = RESET_TOOL_DESCRIPTION
    EXECUTE_TOOL_BEST_PRACTICES = EXECUTE_TOOL_BEST_PRACTICES
    RESET_TOOL_BEST_PRACTICES = RESET_TOOL_BEST_PRACTICES

    DEFAULT_SELF_REFERENCE_BACKEND_NAME = RUNTIME_DEFAULT_SELF_REFERENCE_BACKEND_NAME

    def __init__(
        self,
        execution_timeout_seconds: float = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        input_idle_timeout_seconds: float = DEFAULT_INPUT_IDLE_TIMEOUT_SECONDS,
        working_directory: Optional[Union[str, Path]] = None,
        _install_builtin_packs: bool = True,
    ):
        execution_timeout = float(execution_timeout_seconds)
        if execution_timeout <= 0:
            raise ValueError("execution_timeout_seconds must be greater than 0")

        input_idle_timeout = float(input_idle_timeout_seconds)
        if input_idle_timeout <= 0:
            raise ValueError("input_idle_timeout_seconds must be greater than 0")

        resolved_working_directory: Optional[Path] = None
        if working_directory is not None:
            if not isinstance(working_directory, (str, Path)):
                raise ValueError("working_directory must be a path string or Path")
            resolved = Path(working_directory).expanduser().resolve()
            if not resolved.exists() or not resolved.is_dir():
                raise ValueError("working_directory must be an existing directory")
            resolved_working_directory = resolved

        self.execution_timeout_seconds = execution_timeout
        self.input_idle_timeout_seconds = input_idle_timeout
        self._working_directory = resolved_working_directory

        self.namespace: Dict[str, Any] = {}
        self._tools: Optional[List[Tool]] = None
        self._lock = threading.RLock()
        self._operation_lock = asyncio.Lock()

        self._worker_client = PyReplWorkerClient(self._working_directory)
        self._ctx = self._worker_client._ctx
        self._command_queue: Any = None
        self._event_queue: Any = None
        self._process: Any = None
        self._prefetched_events: List[dict[str, Any]] = self._worker_client.prefetched_events
        self._closed = False

        self._init_primitive_host(_install_builtin_packs=_install_builtin_packs)

        self._instance_id = uuid.uuid4().hex
        self._audit_log = PyReplAuditLog(self._instance_id)

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def audit_log_dir(self) -> str:
        return self._audit_log.log_dir

    @property
    def audit_log_file(self) -> str:
        return self._audit_log.log_file

    @property
    def working_directory(self) -> Optional[str]:
        if self._working_directory is None:
            return None
        return str(self._working_directory)

    @property
    def toolset(self) -> List[Tool]:
        """返回绑定到该 repl 实例的 tool 列表"""
        if self._tools is None:
            self._tools = self._create_tools()
        return self._tools

    @staticmethod
    def _format_execute_tool_output(result: Dict[str, Any]) -> str:
        return format_execute_tool_output(result)

    async def _execute_tool(
        self,
        code: str,
        timeout_seconds: Optional[float] = None,
        event_emitter: Optional[ToolEventEmitter] = None,
    ) -> str:
        return await execute_tool_adapter(
            self,
            code,
            timeout_seconds=timeout_seconds,
            event_emitter=event_emitter,
        )

    def _create_tools(self) -> List[Tool]:
        return create_pyrepl_tools(self)

    def _append_audit_entry(self, payload: dict[str, Any]) -> None:
        self._audit_log.append(payload)

    def close(self) -> None:
        """Close worker process and release resources."""

        with self._lock:
            if self._closed:
                return
            installed_packs = list(self._installed_packs.values())
            self._shutdown_worker_locked()
            self._closed = True

        for pack in installed_packs:
            backend = pack.backend
            if isinstance(backend, RuntimePrimitiveBackend):
                backend.on_close(self)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["PyRepl"]
