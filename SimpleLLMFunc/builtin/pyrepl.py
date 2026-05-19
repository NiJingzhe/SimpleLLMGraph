"""PyRepl builtin tool for SimpleLLMFunc.

轻量级 Python REPL，基于 subprocess + IPython InteractiveShell。
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.logger.logger_config import logger_config
from SimpleLLMFunc.runtime import (
    ForkContext,
    PrimitiveCallContext,
    PrimitivePack,
    PrimitiveRegistry,
    RuntimePrimitiveBackend,
)
from SimpleLLMFunc.runtime.primitives import primitive
from SimpleLLMFunc.runtime.selfref.primitives import (
    DEFAULT_SELF_REFERENCE_BACKEND_NAME as RUNTIME_DEFAULT_SELF_REFERENCE_BACKEND_NAME,
    build_self_reference_pack,
)
from SimpleLLMFunc.tool import TOO_LONG_TO_FILE_MAX_TOKENS, Tool

from SimpleLLMFunc.runtime.selfref import SelfReference

from .pyrepl_worker import (
    COMMAND_EXECUTE,
    COMMAND_INPUT_REPLY,
    COMMAND_PRIMITIVE_RESULT,
    COMMAND_RESET,
    COMMAND_SHUTDOWN,
    EVENT_EXECUTE_RESULT,
    EVENT_INPUT_ACCEPTED,
    EVENT_INPUT_REQUEST,
    EVENT_PRIMITIVE_CALL,
    EVENT_RESET_RESULT,
    EVENT_STDERR,
    EVENT_STDOUT,
    EVENT_WORKER_READY,
    EVENT_WORKER_ERROR,
    run_pyrepl_worker,
)


@dataclass(frozen=True)
class _LegacyPrimitiveRegistration:
    """Replayable low-level primitive registration record."""

    name: str
    handler: Any
    contract: Optional[Any] = None
    description: str = ""
    input_type: str = ""
    output_type: str = ""
    output_parsing: str = ""
    parameters: Optional[List[Dict[str, Any]]] = None
    next_steps: Optional[Any] = None
    backend_name: Optional[str] = None


class PyRepl:
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

    _input_registry_lock = threading.Lock()
    _pending_input_queues: Dict[str, queue.Queue[str]] = {}

    DEFAULT_EXECUTION_TIMEOUT_SECONDS = 600.0
    DEFAULT_INPUT_IDLE_TIMEOUT_SECONDS = 300.0
    INTERRUPT_GRACE_SECONDS = 1.0

    EXECUTE_TOOL_DESCRIPTION = (
        "Run Python code in a persistent REPL session (state persists across "
        "calls). Write direct executable snippets for the active REPL session. "
        "Use top-level executable code. Interactive "
        "`input()` is supported. Use `timeout_seconds` to control per-call "
        "timeout (default 600)."
        f"By the way, if a code snippet produces output more than {TOO_LONG_TO_FILE_MAX_TOKENS} tokens, "
        f"we truncate the tool result to no more than {TOO_LONG_TO_FILE_MAX_TOKENS} tokens, "
        "store the full result in a temporary file, and tell you its path."
    )
    RESET_TOOL_DESCRIPTION = (
        "Reset REPL runtime variables in the current session while preserving "
        "registered runtime primitive backends."
    )
    EXECUTE_TOOL_BEST_PRACTICES = [
        "Primitive = host-registered callable; use runtime.namespace.name(...). Use contains='<namespace>.' for namespace discovery.",
        "Runtime primitives are not standalone tool calls; call them inside execute_code as runtime.namespace.name(...).",
        "Spec lookups return XML by default; use format='dict' for direct field access in code.",
        "Inspect the contracts that support the current step and keep prompt context focused on the selected primitives.",
    ]
    RESET_TOOL_BEST_PRACTICES = [
        "Use reset_repl for REPL variable cleanup while continuing with the same runtime backend state.",
        "Continue the next execution step from a fresh REPL variable namespace after reset.",
    ]

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
        self._runtime_backends: Dict[str, Any] = {}
        self._primitive_pack_installers: Dict[str, Any] = {}
        self._installed_packs: Dict[str, PrimitivePack] = {}
        self._legacy_primitive_registrations: Dict[
            str, _LegacyPrimitiveRegistration
        ] = {}
        self._tools: Optional[List[Tool]] = None
        self._lock = threading.RLock()
        self._operation_lock = asyncio.Lock()

        self._ctx = mp.get_context("spawn")
        self._command_queue: Any = None
        self._event_queue: Any = None
        self._process: Any = None
        self._prefetched_events: List[dict[str, Any]] = []
        self._closed = False

        self._primitive_registry = PrimitiveRegistry()
        self._register_builtin_primitives()
        if _install_builtin_packs:
            self._install_builtin_packs()

        self._instance_id = uuid.uuid4().hex
        self._audit_lock = threading.Lock()
        self._audit_dir = Path(logger_config.LOG_DIR) / "pyrepl" / self._instance_id
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._audit_file = self._audit_dir / "executions.jsonl"

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def audit_log_dir(self) -> str:
        return str(self._audit_dir)

    @property
    def audit_log_file(self) -> str:
        return str(self._audit_file)

    @property
    def working_directory(self) -> Optional[str]:
        if self._working_directory is None:
            return None
        return str(self._working_directory)

    @staticmethod
    def _normalize_backend_name(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("backend name must be a non-empty string")

        normalized = name.strip()
        if not normalized:
            raise ValueError("backend name must be a non-empty string")

        if "." in normalized:
            raise ValueError("backend name must be a single segment")

        return normalized

    def register_runtime_backend(
        self,
        name: str,
        backend: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register one runtime backend object by name."""

        normalized = self._normalize_backend_name(name)
        if backend is None:
            raise ValueError("backend must not be None")

        with self._lock:
            if normalized in self._runtime_backends and not replace:
                raise ValueError(
                    f"runtime backend '{normalized}' is already registered"
                )
            self._runtime_backends[normalized] = backend

    def unregister_runtime_backend(self, name: str) -> None:
        """Unregister one runtime backend by name if it exists."""

        normalized = self._normalize_backend_name(name)
        with self._lock:
            self._runtime_backends.pop(normalized, None)

    def get_runtime_backend(self, name: str) -> Optional[Any]:
        """Get one runtime backend by name."""

        normalized = self._normalize_backend_name(name)
        with self._lock:
            return self._runtime_backends.get(normalized)

    def list_runtime_backends(self) -> List[str]:
        """List registered runtime backend names."""

        with self._lock:
            names = list(self._runtime_backends.keys())
        names.sort()
        return names

    def register_primitive_pack_installer(
        self,
        pack_name: str,
        installer: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register one primitive-pack installer callable."""

        normalized = self._normalize_backend_name(pack_name)
        if not callable(installer):
            raise ValueError("primitive pack installer must be callable")

        with self._lock:
            if normalized in self._primitive_pack_installers and not replace:
                raise ValueError(
                    f"primitive pack installer '{normalized}' is already registered"
                )
            self._primitive_pack_installers[normalized] = installer

    def _install_builtin_packs(self) -> None:
        self.install_pack(
            build_self_reference_pack(
                SelfReference(),
                backend_name=self.DEFAULT_SELF_REFERENCE_BACKEND_NAME,
            )
        )

    def install_primitive_pack(self, pack_name: str, **options: Any) -> None:
        """Install one registered primitive pack into this REPL."""

        if isinstance(pack_name, PrimitivePack):
            replace = bool(options.pop("replace", False))
            if options:
                raise ValueError(
                    "installing a PrimitivePack object only accepts replace=..."
                )
            self.install_pack(pack_name, replace=replace)
            return

        normalized = self._normalize_backend_name(pack_name)
        with self._lock:
            installer = self._primitive_pack_installers.get(normalized)

        if installer is None:
            raise KeyError(f"primitive pack '{normalized}' is not registered")

        installer(**options)

    def install_pack(self, pack: PrimitivePack, *, replace: bool = False) -> None:
        """Install a first-class PrimitivePack into this REPL."""

        if not isinstance(pack, PrimitivePack):
            raise ValueError("pack must be a PrimitivePack instance")
        if pack.backend is None:
            raise ValueError("pack backend must not be None")

        normalized_pack_name = self._normalize_backend_name(pack.name)
        normalized_backend_name = self._normalize_backend_name(pack.backend_name)
        snapshot = pack.clone(backend_name=normalized_backend_name)
        previous_snapshot: Optional[PrimitivePack] = None

        with self._lock:
            if normalized_pack_name in self._installed_packs and not replace:
                raise ValueError(
                    f"primitive pack '{normalized_pack_name}' is already installed"
                )
            previous_snapshot = self._installed_packs.get(normalized_pack_name)

        previous_backend = previous_snapshot.backend if previous_snapshot else None
        previous_primitive_names = (
            {entry.name for entry in previous_snapshot.primitives}
            if previous_snapshot is not None
            else set()
        )
        new_primitive_names = {entry.name for entry in snapshot.primitives}

        self.register_runtime_backend(
            normalized_backend_name,
            snapshot.backend,
            replace=replace,
        )

        for entry in snapshot.primitives:
            contract = entry.contract
            self._primitive_registry.register(
                entry.name,
                entry.handler,
                contract=contract,
                backend_name=normalized_backend_name,
                replace=replace,
            )

        stale_primitive_names = previous_primitive_names - new_primitive_names
        for primitive_name in stale_primitive_names:
            self._primitive_registry.unregister(primitive_name)

        with self._lock:
            self._installed_packs[normalized_pack_name] = snapshot

        if (
            isinstance(previous_backend, RuntimePrimitiveBackend)
            and previous_backend is not snapshot.backend
        ):
            previous_backend.on_close(self)

        if (
            isinstance(snapshot.backend, RuntimePrimitiveBackend)
            and previous_backend is not snapshot.backend
        ):
            snapshot.backend.on_install(self)

    def pack(
        self,
        name: str,
        *,
        backend: Any,
        backend_name: Optional[str] = None,
        guidance: str = "",
    ) -> PrimitivePack:
        """Create a declarative PrimitivePack bound to this REPL host."""

        return PrimitivePack(
            name,
            backend=backend,
            backend_name=backend_name,
            guidance=guidance,
        )

    def _register_builtin_primitives(self) -> None:
        @primitive()
        def runtime_list_primitive_specs(
            _ctx: Any,
            *,
            names: Optional[List[str]] = None,
            prefix: Optional[str] = None,
            contains: Optional[str] = None,
            format: str = "xml",
        ) -> Union[List[Dict[str, Any]], str]:
            """
            Use: Read structured specs for runtime primitives (host-registered callables, no import needed).
            Input: Keyword-only filters. `names` exact names. `contains` substring filter (for example `contains='<namespace>.'`). `format` defaults to `xml`.
            Output: XML when format='xml', or list[dict] when format='dict'.
            Parse: XML: parse <primitive_specs>/<primitive>. Dict: iterate the list.
            Parameters:
            - names: Exact primitive names.
            - prefix: Names starting with prefix.
            - contains: Names containing substring (for example `contains='<namespace>.'`).
            - format: xml (default) or dict.
            Best Practices:
            - Use contains='<namespace>.' for namespace filtering. Use names=[...] for exact set.
            - Specs return XML by default; use format='dict' for direct field access in code.
            """
            return self.list_primitive_specs(
                names=names,
                prefix=prefix,
                contains=contains,
                format=format,
            )

        @primitive(
            next_steps=(
                "Use runtime.get_primitive_spec(name) for one contract, "
                "or runtime.list_primitive_specs(names=[...], contains='...') for batches."
            )
        )
        def runtime_list_primitives(
            _ctx: Any,
            *,
            prefix: Optional[str] = None,
            contains: Optional[str] = None,
        ) -> List[str]:
            """
            Use: Discover runtime primitive names available as `runtime.namespace.name(...)`. Filter by namespace with contains='<namespace>.'.
            Input: Keyword-only. `contains` substring filter (preferred). `prefix` names starting with.
            Output: list[str] of primitive names.
            Parse: Iterate list; call runtime.get_primitive_spec(name) for contracts.
            Parameters:
            - prefix: Names starting with prefix.
            - contains: Names containing substring (for example `contains='<namespace>.'`).
            Best Practices:
            - Filter by namespace with contains='<namespace>.'.
            - After discovery, call runtime.get_primitive_spec(name) or runtime.list_primitive_specs(contains='...').
            """
            return self._primitive_registry.list_names(prefix=prefix, contains=contains)

        @primitive()
        def runtime_get_primitive_spec(
            _ctx: Any,
            name: str,
            *,
            format: str = "xml",
        ) -> Union[Dict[str, Any], str]:
            """
            Use: Read one runtime primitive contract (input/output shape, parameters). Primitive = callable as `runtime.namespace.name(...)`.
            Input: name (full name e.g. selfref.fork.gather_all), format (xml default, dict for field access).
            Output: XML or dict with description, parameters, output_type, output_parsing.
            Parse: XML parse <primitive_spec>. Dict read description, parameters, output_type, output_parsing.
            Parameters:
            - name: Full primitive name.
            - format: xml (default) or dict.
            Best Practices:
            - Default contract lookup for one primitive. Resolve names first via runtime.list_primitives(contains='...').
            - Spec returns XML by default; use format='dict' for direct field access.
            """
            return self.get_primitive_spec(name, format=format)

        @primitive()
        def runtime_list_backends(_ctx: Any) -> List[str]:
            """
            Use: List installed runtime backend packs.
            Input: No arguments.
            Output: `list[str]`. Each item is one backend name such as `selfref`.
            Parse: Treat the result as a plain string list. Check membership before calling backend-specific primitives.
            Best Practices:
            - Check backend availability before using backend-dependent primitives.
            """
            return self.list_runtime_backends()

        self._primitive_registry.register(
            "runtime.list_primitives",
            runtime_list_primitives,
        )

        self._primitive_registry.register(
            "runtime.list_primitive_specs",
            runtime_list_primitive_specs,
            description=(
                "Read structured specs for runtime primitives (host-registered "
                "callables, no import needed)."
            ),
        )

        self._primitive_registry.register(
            "runtime.get_primitive_spec",
            runtime_get_primitive_spec,
        )

        self._primitive_registry.register(
            "runtime.list_backends",
            runtime_list_backends,
        )

    def register_primitive(
        self,
        name: str,
        handler: Any,
        *,
        contract: Optional[Any] = None,
        description: str = "",
        input_type: str = "",
        output_type: str = "",
        output_parsing: str = "",
        parameters: Optional[List[Dict[str, Any]]] = None,
        next_steps: Optional[Any] = None,
        backend_name: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        """Register one host primitive for worker-side runtime calls."""

        self._primitive_registry.register(
            name,
            handler,
            contract=contract,
            description=description,
            input_type=input_type,
            output_type=output_type,
            output_parsing=output_parsing,
            parameters=parameters,
            next_steps=next_steps,
            backend_name=backend_name,
            replace=replace,
        )

        normalized_name = str(name).strip()
        self._legacy_primitive_registrations[normalized_name] = (
            _LegacyPrimitiveRegistration(
                name=normalized_name,
                handler=handler,
                contract=contract,
                description=description,
                input_type=input_type,
                output_type=output_type,
                output_parsing=output_parsing,
                parameters=list(parameters) if parameters is not None else None,
                next_steps=next_steps,
                backend_name=backend_name,
            )
        )

    def unregister_primitive(self, name: str) -> None:
        """Unregister one host primitive by name."""

        self._primitive_registry.unregister(name)
        self._legacy_primitive_registrations.pop(str(name).strip(), None)

    def primitive(
        self,
        name: str,
        *,
        contract: Optional[Any] = None,
        description: str = "",
        input_type: str = "",
        output_type: str = "",
        output_parsing: str = "",
        parameters: Optional[List[Dict[str, Any]]] = None,
        next_steps: Optional[Any] = None,
        backend: Optional[str] = None,
        replace: bool = False,
    ):
        """Decorator sugar for registering one backend-aware primitive."""

        def decorator(handler: Any) -> Any:
            self.register_primitive(
                name,
                handler,
                contract=contract,
                description=description,
                input_type=input_type,
                output_type=output_type,
                output_parsing=output_parsing,
                parameters=parameters,
                next_steps=next_steps,
                backend_name=backend,
                replace=replace,
            )
            return handler

        return decorator

    def list_primitives(self) -> List[str]:
        """List currently registered runtime primitive names."""

        return self._primitive_registry.list_names()

    def list_primitive_contracts(
        self,
        names: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List structured primitive contracts as dict payloads."""

        return self._primitive_registry.list_spec_payloads(
            names=names,
            prefix=prefix,
            contains=contains,
        )

    def get_primitive_contract(self, name: str) -> Dict[str, Any]:
        """Get one structured primitive contract as a dict payload."""

        return self._primitive_registry.get_spec_payload(name)

    def list_installed_packs(self) -> List[str]:
        """List first-class PrimitivePack names installed on this REPL."""

        with self._lock:
            names = list(self._installed_packs.keys())
        names.sort()
        return names

    def _clone_for_fork(
        self,
        *,
        backend_overrides: Optional[Dict[str, Any]] = None,
    ) -> "PyRepl":
        """Clone runtime primitive/backends configuration for forked agents."""

        cloned_repl = PyRepl(
            execution_timeout_seconds=self.execution_timeout_seconds,
            input_idle_timeout_seconds=self.input_idle_timeout_seconds,
            working_directory=self._working_directory,
            _install_builtin_packs=False,
        )

        normalized_overrides: Dict[str, Any] = {}
        if backend_overrides:
            for name, value in backend_overrides.items():
                normalized_overrides[self._normalize_backend_name(name)] = value

        with self._lock:
            installed_packs = [pack.clone() for pack in self._installed_packs.values()]
            runtime_backends = dict(self._runtime_backends)
            legacy_primitives = list(self._legacy_primitive_registrations.values())

        installed_backend_names: set[str] = set()

        for pack in installed_packs:
            override = normalized_overrides.get(pack.backend_name)
            fork_context = ForkContext(
                parent_pack_name=pack.name,
                backend_name=pack.backend_name,
            )
            if override is not None:
                cloned_pack = pack.clone(
                    backend=override,
                    backend_name=pack.backend_name,
                )
            else:
                cloned_pack = pack.clone(
                    backend_name=pack.backend_name,
                    fork_context=fork_context,
                )
            cloned_repl.install_pack(
                cloned_pack,
                replace=True,
            )
            installed_backend_names.add(pack.backend_name)

        for backend_name, backend_value in runtime_backends.items():
            if backend_name in installed_backend_names:
                continue

            override = normalized_overrides.get(backend_name)
            resolved_backend = override if override is not None else backend_value
            if isinstance(resolved_backend, RuntimePrimitiveBackend):
                resolved_backend = resolved_backend.clone_for_fork(
                    context=ForkContext(
                        parent_pack_name=backend_name,
                        backend_name=backend_name,
                    )
                )

            cloned_repl.register_runtime_backend(
                backend_name,
                resolved_backend,
                replace=True,
            )

        for record in legacy_primitives:
            cloned_repl.register_primitive(
                record.name,
                record.handler,
                contract=record.contract,
                description=record.description,
                input_type=record.input_type,
                output_type=record.output_type,
                output_parsing=record.output_parsing,
                parameters=record.parameters,
                next_steps=record.next_steps,
                backend_name=record.backend_name,
                replace=True,
            )

        return cloned_repl

    def list_primitive_specs(
        self,
        names: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        contains: Optional[str] = None,
        format: str = "xml",
    ) -> Union[List[Dict[str, Any]], str]:
        """List primitive contracts as dict payloads or XML string."""

        normalized_format = self._normalize_spec_output_format(format)
        if normalized_format == "xml":
            return self._primitive_registry.list_spec_xml(
                names=names,
                prefix=prefix,
                contains=contains,
            )

        return self._primitive_registry.list_spec_payloads(
            names=names,
            prefix=prefix,
            contains=contains,
        )

    def get_primitive_spec(
        self,
        name: str,
        *,
        format: str = "xml",
    ) -> Union[Dict[str, Any], str]:
        """Get one primitive contract as dict payload or XML string."""

        normalized_format = self._normalize_spec_output_format(format)
        if normalized_format == "xml":
            return self._primitive_registry.get_spec_xml(name)

        return self._primitive_registry.get_spec_payload(name)

    @staticmethod
    def _normalize_spec_output_format(format: str) -> str:
        if not isinstance(format, str):
            raise ValueError("format must be 'dict' or 'xml'")

        normalized = format.strip().lower()
        if normalized in {"", "dict", "json"}:
            return "dict"
        if normalized == "xml":
            return "xml"

        raise ValueError("format must be 'dict' or 'xml'")

    def _build_execute_tool_prompt_injection(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        primitive_names = self.list_primitives()
        with self._lock:
            installed_packs = list(self._installed_packs.values())
        has_context_primitives = any(
            name.startswith("selfref.context.") for name in primitive_names
        )
        has_fork_primitives = any(
            name.startswith("selfref.fork.") for name in primitive_names
        )

        lines: List[str] = [
            "<runtime_primitive_contract>",
            "Runtime primitive = host-registered callable; call it as runtime.namespace.name(...).",
            "Runtime primitives are not standalone tool calls.",
            'Call them inside execute_code as runtime.namespace.name(...). For example: execute_code(code="runtime.selfref.fork.spawn(...)").',
            "Use this block for orientation; use runtime APIs as the source of truth.",
            "Discover names with runtime.list_primitives() and use runtime.list_primitives(contains='<namespace>.') for namespace filtering.",
            "Inspect one primitive: runtime.get_primitive_spec(name). XML by default.",
            "Inspect multiple primitives: runtime.list_primitive_specs(names=[...]) or runtime.list_primitive_specs(contains='...').",
            "Inspect the contracts for the current step and keep prompt context focused on the selected primitives.",
        ]

        pack_guidance_lines: List[str] = []
        seen_pack_names: set[str] = set()
        for pack in sorted(installed_packs, key=lambda item: item.name):
            pack_name = str(pack.name).strip()
            guidance = str(getattr(pack, "guidance", "")).strip()
            if not pack_name or pack_name in seen_pack_names or not guidance:
                continue
            seen_pack_names.add(pack_name)
            pack_guidance_lines.append(f"- {pack_name}: {guidance}")

        if pack_guidance_lines:
            lines.append("Installed primitive packs:")
            lines.extend(pack_guidance_lines)

        if has_fork_primitives:
            lines.extend(
                [
                    "Fork result safety: selfref.fork.gather_all returns compact results by default.",
                    "Read the fields that support the current step, such as status, response, memory_key, and history_count.",
                    "If status == 'error', inspect error_type and error_message before retrying.",
                    "Treat runtime.selfref.fork.gather_all results as dict[fork_id -> ForkResult] and iterate with .items() or .values().",
                    "Summarize the selected result fields in chat responses.",
                    "Use include_history=True when full child history is required.",
                ]
            )

        resolved_memory_key: Optional[str] = None
        if isinstance(context, dict):
            raw_key = context.get("self_reference_key")
            if isinstance(raw_key, str):
                normalized_key = raw_key.strip()
                if normalized_key:
                    resolved_memory_key = normalized_key

        if resolved_memory_key is not None:
            lines.append(f"Active selfref key: {resolved_memory_key}")

        if self._working_directory is not None:
            lines.extend(
                [
                    f"Working directory: {self._working_directory.as_posix()}",
                    "All relative paths resolve from this directory.",
                ]
            )

        lines.extend(
            [
                "Use reset_repl to clear REPL variables while continuing with the current runtime backend state.",
            ]
        )

        if has_context_primitives:
            lines.extend(
                [
                    "Use runtime.selfref.context.inspect() to read the full current context snapshot.",
                    "Use runtime.selfref.context.remember(...) for durable experience, runtime.selfref.context.forget(...) to remove wrong experience, and runtime.selfref.context.compact(...) after a milestone to replace stale working transcript with one structured assistant summary.",
                ]
            )

        lines.append("</runtime_primitive_contract>")
        return "\n".join(lines)

    @classmethod
    def _register_input_queue(cls, request_id: str) -> queue.Queue[str]:
        request_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        with cls._input_registry_lock:
            cls._pending_input_queues[request_id] = request_queue
        return request_queue

    @classmethod
    def _pop_input_queue(cls, request_id: str) -> Optional[queue.Queue[str]]:
        with cls._input_registry_lock:
            return cls._pending_input_queues.pop(request_id, None)

    @classmethod
    def submit_input(cls, request_id: str, value: str) -> bool:
        """Submit a response for a pending ``input()`` request.

        Args:
            request_id: Request ID emitted by ``kernel_input_request`` event.
            value: User-provided input text.

        Returns:
            True if delivered to a live request; False otherwise.
        """

        with cls._input_registry_lock:
            request_queue = cls._pending_input_queues.get(request_id)

        if request_queue is None:
            return False

        try:
            request_queue.put_nowait(value)
            return True
        except queue.Full:
            return False

    @property
    def toolset(self) -> List[Tool]:
        """返回绑定到该 repl 实例的 tool 列表"""
        if self._tools is None:
            self._tools = self._create_tools()
        return self._tools

    @staticmethod
    def _format_execute_tool_output(result: Dict[str, Any]) -> str:
        success = bool(result.get("success"))
        execution_time_ms = result.get("execution_time_ms")
        if isinstance(execution_time_ms, (int, float)):
            duration_text = f"{execution_time_ms:.0f} ms"
        else:
            duration_text = "unknown duration"

        status = "succeeded" if success else "failed"
        lines = [f"Execution {status} in {duration_text}."]

        stdout = result.get("stdout")
        if isinstance(stdout, str) and stdout:
            lines.append("stdout:\n" + stdout)
        else:
            lines.append("stdout: (empty)")

        stderr = result.get("stderr")
        if isinstance(stderr, str) and stderr:
            lines.append("stderr:\n" + stderr)
        else:
            lines.append("stderr: (empty)")

        return_value = result.get("return_value")
        if isinstance(return_value, str) and return_value:
            lines.append(f"return_value: {return_value}")
        else:
            lines.append("return_value: (none)")

        error = result.get("error")
        if isinstance(error, str) and error:
            lines.append(f"error: {error}")

        error_details = result.get("error_details")
        if isinstance(error_details, dict):
            summary = error_details.get("summary")
            if isinstance(summary, str) and summary:
                lines.append(f"error_summary: {summary}")

        return "\n".join(lines)

    async def _execute_tool(
        self,
        code: str,
        timeout_seconds: Optional[float] = None,
        event_emitter: Optional[ToolEventEmitter] = None,
    ) -> str:
        result = await self.execute(
            code,
            timeout_seconds=timeout_seconds,
            event_emitter=event_emitter,
        )
        return self._format_execute_tool_output(result)

    def _create_tools(self) -> List[Tool]:
        tools = []

        execute_tool = Tool(
            name="execute_code",
            description=self.EXECUTE_TOOL_DESCRIPTION,
            func=self._execute_tool,
            best_practices=self.EXECUTE_TOOL_BEST_PRACTICES,
            prompt_injection_builder=self._build_execute_tool_prompt_injection,
            too_long_to_file=True,
        )
        tools.append(execute_tool)

        reset_tool = Tool(
            name="reset_repl",
            description=self.RESET_TOOL_DESCRIPTION,
            func=self.reset,
            best_practices=self.RESET_TOOL_BEST_PRACTICES,
        )
        tools.append(reset_tool)

        return tools

    @staticmethod
    def _format_timeout_seconds(seconds: float) -> str:
        if float(seconds).is_integer():
            return str(int(seconds))
        return f"{seconds:g}"

    @staticmethod
    def _stream_fileno(stream: Any) -> Optional[int]:
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
    def _close_queue_handle(queue_handle: Any) -> None:
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
    def _temporary_valid_stderr(self):
        """Temporarily ensure ``sys.stderr`` has a valid POSIX file descriptor.

        In Textual runtime, ``sys.stderr`` can be a capture proxy with
        ``fileno() == -1``. multiprocessing.resource_tracker forwards
        ``sys.stderr.fileno()`` into ``fds_to_keep`` and crashes when it sees
        invalid values.
        """

        current_stderr = sys.stderr
        current_fd = self._stream_fileno(current_stderr)
        if current_fd is not None and current_fd >= 0:
            yield
            return

        temp_stream = None
        replacement = sys.__stderr__
        replacement_fd = self._stream_fileno(replacement)

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

    def _ensure_worker_locked(self) -> None:
        if self._closed:
            raise RuntimeError("PyRepl is closed")

        if self._process is not None and self._process.is_alive():
            return

        with self._temporary_valid_stderr():
            self._command_queue = self._ctx.Queue()
            self._event_queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=run_pyrepl_worker,
                args=(
                    self._command_queue,
                    self._event_queue,
                    self.working_directory,
                ),
                daemon=True,
            )
            process.start()
        self._process = process

        assert self._event_queue is not None
        startup_deadline = time.monotonic() + 10.0
        while time.monotonic() < startup_deadline:
            if not process.is_alive():
                raise RuntimeError("PyRepl worker exited before startup")

            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            event_type = str(event.get("type", "")) if isinstance(event, dict) else ""
            if event_type == EVENT_WORKER_READY:
                return

            if event_type == EVENT_WORKER_ERROR:
                message = str(event.get("message", "PyRepl worker error"))
                raise RuntimeError(message)

            if isinstance(event, dict):
                self._prefetched_events.append(event)

        raise RuntimeError("Timed out waiting for PyRepl worker startup")

    def _drain_event_queue_locked(self) -> None:
        self._prefetched_events.clear()

        if self._event_queue is None:
            return

        while True:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                return

    def _send_worker_command_locked(self, command: dict[str, Any]) -> None:
        self._ensure_worker_locked()
        assert self._command_queue is not None
        self._command_queue.put(command)

    async def _receive_worker_event(
        self,
        timeout_seconds: float,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            if self._prefetched_events:
                return self._prefetched_events.pop(0)
            event_queue = self._event_queue

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

    def _interrupt_worker_locked(self) -> None:
        process = self._process
        if process is None or not process.is_alive():
            return

        pid = process.pid
        if not pid:
            return

        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def _shutdown_worker_locked(self) -> None:
        process = self._process
        command_queue = self._command_queue
        event_queue = self._event_queue

        if process is not None:
            if process.is_alive():
                try:
                    self._send_worker_command_locked({"type": COMMAND_SHUTDOWN})
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

        self._process = None
        self._command_queue = None
        self._event_queue = None
        self._prefetched_events.clear()

        self._close_queue_handle(command_queue)
        self._close_queue_handle(event_queue)

    def _restart_worker_locked(self) -> None:
        self._shutdown_worker_locked()
        self._ensure_worker_locked()

    async def _execute_primitive_call(
        self,
        message: dict[str, Any],
        event_emitter: Optional[ToolEventEmitter] = None,
    ) -> dict[str, Any]:
        call_id = str(message.get("call_id", ""))
        primitive_name = str(message.get("name", ""))
        execution_id = str(message.get("exec_id", ""))
        args = message.get("args", [])
        kwargs = message.get("kwargs", {})

        if not isinstance(args, list):
            args = []

        if not isinstance(kwargs, dict):
            kwargs = {}

        context = PrimitiveCallContext(
            primitive_name=primitive_name,
            call_id=call_id,
            execution_id=execution_id,
            event_emitter=event_emitter,
            metadata={"pyrepl_instance_id": self._instance_id},
            repl=self,
            registry=self._primitive_registry,
        )

        try:
            result = await self._primitive_registry.call(
                primitive_name,
                args=args,
                kwargs=kwargs,
                context=context,
            )

            return {
                "type": COMMAND_PRIMITIVE_RESULT,
                "call_id": call_id,
                "ok": True,
                "result": result,
            }
        except Exception as exc:
            return {
                "type": COMMAND_PRIMITIVE_RESULT,
                "call_id": call_id,
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

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

    def _append_audit_entry(self, payload: dict[str, Any]) -> None:
        with self._audit_lock:
            with self._audit_file.open("a", encoding="utf-8") as audit_stream:
                json.dump(payload, audit_stream, ensure_ascii=False, default=str)
                audit_stream.write("\n")

    async def execute(
        self,
        code: str,
        timeout_seconds: Optional[float] = None,
        event_emitter: Optional[ToolEventEmitter] = None,
    ) -> Dict[str, Any]:
        """Execute Python snippets in a persistent REPL with streaming output.

        Guidance for LLM tool usage:
        - Write executable snippets directly.
        - Use top-level executable code in the active REPL session.
        - Variables persist across multiple ``execute_code`` calls.
        - Runtime primitives are available via ``runtime.*``. Use
          ``runtime.list_primitives()`` for discovery (or
          ``runtime.list_primitives(contains='...')`` for namespace filtering), then
          ``runtime.get_primitive_spec(name)`` (preferred) or
          ``runtime.list_primitive_specs(names=[...])`` for selected contracts
          (input/output types, parameter docs, and best practices). Spec
          lookups return XML by default; use ``format='dict'`` when code
          needs direct field access.
        - Self-reference primitives are grouped under
          ``runtime.selfref.context.*`` and ``runtime.selfref.fork.*``.
        - ``input()`` is supported. In event mode, callers can reply via
          ``PyRepl.submit_input(request_id, value)``.
        - Use ``reset_repl`` for REPL variable cleanup. Use self-reference
          context methods for runtime-backed context management (for example
          ``runtime.selfref.context.inspect`` /
          ``runtime.selfref.context.remember`` /
          ``runtime.selfref.context.compact``).

        Args:
            code: Python code to execute.
            timeout_seconds: Optional per-call active execution timeout in
                seconds. If omitted, uses the REPL default timeout.
            event_emitter: Optional emitter for real-time stdout/stderr events.

        Returns:
            A dict containing success, stdout, stderr, return_value,
            error, error_details, and execution_time_ms.
        """

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

            try:
                with self._lock:
                    self._ensure_worker_locked()
                    self._drain_event_queue_locked()
                    self._send_worker_command_locked(
                        {
                            "type": COMMAND_EXECUTE,
                            "exec_id": execution_id,
                            "code": code,
                            "input_idle_timeout_seconds": self.input_idle_timeout_seconds,
                            "runtime_enabled": True,
                        }
                    )

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
            with self._lock:
                self.namespace.clear()

                self._ensure_worker_locked()
                self._send_worker_command_locked(
                    {
                        "type": COMMAND_RESET,
                        "request_id": request_id,
                        "runtime_enabled": True,
                    }
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
