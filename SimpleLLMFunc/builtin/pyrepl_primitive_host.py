from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.runtime import (
    ForkContext,
    PrimitiveCallContext,
    PrimitivePack,
    PrimitiveRegistry,
    RuntimePrimitiveBackend,
)
from SimpleLLMFunc.runtime.primitives import primitive
from SimpleLLMFunc.runtime.selfref import SelfReference
from SimpleLLMFunc.runtime.selfref.primitives import build_self_reference_pack

from .pyrepl_worker import COMMAND_PRIMITIVE_RESULT


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


class PyReplPrimitiveHostMixin:
    """Runtime backend and primitive registry behavior for PyRepl."""

    def _init_primitive_host(self, *, _install_builtin_packs: bool) -> None:
        self._runtime_backends: Dict[str, Any] = {}
        self._primitive_pack_installers: Dict[str, Any] = {}
        self._installed_packs: Dict[str, PrimitivePack] = {}
        self._legacy_primitive_registrations: Dict[
            str, _LegacyPrimitiveRegistration
        ] = {}
        self._primitive_registry = PrimitiveRegistry()
        self._register_builtin_primitives()
        if _install_builtin_packs:
            self._install_builtin_packs()

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

        cloned_repl = type(self)(
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




__all__ = ["PyReplPrimitiveHostMixin"]
