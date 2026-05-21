from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, Optional, cast

from SimpleLLMFunc.runtime.selfref.fork_utils import (
    SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM,
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM,
    agent_supports_template_params,
    append_fork_task_user_message,
    build_fork_error_result,
    clone_messages,
    consume_agent_call_output,
    emit_fork_custom_event,
    extract_history_param_name,
    get_agent_fork_toolkit_factory,
    normalize_fork_ids,
    strip_terminal_pending_tool_calls_message,
)


class SelfReferenceInstanceHandle:
    """Container object exposed as ``self_reference.instance``."""

    def __init__(self, owner: Any):
        self._owner = owner

    def is_bound(self) -> bool:
        return self._owner.get_agent_instance() is not None

    def get(self) -> Optional[Any]:
        return self._owner.get_agent_instance()

    async def fork(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
        _event_emitter: Any = None,
        include_history: bool = False,
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        return await self._owner.fork_agent_instance(
            *agent_args,
            source_memory_key=source_memory_key,
            fork_memory_key=fork_memory_key,
            _event_emitter=_event_emitter,
            include_history=include_history,
            **agent_kwargs,
        )

    async def fork_spawn(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
        _event_emitter: Any = None,
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        return await self._owner.spawn_agent_instance(
            *agent_args,
            source_memory_key=source_memory_key,
            fork_memory_key=fork_memory_key,
            _event_emitter=_event_emitter,
            **agent_kwargs,
        )

    async def fork_gather_all(
        self,
        fork_ids: Optional[Any] = None,
        include_history: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        return await self._owner.gather_all_fork_results(
            fork_ids,
            include_history=include_history,
        )

    def has_pending_fork_tasks(self, event_emitter: Any = None) -> bool:
        return self._owner.has_pending_fork_tasks(event_emitter=event_emitter)


class SelfReferenceForkManagerMixin:
    """Fork/spawn/gather orchestration for ``SelfReference``.

    The public ``SelfReference`` object remains the facade.  This mixin owns the
    recursive-agent fork lifecycle that used to live directly in ``state.py``:
    agent binding, fork id/key allocation, child invocation setup, background
    task bookkeeping, event forwarding, and result materialization.
    """

    def bind_agent_instance(
        self,
        agent_instance: Any,
        default_memory_key: Optional[str] = None,
    ) -> None:
        """Bind top-level agent callable for recursive self-fork use-cases."""

        normalized_default_key: Optional[str] = None
        if default_memory_key is not None:
            normalized_default_key = self._normalize_key_for_proxy(default_memory_key)

        with self._lock:
            self._agent_binding.bind(
                agent_instance,
                default_memory_key=normalized_default_key,
            )
            self._agent_instance = self._agent_binding.agent_instance
            self._agent_default_memory_key = self._agent_binding.default_memory_key

    def get_agent_instance(self) -> Optional[Any]:
        with self._lock:
            return self._agent_binding.agent_instance

    def get_agent_default_memory_key(self) -> Optional[str]:
        with self._lock:
            return self._agent_binding.default_memory_key

    def _resolve_source_memory_key_for_fork(
        self,
        source_memory_key: Optional[str],
    ) -> str:
        try:
            return self.resolve_history_key(source_memory_key)
        except ValueError as exc:
            message = str(exc)
            if message == "history key is required because no memory key is available":
                raise ValueError(
                    "source_memory_key is required because no memory key is available"
                ) from exc
            if message == "history key is required when multiple memory keys are bound":
                raise ValueError(
                    "source_memory_key is required when multiple memory keys are bound"
                ) from exc
            raise

    def _build_fork_memory_key(self, source_memory_key: str) -> str:
        with self._lock:
            while True:
                self._fork_counter += 1
                candidate = f"{source_memory_key}::fork::{self._fork_counter}"
                if not self._store.has_history(candidate):
                    return candidate

    def _build_fork_id(self) -> str:
        with self._lock:
            self._fork_id_counter += 1
            return f"fork_{self._fork_id_counter}"

    def _resolve_child_toolkit_override(self, agent_instance: Any) -> Any:
        parent_runtime_toolkit = self._get_active_runtime_toolkit()
        toolkit_factory = get_agent_fork_toolkit_factory(agent_instance)
        if toolkit_factory is None:
            return parent_runtime_toolkit

        try:
            return toolkit_factory(parent_runtime_toolkit)
        except Exception:
            return parent_runtime_toolkit

    def _build_fork_template_params(
        self,
        existing_template_params: Any,
        fork_memory_key: str,
        toolkit_override: Any,
        fork_task_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        active_template_params = self._get_active_template_params()

        if existing_template_params is None:
            merged_template_params = active_template_params or {}
        elif isinstance(existing_template_params, dict):
            merged_template_params = dict(existing_template_params)
            if active_template_params is not None:
                merged_template_params = {
                    **active_template_params,
                    **merged_template_params,
                }
        else:
            raise ValueError("_template_params must be a dict when provided")

        merged_template_params[SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM] = (
            fork_memory_key
        )
        if toolkit_override is not None:
            merged_template_params[SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM] = (
                toolkit_override
            )
        if isinstance(fork_task_message, str) and fork_task_message:
            merged_template_params[SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM] = (
                fork_task_message
            )
        return merged_template_params

    def _history_length(self, key: str) -> int:
        normalized_key = self._normalize_key_for_proxy(key)
        return self._store.get_history_length(normalized_key)

    def _compact_fork_result_payload(self, result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            compact_result = copy.deepcopy(result)
        except Exception:
            compact_result = dict(result)
        compact_result.pop("history", None)

        if "response" not in compact_result and "result" in compact_result:
            compact_result["response"] = compact_result.get("result")
        if "result" not in compact_result and "response" in compact_result:
            compact_result["result"] = compact_result.get("response")

        history_count = compact_result.get("history_count")
        if not isinstance(history_count, int):
            memory_key = compact_result.get("memory_key")
            if isinstance(memory_key, str) and memory_key:
                try:
                    history_count = self._history_length(memory_key)
                except Exception:
                    history_count = 0
            else:
                history_count = 0
            compact_result["history_count"] = history_count

        compact_result["history_included"] = False
        return compact_result

    def _materialize_fork_result_payload(
        self,
        result: Dict[str, Any],
        *,
        include_history: bool,
    ) -> Dict[str, Any]:
        materialized = self._compact_fork_result_payload(result)
        if not include_history:
            return materialized

        memory_key = materialized.get("memory_key")
        history_snapshot = []
        if isinstance(memory_key, str) and memory_key:
            try:
                history_snapshot = self.snapshot_history(memory_key)
            except Exception:
                history_snapshot = []

        materialized["history"] = history_snapshot
        materialized["history_count"] = len(history_snapshot)
        materialized["history_included"] = True
        return materialized

    async def fork_agent_instance(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
        _event_emitter: Any = None,
        include_history: bool = False,
        _fork_id: Optional[str] = None,
        _parent_fork_id: Optional[str] = None,
        _fork_depth: Optional[int] = None,
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        with self._lock:
            agent_instance = self._agent_binding.agent_instance

        if agent_instance is None:
            raise RuntimeError("No agent instance is bound to self_reference")

        source_key = self._resolve_source_memory_key_for_fork(source_memory_key)
        inherited_history = self.snapshot_history(source_key)
        task_message: Optional[str] = None
        if agent_args and isinstance(agent_args[0], str):
            task_message = agent_args[0]
        elif isinstance(agent_kwargs.get("message"), str):
            task_message = cast(str, agent_kwargs.get("message"))

        inherited_history, _ = strip_terminal_pending_tool_calls_message(
            inherited_history
        )

        history_param_name = extract_history_param_name(agent_instance)
        pass_task_via_history = bool(
            task_message
            and history_param_name is not None
            and agent_supports_template_params(agent_instance)
        )
        if pass_task_via_history and task_message is not None:
            inherited_history = append_fork_task_user_message(
                inherited_history,
                task_message=task_message,
            )

        if fork_memory_key is None:
            target_key = self._build_fork_memory_key(source_key)
        else:
            target_key = self._normalize_key_for_proxy(fork_memory_key)

        fork_id = _fork_id if _fork_id is not None else self._build_fork_id()
        parent_fork_id = (
            _parent_fork_id
            if _parent_fork_id is not None
            else self._get_active_fork_id()
        )
        fork_depth = (
            _fork_depth
            if _fork_depth is not None
            else self._get_active_fork_depth() + 1
        )

        await emit_fork_custom_event(
            _event_emitter,
            "selfref_fork_start",
            {
                "fork_id": fork_id,
                "parent_fork_id": parent_fork_id,
                "depth": fork_depth,
                "source_memory_key": source_key,
                "memory_key": target_key,
                "status": "running",
            },
        )

        self.bind_history(target_key, inherited_history)

        call_kwargs = dict(agent_kwargs)
        call_args = list(agent_args)
        if pass_task_via_history and call_args and isinstance(call_args[0], str):
            call_args[0] = ""
        if pass_task_via_history and isinstance(call_kwargs.get("message"), str):
            call_kwargs["message"] = ""
        if history_param_name is not None and history_param_name not in call_kwargs:
            call_kwargs[history_param_name] = clone_messages(inherited_history)

        child_toolkit_override = self._resolve_child_toolkit_override(agent_instance)

        if agent_supports_template_params(agent_instance):
            call_kwargs["_template_params"] = self._build_fork_template_params(
                call_kwargs.get("_template_params"),
                target_key,
                child_toolkit_override,
                task_message,
            )

        active_key_token = self._set_active_memory_key(target_key)
        active_fork_tokens = self._set_active_fork_context(
            fork_id=fork_id,
            depth=fork_depth,
        )
        active_react_state_token = self._active_react_state_var.set(None)
        active_toolkit_token = self._set_active_runtime_toolkit(child_toolkit_override)
        try:
            call_output = agent_instance(*call_args, **call_kwargs)
            response, final_history = await consume_agent_call_output(
                call_output,
                event_emitter=_event_emitter,
                fork_id=fork_id,
                parent_fork_id=parent_fork_id,
                depth=fork_depth,
                source_memory_key=source_key,
                memory_key=target_key,
            )
        except Exception as exc:
            await emit_fork_custom_event(
                _event_emitter,
                "selfref_fork_error",
                {
                    "fork_id": fork_id,
                    "parent_fork_id": parent_fork_id,
                    "depth": fork_depth,
                    "source_memory_key": source_key,
                    "memory_key": target_key,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
        finally:
            self._reset_active_memory_key(active_key_token)
            self._reset_active_fork_context(active_fork_tokens)
            self._active_react_state_var.reset(active_react_state_token)
            self._reset_active_runtime_toolkit(active_toolkit_token)

        if final_history is not None:
            self.bind_history(target_key, final_history)

        completed_result = self._materialize_fork_result_payload(
            {
                "fork_id": fork_id,
                "parent_fork_id": parent_fork_id,
                "depth": fork_depth,
                "source_memory_key": source_key,
                "memory_key": target_key,
                "status": "completed",
                "response": response,
                "result": response,
            },
            include_history=include_history,
        )

        await emit_fork_custom_event(
            _event_emitter,
            "selfref_fork_end",
            {
                "fork_id": fork_id,
                "parent_fork_id": parent_fork_id,
                "depth": fork_depth,
                "source_memory_key": source_key,
                "memory_key": target_key,
                "status": "completed",
            },
        )

        return completed_result

    async def spawn_agent_instance(
        self,
        *agent_args: Any,
        source_memory_key: Optional[str] = None,
        fork_memory_key: Optional[str] = None,
        _event_emitter: Any = None,
        **agent_kwargs: Any,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._agent_binding.agent_instance is None:
                raise RuntimeError("No agent instance is bound to self_reference")

        source_key = self._resolve_source_memory_key_for_fork(source_memory_key)
        if fork_memory_key is None:
            target_key = self._build_fork_memory_key(source_key)
        else:
            target_key = self._normalize_key_for_proxy(fork_memory_key)

        fork_id = self._build_fork_id()
        parent_fork_id = self._get_active_fork_id()
        fork_depth = self._get_active_fork_depth() + 1

        fork_task = asyncio.create_task(
            self.fork_agent_instance(
                *agent_args,
                source_memory_key=source_key,
                fork_memory_key=target_key,
                _event_emitter=_event_emitter,
                _fork_id=fork_id,
                _parent_fork_id=parent_fork_id,
                _fork_depth=fork_depth,
                **agent_kwargs,
            )
        )

        def on_fork_task_done(done_task: asyncio.Task[Dict[str, Any]]) -> None:
            try:
                done_result = done_task.result()
            except BaseException as exc:
                done_result = build_fork_error_result(
                    fork_id=fork_id,
                    source_memory_key=source_key,
                    memory_key=target_key,
                    parent_fork_id=parent_fork_id,
                    depth=fork_depth,
                    error=exc,
                )

            compact_result = self._compact_fork_result_payload(done_result)

            with self._lock:
                self._fork_results[fork_id] = compact_result
                self._fork_tasks.pop(fork_id, None)
                self._fork_emitters.pop(fork_id, None)

        fork_task.add_done_callback(on_fork_task_done)

        with self._lock:
            self._fork_tasks[fork_id] = fork_task
            self._fork_results.pop(fork_id, None)
            self._fork_emitters[fork_id] = _event_emitter

        await emit_fork_custom_event(
            _event_emitter,
            "selfref_fork_spawned",
            {
                "fork_id": fork_id,
                "parent_fork_id": parent_fork_id,
                "depth": fork_depth,
                "source_memory_key": source_key,
                "memory_key": target_key,
                "status": "running",
            },
        )

        return {
            "fork_id": fork_id,
            "parent_fork_id": parent_fork_id,
            "depth": fork_depth,
            "source_memory_key": source_key,
            "memory_key": target_key,
            "status": "running",
        }

    async def wait_fork_result(
        self,
        fork_id: str,
        include_history: bool = False,
    ) -> Dict[str, Any]:
        normalized_fork_id = self._normalize_key_for_proxy(fork_id)

        with self._lock:
            completed_result = self._fork_results.get(normalized_fork_id)
            running_task = self._fork_tasks.get(normalized_fork_id)

        if completed_result is not None:
            return self._materialize_fork_result_payload(
                completed_result,
                include_history=include_history,
            )

        if running_task is None:
            raise KeyError(f"fork_id '{normalized_fork_id}' is not found")

        try:
            await running_task
        except Exception:
            pass

        await asyncio.sleep(0)

        with self._lock:
            result_after_wait = self._fork_results.get(normalized_fork_id)

        if result_after_wait is None and running_task.done():
            try:
                direct_result = running_task.result()
            except Exception as exc:
                result_after_wait = {
                    "fork_id": normalized_fork_id,
                    "parent_fork_id": None,
                    "depth": 0,
                    "source_memory_key": "",
                    "memory_key": "",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "response": None,
                    "result": None,
                    "history": [],
                    "history_count": 0,
                    "history_included": True,
                }
            else:
                result_after_wait = direct_result

            compact_result = self._compact_fork_result_payload(result_after_wait)

            with self._lock:
                self._fork_results[normalized_fork_id] = compact_result

            result_after_wait = compact_result

        if result_after_wait is None:
            raise KeyError(f"fork_id '{normalized_fork_id}' has no result")

        return self._materialize_fork_result_payload(
            result_after_wait,
            include_history=include_history,
        )

    async def gather_all_fork_results(
        self,
        fork_ids: Optional[Any] = None,
        include_history: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Gather fork results; accepts fork_id strings, fork handles, or lists."""
        normalized_ids = normalize_fork_ids(
            fork_ids,
            self._normalize_key_for_proxy,
        )

        if normalized_ids is None:
            with self._lock:
                target_ids = sorted(
                    set(self._fork_tasks.keys()) | set(self._fork_results.keys())
                )
        else:
            target_ids = normalized_ids

        collected: Dict[str, Dict[str, Any]] = {}
        for target_id in target_ids:
            collected[target_id] = await self.wait_fork_result(
                target_id,
                include_history=include_history,
            )

        return collected

    def has_pending_fork_tasks(self, event_emitter: Any = None) -> bool:
        with self._lock:
            if event_emitter is None:
                return any(not task.done() for task in self._fork_tasks.values())

            for fork_id, task in self._fork_tasks.items():
                if task.done():
                    continue
                if self._fork_emitters.get(fork_id) is event_emitter:
                    return True
        return False


__all__ = [
    "SelfReferenceForkManagerMixin",
    "SelfReferenceInstanceHandle",
]
