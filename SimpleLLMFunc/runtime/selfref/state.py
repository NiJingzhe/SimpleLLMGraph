"""Self-reference runtime primitives for agent memory management.

This module provides a framework-level ``SelfReference`` object that can be
shared between ``llm_chat`` and tool runtimes (for example ``PyRepl``).
"""

from __future__ import annotations

import contextvars
import threading
from typing import Any, Dict, List, Optional, cast

from SimpleLLMFunc.base.react_hooks import ReActHookExecutionContext
from SimpleLLMFunc.runtime.primitives import RuntimePrimitiveBackend
from SimpleLLMFunc.runtime.selfref.active_turn import SelfReferenceActiveTurn
from SimpleLLMFunc.runtime.selfref.agent_binding import SelfReferenceAgentBinding
from SimpleLLMFunc.runtime.selfref.mutations import SelfReferenceMutationQueues
from SimpleLLMFunc.runtime.selfref.store import SelfReferenceStore
from SimpleLLMFunc.runtime.selfref.context_memory import (
    HISTORY_PARAM_NAMES,
    MemoryHistory,
    SelfReferenceContextMemoryMixin,
    coerce_history_list as _coerce_history_list,
    clone_messages as _clone_messages,
    normalize_key as _normalize_key,
    parse_data_from_selfref as _parse_data_from_selfref,
)
from SimpleLLMFunc.runtime.selfref.fork_manager import (
    SelfReferenceForkManagerMixin,
    SelfReferenceInstanceHandle,
)
from SimpleLLMFunc.runtime.selfref.fork_utils import (
    SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM,
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM,
)

from SimpleLLMFunc.runtime.selfref.memory_api import (
    SelfReferenceMemoryHandle,
    SelfReferenceMemoryProxy,
)


class SelfReference(
    SelfReferenceContextMemoryMixin,
    SelfReferenceForkManagerMixin,
    RuntimePrimitiveBackend,
):
    """Shared self-reference state object for agent memory operations."""

    def __init__(self):
        self._lock = threading.RLock()
        self._store = SelfReferenceStore(self._lock)
        self._mutations = SelfReferenceMutationQueues(self._lock)
        self._active_turn = SelfReferenceActiveTurn(id(self), self._lock)
        self._memory_proxy = SelfReferenceMemoryProxy(self)
        self._instance_proxy = SelfReferenceInstanceHandle(self)
        self._agent_binding = SelfReferenceAgentBinding()
        # Compatibility attributes for existing internal callers/tests.
        self._agent_instance: Optional[Any] = None
        self._agent_default_memory_key: Optional[str] = None
        self._fork_counter = 0
        self._fork_id_counter = 0
        self._fork_tasks: Dict[str, Any] = {}
        self._fork_results: Dict[str, Dict[str, Any]] = {}
        self._fork_emitters: Dict[str, Any] = {}
        self._experience_id_counter = 0
        self._install_ref_count = 0
        # Compatibility attributes for existing internal callers/tests.
        self._history_store = self._store._history_store
        self._source_store = self._store._source_store
        self._active_destructive_mutation_keys = self._mutations._active_destructive_mutation_keys
        self._pending_compactions = self._mutations._pending_compactions
        self._pending_context_mutations = self._mutations._pending_context_mutations
        self._active_react_states_by_key = self._active_turn._active_react_states_by_key
        self._active_memory_key_var = self._active_turn.active_memory_key_var
        self._active_runtime_toolkit_var = self._active_turn.active_runtime_toolkit_var
        self._active_template_params_var = self._active_turn.active_template_params_var
        self._active_react_state_var = self._active_turn.active_react_state_var

    def _normalize_key_for_proxy(self, key: str) -> str:
        return _normalize_key(key)

    def clone_for_fork(self, *, context) -> "SelfReference":
        _ = context
        return self

    def on_install(self, repl: Any) -> None:
        _ = repl
        with self._lock:
            self._install_ref_count += 1

    def on_close(self, repl: Any) -> None:
        _ = repl
        with self._lock:
            if self._install_ref_count > 0:
                self._install_ref_count -= 1
            remaining = self._install_ref_count

            if remaining > 0:
                return

            pending_tasks = list(self._fork_tasks.values())
            self._fork_tasks.clear()
            self._fork_results.clear()
            self._fork_emitters.clear()
            self._active_turn.clear()
            self._mutations.clear()
            self._store.clear()
            self._agent_binding.clear()
            self._agent_instance = None
            self._agent_default_memory_key = None
            self._fork_counter = 0
            self._fork_id_counter = 0
            self._experience_id_counter = 0

        for task in pending_tasks:
            if not task.done():
                task.cancel()

    @property
    def memory(self) -> SelfReferenceMemoryProxy:
        return self._memory_proxy

    @property
    def instance(self) -> SelfReferenceInstanceHandle:
        return self._instance_proxy

    def _set_active_memory_key(self, key: str) -> contextvars.Token[Optional[str]]:
        return self._active_turn.set_active_memory_key(_normalize_key(key))

    def _reset_active_memory_key(self, token: contextvars.Token[Optional[str]]) -> None:
        self._active_turn.reset_active_memory_key(token)

    def _get_active_memory_key(self) -> Optional[str]:
        return self._active_turn.get_active_memory_key()

    def _set_active_fork_context(
        self,
        fork_id: str,
        depth: int,
    ) -> tuple[contextvars.Token[Optional[str]], contextvars.Token[int]]:
        return self._active_turn.set_active_fork_context(fork_id, depth)

    def _reset_active_fork_context(
        self,
        tokens: tuple[contextvars.Token[Optional[str]], contextvars.Token[int]],
    ) -> None:
        self._active_turn.reset_active_fork_context(tokens)

    def _get_active_fork_id(self) -> Optional[str]:
        return self._active_turn.get_active_fork_id()

    def _get_active_fork_depth(self) -> int:
        return self._active_turn.get_active_fork_depth()

    def _set_active_runtime_toolkit(self, toolkit: Any) -> contextvars.Token[Any]:
        return self._active_turn.set_active_runtime_toolkit(toolkit)

    def _reset_active_runtime_toolkit(self, token: contextvars.Token[Any]) -> None:
        self._active_turn.reset_active_runtime_toolkit(token)

    def _get_active_runtime_toolkit(self) -> Any:
        return self._active_turn.get_active_runtime_toolkit()

    def _set_active_template_params(
        self, template_params: Optional[Dict[str, Any]]
    ) -> contextvars.Token[Optional[Dict[str, Any]]]:
        return self._active_turn.set_active_template_params(template_params)

    def _reset_active_template_params(
        self, token: contextvars.Token[Optional[Dict[str, Any]]]
    ) -> None:
        self._active_turn.reset_active_template_params(token)

    def _get_active_template_params(self) -> Optional[Dict[str, Any]]:
        return self._active_turn.get_active_template_params()

    def _set_active_react_state(
        self, state: ReActHookExecutionContext
    ) -> tuple[contextvars.Token[Optional[ReActHookExecutionContext]], Optional[str]]:
        return self._active_turn.set_active_react_state(state)

    def _reset_active_react_state(
        self,
        token_and_key: tuple[
            contextvars.Token[Optional[ReActHookExecutionContext]],
            Optional[str],
        ],
    ) -> None:
        self._active_turn.reset_active_react_state(token_and_key)

    def _get_active_react_state(self) -> Optional[ReActHookExecutionContext]:
        return self._active_turn.get_active_react_state()

    def _get_active_history_target(self, key: str) -> Optional[MemoryHistory]:
        normalized_key = _normalize_key(key)
        active_state = self._get_active_react_state()
        if active_state is not None and self._get_active_memory_key() == normalized_key:
            return _coerce_history_list(cast(List[Any], active_state.messages))

        mapped_state = self._active_turn.get_mapped_react_state(normalized_key)
        if mapped_state is None:
            return None

        return _coerce_history_list(cast(List[Any], mapped_state.messages))

    def _snapshot_active_history(self, key: str) -> Optional[MemoryHistory]:
        active_messages = self._get_active_history_target(key)
        if active_messages is None:
            return None
        return self._coerce_compiled_context_messages(active_messages)

    def mark_destructive_history_mutation(self, key: str) -> None:
        normalized_key = _normalize_key(key)
        self._mutations.mark_destructive_history_mutation(normalized_key)

    def consume_destructive_history_mutation(self, key: str) -> bool:
        normalized_key = _normalize_key(key)
        return self._mutations.consume_destructive_history_mutation(normalized_key)

    def consume_pending_compaction(self, key: str) -> Optional[Dict[str, Any]]:
        normalized_key = _normalize_key(key)
        return self._mutations.consume_pending_compaction(normalized_key)

    def has_pending_compaction(self, key: str) -> bool:
        normalized_key = _normalize_key(key)
        return self._mutations.has_pending_compaction(normalized_key)

    def queue_context_mutation(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized_key = _normalize_key(key)
        return self._mutations.queue_context_mutation(normalized_key, payload)

    def consume_pending_context_mutations(self, key: str) -> List[Dict[str, Any]]:
        normalized_key = _normalize_key(key)
        return self._mutations.consume_pending_context_mutations(normalized_key)

    def bind_history(self, key: str, history: List[Dict[str, Any]]) -> None:
        normalized_key = _normalize_key(key)
        normalized_history = self._coerce_compiled_context_messages(
            history,
            validate_working_linkage=False,
        )
        source = _parse_data_from_selfref(normalized_history)
        self._store.bind_history(normalized_key, normalized_history)

    def store_history(self, key: str, messages: List[Dict[str, Any]]) -> MemoryHistory:
        normalized_key = _normalize_key(key)
        normalized_history = self._coerce_compiled_context_messages(messages)
        source = _parse_data_from_selfref(normalized_history)
        self._store.store_history(normalized_key, normalized_history)
        return _clone_messages(normalized_history)

    def snapshot_selfref_source(self, key: str) -> DataFromSelfRef:
        normalized_key = _normalize_key(key)
        return self._store.snapshot_source(normalized_key)

    def unbind_history(self, key: str) -> None:
        normalized_key = _normalize_key(key)
        self._store.unbind_history(normalized_key)

    def list_history_keys(self) -> List[str]:
        return self._store.list_history_keys()

    def has_history(self, key: str) -> bool:
        normalized_key = _normalize_key(key)
        return self._store.has_history(normalized_key)

    def snapshot_history(self, key: str) -> MemoryHistory:
        normalized_key = _normalize_key(key)
        active_messages = self._snapshot_active_history(normalized_key)
        if active_messages is not None:
            return active_messages

        return self._store.get_history(normalized_key)

    def resolve_history_key(self, key: Optional[str] = None) -> str:
        """Resolve one usable history key for runtime self-reference operations.

        Resolution order when ``key`` is omitted:
        1. Active key in current execution context.
        2. Bound default key from ``bind_agent_instance``.
        3. The only bound key when exactly one exists.
        """

        if key is not None:
            normalized = _normalize_key(key)
            if not self.has_history(normalized):
                raise KeyError(f"Memory key '{normalized}' is not bound")
            return normalized

        active_key = self._get_active_memory_key()
        if active_key is not None and self.has_history(active_key):
            return active_key

        default_key = self.get_agent_default_memory_key()
        if default_key is not None:
            if not self.has_history(default_key):
                self.bind_history(default_key, [])
            return default_key

        keys = self.list_history_keys()
        if len(keys) == 1:
            return keys[0]

        if not keys:
            raise ValueError(
                "history key is required because no memory key is available"
            )

        raise ValueError("history key is required when multiple memory keys are bound")



__all__ = [
    "SelfReference",
    "SelfReferenceMemoryHandle",
    "SelfReferenceMemoryProxy",
    "SelfReferenceInstanceHandle",
    "SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM",
    "SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM",
    "SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM",
]
