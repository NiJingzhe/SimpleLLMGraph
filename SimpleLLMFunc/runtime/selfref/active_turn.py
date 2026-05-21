from __future__ import annotations

import contextvars
import threading
from typing import Any, Dict, Optional

from SimpleLLMFunc.base.react_hooks import ReActHookExecutionContext


class SelfReferenceActiveTurn:
    """Contextvars and active ReAct-state lookup for SelfReference."""

    def __init__(self, owner_id: int, lock: threading.RLock) -> None:
        self._lock = lock
        self._active_react_states_by_key: Dict[str, ReActHookExecutionContext] = {}
        self._active_memory_key_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_key_{owner_id}",
                default=None,
            )
        )
        self._active_fork_id_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_fork_id_{owner_id}",
                default=None,
            )
        )
        self._active_fork_depth_var: contextvars.ContextVar[int] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_fork_depth_{owner_id}",
                default=0,
            )
        )
        self._active_runtime_toolkit_var: contextvars.ContextVar[Any] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_toolkit_{owner_id}",
                default=None,
            )
        )
        self._active_template_params_var: contextvars.ContextVar[
            Optional[Dict[str, Any]]
        ] = contextvars.ContextVar(
            f"simplellmfunc_self_reference_active_template_params_{owner_id}",
            default=None,
        )
        self._active_react_state_var: contextvars.ContextVar[Optional[ReActHookExecutionContext]] = (
            contextvars.ContextVar(
                f"simplellmfunc_self_reference_active_react_state_{owner_id}",
                default=None,
            )
        )

    @property
    def active_memory_key_var(self):
        return self._active_memory_key_var

    @property
    def active_runtime_toolkit_var(self):
        return self._active_runtime_toolkit_var

    @property
    def active_template_params_var(self):
        return self._active_template_params_var

    @property
    def active_react_state_var(self):
        return self._active_react_state_var

    def clear(self) -> None:
        with self._lock:
            self._active_react_states_by_key.clear()

    def set_active_memory_key(self, key: str) -> contextvars.Token[Optional[str]]:
        return self._active_memory_key_var.set(key)

    def reset_active_memory_key(self, token: contextvars.Token[Optional[str]]) -> None:
        self._active_memory_key_var.reset(token)

    def get_active_memory_key(self) -> Optional[str]:
        return self._active_memory_key_var.get()

    def set_active_fork_context(
        self,
        fork_id: str,
        depth: int,
    ) -> tuple[contextvars.Token[Optional[str]], contextvars.Token[int]]:
        fork_token = self._active_fork_id_var.set(fork_id)
        depth_token = self._active_fork_depth_var.set(depth)
        return fork_token, depth_token

    def reset_active_fork_context(
        self,
        tokens: tuple[contextvars.Token[Optional[str]], contextvars.Token[int]],
    ) -> None:
        fork_token, depth_token = tokens
        self._active_fork_id_var.reset(fork_token)
        self._active_fork_depth_var.reset(depth_token)

    def get_active_fork_id(self) -> Optional[str]:
        return self._active_fork_id_var.get()

    def get_active_fork_depth(self) -> int:
        return self._active_fork_depth_var.get()

    def set_active_runtime_toolkit(self, toolkit: Any) -> contextvars.Token[Any]:
        return self._active_runtime_toolkit_var.set(toolkit)

    def reset_active_runtime_toolkit(self, token: contextvars.Token[Any]) -> None:
        self._active_runtime_toolkit_var.reset(token)

    def get_active_runtime_toolkit(self) -> Any:
        return self._active_runtime_toolkit_var.get()

    def set_active_template_params(
        self, template_params: Optional[Dict[str, Any]]
    ) -> contextvars.Token[Optional[Dict[str, Any]]]:
        copied = dict(template_params) if template_params is not None else None
        return self._active_template_params_var.set(copied)

    def reset_active_template_params(
        self, token: contextvars.Token[Optional[Dict[str, Any]]]
    ) -> None:
        self._active_template_params_var.reset(token)

    def get_active_template_params(self) -> Optional[Dict[str, Any]]:
        value = self._active_template_params_var.get()
        return dict(value) if value is not None else None

    def set_active_react_state(
        self, state: ReActHookExecutionContext
    ) -> tuple[contextvars.Token[Optional[ReActHookExecutionContext]], Optional[str]]:
        token = self._active_react_state_var.set(state)
        active_key = self.get_active_memory_key()
        if active_key is not None:
            with self._lock:
                self._active_react_states_by_key[active_key] = state
        return token, active_key

    def reset_active_react_state(
        self,
        token_and_key: tuple[
            contextvars.Token[Optional[ReActHookExecutionContext]],
            Optional[str],
        ],
    ) -> None:
        token, active_key = token_and_key
        self._active_react_state_var.reset(token)
        if active_key is not None:
            with self._lock:
                self._active_react_states_by_key.pop(active_key, None)

    def get_active_react_state(self) -> Optional[ReActHookExecutionContext]:
        return self._active_react_state_var.get()

    def get_mapped_react_state(self, key: str) -> Optional[ReActHookExecutionContext]:
        with self._lock:
            return self._active_react_states_by_key.get(key)


__all__ = ["SelfReferenceActiveTurn"]
