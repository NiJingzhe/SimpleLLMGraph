"""Invocation-scoped SelfReference session/plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, cast

from SimpleLLMFunc.base.context_source import DataFromSelfRef
from SimpleLLMFunc.base.mutation import (
    ContextMutation,
    ContextSummaryMutation,
    ExperienceForgetMutation,
    ExperienceRememberMutation,
)
from SimpleLLMFunc.runtime.selfref.context_ops import parse_data_from_selfref
from SimpleLLMFunc.runtime.selfref.state import (
    MemoryHistory,
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SelfReference,
    _coerce_history_list,
)


@dataclass
class SelfRefFinalizeResult:
    history: MemoryHistory


class SelfRefSession:
    """Per-invocation adapter around the durable ``SelfReference`` backend.

    The session owns active-key contextvars, mutation collection, source
    snapshots, and finalize-time persistence for one decorated call.
    """

    def __init__(
        self,
        *,
        backend: SelfReference,
        memory_key: str,
        template_params: Optional[Dict[str, Any]],
        runtime_toolkit: Any,
        raw_history_reference: Optional[MemoryHistory],
        agent_instance: Any = None,
        base_system_prompt: str = "",
    ) -> None:
        self.backend = backend
        self.memory_key = memory_key
        self.template_params = template_params
        self.runtime_toolkit = runtime_toolkit
        self.raw_history_reference = raw_history_reference
        self.agent_instance = agent_instance
        self.base_system_prompt = base_system_prompt
        self.history_authority: Literal["external", "selfref", "seed"] = "seed"
        self.baseline_history_count = 0
        self._state_token: Any = None
        self._active_memory_key_token: Any = None
        self._toolkit_context_token: Any = None
        self._active_template_params_token: Any = None
        self._previous_runtime_toolkit: Any = None
        self._previous_memory_key: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        backend: SelfReference,
        memory_key: str,
        template_params: Optional[Dict[str, Any]],
        runtime_toolkit: Any,
        raw_history_reference: Optional[MemoryHistory],
        agent_instance: Any = None,
        base_system_prompt: str = "",
    ) -> "SelfRefSession":
        session = cls(
            backend=backend,
            memory_key=memory_key,
            template_params=template_params,
            runtime_toolkit=runtime_toolkit,
            raw_history_reference=raw_history_reference,
            agent_instance=agent_instance,
            base_system_prompt=base_system_prompt,
        )
        session._prepare_backend()
        return session

    def _prepare_backend(self) -> None:
        self._previous_runtime_toolkit = self.backend._get_active_runtime_toolkit()
        self._toolkit_context_token = self.backend._set_active_runtime_toolkit(
            self.runtime_toolkit
        )
        self._active_template_params_token = self.backend._set_active_template_params(
            self.template_params
        )
        self._previous_memory_key = self.backend._get_active_memory_key()

        if self.agent_instance is not None:
            self.backend.bind_agent_instance(
                self.agent_instance,
                default_memory_key=self.memory_key,
            )

        if self.raw_history_reference is not None:
            self.history_authority = "external"
            self.backend.bind_history(
                self.memory_key,
                cast(List[Dict[str, Any]], self.raw_history_reference),
            )
        elif self.backend.has_history(self.memory_key):
            self.history_authority = "selfref"
        else:
            self.history_authority = "seed"
            self.backend.bind_history(self.memory_key, [])

        self.baseline_history_count = self.backend.filtered_history_count(
            self.memory_key
        )
        self._active_memory_key_token = self.backend._set_active_memory_key(
            self.memory_key
        )

    def set_base_system_prompt(self, base_system_prompt: str) -> None:
        self.base_system_prompt = base_system_prompt

    def snapshot_context_messages(self) -> MemoryHistory:
        return self.backend.snapshot_context_messages(self.memory_key)

    def snapshot_source(self) -> DataFromSelfRef:
        return self.backend.snapshot_selfref_source(self.memory_key)

    def seed_system_prompt_if_missing(self, messages: List[Dict[str, Any]]) -> None:
        if self.backend.get_system_prompt(self.memory_key) is not None:
            return

        system_prompt: Optional[str] = None
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content")
            if isinstance(content, str):
                system_prompt = content
                break

        if system_prompt is None:
            return

        from SimpleLLMFunc.runtime.selfref.context_ops import (
            remove_framework_injected_prompt_blocks,
        )

        cleaned = remove_framework_injected_prompt_blocks(system_prompt)
        system_prompt_for_store = cleaned or system_prompt
        current_history: MemoryHistory = self.backend.snapshot_history(self.memory_key)
        seeded_history: MemoryHistory = [
            {"role": "system", "content": system_prompt_for_store},
            *current_history,
        ]
        self.backend.replace_history(
            key=self.memory_key,
            messages=seeded_history,
            strict=False,
        )

    async def on_run_start(self, state: Any) -> None:
        self._state_token = self.backend._set_active_react_state(state)

    async def before_finalize(self, state: Any) -> None:
        _ = state

    async def before_tool_batch(self, state: Any) -> None:
        _ = state

    async def after_tool_batch(self, state: Any) -> None:
        _ = state

    async def collect_context_mutations(self, state: Any) -> list[ContextMutation]:
        _ = state
        mutations: list[ContextMutation] = []
        pending = self.backend.consume_pending_compaction(self.memory_key)
        if pending is not None:
            mutations.append(
                ContextSummaryMutation(
                    summary_message={
                        "role": "assistant",
                        "content": cast(str, pending["rendered_summary"]),
                    },
                    remember=[
                        {"text": text}
                        for text in cast(List[str], pending.get("remember", []))
                    ],
                )
            )

        for payload in self.backend.consume_pending_context_mutations(self.memory_key):
            payload_type = payload.get("type")
            if payload_type == "experience_remember":
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    mutations.append(ExperienceRememberMutation(text=text))
            elif payload_type == "experience_forget":
                experience_id = payload.get("experience_id")
                if isinstance(experience_id, str) and experience_id.strip():
                    mutations.append(
                        ExperienceForgetMutation(experience_id=experience_id)
                    )

        return mutations

    def finalize(self, history: MemoryHistory) -> SelfRefFinalizeResult:
        resolved_history = cast(MemoryHistory, _coerce_history_list(history))
        resolved_source = parse_data_from_selfref(resolved_history)
        if (
            resolved_source.experiences
            or resolved_source.summary is not None
            or resolved_source.summary_message is not None
        ):
            active_history = self.backend.store_history(
                self.memory_key,
                resolved_history,
            )
        else:
            self.backend.consume_destructive_history_mutation(self.memory_key)
            merged_history = self.backend.merge_turn_history(
                key=self.memory_key,
                baseline_history_count=self.baseline_history_count,
                updated_history=resolved_history,
                commit=True,
            )
            active_history = self.backend.store_history(self.memory_key, merged_history)

        if self.raw_history_reference is not None:
            self.raw_history_reference[:] = cast(List[Dict[str, Any]], active_history)

        return SelfRefFinalizeResult(history=active_history)

    def close(self) -> None:
        if self._state_token is not None:
            try:
                self.backend._reset_active_react_state(self._state_token)
            except Exception:
                pass
            self._state_token = None

        if self._active_memory_key_token is not None:
            try:
                self.backend._reset_active_memory_key(self._active_memory_key_token)
            except ValueError:
                if self._previous_memory_key is None:
                    self.backend._active_memory_key_var.set(None)
                else:
                    self.backend._active_memory_key_var.set(self._previous_memory_key)
            self._active_memory_key_token = None

        if self._active_template_params_token is not None:
            try:
                self.backend._reset_active_template_params(
                    self._active_template_params_token
                )
            except ValueError:
                self.backend._active_template_params_var.set(None)
            self._active_template_params_token = None

        if self._toolkit_context_token is not None:
            try:
                self.backend._reset_active_runtime_toolkit(self._toolkit_context_token)
            except ValueError:
                self.backend._active_runtime_toolkit_var.set(
                    self._previous_runtime_toolkit
                )
            self._toolkit_context_token = None


def resolve_self_reference_key(
    explicit_key: Optional[str],
    func_name: str,
    template_params: Optional[Dict[str, Any]] = None,
) -> str:
    runtime_key = explicit_key
    if template_params is not None:
        override_key = template_params.get(SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM)
        if override_key is not None:
            if not isinstance(override_key, str):
                raise ValueError("self_reference key override must be a non-empty string")
            runtime_key = override_key

    if runtime_key is None:
        return func_name

    normalized = runtime_key.strip()
    if not normalized:
        raise ValueError("self_reference_key must be a non-empty string")
    return normalized


__all__ = ["SelfRefFinalizeResult", "SelfRefSession", "resolve_self_reference_key"]
