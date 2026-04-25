from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from SimpleLLMFunc.base.mutation import (
    ContextMutation,
    ContextSummaryMutation,
    ExperienceForgetMutation,
    ExperienceRememberMutation,
)
from SimpleLLMFunc.runtime.selfref.state import SelfReference
class SelfReferenceReActSyncHooks:
    def __init__(
        self,
        self_reference: SelfReference,
        memory_key: str,
    ) -> None:
        self._state_token = None
        self.self_reference = self_reference
        self.memory_key = memory_key

    async def on_run_start(self, state: Any) -> None:
        self._state_token = self.self_reference._set_active_react_state(state)

    async def before_finalize(self, state: Any) -> None:
        self.self_reference.set_context_messages(
            key=self.memory_key,
            messages=cast(
                List[Dict[str, Any]],
                getattr(state, "protocol_messages", state.messages),
            ),
        )

    async def before_tool_batch(self, state: Any) -> None:
        self.self_reference.bind_history(
            self.memory_key,
            cast(List[Dict[str, Any]], state.messages),
        )

    async def after_tool_batch(self, state: Any) -> None:
        self.self_reference.set_context_messages(
            key=self.memory_key,
            messages=cast(
                List[Dict[str, Any]],
                getattr(state, "protocol_messages", state.messages),
            ),
        )

    async def collect_context_mutations(self, state: Any) -> list[ContextMutation]:
        _ = state
        mutations: list[ContextMutation] = []
        pending = self.self_reference.consume_pending_compaction(self.memory_key)
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

        for payload in self.self_reference.consume_pending_context_mutations(
            self.memory_key
        ):
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

    def close(self) -> None:
        if self._state_token is None:
            return

        self.self_reference._reset_active_react_state(self._state_token)
        self._state_token = None


def build_selfref_react_sync_hooks(
    self_reference: Optional[SelfReference],
    memory_key: Optional[str],
) -> Optional[SelfReferenceReActSyncHooks]:
    if self_reference is None or memory_key is None:
        return None

    return SelfReferenceReActSyncHooks(
        self_reference=self_reference,
        memory_key=memory_key,
    )


__all__ = [
    "SelfReferenceReActSyncHooks",
    "build_selfref_react_sync_hooks",
]
