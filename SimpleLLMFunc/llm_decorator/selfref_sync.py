from __future__ import annotations

from typing import Optional

from SimpleLLMFunc.runtime.selfref.session import SelfRefSession
from SimpleLLMFunc.runtime.selfref.state import SelfReference


class SelfReferenceReActSyncHooks(SelfRefSession):
    def __init__(
        self,
        self_reference: SelfReference,
        memory_key: str,
    ) -> None:
        super().__init__(
            backend=self_reference,
            memory_key=memory_key,
            template_params=None,
            runtime_toolkit=None,
            raw_history_reference=None,
        )


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
