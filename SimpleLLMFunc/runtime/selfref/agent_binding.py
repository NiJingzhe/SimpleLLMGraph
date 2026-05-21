from __future__ import annotations

from typing import Any, Optional


class SelfReferenceAgentBinding:
    """Agent callable binding state for ``SelfReference`` fork operations."""

    def __init__(self) -> None:
        self.agent_instance: Optional[Any] = None
        self.default_memory_key: Optional[str] = None

    def bind(
        self,
        agent_instance: Any,
        *,
        default_memory_key: Optional[str] = None,
    ) -> None:
        if not callable(agent_instance):
            raise ValueError("agent_instance must be callable")
        self.agent_instance = agent_instance
        self.default_memory_key = default_memory_key

    def clear(self) -> None:
        self.agent_instance = None
        self.default_memory_key = None


__all__ = ["SelfReferenceAgentBinding"]
