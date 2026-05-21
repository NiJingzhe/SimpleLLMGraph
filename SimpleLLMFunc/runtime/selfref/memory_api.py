from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from SimpleLLMFunc.runtime.selfref.state import MemoryHistory, SelfReference


class SelfReferenceMemoryHandle:
    """Keyed memory view used by ``self_reference.memory[<key>]``."""

    def __init__(self, owner: "SelfReference", key: str):
        self._owner = owner
        self._key = key

    def count(self) -> int:
        return len(self._owner.snapshot_history(self._key))

    def all(self) -> "MemoryHistory":
        return self._owner.snapshot_history(self._key)

    def get(self, index: int) -> Dict[str, Any]:
        messages = self._owner.snapshot_history(self._key)
        return messages[index].copy()

    def append(self, message: Dict[str, Any]) -> None:
        self._owner.append_message(self._key, message)

    def insert(self, index: int, message: Dict[str, Any]) -> None:
        self._owner.insert_message(self._key, index, message)

    def update(self, index: int, message: Dict[str, Any]) -> None:
        self._owner.update_message(self._key, index, message)

    def delete(self, index: int) -> None:
        self._owner.delete_message(self._key, index)

    def replace(self, messages: List[Dict[str, Any]]) -> None:
        self._owner.replace_history(self._key, messages, strict=True)

    def clear(self) -> None:
        system_prompt = self._owner.get_system_prompt(self._key)
        replacement: List[Dict[str, Any]] = []
        if system_prompt is not None:
            replacement.append({"role": "system", "content": system_prompt})
        self._owner.replace_history(self._key, replacement, strict=True)

    def get_system_prompt(self) -> Optional[str]:
        return self._owner.get_system_prompt(self._key)

    def set_system_prompt(self, text: str) -> None:
        self._owner.set_system_prompt(self._key, text)

    def append_system_prompt(self, text: str) -> None:
        self._owner.append_system_prompt(self._key, text)


class SelfReferenceMemoryProxy:
    """Container object exposed as ``self_reference.memory``."""

    def __init__(self, owner: "SelfReference"):
        self._owner = owner

    def __getitem__(self, key: str) -> SelfReferenceMemoryHandle:
        normalized_key = self._owner._normalize_key_for_proxy(key)
        if not self._owner.has_history(normalized_key):
            raise KeyError(
                f"Memory key '{normalized_key}' is not bound. "
                "Bind it before using self_reference.memory[key]."
            )
        return SelfReferenceMemoryHandle(self._owner, normalized_key)

    def keys(self) -> List[str]:
        return self._owner.list_history_keys()

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._owner.has_history(key)


__all__ = ["SelfReferenceMemoryHandle", "SelfReferenceMemoryProxy"]
