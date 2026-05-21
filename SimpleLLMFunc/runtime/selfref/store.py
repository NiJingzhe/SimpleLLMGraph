from __future__ import annotations

import copy
import threading
from typing import Any, Dict, List

from SimpleLLMFunc.base.types import DataFromSelfRef
from SimpleLLMFunc.runtime.selfref.context_ops import parse_data_from_selfref

MemoryHistory = List[Dict[str, Any]]


class SelfReferenceStore:
    """Durable history/source storage for SelfReference."""

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._history_store: Dict[str, MemoryHistory] = {}
        self._source_store: Dict[str, DataFromSelfRef] = {}

    def clear(self) -> None:
        with self._lock:
            self._history_store.clear()
            self._source_store.clear()

    def has_history(self, key: str) -> bool:
        with self._lock:
            return key in self._history_store

    def bind_history(self, key: str, history: MemoryHistory) -> None:
        source = parse_data_from_selfref(history)
        with self._lock:
            self._history_store[key] = copy.deepcopy(history)
            self._source_store[key] = source

    def store_history(self, key: str, history: MemoryHistory) -> None:
        source = parse_data_from_selfref(history)
        with self._lock:
            if key not in self._history_store:
                raise KeyError(f"Memory key '{key}' is not bound")
            self._history_store[key] = copy.deepcopy(history)
            self._source_store[key] = source

    def get_history(self, key: str) -> MemoryHistory:
        with self._lock:
            if key not in self._history_store:
                raise KeyError(f"Memory key '{key}' is not bound")
            return copy.deepcopy(self._history_store[key])

    def get_history_or_empty(self, key: str) -> MemoryHistory:
        with self._lock:
            return copy.deepcopy(self._history_store.get(key, []))

    def get_history_length(self, key: str) -> int:
        with self._lock:
            if key not in self._history_store:
                raise KeyError(f"Memory key '{key}' is not bound")
            return len(self._history_store[key])

    def ensure_bound(self, key: str) -> None:
        with self._lock:
            if key not in self._history_store:
                raise KeyError(f"Memory key '{key}' is not bound")

    def update_existing(self, key: str, history: MemoryHistory) -> None:
        source = parse_data_from_selfref(history)
        with self._lock:
            if key not in self._history_store:
                raise KeyError(f"Memory key '{key}' is not bound")
            self._history_store[key] = copy.deepcopy(history)
            self._source_store[key] = source

    def snapshot_source(self, key: str) -> DataFromSelfRef:
        with self._lock:
            if key not in self._source_store:
                if key not in self._history_store:
                    raise KeyError(f"Memory key '{key}' is not bound")
                self._source_store[key] = parse_data_from_selfref(
                    self._history_store[key]
                )
            return copy.deepcopy(self._source_store[key])

    def unbind_history(self, key: str) -> None:
        with self._lock:
            self._history_store.pop(key, None)
            self._source_store.pop(key, None)

    def list_history_keys(self) -> List[str]:
        with self._lock:
            keys = list(self._history_store.keys())
        keys.sort()
        return keys

    def key_exists_unlocked_view(self, key: str) -> bool:
        """Compatibility helper for code that already holds the owner lock."""

        return key in self._history_store


__all__ = ["MemoryHistory", "SelfReferenceStore"]
