from __future__ import annotations

import copy
import threading
from typing import Any, Dict, List, Optional


class SelfReferenceMutationQueues:
    """Pending SelfRef compaction and context mutation queues."""

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._active_destructive_mutation_keys: set[str] = set()
        self._pending_compactions: Dict[str, Dict[str, Any]] = {}
        self._pending_context_mutations: Dict[str, List[Dict[str, Any]]] = {}

    def clear(self) -> None:
        with self._lock:
            self._active_destructive_mutation_keys.clear()
            self._pending_compactions.clear()
            self._pending_context_mutations.clear()

    def mark_destructive_history_mutation(self, key: str) -> None:
        with self._lock:
            self._active_destructive_mutation_keys.add(key)

    def consume_destructive_history_mutation(self, key: str) -> bool:
        with self._lock:
            if key in self._active_destructive_mutation_keys:
                self._active_destructive_mutation_keys.remove(key)
                return True
        return False

    def queue_context_compaction(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._pending_compactions[key] = copy.deepcopy(payload)
        return copy.deepcopy(payload)

    def consume_pending_compaction(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            payload = self._pending_compactions.pop(key, None)
        return copy.deepcopy(payload) if payload is not None else None

    def has_pending_compaction(self, key: str) -> bool:
        with self._lock:
            return key in self._pending_compactions

    def queue_context_mutation(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            queue = self._pending_context_mutations.setdefault(key, [])
            queue.append(copy.deepcopy(payload))
        return copy.deepcopy(payload)

    def consume_pending_context_mutations(self, key: str) -> List[Dict[str, Any]]:
        with self._lock:
            payloads = self._pending_context_mutations.pop(key, [])
        return copy.deepcopy(payloads)


__all__ = ["SelfReferenceMutationQueues"]
