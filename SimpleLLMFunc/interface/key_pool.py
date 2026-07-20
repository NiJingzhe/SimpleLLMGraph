from __future__ import annotations

import heapq
import threading
from collections.abc import Sequence


class APIKeyPool:
    """Thread-safe in-memory pool that picks the least busy API key."""

    _instances: dict[str, "APIKeyPool"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, api_keys: Sequence[str], provider_id: str) -> "APIKeyPool":
        with cls._instances_lock:
            if provider_id in cls._instances:
                return cls._instances[provider_id]
            instance = super().__new__(cls)
            cls._instances[provider_id] = instance
            return instance

    def __init__(self, api_keys: Sequence[str], provider_id: str) -> None:
        if getattr(self, "initialized", False):
            return

        keys = list(api_keys)
        if not keys:
            raise ValueError(
                f"API key pool {provider_id} is empty. Please check your configuration."
            )

        self.api_keys = keys
        self.provider_id = provider_id
        self.heap: list[tuple[int, str]] = [(0, key) for key in keys]
        heapq.heapify(self.heap)
        self.key_to_task_count: dict[str, int] = {key: 0 for key in keys}
        self.key_to_index: dict[str, int] = {
            key: index for index, (_, key) in enumerate(self.heap)
        }
        self.lock = threading.Lock()
        self.initialized = True

    def get_least_loaded_key(self) -> str:
        with self.lock:
            if not self.heap:
                raise ValueError(f"{self.provider_id} has no available API keys")
            return self.heap[0][1]

    def increment_task_count(self, api_key: str) -> None:
        with self.lock:
            self._assert_known_key(api_key)
            new_count = self.key_to_task_count[api_key] + 1
            self.key_to_task_count[api_key] = new_count
            self._update_heap(api_key, new_count)

    def decrement_task_count(self, api_key: str) -> None:
        with self.lock:
            self._assert_known_key(api_key)
            new_count = max(0, self.key_to_task_count[api_key] - 1)
            self.key_to_task_count[api_key] = new_count
            self._update_heap(api_key, new_count)

    def _assert_known_key(self, api_key: str) -> None:
        if api_key not in self.key_to_task_count:
            raise ValueError(f"API key {api_key} is not in the pool")

    def _update_heap(self, api_key: str, new_task_count: int) -> None:
        if api_key not in self.key_to_index:
            raise ValueError(f"API key {api_key} is not in the heap")

        index = self.key_to_index[api_key]
        old_count, _ = self.heap[index]
        self.heap[index] = (new_task_count, api_key)

        if new_task_count < old_count:
            self._siftdown_with_index_update(0, index)
        elif new_task_count > old_count:
            self._siftup_with_index_update(index)

    def _siftup_with_index_update(self, pos: int) -> None:
        heap = self.heap
        endpos = len(heap)
        newitem = heap[pos]
        childpos = 2 * pos + 1

        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and heap[childpos][0] > heap[rightpos][0]:
                childpos = rightpos
            if newitem[0] <= heap[childpos][0]:
                break
            heap[pos] = heap[childpos]
            self.key_to_index[heap[childpos][1]] = pos
            pos = childpos
            childpos = 2 * pos + 1

        heap[pos] = newitem
        self.key_to_index[newitem[1]] = pos

    def _siftdown_with_index_update(self, startpos: int, pos: int) -> None:
        heap = self.heap
        newitem = heap[pos]

        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if newitem[0] >= parent[0]:
                break
            heap[pos] = parent
            self.key_to_index[parent[1]] = pos
            pos = parentpos

        heap[pos] = newitem
        self.key_to_index[newitem[1]] = pos
