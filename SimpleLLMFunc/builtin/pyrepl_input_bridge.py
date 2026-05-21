from __future__ import annotations

import queue
import threading
from typing import Dict, Optional


class PyReplInputBridge:
    """Process-wide bridge for worker ``input()`` requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_input_queues: Dict[str, queue.Queue[str]] = {}

    def register_input_queue(self, request_id: str) -> queue.Queue[str]:
        request_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending_input_queues[request_id] = request_queue
        return request_queue

    def pop_input_queue(self, request_id: str) -> Optional[queue.Queue[str]]:
        with self._lock:
            return self._pending_input_queues.pop(request_id, None)

    def submit_input(self, request_id: str, value: str) -> bool:
        with self._lock:
            request_queue = self._pending_input_queues.get(request_id)

        if request_queue is None:
            return False

        try:
            request_queue.put_nowait(value)
            return True
        except queue.Full:
            return False


GLOBAL_PYREPL_INPUT_BRIDGE = PyReplInputBridge()

__all__ = ["GLOBAL_PYREPL_INPUT_BRIDGE", "PyReplInputBridge"]
