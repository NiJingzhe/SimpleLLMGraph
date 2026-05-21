from __future__ import annotations

import queue
import threading
from typing import Optional

from .pyrepl_input_bridge import GLOBAL_PYREPL_INPUT_BRIDGE


class PyReplInputMixin:
    """Class-level input request/reply bridge for PyRepl."""

    _input_registry_lock = threading.Lock()
    _input_bridge = GLOBAL_PYREPL_INPUT_BRIDGE
    _pending_input_queues = _input_bridge._pending_input_queues

    @classmethod
    def _register_input_queue(cls, request_id: str) -> queue.Queue[str]:
        return cls._input_bridge.register_input_queue(request_id)

    @classmethod
    def _pop_input_queue(cls, request_id: str) -> Optional[queue.Queue[str]]:
        return cls._input_bridge.pop_input_queue(request_id)

    @classmethod
    def submit_input(cls, request_id: str, value: str) -> bool:
        """Submit a response for a pending ``input()`` request."""

        return cls._input_bridge.submit_input(request_id, value)


__all__ = ["PyReplInputMixin"]
