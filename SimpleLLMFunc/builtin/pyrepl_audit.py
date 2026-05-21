from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from SimpleLLMFunc.logger.logger_config import logger_config


class PyReplAuditLog:
    """Append-only JSONL audit log for one PyRepl instance."""

    def __init__(self, instance_id: str) -> None:
        self._lock = threading.Lock()
        self._dir = Path(logger_config.LOG_DIR) / "pyrepl" / instance_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "executions.jsonl"

    @property
    def log_dir(self) -> str:
        return str(self._dir)

    @property
    def log_file(self) -> str:
        return str(self._file)

    def append(self, payload: dict[str, Any]) -> None:
        with self._lock:
            with self._file.open("a", encoding="utf-8") as audit_stream:
                json.dump(payload, audit_stream, ensure_ascii=False, default=str)
                audit_stream.write("\n")


__all__ = ["PyReplAuditLog"]
