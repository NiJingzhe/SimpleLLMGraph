"""Log formatters for the SimpleLLMFunc logger.

Two formatters are provided:
    * :class:`ColorFormatter`  — ANSI-colored single-line output for the
      terminal.
    * :class:`JsonlFormatter`  — one structured JSON object per line for
      file-based logging.

Both consume standard :class:`logging.LogRecord` fields (``pathname``,
``lineno``, ``funcName``, ``levelname``, ``message``) populated
automatically by Python's logging machinery when ``stacklevel`` is used.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

# ANSI color codes keyed by level name.
_LEVEL_COLORS: dict[str, str] = {
    "DEBUG": "\033[37m",       # gray
    "INFO": "\033[36m",        # cyan
    "WARNING": "\033[33m",     # yellow
    "ERROR": "\033[31m",       # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """Colorized single-line formatter for terminal consumption."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname, "")
        fname = os.path.basename(record.pathname)
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        ts_str = ts.strftime("%H:%M:%S.%f")[:-3]
        return (
            f"{color}{ts_str} {record.levelname:<8}{_RESET} "
            f"{fname}:{record.lineno} {record.funcName}() | {record.getMessage()}"
        )


class JsonlFormatter(logging.Formatter):
    """JSON-lines formatter for structured file logging."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict[str, Any] = {
            "ts": ts.isoformat(),
            "level": record.levelname,
            "file": record.pathname,
            "line": record.lineno,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)
