"""Core logger setup and public log API for SimpleLLMFunc.

Provides leveled logging (debug / info / warning / error / critical) with
automatic caller-location capture via Python's ``stacklevel`` parameter.
Callsites never need to thread ``location=`` manually — the file, line, and
function name of the caller are populated on the :class:`logging.LogRecord`
by the logging framework itself.

Two handlers are attached to the ``"SimpleLLMFunc"`` logger:
    * **Console** — :class:`~SimpleLLMFunc.logger._formatters.ColorFormatter`
      writing to ``stderr`` at ``INFO`` level.
    * **File**    — :class:`~SimpleLLMFunc.logger._formatters.JsonlFormatter`
      appending JSONL records to ``.simplellmfunc.log.jsonl`` at ``DEBUG``
      level.  The path can be overridden via the ``SIMPLELLMFUNC_LOG_FILE``
      environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from SimpleLLMFunc.logger.formatters import ColorFormatter, JsonlFormatter

# --------------------------------------------------------------------------- #
# Logger setup
# --------------------------------------------------------------------------- #

_LOGGER_NAME = "SimpleLLMFunc"
_DEFAULT_LOG_FILE = ".simplellmfunc.log.jsonl"

logger: logging.Logger = logging.getLogger(_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # avoid double-output via the root logger

if not logger.handlers:
    # --- Console handler (colorized, human-readable) --- #
    _console_handler: logging.Handler = logging.StreamHandler(sys.stderr)
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(ColorFormatter())
    logger.addHandler(_console_handler)

    # --- File handler (JSONL, structured) --- #
    _file_path = os.environ.get("SIMPLELLMFUNC_LOG_FILE", _DEFAULT_LOG_FILE)
    _file_handler: logging.Handler = logging.FileHandler(
        _file_path, mode="a", encoding="utf-8"
    )
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(JsonlFormatter())
    logger.addHandler(_file_handler)


# --------------------------------------------------------------------------- #
# Public log API
# --------------------------------------------------------------------------- #
# ``stacklevel=2`` tells Python's logging to capture *our caller's* frame
# (pathname / lineno / funcName) instead of this module's frame.

def app_log(message: str) -> None:
    """Log at INFO level."""
    logger.info(message, stacklevel=2)


def push_debug(message: str) -> None:
    """Log at DEBUG level."""
    logger.debug(message, stacklevel=2)


def push_warning(message: str) -> None:
    """Log at WARNING level."""
    logger.warning(message, stacklevel=2)


def push_error(message: str) -> None:
    """Log at ERROR level."""
    logger.error(message, stacklevel=2)


def push_critical(message: str) -> None:
    """Log at CRITICAL level."""
    logger.critical(message, stacklevel=2)


# --------------------------------------------------------------------------- #
# Token-counting context stubs
# --------------------------------------------------------------------------- #
# These are placeholders consumed by the adapter layer for running token
# totals. Real context propagation will be introduced with the Event layer.

def get_current_trace_id() -> str:
    """No-op placeholder. Trace-id propagation is deferred to the Event layer."""
    return ""


def get_current_context_attribute(key: str) -> Any:
    """No-op placeholder — returns ``None``."""
    _ = key
    return None


def set_current_context_attribute(key: str, value: Any) -> None:
    """No-op placeholder."""
    _ = key, value
