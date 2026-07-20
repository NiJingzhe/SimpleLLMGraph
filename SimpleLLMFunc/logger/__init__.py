"""Structured logging package for SimpleLLMFunc.

Public API:
    * :func:`app_log`       — INFO level
    * :func:`push_debug`    — DEBUG level
    * :func:`push_warning`  — WARNING level
    * :func:`push_error`    — ERROR level
    * :func:`push_critical` — CRITICAL level

Caller location (file, line, function) is captured automatically via
``stacklevel``; callsites do not pass ``location=`` manually.

Two handlers are configured by default:
    * stderr terminal with ANSI colors (:class:`ColorFormatter`)
    * JSONL file append at ``.simplellmfunc.log.jsonl``
      (:class:`JsonlFormatter`), overridable via ``SIMPLELLMFUNC_LOG_FILE``.
"""

from SimpleLLMFunc.logger.core import (
    app_log,
    get_current_context_attribute,
    get_current_trace_id,
    logger,
    push_critical,
    push_debug,
    push_error,
    push_warning,
    set_current_context_attribute,
)
from SimpleLLMFunc.logger.formatters import ColorFormatter, JsonlFormatter

__all__ = [
    "ColorFormatter",
    "JsonlFormatter",
    "app_log",
    "get_current_context_attribute",
    "get_current_trace_id",
    "logger",
    "push_critical",
    "push_debug",
    "push_error",
    "push_warning",
    "set_current_context_attribute",
]
