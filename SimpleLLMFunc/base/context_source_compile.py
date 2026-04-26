"""Compatibility adapter for the unified Stage 2 compile pipeline.

``build_compiled_messages_from_source`` is kept for existing callers/tests, but
it no longer owns a separate compile path. The implementation delegates to
``base.compile_pipeline``.
"""

from __future__ import annotations

from SimpleLLMFunc.base.compile_pipeline import build_compiled_messages_from_source


__all__ = ["build_compiled_messages_from_source"]
