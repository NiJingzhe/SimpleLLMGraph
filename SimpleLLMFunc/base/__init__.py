"""Baseline modules for SimpleLLMFunc internals.

Keep this package init lightweight. Importing ReAct here creates circular imports
with the Stage 2 InvocationSpec types during package initialization.
"""

__all__ = [
    "ReAct",
    "messages",
    "post_process",
    "tool_call",
    "type_resolve",
]
