"""Internal hook helpers for ReAct execution.

These hooks are intentionally framework-internal for now. They provide a small
set of interception points for mutating active ReAct state without expanding the
public decorator surface yet.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Optional

from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.base.mutation import ContextMutation
from SimpleLLMFunc.type import MessageList


@dataclass
class ReActHookExecutionContext:
    """Mutable execution view exposed to internal ReAct hooks."""

    trace_id: str
    func_name: str
    user_task_prompt: str
    messages: MessageList
    llm_kwargs: dict[str, Any]
    stream: bool
    iteration: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    last_response: Any = None
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    usage: Optional[CompletionUsage] = None
    aborted: bool = False
    final_response: Optional[str] = None


async def run_react_hook(
    hooks: Any,
    hook_name: str,
    state: ReActHookExecutionContext,
) -> None:
    """Invoke one optional internal ReAct hook if present."""

    if hooks is None:
        return

    hook = getattr(hooks, hook_name, None)
    if not callable(hook):
        return

    result = hook(state)
    if inspect.isawaitable(result):
        await result


async def collect_react_context_mutations(
    hooks: Any,
    state: ReActHookExecutionContext,
) -> list[ContextMutation]:
    if hooks is None:
        return []

    hook = getattr(hooks, "collect_context_mutations", None)
    if not callable(hook):
        return []

    result = hook(state)
    if inspect.isawaitable(result):
        result = await result

    if not isinstance(result, list):
        return []
    return result


__all__ = [
    "ReActHookExecutionContext",
    "collect_react_context_mutations",
    "run_react_hook",
]
