"""Compatibility wrappers for the new ReAct core.

The actual implementation now lives in the new event-only core modules:

- ``base/react_loop.py``
- ``base/llm_call.py``

This module intentionally stays small and only preserves the public entrypoints
expected by upper layers and tests.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, Optional

from SimpleLLMFunc.base.context_source import CompileSource
from SimpleLLMFunc.llm_decorator.invocation_spec import InvocationSpec
from SimpleLLMFunc.base.llm_call import execute_single_llm_phase
from SimpleLLMFunc.base.react_loop import run_react_loop
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.stream import EventYield, ReactOutput
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger.context_manager import get_current_trace_id
from SimpleLLMFunc.logger.logger import get_current_context_attribute
from SimpleLLMFunc.observability.langfuse_client import langfuse_client
from SimpleLLMFunc.type import MessageList, ToolDefinitionList


async def execute_single_llm_call(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList = None,
    stream: bool = False,
    trace_id: str = "",
    emit_event: Optional[Any] = None,
    iteration: int = 0,
    abort_signal: Optional[AbortSignal] = None,
    **llm_kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """Preserve the public single-call API while delegating to the new core.

    This wrapper yields only event objects, matching the historical public API.
    """

    func_name = get_current_context_attribute("function_name") or "Unknown Function"
    current_trace_id = trace_id or get_current_trace_id() or ""

    async for output in execute_single_llm_phase(
        llm_interface=llm_interface,
        messages=messages,
        tools=tools,
        llm_kwargs=dict(llm_kwargs),
        trace_id=current_trace_id,
        func_name=func_name,
        iteration=iteration,
        stream=stream,
        emit_event=emit_event,
        abort_signal=abort_signal,
    ):
        if isinstance(output, EventYield):
            yield output.event


async def ReAct_loop(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList,
    tool_map: Dict[str, Any],
    max_tool_calls: Optional[int],
    stream: bool = False,
    trace_id: str = "",
    user_task_prompt: str = "",
    abort_signal: Optional[AbortSignal] = None,
    hooks: Any = None,
    compile_source: Optional[CompileSource] = None,
    tool_prompt_specs: Optional[list[Dict[str, Any]]] = None,
    include_must_principles: bool = False,
    invocation_spec: Optional[InvocationSpec] = None,
    **llm_kwargs: Any,
) -> AsyncGenerator[ReactOutput, None]:
    """Event-only ReAct loop."""
    async for output in run_react_loop(
        llm_interface=llm_interface,
        messages=messages,
        compile_source=compile_source,
        tools=tools,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=stream,
        trace_id=trace_id,
        user_task_prompt=user_task_prompt,
        abort_signal=abort_signal,
        hooks=hooks,
        llm_kwargs=dict(llm_kwargs),
        tool_prompt_specs=tool_prompt_specs,
        include_must_principles=include_must_principles,
        invocation_spec=invocation_spec,
    ):
        yield output


__all__ = ["ReAct_loop", "execute_single_llm_call", "langfuse_client"]
