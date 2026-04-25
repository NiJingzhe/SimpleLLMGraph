"""Step 4: Execute ReAct loop for llm_chat (streaming)."""

from __future__ import annotations

from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from SimpleLLMFunc.base.ReAct import ReAct_loop
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.llm_decorator.utils import process_tools
from SimpleLLMFunc.type import MessageList, ToolDefinitionList
from SimpleLLMFunc.hooks.stream import ReactOutput, responses_only
from SimpleLLMFunc.logger.logger import get_current_context_attribute
from SimpleLLMFunc.logger.context_manager import get_current_trace_id


def prepare_tools_for_execution(
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
    func_name: str,
) -> tuple[ToolDefinitionList, Dict[str, Callable[..., Awaitable[Any]]]]:
    """准备工具供执行使用"""
    return process_tools(toolkit, func_name)


async def _coerce_response_pairs(
    react_stream: AsyncGenerator[Any, None],
) -> AsyncGenerator[Tuple[Any, MessageList], None]:
    async for output in react_stream:
        if isinstance(output, tuple):
            yield cast(Tuple[Any, MessageList], output)
            continue
        if getattr(output, "type", None) == "response":
            typed_output = cast(ReactOutput, output)

            async def _single_item() -> AsyncGenerator[ReactOutput, None]:
                yield typed_output

            async for response, updated_messages in responses_only(_single_item()):
                yield response, updated_messages


async def execute_llm_call(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: Optional[int],
    stream: bool = False,
    enable_event: bool = False,
    trace_id: str = "",
    user_task_prompt: str = "",
    abort_signal: Optional[AbortSignal] = None,
    hooks: Any = None,
    **llm_kwargs: Any,
) -> AsyncGenerator[Union[Tuple[Any, MessageList], ReactOutput], None]:
    """Execute one LLM/ReAct stream.

    The core always emits `ReactOutput`. Legacy tuple output is derived here
    for callers that still request non-event behavior.
    """
    func_name = get_current_context_attribute("function_name") or "Unknown Function"
    current_trace_id = trace_id or get_current_trace_id() or ""

    react_stream = ReAct_loop(
        llm_interface=llm_interface,
        messages=messages,
        tools=tools,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=stream,
        enable_event=enable_event,
        trace_id=current_trace_id,
        user_task_prompt=user_task_prompt,
        abort_signal=abort_signal,
        hooks=hooks,
        **llm_kwargs,
    )

    if enable_event:
        async for output in react_stream:
            yield output
        return

    async for response, updated_messages in _coerce_response_pairs(
        cast(AsyncGenerator[Any, None], react_stream)
    ):
        yield response, updated_messages


async def execute_react_loop_streaming(
    llm_interface: LLM_Interface,
    messages: MessageList,
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
    max_tool_calls: Optional[int],
    stream: bool,
    llm_kwargs: Dict[str, Any],
    func_name: str,
    enable_event: bool = False,
    trace_id: str = "",
    user_task_prompt: str = "",
    abort_signal: Optional[AbortSignal] = None,
    hooks: Any = None,
) -> AsyncGenerator[Union[Tuple[Any, MessageList], ReactOutput], None]:
    """执行 ReAct 循环的流式版本（无重试），返回响应和更新后的消息（或 ReactOutput）"""
    # 1. 准备工具
    tool_param, tool_map = prepare_tools_for_execution(toolkit, func_name)

    # 2. 执行 LLM 调用（流式）
    response_stream = execute_llm_call(
        llm_interface=llm_interface,
        messages=messages,
        tools=tool_param,
        tool_map=tool_map,
        max_tool_calls=max_tool_calls,
        stream=stream,
        enable_event=enable_event,
        trace_id=trace_id,
        user_task_prompt=user_task_prompt,
        abort_signal=abort_signal,
        hooks=hooks,
        **llm_kwargs,
    )

    # 3. 返回响应流和更新后的消息（或 ReactOutput）
    async for output in response_stream:
        yield output
