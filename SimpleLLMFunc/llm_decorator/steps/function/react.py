"""Step 4: Execute ReAct loop for llm_function."""

from __future__ import annotations

import json
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

from SimpleLLMFunc.base.ReAct import ReAct_loop
from SimpleLLMFunc.base.post_process import extract_content_from_response
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location, get_current_context_attribute
from SimpleLLMFunc.logger.context_manager import get_current_trace_id
from SimpleLLMFunc.type import MessageList, ToolDefinitionList
from SimpleLLMFunc.hooks.stream import (
    ReactOutput,
    responses_only,
    is_response_yield,
)


from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.utils import get_last_item_of_async_generator
from SimpleLLMFunc.llm_decorator.utils import process_tools


def prepare_tools_for_execution(
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
    func_name: str,
) -> tuple[ToolDefinitionList, Dict[str, Callable[..., Awaitable[Any]]]]:
    """准备工具供执行使用"""
    return process_tools(toolkit, func_name)


async def _coerce_responses(
    react_stream: AsyncGenerator[Any, None],
) -> AsyncGenerator[Any, None]:
    async for output in react_stream:
        if isinstance(output, tuple):
            response, _messages = output
            yield response
            continue
        if getattr(output, "type", None) == "response":
            typed_output = cast(ReactOutput, output)

            async def _single_item() -> AsyncGenerator[ReactOutput, None]:
                yield typed_output

            async for response, _messages in responses_only(_single_item()):
                yield response


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
    **llm_kwargs: Any,
) -> AsyncGenerator[Union[Any, ReactOutput], None]:
    """Execute one LLM/ReAct stream.

    The core always emits `ReactOutput`. Legacy raw-response iteration is derived
    here for callers that still request non-event behavior.
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
        **llm_kwargs,
    )

    if enable_event:
        async for output in react_stream:
            yield output
        return

    async for response in _coerce_responses(cast(AsyncGenerator[Any, None], react_stream)):
        yield response


async def get_final_response(
    response_stream: AsyncGenerator[Union[Any, ReactOutput], None],
    enable_event: bool = False,
) -> Any:
    """从响应流中获取最后一个响应

    当 enable_event=True 时，从 ReactOutput 中提取最后一个响应
    当 enable_event=False 时，直接获取最后一个响应值
    """
    if enable_event:
        last_response = None
        async for output in response_stream:
            if is_response_yield(output):
                last_response = output.response
        return last_response
    return await get_last_item_of_async_generator(response_stream)


def check_response_content_empty(response: Any, func_name: str) -> bool:
    """检查响应内容是否为空"""
    content = ""
    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        content = message.content if message and hasattr(message, "content") else ""

    return content == ""


async def retry_llm_call(
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: Optional[int],
    retry_times: int,
    func_name: str,
    enable_event: bool = False,
    trace_id: str = "",
    user_task_prompt: str = "",
    abort_signal: Optional[AbortSignal] = None,
    **llm_kwargs: Any,
) -> Any:
    """重试 LLM 调用"""
    final_response = None

    for attempt in range(retry_times + 1):
        if attempt > 0:
            push_debug(
                f"Async LLM function '{func_name}' retry attempt {attempt}...",
                location=get_location(),
            )

        # 执行 LLM 调用
        response_stream = execute_llm_call(
            llm_interface=llm_interface,
            messages=messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=max_tool_calls,
            stream=False,
            enable_event=enable_event,
            trace_id=trace_id,
            user_task_prompt=user_task_prompt,
            abort_signal=abort_signal,
            **llm_kwargs,
        )

        # 获取最终响应
        final_response = await get_final_response(response_stream)

        # 检查内容是否为空
        content = extract_content_from_response(final_response, func_name)
        if content != "":
            break

    # 最终检查
    if final_response:
        content = extract_content_from_response(final_response, func_name)
        if content == "":
            push_error(
                f"Async LLM function '{func_name}' response content still empty, "
                "retry attempts exhausted.",
                location=get_location(),
            )
            raise ValueError("LLM response content is empty after retries.")

    return final_response


async def execute_react_loop(
    llm_interface: LLM_Interface,
    messages: MessageList,
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
    max_tool_calls: Optional[int],
    llm_kwargs: Dict[str, Any],
    func_name: str,
    enable_event: bool = False,
    trace_id: str = "",
    user_task_prompt: str = "",
    abort_signal: Optional[AbortSignal] = None,
) -> Union[Any, AsyncGenerator[ReactOutput, None]]:
    """执行 ReAct 循环的完整流程（包含重试）

    当 enable_event=True 时，返回事件流生成器
    当 enable_event=False 时，返回最终响应值（向后兼容）
    """
    # 1. 准备工具
    tool_param, tool_map = prepare_tools_for_execution(toolkit, func_name)

    if enable_event:
        # 事件模式：返回事件流生成器
        async def event_stream() -> AsyncGenerator[ReactOutput, None]:
            # 执行 LLM 调用
            response_stream = execute_llm_call(
                llm_interface=llm_interface,
                messages=messages,
                tools=tool_param,
                tool_map=tool_map,
                max_tool_calls=max_tool_calls,
                stream=False,
                enable_event=True,
                trace_id=trace_id,
                user_task_prompt=user_task_prompt,
                abort_signal=abort_signal,
                **llm_kwargs,
            )

            # 收集所有输出和最后一个响应
            last_response = None
            async for output in response_stream:
                yield output
                if is_response_yield(output):
                    last_response = output.response

            # 检查响应内容是否为空
            if last_response and check_response_content_empty(last_response, func_name):
                push_warning(
                    f"Async LLM function '{func_name}' returned empty response content, "
                    "will retry automatically.",
                    location=get_location(),
                )

                # 重试 LLM 调用
                retry_times = llm_kwargs.get("retry_times", 2)
                retry_stream = execute_llm_call(
                    llm_interface=llm_interface,
                    messages=messages,
                    tools=tool_param,
                    tool_map=tool_map,
                    max_tool_calls=max_tool_calls,
                    stream=False,
                    enable_event=True,
                    trace_id=trace_id,
                    user_task_prompt=user_task_prompt,
                    abort_signal=abort_signal,
                    **llm_kwargs,
                )

                # Yield 重试的事件流
                async for output in retry_stream:
                    yield output
                    if is_response_yield(output):
                        last_response = output.response

            # 记录最终响应
            if last_response:
                push_debug(
                    f"Async LLM function '{func_name}' received response "
                    f"{json.dumps(last_response, default=str, ensure_ascii=False, indent=2)}",
                    location=get_location(),
                )

        return event_stream()
    else:
        # 向后兼容模式：返回最终响应值
        # 2. 执行 LLM 调用
        response_stream = execute_llm_call(
            llm_interface=llm_interface,
            messages=messages,
            tools=tool_param,
            tool_map=tool_map,
            max_tool_calls=max_tool_calls,
            stream=False,
            enable_event=False,
            trace_id=trace_id,
            user_task_prompt=user_task_prompt,
            abort_signal=abort_signal,
            **llm_kwargs,
        )

        # 3. 获取最终响应
        final_response = await get_final_response(response_stream, enable_event=False)

        # 4. 检查响应内容是否为空
        if check_response_content_empty(final_response, func_name):
            push_warning(
                f"Async LLM function '{func_name}' returned empty response content, "
                "will retry automatically.",
                location=get_location(),
            )

            # 5. 重试 LLM 调用
            retry_times = llm_kwargs.get("retry_times", 2)
            final_response = await retry_llm_call(
                llm_interface=llm_interface,
                messages=messages,
                tools=tool_param,
                tool_map=tool_map,
                max_tool_calls=max_tool_calls,
                retry_times=retry_times,
                func_name=func_name,
                enable_event=False,
                trace_id=trace_id,
                user_task_prompt=user_task_prompt,
                abort_signal=abort_signal,
                **llm_kwargs,
            )

        # 6. 记录最终响应
        push_debug(
            f"Async LLM function '{func_name}' received response "
            f"{json.dumps(final_response, default=str, ensure_ascii=False, indent=2)}",
            location=get_location(),
        )

        return final_response
