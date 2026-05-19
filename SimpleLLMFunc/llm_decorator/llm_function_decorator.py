"""
LLM Function Decorator Module

This module provides LLM function decorators that delegate the execution of ordinary Python
functions to large language models. Using this decorator, simply define the function signature
(parameters and return type), then describe the function's execution strategy in the docstring.

Data Flow:
1. User defines function signature and docstring
2. Decorator captures function calls, extracts parameters and type information
3. Constructs system and user prompts
4. Calls LLM for reasoning
5. Processes tool calls (if necessary)
6. Converts LLM response to specified return type
7. Returns result to caller

Example:
```python
@llm_function(llm_interface=my_llm)
async def generate_summary(text: str) -> str:
    \"\"\"Generate a concise summary from the input text, should contain main points.\"\"\"
    pass
```
"""

import inspect
import json
from typing import (
    List,
    Callable,
    TypeVar,
    Dict,
    Any,
    cast,
    Optional,
    Union,
    Awaitable,
    AsyncGenerator,
)

from SimpleLLMFunc.base.react_loop import ReAct_loop
from SimpleLLMFunc.base.post_process import process_response
from SimpleLLMFunc.llm_decorator.signature import (
    parse_function_signature,
    setup_log_context,
)
from SimpleLLMFunc.llm_decorator.invocation_builder import (
    build_function_invocation_spec,
)
from SimpleLLMFunc.llm_decorator.utils import process_tools
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import push_error
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.observability.langfuse_client import (
    coerce_langfuse_metadata,
    get_langfuse_trace_context,
    langfuse_client,
    propagate_langfuse_trace_name,
    update_langfuse_parent_span,
    update_langfuse_trace_name,
)
from SimpleLLMFunc.hooks.abort import AbortSignal, ABORT_SIGNAL_PARAM
from SimpleLLMFunc.hooks.stream import ReactOutput, is_response_yield

T = TypeVar("T")


class LLMFunction:
    """Callable object returned by :func:`llm_function`.

    The public calling contract remains function-like.  The instance exists so
    framework internals can bind stable decorator state without relying on a
    large closure.
    """

    def __init__(
        self,
        func: Union[Callable[..., T], Callable[..., Awaitable[T]]],
        *,
        llm_interface: LLM_Interface,
        toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]],
        max_tool_calls: Optional[int],
        system_prompt_template: Optional[str],
        user_prompt_template: Optional[str],
        llm_kwargs: Dict[str, Any],
    ) -> None:
        self.func = func
        self.llm_interface = llm_interface
        self.toolkit = toolkit
        self.max_tool_calls = max_tool_calls
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.llm_kwargs = dict(llm_kwargs)

        self.signature = inspect.signature(func)
        self.func_name = func.__name__
        self.__wrapped__ = func
        self.__name__ = func.__name__
        self.__qualname__ = getattr(func, "__qualname__", func.__name__)
        self.__doc__ = func.__doc__
        self.__annotations__ = getattr(func, "__annotations__", {})
        self.__module__ = getattr(func, "__module__", __name__)
        self.__signature__ = self.signature
        if hasattr(inspect, "markcoroutinefunction"):
            inspect.markcoroutinefunction(self)

    async def _execute_function_with_events(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[ReactOutput, None]:
        """统一的执行逻辑，总是返回事件流。"""
        abort_signal = kwargs.pop(ABORT_SIGNAL_PARAM, None)
        if not isinstance(abort_signal, AbortSignal):
            abort_signal = None
        # Step 1: 解析函数签名
        sig, template_params = parse_function_signature(self.func, args, kwargs)

        # Step 2: 设置日志上下文
        async with setup_log_context(
            func_name=sig.func_name,
            trace_id=sig.trace_id,
            arguments=sig.bound_args.arguments,
        ):
            trace_context = get_langfuse_trace_context()
            # 创建 Langfuse parent span
            with langfuse_client.start_as_current_observation(
                as_type="span",
                name=f"{sig.func_name}_function_call",
                input=sig.bound_args.arguments,
                metadata=coerce_langfuse_metadata(
                    {
                        "function_name": sig.func_name,
                        "trace_id": sig.trace_id,
                        "tools_available": len(self.toolkit) if self.toolkit else 0,
                        "max_tool_calls": self.max_tool_calls,
                    }
                ),
                trace_context=trace_context,
            ) as function_span:
                update_langfuse_trace_name(sig.func_name)
                with propagate_langfuse_trace_name(sig.func_name):
                    update_langfuse_parent_span(
                        langfuse_client.get_current_observation_id()
                    )
                    try:
                        # Step 3: 构建 invocation spec（不在 decorator 中构建 provider messages）
                        invocation_spec = build_function_invocation_spec(
                            signature=sig,
                            template_params=template_params,
                            llm_kwargs=self.llm_kwargs,
                            system_prompt_template=self.system_prompt_template,
                            user_prompt_template=self.user_prompt_template,
                            toolkit=self.toolkit,
                        )
                        messages = invocation_spec.transcript_seed.initial_messages

                        # Step 4: 执行 ReAct 循环（返回事件流）
                        user_task_prompt = json.dumps(
                            sig.bound_args.arguments,
                            default=str,
                            ensure_ascii=False,
                        )

                        tool_param, tool_map = process_tools(
                            self.toolkit, sig.func_name
                        )
                        event_stream = ReAct_loop(
                            llm_interface=self.llm_interface,
                            messages=messages,
                            tools=tool_param,
                            tool_map=tool_map,
                            max_tool_calls=self.max_tool_calls,
                            stream=False,
                            trace_id=sig.trace_id,
                            user_task_prompt=user_task_prompt,
                            abort_signal=abort_signal,
                            invocation_spec=invocation_spec,
                            **self.llm_kwargs,
                        )

                        # Step 5: 处理事件流，解析响应后再 yield
                        last_response = None
                        last_messages = None
                        async for output in event_stream:
                            if is_response_yield(output):
                                # 收集原始响应和消息历史
                                last_response = output.response
                                last_messages = output.messages
                                # 不立即 yield ResponseYield，等解析完成后再 yield
                            else:
                                # EventYield 直接透传
                                yield output

                        result: Optional[T] = None

                        # 解析和验证最终响应
                        if last_response is not None:
                            result = process_response(last_response, sig.return_type)

                            # Yield 解析后的响应（而不是原始的 LLM 响应）
                            from SimpleLLMFunc.hooks.stream import ResponseYield

                            yield ResponseYield(
                                type="response",
                                response=result,  # 解析后的结果（str, Pydantic 对象等）
                                messages=last_messages if last_messages else [],
                            )

                        # 更新 Langfuse span
                        output_payload: Dict[str, Any] = {
                            "return_type": str(sig.return_type),
                            "result": result,
                        }
                        function_span.update(output=output_payload)
                    except Exception as exc:
                        # 更新 span 错误信息
                        function_span.update(
                            output={"error": str(exc)},
                        )
                        push_error(
                            f"Async LLM function '{sig.func_name}' execution failed: {str(exc)}",
                            location=get_location(),
                        )
                        raise

    async def stream(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[ReactOutput, None]:
        async for output in self._execute_function_with_events(*args, **kwargs):
            yield output

    async def __call__(self, *args: Any, **kwargs: Any) -> T:
        final_result: Optional[T] = None
        has_result = False
        async for output in self.stream(*args, **kwargs):
            if is_response_yield(output):
                final_result = cast(T, output.response)
                has_result = True

        if has_result:
            return cast(T, final_result)
        raise ValueError("No response received from LLM")


def llm_function(
    llm_interface: LLM_Interface,
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]] = None,
    max_tool_calls: Optional[int] = None,
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    **llm_kwargs: Any,
) -> Any:  # type: ignore
    """
    Async LLM function decorator that delegates function execution to a large language model.

    This decorator returns a callable object with function-like metadata. The public
    call contract is unchanged: use `await decorated_func(...)`, or
    `decorated_func.stream(...)` for the event stream.
    """

    def decorator(
        func: Union[Callable[..., T], Callable[..., Awaitable[T]]],
    ) -> LLMFunction:
        return LLMFunction(
            func,
            llm_interface=llm_interface,
            toolkit=toolkit,
            max_tool_calls=max_tool_calls,
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            llm_kwargs=dict(llm_kwargs),
        )

    return decorator


async_llm_function = llm_function
