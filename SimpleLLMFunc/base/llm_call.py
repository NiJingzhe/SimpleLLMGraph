"""Single-call contracts and helpers for the new ReAct core."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import time
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, cast

from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.base.mutation import AssistantMessageMutation, AssistantTruncatedMutation
from SimpleLLMFunc.base.post_process import (
    extract_content_from_response,
    extract_content_from_stream_response,
)
from SimpleLLMFunc.base.tool_call import (
    accumulate_tool_calls_from_chunks,
    collect_tool_argument_delta_payloads,
    extract_reasoning_details,
    extract_reasoning_details_from_stream,
    extract_tool_calls,
    extract_tool_calls_from_stream_response,
    StreamingToolCallState,
)
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.events import (
    LLMCallEndEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    ReActEventType,
)
from SimpleLLMFunc.hooks.stream import EventYield, ReactOutput, ResponseYield
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.type.tool_call import ToolCall, ToolDefinitionList, dict_to_tool_call

from SimpleLLMFunc.base.mutation import ContextMutation
from SimpleLLMFunc.base.messages import extract_usage_from_response
from SimpleLLMFunc.logger.logger import get_current_context_attribute


@dataclass
class SingleLLMCallResult:
    response: Any = None
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_details: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[CompletionUsage] = None
    execution_time: float = 0.0
    mutations: List[ContextMutation] = field(default_factory=list)
    aborted: bool = False


@dataclass
class SingleLLMPhaseResultYield:
    result: SingleLLMCallResult


def build_llm_call_end_event(
    *,
    trace_id: str,
    func_name: str,
    iteration: int,
    result: SingleLLMCallResult,
    messages: MessageList,
) -> LLMCallEndEvent:
    tool_calls_typed: List[ToolCall] = (
        [dict_to_tool_call(tc) for tc in result.tool_calls] if result.tool_calls else []
    )
    return LLMCallEndEvent(
        event_type=ReActEventType.LLM_CALL_END,
        timestamp=time_now(),
        trace_id=trace_id,
        func_name=func_name,
        iteration=iteration,
        response=result.response,
        messages=messages.copy(),
        tool_calls=tool_calls_typed,
        execution_time=result.execution_time,
        content=result.content,
        reasoning_details=result.reasoning_details,
        usage=result.usage,
    )


def time_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


async def _default_emit_event(event: Any, **_: Any) -> EventYield:
    return EventYield(event=event)


async def _await_next_chunk(stream_iter: Any) -> Any:
    return await stream_iter.__anext__()


async def execute_single_llm_phase(
    *,
    llm_interface: LLM_Interface,
    messages: MessageList,
    tools: ToolDefinitionList,
    llm_kwargs: Dict[str, Any],
    trace_id: str,
    func_name: str,
    iteration: int,
    stream: bool,
    emit_event: Optional[Callable[..., Awaitable[EventYield]]] = None,
    abort_signal: Optional[AbortSignal] = None,
) -> AsyncGenerator[ReactOutput | SingleLLMPhaseResultYield, None]:
    actual_emit_event = emit_event or _default_emit_event
    llm_kwargs_filtered = dict(llm_kwargs)
    if not tools:
        llm_kwargs_filtered.pop("tool_choice", None)

    start_time = time.time()
    yield await actual_emit_event(
        LLMCallStartEvent(
            event_type=ReActEventType.LLM_CALL_START,
            timestamp=time_now(),
            trace_id=trace_id,
            func_name=func_name,
            iteration=iteration,
            messages=messages.copy(),
            tools=tools,
            llm_kwargs=llm_kwargs,
            stream=stream,
        )
    )

    content = ""
    tool_calls: List[Dict[str, Any]] = []
    reasoning_details: List[Dict[str, Any]] = []
    response: Any = None
    aborted = False
    usage = None

    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _read_context_token_counters() -> tuple[int, int]:
        return (
            _as_int(get_current_context_attribute("input_tokens") or 0),
            _as_int(get_current_context_attribute("output_tokens") or 0),
        )

    def _usage_from_context_delta(
        input_before: int,
        output_before: int,
    ) -> Optional[CompletionUsage]:
        input_after, output_after = _read_context_token_counters()
        prompt_tokens = max(0, input_after - input_before)
        completion_tokens = max(0, output_after - output_before)
        if prompt_tokens == 0 and completion_tokens == 0:
            return None
        return CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    llm_input_tokens_before, llm_output_tokens_before = _read_context_token_counters()

    async def _close_stream(stream_obj: Any) -> None:
        close_method = getattr(stream_obj, "aclose", None)
        if callable(close_method):
            result = close_method()
            if inspect.isawaitable(result):
                await cast(Awaitable[Any], result)

    if stream:
        chunk_index = 0
        tool_call_chunks: List[Dict[str, Any]] = []
        stream_tool_call_states: Dict[int, StreamingToolCallState] = {}
        stream_response = llm_interface.chat_stream(
            messages=cast(List[Dict[str, Any]], messages),
            tools=tools,
            **llm_kwargs_filtered,
        )
        stream_iter = stream_response.__aiter__()
        while True:
            try:
                if abort_signal is None:
                    chunk = await stream_iter.__anext__()
                else:
                    abort_task = asyncio.create_task(abort_signal.wait())
                    next_task = asyncio.create_task(_await_next_chunk(stream_iter))
                    done, _ = await asyncio.wait(
                        {abort_task, next_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if abort_task in done:
                        next_task.cancel()
                        await asyncio.gather(next_task, return_exceptions=True)
                        abort_task.cancel()
                        raise asyncio.CancelledError
                    abort_task.cancel()
                    chunk = next_task.result()
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                aborted = True
                await _close_stream(stream_response)
                break

            chunk_content = extract_content_from_stream_response(chunk, func_name)
            content += chunk_content
            chunk_tool_call_chunks = extract_tool_calls_from_stream_response(chunk)
            tool_call_chunks.extend(chunk_tool_call_chunks)
            reasoning_details.extend(extract_reasoning_details_from_stream(chunk))  # type: ignore[arg-type]
            response = chunk

            yield await actual_emit_event(
                LLMChunkArriveEvent(
                    event_type=ReActEventType.LLM_CHUNK_ARRIVE,
                    timestamp=time_now(),
                    trace_id=trace_id,
                    func_name=func_name,
                    iteration=iteration,
                    chunk=chunk,
                    accumulated_content=content,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            if chunk_tool_call_chunks:
                payloads = collect_tool_argument_delta_payloads(
                    chunk_tool_call_chunks,
                    stream_tool_call_states,
                )
                from SimpleLLMFunc.hooks.events import ToolCallArgumentsDeltaEvent

                for tool_name, tool_call_id, argname, argcontent_delta in payloads:
                    origin_overrides: Dict[str, Any] = {"tool_call_id": tool_call_id}
                    if tool_name:
                        origin_overrides["tool_name"] = tool_name
                    yield await actual_emit_event(
                        ToolCallArgumentsDeltaEvent(
                            event_type=ReActEventType.TOOL_CALL_ARGUMENTS_DELTA,
                            timestamp=time_now(),
                            trace_id=trace_id,
                            func_name=func_name,
                            iteration=iteration,
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            argname=argname,
                            argcontent_delta=argcontent_delta,
                        ),
                        origin_overrides=origin_overrides,
                    )
            yield ResponseYield(response=chunk, messages=messages.copy())

            if abort_signal is not None and abort_signal.is_aborted:
                aborted = True
                await _close_stream(stream_response)
                break

        tool_calls = accumulate_tool_calls_from_chunks(tool_call_chunks)
    else:
        if abort_signal is not None and abort_signal.is_aborted:
            aborted = True
        else:
            response = await llm_interface.chat(
                messages=cast(List[Dict[str, Any]], messages),
                tools=tools,
                **llm_kwargs_filtered,
            )
            content = extract_content_from_response(response, func_name)
            tool_calls = extract_tool_calls(response)
            reasoning_details = extract_reasoning_details(response)  # type: ignore[assignment]
            yield ResponseYield(response=response, messages=messages.copy())

    usage = extract_usage_from_response(response)
    if usage is None:
        usage = _usage_from_context_delta(
            llm_input_tokens_before,
            llm_output_tokens_before,
        )

    mutations: List[ContextMutation]
    if aborted:
        mutations = [
            AssistantTruncatedMutation(
                partial_content=content,
                abort_reason=abort_signal.reason if abort_signal is not None else "",
            )
        ]
    elif tool_calls:
        mutations = [
            AssistantMessageMutation(
                content=content or None,
                tool_calls=tool_calls,
                reasoning_details=reasoning_details,
            )
        ]
    else:
        mutations = [AssistantMessageMutation(content=content)] if content else []

    result = SingleLLMCallResult(
        response=response,
        content=content,
        tool_calls=tool_calls,
        reasoning_details=reasoning_details,
        usage=usage,
        execution_time=time.time() - start_time,
        mutations=mutations,
        aborted=aborted,
    )

    yield await actual_emit_event(
        build_llm_call_end_event(
            trace_id=trace_id,
            func_name=func_name,
            iteration=iteration,
            result=result,
            messages=messages,
        )
    )
    yield SingleLLMPhaseResultYield(result=result)


__all__ = [
    "SingleLLMCallResult",
    "SingleLLMPhaseResultYield",
    "build_llm_call_end_event",
    "execute_single_llm_phase",
]
