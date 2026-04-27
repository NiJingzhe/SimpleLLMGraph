"""Tool scheduler contracts and helpers for the new ReAct core."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, cast

from SimpleLLMFunc.base.tool_call.execution import _execute_single_tool_call
from SimpleLLMFunc.base.tool_call.extraction import parse_tool_call_arguments
from SimpleLLMFunc.base.types import (
    ContextMutation,
    ToolCancelledMutation,
    ToolResultMutation,
    ToolSchedulerResult,
    UserMessageMutation,
)
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.event_bus import EventBus
from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.hooks.events import (
    ToolCallEndEvent,
    ToolCallErrorEvent,
    ToolCallResult,
    ToolCallStartEvent,
    ToolCallsBatchEndEvent,
    ToolCallsBatchStartEvent,
    ReActEventType,
)
from SimpleLLMFunc.hooks.stream import EventYield
from SimpleLLMFunc.observability.langfuse_client import get_langfuse_trace_context
from SimpleLLMFunc.type.hooks import ToolResult
from SimpleLLMFunc.type.message import NormalizedMessageList
from SimpleLLMFunc.type.tool_call import ToolCall, dict_to_tool_call

async def schedule_tool_batch(
    *,
    tool_calls: List[Dict[str, Any]],
    messages: NormalizedMessageList,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    trace_id: str,
    func_name: str,
    iteration: int,
    event_bus: Optional[EventBus] = None,
    abort_signal: Optional[AbortSignal] = None,
) -> AsyncGenerator[EventYield | ToolSchedulerResult, None]:
    if not tool_calls:
        yield ToolSchedulerResult(mutations=[], total_tool_calls=0, aborted=False)
        return

    async def _emit_event(event: Any, **kwargs: Any) -> EventYield:
        if event_bus is not None:
            return await event_bus.emit_and_get(cast(Any, event), **kwargs)
        return EventYield(event=event)

    _ = messages
    trace_context = get_langfuse_trace_context()
    tool_results: List[ToolCallResult] = []

    typed_tool_calls: List[ToolCall] = [dict_to_tool_call(tc) for tc in tool_calls]
    yield await _emit_event(
        ToolCallsBatchStartEvent(
            event_type=ReActEventType.TOOL_CALLS_BATCH_START,
            timestamp=_time_now(),
            trace_id=trace_id,
            func_name=func_name,
            iteration=iteration,
            tool_calls=typed_tool_calls,
            batch_size=len(tool_calls),
        )
    )

    async def _run_one_tool(
        tool_call: Dict[str, Any],
    ) -> tuple[List[EventYield], List[ContextMutation]]:
        tool_call_id = tool_call.get("id", "")
        function_call = tool_call.get("function", {})
        tool_name = function_call.get("name", "")
        arguments_str = function_call.get("arguments", "{}")
        start_time = time.time()
        queued_events: List[EventYield] = []

        tool_event_emitter = ToolEventEmitter(
            _trace_id=trace_id,
            _func_name=func_name,
            _iteration=iteration,
            _tool_name=tool_name,
            _tool_call_id=tool_call_id,
            _event_bus=event_bus,
        )

        try:
            tool_call_dict, messages_to_append, _is_multimodal = await _execute_single_tool_call(
                tool_call,
                tool_map,
                event_emitter=tool_event_emitter,
                trace_context=trace_context,
            )

            mutations: List[ContextMutation] = []
            for message in messages_to_append:
                if message.get("role") == "tool":
                    mutations.append(
                        ToolResultMutation(
                            tool_call_id=str(message.get("tool_call_id", tool_call_id)),
                            content=str(message.get("content", "")),
                        )
                    )
                else:
                    mutations.append(UserMessageMutation(message=message))

            parsed_arguments_end = parse_tool_call_arguments(arguments_str, allow_closure=True)
            result_payload: ToolResult = ""
            for mutation in mutations:
                if isinstance(mutation, ToolResultMutation):
                    try:
                        result_payload = json.loads(mutation.content)
                    except Exception:
                        result_payload = mutation.content
                    break
                if isinstance(mutation, UserMessageMutation):
                    result_payload = mutation.message
                    break

            exec_time = time.time() - start_time
            tool_results.append(
                {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "result": result_payload,
                    "execution_time": exec_time,
                    "success": True,
                }
            )
            if tool_event_emitter.has_events():
                queued_events.extend(await tool_event_emitter.get_events())
            queued_events.append(
                await _emit_event(
                ToolCallEndEvent(
                    event_type=ReActEventType.TOOL_CALL_END,
                    timestamp=_time_now(),
                    trace_id=trace_id,
                    func_name=func_name,
                    iteration=iteration,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=parsed_arguments_end if parsed_arguments_end is not None else {},
                    result=result_payload,
                    execution_time=exec_time,
                    success=True,
                ),
                origin_overrides={"tool_name": tool_name, "tool_call_id": tool_call_id},
            )
            )
            return queued_events, mutations
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            exec_time = time.time() - start_time
            parsed_arguments_end = parse_tool_call_arguments(arguments_str, allow_closure=True)
            if tool_event_emitter.has_events():
                queued_events.extend(await tool_event_emitter.get_events())
            queued_events.append(
                await _emit_event(
                ToolCallErrorEvent(
                    event_type=ReActEventType.TOOL_CALL_ERROR,
                    timestamp=_time_now(),
                    trace_id=trace_id,
                    func_name=func_name,
                    iteration=iteration,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=parsed_arguments_end if parsed_arguments_end is not None else {},
                    error=exc,
                    error_message=str(exc),
                    error_type=type(exc).__name__,
                    execution_time=exec_time,
                ),
                origin_overrides={"tool_name": tool_name, "tool_call_id": tool_call_id},
            )
            )
            tool_results.append(
                {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "result": {"error": str(exc)},
                    "execution_time": exec_time,
                    "success": False,
                    "error": exc,
                }
            )
            return queued_events, [
                ToolResultMutation(
                    tool_call_id=tool_call_id,
                    content=json.dumps({"error": str(exc)}, ensure_ascii=False),
                )
            ]

    for tool_call in tool_calls:
        tool_call_id = str(tool_call.get("id", ""))
        function_call = tool_call.get("function", {})
        tool_name = str(function_call.get("name", ""))
        arguments_str = str(function_call.get("arguments", "{}"))
        parsed_arguments = parse_tool_call_arguments(arguments_str, allow_closure=True)
        yield await _emit_event(
            ToolCallStartEvent(
                event_type=ReActEventType.TOOL_CALL_START,
                timestamp=_time_now(),
                trace_id=trace_id,
                func_name=func_name,
                iteration=iteration,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=parsed_arguments if parsed_arguments is not None else {},
                tool_call=dict_to_tool_call(tool_call),
            ),
            origin_overrides={"tool_name": tool_name, "tool_call_id": tool_call_id},
        )

    tasks = [asyncio.create_task(_run_one_tool(tc)) for tc in tool_calls]
    pending_tasks = set(tasks)
    abort_task = asyncio.create_task(abort_signal.wait()) if abort_signal is not None else None
    queued_event_task = (
        asyncio.create_task(event_bus.get())
        if event_bus is not None and pending_tasks
        else None
    )

    mutations: List[ContextMutation] = []

    try:
        while pending_tasks:
            wait_set: set[asyncio.Task[Any]] = set(pending_tasks)
            if abort_task is not None:
                wait_set.add(abort_task)
            if queued_event_task is not None:
                wait_set.add(queued_event_task)

            done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            if abort_task is not None and abort_task in done:
                cancelled_mutations = []
                abort_reason = abort_signal.reason if abort_signal is not None else ""
                for task, tool_call in zip(tasks, tool_calls):
                    if not task.done():
                        task.cancel()
                        function_call = tool_call.get("function", {})
                        cancelled_mutations.append(
                            ToolCancelledMutation(
                                tool_call_id=str(tool_call.get("id", "")),
                                tool_name=str(function_call.get("name", "")),
                                abort_reason=abort_reason,
                            )
                        )
                await asyncio.gather(*tasks, return_exceptions=True)
                if queued_event_task is not None:
                    queued_event_task.cancel()
                    await asyncio.gather(queued_event_task, return_exceptions=True)
                if event_bus is not None:
                    while not event_bus.empty():
                        yield event_bus.get_nowait()
                yield await _emit_batch_end_event(
                    _emit_event,
                    trace_id=trace_id,
                    func_name=func_name,
                    iteration=iteration,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )
                yield ToolSchedulerResult(
                    mutations=cancelled_mutations,
                    total_tool_calls=len(tool_calls),
                    aborted=True,
                )
                return

            queue_event_ready = queued_event_task is not None and queued_event_task in done
            queue_event_output: Optional[EventYield] = None
            if queue_event_ready and queued_event_task is not None:
                queue_event_output = queued_event_task.result()

            if queue_event_output is not None:
                yield queue_event_output

            completed_tasks = [
                task for task in done if task in pending_tasks
            ]
            for completed_task in completed_tasks:
                pending_tasks.remove(completed_task)
                event_outputs, task_mutations = completed_task.result()
                for event in event_outputs:
                    yield event
                mutations.extend(task_mutations)

            if queue_event_ready:
                queued_event_task = (
                    asyncio.create_task(event_bus.get())
                    if event_bus is not None and pending_tasks
                    else None
                )

        if queued_event_task is not None:
            if queued_event_task.done() and not queued_event_task.cancelled():
                yield queued_event_task.result()
            else:
                queued_event_task.cancel()
                await asyncio.gather(queued_event_task, return_exceptions=True)
        if event_bus is not None:
            while not event_bus.empty():
                yield event_bus.get_nowait()
    finally:
        if abort_task is not None:
            abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)

    yield await _emit_batch_end_event(
        _emit_event,
        trace_id=trace_id,
        func_name=func_name,
        iteration=iteration,
        tool_calls=tool_calls,
        tool_results=tool_results,
    )

    yield ToolSchedulerResult(
        mutations=mutations,
        total_tool_calls=len(tool_calls),
        aborted=False,
    )
async def _emit_batch_end_event(
    emit_event: Callable[..., Awaitable[EventYield]],
    *,
    trace_id: str,
    func_name: str,
    iteration: int,
    tool_calls: List[Dict[str, Any]],
    tool_results: List[ToolCallResult],
) -> EventYield:
    success_count = sum(1 for tr in tool_results if tr["success"])
    error_count = len(tool_results) - success_count
    total_execution_time = sum(tr["execution_time"] for tr in tool_results)
    return await emit_event(
        ToolCallsBatchEndEvent(
            event_type=ReActEventType.TOOL_CALLS_BATCH_END,
            timestamp=_time_now(),
            trace_id=trace_id,
            func_name=func_name,
            iteration=iteration,
            tool_results=tool_results,
            batch_size=len(tool_calls),
            total_execution_time=total_execution_time,
            success_count=success_count,
            error_count=error_count,
        )
    )


def _time_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


__all__ = ["ToolSchedulerResult", "schedule_tool_batch"]
