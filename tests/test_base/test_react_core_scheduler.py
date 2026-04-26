from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from SimpleLLMFunc.base.mutation import ToolCancelledMutation, ToolResultMutation, UserMessageMutation
from SimpleLLMFunc.base.tool_scheduler import ToolSchedulerResult, schedule_tool_batch
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.event_bus import EventBus
from SimpleLLMFunc.hooks.events import CustomEvent, ToolCallEndEvent, ToolCallStartEvent, ToolCallsBatchEndEvent, ToolCallsBatchStartEvent
from SimpleLLMFunc.hooks.stream import EventYield


@pytest.mark.asyncio
async def test_schedule_tool_batch_yields_events_and_result_mutation() -> None:
    async def fake_execute_single_tool_call(tool_call, tool_map, event_emitter=None, trace_context=None):
        _ = (tool_map, event_emitter, trace_context)
        return (
            tool_call,
            [{"role": "tool", "tool_call_id": tool_call["id"], "content": '"ok"'}],
            False,
        )

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "test_tool", "arguments": "{}"},
    }

    outputs = []
    with patch(
        "SimpleLLMFunc.base.tool_scheduler._execute_single_tool_call",
        new=fake_execute_single_tool_call,
    ):
        async for output in schedule_tool_batch(
            tool_calls=[tool_call],
            messages=[{"role": "user", "content": "hello"}],
            tool_map={"test_tool": AsyncMock(return_value="ok")},
            trace_id="trace-1",
            func_name="test_func",
            iteration=1,
            event_bus=EventBus(session_id="s", agent_call_id="a"),
        ):
            outputs.append(output)

    assert isinstance(outputs[0], EventYield)
    assert isinstance(outputs[0].event, ToolCallsBatchStartEvent)
    assert isinstance(outputs[1], EventYield)
    assert isinstance(outputs[1].event, ToolCallStartEvent)
    assert isinstance(outputs[2], EventYield)
    assert isinstance(outputs[2].event, ToolCallEndEvent)
    assert isinstance(outputs[3], EventYield)
    assert isinstance(outputs[3].event, ToolCallsBatchEndEvent)
    assert isinstance(outputs[4], ToolSchedulerResult)
    assert isinstance(outputs[4].mutations[0], ToolResultMutation)
    assert outputs[4].mutations[0].tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_schedule_tool_batch_maps_multimodal_messages_to_user_message_mutation() -> None:
    async def fake_execute_single_tool_call(tool_call, tool_map, event_emitter=None, trace_context=None):
        _ = (tool_map, event_emitter, trace_context)
        return (
            tool_call,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "image"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                    ],
                }
            ],
            True,
        )

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "test_tool", "arguments": "{}"},
    }

    outputs = []
    with patch(
        "SimpleLLMFunc.base.tool_scheduler._execute_single_tool_call",
        new=fake_execute_single_tool_call,
    ):
        async for output in schedule_tool_batch(
            tool_calls=[tool_call],
            messages=[{"role": "user", "content": "hello"}],
            tool_map={"test_tool": AsyncMock(return_value="ok")},
            trace_id="trace-1",
            func_name="test_func",
            iteration=1,
        ):
            outputs.append(output)

    result = outputs[-1]
    assert isinstance(result, ToolSchedulerResult)
    assert isinstance(result.mutations[0], UserMessageMutation)
    assert result.mutations[0].message["role"] == "user"


@pytest.mark.asyncio
async def test_schedule_tool_batch_abort_builds_tool_cancelled_mutations() -> None:
    abort_signal = AbortSignal()

    async def fake_execute_single_tool_call(tool_call, tool_map, event_emitter=None, trace_context=None):
        _ = (tool_call, tool_map, event_emitter, trace_context)
        await asyncio.sleep(0.05)
        return ({}, [], False)

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        },
    ]

    async def trigger_abort():
        await asyncio.sleep(0.01)
        abort_signal.abort("user_interrupt")

    outputs = []
    with patch(
        "SimpleLLMFunc.base.tool_scheduler._execute_single_tool_call",
        new=fake_execute_single_tool_call,
    ):
        abort_task = asyncio.create_task(trigger_abort())
        async for output in schedule_tool_batch(
            tool_calls=tool_calls,
            messages=[{"role": "user", "content": "hello"}],
            tool_map={"test_tool": AsyncMock(return_value="ok")},
            trace_id="trace-1",
            func_name="test_func",
            iteration=1,
            abort_signal=abort_signal,
        ):
            outputs.append(output)
        await abort_task

    result = outputs[-1]
    assert isinstance(result, ToolSchedulerResult)
    assert result.aborted is True
    assert all(isinstance(item, ToolCancelledMutation) for item in result.mutations)
    assert {item.tool_call_id for item in result.mutations} == {"call_1", "call_2"}


@pytest.mark.asyncio
async def test_schedule_tool_batch_yields_tool_emitter_events_before_tool_end() -> None:
    async def fake_execute_single_tool_call(tool_call, tool_map, event_emitter=None, trace_context=None):
        _ = (tool_call, tool_map, trace_context)
        assert event_emitter is not None
        await event_emitter.emit(
            "selfref_fork_start",
            {
                "fork_id": "fork_1",
                "depth": 1,
                "memory_key": "agent_main::fork::1",
                "status": "running",
            },
        )
        await asyncio.sleep(0)
        return (
            tool_call,
            [{"role": "tool", "tool_call_id": tool_call["id"], "content": '"ok"'}],
            False,
        )

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "test_tool", "arguments": "{}"},
    }

    outputs = []
    with patch(
        "SimpleLLMFunc.base.tool_scheduler._execute_single_tool_call",
        new=fake_execute_single_tool_call,
    ):
        async for output in schedule_tool_batch(
            tool_calls=[tool_call],
            messages=[{"role": "user", "content": "hello"}],
            tool_map={"test_tool": AsyncMock(return_value="ok")},
            trace_id="trace-1",
            func_name="test_func",
            iteration=1,
            event_bus=EventBus(session_id="s", agent_call_id="a"),
        ):
            outputs.append(output)

    assert isinstance(outputs[0], EventYield)
    assert isinstance(outputs[0].event, ToolCallsBatchStartEvent)
    assert isinstance(outputs[1], EventYield)
    assert isinstance(outputs[1].event, ToolCallStartEvent)
    assert isinstance(outputs[2], EventYield)
    assert isinstance(outputs[2].event, CustomEvent)
    assert outputs[2].event.event_name == "selfref_fork_start"
    assert isinstance(outputs[3], EventYield)
    assert isinstance(outputs[3].event, ToolCallEndEvent)
    assert isinstance(outputs[4], EventYield)
    assert isinstance(outputs[4].event, ToolCallsBatchEndEvent)
    assert isinstance(outputs[5], ToolSchedulerResult)
