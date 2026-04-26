from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from SimpleLLMFunc.base.react_loop import run_react_loop
from SimpleLLMFunc.base.context_source import CompileSource, DataFromAgentConfig, DataFromSelfRef
from SimpleLLMFunc.hooks.events import ReactEndEvent, ReActEventType
from SimpleLLMFunc.hooks.stream import EventYield
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function as OpenAIFunction,
)


def _completion(content: str, *, tool_calls=None) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)
    return ChatCompletion(
        id="test-id",
        choices=[Choice(finish_reason="stop", index=0, message=message)],
        created=123,
        model="test-model",
        object="chat.completion",
    )


class _DummyObservation:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def update(self, **kwargs: Any) -> None:
        _ = kwargs


def _tool_completion() -> ChatCompletion:
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_123",
                type="function",
                function=OpenAIFunction(name="test_tool", arguments='{"arg1":"x"}'),
            )
        ],
    )
    return ChatCompletion(
        id="tool-id",
        choices=[Choice(finish_reason="tool_calls", index=0, message=message)],
        created=123,
        model="test-model",
        object="chat.completion",
    )


def _tool_completion_with_content(content: str) -> ChatCompletion:
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_123",
                type="function",
                function=OpenAIFunction(name="test_tool", arguments='{"arg1":"x"}'),
            )
        ],
    )
    return ChatCompletion(
        id="tool-id",
        choices=[Choice(finish_reason="tool_calls", index=0, message=message)],
        created=123,
        model="test-model",
        object="chat.completion",
    )


@pytest.mark.asyncio
async def test_run_react_loop_no_tools_emits_response_and_react_end() -> None:
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.chat = AsyncMock(return_value=_completion("done"))

    with patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):
        outputs = []
        async for output in run_react_loop(
            llm_interface=llm,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            trace_id="trace-1",
            user_task_prompt="hello",
            abort_signal=None,
            hooks=None,
            llm_kwargs={},
        ):
            outputs.append(output)

    response_yields = [item for item in outputs if getattr(item, "type", None) == "response"]
    event_yields = [item for item in outputs if getattr(item, "type", None) == "event"]
    assert response_yields
    assert response_yields[0].response.choices[0].message.content == "done"
    react_end = next(
        item.event for item in event_yields if isinstance(item.event, ReactEndEvent)
    )
    assert react_end.final_response == "done"
    assert react_end.final_messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "done"},
    ]


@pytest.mark.asyncio
async def test_run_react_loop_event_mode_emits_react_end() -> None:
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.chat = AsyncMock(return_value=_completion("done"))

    with patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):
        outputs = []
        async for output in run_react_loop(
            llm_interface=llm,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            trace_id="trace-1",
            user_task_prompt="hello",
            abort_signal=None,
            hooks=None,
            llm_kwargs={},
        ):
            outputs.append(output)

    assert any(
        isinstance(item, EventYield)
        and isinstance(item.event, ReactEndEvent)
        and item.event.final_response == "done"
        for item in outputs
    )


@pytest.mark.asyncio
async def test_run_react_loop_calls_hooks_in_expected_order_for_no_tool_case() -> None:
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.chat = AsyncMock(return_value=_completion("done"))
    calls = []

    class Hooks:
        async def on_run_start(self, state: Any) -> None:
            calls.append("on_run_start")

        async def before_llm_call(self, state: Any) -> None:
            calls.append("before_llm_call")

        async def after_llm_call(self, state: Any) -> None:
            calls.append("after_llm_call")

        async def before_finalize(self, state: Any) -> None:
            calls.append("before_finalize")

    with patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):
        async for _ in run_react_loop(
            llm_interface=llm,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            trace_id="trace-1",
            user_task_prompt="hello",
            abort_signal=None,
            hooks=Hooks(),
            llm_kwargs={},
        ):
            pass

    assert calls == ["on_run_start", "before_llm_call", "after_llm_call", "before_finalize"]


@pytest.mark.asyncio
async def test_run_react_loop_before_finalize_can_override_final_messages() -> None:
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.chat = AsyncMock(return_value=_completion("done"))

    class Hooks:
        async def before_finalize(self, state: Any) -> None:
            state.messages = [
                {"role": "system", "content": "compacted"},
                {"role": "assistant", "content": "summary"},
            ]
            state.final_response = "compacted-response"

    with patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):
        outputs = []
        async for output in run_react_loop(
            llm_interface=llm,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            trace_id="trace-1",
            user_task_prompt="hello",
            abort_signal=None,
            hooks=Hooks(),
            llm_kwargs={},
        ):
            outputs.append(output)

    react_end = next(
        output.event
        for output in outputs
        if isinstance(output, EventYield) and isinstance(output.event, ReactEndEvent)
    )
    assert react_end.final_response == "compacted-response"
    assert react_end.final_messages == [
        {"role": "system", "content": "compacted"},
        {"role": "assistant", "content": "summary"},
    ]


@pytest.mark.asyncio
async def test_run_react_loop_with_tools_preserves_non_event_terminal_messages() -> None:
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.chat = AsyncMock(side_effect=[_tool_completion(), _completion("done")])

    async def tool_impl(arg1: str, event_emitter: Any = None) -> str:
        _ = (arg1, event_emitter)
        return "result"

    with patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):
        outputs = []
        async for output in run_react_loop(
            llm_interface=llm,
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
            tool_map={"test_tool": tool_impl},
            max_tool_calls=5,
            stream=False,
            trace_id="trace-1",
            user_task_prompt="hello",
            abort_signal=None,
            hooks=None,
            llm_kwargs={},
        ):
            outputs.append(output)

    react_end = next(
        item.event for item in outputs if isinstance(item, EventYield) and isinstance(item.event, ReactEndEvent)
    )
    assert react_end.final_response == "done"
    assert react_end.final_messages == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"arg1":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '"result"'},
        {"role": "assistant", "content": "done"},
    ]


@pytest.mark.asyncio
async def test_run_react_loop_with_tools_keeps_assistant_content_before_tool_results() -> None:
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.chat = AsyncMock(
        side_effect=[_tool_completion_with_content("Let me check that."), _completion("done")]
    )

    async def tool_impl(arg1: str, event_emitter: Any = None) -> str:
        _ = (arg1, event_emitter)
        return "result"

    with patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):
        outputs = []
        async for output in run_react_loop(
            llm_interface=llm,
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
            tool_map={"test_tool": tool_impl},
            max_tool_calls=5,
            stream=False,
            trace_id="trace-1",
            user_task_prompt="hello",
            abort_signal=None,
            hooks=None,
            llm_kwargs={},
        ):
            outputs.append(output)

    react_end = next(
        item.event for item in outputs if isinstance(item, EventYield) and isinstance(item.event, ReactEndEvent)
    )
    assert react_end.final_messages == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"arg1":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '"result"'},
        {"role": "assistant", "content": "done"},
    ]


@pytest.mark.asyncio
async def test_run_react_loop_max_tool_calls_none_keeps_iterating_until_done() -> None:
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.chat = AsyncMock(side_effect=[_tool_completion(), _tool_completion(), _completion("done")])

    async def tool_impl(arg1: str, event_emitter: Any = None) -> str:
        _ = (arg1, event_emitter)
        return "result"

    with patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):
        outputs = []
        async for output in run_react_loop(
            llm_interface=llm,
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
            tool_map={"test_tool": tool_impl},
            max_tool_calls=None,
            stream=False,
            trace_id="trace-1",
            user_task_prompt="hello",
            abort_signal=None,
            hooks=None,
            llm_kwargs={},
        ):
            outputs.append(output)

    assert llm.chat.await_count == 3
    react_end = next(
        item.event for item in outputs if isinstance(item, EventYield) and isinstance(item.event, ReactEndEvent)
    )
    assert react_end.final_response == "done"


@pytest.mark.asyncio
async def test_run_react_loop_preserves_selfref_source_after_tool_batch() -> None:
    llm = MagicMock()
    llm.model_name = "test-model"
    captured_messages: list[list[dict[str, Any]]] = []

    async def chat_side_effect(**kwargs: Any) -> ChatCompletion:
        captured_messages.append(list(kwargs["messages"]))
        if len(captured_messages) == 1:
            return _tool_completion()
        return _completion("done")

    llm.chat = AsyncMock(side_effect=chat_side_effect)

    async def tool_impl(arg1: str, event_emitter: Any = None) -> str:
        _ = (arg1, event_emitter)
        return "result"

    compile_source = CompileSource(
        data_from_agent_config=DataFromAgentConfig(
            base_system_prompt="test agent",
            tool_prompt_specs=[],
            include_must_principles=False,
        ),
        data_from_selfref=DataFromSelfRef(
            base_system_prompt="test agent",
            experiences=[],
            summary=None,
            summary_message=None,
            working_messages=[{"role": "user", "content": "hello"}],
        ),
        input_messages=[
            {"role": "system", "content": "test agent"},
            {"role": "user", "content": "hello"},
        ],
    )

    with patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):
        async for _ in run_react_loop(
            llm_interface=llm,
            messages=[{"role": "system", "content": "test agent"}, {"role": "user", "content": "hello"}],
            compile_source=compile_source,
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
            tool_map={"test_tool": tool_impl},
            max_tool_calls=5,
            stream=False,
            trace_id="trace-1",
            user_task_prompt="hello",
            abort_signal=None,
            hooks=None,
            llm_kwargs={},
        ):
            pass

    assert len(captured_messages) == 2
    assert captured_messages[1] == [
        {"role": "system", "content": "test agent"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"arg1":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '"result"'},
    ]
