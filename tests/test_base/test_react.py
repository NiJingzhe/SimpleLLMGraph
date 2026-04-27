"""Event-only tests for base.react_loop entrypoints."""

from __future__ import annotations

from typing import Any
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function as OpenAIFunction,
)
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.base.react_loop import ReAct_loop, execute_single_llm_call
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.events import LLMCallEndEvent, ReactEndEvent, ReActEventType
from SimpleLLMFunc.hooks.stream import EventYield, ResponseYield


def _completion(content: str, *, tool_calls=None, usage=None) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)
    return ChatCompletion(
        id="test-id",
        choices=[Choice(finish_reason="stop" if not tool_calls else "tool_calls", index=0, message=message)],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=usage,
    )


def _tool_completion() -> ChatCompletion:
    return _completion(
        content=None,
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_123",
                type="function",
                function=OpenAIFunction(name="test_tool", arguments='{"arg1": "value1"}'),
            )
        ],
    )


class _DummyObservation:
    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        _ = kwargs


class TestExecuteLLM:
    def _make_chunk(self, content: str) -> ChatCompletionChunk:
        delta = ChoiceDelta(content=content, role="assistant")
        choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
        return ChatCompletionChunk(
            id="chunk-id",
            choices=[choice],
            created=123,
            model="test-model",
            object="chat.completion.chunk",
        )

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation")
    async def test_execute_non_streaming_no_tools_emits_response_and_react_end(
        self,
        mock_langfuse_start: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_langfuse_start.return_value = _DummyObservation()
        mock_llm_interface.model_name = "test-model"
        mock_llm_interface.chat = AsyncMock(return_value=_completion("Test response"))

        outputs = []
        async for output in ReAct_loop(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
        ):
            outputs.append(output)

        response_yields = [item for item in outputs if isinstance(item, ResponseYield)]
        end_events = [
            item.event
            for item in outputs
            if isinstance(item, EventYield) and isinstance(item.event, ReactEndEvent)
        ]
        assert len(response_yields) == 1
        assert response_yields[0].response.choices[0].message.content == "Test response"
        assert end_events
        assert end_events[-1].final_response == "Test response"

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation")
    async def test_execute_streaming_abort_appends_partial_history(
        self,
        mock_langfuse_start: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_langfuse_start.return_value = _DummyObservation()
        mock_llm_interface.model_name = "test-model"
        abort_signal = AbortSignal()

        async def stream_generator(**kwargs: Any):
            _ = kwargs
            yield self._make_chunk("Hello ")
            abort_signal.abort("user_interrupt")
            await asyncio.sleep(0)
            yield self._make_chunk("world")

        mock_llm_interface.chat_stream = stream_generator

        outputs = []
        async for output in ReAct_loop(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=True,
            abort_signal=abort_signal,
        ):
            outputs.append(output)

        end_events = [
            item.event
            for item in outputs
            if isinstance(item, EventYield) and isinstance(item.event, ReactEndEvent)
        ]
        assert end_events
        final_messages = end_events[-1].final_messages
        assert final_messages[-1]["role"] == "assistant"
        assert "Hello" in final_messages[-1]["content"]
        assert "world" not in final_messages[-1]["content"]
        assert end_events[-1].extra.get("aborted") is True

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation")
    async def test_execute_streaming_emits_tool_argument_delta_event(
        self,
        mock_langfuse_start: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_langfuse_start.return_value = _DummyObservation()
        mock_llm_interface.model_name = "test-model"

        def _chunk(arguments_delta: str, *, include_id: bool, include_name: bool):
            tool_call = ChoiceDeltaToolCall(
                index=0,
                id="call_123" if include_id else None,
                type="function" if include_id else None,
                function=ChoiceDeltaToolCallFunction(
                    name="execute_code" if include_name else None,
                    arguments=arguments_delta,
                ),
            )
            delta = ChoiceDelta(content=None, role="assistant", tool_calls=[tool_call])
            choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
            return ChatCompletionChunk(
                id="chunk-id",
                choices=[choice],
                created=123,
                model="test-model",
                object="chat.completion.chunk",
            )

        async def stream_generator(**kwargs: Any):
            _ = kwargs
            yield _chunk('{{"code":"print(', include_id=True, include_name=True)
            yield _chunk('1)"}', include_id=False, include_name=False)

        mock_llm_interface.chat_stream = stream_generator
        tool_map = {"execute_code": AsyncMock(return_value="ok")}
        tools = [{"type": "function", "function": {"name": "execute_code"}}]

        delta_events = []
        async for output in ReAct_loop(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=tools,
            tool_map=tool_map,
            max_tool_calls=1,
            stream=True,
        ):
            if (
                isinstance(output, EventYield)
                and output.event.event_type == ReActEventType.TOOL_CALL_ARGUMENTS_DELTA
            ):
                delta_events.append(output.event)

        assert delta_events
        merged_delta = "".join(
            event.argcontent_delta for event in delta_events if event.argname == "code"
        )
        assert merged_delta.endswith("print(1)")
        assert all(event.tool_call_id == "call_123" for event in delta_events)

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation")
    async def test_execute_streaming_usage_fallback_without_tools(
        self,
        mock_langfuse_start: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_langfuse_start.return_value = _DummyObservation()
        mock_llm_interface.model_name = "test-model"

        usage_tail_chunk = ChatCompletionChunk(
            id="test-id-usage",
            choices=[],
            created=1234567891,
            model="test-model",
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

        async def stream_generator(**kwargs: Any):
            _ = kwargs
            yield self._make_chunk("chunk")
            yield usage_tail_chunk

        mock_llm_interface.chat_stream = stream_generator

        llm_end_usage = None
        async for output in ReAct_loop(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=True,
        ):
            if isinstance(output, EventYield) and isinstance(output.event, LLMCallEndEvent):
                llm_end_usage = output.event.usage

        assert llm_end_usage is not None
        assert llm_end_usage.total_tokens == 12

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation")
    async def test_execute_event_mode_attaches_origin_metadata(
        self,
        mock_langfuse_start: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_langfuse_start.return_value = _DummyObservation()
        mock_llm_interface.model_name = "test-model"
        mock_llm_interface.chat = AsyncMock(return_value=_completion("Test response"))

        origins = []
        async for output in ReAct_loop(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
        ):
            if isinstance(output, EventYield):
                origins.append(output.origin)

        assert origins
        session_ids = {origin.session_id for origin in origins}
        assert len(session_ids) == 1
        assert "" not in session_ids
        event_seqs = [origin.event_seq for origin in origins]
        assert event_seqs == sorted(event_seqs)
        assert event_seqs[0] == 1

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation")
    async def test_before_llm_call_hook_can_mutate_active_messages(
        self,
        mock_langfuse_start: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_langfuse_start.return_value = _DummyObservation()
        mock_llm_interface.model_name = "test-model"
        mock_llm_interface.chat = AsyncMock(return_value=_completion("Test response"))

        class _Hooks:
            async def before_llm_call(self, state: Any) -> None:
                state.messages = [*state.messages, {"role": "user", "content": "[hook] extra context"}]

        async for _output in ReAct_loop(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            hooks=_Hooks(),
        ):
            pass

        all_called_messages = [call.kwargs["messages"] for call in mock_llm_interface.chat.await_args_list]
        assert any(
            {"role": "user", "content": "[hook] extra context"} in message_list
            for message_list in all_called_messages
        )

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation")
    async def test_before_finalize_hook_can_override_final_messages_and_response(
        self,
        mock_langfuse_start: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_langfuse_start.return_value = _DummyObservation()
        mock_llm_interface.model_name = "test-model"
        mock_llm_interface.chat = AsyncMock(return_value=_completion("Test response"))

        class _Hooks:
            async def before_finalize(self, state: Any) -> None:
                state.messages = [
                    {"role": "system", "content": "compacted"},
                    {"role": "assistant", "content": "summary"},
                ]
                state.final_response = "compacted-response"

        react_end = None
        async for output in ReAct_loop(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            tool_map={},
            max_tool_calls=5,
            stream=False,
            hooks=_Hooks(),
        ):
            if isinstance(output, EventYield) and isinstance(output.event, ReactEndEvent):
                react_end = output.event

        assert react_end is not None
        assert react_end.final_response == "compacted-response"
        assert react_end.final_messages == [
            {"role": "system", "content": "compacted"},
            {"role": "assistant", "content": "summary"},
        ]

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation")
    async def test_react_loop_hook_order_with_tool_iterations(
        self,
        mock_langfuse_start: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_langfuse_start.return_value = _DummyObservation()
        mock_llm_interface.model_name = "test-model"
        mock_llm_interface.chat = AsyncMock(side_effect=[_tool_completion(), _completion("done")])

        async def _tool_impl(arg1: str, event_emitter: Any = None) -> str:
            _ = (arg1, event_emitter)
            return "result"

        calls: list[tuple[str, int, int]] = []

        class Hooks:
            async def on_run_start(self, state: Any) -> None:
                calls.append(("on_run_start", state.iteration, len(state.messages)))

            async def before_llm_call(self, state: Any) -> None:
                calls.append(("before_llm_call", state.iteration, len(state.messages)))

            async def after_llm_call(self, state: Any) -> None:
                calls.append(("after_llm_call", state.iteration, len(state.messages)))

            async def before_tool_batch(self, state: Any) -> None:
                calls.append(("before_tool_batch", state.iteration, len(state.messages)))

            async def after_tool_batch(self, state: Any) -> None:
                calls.append(("after_tool_batch", state.iteration, len(state.messages)))

            async def before_finalize(self, state: Any) -> None:
                calls.append(("before_finalize", state.iteration, len(state.messages)))

        async for _output in ReAct_loop(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
            tool_map={"test_tool": _tool_impl},
            max_tool_calls=5,
            stream=False,
            hooks=Hooks(),
        ):
            pass

        assert [name for name, _, _ in calls] == [
            "on_run_start",
            "before_llm_call",
            "after_llm_call",
            "before_tool_batch",
            "after_tool_batch",
            "before_llm_call",
            "after_llm_call",
            "before_finalize",
        ]


class TestExecuteSingleLLMCall:
    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client")
    @patch("SimpleLLMFunc.base.react_loop.get_current_context_attribute")
    async def test_non_streaming_returns_final_llm_call_end_event(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
        mock_chat_completion: Any,
    ) -> None:
        mock_get_context.return_value = "test_func"
        mock_llm_interface.chat = AsyncMock(return_value=mock_chat_completion)
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        events = []
        async for event in execute_single_llm_call(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            stream=False,
        ):
            events.append(event)

        assert events
        assert isinstance(events[-1], LLMCallEndEvent)
        assert events[-1].content == "Test response"
        assert events[-1].tool_calls == []

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.base.react_loop.langfuse_client")
    @patch("SimpleLLMFunc.base.react_loop.get_current_context_attribute")
    async def test_streaming_accumulates_content_and_tool_calls(
        self,
        mock_get_context: MagicMock,
        mock_langfuse: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        mock_get_context.return_value = "test_func"

        def _chunk(
            arguments_delta: str,
            *,
            include_id: bool,
            include_name: bool,
            content: str | None = None,
        ):
            tool_call = ChoiceDeltaToolCall(
                index=0,
                id="call_123" if include_id else None,
                type="function" if include_id else None,
                function=ChoiceDeltaToolCallFunction(
                    name="execute_code" if include_name else None,
                    arguments=arguments_delta,
                ),
            )
            delta = ChoiceDelta(content=content, role="assistant", tool_calls=[tool_call])
            choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
            return ChatCompletionChunk(
                id="chunk-id",
                choices=[choice],
                created=123,
                model="test-model",
                object="chat.completion.chunk",
            )

        async def stream_generator(**kwargs: Any):
            _ = kwargs
            yield _chunk('{{"code":"print(', include_id=True, include_name=True, content="hel")
            yield _chunk('1)"}', include_id=False, include_name=False, content="lo")

        mock_llm_interface.chat_stream = stream_generator
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_observation)
        mock_observation.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        events = []
        async for event in execute_single_llm_call(
            llm_interface=mock_llm_interface,
            messages=sample_messages,
            tools=None,
            stream=True,
        ):
            events.append(event)

        assert isinstance(events[-1], LLMCallEndEvent)
        assert events[-1].content == "hello"
        assert len(events[-1].tool_calls) == 1
        assert events[-1].tool_calls[0].id == "call_123"
