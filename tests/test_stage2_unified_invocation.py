from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from SimpleLLMFunc import llm_chat, llm_function
from SimpleLLMFunc.base.compile_pipeline import CompiledTurnContext, compile_invocation_turn
from SimpleLLMFunc.hooks.stream import ReactOutput
from SimpleLLMFunc.llm_decorator.invocation_builder import (
    build_chat_invocation_spec,
    build_function_invocation_spec,
)
from SimpleLLMFunc.llm_decorator.llm_chat_decorator import HISTORY_PARAM_NAMES
from SimpleLLMFunc.llm_decorator.signature import parse_function_signature


class _DummyObservation:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def update(self, **kwargs: Any) -> None:
        _ = kwargs


def _completion(content: str) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content)
    return ChatCompletion(
        id="test-id",
        choices=[Choice(finish_reason="stop", index=0, message=message)],
        created=123,
        model="test-model",
        object="chat.completion",
    )


@pytest.mark.asyncio
async def test_function_and_chat_specs_share_compile_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 2 target: both decorators' semantics compile through one boundary."""

    def fake_compile_invocation_turn(*args: Any, **kwargs: Any) -> CompiledTurnContext:
        fake_compile_invocation_turn.calls.append((args, kwargs))
        return compile_invocation_turn(*args, **kwargs)

    fake_compile_invocation_turn.calls = []  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "SimpleLLMFunc.base.react_loop.compile_invocation_turn",
        fake_compile_invocation_turn,
    )

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    mock_llm.chat = AsyncMock(side_effect=[_completion("function done"), _completion("chat done")])

    with patch(
        "SimpleLLMFunc.llm_decorator.llm_function_decorator.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ), patch(
        "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ), patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):

        @llm_function(llm_interface=mock_llm)
        async def summarize(text: str) -> str:
            """Summarize text."""

        @llm_chat(llm_interface=mock_llm)
        async def chat_agent(message: str, history=None):
            """Chat system."""

        assert await summarize("hello") == "function done"

        async for _output in chat_agent("hello", history=[]):
            pass

    assert len(fake_compile_invocation_turn.calls) == 2  # type: ignore[attr-defined]
    modes = [call[0][0].mode for call in fake_compile_invocation_turn.calls]  # type: ignore[attr-defined]
    assert modes == ["function", "chat"]


@pytest.mark.asyncio
async def test_llm_function_invocation_state_is_per_call_not_decorator_shared() -> None:
    """Concurrent calls must not share parsed result state via decorator closure."""

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    first_gate = __import__("asyncio").Event()
    release_first = __import__("asyncio").Event()

    async def chat_side_effect(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        rendered = str(messages)
        if "first" in rendered:
            first_gate.set()
            await release_first.wait()
            return _completion("FIRST")
        await first_gate.wait()
        return _completion("SECOND")

    mock_llm.chat = AsyncMock(side_effect=chat_side_effect)

    with patch(
        "SimpleLLMFunc.llm_decorator.llm_function_decorator.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ), patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):

        @llm_function(llm_interface=mock_llm)
        async def echo(text: str) -> str:
            """Echo the input text."""

        import asyncio

        first_task = asyncio.create_task(echo("first"))
        await first_gate.wait()
        second_task = asyncio.create_task(echo("second"))
        second_result = await second_task
        release_first.set()
        first_result = await first_task

    assert first_result == "FIRST"
    assert second_result == "SECOND"


def test_invocation_builders_create_specs_not_provider_requests() -> None:
    async def summarize(text: str) -> str:
        """Summarize {topic}."""

    sig, template_params = parse_function_signature(
        summarize,
        ("hello",),
        {"_template_params": {"topic": "docs"}},
    )
    spec = build_function_invocation_spec(
        signature=sig,
        template_params=template_params,
        llm_kwargs={"temperature": 0},
    )

    assert spec.mode == "function"
    assert spec.prompt_contract.return_contract is not None
    assert spec.prompt_contract.base_instruction == "Summarize docs."
    assert spec.transcript_seed.initial_messages[0]["role"] == "system"
    assert not hasattr(spec, "llm_messages")

    async def agent(message: str, history=None):
        """Chat about {topic}."""

    chat_sig, chat_template_params = parse_function_signature(
        agent,
        ("hello",),
        {"history": [], "_template_params": {"topic": "docs"}},
    )
    chat_spec = build_chat_invocation_spec(
        signature=chat_sig,
        template_params=chat_template_params,
        llm_kwargs={},
        stream=False,
        return_mode="text",
        runtime_toolkit=None,
    )

    assert chat_spec.mode == "chat"
    assert chat_spec.prompt_contract.return_contract is None
    assert chat_spec.prompt_contract.include_must_principles is True
    assert chat_spec.transcript_seed.history_authority == "seed"
    assert HISTORY_PARAM_NAMES[0] == "history"


def test_decorator_modules_do_not_reintroduce_shared_parsed_result_container() -> None:
    """Architecture guard for Stage 2A isolation."""

    import SimpleLLMFunc.llm_decorator.llm_function_decorator as module

    source = inspect.getsource(module.llm_function)
    assert "parsed_result" not in source
    assert "List[Optional" not in source


def test_decorators_do_not_import_steps_layer() -> None:
    import SimpleLLMFunc.llm_decorator.llm_chat_decorator as chat_module
    import SimpleLLMFunc.llm_decorator.llm_function_decorator as function_module

    chat_source = inspect.getsource(chat_module)
    function_source = inspect.getsource(function_module)

    assert "llm_decorator.steps" not in chat_source
    assert "llm_decorator.steps" not in function_source
    assert "execute_react_loop" not in function_source
    assert "execute_react_loop_streaming" not in chat_source
    assert "parse_and_validate_response" not in function_source
    assert "process_chat_response_stream" not in chat_source


def test_invocation_builder_does_not_import_steps_layer() -> None:
    import SimpleLLMFunc.llm_decorator.invocation_builder as builder_module

    source = inspect.getsource(builder_module)
    assert "llm_decorator.steps" not in source
    assert "build_initial_prompts" not in source
    assert "build_compile_source_for_chat" not in source


@pytest.mark.asyncio
async def test_event_stream_is_the_only_chat_runtime_surface_and_function_has_stream_accessor() -> None:
    """Chat is stream-only; llm_function returns values and exposes events via .stream."""

    from SimpleLLMFunc.base.ReAct import ReAct_loop
    import SimpleLLMFunc.llm_decorator.llm_chat_decorator as chat_module
    import SimpleLLMFunc.llm_decorator.llm_function_decorator as function_module
    from SimpleLLMFunc.hooks.stream import EventYield, ResponseYield, is_response_yield

    assert "enable_event" not in inspect.signature(ReAct_loop).parameters
    assert "enable_event" not in inspect.signature(llm_chat).parameters
    assert "enable_event" not in inspect.signature(llm_function).parameters
    assert "enable_event" not in inspect.signature(build_chat_invocation_spec).parameters
    assert "enable_event" not in inspect.signature(build_function_invocation_spec).parameters
    assert "enable_event" not in chat_module.__dict__
    assert "enable_event" not in function_module.__dict__
    assert "enable_event" not in inspect.getsource(chat_module.llm_chat)
    assert "enable_event" not in inspect.getsource(function_module.llm_function)

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    mock_llm.chat = AsyncMock(return_value=_completion("done"))

    with patch(
        "SimpleLLMFunc.llm_decorator.llm_function_decorator.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ), patch(
        "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ), patch(
        "SimpleLLMFunc.base.react_loop.langfuse_client.start_as_current_observation",
        return_value=_DummyObservation(),
    ):

        @llm_function(llm_interface=mock_llm)
        async def summarize(text: str) -> str:
            """Summarize text."""

        @llm_chat(llm_interface=mock_llm)
        async def chat_agent(message: str, history=None):
            """Chat system."""

        assert await summarize("hello") == "done"

        function_outputs: list[ReactOutput] = []
        async for output in summarize.stream("hello"):
            function_outputs.append(output)

        chat_outputs: list[ReactOutput] = []
        async for output in chat_agent("hello", history=[]):
            chat_outputs.append(output)

    assert any(isinstance(output, EventYield) for output in function_outputs)
    assert any(isinstance(output, ResponseYield) for output in function_outputs)
    assert any(isinstance(output, EventYield) for output in chat_outputs)
    assert any(is_response_yield(output) for output in chat_outputs)


def test_stage2_does_not_keep_wrapper_only_modules_or_steps_package() -> None:
    """Stage 2 should keep hard boundaries without ceremony-only layers."""

    root = Path(__file__).resolve().parents[1]
    forbidden_paths = [
        root / "SimpleLLMFunc" / "llm_decorator" / "attachments.py",
        root / "SimpleLLMFunc" / "llm_decorator" / "invocation_runner.py",
        root / "SimpleLLMFunc" / "llm_decorator" / "result_adapters.py",
        root / "SimpleLLMFunc" / "runtime" / "plugins" / "protocol.py",
        root / "SimpleLLMFunc" / "llm_decorator" / "steps",
    ]

    assert [path for path in forbidden_paths if path.exists()] == []
