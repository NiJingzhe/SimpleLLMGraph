"""Tests for llm_chat decorator self-reference integration behavior."""

from __future__ import annotations

import contextvars
import copy
import inspect
import threading
import json
from types import SimpleNamespace
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as ChatCompletionMessageToolCall,
    Function,
)
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)

from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.interface.openai_responses_compatible import (
    OpenAIResponsesCompatible,
)
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.runtime.selfref import (
    SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM,
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM,
    SelfReference,
)
from SimpleLLMFunc.hooks.events import ReActEventType, ReactEndEvent
from SimpleLLMFunc.hooks.stream import EventOrigin, EventYield, ReactOutput
from SimpleLLMFunc.llm_decorator.llm_chat_decorator import (
    DEFAULT_MAX_TOOL_CALLS,
    llm_chat,
)
from SimpleLLMFunc.observability.langfuse_client import (
    langfuse_client as shared_langfuse_client,
    reset_langfuse_trace_context,
    set_langfuse_trace_context,
)
from SimpleLLMFunc.tool import Tool


_MUST_PROMPT_BLOCK = "<must_principles>"
_MUST_PROMPT_RULE = (
    "Invoke tools through native structured tool_calls / function-calling fields"
)


def _builtin_self_reference(repl: PyRepl) -> SelfReference:
    self_reference = repl.get_runtime_backend("selfref")
    assert isinstance(self_reference, SelfReference)
    return self_reference


class _DummyObservation:
    """Simple context manager used to stub Langfuse observations."""

    def __enter__(self) -> "_DummyObservation":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        return None


class _TrackingObservation:
    """Context manager that tracks nested Langfuse observations."""

    def __init__(
        self,
        tracker: "_TrackingLangfuseClient",
        record: dict[str, Any],
    ) -> None:
        self._tracker = tracker
        self._record = record
        self._trace_token: Optional[contextvars.Token[Optional[str]]] = None
        self._observation_token: Optional[contextvars.Token[Optional[str]]] = None

    def __enter__(self) -> "_TrackingObservation":
        self._trace_token = self._tracker._trace_id_var.set(self._record["trace_id"])
        self._observation_token = self._tracker._observation_id_var.set(
            self._record["span_id"]
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._observation_token is not None:
            self._tracker._observation_id_var.reset(self._observation_token)
        if self._trace_token is not None:
            self._tracker._trace_id_var.reset(self._trace_token)
        return None

    def update(self, **kwargs: Any) -> None:
        self._record.setdefault("updates", []).append(kwargs)

    def set_attribute(self, key: str, value: Any) -> None:
        self._record.setdefault("attributes", {})[key] = value

    def is_recording(self) -> bool:
        return True


class _TrackingLangfuseClient:
    """Tiny contextvar-backed Langfuse stub for trace propagation tests."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._counter = 0
        self.trace_names_by_trace_id: dict[str, str] = {}
        self._trace_id_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar("test_langfuse_trace_id", default=None)
        )
        self._observation_id_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar("test_langfuse_observation_id", default=None)
        )
        self._trace_name_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar("test_langfuse_trace_name", default=None)
        )

    def start_as_current_observation(self, **kwargs: Any) -> _TrackingObservation:
        self._counter += 1
        trace_context = kwargs.get("trace_context")
        current_trace_id = self._trace_id_var.get()
        trace_id = ""
        if isinstance(trace_context, dict):
            raw_trace_id = trace_context.get("trace_id")
            if isinstance(raw_trace_id, str):
                trace_id = raw_trace_id
        if not trace_id:
            trace_id = current_trace_id or f"trace_{self._counter}"

        parent_span_id: Optional[str] = None
        if isinstance(trace_context, dict):
            raw_parent_span_id = trace_context.get("parent_span_id")
            if isinstance(raw_parent_span_id, str) and raw_parent_span_id:
                parent_span_id = raw_parent_span_id
        if parent_span_id is None:
            parent_span_id = self._observation_id_var.get()

        record = {
            "span_id": f"obs_{self._counter}",
            "as_type": kwargs.get("as_type"),
            "name": kwargs.get("name"),
            "input": kwargs.get("input"),
            "metadata": kwargs.get("metadata"),
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "trace_context": trace_context,
            "trace_name": self._trace_name_var.get(),
        }
        self.records.append(record)
        return _TrackingObservation(self, record)

    def propagate_attributes(self, *, trace_name: Optional[str] = None, **kwargs: Any):
        _ = kwargs
        tracker = self

        class _Context:
            def __enter__(self_nonlocal):
                self_nonlocal._token = tracker._trace_name_var.set(trace_name)
                trace_id = tracker._trace_id_var.get()
                if trace_id and trace_name:
                    tracker.trace_names_by_trace_id[trace_id] = trace_name
                return None

            def __exit__(self_nonlocal, exc_type: Any, exc: Any, tb: Any) -> None:
                tracker._trace_name_var.reset(self_nonlocal._token)
                return None

        return _Context()

    def get_current_trace_id(self) -> str:
        return self._trace_id_var.get() or ""

    def get_current_observation_id(self) -> str:
        return self._observation_id_var.get() or ""

    def _active_observation(self) -> Optional[_TrackingObservation]:
        current_observation_id = self._observation_id_var.get()
        if not current_observation_id:
            return None

        for record in reversed(self.records):
            if record.get("span_id") == current_observation_id:
                return _TrackingObservation(self, record)

        return None

    def create_trace_id(self) -> str:
        self._counter += 1
        return f"trace_{self._counter}"


def _make_chat_completion(content: Optional[str]) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content)
    choice = Choice(finish_reason="stop", index=0, message=message)
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )


def _make_responses_text_response(content: str) -> Response:
    return Response(
        id="resp_test",
        created_at=123,
        model="gpt-test",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                role="assistant",
                status="completed",
                type="message",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text=content,
                        annotations=[],
                    )
                ],
            )
        ],
        parallel_tool_calls=True,
        status="completed",
        text={"format": {"type": "text"}},
        tool_choice="auto",
        tools=[],
        truncation="disabled",
        usage=ResponseUsage(
            input_tokens=3,
            input_tokens_details={"cached_tokens": 0},
            output_tokens=2,
            output_tokens_details={"reasoning_tokens": 0},
            total_tokens=5,
        ),
    )


def _make_text_response(content: str) -> Response:
    return _make_responses_text_response(content)


def _make_tool_call_completion(
    tool_name: str, arguments: Dict[str, Any]
) -> ChatCompletion:
    tool_call = ChatCompletionMessageToolCall(
        id="call_execute_code",
        function=Function(
            name=tool_name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
        type="function",
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[tool_call],
    )
    choice = Choice(
        finish_reason="tool_calls",
        index=0,
        message=message,
    )
    return ChatCompletion(
        id="test-tool-call",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )


def test_llm_chat_binds_wrapped_agent_instance_to_self_reference() -> None:
    """SelfReference should mount the decorated callable for recursive fork use."""

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    self_reference = SelfReference()

    @llm_chat(
        llm_interface=mock_llm,
        self_reference=self_reference,
        self_reference_key="agent_main",
    )
    async def agent(message: str, history=None):
        """test agent"""

    assert self_reference.get_agent_instance() is agent


def test_llm_chat_default_max_tool_calls_is_none() -> None:
    """llm_chat should not impose a default tool-call limit."""

    signature = inspect.signature(llm_chat)

    assert DEFAULT_MAX_TOOL_CALLS is None
    assert signature.parameters["max_tool_calls"].default is None


def test_llm_chat_strict_signature_enforces_history_message_shape() -> None:
    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    @llm_chat(llm_interface=mock_llm, strict_signature=True)
    async def agent(history, message: str, _template_params=None):
        """test agent"""

    assert callable(agent)


def test_llm_chat_strict_signature_rejects_non_canonical_shapes() -> None:
    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with pytest.raises(TypeError, match="first parameter"):

        @llm_chat(llm_interface=mock_llm, strict_signature=True)
        async def bad_agent(message: str, history=None):
            """bad agent"""

    with pytest.raises(TypeError, match="second parameter"):

        @llm_chat(llm_interface=mock_llm, strict_signature=True)
        async def bad_agent2(history, message):
            """bad agent"""

    with pytest.raises(TypeError, match="second parameter name"):

        @llm_chat(llm_interface=mock_llm, strict_signature=True)
        async def bad_agent3(history, user_message: str):
            """bad agent"""

    with pytest.raises(TypeError, match="only allows"):

        @llm_chat(llm_interface=mock_llm, strict_signature=True)
        async def bad_agent4(history, message: str, extra: int):
            """bad agent"""


@pytest.mark.asyncio
async def test_llm_chat_auto_resolves_builtin_self_reference_from_pyrepl() -> None:
    """Decorator should pick up the builtin selfref pack from a default PyRepl."""

    captured_system_prompt: str | None = None

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal captured_system_prompt
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                captured_system_prompt = maybe_prompt

        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    repl = PyRepl()
    self_reference = repl.get_runtime_backend("selfref")
    history = [{"role": "user", "content": "seed"}]

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(llm_interface=mock_llm, toolkit=cast(Any, repl.toolset))
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _history in stream:
            pass

    assert isinstance(self_reference, SelfReference)
    assert self_reference.get_agent_instance() is agent
    assert self_reference.list_history_keys() == ["agent"]
    assert repl.namespace.get("self_reference") is None
    assert captured_system_prompt is not None
    assert "[Runtime Primitive Contract]" not in captured_system_prompt
    assert "<tool_best_practices>" in captured_system_prompt
    assert "execute_code" in captured_system_prompt
    assert "<runtime_primitive_contract>" in captured_system_prompt
    assert "Installed primitive packs:" in captured_system_prompt
    assert "- selfref:" in captured_system_prompt
    assert captured_system_prompt.count("runtime.list_primitives()") == 1
    assert (
        captured_system_prompt.count("runtime.list_primitives(contains='<namespace>.')")
        == 1
    )
    assert captured_system_prompt.count("runtime.get_primitive_spec(name)") == 1
    assert (
        captured_system_prompt.count("runtime.list_primitive_specs(contains='...')")
        == 1
    )
    assert _MUST_PROMPT_BLOCK in captured_system_prompt
    assert _MUST_PROMPT_RULE in captured_system_prompt
    assert "Active selfref key: agent" in captured_system_prompt
    assert (
        "Use assistant content for natural-language reasoning and final responses."
        in captured_system_prompt
    )
    assert (
        "Keep tool invocation payloads in the native tool channel."
        in captured_system_prompt
    )
    assert captured_system_prompt.index(
        "<tool_best_practices>"
    ) < captured_system_prompt.index("test agent")
    assert captured_system_prompt.rfind(
        _MUST_PROMPT_BLOCK
    ) > captured_system_prompt.index("test agent")
    assert (
        "You can use the following tools flexibly according to the real case and tool description:"
        not in captured_system_prompt
    )
    assert (
        "For fork results, read status/response/result/memory_key/history_count first; if status is error, inspect error_type/error_message before retrying."
        not in captured_system_prompt
    )
    assert "Mounted primitive summary:" not in captured_system_prompt
    assert "Use memory key" not in captured_system_prompt


@pytest.mark.asyncio
async def test_llm_chat_auto_resolves_self_reference_from_pyrepl_backend() -> None:
    """Decorator should auto-resolve SelfReference from mounted PyRepl runtime backend."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    captured_system_prompt: str | None = None

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal captured_system_prompt
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                captured_system_prompt = maybe_prompt

        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(llm_interface=mock_llm, toolkit=cast(Any, repl.toolset))
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _history in stream:
            pass

    assert self_reference.get_agent_instance() is agent
    assert self_reference.list_history_keys() == ["agent"]
    assert captured_system_prompt is not None
    assert "[Runtime Primitive Contract]" not in captured_system_prompt
    assert "<tool_best_practices>" in captured_system_prompt
    assert "<runtime_primitive_contract>" in captured_system_prompt
    assert captured_system_prompt.count("runtime.get_primitive_spec(name)") == 1
    assert (
        "keep prompt context focused on the selected primitives"
        in captured_system_prompt
    )
    assert "Installed primitive packs:" in captured_system_prompt
    assert "- selfref:" in captured_system_prompt
    assert captured_system_prompt.count("selfref = your agent context") == 1
    assert (
        "Summarize the selected result fields in chat responses."
        in captured_system_prompt
    )
    assert (
        "Treat runtime.selfref.fork.gather_all results as dict[fork_id -> ForkResult] and iterate with .items() or .values()."
        in captured_system_prompt
    )
    assert _MUST_PROMPT_BLOCK in captured_system_prompt
    assert _MUST_PROMPT_RULE in captured_system_prompt
    assert "Active selfref key: agent" in captured_system_prompt
    assert (
        "You can use the following tools flexibly according to the real case and tool description:"
        not in captured_system_prompt
    )
    assert (
        "For fork results, read status/response/result/memory_key/history_count first; if status is error, inspect error_type/error_message before retrying."
        not in captured_system_prompt
    )
    assert "Mounted primitive summary:" not in captured_system_prompt


@pytest.mark.asyncio
async def test_llm_chat_injects_active_selfref_key_for_runtime_context_ops() -> None:
    """Runtime selfref context calls without key should use decorator memory key."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = args

        runtime_toolkit = kwargs["toolkit"]
        execute_tool = next(
            tool
            for tool in runtime_toolkit
            if isinstance(tool, Tool) and tool.name == "execute_code"
        )
        execute_result = await execute_tool.run(
            "runtime.selfref.context.remember('from-selfref')"
        )
        assert "Execution succeeded" in execute_result

        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)
    self_reference.bind_history(
        "agent_main", [{"role": "user", "content": "seed-main"}]
    )
    self_reference.bind_history("other", [{"role": "user", "content": "seed-other"}])

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _updated_history in stream:
            pass

    main_history = self_reference.snapshot_history("agent_main")
    other_history = self_reference.snapshot_history("other")
    assert main_history[0]["role"] == "system"
    assert "from-selfref" in str(main_history[0]["content"])
    assert other_history == [{"role": "user", "content": "seed-other"}]


@pytest.mark.asyncio
async def test_llm_chat_runtime_context_forget_uses_mutation_only_path() -> None:
    """runtime.selfref.context.forget inside a turn should apply through mutations, not live context edits."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    call_counter = 0

    first_response = _make_tool_call_completion(
        "execute_code",
        {
            "code": (
                "snapshot = runtime.selfref.context.inspect()\n"
                "exp_id = snapshot['experiences'][0]['id']\n"
                "runtime.selfref.context.forget(exp_id)"
            )
        },
    )
    second_response = _make_chat_completion("done")

    async def chat_side_effect(**kwargs: Any) -> Any:
        nonlocal call_counter
        response = [first_response, second_response][call_counter]
        call_counter += 1
        return response

    mock_llm.chat = AsyncMock(side_effect=chat_side_effect)

    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)
    self_reference.bind_history("agent_main", [{"role": "user", "content": "seed-main"}])
    self_reference.remember_experience("agent_main", "to-remove")

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
        patch(
            "SimpleLLMFunc.base.ReAct.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
            enable_event=True,
        )
        async def agent(message: str, history=None):
            """test agent"""

        outputs: list[ReactOutput] = []
        async for output in cast(
            AsyncGenerator[ReactOutput, None], agent("hello", history=history)
        ):
            outputs.append(output)

    react_end = next(
        output.event
        for output in outputs
        if isinstance(output, EventYield) and isinstance(output.event, ReactEndEvent)
    )
    assert react_end.final_messages == self_reference.snapshot_history("agent_main")
    assert "to-remove" not in str(react_end.final_messages[0]["content"])


@pytest.mark.asyncio
async def test_llm_chat_runtime_context_compact_rewrites_final_messages() -> None:
    """runtime.selfref.context.compact should replace final context with system + assistant summary."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    call_counter = 0

    first_response = _make_tool_call_completion(
        "execute_code",
        {
            "code": (
                "payload = runtime.selfref.context.compact(\n"
                "    goal='Goal A',\n"
                "    instruction='Instruction A',\n"
                "    discoveries=['Discovery A'],\n"
                "    completed=['Completed A'],\n"
                "    current_status='Ready for the next milestone.',\n"
                "    likely_next_work=['Next A'],\n"
                "    relevant_files_directories=['src/a.py'],\n"
                ")\n"
                "print(payload['assistant_message'])"
            )
        },
    )
    second_response = _make_chat_completion("done")

    async def chat_side_effect(**kwargs: Any) -> Any:
        nonlocal call_counter
        response = [first_response, second_response][call_counter]
        call_counter += 1
        return response

    mock_llm.chat = AsyncMock(side_effect=chat_side_effect)

    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
        patch(
            "SimpleLLMFunc.base.ReAct.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
            enable_event=True,
        )
        async def agent(message: str, history=None):
            """test agent"""

        outputs: list[ReactOutput] = []
        async for output in cast(
            AsyncGenerator[ReactOutput, None], agent("hello", history=history)
        ):
            outputs.append(output)

        react_end = next(
            output.event
            for output in outputs
            if isinstance(output, EventYield)
            and isinstance(output.event, ReactEndEvent)
        )
    assert react_end.final_messages == self_reference.snapshot_history("agent_main")
    assert history == react_end.final_messages
    assert react_end.final_messages[0] == {"role": "system", "content": "test agent"}
    assert react_end.final_messages[1]["role"] == "assistant"
    assert "<context_compaction_summary>" in react_end.final_messages[1]["content"]
    assert "## Goal" in react_end.final_messages[1]["content"]
    assert "Goal A" in react_end.final_messages[1]["content"]
    assert react_end.final_messages[2] == {"role": "assistant", "content": "done"}
    assert len(react_end.final_messages) == 3


@pytest.mark.asyncio
async def test_llm_chat_runtime_context_compact_can_remember_experience() -> None:
    """runtime.selfref.context.compact should also persist remembered experiences into system prompt."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    call_counter = 0

    first_response = _make_tool_call_completion(
        "execute_code",
        {
            "code": (
                "runtime.selfref.context.compact(\n"
                "    goal='Goal B',\n"
                "    instruction='Instruction B',\n"
                "    discoveries=['Discovery B'],\n"
                "    completed=['Completed B'],\n"
                "    current_status='Status B',\n"
                "    likely_next_work=['Next B'],\n"
                "    relevant_files_directories=['src/b.py'],\n"
                "    remember=['Preference B'],\n"
                ")"
            )
        },
    )
    second_response = _make_chat_completion("done")

    async def chat_side_effect(**kwargs: Any) -> Any:
        nonlocal call_counter
        response = [first_response, second_response][call_counter]
        call_counter += 1
        return response

    mock_llm.chat = AsyncMock(side_effect=chat_side_effect)

    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
        patch(
            "SimpleLLMFunc.base.ReAct.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
            enable_event=True,
        )
        async def agent(message: str, history=None):
            """test agent"""

        outputs: list[ReactOutput] = []
        async for output in cast(
            AsyncGenerator[ReactOutput, None], agent("hello", history=history)
        ):
            outputs.append(output)

    react_end = next(
        output.event
        for output in outputs
        if isinstance(output, EventYield) and isinstance(output.event, ReactEndEvent)
    )
    assert react_end.final_messages == self_reference.snapshot_history("agent_main")
    assert history == react_end.final_messages
    assert react_end.final_messages[0]["role"] == "system"
    assert "Preference B" in str(react_end.final_messages[0]["content"])
    assert "<experience>" in str(react_end.final_messages[0]["content"])
    assert react_end.final_messages[1]["role"] == "assistant"
    assert "Goal B" in str(react_end.final_messages[1]["content"])
    assert react_end.final_messages[2] == {"role": "assistant", "content": "done"}
    assert len(react_end.final_messages) == 3


@pytest.mark.asyncio
async def test_llm_chat_runtime_context_compact_applies_before_next_llm_call() -> None:
    """runtime.selfref.context.compact should affect the next same-turn LLM call."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    captured_messages: list[list[dict[str, Any]]] = []
    call_counter = 0

    first_response = _make_tool_call_completion(
        "execute_code",
        {
            "code": (
                "runtime.selfref.context.compact(\n"
                "    goal='Goal C',\n"
                "    instruction='Instruction C',\n"
                "    discoveries=['Discovery C'],\n"
                "    completed=['Completed C'],\n"
                "    current_status='Status C',\n"
                "    likely_next_work=['Next C'],\n"
                "    relevant_files_directories=['src/c.py'],\n"
                ")"
            )
        },
    )
    second_response = _make_chat_completion("done")

    async def chat_side_effect(**kwargs: Any) -> Any:
        nonlocal call_counter
        captured_messages.append(
            copy.deepcopy(cast(list[dict[str, Any]], kwargs["messages"]))
        )
        response = [first_response, second_response][call_counter]
        call_counter += 1
        return response

    mock_llm.chat = AsyncMock(side_effect=chat_side_effect)

    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
        patch(
            "SimpleLLMFunc.base.ReAct.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _updated_history in stream:
            pass

    assert len(captured_messages) == 2
    second_call_messages = captured_messages[1]
    assert second_call_messages[0] == {"role": "system", "content": "test agent"}
    assert second_call_messages[1]["role"] == "assistant"
    assert "<context_compaction_summary>" in str(second_call_messages[1]["content"])
    assert "Goal C" in str(second_call_messages[1]["content"])
    assert len(second_call_messages) == 2


@pytest.mark.asyncio
async def test_llm_chat_selfref_clears_active_react_state_after_run() -> None:
    """Self-reference sync should not leave active ReAct state bound after run completion."""

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    mock_llm.chat = AsyncMock(return_value=_make_chat_completion("done"))

    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)
    observed_state_during_run: list[Any] = []

    async def fake_execute_react_loop_streaming(*args: Any, **kwargs: Any):
        hooks = kwargs.get("hooks")
        state = SimpleNamespace(
            messages=copy.deepcopy(cast(list[dict[str, Any]], kwargs["messages"]))
        )
        if hooks is not None:
            await hooks.on_run_start(state)
            observed_state_during_run.append(self_reference._get_active_react_state())
            await hooks.before_finalize(state)
            hooks.close()
        yield "done", state.messages

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=[]),
        )

        async for _content, _updated_history in stream:
            pass

    assert observed_state_during_run
    assert observed_state_during_run[0] is not None
    assert self_reference._get_active_react_state() is None


@pytest.mark.asyncio
async def test_llm_chat_fork_uses_isolated_pyrepl_session_toolkit() -> None:
    """Forked child should run with a cloned PyRepl toolkit session."""

    observed_toolkits: list[Any] = []

    async def fake_execute_react_loop_streaming(*args: Any, **kwargs: Any):
        _ = args
        observed_toolkits.append(kwargs.get("toolkit"))
        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    root_repl = PyRepl()
    self_reference = _builtin_self_reference(root_repl)

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, root_repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        self_reference.bind_history("agent_main", [])
        await self_reference.instance.fork("hello")

    assert observed_toolkits
    toolkit_used = observed_toolkits[-1]
    assert isinstance(toolkit_used, list)

    execute_tool = next(
        tool
        for tool in toolkit_used
        if isinstance(tool, Tool) and tool.name == "execute_code"
    )
    child_repl = getattr(execute_tool.func, "__self__", None)

    assert isinstance(child_repl, PyRepl)
    assert child_repl is not root_repl
    assert child_repl.get_runtime_backend("selfref") is self_reference
    assert child_repl._closed is True
    assert root_repl._closed is False


@pytest.mark.asyncio
async def test_llm_chat_fork_clones_custom_pyrepl_pack_primitives() -> None:
    """Forked child should preserve first-class custom PrimitivePack installs."""

    observed_toolkits: list[Any] = []

    async def fake_execute_react_loop_streaming(*args: Any, **kwargs: Any):
        _ = args
        observed_toolkits.append(kwargs.get("toolkit"))
        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    root_repl = PyRepl()
    self_reference = _builtin_self_reference(root_repl)

    constants_pack = root_repl.pack(
        "constants",
        backend={"project": "SimpleLLMFunc", "branch": "feat/better-primitive-dev"},
    )

    @constants_pack.primitive("get")
    def constants_get(ctx, key: str):
        """
        Use: Read one value from constants backend.
        Input: `key: str`.
        Output: `str | None`.
        Best Practices:
        - Prefer reading a single key per call.
        """
        backend = ctx.backend
        if not isinstance(backend, dict):
            raise RuntimeError("constants backend must be a dict")
        return backend.get(key)

    root_repl.install_pack(constants_pack)

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, root_repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        self_reference.bind_history("agent_main", [])
        await self_reference.instance.fork("hello")

    assert observed_toolkits
    toolkit_used = observed_toolkits[-1]
    execute_tool = next(
        tool
        for tool in toolkit_used
        if isinstance(tool, Tool) and tool.name == "execute_code"
    )
    child_repl = getattr(execute_tool.func, "__self__", None)

    assert isinstance(child_repl, PyRepl)
    assert child_repl is not root_repl
    assert child_repl.get_runtime_backend("constants") == {
        "project": "SimpleLLMFunc",
        "branch": "feat/better-primitive-dev",
    }
    assert "constants.get" in child_repl.list_primitives()
    assert child_repl.list_installed_packs() == ["constants", "selfref"]


@pytest.mark.asyncio
async def test_llm_chat_selfref_fork_spawn_preserves_langfuse_trace_context() -> None:
    """Child agent spans should stay on the same trace and nest under execute_code."""

    fork_code = (
        "handle = runtime.selfref.fork.spawn('child task')\n"
        "results = runtime.selfref.fork.gather_all(handle)\n"
        "print(results[handle['fork_id']]['response'])"
    )

    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)
    tracker = _TrackingLangfuseClient()
    child_messages_seen: list[list[dict[str, Any]]] = []

    async def fake_chat(messages: list[dict[str, Any]], tools=None, **kwargs: Any):
        _ = kwargs

        if self_reference._get_active_fork_id() is not None:
            child_messages_seen.append(copy.deepcopy(messages))
            return _make_chat_completion("child-ok")

        has_tool_result = any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in messages
        )
        if tools and not has_tool_result:
            return _make_tool_call_completion(
                "execute_code",
                {"code": fork_code},
            )

        return _make_chat_completion("parent-ok")

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    mock_llm.chat.side_effect = fake_chat

    history = [{"role": "user", "content": "seed"}]

    class _FakeOtelSpanContext:
        def __init__(self, trace_id: str, span_id: str) -> None:
            self.trace_id = trace_id
            self.span_id = span_id
            self.is_valid = True

    class _FakeOtelSpan:
        def __init__(self, trace_id: str, span_id: str) -> None:
            self._span_context = _FakeOtelSpanContext(trace_id, span_id)

        def get_span_context(self):
            return self._span_context

        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key: str, value: Any) -> None:
            _ = (key, value)

    def _current_otel_span() -> Any:
        current_trace_id = tracker.get_current_trace_id()
        current_observation_id = tracker.get_current_observation_id()
        if current_trace_id and current_observation_id:
            return _FakeOtelSpan(current_trace_id, current_observation_id)
        from SimpleLLMFunc.observability.langfuse_client import otel_trace_api

        return otel_trace_api.INVALID_SPAN

    with (
        patch.object(
            shared_langfuse_client,
            "start_as_current_observation",
            side_effect=tracker.start_as_current_observation,
        ),
        patch.object(
            shared_langfuse_client,
            "get_current_trace_id",
            side_effect=tracker.get_current_trace_id,
        ),
        patch.object(
            shared_langfuse_client,
            "get_current_observation_id",
            side_effect=tracker.get_current_observation_id,
        ),
        patch.object(
            shared_langfuse_client,
            "create_trace_id",
            side_effect=tracker.create_trace_id,
        ),
        patch(
            "SimpleLLMFunc.observability.langfuse_client.otel_trace_api.get_current_span",
            side_effect=_current_otel_span,
        ),
    ):
        trace_token = set_langfuse_trace_context({"trace_id": "trace-root"})
        try:

            @llm_chat(
                llm_interface=mock_llm,
                toolkit=cast(Any, repl.toolset),
                self_reference=self_reference,
                self_reference_key="agent_main",
                max_tool_calls=None,
            )
            async def agent(message: str, history=None):
                """test agent"""

            stream = cast(
                AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
                agent("root task", history=history),
            )

            async for _content, _updated_history in stream:
                pass
        finally:
            reset_langfuse_trace_context(trace_token)

    parent_chat_span = next(
        record
        for record in tracker.records
        if record.get("name") == "agent_chat_call"
        and isinstance(record.get("input"), dict)
        and record["input"].get("message") == "root task"
    )
    execute_code_span = next(
        record
        for record in tracker.records
        if record.get("as_type") == "tool" and record.get("name") == "execute_code"
    )
    child_chat_span = next(
        (
            record
            for record in tracker.records
            if record.get("name") == "agent_chat_call"
            and record.get("parent_span_id") == execute_code_span["span_id"]
        ),
        None,
    )

    assert parent_chat_span["trace_id"] == "trace-root"
    assert execute_code_span["trace_id"] == "trace-root"
    assert execute_code_span["parent_span_id"] == parent_chat_span["span_id"]
    if child_chat_span is not None:
        assert child_chat_span["trace_id"] == "trace-root"
        assert child_chat_span["parent_span_id"] == execute_code_span["span_id"]

    assert child_messages_seen
    assert any(
        len(candidate) >= 4
        and candidate[0].get("role") == "system"
        and "test agent" in str(candidate[0].get("content", ""))
        and candidate[1] == {"role": "user", "content": "seed"}
        and candidate[2] == {"role": "user", "content": "message: root task"}
        and candidate[3].get("role") == "user"
        and "You are now already a forked subagent."
        in str(candidate[3].get("content", ""))
        and "you have no need to care about whether the previous fork was correct or not"
        in str(candidate[3].get("content", ""))
        and "because it has already succeeded." in str(candidate[3].get("content", ""))
        and "the only thing you are required to do is:"
        in str(candidate[3].get("content", ""))
        and "child task" in str(candidate[3].get("content", ""))
        and str(candidate[3].get("content", "")).endswith(
            "Follow the instructions above to finish this task."
        )
        and not any(
            message.get("role") == "assistant" and message.get("tool_calls")
            for message in candidate[:4]
            if isinstance(message, dict)
        )
        for candidate in child_messages_seen
    )


@pytest.mark.asyncio
async def test_llm_chat_sets_explicit_langfuse_trace_name() -> None:
    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    tracker = _TrackingLangfuseClient()

    async def fake_chat(messages: list[dict[str, Any]], tools=None, **kwargs: Any):
        _ = messages, tools, kwargs
        return _make_chat_completion("ok")

    mock_llm.chat.side_effect = fake_chat

    with (
        patch.object(
            shared_langfuse_client,
            "start_as_current_observation",
            side_effect=tracker.start_as_current_observation,
        ),
        patch.object(
            shared_langfuse_client,
            "get_current_observation_id",
            side_effect=tracker.get_current_observation_id,
        ),
        patch(
            "SimpleLLMFunc.observability.langfuse_client.otel_trace_api.get_current_span",
            side_effect=tracker._active_observation,
        ),
    ):

        @llm_chat(llm_interface=mock_llm)
        async def core_agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
            core_agent("hello", history=[]),
        )

        async for _content, _updated_history in stream:
            pass

    parent_chat_span = next(
        record
        for record in tracker.records
        if record.get("name") == "core_agent_chat_call"
    )

    assert parent_chat_span["attributes"]["langfuse.trace.name"] == "core_agent"


@pytest.mark.asyncio
async def test_llm_chat_propagates_trace_name_to_child_observations() -> None:
    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    tracker = _TrackingLangfuseClient()

    async def fake_chat(messages: list[dict[str, Any]], tools=None, **kwargs: Any):
        _ = messages, tools, kwargs
        with tracker.start_as_current_observation(
            as_type="generation",
            name="react_generation",
        ):
            return _make_chat_completion("ok")

    mock_llm.chat.side_effect = fake_chat

    with (
        patch.object(
            shared_langfuse_client,
            "start_as_current_observation",
            side_effect=tracker.start_as_current_observation,
        ),
        patch.object(
            shared_langfuse_client,
            "get_current_observation_id",
            side_effect=tracker.get_current_observation_id,
        ),
        patch(
            "SimpleLLMFunc.observability.langfuse_client.otel_trace_api.get_current_span",
            side_effect=tracker._active_observation,
        ),
        patch(
            "SimpleLLMFunc.observability.langfuse_client.langfuse_propagate_attributes",
            side_effect=tracker.propagate_attributes,
        ),
    ):

        @llm_chat(llm_interface=mock_llm)
        async def core_agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
            core_agent("hello", history=[]),
        )

        async for _content, _updated_history in stream:
            pass

    generation_span = next(
        record for record in tracker.records if record.get("name") == "react_generation"
    )

    assert generation_span["trace_name"] == "core_agent"
    assert tracker.trace_names_by_trace_id[generation_span["trace_id"]] == "core_agent"


@pytest.mark.asyncio
async def test_llm_chat_forwards_reasoning_kwargs_to_openai_responses_interface() -> (
    None
):
    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-responses-llm-chat-reasoning",
    )
    llm = OpenAIResponsesCompatible(
        api_key_pool=key_pool,
        model_name="gpt-test",
        base_url="https://example.com/v1",
        max_retries=1,
        retry_delay=0.0,
    )
    llm.token_bucket.acquire = AsyncMock(return_value=True)

    create_mock = AsyncMock(return_value=_make_text_response("ok"))
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    @llm_chat(llm_interface=llm, reasoning={"effort": "xhigh"})
    async def core_agent(message: str, history=None):
        """test agent"""

    stream = cast(
        AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
        core_agent("hello", history=[]),
    )

    async for _content, _updated_history in stream:
        pass

    request_kwargs = create_mock.await_args.kwargs
    assert request_kwargs["reasoning"] == {"effort": "xhigh"}


@pytest.mark.asyncio
async def test_llm_chat_maps_docstring_system_prompt_to_responses_instructions() -> (
    None
):
    key_pool = APIKeyPool(
        api_keys=["test-key"],
        provider_id="test-responses-llm-chat-instructions",
    )
    llm = OpenAIResponsesCompatible(
        api_key_pool=key_pool,
        model_name="gpt-test",
        base_url="https://example.com/v1",
        max_retries=1,
        retry_delay=0.0,
    )
    llm.token_bucket.acquire = AsyncMock(return_value=True)

    create_mock = AsyncMock(return_value=_make_text_response("ok"))
    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    llm._get_or_create_client = AsyncMock(return_value=fake_client)

    @llm_chat(llm_interface=llm)
    async def core_agent(message: str, history=None):
        """You are helpful."""

    stream = cast(
        AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
        core_agent("hello", history=[]),
    )

    async for _content, _updated_history in stream:
        pass

    request_kwargs = create_mock.await_args.kwargs
    assert "You are helpful." in request_kwargs["instructions"]
    assert "<must_principles>" in request_kwargs["instructions"]
    assert all(item.get("role") != "system" for item in request_kwargs["input"])


@pytest.mark.asyncio
async def test_llm_chat_fork_inherits_parent_template_params_for_docstring_rendering() -> (
    None
):
    """Forked child should inherit parent template params such as environment_block."""

    self_reference = SelfReference()
    self_reference.bind_history(
        "agent_main",
        [
            {"role": "user", "content": "seed"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_fork_1",
                        "type": "function",
                        "function": {
                            "name": "execute_code",
                            "arguments": "runtime.selfref.fork.spawn('child task')",
                        },
                    }
                ],
            },
        ],
    )

    captured_template_params: dict[str, Any] = {}

    async def fake_agent(message: str, history=None, _template_params=None):
        captured_template_params.update(cast(dict[str, Any], _template_params or {}))
        return (
            "done",
            [
                *(history or []),
                {"role": "assistant", "content": "child done"},
            ],
        )

    self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
    toolkit_override = ["tool-a"]
    toolkit_token = self_reference._set_active_runtime_toolkit(toolkit_override)
    template_token = self_reference._set_active_template_params(
        {"environment_block": "ENV: /tmp/workspace"}
    )
    memory_token = self_reference._set_active_memory_key("agent_main")
    try:
        completed = await self_reference.instance.fork(
            "child task", include_history=True
        )
    finally:
        self_reference._reset_active_memory_key(memory_token)
        self_reference._reset_active_template_params(template_token)
        self_reference._reset_active_runtime_toolkit(toolkit_token)

    assert completed["status"] == "completed"
    assert captured_template_params[SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM] != ""
    assert (
        captured_template_params[SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM]
        == toolkit_override
    )
    assert (
        captured_template_params[SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM]
        == "child task"
    )
    assert captured_template_params["environment_block"] == "ENV: /tmp/workspace"


def test_self_reference_active_template_params_do_not_deepcopy_uncopyable_objects() -> (
    None
):
    """Active template params should tolerate runtime objects like toolkit overrides."""

    self_reference = SelfReference()
    lock = threading.RLock()

    token = self_reference._set_active_template_params(
        {
            "environment_block": "ENV: /tmp/workspace",
            SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM: [lock],
        }
    )
    try:
        merged = self_reference._build_fork_template_params(
            None,
            "fork_key",
            None,
            "child task",
        )
    finally:
        self_reference._reset_active_template_params(token)

    assert merged["environment_block"] == "ENV: /tmp/workspace"
    assert merged[SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM] == [lock]


@pytest.mark.asyncio
async def test_llm_chat_event_mode_merges_self_reference_memory_mutations() -> None:
    """Event mode should merge memory-handle edits into final history."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[ReactOutput, None]:
        _ = (args, kwargs)

        self_reference.memory["agent_main"].append(
            {"role": "user", "content": "[plan] keep me"}
        )

        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=datetime.now(timezone.utc),
                trace_id="trace-test",
                func_name="agent",
                iteration=1,
                final_response="done",
                final_messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "seed"},
                    {"role": "user", "content": "message: hello"},
                    {"role": "assistant", "content": "done"},
                ],
                total_iterations=1,
                total_execution_time=0.01,
                total_tool_calls=1,
                total_llm_calls=1,
            )
        )

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            enable_event=True,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[ReactOutput, None],
            agent("hello", history=history),
        )

        outputs: list[ReactOutput] = []
        async for output in stream:
            outputs.append(output)

    assert len(outputs) == 1
    final_output = outputs[0]
    assert isinstance(final_output, EventYield)
    assert isinstance(final_output.event, ReactEndEvent)

    final_history = final_output.event.final_messages
    assert final_history[0].get("role") == "system"
    assert final_history[0].get("content") == "test agent"
    assert final_history[1:] == [
        {"role": "user", "content": "seed"},
        {"role": "user", "content": "[plan] keep me"},
        {"role": "user", "content": "message: hello"},
        {"role": "assistant", "content": "done"},
    ]
    assert history == final_history
    assert self_reference.snapshot_history("agent_main") == final_history


@pytest.mark.asyncio
async def test_llm_chat_event_mode_ignores_fork_react_end_for_history_merge() -> None:
    """Fork-scoped ReactEndEvent should not be merged into main memory."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()

    child_final_messages = [
        {"role": "system", "content": "child system"},
        {"role": "user", "content": "child prompt"},
        {"role": "assistant", "content": "child done"},
    ]

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[ReactOutput, None]:
        _ = (args, kwargs)

        self_reference.memory["agent_main"].append(
            {"role": "user", "content": "[plan] keep me"}
        )

        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=datetime.now(timezone.utc),
                trace_id="trace-child",
                func_name="agent",
                iteration=1,
                final_response="child done",
                final_messages=child_final_messages,
                total_iterations=1,
                total_execution_time=0.01,
                total_tool_calls=0,
                total_llm_calls=1,
            ),
            origin=EventOrigin(
                session_id="trace-main",
                agent_call_id="agent-main",
                event_seq=1,
                fork_id="fork_1",
                fork_depth=1,
                source_memory_key="agent_main",
                memory_key="agent_main::fork::1",
            ),
        )

        yield EventYield(
            event=ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=datetime.now(timezone.utc),
                trace_id="trace-main",
                func_name="agent",
                iteration=1,
                final_response="done",
                final_messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "seed"},
                    {"role": "user", "content": "message: hello"},
                    {"role": "assistant", "content": "done"},
                ],
                total_iterations=1,
                total_execution_time=0.01,
                total_tool_calls=1,
                total_llm_calls=1,
            )
        )

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            enable_event=True,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[ReactOutput, None],
            agent("hello", history=history),
        )

        outputs: list[ReactOutput] = []
        async for output in stream:
            outputs.append(output)

    assert len(outputs) == 2

    child_output = outputs[0]
    assert isinstance(child_output, EventYield)
    assert isinstance(child_output.event, ReactEndEvent)
    assert child_output.event.final_messages == child_final_messages

    final_output = outputs[-1]
    assert isinstance(final_output, EventYield)
    assert isinstance(final_output.event, ReactEndEvent)

    final_history = final_output.event.final_messages
    assert final_history[0].get("role") == "system"
    assert final_history[0].get("content") == "test agent"
    assert final_history[1:] == [
        {"role": "user", "content": "seed"},
        {"role": "user", "content": "[plan] keep me"},
        {"role": "user", "content": "message: hello"},
        {"role": "assistant", "content": "done"},
    ]
    assert not any(item.get("content") == "child prompt" for item in final_history)
    assert history == final_history
    assert self_reference.snapshot_history("agent_main") == final_history


@pytest.mark.asyncio
async def test_llm_chat_non_event_mode_merges_self_reference_memory_mutations() -> None:
    """Tuple mode should also merge memory-handle edits into updated history."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (args, kwargs)

        self_reference.memory["agent_main"].append(
            {"role": "user", "content": "[plan] keep me"}
        )

        yield (
            "done",
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "seed"},
                {"role": "user", "content": "message: hello"},
                {"role": "assistant", "content": "done"},
            ],
        )

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        outputs: list[tuple[str, list[dict[str, Any]]]] = []
        async for output in stream:
            outputs.append(output)

    assert len(outputs) == 1
    output_content, output_history = outputs[0]
    assert output_content == "done"
    assert output_history[0].get("role") == "system"
    assert output_history[0].get("content") == "test agent"
    assert output_history[1:] == [
        {"role": "user", "content": "seed"},
        {"role": "user", "content": "[plan] keep me"},
        {"role": "user", "content": "message: hello"},
        {"role": "assistant", "content": "done"},
    ]
    assert history == output_history
    assert self_reference.snapshot_history("agent_main") == output_history


@pytest.mark.asyncio
async def test_llm_chat_uses_function_name_as_default_self_reference_key() -> None:
    """When no key is provided, llm_chat should use function name."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()
    captured_system_prompt: str | None = None

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal captured_system_prompt
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                captured_system_prompt = maybe_prompt

        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(llm_interface=mock_llm, self_reference=self_reference)
        async def agent(message: str, history=None):
            """test agent"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _history in stream:
            pass

    assert self_reference.list_history_keys() == ["agent"]
    assert self_reference.snapshot_history("agent") == history
    assert captured_system_prompt is not None
    assert "test agent" in captured_system_prompt
    assert _MUST_PROMPT_BLOCK in captured_system_prompt
    assert _MUST_PROMPT_RULE in captured_system_prompt
    assert captured_system_prompt.strip().endswith("</must_principles>")
    assert "[Runtime Primitive Contract]" not in captured_system_prompt


@pytest.mark.asyncio
async def test_llm_chat_persists_runtime_system_prompt_across_turns() -> None:
    """Runtime system prompt edits should be reused on subsequent turns."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    self_reference = SelfReference()
    observed_system_prompts: list[str] = []
    call_count = 0

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal call_count
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                observed_system_prompts.append(maybe_prompt)

        current_call = call_count
        call_count += 1

        if current_call == 0:
            self_reference.memory["agent_main"].set_system_prompt("runtime system")

        yield (
            f"done-{current_call + 1}",
            [
                *messages,
                {"role": "assistant", "content": f"done-{current_call + 1}"},
            ],
        )

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """docstring system"""

        first_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 1", history=history),
        )
        async for _content, _history in first_stream:
            pass

        second_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 2", history=history),
        )
        async for _content, _history in second_stream:
            pass

    assert call_count == 2
    assert len(observed_system_prompts) == 2
    assert "docstring system" in observed_system_prompts[0]
    assert "runtime system" in observed_system_prompts[1]
    assert _MUST_PROMPT_BLOCK in observed_system_prompts[0]
    assert _MUST_PROMPT_BLOCK in observed_system_prompts[1]
    assert "[Runtime Primitive Contract]" not in observed_system_prompts[0]
    assert "[Runtime Primitive Contract]" not in observed_system_prompts[1]
    assert history[0] == {"role": "system", "content": "runtime system"}
    assert self_reference.snapshot_history("agent_main")[0] == {
        "role": "system",
        "content": "runtime system",
    }


@pytest.mark.asyncio
async def test_append_system_prompt_persists_without_contract_pollution() -> None:
    """append_system_prompt should persist durable text, not runtime contract block."""

    history: list[dict[str, Any]] = []
    self_reference = SelfReference()
    observed_system_prompts: list[str] = []
    call_count = 0

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal call_count
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                observed_system_prompts.append(maybe_prompt)

        if call_count == 0:
            self_reference.memory["agent_main"].append_system_prompt("Preference A")

        call_count += 1
        yield (
            f"done-{call_count}",
            [
                *messages,
                {"role": "assistant", "content": f"done-{call_count}"},
            ],
        )

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """docstring system"""

        first_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 1", history=history),
        )
        async for _content, _history in first_stream:
            pass

        second_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 2", history=history),
        )
        async for _content, _history in second_stream:
            pass

    assert call_count == 2
    assert len(observed_system_prompts) == 2
    assert "Preference A" in observed_system_prompts[1]
    assert _MUST_PROMPT_BLOCK in observed_system_prompts[0]
    assert _MUST_PROMPT_BLOCK in observed_system_prompts[1]
    assert "[Runtime Primitive Contract]" not in observed_system_prompts[1]

    persisted_system_prompt = self_reference.memory["agent_main"].get_system_prompt()
    assert persisted_system_prompt == "docstring system\nPreference A"


@pytest.mark.asyncio
async def test_llm_chat_deduplicates_runtime_primitive_contract_prompt() -> None:
    """Runtime guidance should stay deduplicated in Tool Best Practices."""

    history: list[dict[str, Any]] = [{"role": "user", "content": "seed"}]
    repl = PyRepl()
    self_reference = _builtin_self_reference(repl)
    observed_system_prompts: list[str] = []

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = args
        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                observed_system_prompts.append(maybe_prompt)
        yield "ok", messages

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            toolkit=cast(Any, repl.toolset),
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """docstring system"""

        first_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 1", history=history),
        )
        async for _content, _history in first_stream:
            pass

        second_stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello 2", history=history),
        )
        async for _content, _history in second_stream:
            pass

    assert len(observed_system_prompts) == 2
    assert observed_system_prompts[0].count("[Runtime Primitive Contract]") == 0
    assert observed_system_prompts[1].count("[Runtime Primitive Contract]") == 0
    assert observed_system_prompts[0].count("<tool_best_practices>") == 1
    assert observed_system_prompts[1].count("<tool_best_practices>") == 1
    assert observed_system_prompts[0].count("<runtime_primitive_contract>") == 1
    assert observed_system_prompts[1].count("<runtime_primitive_contract>") == 1
    assert observed_system_prompts[0].count(_MUST_PROMPT_BLOCK) == 1
    assert observed_system_prompts[1].count(_MUST_PROMPT_BLOCK) == 1


@pytest.mark.asyncio
async def test_llm_chat_seeds_system_prompt_into_empty_self_reference_memory() -> None:
    """Empty memory should contain seeded system prompt before first tool run."""

    history: list[dict[str, Any]] = []
    self_reference = SelfReference()
    observed_memory_lengths: list[int] = []
    observed_first_roles: list[str | None] = []

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = args
        memory_snapshot = self_reference.memory["agent_main"].all()
        observed_memory_lengths.append(len(memory_snapshot))
        if memory_snapshot:
            observed_first_roles.append(memory_snapshot[0].get("role"))
        else:
            observed_first_roles.append(None)
        yield "ok", kwargs["messages"]

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(
            llm_interface=mock_llm,
            self_reference=self_reference,
            self_reference_key="agent_main",
        )
        async def agent(message: str, history=None):
            """docstring system"""

        stream = cast(
            AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
            agent("hello", history=history),
        )

        async for _content, _history in stream:
            pass


@pytest.mark.asyncio
async def test_llm_chat_renders_docstring_template_params_into_system_prompt() -> None:
    """llm_chat should render _template_params into the docstring system prompt."""

    captured_system_prompt: str | None = None

    async def fake_execute_react_loop_streaming(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        nonlocal captured_system_prompt
        _ = args

        messages = kwargs["messages"]
        if messages:
            maybe_prompt = messages[0].get("content")
            if isinstance(maybe_prompt, str):
                captured_system_prompt = maybe_prompt

        yield "ok", messages

    async def passthrough_process_chat_response_stream(
        response_stream: AsyncGenerator[tuple[str, list[dict[str, Any]]], None],
        return_mode: str,
        messages: list[dict[str, Any]],
        func_name: str,
        stream: bool,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        _ = (return_mode, messages, func_name, stream)
        async for response, updated_history in response_stream:
            yield response, updated_history

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.execute_react_loop_streaming",
            new=fake_execute_react_loop_streaming,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.process_chat_response_stream",
            new=passthrough_process_chat_response_stream,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_chat_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_chat(llm_interface=mock_llm)
        async def agent(message: str, history=None):
            """Workspace: {workspace_dir}"""

        stream = cast(
            AsyncGenerator[tuple[Any, list[dict[str, Any]]], None],
            agent(
                "hello",
                history=[],
                _template_params={"workspace_dir": "/tmp/chat-workspace"},
            ),
        )

        async for _content, _history in stream:
            pass

    assert captured_system_prompt is not None
    assert "Workspace: /tmp/chat-workspace" in captured_system_prompt
    assert "{workspace_dir}" not in captured_system_prompt
