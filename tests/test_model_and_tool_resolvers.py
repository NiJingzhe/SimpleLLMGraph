from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Coroutine, NamedTuple, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from SimpleLLMFunc.context.ir import (
    AssistantMessage,
    Completion,
    CompletionChoice,
    FinishReason,
    Request,
    ToolResult,
    UserMessage,
)
from SimpleLLMFunc.event import EventView, UserMessageEvent
from SimpleLLMFunc.context.ir._enums import ReasoningEffort
from SimpleLLMFunc.interface import APIKeyPool, LLM_Interface
from SimpleLLMFunc.loop import (
    CancellationToken,
    CompiledContext,
    Effect,
    EventSnapshot,
    ExecutionMode,
    FunctionTool,
    ModelCallEffect,
    ModelCallResolution,
    ModelCallResolver,
    ResolutionStatus,
    RunJournal,
    Runtime,
    Step,
    ToolCallEffect,
    ToolCallResolution,
    ToolCallResolver,
    tool,
)


class FakeLLM(LLM_Interface):
    def __init__(self, completion: Completion) -> None:
        super().__init__(APIKeyPool(["test"], "fake-loop-tests"), "fake-model")
        self.completion = completion
        self.requests: list[Request] = []
        self.cancellations: list[CancellationToken | None] = []

    async def chat(
        self,
        request: Request,
        *,
        trace_id: str | None = None,
        timeout: int | None = 30,
        cancellation: CancellationToken | None = None,
    ) -> Completion:
        del trace_id, timeout
        self.requests.append(request)
        self.cancellations.append(cancellation)
        return self.completion

    async def chat_stream(
        self,
        request: Request,
        *,
        trace_id: str | None = None,
        timeout: int | None = 30,
        cancellation: CancellationToken | None = None,
    ):
        del request, trace_id, timeout, cancellation
        if False:
            yield


def completion(content: str = "done") -> Completion:
    return Completion(
        id="completion-1",
        created=1,
        model="fake-model",
        choices=[
            CompletionChoice(
                index=0,
                message=AssistantMessage(content=content),
                finish_reason=FinishReason.STOP,
            )
        ],
    )


def empty_events(run_id: str = "tool-test") -> EventView:
    return EventView.create(run_id, 0, [])


def model_effect() -> ModelCallEffect:
    context = CompiledContext.create(
        request=Request(
            model="fake-model",
            messages=[UserMessage(content="hello")],
        ),
        source_revision=0,
        compiler_revision="test@1",
    )
    return ModelCallEffect(
        id="effect-model",
        attempt=1,
        idempotency_key="run-1:model:1",
        context=context,
    )


@pytest.mark.asyncio
async def test_model_resolver_calls_llm_and_records_trace() -> None:
    llm = FakeLLM(completion())
    resolver = ModelCallResolver(llm)

    resolution = await resolver.resolve(model_effect(), CancellationToken())

    assert isinstance(resolution, ModelCallResolution)
    assert resolution.status is ResolutionStatus.COMPLETED
    assert resolution.completion.choices[0].message.content == "done"
    assert resolution.trace.actual_model == "fake-model"
    assert resolution.trace.context_digest == model_effect().context.digest
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_model_resolver_passes_and_honors_the_shared_cancellation_token() -> None:
    class BlockingLLM(FakeLLM):
        async def chat(
            self,
            request: Request,
            *,
            trace_id: str | None = None,
            timeout: int | None = 30,
            cancellation: CancellationToken | None = None,
        ) -> Completion:
            del request, trace_id, timeout
            assert cancellation is not None
            self.cancellations.append(cancellation)
            await cancellation.wait()
            raise asyncio.CancelledError

    llm = BlockingLLM(completion())
    resolver = ModelCallResolver(llm)
    cancellation = CancellationToken()
    task = asyncio.create_task(resolver.resolve(model_effect(), cancellation))

    await asyncio.sleep(0)
    cancellation.cancel()
    resolution = await task

    assert resolution.status is ResolutionStatus.CANCELLED
    assert llm.cancellations == [cancellation]


@pytest.mark.asyncio
async def test_model_resolver_propagates_unrelated_task_cancellation() -> None:
    class CancelledLLM(FakeLLM):
        async def chat(
            self,
            request: Request,
            *,
            trace_id: str | None = None,
            timeout: int | None = 30,
            cancellation: CancellationToken | None = None,
        ) -> Completion:
            del request, trace_id, timeout, cancellation
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ModelCallResolver(CancelledLLM(completion())).resolve(
            model_effect(),
            CancellationToken(),
        )


@pytest.mark.asyncio
async def test_model_resolver_honors_pre_cancellation_and_model_mismatch() -> None:
    llm = FakeLLM(completion())
    resolver = ModelCallResolver(llm)
    cancellation = CancellationToken()
    cancellation.cancel()

    cancelled = await resolver.resolve(model_effect(), cancellation)
    assert cancelled.status is ResolutionStatus.CANCELLED
    assert not llm.requests

    mismatch_context = CompiledContext.create(
        request=Request(
            model="different-model",
            messages=[UserMessage(content="hello")],
        ),
        source_revision=0,
        compiler_revision="test@1",
    )
    mismatch = model_effect().model_copy(update={"context": mismatch_context})

    with pytest.raises(ValueError, match="model assertion"):
        await resolver.resolve(mismatch, CancellationToken())

    forged_context = model_effect().context.model_copy(
        update={"request_json": mismatch_context.request_json}
    )
    forged = model_effect().model_copy(update={"context": forged_context})
    with pytest.raises(ValueError, match="request_digest"):
        await resolver.resolve(forged, CancellationToken())


@pytest.mark.asyncio
async def test_tool_resolver_success_failure_unknown_and_cancellation() -> None:
    calls: list[tuple[str, bool]] = []

    @tool
    async def echo(value: str, fail: bool = False) -> dict[str, str]:
        """Echo a value."""

        calls.append((value, fail))
        if fail:
            raise RuntimeError("tool failed")
        return {"echo": value}

    resolver = ToolCallResolver([echo])

    effect = ToolCallEffect.from_call(
        id="effect-tool",
        attempt=1,
        idempotency_key="run-1:tool:1",
        tool_call_id="call-1",
        name="echo",
        arguments={"value": "hello"},
        events=empty_events(),
    )
    success = await resolver.resolve(effect, CancellationToken())
    assert isinstance(success, ToolCallResolution)
    assert success.status is ResolutionStatus.COMPLETED
    assert success.value == {"echo": "hello"}
    assert calls == [("hello", False)]

    failing = ToolCallEffect.from_call(
        id="effect-fail",
        attempt=1,
        idempotency_key="run-1:tool:2",
        tool_call_id="call-2",
        name="echo",
        arguments={"value": "hello", "fail": True},
        events=empty_events(),
    )
    failed = await resolver.resolve(failing, CancellationToken())
    assert failed.status is ResolutionStatus.FAILED
    assert failed.error is not None
    assert failed.error.message == "tool failed"

    unknown = ToolCallEffect.from_call(
        id="effect-unknown",
        attempt=1,
        idempotency_key="run-1:tool:3",
        tool_call_id="call-3",
        name="missing",
        arguments={},
        events=empty_events(),
    )
    denied = await resolver.resolve(unknown, CancellationToken())
    assert denied.status is ResolutionStatus.DENIED

    cancellation = CancellationToken()
    cancellation.cancel()
    cancelled = await resolver.resolve(effect, cancellation)
    assert cancelled.status is ResolutionStatus.CANCELLED


def test_tool_effect_canonicalizes_and_validates_arguments() -> None:
    effect = ToolCallEffect.from_call(
        id="effect-tool",
        attempt=1,
        idempotency_key="key",
        tool_call_id="call-1",
        name="echo",
        arguments={"b": 2, "a": 1},
        events=empty_events(),
    )

    assert effect.arguments == {"a": 1, "b": 2}
    assert json.loads(effect.arguments_json) == {"a": 1, "b": 2}

    with pytest.raises(ValueError, match="JSON object"):
        ToolCallEffect(
            id="bad",
            attempt=1,
            idempotency_key="bad",
            tool_call_id="call",
            name="echo",
            arguments_json="[]",
            event_snapshot=EventSnapshot.create(empty_events()),
        )

    with pytest.raises(ValueError, match="valid JSON"):
        ToolCallEffect(
            id="bad-json",
            attempt=1,
            idempotency_key="bad-json",
            tool_call_id="call",
            name="echo",
            arguments_json="{",
            event_snapshot=EventSnapshot.create(empty_events()),
        )

    with pytest.raises(ValueError, match="non-JSON constant"):
        ToolCallEffect(
            id="bad-constant",
            attempt=1,
            idempotency_key="bad-constant",
            tool_call_id="call",
            name="echo",
            arguments_json='{"value":NaN}',
            event_snapshot=EventSnapshot.create(empty_events()),
        )

    with pytest.raises(ValueError, match="JSON compliant"):
        ToolCallEffect.from_call(
            id="bad-float",
            attempt=1,
            idempotency_key="bad-float",
            tool_call_id="call",
            name="echo",
            arguments={"value": float("inf")},
            events=empty_events(),
        )

    invalid = ToolCallEffect.model_construct(
        id="constructed",
        attempt=1,
        idempotency_key="constructed",
        tool_call_id="call",
        name="echo",
        arguments_json="[]",
    )
    with pytest.raises(AssertionError, match="not an object"):
        _ = invalid.arguments


@pytest.mark.asyncio
async def test_builtin_resolvers_require_their_concrete_effect_types() -> None:
    effect = Effect(id="base", attempt=1, idempotency_key="base")

    with pytest.raises(TypeError, match="ModelCallEffect"):
        await ModelCallResolver(FakeLLM(completion())).resolve(
            effect,
            CancellationToken(),
        )

    with pytest.raises(TypeError, match="ToolCallEffect"):
        await ToolCallResolver([]).resolve(effect, CancellationToken())


def test_tool_resolver_rejects_duplicate_tool_names() -> None:
    @tool
    def echo(value: str) -> str:
        """Echo a value."""

        return value

    with pytest.raises(ValueError, match="duplicate tool echo"):
        ToolCallResolver([echo, echo])


def test_tool_decorator_builds_schema_and_preserves_function_behavior() -> None:
    @tool
    def repeat(value: str, count: int = 1) -> str:
        """Repeat a value."""

        return value * count

    assert isinstance(repeat, FunctionTool)
    assert repeat.__name__ == "repeat"
    assert repeat.__doc__ == "Repeat a value."
    assert repeat.name == "repeat"
    assert repeat("x", count=3) == "xxx"
    assert repeat.invoke({"value": "y", "count": 2}) == "yy"
    assert repeat.schema.function.description == "Repeat a value."
    assert repeat.return_schema == {"type": "string"}
    parameters = repeat.schema.function.parameters
    assert parameters["type"] == "object"
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["value"]
    assert parameters["properties"] == {
        "value": {"title": "Value", "type": "string"},
        "count": {"default": 1, "title": "Count", "type": "integer"},
    }

    schema = repeat.schema
    schema.function.name = "changed"
    assert repeat.name == "repeat"


class NestedToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int


class SerializableToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    nested: NestedToolResult
    tags: tuple[str, ...]


class RecursiveNode(BaseModel):
    value: str
    child: RecursiveNode | None = None


class TypedDictResult(TypedDict):
    value: str


class NamedTupleResult(NamedTuple):
    value: str


@pytest.mark.asyncio
async def test_tool_resolver_round_trips_pydantic_udt_results() -> None:
    @tool
    def inspect_value(value: str) -> SerializableToolResult:
        """Return a nested typed result."""

        return SerializableToolResult(
            label=value,
            nested=NestedToolResult(count=2),
            tags=("a", "b"),
        )

    effect = ToolCallEffect.from_call(
        id="typed-result",
        attempt=1,
        idempotency_key="typed-result:1",
        tool_call_id="typed-result-call",
        name="inspect_value",
        arguments={"value": "hello"},
        events=empty_events(),
    )
    resolution = await ToolCallResolver([inspect_value]).resolve(
        effect,
        CancellationToken(),
    )

    assert isinstance(resolution, ToolCallResolution)
    assert resolution.status is ResolutionStatus.COMPLETED
    assert resolution.value == {
        "label": "hello",
        "nested": {"count": 2},
        "tags": ["a", "b"],
    }
    assert inspect_value.restore_result(resolution.value) == SerializableToolResult(
        label="hello",
        nested=NestedToolResult(count=2),
        tags=("a", "b"),
    )
    assert inspect_value.return_schema["type"] == "object"

    step = Step(
        id="typed-result-step",
        run_id="typed-result-run",
        source_revision=0,
        policy_revision="typed-result-policy",
        kind="tools",
        effects=(effect,),
    )
    journal = RunJournal(steps=(step,), resolutions=(resolution,))
    restored_step = Step.model_validate_json(step.model_dump_json())
    restored_journal = RunJournal.model_validate_json(journal.model_dump_json())
    restored_effect = restored_step.effects[0]
    restored_resolution = restored_journal.resolutions[0]
    assert isinstance(restored_effect, ToolCallEffect)
    assert restored_effect.event_snapshot.view.digest == effect.event_snapshot.digest
    assert isinstance(restored_resolution, ToolCallResolution)
    assert restored_resolution.result == resolution.result
    assert restored_resolution.value == resolution.value


@pytest.mark.asyncio
async def test_async_tool_result_compiler_receives_the_awaited_type() -> None:
    observed: list[SerializableToolResult] = []

    def compile_async_result(
        result: SerializableToolResult,
        events: EventView,
    ) -> ToolResult:
        del events
        observed.append(result)
        return ToolResult(content=result.label)

    @tool(result_compiler=compile_async_result)
    async def inspect_async(value: str) -> SerializableToolResult:
        """Return a typed result asynchronously."""

        return SerializableToolResult(
            label=value,
            nested=NestedToolResult(count=1),
            tags=(),
        )

    effect = ToolCallEffect.from_call(
        id="async-typed-result",
        attempt=1,
        idempotency_key="async-typed-result:1",
        tool_call_id="async-typed-result-call",
        name="inspect_async",
        arguments={"value": "awaited"},
        events=empty_events(),
    )

    resolution = await ToolCallResolver([inspect_async]).resolve(
        effect,
        CancellationToken(),
    )

    assert isinstance(resolution, ToolCallResolution)
    assert resolution.result == ToolResult(content="awaited")
    assert observed == [
        SerializableToolResult(
            label="awaited",
            nested=NestedToolResult(count=1),
            tags=(),
        )
    ]


@pytest.mark.asyncio
async def test_tool_resolver_rejects_results_outside_the_return_contract() -> None:
    @tool
    def invalid_result() -> SerializableToolResult:
        """Return data that violates the declared result contract."""

        return {"label": "wrong"}  # type: ignore[return-value]

    effect = ToolCallEffect.from_call(
        id="invalid-result",
        attempt=1,
        idempotency_key="invalid-result:1",
        tool_call_id="invalid-result-call",
        name="invalid_result",
        arguments={},
        events=empty_events(),
    )

    resolution = await ToolCallResolver([invalid_result]).resolve(
        effect,
        CancellationToken(),
    )

    assert resolution.status is ResolutionStatus.FAILED
    assert resolution.error is not None
    assert resolution.error.type == "ValidationError"


def test_tool_decorator_rejects_unserializable_or_non_pydantic_udts() -> None:
    class UnboundedResult(BaseModel):
        value: object

    with pytest.raises(TypeError, match="Any or object"):

        @tool
        def unbounded() -> UnboundedResult:
            """Return an unbounded field."""

            return UnboundedResult(value="anything")

    class AnyResult(BaseModel):
        value: Any

    with pytest.raises(TypeError, match="Any or object"):

        @tool
        def any_result() -> AnyResult:
            """Return an Any field."""

            return AnyResult(value="anything")

    class BareContainerResult(BaseModel):
        values: list  # pyright: ignore[reportMissingTypeArgument]

    with pytest.raises(TypeError, match="parameterize containers"):

        @tool
        def bare_container() -> BareContainerResult:
            """Return an unparameterized container field."""

            return BareContainerResult(values=[])

    with pytest.raises(TypeError, match="parameterize containers"):

        @tool  # pyright: ignore[reportUnknownArgumentType]
        def bare_mapping() -> dict:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
            """Return an unparameterized mapping."""

            return {}  # pyright: ignore[reportUnknownVariableType]

    @dataclass
    class DataclassResult:
        value: str

    with pytest.raises(TypeError, match="pydantic BaseModel"):

        @tool
        def dataclass_result() -> DataclassResult:
            """Return a dataclass instead of a Pydantic UDT."""

            return DataclassResult(value="wrong UDT")

    with pytest.raises(TypeError, match="pydantic BaseModel"):

        @tool
        def typed_dict_result() -> TypedDictResult:
            """Return a TypedDict instead of a Pydantic UDT."""

            return {"value": "wrong UDT"}

    with pytest.raises(TypeError, match="pydantic BaseModel"):

        @tool
        def named_tuple_result() -> NamedTupleResult:
            """Return a NamedTuple instead of a Pydantic UDT."""

            return NamedTupleResult(value="wrong UDT")

    class CallableResult(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        callback: Callable[[], str]

    with pytest.raises(TypeError, match="not JSON serializable"):

        @tool
        def callable_result() -> CallableResult:
            """Return a callable field."""

            return CallableResult(callback=lambda: "no")


def test_tool_result_compiler_contract_is_explicit() -> None:
    def compile_result(
        result: SerializableToolResult,
        events: EventView,
    ) -> ToolResult:
        return ToolResult(
            content=f"{events.run_id}:{result.label}:{len(events)}"
        )

    @tool(result_compiler=compile_result)
    def inspect_value() -> SerializableToolResult:
        """Return a compiler-aware result."""

        return SerializableToolResult(
            label="value",
            nested=NestedToolResult(count=1),
            tags=(),
        )

    events = EventView.create(
        "compiler-run",
        3,
        [UserMessageEvent(id="user", content="question")],
    )
    value = inspect_value.serialize_result(inspect_value())

    assert inspect_value.compile_result(value, events) == ToolResult(
        content="compiler-run:value:1"
    )

    def other_compiler(
        result: SerializableToolResult,
        events: EventView,
    ) -> ToolResult:
        del events
        return ToolResult(content=f"other:{result.label}")

    @tool(result_compiler=other_compiler)
    def other_inspect_value() -> SerializableToolResult:
        """Return the same schema with another compiler."""

        return inspect_value()

    assert inspect_value.revision != other_inspect_value.revision


def test_tool_result_compiler_rejects_invalid_signatures_and_outputs() -> None:
    async def async_compiler(
        result: SerializableToolResult,
        events: EventView,
    ) -> ToolResult:
        del events
        return ToolResult(content=result.label)

    with pytest.raises(TypeError, match="synchronous"):

        @tool(result_compiler=async_compiler)  # type: ignore[arg-type]
        def with_async_compiler() -> SerializableToolResult:
            """Use an invalid async compiler."""

            raise AssertionError

    def missing_events(result: SerializableToolResult) -> ToolResult:
        return ToolResult(content=result.label)

    with pytest.raises(TypeError, match="positional arguments"):

        @tool(result_compiler=missing_events)  # type: ignore[arg-type]
        def with_missing_events() -> SerializableToolResult:
            """Use a compiler missing its event snapshot."""

            raise AssertionError

    def wrong_result_type(result: str, events: EventView) -> ToolResult:
        del events
        return ToolResult(content=result)

    with pytest.raises(TypeError, match="match the tool return type"):

        @tool(result_compiler=wrong_result_type)  # type: ignore[arg-type]
        def with_wrong_result_type() -> SerializableToolResult:
            """Use a compiler for another result type."""

            raise AssertionError

    def wrong_events_type(
        result: SerializableToolResult,
        events: tuple[object, ...],
    ) -> ToolResult:
        del events
        return ToolResult(content=result.label)

    with pytest.raises(TypeError, match="must be EventView"):

        @tool(result_compiler=wrong_events_type)  # type: ignore[arg-type]
        def with_wrong_events_type() -> SerializableToolResult:
            """Use an untyped event sequence."""

            raise AssertionError

    def wrong_compiler_return(
        result: SerializableToolResult,
        events: EventView,
    ) -> str:
        del events
        return result.label

    with pytest.raises(TypeError, match="declare a ToolResult"):

        @tool(result_compiler=wrong_compiler_return)  # type: ignore[arg-type]
        def with_wrong_compiler_return() -> SerializableToolResult:
            """Declare the wrong compiler return type."""

            raise AssertionError

    class CallableCompiler:
        def __call__(
            self,
            result: SerializableToolResult,
            events: EventView,
        ) -> ToolResult:
            del events
            return ToolResult(content=result.label)

    with pytest.raises(TypeError, match="must be a function"):
        def callable_target() -> SerializableToolResult:
            """Use a callable object as a compiler."""

            return SerializableToolResult(
                label="value",
                nested=NestedToolResult(count=1),
                tags=(),
            )

        FunctionTool(
            callable_target,
            result_compiler=CallableCompiler(),  # type: ignore[arg-type]
        )

    def invalid_output(
        result: SerializableToolResult,
        events: EventView,
    ) -> ToolResult:
        del events
        return result.label  # type: ignore[return-value]

    @tool(result_compiler=invalid_output)
    def with_invalid_output() -> SerializableToolResult:
        """Return a value for a compiler with invalid runtime behavior."""

        return SerializableToolResult(
            label="value",
            nested=NestedToolResult(count=1),
            tags=(),
        )

    with pytest.raises(TypeError, match="must return ToolResult"):
        with_invalid_output.compile_result(
            with_invalid_output.serialize_result(with_invalid_output()),
            EventView.create("run", 0, []),
        )


def test_tool_return_annotation_supports_awaitable_and_recursive_models() -> None:
    @tool
    def recursive() -> RecursiveNode:
        """Return a recursive model."""

        return RecursiveNode(
            value="root",
            child=RecursiveNode(value="leaf"),
        )

    @tool
    def coroutine_result() -> Coroutine[object, object, str]:
        """Return an explicitly typed coroutine."""

        async def finish() -> str:
            return "done"

        return finish()

    assert recursive.restore_result(recursive.serialize_result(recursive())) == recursive()
    assert coroutine_result.return_schema == {"type": "string"}

    @tool
    def annotated_result() -> Annotated[str, Field(min_length=1)]:
        """Return an annotated string."""

        return "value"

    assert annotated_result.serialize_result("value") == "value"


def test_tool_result_serialization_must_be_lossless() -> None:
    class LossyResult(BaseModel):
        value: int

        @field_serializer("value")
        def serialize_value(self, value: int) -> int:
            return value + 1

    @tool
    def lossy() -> LossyResult:
        """Use a deliberately lossy serializer."""

        return LossyResult(value=1)

    with pytest.raises(ValueError, match="not losslessly"):
        lossy.serialize_result(lossy())


def test_event_snapshot_and_completed_resolution_evidence_are_validated() -> None:
    view = EventView.create(
        "snapshot-run",
        2,
        [UserMessageEvent(id="snapshot-user", content="hello")],
    )
    snapshot = EventSnapshot.create(view)
    restored = EventSnapshot.model_validate(snapshot.model_dump(mode="json"))

    assert restored.view.digest == view.digest
    assert EventSnapshot.restore_events("not-events") == "not-events"
    with pytest.raises(ValueError, match="digest"):
        EventSnapshot.model_validate(
            {**snapshot.model_dump(mode="json"), "digest": "forged"}
        )

    effect = ToolCallEffect.from_call(
        id="missing-evidence",
        attempt=1,
        idempotency_key="missing-evidence:1",
        tool_call_id="missing-evidence-call",
        name="echo",
        arguments={},
        events=view,
    )
    with pytest.raises(ValueError, match="compilation evidence"):
        ToolCallResolution.from_effect(
            effect,
            tool_call_id=effect.tool_call_id,
            name=effect.name,
            value="result",
            result=ToolResult(content='"result"'),
        )

    with pytest.raises(ValueError, match="JSON compliant"):
        ToolCallResolution(
            effect_id=effect.id,
            attempt=effect.attempt,
            status=ResolutionStatus.FAILED,
            tool_call_id=effect.tool_call_id,
            name=effect.name,
            value=float("inf"),
        )


@pytest.mark.asyncio
async def test_tool_resolver_validates_arguments_before_calling_function() -> None:
    called = False

    @tool
    def typed(value: int) -> int:
        """Accept one integer."""

        nonlocal called
        called = True
        return value

    effect = ToolCallEffect.from_call(
        id="invalid-tool-arguments",
        attempt=1,
        idempotency_key="invalid-tool-arguments:1",
        tool_call_id="call-invalid",
        name="typed",
        arguments={"value": "1", "extra": True},
        events=empty_events(),
    )

    resolution = await ToolCallResolver([typed]).resolve(
        effect,
        CancellationToken(),
    )

    assert resolution.status is ResolutionStatus.FAILED
    assert resolution.error is not None
    assert resolution.error.type == "ValidationError"
    assert called is False


@pytest.mark.asyncio
async def test_parallel_sync_tools_run_off_the_event_loop() -> None:
    rendezvous = threading.Barrier(2)

    @tool
    def meet(value: str) -> str:
        """Wait for the other parallel tool call."""

        rendezvous.wait(timeout=2)
        return value

    effects = tuple(
        ToolCallEffect.from_call(
            id=f"effect-{value}",
            attempt=1,
            idempotency_key=f"effect-{value}:1",
            tool_call_id=f"call-{value}",
            name="meet",
            arguments={"value": value},
            events=empty_events("parallel-sync-tools"),
        )
        for value in ("a", "b")
    )
    step = Step(
        id="parallel-sync-tools",
        run_id="parallel-sync-tools",
        source_revision=0,
        policy_revision="test@1",
        kind="tools",
        effects=effects,
        execution=ExecutionMode.PARALLEL,
    )

    resolutions = await asyncio.wait_for(
        Runtime([ToolCallResolver([meet])]).resolve_step(step),
        timeout=3,
    )

    assert [resolution.status for resolution in resolutions] == [
        ResolutionStatus.COMPLETED,
        ResolutionStatus.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_cancelling_sync_tool_waits_for_its_worker_thread() -> None:
    started = threading.Event()
    release = threading.Event()

    @tool
    def blocking(value: str) -> str:
        """Block until the test releases this tool."""

        started.set()
        release.wait(timeout=2)
        return value

    effect = ToolCallEffect.from_call(
        id="blocking-effect",
        attempt=1,
        idempotency_key="blocking-effect:1",
        tool_call_id="blocking-call",
        name="blocking",
        arguments={"value": "done"},
        events=empty_events("blocking-run"),
    )
    step = Step(
        id="blocking-step",
        run_id="blocking-run",
        source_revision=0,
        policy_revision="test@1",
        kind="tools",
        effects=(effect,),
    )
    cancellation = CancellationToken()
    task = asyncio.create_task(
        Runtime([ToolCallResolver([blocking])]).resolve_step(
            step,
            cancellation=cancellation,
        )
    )
    assert await asyncio.to_thread(started.wait, 1)

    cancellation.cancel()
    await asyncio.sleep(0)
    release.set()
    resolution = (await task)[0]

    assert resolution.status is ResolutionStatus.COMPLETED
    assert isinstance(resolution, ToolCallResolution)
    assert resolution.value == "done"


@pytest.mark.asyncio
async def test_cancelling_tool_resolver_task_waits_for_sync_worker() -> None:
    started = threading.Event()
    release = threading.Event()

    @tool
    def blocking(value: str) -> str:
        """Block until the test releases this tool."""

        started.set()
        release.wait(timeout=2)
        return value

    effect = ToolCallEffect.from_call(
        id="direct-blocking-effect",
        attempt=1,
        idempotency_key="direct-blocking-effect:1",
        tool_call_id="direct-blocking-call",
        name="blocking",
        arguments={"value": "done"},
        events=empty_events("direct-blocking-run"),
    )
    task = asyncio.create_task(
        ToolCallResolver([blocking]).resolve(effect, CancellationToken())
    )
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    release.set()
    resolution = await task

    assert resolution.status is ResolutionStatus.COMPLETED
    assert isinstance(resolution, ToolCallResolution)
    assert resolution.value == "done"


@pytest.mark.asyncio
async def test_cancelling_async_tool_interrupts_the_coroutine() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @tool
    async def blocking(value: str) -> str:
        """Block until the runtime cancels this tool."""

        del value
        started.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    effect = ToolCallEffect.from_call(
        id="async-blocking-effect",
        attempt=1,
        idempotency_key="async-blocking-effect:1",
        tool_call_id="async-blocking-call",
        name="blocking",
        arguments={"value": "unused"},
        events=empty_events("async-blocking-run"),
    )
    step = Step(
        id="async-blocking-step",
        run_id="async-blocking-run",
        source_revision=0,
        policy_revision="test@1",
        kind="tools",
        effects=(effect,),
    )
    cancellation = CancellationToken()
    task = asyncio.create_task(
        Runtime([ToolCallResolver([blocking])]).resolve_step(
            step,
            cancellation=cancellation,
        )
    )
    await started.wait()

    cancellation.cancel()
    resolution = (await task)[0]

    assert resolution.status is ResolutionStatus.CANCELLED
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_cancelling_sync_tool_does_not_wait_for_returned_awaitable() -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    awaitable_started = asyncio.Event()

    @tool
    def blocking(value: str) -> Awaitable[str]:
        """Return cancellable async work after a blocking sync phase."""

        worker_started.set()
        release_worker.wait(timeout=2)

        async def finish() -> str:
            awaitable_started.set()
            await asyncio.Future[None]()
            return value

        return finish()

    effect = ToolCallEffect.from_call(
        id="sync-awaitable-effect",
        attempt=1,
        idempotency_key="sync-awaitable-effect:1",
        tool_call_id="sync-awaitable-call",
        name="blocking",
        arguments={"value": "unused"},
        events=empty_events("sync-awaitable-run"),
    )
    step = Step(
        id="sync-awaitable-step",
        run_id="sync-awaitable-run",
        source_revision=0,
        policy_revision="test@1",
        kind="tools",
        effects=(effect,),
    )
    cancellation = CancellationToken()
    task = asyncio.create_task(
        Runtime([ToolCallResolver([blocking])]).resolve_step(
            step,
            cancellation=cancellation,
        )
    )
    assert await asyncio.to_thread(worker_started.wait, 1)

    cancellation.cancel()
    release_worker.set()
    resolution = (await task)[0]

    assert resolution.status is ResolutionStatus.CANCELLED
    assert not awaitable_started.is_set()


def test_tool_revision_binds_implementation_and_closure_configuration() -> None:
    def create(prefix: str):
        @tool
        def prefix_value(value: str) -> str:
            """Prefix a value."""

            return prefix + value

        return prefix_value

    left = create("left:")
    right = create("right:")

    assert left.schema == right.schema
    assert left.revision != right.revision

    class Dependency:
        def __init__(self, revision: str) -> None:
            self.revision = revision

    def with_dependency(dependency: Dependency):
        @tool
        def dependency_value(value: str) -> str:
            """Use a revisioned closure dependency."""

            return dependency.revision + value

        return dependency_value

    assert with_dependency(Dependency("v1")).revision != with_dependency(
        Dependency("v2")
    ).revision

    config = {
        "effort": ReasoningEffort.HIGH,
        "numbers": [1.5],
        "path": Path("."),
    }

    @tool
    def configured(value: str) -> str:
        """Use structured closure configuration."""

        return f"{config!r}:{value}"

    assert configured.revision.startswith("tool@sha256:")


def test_tool_decorator_rejects_functions_without_a_typed_contract() -> None:
    with pytest.raises(TypeError, match="function or result_compiler"):
        tool()  # pyright: ignore[reportCallIssue]

    with pytest.raises(ValueError, match="docstring"):

        @tool
        def undocumented(value: str) -> str:
            return value  # pyright: ignore[reportUnknownVariableType]

    with pytest.raises(TypeError, match="parameter value"):

        @tool  # pyright: ignore[reportUnknownArgumentType]
        def untyped(value) -> object:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
            """Missing a parameter annotation."""

            return value  # pyright: ignore[reportUnknownVariableType]

    with pytest.raises(TypeError, match="return annotation"):

        @tool
        def missing_return(value: str):
            """Missing a return annotation."""

            return value

    with pytest.raises(TypeError, match="JSON object field"):

        @tool
        def variadic(*values: str) -> tuple[str, ...]:
            """Use variadic parameters."""

            return values

    with pytest.raises(TypeError, match="free functions"):

        @tool
        def invalid_method(self: object, value: str) -> str:
            """Do not decorate methods."""

            return value

    class BoundTools:
        def method(self, value: str) -> str:
            """Do not decorate bound methods."""

            return value

    with pytest.raises(TypeError, match="free functions"):
        tool(BoundTools().method)

    def local_class_method() -> type[object]:
        class LocalTools:
            @tool
            def method(receiver, value: str) -> str:
                """Do not decorate methods on local classes."""

                del receiver
                return value

        return LocalTools

    with pytest.raises(TypeError, match="free functions"):
        local_class_method()
