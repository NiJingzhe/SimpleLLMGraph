from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, Field

from SimpleLLMFunc.context.ir import (
    AssistantMessage,
    Completion,
    CompletionChoice,
    ContentPart,
    FinishReason,
    Image,
    InputImagePart,
    InputTextPart,
    OutputAudioData,
    OutputAudioPart,
    OutputTextPart,
    ReasoningPart,
    ToolCall,
    ToolCallFunction,
    ToolResult,
)
from SimpleLLMFunc.event import (
    AssistantMessageEvent,
    EventView,
    ToolResultEvent,
    UserMessageEvent,
)
from SimpleLLMFunc.loop import (
    CallTrace,
    CancellationToken,
    CompiledContext,
    ContextStrategy,
    ContextStrategyEvent,
    ContextStrategyName,
    DefaultController,
    Effect,
    EffectResolution,
    ExecutionMode,
    FunctionTool,
    InMemoryRunStore,
    Loop,
    LoopInput,
    LoopPhase,
    LoopRunState,
    ModelCallEffect,
    ModelCallResolution,
    Resolver,
    ResolutionStatus,
    RunStatus,
    Runtime,
    Step,
    ToolCallEffect,
    ToolCallResolution,
    ToolCallResolver,
    tool,
)


@tool
def echo(value: str) -> str:
    """Echo a value."""

    return value


@tool
def set_context_strategy(
    name: ContextStrategyName,
    instruction: str,
    recent_message_limit: Annotated[int, Field(ge=1)],
) -> ContextStrategy:
    """Change the context strategy."""

    return ContextStrategy(
        name=name,
        instruction=instruction,
        recent_message_limit=recent_message_limit,
    )


def make_loop(
    *,
    tools: Sequence[FunctionTool[..., object]] = (echo,),
    context_tool: FunctionTool[..., object] | None = None,
    max_model_calls: int = 3,
    tool_execution: ExecutionMode = ExecutionMode.SEQUENTIAL,
) -> Loop:
    return Loop(
        model="test-model",
        system_prompt="Be useful.",
        tools=tools,
        context_tool=context_tool,
        max_model_calls=max_model_calls,
        tool_execution=tool_execution,
    )


def completion(
    content: str | Sequence[ContentPart] | None = "done",
    *,
    tool_calls: list[ToolCall] | None = None,
) -> Completion:
    return Completion(
        id="completion",
        created=1,
        model="test-model",
        choices=[
            CompletionChoice(
                index=0,
                message=AssistantMessage(
                    content=(
                        content
                        if isinstance(content, str) or content is None
                        else list(content)
                    ),
                    tool_calls=tool_calls,
                ),
                finish_reason=(
                    FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP
                ),
            )
        ],
    )


def model_resolution(step: Step, value: Completion) -> ModelCallResolution:
    effect = step.effects[0]
    assert isinstance(effect, ModelCallEffect)
    return ModelCallResolution(
        effect_id=effect.id,
        attempt=effect.attempt,
        status=ResolutionStatus.COMPLETED,
        completion=value,
        trace=CallTrace(
            actual_model="test-model",
            context_digest=effect.context.digest,
            duration_seconds=0,
        ),
    )


def tool_call(call_id: str = "call-1", name: str = "echo") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=ToolCallFunction(name=name, arguments='{"value":"hello"}'),
    )


def enter_tool_phase(
    loop: Loop,
    *,
    call: ToolCall | None = None,
) -> tuple[LoopRunState, Step]:
    state = loop.initialize("Use a tool", run_id="tool-run")
    step = loop.step(state)
    state = loop.reduce(
        state,
        step,
        (model_resolution(step, completion(None, tool_calls=[call or tool_call()])),),
    ).next_state
    return state, loop.step(state)


def completed_tool_resolution(
    effect: ToolCallEffect,
    function_tool: FunctionTool[..., object],
    value: object,
) -> ToolCallResolution:
    serialized, result = function_tool.prepare_result(
        value,
        effect.event_snapshot.view,
    )
    return ToolCallResolution.from_effect(
        effect,
        tool_call_id=effect.tool_call_id,
        name=effect.name,
        value=serialized,
        result=result,
        tool_revision=function_tool.revision,
        event_digest=effect.event_snapshot.digest,
    )


def test_loop_declaration_compiles_exact_inspectable_context() -> None:
    loop = make_loop(tool_execution=ExecutionMode.PARALLEL)
    state = loop.initialize(
        LoopInput(
            content=[
                InputTextPart(text="Inspect this"),
                InputImagePart(image_url="https://example.test/image.png"),
            ],
            context=ContextStrategy(
                name=ContextStrategyName.RECENT,
                instruction="Keep recent evidence.",
                recent_message_limit=4,
            ),
        ),
        run_id="inspectable",
    )

    step = loop.step(state)
    effect = step.effects[0]

    assert isinstance(effect, ModelCallEffect)
    assert effect.context.source_revision == 0
    assert effect.context.compiler_revision.endswith(state.strategy.revision)
    assert effect.context.request.messages[1].content == [
        InputTextPart(text="Inspect this"),
        InputImagePart(image_url="https://example.test/image.png"),
    ]
    assert effect.context.request.tools == [echo.schema]
    assert effect.context.provenance[0].event_ids == (
        "inspectable:system",
        "inspectable:strategy:0",
    )
    assert loop.policy_revision == make_loop(
        tool_execution=ExecutionMode.PARALLEL
    ).policy_revision
    assert loop.policy_revision != make_loop().policy_revision

    request = effect.context.request
    request.max_tokens = 17
    changed_context = effect.context.derive(
        request=request,
        transformation="test.change_request",
    )
    changed_step = step.model_copy(
        update={
            "effects": (effect.model_copy(update={"context": changed_context}),)
        }
    )
    assert changed_step.digest != step.digest
    dumped = step.model_dump(mode="json")
    assert dumped["effects"][0]["context"]["digest"] == effect.context.digest


def test_loop_completes_text_audio_and_refusal_results() -> None:
    loop = make_loop(tools=())

    for run_id, value, expected in (
        ("string", "finished", "finished"),
        ("text", [OutputTextPart(text="finished")], "finished"),
        (
            "audio",
            [
                OutputAudioPart(
                    output_audio=OutputAudioData(
                        id="audio-1",
                        transcript="spoken result",
                    )
                )
            ],
            "spoken result",
        ),
        ("empty", None, ""),
    ):
        state = loop.initialize("Finish", run_id=run_id)
        step = loop.step(state)
        completed = loop.reduce(
            state,
            step,
            (model_resolution(step, completion(value)),),
        ).next_state
        assert completed.status is RunStatus.COMPLETED
        assert loop.result(completed) == expected

    state = loop.initialize("Refuse", run_id="refusal")
    step = loop.step(state)
    refusal = completion(None)
    refusal.choices[0].message.refusal = "cannot comply"
    completed = loop.reduce(
        state,
        step,
        (model_resolution(step, refusal),),
    ).next_state
    assert loop.result(completed) == "cannot comply"


def test_loop_model_failures_are_terminal() -> None:
    loop = make_loop()
    state = loop.initialize("Fail", run_id="model-failure")
    step = loop.step(state)
    effect = step.effects[0]

    failed = loop.reduce(
        state,
        step,
        (EffectResolution.failed(effect, RuntimeError("provider failed")),),
    ).next_state
    assert failed.status is RunStatus.FAILED
    assert failed.error == "provider failed"

    state = loop.initialize("Cancel", run_id="model-cancel")
    step = loop.step(state)
    cancelled = loop.reduce(
        state,
        step,
        (EffectResolution.cancelled(step.effects[0]),),
    ).next_state
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.error == ResolutionStatus.CANCELLED.value

    state = loop.initialize("Empty", run_id="empty-completion")
    step = loop.step(state)
    empty = Completion(
        id="empty",
        created=1,
        model="test-model",
        choices=[],
    )
    no_choices = loop.reduce(
        state,
        step,
        (model_resolution(step, empty),),
    ).next_state
    assert no_choices.status is RunStatus.FAILED
    assert no_choices.error == "model completion contains no choices"

    for finish_reason in (FinishReason.LENGTH, FinishReason.CONTENT_FILTER):
        state = loop.initialize(
            "Partial",
            run_id=f"partial-{finish_reason.value}",
        )
        step = loop.step(state)
        partial = completion("partial")
        partial.choices[0].finish_reason = finish_reason
        failed = loop.reduce(
            state,
            step,
            (model_resolution(step, partial),),
        ).next_state
        assert failed.status is RunStatus.FAILED
        assert failed.error == (
            f"model completion has unexpected finish reason: {finish_reason.value}"
        )


def test_loop_tool_results_return_to_model_or_cancel() -> None:
    loop = make_loop(tool_execution=ExecutionMode.PARALLEL)
    state, step = enter_tool_phase(loop)
    effect = step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    assert step.execution is ExecutionMode.PARALLEL

    changed_effect = effect.model_copy(update={"arguments_json": '{"value":"bye"}'})
    changed_step = step.model_copy(update={"effects": (changed_effect,)})
    assert changed_step.digest != step.digest
    assert step.model_dump(mode="json")["effects"][0]["arguments_json"] == (
        '{"value":"hello"}'
    )

    denied = loop.reduce(
        state,
        step,
        (EffectResolution.denied(effect, "operator denied"),),
    ).next_state
    assert denied.status is RunStatus.READY
    assert denied.next_action is LoopPhase.MODEL
    denied_event = denied.event_view.latest(ToolResultEvent)
    assert denied_event.is_error
    assert "operator denied" in denied_event.result.content

    state, step = enter_tool_phase(loop)
    cancelled = loop.reduce(
        state,
        step,
        (EffectResolution.cancelled(step.effects[0]),),
    ).next_state
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.next_action is LoopPhase.DONE


def test_loop_context_tool_activation_is_declared() -> None:
    tools = (echo, set_context_strategy)
    loop = make_loop(tools=tools, context_tool=set_context_strategy)
    call = ToolCall(
        id="context-call",
        function=ToolCallFunction(
            name="set_context_strategy",
            arguments=(
                '{"instruction":"Recent only","name":"recent",'
                '"recent_message_limit":2}'
            ),
        ),
    )
    state, step = enter_tool_phase(loop, call=call)
    effect = step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    strategy = ContextStrategy(
        name=ContextStrategyName.RECENT,
        instruction="Recent only",
        recent_message_limit=2,
    )
    resolution = completed_tool_resolution(effect, set_context_strategy, strategy)

    updated = loop.reduce(state, step, (resolution,)).next_state

    assert updated.strategy == strategy
    assert updated.event_view.latest(ContextStrategyEvent).strategy == strategy
    model_step = loop.step(updated)
    model_effect = model_step.effects[0]
    assert isinstance(model_effect, ModelCallEffect)
    assert model_effect.context.compiler_revision.endswith(strategy.revision)


class ScreenshotResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    caption: str
    image: Image
    gallery: tuple[Image, ...] = ()


def test_multimodal_tool_result_rewrites_the_complete_tool_batch() -> None:
    @tool
    def screenshot() -> ScreenshotResult:
        """Capture a screenshot."""

        return ScreenshotResult(
            caption="dashboard",
            image=Image.from_base64(b"png", media_type="image/png"),
            gallery=(Image.from_url("https://example.test/detail.png"),),
        )

    loop = make_loop(
        tools=(echo, screenshot),
        tool_execution=ExecutionMode.PARALLEL,
    )
    state = loop.initialize("Inspect the dashboard", run_id="multimodal-tool")
    model_step = loop.step(state)
    calls = [
        ToolCall(
            id="image-call",
            function=ToolCallFunction(name="screenshot", arguments="{}"),
        ),
        tool_call(call_id="echo-call"),
    ]
    state = loop.reduce(
        state,
        model_step,
        (model_resolution(model_step, completion(None, tool_calls=calls)),),
    ).next_state
    tool_step = loop.step(state)
    image_effect, echo_effect = tool_step.effects
    assert isinstance(image_effect, ToolCallEffect)
    assert isinstance(echo_effect, ToolCallEffect)
    resolutions = (
        completed_tool_resolution(image_effect, screenshot, screenshot()),
        completed_tool_resolution(echo_effect, echo, "hello"),
    )

    state = loop.reduce(state, tool_step, resolutions).next_state
    result_events = state.event_view.all(ToolResultEvent)
    assert isinstance(result_events[0].result.content, list)
    assert isinstance(result_events[0].result.content[-1], Image)
    assert sum(
        isinstance(part, Image) for part in result_events[0].result.content
    ) == 2
    assert result_events[1].result == ToolResult(content='"hello"')

    context = loop.step(state).effects[0]
    assert isinstance(context, ModelCallEffect)
    messages = context.context.request.messages
    assert not any(message.role.value == "tool" for message in messages)
    assert not any(
        isinstance(message, AssistantMessage) and message.tool_calls
        for message in messages
    )
    assert [message.role.value for message in messages[-4:]] == [
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    image_message = messages[-3]
    assert image_message.role.value == "user"
    assert isinstance(image_message.content, list)
    assert isinstance(image_message.content[-1], Image)
    assert context.context.provenance[-3].event_ids == (
        "multimodal-tool:step:0:assistant",
        "multimodal-tool:step:1:tool-result:0",
    )


@pytest.mark.asyncio
async def test_tool_result_compiler_receives_an_isolated_pre_execution_event_view() -> None:
    snapshots: list[tuple[int, str, int]] = []

    class Observation(BaseModel):
        value: str

    def compile_observation(
        result: Observation,
        events: EventView,
    ) -> ToolResult:
        user = events.latest(UserMessageEvent)
        assert isinstance(user.content, list)
        user.content.append(InputTextPart(text="compiler mutation"))
        snapshots.append((events.revision, events.digest, len(events)))
        return ToolResult(content=f"{result.value}:{len(user.content)}")

    @tool(result_compiler=compile_observation)
    def observe(value: str) -> Observation:
        """Observe prior tool activity."""

        return Observation(value=value)

    loop = make_loop(tools=(observe,))
    state = loop.initialize(
        LoopInput(content=[InputTextPart(text="before")]),
        run_id="compiler-events",
    )
    model_step = loop.step(state)
    call = ToolCall(
        id="observe-call",
        function=ToolCallFunction(
            name="observe",
            arguments='{"value":"result"}',
        ),
    )
    state = loop.reduce(
        state,
        model_step,
        (model_resolution(model_step, completion(None, tool_calls=[call])),),
    ).next_state
    before = state.event_view
    tool_step = loop.step(state)
    effect = tool_step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    resolution = await ToolCallResolver([observe]).resolve(
        effect,
        CancellationToken(),
    )
    assert isinstance(resolution, ToolCallResolution)

    reduced = loop.reduce(state, tool_step, (resolution,)).next_state

    assert snapshots == [(before.revision, before.digest, len(before))]
    assert len(before.all(ToolResultEvent)) == 0
    original_user = state.event_view.latest(UserMessageEvent)
    assert original_user.content == [InputTextPart(text="before")]
    compiled = reduced.event_view.latest(ToolResultEvent)
    assert compiled.result == ToolResult(content="result:2")


def test_recent_context_keeps_multimodal_result_pairs_atomic() -> None:
    @tool
    def screenshot() -> ScreenshotResult:
        """Capture a screenshot."""

        return ScreenshotResult(
            caption="view",
            image=Image.from_url("https://example.test/view.png"),
        )

    loop = make_loop(tools=(screenshot,))
    state = loop.initialize(
        LoopInput(
            content="Inspect",
            context=ContextStrategy(
                name=ContextStrategyName.RECENT,
                instruction="Keep the latest result.",
                recent_message_limit=1,
            ),
        ),
        run_id="recent-image",
    )
    model_step = loop.step(state)
    call = ToolCall(
        id="recent-image-call",
        function=ToolCallFunction(name="screenshot", arguments="{}"),
    )
    state = loop.reduce(
        state,
        model_step,
        (model_resolution(model_step, completion(None, tool_calls=[call])),),
    ).next_state
    tool_step = loop.step(state)
    effect = tool_step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    state = loop.reduce(
        state,
        tool_step,
        (
            completed_tool_resolution(effect, screenshot, screenshot()),
        ),
    ).next_state

    model_effect = loop.step(state).effects[0]
    assert isinstance(model_effect, ModelCallEffect)
    assert [message.role.value for message in model_effect.context.request.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_image_return_annotation_forces_custom_results_to_user_messages() -> None:
    def compile_screenshot(
        result: ScreenshotResult,
        events: EventView,
    ) -> ToolResult:
        del events
        return ToolResult(content=result.caption)

    @tool(result_compiler=compile_screenshot)
    def screenshot() -> ScreenshotResult:
        """Capture an image with a text-only custom presentation."""

        return ScreenshotResult(
            caption="dashboard",
            image=Image.from_url("https://example.test/dashboard.png"),
        )

    loop = make_loop(tools=(screenshot,))
    state = loop.initialize("Inspect", run_id="custom-image-result")
    model_step = loop.step(state)
    call = ToolCall(
        id="custom-image-call",
        function=ToolCallFunction(name="screenshot", arguments="{}"),
    )
    state = loop.reduce(
        state,
        model_step,
        (model_resolution(model_step, completion(None, tool_calls=[call])),),
    ).next_state
    tool_step = loop.step(state)
    effect = tool_step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    resolution = completed_tool_resolution(effect, screenshot, screenshot())

    state = loop.reduce(state, tool_step, (resolution,)).next_state

    event = state.event_view.latest(ToolResultEvent)
    assert event.result == ToolResult(content="dashboard", as_user_message=True)
    model_effect = loop.step(state).effects[0]
    assert isinstance(model_effect, ModelCallEffect)
    assert [message.role.value for message in model_effect.context.request.messages[-2:]] == [
        "assistant",
        "user",
    ]


def test_loop_rejects_orphan_results_and_mismatched_compilation_evidence() -> None:
    loop = make_loop()
    state = loop.initialize("Start", run_id="orphan-result")
    orphan = ToolResultEvent(
        id="orphan-result:event",
        tool_call_id="missing-call",
        tool_name="echo",
        result=ToolResult(content='"orphan"'),
    )
    invalid = state.model_copy(update={"events": (*state.events, orphan)})
    with pytest.raises(ValueError, match="orphan tool result"):
        loop.step(invalid)

    assistant = AssistantMessageEvent(
        id="orphan-result:assistant",
        tool_calls=(
            tool_call(call_id="first-call"),
            tool_call(call_id="second-call"),
        ),
    )
    partial = ToolResultEvent(
        id="orphan-result:partial",
        tool_call_id="first-call",
        tool_name="echo",
        result=ToolResult(content='"partial"'),
    )
    incomplete = state.model_copy(
        update={"events": (*state.events, assistant, partial)}
    )
    with pytest.raises(ValueError, match="incomplete tool result batch"):
        loop.step(incomplete)

    state, tool_step = enter_tool_phase(loop)
    effect = tool_step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    resolution = completed_tool_resolution(effect, echo, "hello").model_copy(
        update={"event_digest": "different"}
    )
    with pytest.raises(ValueError, match="compilation evidence"):
        loop.reduce(state, tool_step, (resolution,))


def test_context_tool_requires_a_context_strategy_result() -> None:
    @tool
    def invalid_context() -> dict[str, str]:
        """Return serializable data that is not a ContextStrategy."""

        return {"value": "invalid"}

    loop = make_loop(tools=(invalid_context,), context_tool=invalid_context)
    call = ToolCall(
        id="invalid-context-call",
        function=ToolCallFunction(name="invalid_context", arguments="{}"),
    )
    state, step = enter_tool_phase(loop, call=call)
    effect = step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    resolution = completed_tool_resolution(
        effect,
        invalid_context,
        {"value": "invalid"},
    )

    failed = loop.reduce(state, step, (resolution,)).next_state

    assert failed.status is RunStatus.FAILED
    assert failed.error is not None
    assert failed.error.startswith("invalid context tool result:")


def test_loop_rejects_invalid_declarations_and_states() -> None:
    with pytest.raises(ValueError, match="model"):
        Loop(model="", system_prompt="system")
    with pytest.raises(ValueError, match="system_prompt"):
        Loop(model="model", system_prompt="")
    with pytest.raises(ValueError, match="max_model_calls"):
        Loop(model="model", system_prompt="system", max_model_calls=0)
    with pytest.raises(ValueError, match="unique"):
        Loop(model="model", system_prompt="system", tools=[echo, echo])
    with pytest.raises(ValueError, match="context_tool"):
        make_loop(context_tool=set_context_strategy)

    loop = make_loop()
    with pytest.raises(ValueError, match="run_id"):
        loop.initialize("task", run_id="")
    state = loop.initialize("task", run_id="terminal")
    terminal = state.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "settled": True,
            "next_action": LoopPhase.DONE,
            "output": "done",
        }
    )
    with pytest.raises(ValueError, match="cannot calculate"):
        loop.step(terminal)
    with pytest.raises(RuntimeError, match="not complete"):
        loop.result(state)


def test_loop_rejects_invalid_step_and_resolution_shapes() -> None:
    loop = make_loop()
    state = loop.initialize("task", run_id="invalid-shape")
    model_step = loop.step(state)
    wrong_kind = model_step.model_copy(update={"kind": "other"})
    with pytest.raises(ValueError, match="shape"):
        loop.reduce(
            state,
            wrong_kind,
            (model_resolution(wrong_kind, completion()),),
        )

    with pytest.raises(TypeError, match="ModelCallResolution"):
        loop.reduce(
            state,
            model_step,
            (EffectResolution.completed(model_step.effects[0]),),
        )

    mismatched_trace = model_resolution(model_step, completion()).model_copy(
        update={
            "trace": model_resolution(model_step, completion()).trace.model_copy(
                update={"context_digest": "different"}
            )
        }
    )
    with pytest.raises(ValueError, match="compiled context"):
        loop.reduce(state, model_step, (mismatched_trace,))

    mismatched_model = model_resolution(model_step, completion()).model_copy(
        update={
            "trace": model_resolution(model_step, completion()).trace.model_copy(
                update={"actual_model": "different-model"}
            )
        }
    )
    with pytest.raises(ValueError, match="completion model"):
        loop.reduce(state, model_step, (mismatched_model,))

    wrong_model_effect = ToolCallEffect.from_call(
        id=model_step.effects[0].id,
        attempt=1,
        idempotency_key="wrong-model-effect",
        tool_call_id="wrong-model-call",
        name="echo",
        arguments={"value": "hello"},
        events=state.event_view,
    )
    malformed_model = model_step.model_copy(
        update={"effects": (wrong_model_effect,)}
    )
    with pytest.raises(TypeError, match="ModelCallEffect"):
        loop.reduce(
            state,
            malformed_model,
            (model_resolution(model_step, completion()),),
        )

    model_effect = model_step.effects[0]
    assert isinstance(model_effect, ModelCallEffect)
    unrelated_context = CompiledContext.create(
        request=model_effect.context.request,
        source_revision=state.revision,
        compiler_revision="unrelated-compiler@1",
        provenance=model_effect.context.provenance,
    )
    unrelated_step = model_step.model_copy(
        update={
            "effects": (
                model_effect.model_copy(update={"context": unrelated_context}),
            )
        }
    )
    with pytest.raises(ValueError, match="not derived"):
        loop.validate_step(state, unrelated_step)

    tool_state = state.model_copy(update={"next_action": LoopPhase.TOOLS})
    with pytest.raises(ValueError, match="pending ToolCallEvents"):
        loop.step(tool_state)

    entered, tool_step = enter_tool_phase(loop)
    wrong_effect = model_step.effects[0]
    malformed = tool_step.model_copy(update={"effects": (wrong_effect,)})
    with pytest.raises(ValueError, match="pending tool calls"):
        loop.reduce(
            entered,
            malformed,
            (EffectResolution.completed(wrong_effect),),
        )

    effect = tool_step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    with pytest.raises(TypeError, match="ToolCallResolution"):
        loop.reduce(
            entered,
            tool_step,
            (EffectResolution.completed(effect),),
        )

    mismatched = ToolCallResolution.from_effect(
        effect,
        tool_call_id="different",
        name=effect.name,
        value="hello",
        result=ToolResult(content='"hello"'),
        tool_revision=echo.revision,
        event_digest=effect.event_snapshot.digest,
    )
    with pytest.raises(ValueError, match="metadata"):
        loop.reduce(entered, tool_step, (mismatched,))


@pytest.mark.asyncio
async def test_controller_rejects_malformed_declarative_step_before_resolution() -> None:
    class TrackingModelResolver(Resolver):
        effect_type = ModelCallEffect

        def __init__(self) -> None:
            self.calls = 0

        async def resolve(
            self,
            effect: Effect,
            cancellation: CancellationToken,
        ) -> EffectResolution:
            del cancellation
            assert isinstance(effect, ModelCallEffect)
            self.calls += 1
            return EffectResolution.completed(effect)

    loop = make_loop()
    state = loop.initialize("task", run_id="malformed-pending")
    step = loop.step(state)
    effect = step.effects[0]
    assert isinstance(effect, ModelCallEffect)
    malformed = step.model_copy(
        update={"effects": (effect.model_copy(update={"timeout": 99}),)}
    )
    store = InMemoryRunStore[LoopRunState]()
    await store.create(state)
    await store.record_step(malformed, expected_revision=0)
    resolver = TrackingModelResolver()

    with pytest.raises(ValueError, match="metadata"):
        await DefaultController().resume(
            loop,
            state.run_id,
            Runtime([resolver]),
            store,
        )

    assert resolver.calls == 0


def test_context_editor_can_recompile_an_explicit_event_projection() -> None:
    strategy = ContextStrategy(
        name=ContextStrategyName.RECENT,
        instruction="Keep complete tool-call batches.",
        recent_message_limit=1,
    )
    loop = make_loop()
    state = loop.initialize(
        LoopInput(content="Start", context=strategy),
        run_id="edited-context",
    )
    assistant = AssistantMessageEvent(
        id="edited-context:assistant",
        content=[
            ReasoningPart(reasoning="Use the echo tool", signature="signature"),
            OutputTextPart(text="Calling a tool"),
        ],
        tool_calls=(tool_call(),),
    )
    result = ToolResultEvent(
        id="edited-context:tool-result",
        causation_id="edited-context:effect",
        tool_call_id="call-1",
        tool_name="echo",
        result=ToolResult(content='{"value":"hello"}'),
    )
    edited = state.model_copy(
        update={
            "revision": 1,
            "events": (*state.events, assistant, result),
        }
    )

    step = loop.step(edited)
    effect = step.effects[0]

    assert isinstance(effect, ModelCallEffect)
    messages = effect.context.request.messages
    assert isinstance(messages[-2], AssistantMessage)
    assert messages[-2].content == [
        ReasoningPart(reasoning="Use the echo tool", signature="signature"),
        OutputTextPart(text="Calling a tool"),
    ]
    assert messages[-1].role.value == "tool"

    invalid = edited.model_copy(
        update={
            "events": tuple(
                event
                for event in edited.events
                if not event.id.endswith(":user:0")
            )
        }
    )
    with pytest.raises(ValueError, match="initial user message"):
        loop.step(invalid)


def test_loop_rejects_duplicate_tool_call_ids_in_one_model_turn() -> None:
    loop = make_loop()
    state = loop.initialize("Use tools", run_id="duplicate-calls")
    step = loop.step(state)
    duplicate = tool_call()

    failed = loop.reduce(
        state,
        step,
        (
            model_resolution(
                step,
                completion(None, tool_calls=[duplicate, duplicate]),
            ),
        ),
    ).next_state

    assert failed.status is RunStatus.FAILED
    assert failed.error == "model produced duplicate tool-call IDs"


def test_loop_rejects_reused_and_invalid_tool_calls() -> None:
    loop = make_loop()
    state, tool_step = enter_tool_phase(loop)
    effect = tool_step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    state = loop.reduce(
        state,
        tool_step,
        (
            completed_tool_resolution(effect, echo, "hello"),
        ),
    ).next_state
    model_step = loop.step(state)

    reused = loop.reduce(
        state,
        model_step,
        (
            model_resolution(
                model_step,
                completion(None, tool_calls=[tool_call()]),
            ),
        ),
    ).next_state
    assert reused.status is RunStatus.FAILED
    assert reused.error == "model reused a tool-call ID from an earlier turn"

    state = loop.initialize("Use tools", run_id="invalid-tool-call")
    model_step = loop.step(state)
    invalid = ToolCall(
        id="invalid-call",
        function=ToolCallFunction(name="echo", arguments="{"),
    )
    failed = loop.reduce(
        state,
        model_step,
        (
            model_resolution(
                model_step,
                completion(None, tool_calls=[invalid]),
            ),
        ),
    ).next_state
    assert failed.status is RunStatus.FAILED
    assert failed.error is not None
    assert failed.error.startswith("invalid model tool call at index 0:")


def test_loop_fails_before_tools_when_model_call_limit_is_reached() -> None:
    loop = make_loop(max_model_calls=1)
    state = loop.initialize("Use a tool", run_id="model-call-limit")
    step = loop.step(state)

    failed = loop.reduce(
        state,
        step,
        (
            model_resolution(
                step,
                completion(None, tool_calls=[tool_call()]),
            ),
        ),
    ).next_state

    assert failed.status is RunStatus.FAILED
    assert failed.settled
    assert failed.next_action is LoopPhase.DONE
    assert failed.error == "model-call limit reached before requested tools could run"


def test_invalid_context_tool_result_reduces_to_terminal_failure() -> None:
    loop = make_loop(
        tools=(echo, set_context_strategy),
        context_tool=set_context_strategy,
    )
    call = ToolCall(
        id="bad-context",
        function=ToolCallFunction(
            name="set_context_strategy",
            arguments=(
                '{"instruction":"Recent","name":"recent",'
                '"recent_message_limit":2}'
            ),
        ),
    )
    state, step = enter_tool_phase(loop, call=call)
    effect = step.effects[0]
    assert isinstance(effect, ToolCallEffect)
    resolution = ToolCallResolution.from_effect(
        effect,
        tool_call_id=effect.tool_call_id,
        name=effect.name,
        value={"invalid": True},
        result=ToolResult(content='{"invalid":true}'),
        tool_revision=set_context_strategy.revision,
        event_digest=effect.event_snapshot.digest,
    )

    failed = loop.reduce(state, step, (resolution,)).next_state

    assert failed.status is RunStatus.FAILED
    assert failed.settled
    assert failed.error is not None
    assert failed.error.startswith("invalid tool result:")

    with pytest.raises(ValueError, match="valid JSON"):
        ToolCallResolution.from_effect(
            effect,
            tool_call_id=effect.tool_call_id,
            name=effect.name,
            value={"bad": {1}},
            result=ToolResult(content='{"bad":"invalid"}'),
            tool_revision=set_context_strategy.revision,
            event_digest=effect.event_snapshot.digest,
        )
