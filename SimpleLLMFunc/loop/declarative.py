"""Declarative model/tool Loop implemented entirely on the public L1 protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from SimpleLLMFunc.context.ir import (
    AssistantMessage,
    ContentPart,
    Conversation,
    FinishReason,
    InputTextPart,
    OutputAudioPart,
    OutputTextPart,
    Request,
    SystemMessage,
    Tool,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from SimpleLLMFunc.event import (
    AssistantMessageEvent,
    BaseEvent,
    CtxMixin,
    SystemPromptEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from SimpleLLMFunc.loop.context import CompiledContext, ContextProvenance
from SimpleLLMFunc.loop.core import (
    EffectResolution,
    ExecutionMode,
    LoopPolicy,
    LoopState,
    Reduction,
    ResolutionStatus,
    RunStatus,
    Step,
)
from SimpleLLMFunc.loop.model_call import (
    ModelCallEffect,
    ModelCallResolution,
    ModelCallResolver,
)
from SimpleLLMFunc.loop.tool import FunctionTool
from SimpleLLMFunc.loop.tool_call import (
    EventSnapshot,
    ToolCallEffect,
    ToolCallResolution,
)
from SimpleLLMFunc.loop.tool_runtime import ToolCallResolver


_IMPLEMENTATION_REVISION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

ProjectedMessage: TypeAlias = tuple[
    UserMessage | AssistantMessage | ToolMessage,
    tuple[str, ...],
]


class ContextStrategyName(str, Enum):
    """Built-in deterministic context projections."""

    FULL = "full"
    RECENT = "recent"


class ContextStrategy(BaseModel):
    """Editable context projection used by the standard Loop compiler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ContextStrategyName = ContextStrategyName.FULL
    instruction: str = Field(
        default="Preserve relevant conversation context.",
        min_length=1,
    )
    recent_message_limit: int = Field(default=12, ge=1)

    @property
    def revision(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return (
            f"context/{self.name.value}@sha256:"
            f"{hashlib.sha256(encoded).hexdigest()}"
        )


class ContextStrategyEvent(BaseEvent, CtxMixin):
    """A context strategy activated for subsequent model calls."""

    type: Literal["context_strategy"] = "context_strategy"
    strategy: ContextStrategy


class LoopInput(BaseModel):
    """Input for one declarative Loop run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | list[ContentPart]
    context: ContextStrategy = ContextStrategy()


class LoopPhase(str, Enum):
    MODEL = "model"
    TOOLS = "tools"
    DONE = "done"


class LoopRunState(LoopState):
    """Semantic state of a declarative model/tool Loop."""

    model: str = Field(min_length=1)
    next_action: LoopPhase = LoopPhase.MODEL
    model_calls: int = Field(default=0, ge=0)
    output: str | None = None
    error: str | None = None

    @property
    def strategy(self) -> ContextStrategy:
        return self.event_view.latest(ContextStrategyEvent).strategy


def _content_text(content: str | list[ContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text: list[str] = []
    for part in content:
        if isinstance(part, (InputTextPart, OutputTextPart)):
            text.append(part.text)
        elif (
            isinstance(part, OutputAudioPart)
            and part.output_audio.transcript is not None
        ):
            text.append(part.output_audio.transcript)
    return "\n".join(text)


class _ContextCompiler:
    REVISION = f"declarative-context-compiler@sha256:{_IMPLEMENTATION_REVISION}"

    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools = tuple(tool.model_copy(deep=True) for tool in tools)

    def compile(self, state: LoopRunState) -> CompiledContext:
        system = state.event_view.latest(SystemPromptEvent)
        strategy_event = state.event_view.latest(ContextStrategyEvent)
        strategy = strategy_event.strategy
        system_content = (
            f"{system.content}\n\nContext strategy ({strategy.name.value}):\n"
            f"{strategy.instruction}"
        )
        projected: list[list[ProjectedMessage]] = []
        results_by_call = {
            event.tool_call_id: event
            for event in state.events
            if isinstance(event, ToolResultEvent)
        }
        consumed_results: set[str] = set()

        def project_result(
            event: ToolResultEvent,
            assistant_id: str | None = None,
            force_user_message: bool = False,
        ) -> list[ProjectedMessage]:
            event_ids = (
                (assistant_id, event.id) if assistant_id is not None else (event.id,)
            )
            if force_user_message or event.result.requires_user_message:
                return [
                    (
                        AssistantMessage(
                            content=(
                                f"I will use the {event.tool_name} tool to inspect "
                                "its result."
                            )
                        ),
                        event_ids,
                    ),
                    (
                        UserMessage.model_validate(
                            {"content": event.result.content}
                        ),
                        event_ids,
                    ),
                ]
            assert isinstance(event.result.content, str)
            return [
                (
                    ToolMessage(
                        tool_call_id=event.tool_call_id,
                        content=event.result.content,
                    ),
                    event_ids,
                )
            ]

        for event in state.events:
            if isinstance(event, UserMessageEvent):
                projected.append([(UserMessage(content=event.content), (event.id,))])
            elif isinstance(event, AssistantMessageEvent):
                calls = tuple(event.tool_calls)
                results = [
                    results_by_call[call.id]
                    for call in calls
                    if call.id in results_by_call
                ]
                if results and len(results) != len(calls):
                    raise ValueError("Loop context contains an incomplete tool result batch")
                rewrite_batch = any(
                    result.result.requires_user_message for result in results
                )
                unit: list[ProjectedMessage] = []
                if not rewrite_batch or event.content is not None or event.refusal:
                    unit.append(
                        (
                            AssistantMessage(
                                content=event.content,
                                refusal=event.refusal,
                                name=event.name,
                                tool_calls=(
                                    None if rewrite_batch else list(calls) or None
                                ),
                            ),
                            (event.id,),
                        )
                    )
                for result in results:
                    consumed_results.add(result.id)
                    unit.extend(
                        project_result(
                            result,
                            event.id,
                            force_user_message=rewrite_batch,
                        )
                    )
                if unit:
                    projected.append(unit)
            elif isinstance(event, ToolResultEvent):
                continue

        result_ids = {
            event.id
            for event in state.events
            if isinstance(event, ToolResultEvent)
        }
        if consumed_results != result_ids:
            raise ValueError("Loop context contains an orphan tool result")

        if not projected or not isinstance(projected[0][0][0], UserMessage):
            raise ValueError("Loop context requires an initial user message")
        first_user = projected[0]
        history = projected[1:]
        if strategy.name is ContextStrategyName.RECENT:
            start = len(history)
            selected_count = 0
            while start > 0 and selected_count < strategy.recent_message_limit:
                start -= 1
                selected_count += len(history[start])
            history = history[start:]

        selected = [item for unit in [first_user, *history] for item in unit]
        messages: Conversation = [SystemMessage(content=system_content)]
        provenance = [
            ContextProvenance(
                path="messages.0",
                event_ids=(system.id, strategy_event.id),
            )
        ]
        for message, event_ids in selected:
            messages.append(message)
            provenance.append(
                ContextProvenance(
                    path=f"messages.{len(messages) - 1}",
                    event_ids=event_ids,
                )
            )

        return CompiledContext.create(
            request=Request(
                model=state.model,
                messages=messages,
                tools=[tool.model_copy(deep=True) for tool in self._tools] or None,
            ),
            source_revision=state.revision,
            compiler_revision=f"{self.REVISION}/{strategy.revision}",
            provenance=tuple(provenance),
        )


class Loop(LoopPolicy[LoopRunState, str | LoopInput, str]):
    """A declaratively configured model/tool Loop with explicit L1 artifacts."""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        tools: Sequence[FunctionTool[..., object]] = (),
        context_tool: FunctionTool[..., object] | None = None,
        max_model_calls: int = 12,
        tool_execution: ExecutionMode = ExecutionMode.SEQUENTIAL,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if not system_prompt:
            raise ValueError("system_prompt must not be empty")
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be at least 1")
        declared = tuple(tools)
        names = [tool.name for tool in declared]
        if len(names) != len(set(names)):
            raise ValueError("Loop tools must have unique names")
        if context_tool is not None and context_tool.name not in names:
            raise ValueError("context_tool must be included in tools")

        self.model = model
        self.system_prompt = system_prompt
        self.tools = declared
        self._tools_by_name = {tool.name: tool for tool in declared}
        self.context_tool = context_tool
        self.max_model_calls = max_model_calls
        self.tool_execution = tool_execution
        self.compiler = _ContextCompiler([tool.schema for tool in declared])
        declaration = {
            "model": model,
            "system_prompt": system_prompt,
            "tools": [tool.schema.model_dump(mode="json") for tool in declared],
            "tool_implementations": [tool.revision for tool in declared],
            "context_tool": context_tool.name if context_tool else None,
            "context_compiler": self.compiler.REVISION,
            "max_model_calls": max_model_calls,
            "model_resolver": ModelCallResolver.revision,
            "tool_execution": tool_execution.value,
            "tool_resolver": ToolCallResolver.revision,
        }
        encoded = json.dumps(
            declaration,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.policy_revision = f"loop@sha256:{hashlib.sha256(encoded).hexdigest()}"

    def initialize(
        self,
        value: str | LoopInput,
        *,
        run_id: str,
    ) -> LoopRunState:
        if not run_id:
            raise ValueError("run_id must not be empty")
        loop_input = LoopInput(content=value) if isinstance(value, str) else value
        events: tuple[BaseEvent, ...] = (
            SystemPromptEvent(id=f"{run_id}:system", content=self.system_prompt),
            ContextStrategyEvent(
                id=f"{run_id}:strategy:0",
                strategy=loop_input.context,
            ),
            UserMessageEvent(id=f"{run_id}:user:0", content=loop_input.content),
        )
        return LoopRunState(
            run_id=run_id,
            revision=0,
            status=RunStatus.READY,
            settled=False,
            events=events,
            model=self.model,
        )

    def can_step(self, state: LoopRunState) -> bool:
        return state.status is RunStatus.READY and state.next_action is not LoopPhase.DONE

    def step(self, state: LoopRunState) -> Step:
        if not self.can_step(state):
            raise ValueError("Loop state cannot calculate another Step")
        step_id = f"{state.run_id}:step:{state.revision}"
        if state.next_action is LoopPhase.MODEL:
            return Step(
                id=step_id,
                run_id=state.run_id,
                source_revision=state.revision,
                policy_revision=self.policy_revision,
                kind="model",
                effects=(
                    ModelCallEffect(
                        id=f"{step_id}:model",
                        attempt=1,
                        idempotency_key=f"{step_id}:model:1",
                        context=self.compiler.compile(state),
                    ),
                ),
            )

        pending = self._pending_tool_calls(state)
        if not pending:
            raise ValueError("tool Step requires pending ToolCallEvents")
        effects = tuple(
            ToolCallEffect(
                id=f"{step_id}:tool:{index}",
                attempt=1,
                idempotency_key=f"{step_id}:tool:{call.tool_call_id}:1",
                tool_call_id=call.tool_call_id,
                name=call.tool_name,
                arguments_json=call.arguments,
                event_snapshot=EventSnapshot.create(state.event_view),
            )
            for index, call in enumerate(pending)
        )
        return Step(
            id=step_id,
            run_id=state.run_id,
            source_revision=state.revision,
            policy_revision=self.policy_revision,
            kind="tools",
            effects=effects,
            execution=self.tool_execution,
        )

    def validate_step(self, state: LoopRunState, step: Step) -> None:
        super().validate_step(state, step)
        expected = self.step(state)
        if (
            step.id != expected.id
            or step.kind != expected.kind
            or step.execution is not expected.execution
            or len(step.effects) != len(expected.effects)
        ):
            raise ValueError("Step shape does not match declarative Loop state")
        if step.kind == "tools":
            if step != expected:
                raise ValueError("tool Step does not match pending tool calls")
            return

        effect = step.effects[0]
        expected_effect = expected.effects[0]
        if not isinstance(effect, ModelCallEffect) or not isinstance(
            expected_effect, ModelCallEffect
        ):
            raise TypeError("model Step requires one ModelCallEffect")
        if (
            effect.id != expected_effect.id
            or effect.attempt != expected_effect.attempt
            or effect.idempotency_key != expected_effect.idempotency_key
            or effect.timeout != expected_effect.timeout
        ):
            raise ValueError("model Effect metadata does not match Loop state")
        context = effect.context
        CompiledContext.model_validate(context.model_dump(mode="python"))
        expected_context = expected_effect.context
        if (
            context.source_revision != state.revision
            or context.compiler_revision != expected_context.compiler_revision
            or context.provenance != expected_context.provenance
            or context.request.model != state.model
            or (
                context.digest != expected_context.digest
                and context.parent_digest != expected_context.digest
            )
        ):
            raise ValueError("model context is not derived from the current Loop state")

    def reduce(
        self,
        state: LoopRunState,
        step: Step,
        resolutions: Sequence[EffectResolution],
    ) -> Reduction[LoopRunState]:
        self.validate_cycle(state, step, resolutions)
        if step.kind == "model":
            return self._reduce_model(state, step, resolutions[0])
        return self._reduce_tools(state, step, resolutions)

    def result(self, state: LoopRunState) -> str:
        if state.status is not RunStatus.COMPLETED or state.output is None:
            raise RuntimeError("Loop run is not complete")
        return state.output

    @staticmethod
    def _pending_tool_calls(state: LoopRunState) -> tuple[ToolCallEvent, ...]:
        latest_assistant = state.event_view.latest_or_none(AssistantMessageEvent)
        if latest_assistant is None:
            return ()
        return tuple(
            event
            for event in state.events
            if isinstance(event, ToolCallEvent)
            and event.causation_id == latest_assistant.id
        )

    @staticmethod
    def _validate_tool_calls(
        state: LoopRunState,
        message: AssistantMessage,
    ) -> str | None:
        calls = tuple(message.tool_calls or ())
        call_ids = [call.id for call in calls]
        if len(call_ids) != len(set(call_ids)):
            return "model produced duplicate tool-call IDs"
        previous_ids = {
            event.tool_call_id
            for event in state.events
            if isinstance(event, ToolCallEvent)
        }
        if previous_ids.intersection(call_ids):
            return "model reused a tool-call ID from an earlier turn"
        for index, call in enumerate(calls):
            try:
                ToolCallEffect(
                    id=f"validation:{index}",
                    attempt=1,
                    idempotency_key=f"validation:{index}",
                    tool_call_id=call.id,
                    name=call.function.name,
                    arguments_json=call.function.arguments,
                    event_snapshot=EventSnapshot.create(state.event_view),
                )
            except ValueError as exc:
                return f"invalid model tool call at index {index}: {exc}"
        return None

    def _reduce_model(
        self,
        state: LoopRunState,
        step: Step,
        resolution: EffectResolution,
    ) -> Reduction[LoopRunState]:
        if resolution.status is not ResolutionStatus.COMPLETED:
            status = (
                RunStatus.CANCELLED
                if resolution.status is ResolutionStatus.CANCELLED
                else RunStatus.FAILED
            )
            error = resolution.error
            next_state = state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "status": status,
                    "settled": True,
                    "next_action": LoopPhase.DONE,
                    "error": error.message if error else resolution.status.value,
                }
            )
            return Reduction.from_cycle(
                step=step,
                resolutions=(resolution,),
                next_state=next_state,
            )
        if not isinstance(resolution, ModelCallResolution):
            raise TypeError("completed model Effect requires ModelCallResolution")
        model_effect = step.effects[0]
        assert isinstance(model_effect, ModelCallEffect)
        if resolution.trace.context_digest != model_effect.context.digest:
            raise ValueError("model Resolution trace does not match compiled context")
        if (
            resolution.completion.model
            and resolution.trace.actual_model != resolution.completion.model
        ):
            raise ValueError("model Resolution trace does not match completion model")
        if not resolution.completion.choices:
            next_state = state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "status": RunStatus.FAILED,
                    "settled": True,
                    "next_action": LoopPhase.DONE,
                    "model_calls": state.model_calls + 1,
                    "error": "model completion contains no choices",
                }
            )
            return Reduction.from_cycle(
                step=step,
                resolutions=(resolution,),
                next_state=next_state,
            )

        choice = resolution.completion.choices[0]
        message = choice.message
        assistant_id = f"{step.id}:assistant"
        assistant_event = AssistantMessageEvent(
            id=assistant_id,
            causation_id=step.effects[0].id,
            content=message.content,
            refusal=message.refusal,
            name=message.name,
            tool_calls=tuple(message.tool_calls or ()),
        )
        expected_finish_reason = (
            FinishReason.TOOL_CALLS
            if message.tool_calls
            else FinishReason.STOP
        )
        invalid = (
            None
            if choice.finish_reason is expected_finish_reason
            else (
                "model completion has unexpected finish reason: "
                f"{choice.finish_reason.value if choice.finish_reason else 'none'}"
            )
        )
        if invalid is None:
            invalid = self._validate_tool_calls(state, message)
        if invalid is not None:
            next_state = state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "status": RunStatus.FAILED,
                    "settled": True,
                    "next_action": LoopPhase.DONE,
                    "model_calls": state.model_calls + 1,
                    "error": invalid,
                    "events": (*state.events, assistant_event),
                }
            )
            return Reduction.from_cycle(
                step=step,
                resolutions=(resolution,),
                next_state=next_state,
                events=(assistant_event,),
            )
        tool_events = tuple(
            ToolCallEvent(
                id=f"{step.id}:tool-call:{index}",
                causation_id=assistant_id,
                tool_call_id=call.id,
                tool_name=call.function.name,
                arguments=call.function.arguments,
            )
            for index, call in enumerate(message.tool_calls or ())
        )
        appended: tuple[BaseEvent, ...] = (assistant_event, *tool_events)
        model_calls = state.model_calls + 1
        if tool_events and model_calls >= self.max_model_calls:
            status = RunStatus.FAILED
            settled = True
            next_action = LoopPhase.DONE
            output = None
            error = "model-call limit reached before requested tools could run"
        elif tool_events:
            status = RunStatus.READY
            settled = False
            next_action = LoopPhase.TOOLS
            output = None
            error = None
        else:
            status = RunStatus.COMPLETED
            settled = True
            next_action = LoopPhase.DONE
            output = _content_text(message.content) or (message.refusal or "")
            error = None
        next_state = state.model_copy(
            update={
                "revision": state.revision + 1,
                "status": status,
                "settled": settled,
                "next_action": next_action,
                "model_calls": model_calls,
                "output": output,
                "error": error,
                "events": (*state.events, *appended),
            }
        )
        return Reduction.from_cycle(
            step=step,
            resolutions=(resolution,),
            next_state=next_state,
            events=appended,
        )

    def _reduce_tools(
        self,
        state: LoopRunState,
        step: Step,
        resolutions: Sequence[EffectResolution],
    ) -> Reduction[LoopRunState]:
        appended: list[BaseEvent] = []
        cancelled = False
        reduction_error: str | None = None
        for index, (effect, resolution) in enumerate(
            zip(step.effects, resolutions, strict=True)
        ):
            assert isinstance(effect, ToolCallEffect)
            is_error = resolution.status is not ResolutionStatus.COMPLETED
            if resolution.status is ResolutionStatus.CANCELLED:
                cancelled = True
            if resolution.status is ResolutionStatus.COMPLETED:
                if not isinstance(resolution, ToolCallResolution):
                    raise TypeError(
                        "completed tool Effect requires ToolCallResolution"
                    )
                if (
                    resolution.tool_call_id != effect.tool_call_id
                    or resolution.name != effect.name
                ):
                    raise ValueError(
                        "tool Resolution metadata does not match ToolCallEffect"
                    )
                tool = self._tools_by_name.get(effect.name)
                if (
                    tool is None
                    or resolution.tool_revision != tool.revision
                    or resolution.event_digest != effect.event_snapshot.digest
                ):
                    raise ValueError(
                        "tool Resolution compilation evidence does not match Effect"
                    )
            value = (
                resolution.value
                if isinstance(resolution, ToolCallResolution)
                else None
            )
            if is_error:
                error = resolution.error
                result = ToolResult(
                    content=json.dumps(
                        {
                            "error": (
                                error.model_dump(mode="json") if error else None
                            ),
                            "status": resolution.status.value,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
            else:
                try:
                    tool = self._tools_by_name[effect.name]
                    tool.restore_result(value)
                    assert isinstance(resolution, ToolCallResolution)
                    assert resolution.result is not None
                    result = resolution.result
                except (KeyError, TypeError, ValueError) as exc:
                    is_error = True
                    reduction_error = f"invalid tool result: {exc}"
                    result = ToolResult(
                        content=json.dumps(
                            {
                                "error": {
                                    "message": reduction_error,
                                    "type": "InvalidToolResult",
                                },
                                "status": ResolutionStatus.FAILED.value,
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                    )
            appended.append(
                ToolResultEvent(
                    id=f"{step.id}:tool-result:{index}",
                    causation_id=effect.id,
                    tool_call_id=effect.tool_call_id,
                    tool_name=effect.name,
                    result=result,
                    is_error=is_error,
                )
            )

            if (
                self.context_tool is not None
                and effect.name == self.context_tool.name
                and resolution.status is ResolutionStatus.COMPLETED
                and reduction_error is None
            ):
                try:
                    restored = self.context_tool.restore_result(value)
                    strategy = ContextStrategy.model_validate(restored)
                except ValueError as exc:
                    reduction_error = f"invalid context tool result: {exc}"
                else:
                    appended.append(
                        ContextStrategyEvent(
                            id=f"{step.id}:strategy:{index}",
                            causation_id=effect.id,
                            strategy=strategy,
                        )
                    )

        if reduction_error is not None:
            status = RunStatus.FAILED
            settled = True
            next_action = LoopPhase.DONE
        elif cancelled:
            status = RunStatus.CANCELLED
            settled = True
            next_action = LoopPhase.DONE
        else:
            status = RunStatus.READY
            settled = False
            next_action = LoopPhase.MODEL
        events = tuple(appended)
        next_state = state.model_copy(
            update={
                "revision": state.revision + 1,
                "status": status,
                "settled": settled,
                "next_action": next_action,
                "error": reduction_error,
                "events": (*state.events, *events),
            }
        )
        return Reduction.from_cycle(
            step=step,
            resolutions=resolutions,
            next_state=next_state,
            events=events,
        )
