"""New ReAct core loop contracts and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, cast

from SimpleLLMFunc.base.compile_pipeline import compile_invocation_turn
from SimpleLLMFunc.base.context_compile import CompiledContext, ContextState, compile_context, clone_messages
from SimpleLLMFunc.base.context_source import CompileSource
from SimpleLLMFunc.base.llm_call import SingleLLMCallResult, SingleLLMPhaseResultYield, execute_single_llm_phase
from SimpleLLMFunc.base.mutation import ContextMutation
from SimpleLLMFunc.base.react_hooks import (
    ReActHookExecutionContext,
    collect_react_context_mutations,
    run_react_hook,
)
from SimpleLLMFunc.base.tool_scheduler import ToolSchedulerResult, schedule_tool_batch
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.hooks.event_bus import EventBus
from SimpleLLMFunc.hooks.events import (
    ReactEndEvent,
    ReactIterationEndEvent,
    ReactIterationStartEvent,
    ReactStartEvent,
    ReActEventType,
)
from SimpleLLMFunc.hooks.stream import EventYield, ReactOutput, ResponseYield
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.logger import push_debug
from SimpleLLMFunc.logger.context_manager import get_current_trace_id
from SimpleLLMFunc.logger.logger import get_current_context_attribute, get_location
from SimpleLLMFunc.observability.langfuse_client import (
    coerce_langfuse_metadata,
    get_langfuse_trace_context,
    langfuse_client,
)
from SimpleLLMFunc.llm_decorator.invocation_spec import InvocationSpec
from SimpleLLMFunc.type.message import MessageList, NormalizedMessageList
from SimpleLLMFunc.type.tool_call import ToolDefinitionList


@dataclass
class ReactLoopState:
    context_state: ContextState
    pending_mutations: List[ContextMutation] = field(default_factory=list)
    iteration: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0


def _build_react_loop_invocation_spec(
    *,
    compile_source: Optional[CompileSource],
    messages: MessageList,
    tool_prompt_specs: Optional[List[Dict[str, Any]]],
    include_must_principles: bool,
    stream: bool,
    trace_id: str,
    func_name: str,
    llm_kwargs: Dict[str, Any],
):
    from SimpleLLMFunc.llm_decorator.invocation_spec import PromptContract, TranscriptSeed

    if compile_source is not None:
        prompt_contract = PromptContract(
            base_instruction=compile_source.data_from_agent_config.base_system_prompt,
            tool_prompt_specs=list(compile_source.data_from_agent_config.tool_prompt_specs),
            include_must_principles=compile_source.data_from_agent_config.include_must_principles,
        )
        template_params = compile_source.data_from_agent_config.template_params
        data_from_selfref = compile_source.data_from_selfref
    else:
        prompt_contract = PromptContract(
            base_instruction="",
            tool_prompt_specs=list(tool_prompt_specs or []),
            include_must_principles=include_must_principles,
        )
        template_params = None
        data_from_selfref = None

    return InvocationSpec(
        mode="chat",
        func_name=func_name,
        trace_id=trace_id,
        docstring=prompt_contract.base_instruction,
        bound_args={},
        type_hints={},
        return_type=None,
        template_params=template_params,
        llm_kwargs=dict(llm_kwargs),
        stream=stream,
        return_mode="text",
        prompt_contract=prompt_contract,
        transcript_seed=TranscriptSeed(initial_messages=cast(NormalizedMessageList, messages)),
        data_from_selfref=data_from_selfref,
    )


async def run_react_loop(
    *,
    llm_interface: LLM_Interface,
    messages: MessageList,
    compile_source: Optional[CompileSource] = None,
    tools: ToolDefinitionList,
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    max_tool_calls: Optional[int],
    stream: bool,
    trace_id: str,
    user_task_prompt: str,
    abort_signal: Optional[AbortSignal],
    hooks: Any,
    llm_kwargs: Dict[str, Any],
    tool_prompt_specs: Optional[List[Dict[str, Any]]] = None,
    include_must_principles: bool = False,
    invocation_spec: Optional[InvocationSpec] = None,
) -> AsyncGenerator[ReactOutput, None]:
    func_name = get_current_context_attribute("function_name") or "Unknown Function"
    current_trace_id = trace_id or get_current_trace_id() or f"trace_{int(time.time() * 1000)}"
    model_parameters = {k: v for k, v in llm_kwargs.items() if k != "retry_times"}
    trace_context = get_langfuse_trace_context()
    event_bus = EventBus(session_id=current_trace_id, agent_call_id=f"agent_{current_trace_id}")
    start_time = time.time()

    state = ReActHookExecutionContext(
        trace_id=current_trace_id,
        func_name=func_name,
        user_task_prompt=user_task_prompt,
        messages=messages.copy(),
        llm_kwargs=dict(llm_kwargs),
        stream=stream,
    )
    await run_react_hook(hooks, "on_run_start", state)

    loop_state = ReactLoopState(
        context_state=ContextState(
            messages=state.messages,
            data_from_selfref=(compile_source.data_from_selfref if compile_source is not None else None),
        )
    )

    async def _emit_event(event: Any, **kwargs: Any) -> EventYield:
        return await event_bus.emit_and_get(event, **kwargs)

    yield await _emit_event(
        ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id=current_trace_id,
            func_name=func_name,
            iteration=0,
            user_task_prompt=user_task_prompt,
            initial_messages=state.messages.copy(),
            available_tools=tools,
        )
    )

    while True:
        loop_state.pending_mutations.extend(
            await collect_react_context_mutations(hooks, state)
        )
        compiled = compile_context(loop_state.context_state, loop_state.pending_mutations)
        loop_state.context_state = ContextState(
            messages=compiled.messages,
            data_from_selfref=compiled.data_from_selfref,
        )
        loop_state.pending_mutations = []

        state.iteration = loop_state.iteration
        state.messages = compiled.messages
        state.total_llm_calls = loop_state.total_llm_calls
        state.total_tool_calls = loop_state.total_tool_calls
        await run_react_hook(hooks, "before_llm_call", state)
        compiled = CompiledContext(
            messages=state.messages,
            data_from_selfref=loop_state.context_state.data_from_selfref,
        )
        turn_spec = invocation_spec or _build_react_loop_invocation_spec(
            compile_source=compile_source,
            messages=compiled.messages,
            tool_prompt_specs=tool_prompt_specs,
            include_must_principles=include_must_principles,
            stream=stream,
            trace_id=current_trace_id,
            func_name=func_name,
            llm_kwargs=llm_kwargs,
        )
        compiled_turn = compile_invocation_turn(
            turn_spec,
            compiled.messages,
            [],
            compiled.data_from_selfref,
        )
        llm_input_messages = compiled_turn.llm_messages

        if loop_state.iteration > 0:
            yield await _emit_event(
                ReactIterationStartEvent(
                    event_type=ReActEventType.REACT_ITERATION_START,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=current_trace_id,
                    func_name=func_name,
                    iteration=loop_state.iteration,
                    current_messages=llm_input_messages.copy(),
                )
            )

        loop_state.total_llm_calls += 1
        llm_result = SingleLLMCallResult()
        llm_start = time.time()

        with langfuse_client.start_as_current_observation(
            as_type="generation",
            name=f"{func_name}_llm_call_{loop_state.total_llm_calls}",
            input=llm_input_messages,
            model=llm_interface.model_name,
            model_parameters=model_parameters,
            metadata=coerce_langfuse_metadata(
                {
                    "stream": stream,
                    "iteration": loop_state.iteration,
                    "tools_available": len(tools) if tools else 0,
                }
            ),
            completion_start_time=datetime.now(timezone.utc),
            trace_context=trace_context,
        ) as generation_span:
            async for output in execute_single_llm_phase(
                llm_interface=llm_interface,
                messages=llm_input_messages,
                tools=tools,
                llm_kwargs=llm_kwargs,
                trace_id=current_trace_id,
                func_name=func_name,
                iteration=loop_state.iteration,
                stream=stream,
                emit_event=_emit_event,
                abort_signal=abort_signal,
            ):
                if isinstance(output, SingleLLMPhaseResultYield):
                    llm_result = output.result
                    continue
                if isinstance(output, ResponseYield):
                    yield output
                    continue
                yield output

            state.messages = compiled.messages
            state.last_response = llm_result.response
            state.content = llm_result.content
            state.tool_calls = list(llm_result.tool_calls)
            state.reasoning_details = list(llm_result.reasoning_details)
            state.usage = llm_result.usage
            state.aborted = llm_result.aborted
            await run_react_hook(hooks, "after_llm_call", state)
            generation_span.update(
                output={"content": llm_result.content, "tool_calls": llm_result.tool_calls},
            )

        llm_mutations = list(llm_result.mutations)

        if llm_result.aborted:
            compiled_final = compile_context(loop_state.context_state, llm_mutations)
            loop_state.context_state = ContextState(
                messages=clone_messages(compiled_final.messages),
                data_from_selfref=compiled_final.data_from_selfref,
            )
            state.iteration = loop_state.iteration
            state.messages = compiled_final.messages
            state.final_response = llm_result.content
            state.aborted = True
            await run_react_hook(hooks, "before_finalize", state)
            react_end = ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=datetime.now(timezone.utc),
                trace_id=current_trace_id,
                func_name=func_name,
                iteration=loop_state.iteration,
                final_response=state.final_response or "",
                final_messages=state.messages.copy(),
                total_iterations=loop_state.iteration,
                total_execution_time=time.time() - start_time,
                total_tool_calls=loop_state.total_tool_calls,
                total_llm_calls=loop_state.total_llm_calls,
                total_token_usage=llm_result.usage,
                extra={"aborted": True, "abort_reason": abort_signal.reason if abort_signal is not None else ""},
            )
            yield await _emit_event(react_end)
            return

        if not llm_result.tool_calls:
            compiled_final = compile_context(loop_state.context_state, llm_mutations)
            loop_state.context_state = ContextState(
                messages=clone_messages(compiled_final.messages),
                data_from_selfref=compiled_final.data_from_selfref,
            )
            state.iteration = loop_state.iteration
            state.messages = compiled_final.messages
            state.final_response = llm_result.content
            state.aborted = False
            await run_react_hook(hooks, "before_finalize", state)
            react_end = ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=datetime.now(timezone.utc),
                trace_id=current_trace_id,
                func_name=func_name,
                iteration=loop_state.iteration,
                final_response=state.final_response or "",
                final_messages=state.messages.copy(),
                total_iterations=loop_state.iteration,
                total_execution_time=time.time() - start_time,
                total_tool_calls=loop_state.total_tool_calls,
                total_llm_calls=loop_state.total_llm_calls,
                total_token_usage=llm_result.usage,
            )
            yield await _emit_event(react_end)
            return

        if max_tool_calls is not None and loop_state.total_tool_calls + len(llm_result.tool_calls) > max_tool_calls:
            final_compiled = compile_context(loop_state.context_state, [])
            final_call_result = SingleLLMCallResult()
            loop_state.total_llm_calls += 1
            state.iteration = loop_state.iteration + 1
            state.messages = final_compiled.messages
            state.total_llm_calls = loop_state.total_llm_calls
            await run_react_hook(hooks, "before_llm_call", state)

            with langfuse_client.start_as_current_observation(
                as_type="generation",
                name=f"{func_name}_final_llm_call",
                input=compile_invocation_turn(
                    invocation_spec or _build_react_loop_invocation_spec(
                        compile_source=compile_source,
                        messages=state.messages,
                        tool_prompt_specs=tool_prompt_specs,
                        include_must_principles=include_must_principles,
                        stream=False,
                        trace_id=current_trace_id,
                        func_name=func_name,
                        llm_kwargs=llm_kwargs,
                    ),
                    state.messages,
                    [],
                    final_compiled.data_from_selfref,
                ).llm_messages,
                model=llm_interface.model_name,
                model_parameters=model_parameters,
                metadata=coerce_langfuse_metadata(
                    {
                        "stream": False,
                        "reason": "max_tool_calls_reached",
                        "iteration": loop_state.iteration + 1,
                    }
                ),
                completion_start_time=datetime.now(timezone.utc),
                trace_context=trace_context,
            ) as generation_span:
                final_llm_input = compile_invocation_turn(
                    invocation_spec or _build_react_loop_invocation_spec(
                        compile_source=compile_source,
                        messages=state.messages,
                        tool_prompt_specs=tool_prompt_specs,
                        include_must_principles=include_must_principles,
                        stream=False,
                        trace_id=current_trace_id,
                        func_name=func_name,
                        llm_kwargs=llm_kwargs,
                    ),
                    state.messages,
                    [],
                    final_compiled.data_from_selfref,
                ).llm_messages
                async for output in execute_single_llm_phase(
                    llm_interface=llm_interface,
                    messages=final_llm_input,
                    tools=None,
                    llm_kwargs=llm_kwargs,
                    trace_id=current_trace_id,
                    func_name=func_name,
                    iteration=loop_state.iteration + 1,
                    stream=False,
                    emit_event=_emit_event,
                    abort_signal=abort_signal,
                ):
                    if isinstance(output, SingleLLMPhaseResultYield):
                        final_call_result = output.result
                        continue
                    yield output

                state.messages = final_compiled.messages
                state.last_response = final_call_result.response
                state.content = final_call_result.content
                state.tool_calls = list(final_call_result.tool_calls)
                state.reasoning_details = list(final_call_result.reasoning_details)
                state.usage = final_call_result.usage
                state.aborted = final_call_result.aborted
                await run_react_hook(hooks, "after_llm_call", state)
                generation_span.update(
                    output={
                        "content": final_call_result.content,
                        "tool_calls": final_call_result.tool_calls,
                    }
                )

            final_mutations = list(final_call_result.mutations)
            compiled_final = compile_context(loop_state.context_state, final_mutations)
            loop_state.context_state = ContextState(
                messages=clone_messages(compiled_final.messages),
                data_from_selfref=compiled_final.data_from_selfref,
            )
            state.iteration = loop_state.iteration + 1
            state.messages = compiled_final.messages
            state.final_response = final_call_result.content
            state.aborted = False
            await run_react_hook(hooks, "before_finalize", state)
            react_end = ReactEndEvent(
                event_type=ReActEventType.REACT_END,
                timestamp=datetime.now(timezone.utc),
                trace_id=current_trace_id,
                func_name=func_name,
                iteration=loop_state.iteration + 1,
                final_response=state.final_response or "",
                final_messages=state.messages.copy(),
                total_iterations=loop_state.iteration + 1,
                total_execution_time=time.time() - start_time,
                total_tool_calls=loop_state.total_tool_calls,
                total_llm_calls=loop_state.total_llm_calls,
                total_token_usage=final_call_result.usage,
            )
            yield await _emit_event(react_end)
            return

        compiled_tool_context = CompiledContext(
            messages=compiled.messages.copy(),
            data_from_selfref=compiled.data_from_selfref,
        )
        state.iteration = loop_state.iteration + 1
        state.messages = compiled_tool_context.messages
        await run_react_hook(hooks, "before_tool_batch", state)
        pre_tool_messages = clone_messages(state.messages)

        scheduler_result = ToolSchedulerResult()
        async for item in schedule_tool_batch(
            tool_calls=llm_result.tool_calls,
            messages=cast(NormalizedMessageList, state.messages),
            tool_map=tool_map,
            trace_id=current_trace_id,
            func_name=func_name,
            iteration=loop_state.iteration + 1,
            event_bus=event_bus,
            abort_signal=abort_signal,
        ):
            if isinstance(item, EventYield):
                yield item
            else:
                scheduler_result = item

        loop_state.pending_mutations = llm_mutations + scheduler_result.mutations
        loop_state.total_tool_calls += scheduler_result.total_tool_calls
        state.total_tool_calls = loop_state.total_tool_calls
        active_messages_after_tools = clone_messages(state.messages)

        if active_messages_after_tools != pre_tool_messages:
            state.messages = active_messages_after_tools
        else:
            compiled_after_tools = compile_context(
                loop_state.context_state,
                loop_state.pending_mutations,
            )
            state.messages = compiled_after_tools.messages
        await run_react_hook(hooks, "after_tool_batch", state)
        # Hooks may replace the compiled context entirely (for example selfref
        # compaction). After this point the hook-updated state becomes the next
        # baseline, and no pre-hook pending mutations should be replayed again.
        loop_state.context_state = ContextState(
            messages=clone_messages(state.messages),
            data_from_selfref=(
                compiled_after_tools.data_from_selfref
                if active_messages_after_tools == pre_tool_messages
                else compiled_tool_context.data_from_selfref
            ),
        )
        loop_state.pending_mutations = []

        yield await _emit_event(
            ReactIterationEndEvent(
                event_type=ReActEventType.REACT_ITERATION_END,
                timestamp=datetime.now(timezone.utc),
                trace_id=current_trace_id,
                func_name=func_name,
                iteration=loop_state.iteration + 1,
                messages=state.messages.copy(),
                iteration_time=time.time() - llm_start,
                tool_calls_count=scheduler_result.total_tool_calls,
            )
        )

        loop_state.iteration += 1
        push_debug(
            f"LLM function '{func_name}' completed loop iteration {loop_state.iteration}",
            location=get_location(),
        )


__all__ = ["ReactLoopState", "run_react_loop"]
