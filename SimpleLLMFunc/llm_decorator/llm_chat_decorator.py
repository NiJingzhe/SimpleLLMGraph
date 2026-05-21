from __future__ import annotations

import inspect
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    cast,
)

from SimpleLLMFunc.base.react_loop import ReAct_loop
from SimpleLLMFunc.base.types import CompileSource, DataFromAgentConfig
from SimpleLLMFunc.hooks.events import ReactEndEvent
from SimpleLLMFunc.hooks.stream import ReactOutput, ResponseYield, is_event_yield
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.llm_decorator.chat_call_context import build_chat_call_context
from SimpleLLMFunc.llm_decorator.chat_selfref import (
    build_must_principles_prompt_block as _build_must_principles_prompt_block,
    create_selfref_session,
    extract_first_system_prompt_from_messages as _extract_first_system_prompt_from_messages,
    extract_raw_history_reference as _extract_raw_history_reference,
    finalize_self_reference_history as _finalize_self_reference_history,
    react_end_event_has_fork_origin as _react_end_event_has_fork_origin,
    remove_injected_prompt_blocks as _remove_injected_prompt_blocks,
    remove_must_principles_prompt_block as _remove_must_principles_prompt_block,
    remove_prompt_block as _remove_prompt_block,
    remove_runtime_primitive_prompt_block as _remove_runtime_primitive_prompt_block,
    resolve_runtime_self_reference_key,
    resolve_self_reference_key as _resolve_self_reference_key,
    seed_self_reference_system_prompt_if_missing as _seed_self_reference_system_prompt_if_missing,
    set_history_argument as _set_history_argument,
)
from SimpleLLMFunc.llm_decorator.chat_toolkit import (
    clone_toolkit_for_fork as _clone_toolkit_for_fork,
    close_fork_cloned_pyrepls as _close_fork_cloned_pyrepls,
    extract_self_reference_from_toolkit as _extract_self_reference_from_toolkit,
    resolve_effective_self_reference as _resolve_effective_self_reference,
    resolve_runtime_toolkit as _resolve_runtime_toolkit,
)
from SimpleLLMFunc.llm_decorator.chat_types import (
    AGENT_FORK_TOOLKIT_FACTORY_ATTR,
    AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR,
    DEFAULT_MAX_TOOL_CALLS,
    ToolkitList,
)
from SimpleLLMFunc.llm_decorator.invocation_builder import build_chat_invocation_spec
from SimpleLLMFunc.llm_decorator.prompt_contract import HISTORY_PARAM_NAMES
from SimpleLLMFunc.llm_decorator.signature import setup_log_context
from SimpleLLMFunc.llm_decorator.utils import process_tools
from SimpleLLMFunc.observability.langfuse_client import (
    coerce_langfuse_metadata,
    get_langfuse_trace_context,
    langfuse_client,
    propagate_langfuse_trace_name,
    update_langfuse_parent_span,
    update_langfuse_trace_name,
)
from SimpleLLMFunc.runtime.selfref.state import MemoryHistory, SelfReference
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.type import HistoryList, MessageList, NormalizedMessageParam

T = TypeVar("T")
P = ParamSpec("P")


def _build_chat_compile_source(invocation_spec: Any) -> CompileSource:
    """Build the compatibility CompileSource payload from an InvocationSpec."""

    return CompileSource(
        data_from_agent_config=DataFromAgentConfig(
            base_system_prompt=invocation_spec.prompt_contract.base_instruction,
            template_params=(
                dict(invocation_spec.template_params)
                if invocation_spec.template_params is not None
                else None
            ),
            tool_prompt_specs=list(invocation_spec.prompt_contract.tool_prompt_specs),
            include_must_principles=invocation_spec.prompt_contract.include_must_principles,
        ),
        data_from_selfref=invocation_spec.data_from_selfref,
        input_messages=invocation_spec.transcript_seed.initial_messages,
    )


class LLMChat:
    """Callable object returned by :func:`llm_chat`.

    The public call contract remains an async generator.  The instance provides
    a stable internal agent identity for SelfRef binding and fork rebinding.
    """

    def __init__(
        self,
        func: Union[Callable[P, Any], Callable[P, Awaitable[Any]]],
        *,
        llm_interface: LLM_Interface,
        toolkit: Optional[ToolkitList],
        max_tool_calls: Optional[int],
        stream: bool,
        strict_signature: bool,
        self_reference: Optional[SelfReference],
        self_reference_key: Optional[str],
        llm_kwargs: Dict[str, Any],
    ) -> None:
        self.func = func
        self.llm_interface = llm_interface
        self.toolkit = toolkit
        self.max_tool_calls = max_tool_calls
        self.stream = stream
        self.strict_signature = strict_signature
        self.self_reference = self_reference
        self.self_reference_key = self_reference_key
        self.llm_kwargs = dict(llm_kwargs)
        self.signature_meta = inspect.signature(func)
        self.func_name = func.__name__

        if strict_signature:
            self._validate_strict_signature()

        self.resolved_default_self_reference_key: Optional[str] = None
        if self_reference is not None or self_reference_key is not None:
            self.resolved_default_self_reference_key = _resolve_self_reference_key(
                self_reference_key,
                self.func_name,
            )

        self.__wrapped__ = func
        self.__name__ = func.__name__
        self.__qualname__ = getattr(func, "__qualname__", func.__name__)
        self.__doc__ = func.__doc__
        self.__annotations__ = getattr(func, "__annotations__", {})
        self.__module__ = getattr(func, "__module__", __name__)
        self.__signature__ = self.signature_meta
        setattr(self, AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR, True)
        setattr(self, AGENT_FORK_TOOLKIT_FACTORY_ATTR, self._build_fork_toolkit)

        if self.self_reference is not None:
            self.self_reference.bind_agent_instance(
                self,
                default_memory_key=self.resolved_default_self_reference_key,
            )

    def _validate_strict_signature(self) -> None:
        signature = self.signature_meta
        parameters = list(signature.parameters.values())

        allowed_extra_param_names = {"_template_params"}

        if len(parameters) < 2:
            raise TypeError(
                "llm_chat(strict_signature=True) requires function signature "
                "`agent(history, message: str, ...)` with at least two parameters."
            )

        if any(
            param.kind
            in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            for param in parameters
        ):
            raise TypeError(
                "llm_chat(strict_signature=True) does not allow *args/**kwargs. "
                "Use `agent(history, message: str, _template_params=None)`."
            )

        history_param = parameters[0]
        message_param = parameters[1]

        if history_param.name not in HISTORY_PARAM_NAMES:
            raise TypeError(
                "llm_chat(strict_signature=True) requires the first parameter to be "
                "`history` or `chat_history` (see HISTORY_PARAM_NAMES)."
            )

        message_annotation = message_param.annotation
        if (
            message_annotation is inspect.Signature.empty
            or message_annotation is None
            or (
                message_annotation is not str
                and not (
                    isinstance(message_annotation, str) and message_annotation == "str"
                )
            )
        ):
            raise TypeError(
                "llm_chat(strict_signature=True) requires the second parameter to be "
                "annotated as `str` for the user message, e.g. "
                "`async def agent(history, message: str, ...)`."
            )

        if message_param.name != "message":
            raise TypeError(
                "llm_chat(strict_signature=True) requires the second parameter name to be "
                "`message` so fork delegation can pass it by keyword."
            )

        for extra_param in parameters[2:]:
            if extra_param.name in allowed_extra_param_names:
                continue
            raise TypeError(
                "llm_chat(strict_signature=True) only allows an optional `_template_params` "
                "parameter in addition to `(history, message: str)`. "
                f"Unexpected parameter: {extra_param.name!r}."
            )

    def _build_fork_toolkit(self, parent_toolkit: Any) -> Optional[ToolkitList]:
        candidate_toolkit: Optional[ToolkitList]
        if isinstance(parent_toolkit, list):
            candidate_toolkit = cast(Optional[ToolkitList], parent_toolkit)
        else:
            candidate_toolkit = self.toolkit

        effective_for_fork = _resolve_effective_self_reference(
            self.self_reference,
            candidate_toolkit,
        )

        return _clone_toolkit_for_fork(
            candidate_toolkit,
            effective_for_fork,
        )

    async def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[ReactOutput, None]:
        call_context = build_chat_call_context(
            func=self.func,
            args=args,
            kwargs=kwargs,
            default_toolkit=self.toolkit,
            explicit_self_reference=self.self_reference,
        )
        function_signature = call_context.signature
        template_params = call_context.template_params
        runtime_toolkit = call_context.runtime_toolkit
        effective_self_reference = call_context.effective_self_reference

        toolkit_context_token = None
        active_memory_key_token = None
        active_template_params_token = None
        react_hooks = None
        previous_runtime_toolkit: Any = None
        previous_memory_key: Optional[str] = None
        if effective_self_reference is not None:
            previous_runtime_toolkit = (
                effective_self_reference._get_active_runtime_toolkit()
            )
            toolkit_context_token = (
                effective_self_reference._set_active_runtime_toolkit(runtime_toolkit)
            )
            active_template_params_token = (
                effective_self_reference._set_active_template_params(template_params)
            )

        try:
            async with setup_log_context(
                func_name=function_signature.func_name,
                trace_id=function_signature.trace_id,
                arguments=function_signature.bound_args.arguments,
            ):
                trace_context = get_langfuse_trace_context()
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"{function_signature.func_name}_chat_call",
                    input=function_signature.bound_args.arguments,
                    metadata=coerce_langfuse_metadata(
                        {
                            "function_name": function_signature.func_name,
                            "trace_id": function_signature.trace_id,
                            "tools_available": len(runtime_toolkit)
                            if runtime_toolkit
                            else 0,
                            "max_tool_calls": self.max_tool_calls,
                            "stream": self.stream,
                            "self_reference_enabled": (
                                effective_self_reference is not None
                            ),
                            "self_reference_key": self.self_reference_key,
                        }
                    ),
                    trace_context=trace_context,
                ) as chat_span:
                    update_langfuse_trace_name(function_signature.func_name)
                    with propagate_langfuse_trace_name(function_signature.func_name):
                        update_langfuse_parent_span(
                            langfuse_client.get_current_observation_id()
                        )
                        try:
                            raw_history_reference = _extract_raw_history_reference(
                                function_signature.bound_args.arguments
                            )

                            resolved_self_reference_key: Optional[str] = None
                            baseline_history_count = 0

                            if effective_self_reference is not None:
                                previous_memory_key = (
                                    effective_self_reference._get_active_memory_key()
                                )
                                resolved_self_reference_key = (
                                    resolve_runtime_self_reference_key(
                                        explicit_key=self.self_reference_key,
                                        func_name=function_signature.func_name,
                                        template_params=template_params,
                                    )
                                )

                                effective_self_reference.bind_agent_instance(
                                    self,
                                    default_memory_key=resolved_self_reference_key,
                                )

                                if raw_history_reference is not None:
                                    effective_self_reference.bind_history(
                                        resolved_self_reference_key,
                                        cast(
                                            List[Dict[str, Any]],
                                            raw_history_reference,
                                        ),
                                    )
                                elif not effective_self_reference.has_history(
                                    resolved_self_reference_key
                                ):
                                    effective_self_reference.bind_history(
                                        resolved_self_reference_key,
                                        [],
                                    )

                                history_snapshot = (
                                    effective_self_reference.snapshot_context_messages(
                                        resolved_self_reference_key
                                    )
                                )
                                _set_history_argument(
                                    function_signature.bound_args.arguments,
                                    history_snapshot,
                                )
                                baseline_history_count = (
                                    effective_self_reference.filtered_history_count(
                                        resolved_self_reference_key
                                    )
                                )

                                active_memory_key_token = (
                                    effective_self_reference._set_active_memory_key(
                                        resolved_self_reference_key
                                    )
                                )

                            selfref_session = None
                            if (
                                effective_self_reference is not None
                                and resolved_self_reference_key is not None
                            ):
                                selfref_session = create_selfref_session(
                                    backend=effective_self_reference,
                                    memory_key=resolved_self_reference_key,
                                    template_params=template_params,
                                    runtime_toolkit=runtime_toolkit,
                                    raw_history_reference=raw_history_reference,
                                    agent_instance=self,
                                    baseline_history_count=baseline_history_count,
                                )

                            invocation_spec = build_chat_invocation_spec(
                                signature=function_signature,
                                template_params=template_params,
                                llm_kwargs=self.llm_kwargs,
                                stream=self.stream,
                                runtime_toolkit=runtime_toolkit,
                                selfref_session=selfref_session,
                                raw_history_reference=raw_history_reference,
                            )
                            messages = invocation_spec.transcript_seed.initial_messages
                            compile_source = _build_chat_compile_source(invocation_spec)

                            if selfref_session is not None:
                                _seed_self_reference_system_prompt_if_missing(
                                    effective_self_reference,
                                    cast(str, resolved_self_reference_key),
                                    messages,
                                )
                                react_hooks = selfref_session

                            tool_param, tool_map = process_tools(
                                runtime_toolkit,
                                function_signature.func_name,
                            )
                            response_stream = ReAct_loop(
                                llm_interface=self.llm_interface,
                                messages=messages,
                                compile_source=compile_source,
                                tools=tool_param,
                                tool_map=tool_map,
                                max_tool_calls=self.max_tool_calls,
                                stream=self.stream,
                                trace_id=function_signature.trace_id,
                                user_task_prompt=call_context.user_task_prompt,
                                abort_signal=call_context.abort_signal,
                                hooks=react_hooks,
                                invocation_spec=invocation_spec,
                                toolkit=runtime_toolkit,
                                **self.llm_kwargs,
                            )

                            collected_responses = []
                            final_history = None

                            typed_event_stream = cast(
                                AsyncGenerator[ReactOutput, None],
                                response_stream,
                            )
                            async for raw_output in typed_event_stream:
                                if isinstance(raw_output, tuple):
                                    raw_response, history = raw_output
                                    output = ResponseYield(
                                        type="response",
                                        response=raw_response,
                                        messages=cast(MessageList, history),
                                    )
                                    if (
                                        effective_self_reference is not None
                                        and resolved_self_reference_key is not None
                                    ):
                                        active_history = _finalize_self_reference_history(
                                            effective_self_reference,
                                            resolved_self_reference_key,
                                            cast(MemoryHistory, history),
                                            baseline_history_count=baseline_history_count,
                                            base_system_prompt=compile_source.data_from_agent_config.base_system_prompt,
                                        )
                                        output.messages = cast(
                                            MessageList, active_history
                                        )
                                        final_history = cast(
                                            HistoryList,
                                            cast(
                                                List[NormalizedMessageParam],
                                                active_history,
                                            ),
                                        )
                                        if raw_history_reference is not None:
                                            raw_history_reference[:] = cast(
                                                List[Dict[str, Any]], active_history
                                            )
                                else:
                                    output = raw_output

                                if (
                                    effective_self_reference is not None
                                    and resolved_self_reference_key is not None
                                    and is_event_yield(output)
                                    and isinstance(output.event, ReactEndEvent)
                                    and not _react_end_event_has_fork_origin(
                                        output.event,
                                        output.origin,
                                    )
                                ):
                                    active_history = _finalize_self_reference_history(
                                        effective_self_reference,
                                        resolved_self_reference_key,
                                        cast(
                                            MemoryHistory, output.event.final_messages
                                        ),
                                        baseline_history_count=baseline_history_count,
                                        base_system_prompt=compile_source.data_from_agent_config.base_system_prompt,
                                    )
                                    output.event.final_messages = cast(
                                        HistoryList,
                                        cast(
                                            List[NormalizedMessageParam], active_history
                                        ),
                                    )
                                    final_history = cast(
                                        HistoryList,
                                        cast(
                                            List[NormalizedMessageParam], active_history
                                        ),
                                    )
                                    if raw_history_reference is not None:
                                        raw_history_reference[:] = cast(
                                            List[Dict[str, Any]], active_history
                                        )

                                yield output

                            chat_span.update(
                                output={
                                    "responses": collected_responses,
                                    "final_history": final_history,
                                    "total_responses": len(collected_responses),
                                },
                            )
                        except Exception as exc:
                            chat_span.update(
                                output={"error": str(exc)},
                            )
                            raise
        finally:
            if react_hooks is not None:
                react_hooks.close()
            _close_fork_cloned_pyrepls(runtime_toolkit)
            if (
                effective_self_reference is not None
                and active_memory_key_token is not None
            ):
                try:
                    effective_self_reference._reset_active_memory_key(
                        active_memory_key_token
                    )
                except ValueError:
                    if previous_memory_key is None:
                        effective_self_reference._active_memory_key_var.set(None)
                    else:
                        effective_self_reference._active_memory_key_var.set(
                            previous_memory_key
                        )
            if (
                effective_self_reference is not None
                and active_template_params_token is not None
            ):
                try:
                    effective_self_reference._reset_active_template_params(
                        active_template_params_token
                    )
                except ValueError:
                    effective_self_reference._active_template_params_var.set(None)
            if (
                effective_self_reference is not None
                and toolkit_context_token is not None
            ):
                try:
                    effective_self_reference._reset_active_runtime_toolkit(
                        toolkit_context_token
                    )
                except ValueError:
                    effective_self_reference._active_runtime_toolkit_var.set(
                        previous_runtime_toolkit
                    )


def llm_chat(
    llm_interface: LLM_Interface,
    toolkit: Optional[ToolkitList] = None,
    max_tool_calls: Optional[int] = DEFAULT_MAX_TOOL_CALLS,
    stream: bool = False,
    strict_signature: bool = False,
    self_reference: Optional[SelfReference] = None,
    self_reference_key: Optional[str] = None,
    **llm_kwargs: Any,
) -> Callable[
    [Union[Callable[P, Any], Callable[P, Awaitable[Any]]]],
    Callable[P, AsyncGenerator[ReactOutput, None]],
]:
    """
    Async LLM chat decorator for implementing asynchronous conversational interactions with
    large language models, with support for tool calling and conversation history management.

    This decorator returns a callable object whose ``__call__`` produces the
    same AsyncGenerator as the previous function wrapper.  Keeping the
    decorated value as an instance gives SelfRef a stable agent identity for
    rebinding and fork execution while preserving function-like metadata.
    """

    def decorator(
        func: Union[Callable[P, Any], Callable[P, Awaitable[Any]]],
    ) -> Callable[P, AsyncGenerator[ReactOutput, None]]:
        return cast(
            Callable[P, AsyncGenerator[ReactOutput, None]],
            LLMChat(
                func,
                llm_interface=llm_interface,
                toolkit=toolkit,
                max_tool_calls=max_tool_calls,
                stream=stream,
                strict_signature=strict_signature,
                self_reference=self_reference,
                self_reference_key=self_reference_key,
                llm_kwargs=dict(llm_kwargs),
            ),
        )

    return decorator


async_llm_chat = llm_chat

__all__ = [
    "DEFAULT_MAX_TOOL_CALLS",
    "LLMChat",
    "ToolkitList",
    "async_llm_chat",
    "llm_chat",
    "_build_must_principles_prompt_block",
    "_clone_toolkit_for_fork",
    "_close_fork_cloned_pyrepls",
    "_extract_first_system_prompt_from_messages",
    "_extract_raw_history_reference",
    "_extract_self_reference_from_toolkit",
    "_finalize_self_reference_history",
    "_react_end_event_has_fork_origin",
    "_remove_injected_prompt_blocks",
    "_remove_must_principles_prompt_block",
    "_remove_prompt_block",
    "_remove_runtime_primitive_prompt_block",
    "_resolve_effective_self_reference",
    "_resolve_runtime_toolkit",
    "_resolve_self_reference_key",
    "_seed_self_reference_system_prompt_if_missing",
    "_set_history_argument",
]
