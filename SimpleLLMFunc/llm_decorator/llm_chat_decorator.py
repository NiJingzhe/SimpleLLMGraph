import inspect
import json
from functools import wraps
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
    Literal,
)

from SimpleLLMFunc.base.react_loop import ReAct_loop
from SimpleLLMFunc.llm_decorator.signature import (
    parse_function_signature,
    setup_log_context,
)
from SimpleLLMFunc.llm_decorator.invocation_builder import build_chat_invocation_spec
from SimpleLLMFunc.runtime.selfref.session import SelfRefSession
from SimpleLLMFunc.llm_decorator.prompt_contract import HISTORY_PARAM_NAMES
from SimpleLLMFunc.llm_decorator.utils import (
    process_tools,
    remove_tool_best_practices_prompt_block,
)
from SimpleLLMFunc.base.types import CompileSource, DataFromAgentConfig
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.runtime.selfref.state import (
    MemoryHistory,
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM,
    SelfReference,
)
from SimpleLLMFunc.runtime.selfref.state import _coerce_history_list
from SimpleLLMFunc.runtime.selfref.context_ops import parse_data_from_selfref
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.type import HistoryList, MessageList, NormalizedMessageParam
from SimpleLLMFunc.hooks.events import ReactEndEvent
from SimpleLLMFunc.hooks.stream import ReactOutput, ResponseYield, is_event_yield
from SimpleLLMFunc.hooks.abort import AbortSignal, ABORT_SIGNAL_PARAM
from SimpleLLMFunc.observability.langfuse_client import (
    coerce_langfuse_metadata,
    get_langfuse_trace_context,
    langfuse_client,
    propagate_langfuse_trace_name,
    update_langfuse_parent_span,
    update_langfuse_trace_name,
)

# Type aliases
ToolkitList = List[Union[Tool, Callable[..., Awaitable[Any]]]]

# Type variables
T = TypeVar("T")
P = ParamSpec("P")

# Constants
DEFAULT_MAX_TOOL_CALLS: Optional[int] = None  # No default tool-call cap
_RUNTIME_PRIMITIVE_PROMPT_BLOCK_START = "<runtime_primitive_contract>"
_RUNTIME_PRIMITIVE_PROMPT_BLOCK_END = "</runtime_primitive_contract>"
_LEGACY_SELF_REFERENCE_PROMPT_BLOCK_START = "[SelfReference Memory Contract]"
_LEGACY_SELF_REFERENCE_PROMPT_BLOCK_END = "[/SelfReference Memory Contract]"
_MUST_PRINCIPLES_PROMPT_BLOCK_START = "<must_principles>"
_MUST_PRINCIPLES_PROMPT_BLOCK_END = "</must_principles>"
_AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR = "__simplellmfunc_accepts_template_params__"
_AGENT_FORK_TOOLKIT_FACTORY_ATTR = "__simplellmfunc_fork_toolkit_factory__"
_FORK_CLONED_PYREPL_ATTR = "__simplellmfunc_fork_cloned_pyrepl__"


def _extract_raw_history_reference(
    arguments: dict[str, Any],
) -> Optional[MemoryHistory]:
    """Extract the original history list object from bound call arguments."""

    for history_param_name in HISTORY_PARAM_NAMES:
        if history_param_name not in arguments:
            continue

        history = arguments[history_param_name]
        if history is None:
            return None

        if isinstance(history, list) and all(
            isinstance(item, dict) for item in history
        ):
            return cast(MemoryHistory, history)

        return None

    return None


def _set_history_argument(
    arguments: dict[str, Any],
    history: MemoryHistory,
) -> bool:
    """Inject a history snapshot into bound arguments when history param exists."""

    for history_param_name in HISTORY_PARAM_NAMES:
        if history_param_name in arguments:
            arguments[history_param_name] = list(history)
            return True
    return False


def _clone_toolkit_for_fork(
    base_toolkit: Optional[ToolkitList],
    self_reference_override: Optional[SelfReference],
) -> Optional[ToolkitList]:
    if base_toolkit is None:
        return None

    from SimpleLLMFunc.builtin import PyRepl

    cloned_toolkit: ToolkitList = []
    repl_clones: Dict[int, PyRepl] = {}
    backend_overrides: Optional[Dict[str, Any]] = None
    if self_reference_override is not None:
        backend_overrides = {
            PyRepl.DEFAULT_SELF_REFERENCE_BACKEND_NAME: self_reference_override,
        }

    for item in base_toolkit:
        if isinstance(item, Tool):
            bound_instance = getattr(item.func, "__self__", None)
            if isinstance(bound_instance, PyRepl):
                original_repl_id = id(bound_instance)
                if original_repl_id not in repl_clones:
                    replacement_repl = bound_instance._clone_for_fork(
                        backend_overrides=backend_overrides,
                    )
                    setattr(replacement_repl, _FORK_CLONED_PYREPL_ATTR, True)

                    repl_clones[original_repl_id] = replacement_repl

                replacement_repl = repl_clones[original_repl_id]
                replacement_tool = next(
                    tool for tool in replacement_repl.toolset if tool.name == item.name
                )
                cloned_toolkit.append(replacement_tool)
                continue

        cloned_toolkit.append(item)

    return cloned_toolkit


def _close_fork_cloned_pyrepls(toolkit: Optional[ToolkitList]) -> None:
    if not toolkit:
        return

    from SimpleLLMFunc.builtin import PyRepl

    closed_repl_ids: set[int] = set()

    for item in toolkit:
        if not isinstance(item, Tool):
            continue

        bound_instance = getattr(item.func, "__self__", None)
        if not isinstance(bound_instance, PyRepl):
            continue
        if not bool(getattr(bound_instance, _FORK_CLONED_PYREPL_ATTR, False)):
            continue

        repl_id = id(bound_instance)
        if repl_id in closed_repl_ids:
            continue
        closed_repl_ids.add(repl_id)

        try:
            bound_instance.close()
        except Exception:
            continue


def _extract_self_reference_from_toolkit(
    toolkit: Optional[ToolkitList],
) -> Optional[SelfReference]:
    if not toolkit:
        return None

    from SimpleLLMFunc.builtin import PyRepl

    discovered: Dict[int, SelfReference] = {}

    for item in toolkit:
        if not isinstance(item, Tool):
            continue

        bound_instance = getattr(item.func, "__self__", None)
        if not isinstance(bound_instance, PyRepl):
            continue

        default_backend = bound_instance.get_runtime_backend(
            PyRepl.DEFAULT_SELF_REFERENCE_BACKEND_NAME
        )
        if isinstance(default_backend, SelfReference):
            discovered[id(default_backend)] = default_backend
            continue

        for backend_name in bound_instance.list_runtime_backends():
            backend_value = bound_instance.get_runtime_backend(backend_name)
            if isinstance(backend_value, SelfReference):
                discovered[id(backend_value)] = backend_value

    if not discovered:
        return None

    return next(iter(discovered.values()))


def _resolve_effective_self_reference(
    explicit_self_reference: Optional[SelfReference],
    toolkit: Optional[ToolkitList],
) -> Optional[SelfReference]:
    if explicit_self_reference is not None:
        return explicit_self_reference
    return _extract_self_reference_from_toolkit(toolkit)


def _resolve_runtime_toolkit(
    default_toolkit: Optional[ToolkitList],
    template_params: Optional[Dict[str, Any]],
) -> Optional[ToolkitList]:
    if template_params is None:
        return default_toolkit

    override_toolkit = template_params.get(
        SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM
    )
    if override_toolkit is None:
        return default_toolkit

    if not isinstance(override_toolkit, list):
        raise ValueError("self_reference toolkit override must be a list")

    if not default_toolkit:
        return cast(Optional[ToolkitList], override_toolkit)

    # Merge: start with default tools, then add override tools.
    # When a tool with the same name exists in both, the override takes priority.
    from SimpleLLMFunc.tool import Tool

    seen_names: set[str] = set()
    merged: ToolkitList = []

    # Collect override tool names first (they take priority)
    for item in override_toolkit:
        if isinstance(item, Tool):
            seen_names.add(item.name)
        elif callable(item):
            seen_names.add(getattr(item, "__name__", ""))

    # Add default tools that are not overridden
    for item in default_toolkit:
        if isinstance(item, Tool):
            if item.name not in seen_names:
                merged.append(item)
        elif callable(item):
            name = getattr(item, "__name__", "")
            if name not in seen_names:
                merged.append(item)

    # Append all override tools
    merged.extend(override_toolkit)

    return merged


def _resolve_self_reference_key(
    explicit_key: Optional[str],
    func_name: str,
) -> str:
    if explicit_key is None:
        return func_name

    normalized = explicit_key.strip()
    if not normalized:
        raise ValueError("self_reference_key must be a non-empty string")
    return normalized


def _remove_prompt_block(system_prompt: str, start_marker: str, end_marker: str) -> str:
    cleaned_prompt = system_prompt

    while True:
        start_index = cleaned_prompt.find(start_marker)
        if start_index < 0:
            break

        end_index = cleaned_prompt.find(end_marker, start_index)
        if end_index < 0:
            cleaned_prompt = cleaned_prompt[:start_index]
            break

        cleaned_prompt = (
            cleaned_prompt[:start_index] + cleaned_prompt[end_index + len(end_marker) :]
        )

    return cleaned_prompt


def _remove_runtime_primitive_prompt_block(system_prompt: str) -> str:
    cleaned_prompt = system_prompt
    for start_marker, end_marker in (
        (
            _RUNTIME_PRIMITIVE_PROMPT_BLOCK_START,
            _RUNTIME_PRIMITIVE_PROMPT_BLOCK_END,
        ),
        (
            _LEGACY_SELF_REFERENCE_PROMPT_BLOCK_START,
            _LEGACY_SELF_REFERENCE_PROMPT_BLOCK_END,
        ),
    ):
        cleaned_prompt = _remove_prompt_block(
            cleaned_prompt,
            start_marker,
            end_marker,
        )

    return cleaned_prompt.strip()


def _remove_injected_prompt_blocks(system_prompt: str) -> str:
    cleaned_prompt = _remove_runtime_primitive_prompt_block(system_prompt)
    cleaned_prompt = remove_tool_best_practices_prompt_block(cleaned_prompt)
    cleaned_prompt = _remove_must_principles_prompt_block(cleaned_prompt)
    return cleaned_prompt.strip()


def _build_must_principles_prompt_block() -> str:
    lines = [
        _MUST_PRINCIPLES_PROMPT_BLOCK_START,
        "<rule>Invoke tools through native structured tool_calls / function-calling fields.</rule>",
        "<rule>Use assistant content for natural-language reasoning and final responses.</rule>",
        "<rule>Keep tool invocation payloads in the native tool channel.</rule>",
        _MUST_PRINCIPLES_PROMPT_BLOCK_END,
    ]
    return "\n".join(lines)


def _remove_must_principles_prompt_block(system_prompt: str) -> str:
    cleaned_prompt = _remove_prompt_block(
        system_prompt,
        _MUST_PRINCIPLES_PROMPT_BLOCK_START,
        _MUST_PRINCIPLES_PROMPT_BLOCK_END,
    )
    return cleaned_prompt.strip()


def _extract_first_system_prompt_from_messages(messages: MessageList) -> Optional[str]:
    for message in messages:
        if message.get("role") != "system":
            continue

        content = message.get("content")
        if isinstance(content, str):
            return content

    return None


def _seed_self_reference_system_prompt_if_missing(
    self_reference: SelfReference,
    memory_key: str,
    messages: MessageList,
) -> None:
    if self_reference.get_system_prompt(memory_key) is not None:
        return

    system_prompt = _extract_first_system_prompt_from_messages(messages)
    if system_prompt is None:
        return

    # Keep memory store focused on durable prompt content. If the system prompt
    # already contains auto-injected tool/runtime guidance blocks, store a
    # cleaned version and let llm_chat inject guidance again per turn.
    cleaned_system_prompt = _remove_injected_prompt_blocks(system_prompt)
    system_prompt_for_store = cleaned_system_prompt or system_prompt

    current_history: MemoryHistory = self_reference.snapshot_history(memory_key)
    seeded_history: MemoryHistory = [
        {"role": "system", "content": system_prompt_for_store},
        *current_history,
    ]
    self_reference.replace_history(
        key=memory_key,
        messages=seeded_history,
        strict=False,
    )


def _react_end_event_has_fork_origin(event: ReactEndEvent, origin: Any) -> bool:
    origin_fork_id = getattr(origin, "fork_id", None)
    if isinstance(origin_fork_id, str) and origin_fork_id:
        return True

    event_extra = getattr(event, "extra", None)
    if not isinstance(event_extra, dict):
        return False

    raw_origin = event_extra.get("origin")
    if not isinstance(raw_origin, dict):
        return False

    raw_fork_id = raw_origin.get("fork_id")
    return isinstance(raw_fork_id, str) and bool(raw_fork_id)


def _finalize_self_reference_history(
    self_reference: SelfReference,
    memory_key: str,
    history: MemoryHistory,
    *,
    baseline_history_count: int,
    base_system_prompt: str,
) -> MemoryHistory:
    resolved_history = cast(MemoryHistory, _coerce_history_list(history))
    resolved_source = parse_data_from_selfref(resolved_history)
    if (
        resolved_source.experiences
        or resolved_source.summary is not None
        or resolved_source.summary_message is not None
    ):
        return self_reference.store_history(memory_key, resolved_history)

    self_reference.consume_destructive_history_mutation(memory_key)
    resolved_history = self_reference.merge_turn_history(
        key=memory_key,
        baseline_history_count=baseline_history_count,
        updated_history=resolved_history,
        commit=True,
    )
    return self_reference.store_history(memory_key, resolved_history)


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
        setattr(self, _AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR, True)
        setattr(self, _AGENT_FORK_TOOLKIT_FACTORY_ATTR, self._build_fork_toolkit)

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
        abort_signal = kwargs.pop(ABORT_SIGNAL_PARAM, None)
        if not isinstance(abort_signal, AbortSignal):
            abort_signal = None
        # Step 1: 解析函数签名
        function_signature, template_params = parse_function_signature(
            self.func,
            args,
            kwargs,
        )
        runtime_toolkit = _resolve_runtime_toolkit(self.toolkit, template_params)
        effective_self_reference = _resolve_effective_self_reference(
            self.self_reference,
            runtime_toolkit,
        )

        # 构建用户任务提示（用于事件）
        user_task_prompt = json.dumps(
            function_signature.bound_args.arguments,
            default=str,
            ensure_ascii=False,
        )

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
            # Step 2: 设置日志上下文
            async with setup_log_context(
                func_name=function_signature.func_name,
                trace_id=function_signature.trace_id,
                arguments=function_signature.bound_args.arguments,
            ):
                trace_context = get_langfuse_trace_context()
                # 创建 Langfuse parent span
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
                                runtime_self_reference_key = self.self_reference_key
                                if template_params is not None:
                                    override_key = template_params.get(
                                        SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM
                                    )
                                    if override_key is not None:
                                        if not isinstance(override_key, str):
                                            raise ValueError(
                                                "self_reference key override must be a non-empty string"
                                            )
                                        runtime_self_reference_key = override_key

                                resolved_self_reference_key = (
                                    _resolve_self_reference_key(
                                        runtime_self_reference_key,
                                        function_signature.func_name,
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

                            # Step 3: 构建 invocation spec 与 selfref session
                            selfref_session = None
                            if (
                                effective_self_reference is not None
                                and resolved_self_reference_key is not None
                            ):
                                selfref_session = SelfRefSession(
                                    backend=effective_self_reference,
                                    memory_key=resolved_self_reference_key,
                                    template_params=template_params,
                                    runtime_toolkit=runtime_toolkit,
                                    raw_history_reference=raw_history_reference,
                                    agent_instance=self,
                                )
                                selfref_session.history_authority = (
                                    "external"
                                    if raw_history_reference is not None
                                    else (
                                        "selfref"
                                        if effective_self_reference.has_history(
                                            resolved_self_reference_key
                                        )
                                        else "seed"
                                    )
                                )
                                selfref_session.baseline_history_count = (
                                    baseline_history_count
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
                            # Compatibility payload for existing step tests / patched runners.
                            # The formal LLM conversion still uses invocation_spec -> compile_pipeline.
                            compile_source = CompileSource(
                                data_from_agent_config=DataFromAgentConfig(
                                    base_system_prompt=invocation_spec.prompt_contract.base_instruction,
                                    template_params=(
                                        dict(invocation_spec.template_params)
                                        if invocation_spec.template_params is not None
                                        else None
                                    ),
                                    tool_prompt_specs=list(
                                        invocation_spec.prompt_contract.tool_prompt_specs
                                    ),
                                    include_must_principles=invocation_spec.prompt_contract.include_must_principles,
                                ),
                                data_from_selfref=invocation_spec.data_from_selfref,
                                input_messages=invocation_spec.transcript_seed.initial_messages,
                            )

                            if selfref_session is not None:
                                _seed_self_reference_system_prompt_if_missing(
                                    effective_self_reference,
                                    resolved_self_reference_key,
                                    messages,
                                )
                                react_hooks = selfref_session

                            # Step 4: 执行 ReAct 循环（流式）
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
                                user_task_prompt=user_task_prompt,
                                abort_signal=abort_signal,
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
                                            base_system_prompt=(
                                                compile_source.data_from_agent_config.base_system_prompt
                                                if compile_source is not None
                                                else ""
                                            ),
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
                                        base_system_prompt=(
                                            compile_source.data_from_agent_config.base_system_prompt
                                            if compile_source is not None
                                            else ""
                                        ),
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
                            # 更新 span 错误信息
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
