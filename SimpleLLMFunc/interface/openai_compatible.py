from __future__ import annotations

import asyncio
import json
import os
from inspect import isawaitable
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Optional,
    cast,
    override,
)

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails as OAITokensDetails

from SimpleLLMFunc.cancellation import (
    CancellationToken,
    await_with_cancellation,
)
from SimpleLLMFunc.context.ir import (
    AssistantMessage,
    Chunk,
    Choice,
    Completion,
    CompletionChoice,
    CompletionTokensDetails,
    Conversation,
    Delta,
    FinishReason,
    Image,
    InputAudioPart,
    InputTextPart,
    OutputTextPart,
    ReasoningEffort,
    Request as IRRequest,
    Role,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolCallDeltaFunction,
    ToolCallFunction,
    Usage,
    UserMessage,
)
from SimpleLLMFunc.context.ir._enums import ToolCallType
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.interface.llm_interface import DEFAULT_CONTEXT_WINDOW, LLM_Interface
from SimpleLLMFunc.interface.token_bucket import rate_limit_manager
from SimpleLLMFunc.logger import (
    app_log,
    get_current_context_attribute,
    get_current_trace_id,
    push_critical,
    push_debug,
    push_error,
    push_warning,
    set_current_context_attribute,
)


# --------------------------------------------------------------------------- #
# IR -> OpenAI wire format translation (inline, pure functions)
# --------------------------------------------------------------------------- #


def _ir_messages_to_openai(messages: Conversation) -> list[dict[str, object]]:
    """Translate IR :class:`Conversation` to the OpenAI ``messages`` wire list."""
    out: list[dict[str, object]] = []
    for message in messages:
        if isinstance(message, UserMessage) and isinstance(message.content, list):
            content: list[dict[str, object]] = []
            for part in message.content:
                if isinstance(part, InputTextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, Image):
                    image_url = part.image_url
                    payload = (
                        {"url": image_url}
                        if isinstance(image_url, str)
                        else image_url.model_dump(mode="json", exclude_none=True)
                    )
                    content.append({"type": "image_url", "image_url": payload})
                elif isinstance(part, InputAudioPart):
                    content.append(part.model_dump(mode="json"))
                else:
                    raise ValueError(
                        "OpenAI Chat API does not support this user content part"
                    )
            out.append({"role": "user", "content": content})
            continue
        if isinstance(message, AssistantMessage) and isinstance(
            message.content, list
        ):
            assistant_content: list[dict[str, object]] = []
            for part in message.content:
                if not isinstance(part, OutputTextPart):
                    raise ValueError(
                        "OpenAI Chat API cannot losslessly replay this assistant "
                        "content part"
                    )
                assistant_content.append({"type": "text", "text": part.text})
            payload = message.model_dump(
                mode="json",
                exclude={"content"},
                exclude_none=True,
            )
            payload["content"] = assistant_content
            out.append(payload)
            continue
        out.append(message.model_dump(mode="json", exclude_none=True))
    return out


def _ir_tools_to_openai(tools: Optional[list[Tool]]) -> Optional[list[dict[str, object]]]:
    if not tools:
        return None
    return [t.model_dump(mode="json") for t in tools]


def _request_to_create_kwargs(
    request: IRRequest,
    *,
    stream: bool,
    stream_options: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Translate an IR :class:`Request` to ``client.chat.completions.create``
    kwargs (excluding ``model`` / ``stream`` / ``timeout``, which the adapter
    adds separately).

    The returned dict is typed ``dict[str, Any]`` because it is splatted into
    the OpenAI SDK's overloaded ``create`` call; the SDK needs per-value
    ``Any`` to pick the right overload and narrow each parameter type.
    """
    kwargs: dict[str, Any] = {
        "messages": _ir_messages_to_openai(request.messages),
    }
    tools = _ir_tools_to_openai(request.tools)
    if tools is not None:
        kwargs["tools"] = tools
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if (
        request.reasoning_effort is not None
        and request.reasoning_effort is not ReasoningEffort.NONE
    ):
        kwargs["reasoning_effort"] = request.reasoning_effort.value
    if stream and stream_options is not None:
        kwargs["stream_options"] = stream_options
    if request.extra:
        for k, v in request.extra.items():
            kwargs[k] = v
    return kwargs


def _usage_from_openai(usage: Optional[CompletionUsage]) -> Optional[Usage]:
    if usage is None:
        return None
    prompt_tokens = int(usage.prompt_tokens)
    completion_tokens = int(usage.completion_tokens)
    total_tokens = int(usage.total_tokens)
    reasoning_tokens: Optional[int] = None
    details: Optional[OAITokensDetails] = usage.completion_tokens_details
    if details is not None:
        reasoning_tokens = details.reasoning_tokens
    completion_tokens_details = (
        CompletionTokensDetails(reasoning_tokens=reasoning_tokens)
        if reasoning_tokens is not None
        else None
    )
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_tokens_details=completion_tokens_details,
    )


def _tool_calls_from_openai(
    openai_tool_calls: Optional[list[ChatCompletionMessageToolCall]],
) -> Optional[list[ToolCall]]:
    if not openai_tool_calls:
        return None
    out: list[ToolCall] = []
    for tc in openai_tool_calls:
        fn = tc.function
        out.append(
            ToolCall(
                id=tc.id or "",
                type=ToolCallType(tc.type or "function"),
                function=ToolCallFunction(name=fn.name or "", arguments=fn.arguments or ""),
            )
        )
    return out


def _completion_from_chat_completion(cc: ChatCompletion) -> Completion:
    choices: list[CompletionChoice] = []
    for ch in cc.choices or []:
        msg = ch.message
        assistant = AssistantMessage(
            content=msg.content,
            tool_calls=_tool_calls_from_openai(
                cast(Optional[list[ChatCompletionMessageToolCall]], msg.tool_calls)
            ),
            refusal=msg.refusal,
        )
        fr = ch.finish_reason
        finish_reason = FinishReason(fr) if fr else None
        choices.append(
            CompletionChoice(
                index=ch.index,
                message=assistant,
                finish_reason=finish_reason,
                logprobs=ch.logprobs,
            )
        )
    return Completion(
        id=cc.id,
        created=cc.created,
        model=cc.model,
        choices=choices,
        usage=_usage_from_openai(cc.usage),
    )


def _chunk_from_chat_completion_chunk(ch: ChatCompletionChunk) -> Chunk:
    choices: list[Choice] = []
    for c in ch.choices or []:
        d = c.delta
        role: Optional[Role] = None
        if d.role is not None:
            try:
                role = Role(d.role)
            except ValueError:
                role = None
        tool_calls: Optional[list[ToolCallDelta]] = None
        if d.tool_calls:
            tool_calls = []
            for otc in d.tool_calls:
                fn = otc.function
                tc_type: Optional[ToolCallType] = None
                if otc.type is not None:
                    tc_type = ToolCallType(otc.type)
                tool_calls.append(
                    ToolCallDelta(
                        index=otc.index,
                        id=otc.id,
                        type=tc_type,
                        function=ToolCallDeltaFunction(
                            name=fn.name if fn is not None else None,
                            arguments=fn.arguments or "" if fn is not None else "",
                        ),
                    )
                )
        fr = c.finish_reason
        finish_reason = FinishReason(fr) if fr else None
        choices.append(
            Choice(
                index=c.index,
                delta=Delta(
                    role=role,
                    content=d.content,
                    reasoning=None,
                    tool_calls=tool_calls,
                    refusal=d.refusal if hasattr(d, "refusal") else None,
                ),
                finish_reason=finish_reason,
                logprobs=c.logprobs,
            )
        )
    return Chunk(
        id=ch.id,
        created=ch.created,
        model=ch.model,
        choices=choices,
        usage=_usage_from_openai(ch.usage),
    )


# --------------------------------------------------------------------------- #
# Provider config TypedDict (for load_from_json_file)
# --------------------------------------------------------------------------- #


class _ProviderModelInfo:
    """Typed view of a single model entry in the provider JSON config."""

    model_name: str
    api_keys: list[str]
    base_url: str
    context_window: int
    max_retries: int
    retry_delay: float
    rate_limit_capacity: int
    rate_limit_refill_rate: float
    api_params: Optional[dict[str, Any]]

    def __init__(self, raw: dict[str, Any]) -> None:
        self.model_name = str(raw["model_name"])
        self.api_keys = [str(k) for k in cast(list[Any], raw["api_keys"])]
        self.base_url = str(raw["base_url"])
        self.context_window = int(cast(int, raw.get("context_window", DEFAULT_CONTEXT_WINDOW)))
        self.max_retries = int(cast(int, raw.get("max_retries", 5)))
        self.retry_delay = float(cast(float, raw.get("retry_delay", 1.0)))
        self.rate_limit_capacity = int(cast(int, raw.get("rate_limit_capacity", 10)))
        self.rate_limit_refill_rate = float(cast(float, raw.get("rate_limit_refill_rate", 1.0)))
        ap = raw.get("api_params")
        self.api_params = cast(Optional[dict[str, Any]], ap) if isinstance(ap, dict) else None


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


class OpenAICompatible(LLM_Interface):
    """OpenAI Chat-Completions-compatible adapter.

    Speaks the neutral context IR on its inner side: :meth:`chat` consumes an
    IR :class:`Request` and returns an IR :class:`Completion`;
    :meth:`chat_stream` consumes an IR :class:`Request` and yields IR
    :class:`Chunk` items. Translation to/from the OpenAI wire format happens
    inline via the module-level helpers above.
    """

    def __init__(
        self,
        api_key_pool: APIKeyPool,
        model_name: str,
        base_url: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        rate_limit_capacity: int = 10,
        rate_limit_refill_rate: float = 1.0,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        api_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            api_key_pool,
            model_name,
            base_url=base_url,
            context_window=context_window,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self.model_name = model_name
        self.key_pool = api_key_pool

        bucket_id = f"{base_url}_{model_name}"
        self.token_bucket = rate_limit_manager.get_or_create_bucket(
            bucket_id=bucket_id,
            capacity=rate_limit_capacity,
            refill_rate=rate_limit_refill_rate,
        )

        self._api_params: dict[str, Any] = dict(api_params) if api_params else {}

        initial_key = api_key_pool.get_least_loaded_key()
        self.client: Optional[AsyncOpenAI] = AsyncOpenAI(
            api_key=initial_key,
            base_url=self.base_url,
        )
        self._current_key = initial_key
        self._clients: dict[str, AsyncOpenAI] = {initial_key: self.client}
        self._stream_finish_grace_timeout = 1.0

    def __repr__(self) -> str:
        return (
            f"OpenAICompatible(model_name={self.model_name}, base_url={self.base_url})"
        )

    def get_rate_limit_status(self) -> dict[str, Any]:
        return self.token_bucket.get_info()

    def reset_rate_limit(self) -> None:
        self.token_bucket.reset()

    @classmethod
    def load_from_json_file(
        cls, json_path: str
    ) -> dict[str, dict[str, "OpenAICompatible"]]:
        if not os.path.exists(json_path):
            push_critical(
                f"JSON 文件 {json_path} 不存在。请检查您的配置。",
            )
            raise FileNotFoundError(f"JSON 文件 {json_path} 不存在。")

        with open(json_path, "r", encoding="utf-8") as f:
            json_str = f.read()
        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError as e:
            push_critical(f"Failed to parse JSON string: {e}")
            raise ValueError(f"Failed to parse JSON string: {e}") from e

        if not isinstance(raw, dict):
            push_critical(
                f"Top-level JSON must be an object, got {type(raw).__name__}",
            )
            raise TypeError(
                f"Top-level JSON must be an object, got {type(raw).__name__}"
            )

        raw_providers: dict[str, Any] = cast(dict[str, Any], raw)
        all_providers_dict: dict[str, dict[str, OpenAICompatible]] = {}
        try:
            for provider_id, models in raw_providers.items():
                all_providers_dict[provider_id] = {}
                app_log(
                    f"Loading OpenAICompatible instances for provider: {provider_id}",
                )

                if not isinstance(models, list):
                    push_critical(
                        f"Invalid model format under provider {provider_id}. Expected a list.",
                    )
                    raise TypeError(
                        f"Invalid model format under provider {provider_id}. Expected a list."
                    )

                models_list: list[Any] = cast(list[Any], models)
                for model_raw in models_list:
                    if not isinstance(model_raw, dict):
                        raise TypeError(
                            f"Invalid model entry under provider {provider_id}. Expected an object."
                        )
                    info = _ProviderModelInfo(
                        cast(dict[str, Any], model_raw)
                    )

                    key_pool = APIKeyPool(info.api_keys, f"{provider_id}-{info.model_name}")
                    instance = cls(
                        api_key_pool=key_pool,
                        model_name=info.model_name,
                        base_url=info.base_url,
                        context_window=info.context_window,
                        max_retries=info.max_retries,
                        retry_delay=info.retry_delay,
                        rate_limit_capacity=info.rate_limit_capacity,
                        rate_limit_refill_rate=info.rate_limit_refill_rate,
                        api_params=info.api_params,
                    )
                    all_providers_dict[provider_id][info.model_name] = instance
                    app_log(
                        f"Loaded OpenAICompatible instance for provider {provider_id} model {info.model_name}",
                    )
        except (ValueError, TypeError, KeyError) as e:
            push_critical(
                f"Error while loading OpenAICompatible instances: {e}",
            )
            raise ValueError(f"Error while loading OpenAICompatible instances: {e}") from e
        except Exception as e:
            push_critical(
                f"Unknown error while loading OpenAICompatible instances: {e}",
            )
            raise ValueError(
                f"Unknown error while loading OpenAICompatible instances: {e}"
            ) from e

        return all_providers_dict

    async def _get_or_create_client(
        self,
        key: str,
        cancellation: CancellationToken | None = None,
    ) -> AsyncOpenAI:
        if cancellation is not None and cancellation.cancelled:
            raise asyncio.CancelledError
        if self._current_key == key and self.client is not None:
            self._clients[key] = self.client
            return self.client
        client = self._clients.get(key)
        if client is None:
            client = AsyncOpenAI(api_key=key, base_url=self.base_url)
            self._clients[key] = client
        self.client = client
        self._current_key = key
        return client

    async def aclose(self) -> None:
        clients = tuple({id(client): client for client in self._clients.values()}.values())
        closed: set[int] = set()
        try:
            for client in clients:
                try:
                    await client.close()
                except Exception:
                    continue
                closed.add(id(client))
        finally:
            self._clients = {
                key: client
                for key, client in self._clients.items()
                if id(client) not in closed
            }
            if self.client is not None and id(self.client) in closed:
                replacement = next(iter(self._clients.items()), None)
                if replacement is None:
                    self.client = None
                else:
                    self._current_key, self.client = replacement

    async def _close_stream_response(self, response: object) -> None:
        close_method = getattr(response, "close", None)
        if callable(close_method):
            try:
                close_result = close_method()
                if isawaitable(close_result):
                    await cast(Awaitable[None], close_result)
            except Exception as close_exc:
                push_warning(
                    f"{self.model_name} failed to close stream response: {close_exc}",
                )
            return

        aclose_method = getattr(response, "aclose", None)
        if callable(aclose_method):
            try:
                aclose_result = aclose_method()
                if isawaitable(aclose_result):
                    await cast(Awaitable[None], aclose_result)
            except Exception as close_exc:
                push_warning(
                    f"{self.model_name} failed to close stream response: {close_exc}",
                )

    def _count_tokens(self, response: ChatCompletion | ChatCompletionChunk) -> tuple[int, int]:
        usage = response.usage
        if usage is None:
            return 0, 0
        return int(usage.prompt_tokens), int(usage.completion_tokens)

    def _merge_call_kwargs(self, request: IRRequest) -> dict[str, Any]:
        call_kwargs: dict[str, Any] = dict(self._api_params)
        call_kwargs.update(_request_to_create_kwargs(request, stream=False, stream_options=None))
        return call_kwargs

    @override
    async def chat(
        self,
        request: IRRequest,
        *,
        trace_id: Optional[str] = None,
        timeout: Optional[int] = 30,
        cancellation: CancellationToken | None = None,
    ) -> Completion:
        _ = trace_id or get_current_trace_id()
        if cancellation is not None and cancellation.cancelled:
            raise asyncio.CancelledError
        key = self.key_pool.get_least_loaded_key()
        client = await self._get_or_create_client(key, cancellation)

        attempt = 0
        task_counted = False
        while attempt < self.max_retries:
            try:
                token_acquired = await await_with_cancellation(
                    lambda: self.token_bucket.acquire(
                        tokens_needed=1,
                        timeout=30.0,
                    ),
                    cancellation,
                )
                if not token_acquired:
                    push_warning(
                        f"{self.model_name} token bucket acquire timed out; skipping request",
                    )
                    raise Exception("Rate limit: token bucket acquire timed out")

                self.key_pool.increment_task_count(key)
                task_counted = True
                push_debug(
                    f"OpenAICompatible::chat: model={self.model_name} "
                    f"message_count={len(request.messages)}",
                )

                call_kwargs = self._merge_call_kwargs(request)
                response = await await_with_cancellation(
                    lambda: client.chat.completions.create(
                        model=self.model_name,
                        stream=False,
                        timeout=timeout,
                        **call_kwargs,
                    ),
                    cancellation,
                )

                if not (
                    response.choices
                    and response.choices[0].message
                    and response.choices[0].message.tool_calls
                ):
                    prompt_tokens, completion_tokens = self._count_tokens(response)
                    input_tokens = get_current_context_attribute("input_tokens") or 0
                    output_tokens = get_current_context_attribute("output_tokens") or 0
                    set_current_context_attribute(
                        "input_tokens", input_tokens + prompt_tokens
                    )
                    set_current_context_attribute(
                        "output_tokens", output_tokens + completion_tokens
                    )

                self.key_pool.decrement_task_count(key)
                task_counted = False
                return _completion_from_chat_completion(response)

            except BaseException as e:
                if task_counted:
                    self.key_pool.decrement_task_count(key)
                    task_counted = False
                if not isinstance(e, Exception):
                    raise
                attempt += 1
                push_warning(
                    f"{self.model_name} Interface attempt {attempt} failed for "
                    f"message_count={len(request.messages)}: {e}",
                )

                key = self.key_pool.get_least_loaded_key()
                client = await self._get_or_create_client(key, cancellation)

                if attempt >= self.max_retries:
                    push_error(
                        f"Max retries reached for {self.model_name} "
                        f"message_count={len(request.messages)}",
                    )
                    raise
                await await_with_cancellation(
                    lambda: asyncio.sleep(self.retry_delay),
                    cancellation,
                )

        return Completion(id="", created=0, model="", choices=[])

    @override
    async def chat_stream(
        self,
        request: IRRequest,
        *,
        trace_id: Optional[str] = None,
        timeout: Optional[int] = 30,
        cancellation: CancellationToken | None = None,
    ) -> AsyncGenerator[Chunk, None]:
        _ = trace_id or get_current_trace_id()
        if cancellation is not None and cancellation.cancelled:
            raise asyncio.CancelledError
        key = self.key_pool.get_least_loaded_key()
        client = await self._get_or_create_client(key, cancellation)

        attempt = 0
        task_counted = False
        yielded_output = False
        while attempt < self.max_retries:
            yielded_output = False
            try:
                token_acquired = await await_with_cancellation(
                    lambda: self.token_bucket.acquire(
                        tokens_needed=1,
                        timeout=30.0,
                    ),
                    cancellation,
                )
                if not token_acquired:
                    push_warning(
                        f"{self.model_name} stream token bucket acquire timed out; skipping request",
                    )
                    raise Exception("Rate limit: token bucket acquire timed out")

                self.key_pool.increment_task_count(key)
                task_counted = True
                push_debug(
                    f"OpenAICompatible::chat_stream: model={self.model_name} "
                    f"message_count={len(request.messages)}",
                )

                base_kwargs: dict[str, Any] = dict(self._api_params)
                base_kwargs.update(
                    _request_to_create_kwargs(
                        request,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                )

                auto_stream_options_added = "stream_options" not in base_kwargs
                if auto_stream_options_added:
                    base_kwargs["stream_options"] = {"include_usage": True}

                create_stream = cast(
                    Callable[..., Awaitable[AsyncStream[ChatCompletionChunk]]],
                    client.chat.completions.create,
                )
                stream = await await_with_cancellation(
                    lambda: create_stream(
                        model=self.model_name,
                        stream=True,
                        timeout=timeout,
                        **base_kwargs,
                    ),
                    cancellation,
                )

                try:
                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    saw_finish_reason = False
                    saw_usage_chunk = False
                    finish_deadline: Optional[float] = None

                    loop = asyncio.get_running_loop()
                    response_iter = stream.__aiter__()

                    while True:
                        try:
                            if saw_finish_reason:
                                if finish_deadline is None:
                                    finish_deadline = (
                                        loop.time() + self._stream_finish_grace_timeout
                                    )
                                remaining = finish_deadline - loop.time()
                                if remaining <= 0:
                                    break
                                chunk = await await_with_cancellation(
                                    lambda: asyncio.wait_for(
                                        response_iter.__anext__(),
                                        timeout=remaining,
                                    ),
                                    cancellation,
                                )
                            else:
                                chunk = await await_with_cancellation(
                                    response_iter.__anext__,
                                    cancellation,
                                )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            break

                        ir_chunk = _chunk_from_chat_completion_chunk(chunk)
                        yielded_output = True
                        yield ir_chunk

                        first_choice = chunk.choices[0] if chunk.choices else None
                        usage = chunk.usage

                        if usage is not None:
                            saw_usage_chunk = True
                            prompt_tokens, completion_tokens = self._count_tokens(chunk)
                            total_prompt_tokens = prompt_tokens
                            total_completion_tokens = completion_tokens

                        if saw_finish_reason and saw_usage_chunk:
                            break

                        if first_choice is not None and first_choice.finish_reason is not None:
                            if usage is not None:
                                break
                            saw_finish_reason = True

                    input_tokens = get_current_context_attribute("input_tokens") or 0
                    output_tokens = get_current_context_attribute("output_tokens") or 0
                    set_current_context_attribute(
                        "input_tokens", input_tokens + total_prompt_tokens
                    )
                    set_current_context_attribute(
                        "output_tokens", output_tokens + total_completion_tokens
                    )
                finally:
                    await self._close_stream_response(stream)

                self.key_pool.decrement_task_count(key)
                task_counted = False
                break
            except BaseException as e:
                if task_counted:
                    self.key_pool.decrement_task_count(key)
                    task_counted = False
                if not isinstance(e, Exception):
                    raise
                if yielded_output:
                    raise
                attempt += 1
                push_warning(
                    f"{self.model_name} Interface attempt {attempt} failed for "
                    f"message_count={len(request.messages)}: {e}",
                )

                key = self.key_pool.get_least_loaded_key()
                client = await self._get_or_create_client(key, cancellation)

                if attempt >= self.max_retries:
                    push_error(
                        f"Max retries reached for {self.model_name} "
                        f"message_count={len(request.messages)}",
                    )
                    raise
                await await_with_cancellation(
                    lambda: asyncio.sleep(self.retry_delay),
                    cancellation,
                )

        if False:
            yield Chunk(id="", created=0, model="", choices=[])
