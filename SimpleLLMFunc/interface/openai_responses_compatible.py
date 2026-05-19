from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, Iterable, Literal, Optional, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import Response, ResponseStreamEvent
from typing_extensions import override

from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.interface.llm_interface import DEFAULT_CONTEXT_WINDOW, LLM_Interface
from SimpleLLMFunc.interface.token_bucket import rate_limit_manager
from SimpleLLMFunc.logger import (
    get_current_trace_id,
    get_location,
    push_debug,
    push_error,
    push_warning,
)
from SimpleLLMFunc.logger.logger import (
    get_current_context_attribute,
    push_critical,
    set_current_context_attribute,
)


_RESPONSES_TOOL_HIDDEN_PROPERTIES = {"event_emitter"}
_RESPONSES_UNSUPPORTED_REQUEST_KWARGS = {"temperature", "top_p", "n"}


def _extract_response_output_text(response: Response) -> str:
    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", None)
                if isinstance(text, str) and text:
                    fragments.append(text)
    return "".join(fragments)


def _extract_response_reasoning_details(response: Response) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "reasoning":
            continue

        content_parts = getattr(item, "content", None) or []
        for index, part in enumerate(content_parts):
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                details.append(
                    {
                        "id": getattr(item, "id", "") or "",
                        "format": "text",
                        "index": index,
                        "type": "reasoning.text",
                        "data": text,
                    }
                )

        summary_parts = getattr(item, "summary", None) or []
        for index, part in enumerate(summary_parts, start=len(details)):
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                details.append(
                    {
                        "id": getattr(item, "id", "") or "",
                        "format": "text",
                        "index": index,
                        "type": "reasoning.summary_text",
                        "data": text,
                    }
                )
    return details


def _extract_response_tool_calls(
    response: Response,
) -> list[ChatCompletionMessageFunctionToolCall]:
    tool_calls: list[ChatCompletionMessageFunctionToolCall] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        tool_calls.append(
            ChatCompletionMessageFunctionToolCall(
                id=getattr(item, "call_id", None) or getattr(item, "id", "") or "",
                type="function",
                function=Function(
                    name=getattr(item, "name", "") or "",
                    arguments=getattr(item, "arguments", "") or "{}",
                ),
            )
        )
    return tool_calls


def _extract_usage(response: Response) -> Optional[CompletionUsage]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        or (prompt_tokens + completion_tokens)
    )
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _response_to_chat_completion(response: Response) -> ChatCompletion:
    tool_calls = _extract_response_tool_calls(response)
    content = None if tool_calls else _extract_response_output_text(response)
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )

    reasoning_details = _extract_response_reasoning_details(response)
    if reasoning_details:
        message.reasoning_details = reasoning_details

    finish_reason = "tool_calls" if tool_calls else "stop"
    choice = Choice(
        finish_reason=finish_reason,
        index=0,
        message=message,
    )
    completion = ChatCompletion(
        id=getattr(response, "id", "") or "",
        choices=[choice],
        created=int(getattr(response, "created_at", 0) or 0),
        model=getattr(response, "model", "") or "",
        object="chat.completion",
        usage=_extract_usage(response),
    )
    completion.response_id = getattr(response, "id", None)
    return completion


def _messages_to_response_input(
    messages: Iterable[Dict[str, Any]],
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            continue

        if role == "user":
            text = "" if content is None else str(content)
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": text}],
                }
            )
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function_payload = tool_call.get("function")
                    if not isinstance(function_payload, dict):
                        continue
                    call_id = tool_call.get("id") or ""
                    name = function_payload.get("name") or ""
                    arguments = function_payload.get("arguments") or "{}"
                    if not call_id or not name:
                        continue
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": str(call_id),
                            "name": str(name),
                            "arguments": str(arguments),
                        }
                    )

            text = "" if content is None else str(content)
            if text:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id") or message.get("id") or ""
            output = "" if content is None else str(content)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": output,
                }
            )
    return input_items


def _extract_response_instructions(
    messages: Iterable[Dict[str, Any]],
) -> Optional[str]:
    instructions: list[str] = []
    for message in messages:
        if message.get("role") != "system":
            continue

        content = message.get("content")
        if content is None:
            continue

        text = str(content)
        if text:
            instructions.append(text)

    if not instructions:
        return None

    return "\n\n".join(instructions)


def _tools_to_response_tools(
    tools: Optional[Iterable[Dict[str, Any]]],
) -> Optional[list[dict[str, Any]]]:
    if not tools:
        return None

    response_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        function_spec = tool.get("function", {})
        if not isinstance(function_spec, dict):
            continue
        response_tools.append(
            {
                "type": "function",
                "name": function_spec.get("name", ""),
                "description": function_spec.get("description"),
                "parameters": _normalize_tool_parameters_for_responses(
                    function_spec.get("parameters")
                ),
                "strict": True,
            }
        )

    return response_tools or None


def _normalize_schema_for_responses(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_normalize_schema_for_responses(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    normalized = {
        key: _normalize_schema_for_responses(value) for key, value in schema.items()
    }

    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            key: _normalize_schema_for_responses(value)
            for key, value in normalized["properties"].items()
        }

    schema_type = normalized.get("type")
    is_object_schema = (
        schema_type == "object"
        or (isinstance(schema_type, list) and "object" in schema_type)
        or "properties" in normalized
    )

    if is_object_schema and "additionalProperties" not in normalized:
        normalized["additionalProperties"] = False

    if "additionalProperties" in normalized and isinstance(
        normalized["additionalProperties"], dict
    ):
        normalized["additionalProperties"] = _normalize_schema_for_responses(
            normalized["additionalProperties"]
        )

    return normalized


def _convert_nullable_type_list_to_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    schema_type = schema.get("type")
    if not isinstance(schema_type, list):
        return schema

    non_null_types = [item for item in schema_type if item != "null"]
    has_null = len(non_null_types) != len(schema_type)
    if not has_null or not non_null_types:
        return schema

    converted = dict(schema)
    converted.pop("type", None)
    converted["anyOf"] = [{"type": value} for value in non_null_types] + [
        {"type": "null"}
    ]
    return converted


def _normalize_tool_parameters_for_responses(parameters: Any) -> Any:
    normalized = _normalize_schema_for_responses(parameters)
    if not isinstance(normalized, dict):
        return normalized

    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        return normalized

    filtered_properties: dict[str, Any] = {}
    for key, value in properties.items():
        if key in _RESPONSES_TOOL_HIDDEN_PROPERTIES:
            continue
        if isinstance(value, dict):
            filtered_properties[key] = _convert_nullable_type_list_to_anyof(value)
        else:
            filtered_properties[key] = value

    normalized["properties"] = filtered_properties
    normalized["required"] = list(filtered_properties.keys())
    normalized["additionalProperties"] = False
    return normalized


def _translate_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function_spec = tool_choice.get("function", {})
        if isinstance(function_spec, dict) and function_spec.get("name"):
            return {"type": "function", "name": function_spec["name"]}
    return tool_choice


def _make_text_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    delta_text: str,
) -> ChatCompletionChunk:
    delta = ChoiceDelta(content=delta_text, role="assistant")
    choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
    return ChatCompletionChunk(
        id=response_id,
        choices=[choice],
        created=created,
        model=model,
        object="chat.completion.chunk",
    )


def _make_reasoning_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    delta_text: str,
) -> ChatCompletionChunk:
    delta = ChoiceDelta(content=None, role="assistant")
    delta.reasoning_details = [
        {
            "id": "",
            "format": "text",
            "index": 0,
            "type": "reasoning.text",
            "data": delta_text,
        }
    ]
    choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
    return ChatCompletionChunk(
        id=response_id,
        choices=[choice],
        created=created,
        model=model,
        object="chat.completion.chunk",
    )


def _make_tool_call_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    tool_call_index: int,
    tool_call_id: str,
    name: Optional[str],
    arguments_delta: str,
) -> ChatCompletionChunk:
    function = SimpleNamespace(name=name, arguments=arguments_delta)
    tool_call = SimpleNamespace(
        index=tool_call_index,
        id=tool_call_id,
        type="function",
        function=function,
    )
    delta = ChoiceDelta(content=None, role="assistant")
    delta.tool_calls = [tool_call]
    choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
    return ChatCompletionChunk(
        id=response_id,
        choices=[choice],
        created=created,
        model=model,
        object="chat.completion.chunk",
    )


def _make_finish_chunk(
    *,
    response: Response,
) -> ChatCompletionChunk:
    tool_calls = _extract_response_tool_calls(response)
    finish_reason = "tool_calls" if tool_calls else "stop"
    delta = ChoiceDelta(content=None, role="assistant")
    choice = ChunkChoice(delta=delta, finish_reason=finish_reason, index=0)
    return ChatCompletionChunk(
        id=getattr(response, "id", "") or "",
        choices=[choice],
        created=int(getattr(response, "created_at", 0) or 0),
        model=getattr(response, "model", "") or "",
        object="chat.completion.chunk",
        usage=_extract_usage(response),
    )


class OpenAIResponsesCompatible(LLM_Interface):
    def __repr__(self) -> str:
        return f"OpenAIResponsesCompatible(model_name={self.model_name}, base_url={self.base_url})"

    @classmethod
    def load_from_json_file(
        cls, json_path: str
    ) -> Dict[str, Dict[str, "OpenAIResponsesCompatible"]]:
        if not os.path.exists(json_path):
            push_critical(
                f"JSON file {json_path} does not exist. Please check your configuration.",
                location=get_location(),
            )
            raise FileNotFoundError(
                f"JSON file {json_path} does not exist."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        all_providers_dict: Dict[str, Dict[str, OpenAIResponsesCompatible]] = {}
        for provider_id, models in payload.items():
            all_providers_dict[provider_id] = {}
            if not isinstance(models, list):
                raise TypeError(
                    f"Invalid model format under provider {provider_id}. Expected a list."
                )
            for model_info in models:
                model_name = model_info["model_name"]
                key_pool = APIKeyPool(
                    model_info["api_keys"], f"{provider_id}-{model_name}"
                )
                instance = cls(
                    api_key_pool=key_pool,
                    model_name=model_name,
                    base_url=model_info["base_url"],
                    context_window=model_info.get(
                        "context_window", DEFAULT_CONTEXT_WINDOW
                    ),
                    max_retries=model_info.get("max_retries", 5),
                    retry_delay=model_info.get("retry_delay", 1.0),
                    rate_limit_capacity=model_info.get("rate_limit_capacity", 10),
                    rate_limit_refill_rate=model_info.get(
                        "rate_limit_refill_rate", 1.0
                    ),
                )
                all_providers_dict[provider_id][model_name] = instance
        return all_providers_dict

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
    ):
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
        bucket_id = f"responses_{base_url}_{model_name}"
        self.token_bucket = rate_limit_manager.get_or_create_bucket(
            bucket_id=bucket_id,
            capacity=rate_limit_capacity,
            refill_rate=rate_limit_refill_rate,
        )
        self.client = AsyncOpenAI(
            api_key=api_key_pool.get_least_loaded_key(),
            base_url=self.base_url,
        )

    async def _get_or_create_client(self, key: str) -> AsyncOpenAI:
        if (
            not hasattr(self, "_current_key")
            or self._current_key != key  # type: ignore[attr-defined]
            or not hasattr(self, "client")
            or self.client is None
        ):
            if hasattr(self, "client") and self.client is not None:
                try:
                    await self.client.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            self.client = AsyncOpenAI(api_key=key, base_url=self.base_url)
            self._current_key = key
        return self.client

    async def _close_stream_response(self, response: Any) -> None:
        close_method = getattr(response, "close", None)
        if callable(close_method):
            result = close_method()
            if hasattr(result, "__await__"):
                await result
            return

        aclose_method = getattr(response, "aclose", None)
        if callable(aclose_method):
            result = aclose_method()
            if hasattr(result, "__await__"):
                await result

    def _count_tokens(self, response: Response) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        return (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )

    def _build_request_kwargs(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_kwargs = dict(kwargs)
        for unsupported_key in _RESPONSES_UNSUPPORTED_REQUEST_KWARGS:
            request_kwargs.pop(unsupported_key, None)
        request_kwargs["input"] = _messages_to_response_input(messages)
        instructions = _extract_response_instructions(messages)
        if instructions is not None:
            request_kwargs["instructions"] = instructions
        tools = _tools_to_response_tools(
            cast(Optional[Iterable[Dict[str, Any]]], request_kwargs.pop("tools", None))
        )
        if tools:
            request_kwargs["tools"] = tools

        tool_choice = _translate_tool_choice(request_kwargs.get("tool_choice"))
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice

        return request_kwargs

    def _apply_usage_to_context(self, response: Response) -> None:
        prompt_tokens, completion_tokens = self._count_tokens(response)
        input_tokens = int(get_current_context_attribute("input_tokens") or 0)
        output_tokens = int(get_current_context_attribute("output_tokens") or 0)
        set_current_context_attribute("input_tokens", input_tokens + prompt_tokens)
        set_current_context_attribute(
            "output_tokens", output_tokens + completion_tokens
        )

    @override
    async def chat(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[False] = False,
        messages: Iterable[Dict[str, str]] = [{"role": "user", "content": ""}],
        timeout: Optional[int] = 30,
        *args: Any,
        **kwargs: Any,
    ) -> ChatCompletion:
        _ = trace_id, args
        key = self.key_pool.get_least_loaded_key()
        client = await self._get_or_create_client(key)

        attempt = 0
        while attempt < self.max_retries:
            try:
                token_acquired = await self.token_bucket.acquire(
                    tokens_needed=1, timeout=30.0
                )
                if not token_acquired:
                    raise Exception("Rate limit: token bucket acquire timed out")

                self.key_pool.increment_task_count(key)
                data = json.dumps(list(messages), ensure_ascii=False, indent=4)
                push_debug(
                    f"OpenAIResponsesCompatible::chat: {self.model_name} request with API key: {key}, and message: {data}",
                    location=get_location(),
                )
                request_kwargs = self._build_request_kwargs(
                    messages=messages, kwargs=kwargs
                )
                response = await client.responses.create(
                    model=self.model_name,
                    stream=stream,
                    timeout=timeout,
                    *args,
                    **request_kwargs,
                )
                response = cast(Response, response)
                self._apply_usage_to_context(response)
                self.key_pool.decrement_task_count(key)
                return _response_to_chat_completion(response)
            except Exception as exc:
                self.key_pool.decrement_task_count(key)
                attempt += 1
                data = json.dumps(list(messages), ensure_ascii=False, indent=4)
                push_warning(
                    f"{self.model_name} Responses interface attempt {attempt} failed: With message : {data} send, \n but exception : {str(exc)} was caught",
                    location=get_location(),
                )
                key = self.key_pool.get_least_loaded_key()
                client = await self._get_or_create_client(key)
                if attempt >= self.max_retries:
                    push_error(
                        f"Max retries reached. {self.model_name} Failed to get a response for {data}",
                        location=get_location(),
                    )
                    raise
                await asyncio.sleep(self.retry_delay)

        return ChatCompletion(
            id="",
            choices=[],
            created=0,
            model="",
            object="chat.completion",
            usage=None,
        )

    @override
    async def chat_stream(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[True] = True,
        messages: Iterable[Dict[str, str]] = [{"role": "user", "content": ""}],
        timeout: Optional[int] = 30,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        _ = trace_id, args
        key = self.key_pool.get_least_loaded_key()
        client = await self._get_or_create_client(key)

        attempt = 0
        while attempt < self.max_retries:
            try:
                token_acquired = await self.token_bucket.acquire(
                    tokens_needed=1, timeout=30.0
                )
                if not token_acquired:
                    raise Exception("Rate limit: 令牌桶获取令牌超时")

                self.key_pool.increment_task_count(key)
                data = json.dumps(list(messages), ensure_ascii=False, indent=4)
                push_debug(
                    f"OpenAIResponsesCompatible::chat_stream: {self.model_name} request with API key: {key}, and message: {data}",
                    location=get_location(),
                )
                request_kwargs = self._build_request_kwargs(
                    messages=messages, kwargs=kwargs
                )
                response_stream = await client.responses.create(
                    model=self.model_name,
                    stream=stream,
                    timeout=timeout,
                    *args,
                    **request_kwargs,
                )

                response_stream = cast(
                    AsyncGenerator[ResponseStreamEvent, None], response_stream
                )

                current_response_id = ""
                current_created = 0
                current_model = self.model_name
                tool_call_contexts_by_item_id: dict[str, dict[str, Any]] = {}
                tool_call_contexts_by_output_index: dict[int, dict[str, Any]] = {}
                completed_response: Optional[Response] = None

                try:
                    async for event in response_stream:
                        event_type = getattr(event, "type", "")
                        if event_type == "response.created":
                            response_obj = getattr(event, "response", None)
                            if response_obj is not None:
                                current_response_id = (
                                    getattr(response_obj, "id", "")
                                    or current_response_id
                                )
                                current_created = int(
                                    getattr(response_obj, "created_at", 0)
                                    or current_created
                                )
                                current_model = (
                                    getattr(response_obj, "model", current_model)
                                    or current_model
                                )
                            continue

                        if event_type == "response.output_text.delta":
                            yield _make_text_chunk(
                                response_id=current_response_id,
                                created=current_created,
                                model=current_model,
                                delta_text=getattr(event, "delta", "") or "",
                            )
                            continue

                        if event_type in {
                            "response.reasoning_text.delta",
                            "response.reasoning_summary_text.delta",
                        }:
                            yield _make_reasoning_chunk(
                                response_id=current_response_id,
                                created=current_created,
                                model=current_model,
                                delta_text=getattr(event, "delta", "") or "",
                            )
                            continue

                        if event_type == "response.output_item.added":
                            item = getattr(event, "item", None)
                            if getattr(item, "type", None) == "function_call":
                                item_id = getattr(item, "id", "") or ""
                                output_index_raw = getattr(event, "output_index", None)
                                output_index = (
                                    int(output_index_raw)
                                    if isinstance(output_index_raw, int)
                                    else len(tool_call_contexts_by_output_index)
                                )
                                context = {
                                    "index": output_index,
                                    "tool_call_id": (
                                        getattr(item, "call_id", "")
                                        or getattr(item, "id", "")
                                        or ""
                                    ),
                                    "name": getattr(item, "name", None),
                                }
                                if item_id:
                                    tool_call_contexts_by_item_id[item_id] = context
                                tool_call_contexts_by_output_index[output_index] = (
                                    context
                                )
                            continue

                        if event_type == "response.function_call_arguments.delta":
                            event_item_id = getattr(event, "item_id", "") or ""
                            event_output_index = getattr(event, "output_index", None)
                            tool_call_context = None
                            if event_item_id:
                                tool_call_context = tool_call_contexts_by_item_id.get(
                                    event_item_id
                                )
                            if tool_call_context is None and isinstance(
                                event_output_index, int
                            ):
                                tool_call_context = (
                                    tool_call_contexts_by_output_index.get(
                                        event_output_index
                                    )
                                )
                            if tool_call_context is None:
                                fallback_index = (
                                    event_output_index
                                    if isinstance(event_output_index, int)
                                    else 0
                                )
                                tool_call_context = {
                                    "index": fallback_index,
                                    "tool_call_id": event_item_id,
                                    "name": None,
                                }
                            yield _make_tool_call_chunk(
                                response_id=current_response_id,
                                created=current_created,
                                model=current_model,
                                tool_call_index=int(tool_call_context["index"]),
                                tool_call_id=str(
                                    tool_call_context.get("tool_call_id", "") or ""
                                ),
                                name=cast(
                                    Optional[str], tool_call_context.get("name", None)
                                ),
                                arguments_delta=getattr(event, "delta", "") or "",
                            )
                            continue

                        if event_type == "response.completed":
                            completed_response = cast(
                                Optional[Response], getattr(event, "response", None)
                            )
                            if completed_response is not None:
                                self._apply_usage_to_context(completed_response)
                                yield _make_finish_chunk(response=completed_response)
                            break
                finally:
                    await self._close_stream_response(response_stream)

                self.key_pool.decrement_task_count(key)
                break
            except Exception as exc:
                self.key_pool.decrement_task_count(key)
                attempt += 1
                data = json.dumps(list(messages), ensure_ascii=False, indent=4)
                push_warning(
                    f"{self.model_name} Responses interface attempt {attempt} failed: With message : {data} send, \n but exception : {str(exc)} was caught",
                    location=get_location(),
                )
                key = self.key_pool.get_least_loaded_key()
                client = await self._get_or_create_client(key)
                if attempt >= self.max_retries:
                    push_error(
                        f"Max retries reached. {self.model_name} Failed to get a response for {data}",
                        location=get_location(),
                    )
                    raise
                await asyncio.sleep(self.retry_delay)

        if False:
            yield ChatCompletionChunk(
                id="",
                choices=[],
                created=0,
                model="",
                object="chat.completion.chunk",
            )
