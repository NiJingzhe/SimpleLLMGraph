from __future__ import annotations

import asyncio
import json
import os
from inspect import isawaitable
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Iterable,
    Optional,
    cast,
    override,
)

from openai import AsyncOpenAI
from openai.types.responses import Response, ResponseOutputItem, ResponseStreamEvent

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
    Delta,
    FinishReason,
    Request as IRRequest,
    Role,
    ToolCall,
    ToolCallDelta,
    ToolCallDeltaFunction,
    ToolCallFunction,
    Usage,
)
from SimpleLLMFunc.context.ir._enums import ToolCallType
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.interface.llm_interface import DEFAULT_CONTEXT_WINDOW, LLM_Interface
from SimpleLLMFunc.interface.token_bucket import rate_limit_manager
from SimpleLLMFunc.logger import (
    get_current_context_attribute,
    get_current_trace_id,
    push_critical,
    push_debug,
    push_error,
    push_warning,
    set_current_context_attribute,
)


_RESPONSES_TOOL_HIDDEN_PROPERTIES = {"event_emitter"}
_RESPONSES_REASONING_SIGNATURE_PREFIX = "openai-responses:"


# --------------------------------------------------------------------------- #
# IR -> Responses wire format translation
# --------------------------------------------------------------------------- #


def _ir_user_content_to_response_content(content: object) -> list[dict[str, object]]:
    """Translate IR user content (str or list of content parts) into the
    Responses API ``input_text`` / ``input_image`` content list.
    """
    if isinstance(content, list):
        parts = cast(list[object], content)
        converted: list[dict[str, object]] = []
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("unsupported user content part for Responses API")
            pd = cast(dict[str, object], part)
            part_type = pd.get("type")
            if part_type in ("input_text", "text"):
                converted.append(
                    {"type": "input_text", "text": str(pd.get("text", ""))}
                )
                continue
            if part_type in ("input_image", "image_url"):
                image_url = pd.get("image_url")
                if isinstance(image_url, str) and image_url:
                    converted.append(
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "auto",
                        }
                    )
                    continue
                if isinstance(image_url, dict):
                    iu = cast(dict[str, object], image_url)
                    url = iu.get("url")
                    if isinstance(url, str) and url:
                        image_part: dict[str, object] = {
                            "type": "input_image",
                            "image_url": url,
                            "detail": "auto",
                        }
                        detail = iu.get("detail")
                        if detail in ("low", "high", "auto"):
                            image_part["detail"] = detail
                        converted.append(image_part)
                        continue
            raise ValueError(
                f"Responses API does not support user content part {part_type!r}"
            )
        return converted or [{"type": "input_text", "text": ""}]
    text = "" if content is None else str(content)
    return [{"type": "input_text", "text": text}]


def _assistant_text_item(text: str) -> dict[str, object]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "input_text", "text": text}],
    }


def _reasoning_text(payload: dict[str, object]) -> str:
    fragments: list[str] = []
    for key in ("content", "summary"):
        parts = payload.get(key)
        if not isinstance(parts, list):
            continue
        for part in cast(list[object], parts):
            if isinstance(part, dict):
                text = cast(dict[str, object], part).get("text")
                if isinstance(text, str) and text:
                    fragments.append(text)
    return "\n".join(fragments)


def _decode_responses_reasoning_part(payload: dict[str, object]) -> dict[str, object]:
    signature = payload.get("signature")
    if not isinstance(signature, str) or not signature.startswith(
        _RESPONSES_REASONING_SIGNATURE_PREFIX
    ):
        raise ValueError(
            "Responses API cannot losslessly replay this reasoning signature"
        )
    try:
        item = json.loads(signature.removeprefix(_RESPONSES_REASONING_SIGNATURE_PREFIX))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid Responses reasoning signature") from exc
    if not isinstance(item, dict):
        raise ValueError("invalid Responses reasoning signature")
    reasoning_item = cast(dict[str, object], item)
    if (
        reasoning_item.get("type") != "reasoning"
        or not isinstance(reasoning_item.get("id"), str)
        or not isinstance(reasoning_item.get("summary"), list)
    ):
        raise ValueError("invalid Responses reasoning signature")
    visible_reasoning = payload.get("reasoning")
    if visible_reasoning != _reasoning_text(reasoning_item):
        raise ValueError("Responses reasoning text does not match its signature")
    return reasoning_item


def _ir_assistant_content_to_response_items(
    content: object,
) -> list[dict[str, object]]:
    """Translate assistant content into valid, ordered Responses input items."""

    if content is None:
        return []
    if isinstance(content, str):
        return [_assistant_text_item(content)] if content else []
    if not isinstance(content, list):
        raise ValueError("unsupported assistant content for Responses API")

    converted: list[dict[str, object]] = []
    text_parts: list[dict[str, object]] = []

    def flush_text() -> None:
        if not text_parts:
            return
        converted.append(
            {
                "type": "message",
                "role": "assistant",
                "content": list(text_parts),
            }
        )
        text_parts.clear()

    for part in cast(list[object], content):
        if not isinstance(part, dict):
            raise ValueError("unsupported assistant content part for Responses API")
        payload = cast(dict[str, object], part)
        if payload.get("type") == "output_text":
            text = str(payload.get("text", ""))
            if text:
                text_parts.append({"type": "input_text", "text": text})
            continue
        if payload.get("type") == "reasoning":
            flush_text()
            converted.append(_decode_responses_reasoning_part(payload))
            continue
        raise ValueError(
            "Responses API cannot losslessly replay this assistant content part"
        )
    flush_text()
    return converted


def _ir_messages_to_response_input(
    messages: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Translate IR messages (already JSON-dumped) to Responses API ``input``."""
    input_items: list[dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            continue
        if role in ("user", "developer"):
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": _ir_user_content_to_response_content(content),
                }
            )
            continue
        if role == "assistant":
            content_items = _ir_assistant_content_to_response_items(content)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                if any(item.get("type") == "message" for item in content_items):
                    raise ValueError(
                        "Responses API cannot preserve mixed assistant text and tool calls"
                    )
                input_items.extend(content_items)
                tc_list = cast(list[object], tool_calls)
                for tool_call in tc_list:
                    if not isinstance(tool_call, dict):
                        continue
                    tc = cast(dict[str, object], tool_call)
                    function_payload = tc.get("function")
                    if not isinstance(function_payload, dict):
                        continue
                    fn = cast(dict[str, object], function_payload)
                    call_id = tc.get("id") or ""
                    name = fn.get("name") or ""
                    arguments = fn.get("arguments") or "{}"
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
            else:
                input_items.extend(content_items)
            refusal = message.get("refusal")
            if isinstance(refusal, str) and refusal:
                raise ValueError(
                    "Responses API cannot losslessly replay an assistant refusal"
                )
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id") or message.get("id") or ""
            output = "" if content is None else str(content)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(tool_call_id),
                    "output": output,
                }
            )
    return input_items


def _ir_messages_to_instructions(
    messages: Iterable[dict[str, object]],
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
    return "\n\n".join(instructions) if instructions else None


# --- JSON Schema normalization for Responses tools --- #


def _normalize_schema_for_responses(schema: object) -> object:
    """Recursively normalize a JSON Schema for the Responses API:
    add ``additionalProperties: false`` to object schemas.
    """
    if isinstance(schema, list):
        items = cast(list[object], schema)
        return [_normalize_schema_for_responses(item) for item in items]
    if not isinstance(schema, dict):
        return schema
    src = cast(dict[str, object], schema)
    normalized: dict[str, object] = {
        key: _normalize_schema_for_responses(value) for key, value in src.items()
    }
    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            key: _normalize_schema_for_responses(value)
            for key, value in cast(dict[str, object], normalized["properties"]).items()
        }
    schema_type = normalized.get("type")
    is_object_schema = (
        schema_type == "object"
        or (isinstance(schema_type, list) and "object" in cast(list[object], schema_type))
        or "properties" in normalized
    )
    if is_object_schema and "additionalProperties" not in normalized:
        normalized["additionalProperties"] = False
    ap = normalized.get("additionalProperties")
    if isinstance(ap, dict):
        normalized["additionalProperties"] = _normalize_schema_for_responses(
            cast(dict[str, object], ap)
        )
    return normalized


def _convert_nullable_type_list_to_anyof(schema: dict[str, object]) -> dict[str, object]:
    schema_type = schema.get("type")
    if not isinstance(schema_type, list):
        return schema
    type_list = cast(list[object], schema_type)
    non_null_types = [item for item in type_list if item != "null"]
    has_null = len(non_null_types) != len(type_list)
    if not has_null or not non_null_types:
        return schema
    converted = dict(schema)
    converted.pop("type", None)
    converted["anyOf"] = [{"type": value} for value in non_null_types] + [
        {"type": "null"}
    ]
    return converted


def _normalize_tool_parameters_for_responses(parameters: object) -> object:
    normalized = _normalize_schema_for_responses(parameters)
    if not isinstance(normalized, dict):
        return normalized
    result = cast(dict[str, object], normalized)
    properties = result.get("properties")
    if not isinstance(properties, dict):
        return result
    props = cast(dict[str, object], properties)
    filtered_properties: dict[str, object] = {}
    for key, value in props.items():
        if key in _RESPONSES_TOOL_HIDDEN_PROPERTIES:
            continue
        if isinstance(value, dict):
            filtered_properties[key] = _convert_nullable_type_list_to_anyof(
                cast(dict[str, object], value)
            )
        else:
            filtered_properties[key] = value
    result["properties"] = filtered_properties
    result["required"] = list(filtered_properties.keys())
    result["additionalProperties"] = False
    return result


def _ir_tools_to_response_tools(
    tools: Optional[list[dict[str, object]]],
) -> Optional[list[dict[str, object]]]:
    if not tools:
        return None
    response_tools: list[dict[str, object]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function_spec = tool.get("function")
        if not isinstance(function_spec, dict):
            continue
        fn = cast(dict[str, object], function_spec)
        response_tools.append(
            {
                "type": "function",
                "name": str(fn.get("name", "")),
                "description": fn.get("description"),
                "parameters": _normalize_tool_parameters_for_responses(
                    fn.get("parameters")
                ),
                "strict": True,
            }
        )
    return response_tools or None


def _request_to_responses_kwargs(request: IRRequest) -> dict[str, Any]:
    """Translate an IR :class:`Request` to ``client.responses.create`` kwargs
    (excluding ``model`` / ``stream`` / ``timeout``).
    """
    messages_dump: list[dict[str, object]] = [
        m.model_dump(mode="json") for m in request.messages
    ]
    kwargs: dict[str, Any] = {
        "input": _ir_messages_to_response_input(messages_dump),
    }
    instructions = _ir_messages_to_instructions(messages_dump)
    if instructions is not None:
        kwargs["instructions"] = instructions
    tools = _ir_tools_to_response_tools(
        [t.model_dump(mode="json") for t in request.tools] if request.tools else None
    )
    if tools:
        kwargs["tools"] = tools
    if request.max_tokens is not None:
        kwargs["max_output_tokens"] = request.max_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": request.reasoning_effort.value}
    if request.extra:
        for k, v in request.extra.items():
            kwargs[k] = v
    return kwargs


# --------------------------------------------------------------------------- #
# Responses wire format -> IR translation
# --------------------------------------------------------------------------- #


def _extract_response_output_text(response: Response) -> str:
    fragments: list[str] = []
    output: list[ResponseOutputItem] = getattr(response, "output", []) or []
    unsupported = [
        getattr(item, "type", None)
        for item in output
        if getattr(item, "type", None)
        not in {"reasoning", "message", "function_call"}
    ]
    if unsupported:
        raise ValueError(f"unsupported Responses output item types: {unsupported}")
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        contents: list[Any] = getattr(item, "content", []) or []
        for content in contents:
            if getattr(content, "type", None) == "output_text":
                annotations = cast(
                    list[object], getattr(content, "annotations", None) or []
                )
                if annotations:
                    raise ValueError(
                        "Responses output annotations cannot be represented losslessly"
                    )
                text = getattr(content, "text", None)
                if isinstance(text, str) and text:
                    fragments.append(text)
    return "".join(fragments)


def _extract_response_refusal(response: Response) -> str | None:
    fragments: list[str] = []
    output: list[ResponseOutputItem] = getattr(response, "output", []) or []
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        contents: list[Any] = getattr(item, "content", []) or []
        for content in contents:
            if getattr(content, "type", None) == "refusal":
                refusal = getattr(content, "refusal", None)
                if isinstance(refusal, str) and refusal:
                    fragments.append(refusal)
    return "".join(fragments) or None


def _extract_response_reasoning_parts(response: Response) -> list[object]:
    from SimpleLLMFunc.context.ir.parts import ReasoningPart

    parts: list[object] = []
    output: list[ResponseOutputItem] = getattr(response, "output", []) or []
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        model_dump = getattr(item, "model_dump", None)
        if not callable(model_dump):
            raise ValueError("Responses reasoning item cannot be serialized")
        payload = cast(dict[str, object], model_dump(mode="json", exclude_none=True))
        signature = _RESPONSES_REASONING_SIGNATURE_PREFIX + json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        parts.append(
            ReasoningPart(reasoning=_reasoning_text(payload), signature=signature)
        )
    return parts


def _extract_response_tool_calls(response: Response) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    output: list[ResponseOutputItem] = getattr(response, "output", []) or []
    for item in output:
        if getattr(item, "type", None) != "function_call":
            continue
        tool_calls.append(
            ToolCall(
                id=getattr(item, "call_id", None) or getattr(item, "id", "") or "",
                type=ToolCallType.FUNCTION,
                function=ToolCallFunction(
                    name=getattr(item, "name", "") or "",
                    arguments=getattr(item, "arguments", "") or "{}",
                ),
            )
        )
    return tool_calls


def _usage_from_response(response: Response) -> Optional[Usage]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        or (prompt_tokens + completion_tokens)
    )
    reasoning_tokens: Optional[int] = None
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        rt = getattr(details, "reasoning_tokens", None)
        if rt is not None:
            reasoning_tokens = int(rt)
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


def _response_to_completion(response: Response) -> Completion:
    status = getattr(response, "status", None)
    incomplete_reason = None
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(details, "reason", None)
        if incomplete_reason not in ("max_output_tokens", "content_filter"):
            raise RuntimeError(
                f"Responses request ended incomplete for unsupported reason: "
                f"{incomplete_reason}"
            )
    elif status not in (None, "completed"):
        detail = getattr(response, "error", None) or getattr(
            response, "incomplete_details", None
        )
        raise RuntimeError(f"Responses request ended with status {status}: {detail}")
    output: list[ResponseOutputItem] = getattr(response, "output", []) or []
    seen_non_reasoning = False
    for item in output:
        if getattr(item, "type", None) == "reasoning":
            if seen_non_reasoning:
                raise ValueError("Responses reasoning item order cannot be represented")
        else:
            seen_non_reasoning = True
    if len([item for item in output if getattr(item, "type", None) == "message"]) > 1:
        raise ValueError("multiple Responses output messages cannot be represented")

    tool_calls = _extract_response_tool_calls(response)
    reasoning_parts = _extract_response_reasoning_parts(response)
    text = _extract_response_output_text(response)
    refusal = _extract_response_refusal(response)
    if tool_calls and (text or refusal):
        raise ValueError(
            "Responses output mixes text or refusal with function calls and cannot "
            "be represented losslessly"
        )
    content_parts: list[object] = []
    content_parts.extend(reasoning_parts)
    if text:
        content_parts.append({"type": "output_text", "text": text})
    content_value: object = None
    if content_parts:
        content_value = content_parts
    elif not tool_calls:
        content_value = text or None

    assistant = AssistantMessage(
        content=content_value,  # type: ignore[arg-type]
        tool_calls=tool_calls or None,
        refusal=refusal,
    )
    if tool_calls:
        finish_reason = FinishReason.TOOL_CALLS
    elif incomplete_reason == "max_output_tokens":
        finish_reason = FinishReason.LENGTH
    elif incomplete_reason == "content_filter":
        finish_reason = FinishReason.CONTENT_FILTER
    else:
        finish_reason = FinishReason.STOP
    choice = CompletionChoice(
        index=0,
        message=assistant,
        finish_reason=finish_reason,
        logprobs=None,
    )
    return Completion(
        id=getattr(response, "id", "") or "",
        created=int(getattr(response, "created_at", 0) or 0),
        model=getattr(response, "model", "") or "",
        choices=[choice],
        usage=_usage_from_response(response),
    )


# --- streaming chunk builders (IR) --- #


def _make_text_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    delta_text: str,
) -> Chunk:
    return Chunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            Choice(
                index=0,
                delta=Delta(role=Role.ASSISTANT, content=delta_text),
                finish_reason=None,
                logprobs=None,
            )
        ],
        usage=None,
    )


def _make_reasoning_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    delta_text: str,
    signature: str | None = None,
) -> Chunk:
    return Chunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            Choice(
                index=0,
                delta=Delta(
                    role=Role.ASSISTANT,
                    reasoning=delta_text,
                    reasoning_signature=signature,
                ),
                finish_reason=None,
                logprobs=None,
            )
        ],
        usage=None,
    )


def _make_refusal_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    delta_text: str,
) -> Chunk:
    return Chunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            Choice(
                index=0,
                delta=Delta(role=Role.ASSISTANT, refusal=delta_text),
                finish_reason=None,
                logprobs=None,
            )
        ],
        usage=None,
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
) -> Chunk:
    return Chunk(
        id=response_id,
        created=created,
        model=model,
        choices=[
            Choice(
                index=0,
                delta=Delta(
                    role=Role.ASSISTANT,
                    tool_calls=[
                        ToolCallDelta(
                            index=tool_call_index,
                            id=tool_call_id or None,
                            type=ToolCallType.FUNCTION if tool_call_id else None,
                            function=ToolCallDeltaFunction(
                                name=name,
                                arguments=arguments_delta or "",
                            ),
                        )
                    ],
                ),
                finish_reason=None,
                logprobs=None,
            )
        ],
        usage=None,
    )


def _make_finish_chunk(*, response: Response) -> Chunk:
    completion = _response_to_completion(response)
    finish_reason = completion.choices[0].finish_reason or FinishReason.STOP
    return Chunk(
        id=getattr(response, "id", "") or "",
        created=int(getattr(response, "created_at", 0) or 0),
        model=getattr(response, "model", "") or "",
        choices=[
            Choice(
                index=0,
                delta=Delta(role=Role.ASSISTANT),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ],
        usage=completion.usage,
    )


# --------------------------------------------------------------------------- #
# Provider config loader (typed)
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

    def __init__(self, raw: dict[str, Any]) -> None:
        self.model_name = str(raw["model_name"])
        self.api_keys = [str(k) for k in cast(list[Any], raw["api_keys"])]
        self.base_url = str(raw["base_url"])
        self.context_window = int(cast(int, raw.get("context_window", DEFAULT_CONTEXT_WINDOW)))
        self.max_retries = int(cast(int, raw.get("max_retries", 5)))
        self.retry_delay = float(cast(float, raw.get("retry_delay", 1.0)))
        self.rate_limit_capacity = int(cast(int, raw.get("rate_limit_capacity", 10)))
        self.rate_limit_refill_rate = float(cast(float, raw.get("rate_limit_refill_rate", 1.0)))


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


class OpenAIResponsesCompatible(LLM_Interface):
    """OpenAI Responses API adapter.

    Speaks the neutral context IR on its inner side, translating to/from the
    Responses API wire format inline.
    """

    def __repr__(self) -> str:
        return f"OpenAIResponsesCompatible(model_name={self.model_name}, base_url={self.base_url})"

    @classmethod
    def load_from_json_file(
        cls, json_path: str
    ) -> dict[str, dict[str, "OpenAIResponsesCompatible"]]:
        if not os.path.exists(json_path):
            push_critical(
                f"JSON file {json_path} does not exist. Please check your configuration.",
            )
            raise FileNotFoundError(
                f"JSON file {json_path} does not exist."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise TypeError(
                f"Top-level JSON must be an object, got {type(payload).__name__}"
            )

        raw_providers: dict[str, Any] = cast(dict[str, Any], payload)
        all_providers_dict: dict[str, dict[str, OpenAIResponsesCompatible]] = {}
        for provider_id, models in raw_providers.items():
            all_providers_dict[provider_id] = {}
            if not isinstance(models, list):
                raise TypeError(
                    f"Invalid model format under provider {provider_id}. Expected a list."
                )
            models_list: list[Any] = cast(list[Any], models)
            for model_raw in models_list:
                if not isinstance(model_raw, dict):
                    raise TypeError(
                        f"Invalid model entry under provider {provider_id}. Expected an object."
                    )
                info = _ProviderModelInfo(cast(dict[str, Any], model_raw))
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
                )
                all_providers_dict[provider_id][info.model_name] = instance
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
        bucket_id = f"responses_{base_url}_{model_name}"
        self.token_bucket = rate_limit_manager.get_or_create_bucket(
            bucket_id=bucket_id,
            capacity=rate_limit_capacity,
            refill_rate=rate_limit_refill_rate,
        )
        initial_key = api_key_pool.get_least_loaded_key()
        self.client: Optional[AsyncOpenAI] = AsyncOpenAI(
            api_key=initial_key,
            base_url=self.base_url,
        )
        self._current_key = initial_key
        self._clients: dict[str, AsyncOpenAI] = {initial_key: self.client}

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

    async def _close_stream_response(self, response: object) -> None:
        close_method = getattr(response, "close", None)
        if callable(close_method):
            try:
                result = close_method()
                if isawaitable(result):
                    await cast(Awaitable[None], result)
            except Exception as close_exc:
                push_warning(
                    f"{self.model_name} failed to close stream response: {close_exc}",
                )
            return
        aclose_method = getattr(response, "aclose", None)
        if callable(aclose_method):
            try:
                result = aclose_method()
                if isawaitable(result):
                    await cast(Awaitable[None], result)
            except Exception as close_exc:
                push_warning(
                    f"{self.model_name} failed to close stream response: {close_exc}",
                )

    async def aclose(self) -> None:
        """Close the underlying provider client."""

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

    def _count_tokens(self, response: Response) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        return (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )

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
                    raise Exception("Rate limit: token bucket acquire timed out")

                self.key_pool.increment_task_count(key)
                task_counted = True
                push_debug(
                    f"OpenAIResponsesCompatible::chat: model={self.model_name} "
                    f"message_count={len(request.messages)}",
                )
                request_kwargs = _request_to_responses_kwargs(request)
                response = await await_with_cancellation(
                    lambda: client.responses.create(
                        model=self.model_name,
                        stream=False,
                        timeout=timeout,
                        **request_kwargs,
                    ),
                    cancellation,
                )
                self._apply_usage_to_context(response)
                completion = _response_to_completion(response)
                self.key_pool.decrement_task_count(key)
                task_counted = False
                return completion
            except BaseException as exc:
                if task_counted:
                    self.key_pool.decrement_task_count(key)
                    task_counted = False
                if not isinstance(exc, Exception):
                    raise
                attempt += 1
                push_warning(
                    f"{self.model_name} Responses interface attempt {attempt} "
                    f"failed for message_count={len(request.messages)}: {exc}",
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
                    raise Exception("Rate limit: 令牌桶获取令牌超时")

                self.key_pool.increment_task_count(key)
                task_counted = True
                push_debug(
                    f"OpenAIResponsesCompatible::chat_stream: model={self.model_name} "
                    f"message_count={len(request.messages)}",
                )
                request_kwargs = _request_to_responses_kwargs(request)
                response_stream = await await_with_cancellation(
                    lambda: client.responses.create(
                        model=self.model_name,
                        stream=True,
                        timeout=timeout,
                        **request_kwargs,
                    ),
                    cancellation,
                )
                response_stream = cast(
                    AsyncGenerator[ResponseStreamEvent, None], response_stream
                )

                current_response_id = ""
                current_created = 0
                current_model = self.model_name
                tool_call_contexts_by_item_id: dict[str, dict[str, object]] = {}
                tool_call_contexts_by_output_index: dict[int, dict[str, object]] = {}
                next_tool_call_index = 0
                terminal_received = False

                try:
                    response_iter = response_stream.__aiter__()
                    while True:
                        try:
                            event = await await_with_cancellation(
                                response_iter.__anext__,
                                cancellation,
                            )
                        except StopAsyncIteration:
                            break
                        event_type = getattr(event, "type", "")
                        if event_type == "response.created":
                            response_obj = getattr(event, "response", None)
                            if response_obj is not None:
                                rid = getattr(response_obj, "id", "") or ""
                                if rid:
                                    current_response_id = rid
                                ca = getattr(response_obj, "created_at", 0) or 0
                                if ca:
                                    current_created = int(ca)
                                m = getattr(response_obj, "model", "") or ""
                                if m:
                                    current_model = m
                            continue

                        if event_type == "response.output_text.delta":
                            yielded_output = True
                            yield _make_text_chunk(
                                response_id=current_response_id,
                                created=current_created,
                                model=current_model,
                                delta_text=getattr(event, "delta", "") or "",
                            )
                            continue

                        if event_type in (
                            "response.reasoning_text.delta",
                            "response.reasoning_summary_text.delta",
                        ):
                            yielded_output = True
                            yield _make_reasoning_chunk(
                                response_id=current_response_id,
                                created=current_created,
                                model=current_model,
                                delta_text=getattr(event, "delta", "") or "",
                            )
                            continue

                        if event_type == "response.refusal.delta":
                            yielded_output = True
                            yield _make_refusal_chunk(
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
                                context: dict[str, object] = {
                                    "index": next_tool_call_index,
                                    "tool_call_id": (
                                        getattr(item, "call_id", "")
                                        or getattr(item, "id", "")
                                        or ""
                                    ),
                                    "name": getattr(item, "name", None),
                                }
                                if item_id:
                                    tool_call_contexts_by_item_id[item_id] = context
                                tool_call_contexts_by_output_index[output_index] = context
                                next_tool_call_index += 1
                                yielded_output = True
                                yield _make_tool_call_chunk(
                                    response_id=current_response_id,
                                    created=current_created,
                                    model=current_model,
                                    tool_call_index=cast(int, context["index"]),
                                    tool_call_id=cast(str, context["tool_call_id"]),
                                    name=cast(Optional[str], context["name"]),
                                    arguments_delta="",
                                )
                            continue

                        if event_type == "response.function_call_arguments.delta":
                            event_item_id = getattr(event, "item_id", "") or ""
                            event_output_index = getattr(event, "output_index", None)
                            tool_call_context: Optional[dict[str, object]] = None
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
                                tool_call_context = {
                                    "index": next_tool_call_index,
                                    "tool_call_id": event_item_id,
                                    "name": None,
                                }
                                next_tool_call_index += 1
                            idx = cast(int, tool_call_context.get("index"))
                            tcid = cast(str, tool_call_context.get("tool_call_id", "") or "")
                            tname = cast(Optional[str], tool_call_context.get("name", None))
                            delta = getattr(event, "delta", "") or ""
                            emitted = cast(
                                str, tool_call_context.get("arguments_emitted", "")
                            )
                            tool_call_context["arguments_emitted"] = emitted + delta
                            yielded_output = True
                            yield _make_tool_call_chunk(
                                response_id=current_response_id,
                                created=current_created,
                                model=current_model,
                                tool_call_index=idx,
                                tool_call_id=tcid,
                                name=tname,
                                arguments_delta=delta,
                            )
                            continue

                        if event_type == "response.function_call_arguments.done":
                            event_item_id = getattr(event, "item_id", "") or ""
                            event_output_index = getattr(event, "output_index", None)
                            call_context: dict[str, object] | None = (
                                tool_call_contexts_by_item_id.get(event_item_id)
                                if event_item_id
                                else None
                            )
                            if call_context is None and isinstance(event_output_index, int):
                                call_context = tool_call_contexts_by_output_index.get(
                                    event_output_index
                                )
                            if call_context is None:
                                raise RuntimeError(
                                    "Responses tool-call completion has no context"
                                )
                            arguments = getattr(event, "arguments", "") or ""
                            emitted = cast(
                                str, call_context.get("arguments_emitted", "")
                            )
                            if not arguments.startswith(emitted):
                                raise RuntimeError(
                                    "Responses tool-call arguments are not cumulative"
                                )
                            suffix = arguments[len(emitted) :]
                            call_context["arguments_emitted"] = arguments
                            if suffix:
                                yielded_output = True
                                yield _make_tool_call_chunk(
                                    response_id=current_response_id,
                                    created=current_created,
                                    model=current_model,
                                    tool_call_index=cast(int, call_context["index"]),
                                    tool_call_id=cast(
                                        str, call_context["tool_call_id"]
                                    ),
                                    name=cast(Optional[str], call_context["name"]),
                                    arguments_delta=suffix,
                                )
                            continue

                        if event_type == "response.output_item.done":
                            item = getattr(event, "item", None)
                            if getattr(item, "type", None) == "reasoning":
                                model_dump = getattr(item, "model_dump", None)
                                if not callable(model_dump):
                                    raise RuntimeError(
                                        "Responses reasoning item cannot be serialized"
                                    )
                                payload = cast(
                                    dict[str, object],
                                    model_dump(mode="json", exclude_none=True),
                                )
                                signature = (
                                    _RESPONSES_REASONING_SIGNATURE_PREFIX
                                    + json.dumps(
                                        payload,
                                        ensure_ascii=True,
                                        separators=(",", ":"),
                                        sort_keys=True,
                                    )
                                )
                                yielded_output = True
                                yield _make_reasoning_chunk(
                                    response_id=current_response_id,
                                    created=current_created,
                                    model=current_model,
                                    delta_text="",
                                    signature=signature,
                                )
                            continue

                        if event_type == "response.completed":
                            completed_response = getattr(event, "response", None)
                            if completed_response is not None:
                                cr = cast(Response, completed_response)
                                self._apply_usage_to_context(cr)
                                terminal_received = True
                                yield _make_finish_chunk(response=cr)
                            break

                        if event_type == "response.incomplete":
                            terminal_response = getattr(event, "response", None)
                            reason = getattr(
                                getattr(terminal_response, "incomplete_details", None),
                                "reason",
                                None,
                            )
                            if reason not in ("max_output_tokens", "content_filter"):
                                raise RuntimeError(
                                    "Responses stream ended incomplete for "
                                    f"unsupported reason: {reason}"
                                )
                            if terminal_response is None:
                                raise RuntimeError(
                                    "Responses incomplete event has no response"
                                )
                            terminal_received = True
                            yield _make_finish_chunk(response=cast(Response, terminal_response))
                            break

                        if event_type == "response.failed":
                            terminal_response = getattr(event, "response", None)
                            status = getattr(terminal_response, "status", event_type)
                            detail = getattr(terminal_response, "error", None)
                            raise RuntimeError(
                                f"Responses stream ended with status {status}: {detail}"
                            )
                    if not terminal_received:
                        raise RuntimeError(
                            "Responses stream ended without a terminal event"
                        )
                finally:
                    await self._close_stream_response(response_stream)

                self.key_pool.decrement_task_count(key)
                task_counted = False
                break
            except BaseException as exc:
                if task_counted:
                    self.key_pool.decrement_task_count(key)
                    task_counted = False
                if not isinstance(exc, Exception):
                    raise
                if yielded_output:
                    raise
                attempt += 1
                push_warning(
                    f"{self.model_name} Responses interface attempt {attempt} "
                    f"failed for message_count={len(request.messages)}: {exc}",
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
