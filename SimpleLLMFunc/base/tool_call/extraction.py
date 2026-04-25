"""Tool call extraction helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from SimpleLLMFunc.logger import push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location

# 从统一类型系统导入类型
from SimpleLLMFunc.type.message import ReasoningDetail
from SimpleLLMFunc.type.tool_call import (
    AccumulatedToolCall,
    ToolCallArguments,
    ToolCallFunctionInfo,
)


def _try_parse_tool_call_arguments(candidate: str) -> ToolCallArguments | None:
    """Try parsing one candidate JSON string as tool-call arguments dict."""

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    if isinstance(parsed, dict):
        return parsed

    return None


def _trim_trailing_char(value: str, char: str, count: int) -> str:
    """Trim up to `count` trailing `char` from value."""

    trimmed = value
    while count > 0 and trimmed.endswith(char):
        trimmed = trimmed[:-1]
        count -= 1
    return trimmed


def _close_unfinished_json_payload(arguments_str: str) -> str:
    """Close common unfinished JSON structures for streaming arguments."""

    value = arguments_str.strip()
    if not value:
        return "{}"

    in_string = False
    escaped = False
    brace_balance = 0
    bracket_balance = 0

    for char in value:
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            brace_balance += 1
        elif char == "}":
            brace_balance -= 1
        elif char == "[":
            bracket_balance += 1
        elif char == "]":
            bracket_balance -= 1

    candidate = value

    if in_string:
        candidate += '"'

    if bracket_balance < 0:
        candidate = _trim_trailing_char(candidate, "]", -bracket_balance)
    if brace_balance < 0:
        candidate = _trim_trailing_char(candidate, "}", -brace_balance)

    if bracket_balance > 0:
        candidate += "]" * bracket_balance
    if brace_balance > 0:
        candidate += "}" * brace_balance

    return candidate


def _build_argument_candidates(
    arguments_str: str,
    *,
    allow_closure: bool,
) -> List[str]:
    """Build deterministic candidates for malformed tool-call arguments."""

    value = arguments_str.strip()
    if not value:
        return ["{}"]

    candidates: List[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(value)

    if value.startswith("{{"):
        add(value[1:])
    if value.endswith("}}"):
        add(value[:-1])
    if value.startswith("{{") and value.endswith("}}"):
        add(value[1:-1])

    if allow_closure:
        closed = _close_unfinished_json_payload(value)
        add(closed)
        if closed.startswith("{{"):
            add(closed[1:])
        if closed.endswith("}}"):
            add(closed[:-1])
        if closed.startswith("{{") and closed.endswith("}}"):
            add(closed[1:-1])

    return candidates


def parse_tool_call_arguments(
    arguments_str: str,
    *,
    allow_closure: bool = False,
) -> ToolCallArguments | None:
    """Parse tool-call arguments using deterministic repair candidates.

    Args:
        arguments_str: Raw JSON argument string from model output.
        allow_closure: Whether to try auto-closing unfinished JSON payload.

    Returns:
        Parsed dict when successful, otherwise None.
    """

    for candidate in _build_argument_candidates(
        arguments_str,
        allow_closure=allow_closure,
    ):
        parsed = _try_parse_tool_call_arguments(candidate)
        if parsed is not None:
            return parsed

    return None


def repair_tool_call_arguments(arguments_str: str) -> str:
    """Repair malformed tool-call arguments and return best JSON string."""

    for candidate in _build_argument_candidates(arguments_str, allow_closure=True):
        if _try_parse_tool_call_arguments(candidate) is not None:
            return candidate

    value = arguments_str.strip()
    return value if value else "{}"


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Extract tool-call metadata from a synchronous response."""

    tool_calls: List[Dict[str, Any]] = []

    try:
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": getattr(tool_call, "type", "function"),
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
    except Exception as exc:
        push_error(f"Error extracting tool calls: {str(exc)}")
    finally:
        return tool_calls


def accumulate_tool_calls_from_chunks(
    tool_call_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge tool-call chunks emitted during streaming responses."""

    accumulated_calls: Dict[int, AccumulatedToolCall] = {}

    for chunk in tool_call_chunks:
        index = chunk.get("index")
        if index is None:
            push_warning(
                "Tool call chunk is missing 'index'; skipping",
                location=get_location(),
            )
            continue

        if index not in accumulated_calls:
            accumulated_calls[index] = AccumulatedToolCall(
                id=None,
                type=None,
                function=ToolCallFunctionInfo(name=None, arguments=""),
            )

        if chunk.get("id"):
            accumulated_calls[index]["id"] = chunk["id"]
        if chunk.get("type"):
            accumulated_calls[index]["type"] = chunk["type"]

        if "function" in chunk:
            function_chunk = chunk["function"]
            func_info = accumulated_calls[index]["function"]
            if function_chunk.get("name"):
                func_info["name"] = function_chunk["name"]
            if function_chunk.get("arguments"):
                func_info["arguments"] += function_chunk["arguments"]

    complete_tool_calls: List[Dict[str, Any]] = []
    for call in accumulated_calls.values():
        if call["id"] and call["function"]["name"]:
            if not call["type"]:
                call["type"] = "function"
            repaired_arguments = repair_tool_call_arguments(
                call["function"]["arguments"]
            )
            complete_tool_calls.append(
                {
                    "id": call["id"],
                    "type": call["type"],
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": repaired_arguments,
                    },
                }
            )

    return complete_tool_calls


def extract_tool_calls_from_stream_response(chunk: Any) -> List[Dict[str, Any]]:
    """Extract tool-call fragments from a streaming chunk."""

    tool_call_chunks: List[Dict[str, Any]] = []

    try:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        tool_call_chunk: Dict[str, Any] = {
                            "index": getattr(tool_call, "index", None),
                            "id": getattr(tool_call, "id", None),
                            "type": getattr(tool_call, "type", None),
                        }

                        if hasattr(tool_call, "function") and tool_call.function:
                            function_info: Dict[str, Any] = {}
                            if (
                                hasattr(tool_call.function, "name")
                                and tool_call.function.name
                            ):
                                function_info["name"] = tool_call.function.name
                            if (
                                hasattr(tool_call.function, "arguments")
                                and tool_call.function.arguments
                            ):
                                function_info["arguments"] = (
                                    tool_call.function.arguments
                                )

                            if function_info:
                                tool_call_chunk["function"] = function_info

                        tool_call_chunks.append(tool_call_chunk)
    except Exception as exc:
        push_error(f"Error extracting streaming tool calls: {str(exc)}")

    return tool_call_chunks


def extract_reasoning_details(response: Any) -> List[ReasoningDetail]:
    """从非流式响应中提取 reasoning_details。

    Args:
        response: LLM 响应对象

    Returns:
        reasoning_details 列表
    """
    reasoning_details: List[ReasoningDetail] = []

    try:
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message

            # 尝试从 message 中获取 reasoning_details
            reasoning_details_raw = None
            if hasattr(message, "reasoning_details"):
                reasoning_details_raw = message.reasoning_details
            elif isinstance(message, dict) and "reasoning_details" in message:
                reasoning_details_raw = message["reasoning_details"]

            if reasoning_details_raw:
                for detail in reasoning_details_raw:
                    # 处理 detail 可能是字典或对象的情况
                    if isinstance(detail, dict):
                        reasoning_details.append(
                            ReasoningDetail(
                                id=detail.get("id", ""),
                                format=detail.get("format", ""),
                                index=detail.get("index", 0),
                                type=detail.get("type", "reasoning.encrypted"),
                                data=detail.get("data", ""),
                            )
                        )
                    else:
                        # detail 是对象，尝试获取属性
                        reasoning_details.append(
                            ReasoningDetail(
                                id=getattr(detail, "id", "")
                                if hasattr(detail, "id")
                                else "",
                                format=getattr(detail, "format", "")
                                if hasattr(detail, "format")
                                else "",
                                index=getattr(detail, "index", 0)
                                if hasattr(detail, "index")
                                else 0,
                                type=getattr(detail, "type", "reasoning.encrypted")
                                if hasattr(detail, "type")
                                else "reasoning.encrypted",
                                data=getattr(detail, "data", "")
                                if hasattr(detail, "data")
                                else "",
                            )
                        )
    except Exception as exc:
        push_error(f"Error extracting reasoning_details: {str(exc)}")
        import traceback

        push_error(f"Detailed error: {traceback.format_exc()}")

    return reasoning_details


def extract_reasoning_details_from_stream(chunk: Any) -> List[ReasoningDetail]:
    """从流式响应 chunk 中提取 reasoning_details。

    Args:
        chunk: 流式响应的一个 chunk

    Returns:
        reasoning_details 列表
    """
    reasoning_details: List[ReasoningDetail] = []

    try:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta

                # 尝试从 delta 中获取 reasoning_details
                reasoning_details_raw = None
                if hasattr(delta, "reasoning_details"):
                    reasoning_details_raw = delta.reasoning_details
                elif isinstance(delta, dict) and "reasoning_details" in delta:
                    reasoning_details_raw = delta["reasoning_details"]

                if reasoning_details_raw:
                    for detail in reasoning_details_raw:
                        # 处理 detail 可能是字典或对象的情况
                        if isinstance(detail, dict):
                            reasoning_details.append(
                                ReasoningDetail(
                                    id=detail.get("id", ""),
                                    format=detail.get("format", ""),
                                    index=detail.get("index", 0),
                                    type=detail.get("type", "reasoning.encrypted"),
                                    data=detail.get("data", ""),
                                )
                            )
                        else:
                            # detail 是对象，尝试获取属性
                            reasoning_details.append(
                                ReasoningDetail(
                                    id=getattr(detail, "id", "")
                                    if hasattr(detail, "id")
                                    else "",
                                    format=getattr(detail, "format", "")
                                    if hasattr(detail, "format")
                                    else "",
                                    index=getattr(detail, "index", 0)
                                    if hasattr(detail, "index")
                                    else 0,
                                    type=getattr(detail, "type", "reasoning.encrypted")
                                    if hasattr(detail, "type")
                                    else "reasoning.encrypted",
                                    data=getattr(detail, "data", "")
                                    if hasattr(detail, "data")
                                    else "",
                                )
                            )
    except Exception as exc:
        push_error(f"Error extracting streaming reasoning_details: {str(exc)}")
        import traceback

        push_error(f"Detailed error: {traceback.format_exc()}")

    return reasoning_details
