"""Message protocol normalization and validation helpers."""

from __future__ import annotations

import copy
from typing import Any, cast

from SimpleLLMFunc.type.message import MessageList, NormalizedMessageList, NormalizedMessageParam

ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool", "function"}


def message_to_dict(message: Any, index: int) -> NormalizedMessageParam:
    if isinstance(message, dict):
        return cast(NormalizedMessageParam, copy.deepcopy(message))

    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=False)
        if isinstance(dumped, dict):
            return cast(NormalizedMessageParam, dumped)

    raise ValueError(f"message at index {index} must be a dict-like chat message")


def is_valid_content_for_role(role: str, content: Any) -> bool:
    if role == "system":
        return isinstance(content, str)
    if role == "user":
        return isinstance(content, (str, list))
    if role == "assistant":
        return content is None or isinstance(content, (str, list))
    if role in {"tool", "function"}:
        return isinstance(content, str)
    return False


def validate_message_shape(message: NormalizedMessageParam, index: int) -> None:
    role = message.get("role")
    if not isinstance(role, str) or role not in ALLOWED_MESSAGE_ROLES:
        raise ValueError(f"Invalid message role at index {index}: {role!r}")

    if "content" not in message:
        raise ValueError(
            f"Message at index {index} is missing required field 'content'"
        )

    if not is_valid_content_for_role(role, message.get("content")):
        raise ValueError(
            f"Invalid content for role '{role}' at index {index}: "
            f"{type(message.get('content')).__name__}"
        )

    if role == "assistant" and "tool_calls" in message:
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ValueError("assistant.tool_calls must be a list when present")
        if isinstance(tool_calls, list):
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    raise ValueError(
                        "assistant.tool_calls entries must be dict objects "
                        f"(index {index}, tool_call {call_index})"
                    )
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(
                        "assistant.tool_calls entries must contain non-empty id "
                        f"(index {index}, tool_call {call_index})"
                    )

    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValueError("tool messages must contain non-empty tool_call_id")


def validate_tool_linkage(messages: MessageList | NormalizedMessageList) -> None:
    pending_tool_call_ids = []

    for index, message in enumerate(messages):
        role = message.get("role")

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if pending_tool_call_ids and not tool_calls:
                raise ValueError(
                    "Missing tool results before next assistant message "
                    f"(index {index})"
                )
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    call_id = tool_call.get("id")
                    if isinstance(call_id, str) and call_id:
                        pending_tool_call_ids.append(call_id)
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not pending_tool_call_ids:
                raise ValueError(
                    "Tool message appears without preceding assistant tool_calls "
                    f"(index {index})"
                )
            if tool_call_id not in pending_tool_call_ids:
                raise ValueError(
                    "tool_call_id does not match pending assistant tool_calls "
                    f"(index {index})"
                )
            pending_tool_call_ids.remove(tool_call_id)
            continue

        if pending_tool_call_ids:
            raise ValueError(
                "Pending assistant tool_calls must be followed by matching tool "
                f"messages before role '{role}' (index {index})"
            )

    if pending_tool_call_ids:
        raise ValueError("Unmatched assistant tool_calls without tool results")


def normalize_and_validate_messages(messages: MessageList) -> NormalizedMessageList:
    normalized: NormalizedMessageList = [
        message_to_dict(message, index) for index, message in enumerate(messages)
    ]
    for index, message in enumerate(normalized):
        validate_message_shape(message, index)
    validate_tool_linkage(cast(NormalizedMessageList, normalized))
    return cast(NormalizedMessageList, normalized)


__all__ = [
    "ALLOWED_MESSAGE_ROLES",
    "is_valid_content_for_role",
    "message_to_dict",
    "normalize_and_validate_messages",
    "validate_message_shape",
    "validate_tool_linkage",
]
