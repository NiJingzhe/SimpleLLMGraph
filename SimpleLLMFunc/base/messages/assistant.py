"""Assistant message construction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from SimpleLLMFunc.type.message import NormalizedMessageParam


def build_assistant_tool_message(
    tool_calls: List[Dict[str, Any]],
    reasoning_details: Optional[List[Dict[str, Any]]] = None,
) -> NormalizedMessageParam:
    """Construct the assistant message containing tool call descriptors.

    Args:
        tool_calls: 工具调用列表
        reasoning_details: 可选的推理细节（如 Google Gemini 的 reasoning_details）

    Returns:
        assistant 消息字典
    """
    _ = reasoning_details

    if tool_calls:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
    return {}


def build_assistant_response_message(content: str) -> NormalizedMessageParam:
    """Construct a plain assistant response message."""

    return {
        "role": "assistant",
        "content": content,
    }
