"""Tool result validation and serialization helpers."""

from __future__ import annotations

import json
from typing import Any

from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


def _is_image_payload(value: Any) -> bool:
    return isinstance(value, (ImgPath, ImgUrl))


def _is_image_payload_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_is_image_payload(item) for item in value)
    )


def serialize_tool_output_for_langfuse(result: Any) -> Any:
    """序列化工具输出以便langfuse记录。

    Args:
        result: 工具返回的原始结果

    Returns:
        序列化后的结果，适合langfuse记录
    """
    if isinstance(result, ImgPath):
        return {
            "type": "image_path",
            "path": str(result.path),
            "detail": result.detail,
        }

    if isinstance(result, ImgUrl):
        return {
            "type": "image_url",
            "url": result.url,
            "detail": result.detail,
        }

    if isinstance(result, tuple) and len(result) == 2:
        text_part, img_part = result
        if isinstance(text_part, str) and isinstance(img_part, (ImgPath, ImgUrl)):
            return {
                "type": "text_with_image",
                "text": text_part,
                "image": serialize_tool_output_for_langfuse(img_part),
            }
        if isinstance(text_part, str) and _is_image_payload_list(img_part):
            return {
                "type": "text_with_images",
                "text": text_part,
                "images": [
                    serialize_tool_output_for_langfuse(item) for item in img_part
                ],
            }

    if isinstance(result, Text):
        return str(result.content)

    # 对于其他类型，尝试直接返回（JSON可序列化的对象）或转为字符串
    try:
        json.dumps(result)
        return result
    except (TypeError, ValueError):
        return str(result)


def is_valid_tool_result(result: Any) -> bool:
    """Validate whether a tool return value is supported."""

    if isinstance(result, (ImgPath, ImgUrl)):
        return True

    if _is_image_payload_list(result):
        return True

    if isinstance(result, str):
        return True

    if isinstance(result, tuple) and len(result) == 2:
        text_part, img_part = result
        if isinstance(text_part, str) and isinstance(img_part, (ImgPath, ImgUrl)):
            return True
        if isinstance(text_part, str) and _is_image_payload_list(img_part):
            return True
        return False

    try:
        json.dumps(result)
        return True
    except (TypeError, ValueError):
        return False
