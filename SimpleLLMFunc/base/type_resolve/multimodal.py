"""Multimodal type checking helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from SimpleLLMFunc.type.chat_input import UserChatMessage
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


def has_multimodal_content(
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    exclude_params: Optional[List[str]] = None,
) -> bool:
    """Check whether arguments contain multimodal payloads."""

    exclude_params = exclude_params or []

    for param_name, param_value in arguments.items():
        if param_name in exclude_params:
            continue

        if param_name in type_hints:
            annotation = type_hints[param_name]
            if is_multimodal_type(param_value, annotation):
                return True
        elif isinstance(param_value, UserChatMessage):
            return True
    return False


def is_multimodal_type(value: Any, annotation: Any) -> bool:
    """Determine whether a value/annotation pair represents multimodal content."""

    from typing import List as TypingList, Union, get_args, get_origin

    if isinstance(value, (Text, ImgUrl, ImgPath, UserChatMessage)):
        return True

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        for arg_type in non_none_args:
            if is_multimodal_type(value, arg_type):
                return True
        return False

    if origin in (list, TypingList):
        if not args:
            return False
        element_type = args[0]
        if element_type in (Text, ImgUrl, ImgPath, UserChatMessage):
            return True
        if isinstance(value, (list, tuple)):
            return any(
                isinstance(item, (Text, ImgUrl, ImgPath, UserChatMessage))
                for item in value
            )
        return False

    if annotation in (Text, ImgUrl, ImgPath, UserChatMessage):
        return True

    return False

