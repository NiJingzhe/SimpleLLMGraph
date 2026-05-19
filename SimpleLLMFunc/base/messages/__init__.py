"""Helpers for constructing structured assistant messages."""

from SimpleLLMFunc.base.messages.assistant import (
    build_assistant_response_message,
    build_assistant_tool_message,
)
from SimpleLLMFunc.base.messages.extraction import extract_usage_from_response
from SimpleLLMFunc.base.messages.multimodal import (
    build_multimodal_content,
    create_image_path_content,
    create_image_url_content,
    create_text_content,
    parse_multimodal_parameter,
)
from SimpleLLMFunc.base.messages.validation import (
    ALLOWED_MESSAGE_ROLES,
    is_valid_content_for_role,
    message_to_dict,
    normalize_and_validate_messages,
    validate_message_shape,
    validate_tool_linkage,
)

__all__ = [
    "build_assistant_tool_message",
    "build_assistant_response_message",
    "extract_usage_from_response",
    "build_multimodal_content",
    "parse_multimodal_parameter",
    "create_text_content",
    "create_image_url_content",
    "create_image_path_content",
    "ALLOWED_MESSAGE_ROLES",
    "is_valid_content_for_role",
    "message_to_dict",
    "normalize_and_validate_messages",
    "validate_message_shape",
    "validate_tool_linkage",
]
