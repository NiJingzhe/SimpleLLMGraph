"""Canonical user-message input object for ``llm_chat``.

``UserChatMessage`` is the one supported multimodal input shape for chat
agents.  It mirrors the OpenAI Chat Completions user-message schema while
keeping construction Pythonic.  Future input modalities should extend this
object instead of adding parallel chat input aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypeAlias, Union, cast

from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl

UserChatContentPart: TypeAlias = Dict[str, Any]
UserChatContent: TypeAlias = Union[str, List[UserChatContentPart]]
UserChatContentInput: TypeAlias = Union[str, ImgUrl, ImgPath, UserChatContentPart]


def _text_part(text: str) -> UserChatContentPart:
    return {"type": "text", "text": text}


def _image_url_part(image: ImgUrl) -> UserChatContentPart:
    image_url_data = {"url": image.url}
    if image.detail != "auto":
        image_url_data["detail"] = image.detail
    return {"type": "image_url", "image_url": image_url_data}


def _image_path_part(image: ImgPath) -> UserChatContentPart:
    data_url = f"data:{image.get_mime_type()};base64,{image.to_base64()}"
    image_url_data = {"url": data_url}
    if image.detail != "auto":
        image_url_data["detail"] = image.detail
    return {"type": "image_url", "image_url": image_url_data}


def _normalize_content_part(part: UserChatContentInput) -> UserChatContentPart:
    if isinstance(part, str):
        return _text_part(part)
    if isinstance(part, ImgUrl):
        return _image_url_part(part)
    if isinstance(part, ImgPath):
        return _image_path_part(part)
    if isinstance(part, dict):
        normalized = dict(part)
        part_type = normalized.get("type")
        if part_type == "text" and isinstance(normalized.get("text"), str):
            return normalized
        if part_type == "image_url" and isinstance(normalized.get("image_url"), dict):
            return normalized
        raise ValueError(
            "UserChatMessage content dict parts must be OpenAI-compatible "
            "text or image_url parts"
        )
    raise ValueError(f"Unsupported user chat content part: {type(part).__name__}")


def normalize_user_chat_content(value: Any) -> UserChatContent:
    """Normalize ``UserChatMessage`` content to OAI-compatible user content."""

    if isinstance(value, UserChatMessage):
        return normalize_user_chat_content(value.content)

    if isinstance(value, str):
        return value

    if isinstance(value, (ImgUrl, ImgPath)):
        return [_normalize_content_part(cast(UserChatContentInput, value))]

    if isinstance(value, dict):
        if value.get("role") == "user" and "content" in value:
            return normalize_user_chat_content(value.get("content"))
        return [_normalize_content_part(cast(UserChatContentInput, value))]

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        return [
            _normalize_content_part(cast(UserChatContentInput, part))
            for part in value
        ]

    raise ValueError(f"Unsupported user chat content: {type(value).__name__}")


@dataclass(frozen=True)
class UserChatMessage:
    """OpenAI-compatible user message object for ``llm_chat``.

    This is the canonical way to pass multimodal user input to chat agents.
    Plain-text-only agents can still use ``message: str`` for backwards
    compatibility, but multimodal input should go through this object.
    """

    content: UserChatContent
    role: Literal["user"] = "user"

    @classmethod
    def text(cls, text: str) -> "UserChatMessage":
        return cls(content=str(text))

    @classmethod
    def multimodal(cls, *parts: UserChatContentInput) -> "UserChatMessage":
        return cls(content=normalize_user_chat_content(list(parts)))

    def to_message(self) -> Dict[str, Any]:
        return {"role": self.role, "content": normalize_user_chat_content(self.content)}


def normalize_user_chat_message(value: Any) -> Dict[str, Any]:
    """Normalize canonical chat input to ``{"role": "user", "content": ...}``."""

    if isinstance(value, UserChatMessage):
        return value.to_message()

    if isinstance(value, dict) and value.get("role") == "user" and "content" in value:
        return {**value, "content": normalize_user_chat_content(value.get("content"))}

    raise ValueError(
        "llm_chat multimodal input must be a UserChatMessage or an "
        "OpenAI-compatible user message dict"
    )


__all__ = [
    "UserChatContent",
    "UserChatContentInput",
    "UserChatContentPart",
    "UserChatMessage",
    "normalize_user_chat_content",
    "normalize_user_chat_message",
]
