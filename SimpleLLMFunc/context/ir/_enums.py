"""Enumerations for the provider-neutral context entry IR.

The IR is modelled on the OpenAI Chat Completions streaming spec, extended
with minimal first-class reasoning support (an effort parameter plus
reasoning content parts/deltas) so providers that surface thinking -- e.g.
Anthropic extended thinking or the OpenAI o-series -- can be represented
faithfully without inventing concepts outside the OpenAI-equivalent set.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """Conversation entry role, mirroring the OpenAI ``messages`` roles."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reason a choice stopped generating, per the Chat Completions spec."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"


class ReasoningEffort(str, Enum):
    """Requested reasoning intensity for reasoning-capable models.

    Carried on the request envelope (the OpenAI ``reasoning_effort`` field);
    ``NONE`` opts out of surfaced reasoning where the provider supports it.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolCallType(str, Enum):
    """Tool call discriminator. Only ``function`` is defined by the spec."""

    FUNCTION = "function"


class ImageDetail(str, Enum):
    """Fidelity hint for image parts, per the OpenAI ``image_url.detail``."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class AudioFormat(str, Enum):
    """Container format for inline audio parts."""

    WAV = "wav"
    MP3 = "mp3"


class StreamObject(str, Enum):
    """``object`` discriminator of a streamed chunk."""

    CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
