"""Multimodal content parts for context entries.

Mirrors the OpenAI Chat Completions content-part taxonomy -- ``input_text``,
``input_image``, ``input_audio``, ``output_text`` and ``output_audio`` --
plus a ``reasoning`` part that carries surfaced thinking. ``ContentPart`` is
a discriminated union keyed on ``type``.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
from pathlib import Path
from typing import Annotated, Literal, Self, Union
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from SimpleLLMFunc.context.ir._enums import AudioFormat, ImageDetail


class InputTextPart(BaseModel):
    """A text fragment supplied by the user or embedded in a user turn."""

    type: Literal["input_text"] = "input_text"
    text: str


class OutputTextPart(BaseModel):
    """A text fragment produced by the assistant in a prior turn."""

    type: Literal["output_text"] = "output_text"
    text: str


class InputImageURL(BaseModel):
    """Reference to an image, either a URL or a ``data:`` URI."""

    url: str
    detail: ImageDetail | None = None


class Image(BaseModel):
    """A web or inline image usable in inputs and typed tool results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["input_image"] = "input_image"
    image_url: str | InputImageURL

    @field_validator("image_url")
    @classmethod
    def validate_image_url(
        cls,
        value: str | InputImageURL,
    ) -> str | InputImageURL:
        url = value if isinstance(value, str) else value.url
        if url.startswith("data:image/"):
            header, separator, encoded = url.partition(",")
            if not separator or not header.endswith(";base64") or not encoded:
                raise ValueError("image data URL must contain base64 image data")
            try:
                base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image data URL contains invalid base64") from exc
            return value
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("image URL must use http, https, or a base64 data URL")
        return value

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        detail: ImageDetail | None = None,
    ) -> Self:
        """Create an image from a public HTTP(S) URL or image data URL."""

        image_url: str | InputImageURL = (
            url if detail is None else InputImageURL(url=url, detail=detail)
        )
        return cls(image_url=image_url)

    @classmethod
    def from_base64(
        cls,
        data: bytes | str,
        *,
        media_type: str,
        detail: ImageDetail | None = None,
    ) -> Self:
        """Create an inline image from bytes or an already-base64 string."""

        if (
            not media_type.startswith("image/")
            or len(media_type) == len("image/")
            or any(marker in media_type for marker in (";", ",", " "))
        ):
            raise ValueError("media_type must be an image MIME type")
        if isinstance(data, bytes):
            encoded = base64.b64encode(data).decode("ascii")
        else:
            try:
                base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("data must be valid base64") from exc
            encoded = data
        return cls.from_url(
            f"data:{media_type};base64,{encoded}",
            detail=detail,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        media_type: str | None = None,
        detail: ImageDetail | None = None,
    ) -> Self:
        """Read a local image immediately and embed it as a base64 data URL."""

        resolved = Path(path)
        detected = media_type or mimetypes.guess_type(resolved.name)[0]
        if detected is None:
            raise ValueError("cannot determine image MIME type from path")
        return cls.from_base64(
            resolved.read_bytes(),
            media_type=detected,
            detail=detail,
        )


InputImagePart = Image


class InputAudioData(BaseModel):
    """Inline audio payload, base64-encoded."""

    data: str
    format: AudioFormat


class InputAudioPart(BaseModel):
    """Audio supplied to the model."""

    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudioData


class OutputAudioData(BaseModel):
    """Audio produced by the assistant. Fields follow the OpenAI output
    audio object; ``data`` / ``transcript`` / ``expires_at`` may be absent
    depending on the provider and transport.
    """

    id: str
    data: str | None = None
    expires_at: int | None = None
    transcript: str | None = None
    format: AudioFormat | None = None


class OutputAudioPart(BaseModel):
    """Audio produced by the assistant."""

    type: Literal["output_audio"] = "output_audio"
    output_audio: OutputAudioData


class ReasoningPart(BaseModel):
    """Surfaced reasoning / thinking content.

    Carries the reasoning text that a provider chose to expose (e.g.
    Anthropic extended thinking). ``signature`` preserves provider
    round-trip metadata where the provider requires it to be echoed back.
    """

    type: Literal["reasoning"] = "reasoning"
    reasoning: str
    signature: str | None = None


ContentPart = Annotated[
    Union[
        InputTextPart,
        OutputTextPart,
        Image,
        InputAudioPart,
        OutputAudioPart,
        ReasoningPart,
    ],
    Field(discriminator="type"),
]
