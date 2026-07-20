"""Provider-neutral compiled tool-result content."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from SimpleLLMFunc.context.ir.parts import Image, InputTextPart


class ToolResult(BaseModel):
    """Model-visible result produced by one tool's result compiler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | list[InputTextPart | Image]
    as_user_message: bool = False

    @model_validator(mode="after")
    def validate_content(self) -> "ToolResult":
        if isinstance(self.content, list) and not self.content:
            raise ValueError("multimodal tool result content must not be empty")
        return self

    @property
    def requires_user_message(self) -> bool:
        return self.as_user_message or isinstance(self.content, list)
