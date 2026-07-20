"""Request envelope for a provider-neutral Chat-Completions-style call.

``Request`` is the IR view of "what to send": model, conversation, tools and
the small set of OpenAI-equivalent sampling / reasoning parameters. Provider
adapters translate this to their wire format. Whether the call is streamed
is expressed by which :class:`LLM_Interface` method is invoked
(``chat`` vs ``chat_stream``), so the request itself carries no ``stream``
flag.

``extra`` is an escape hatch for provider-specific parameters that are not
yet first-class IR fields; adapters merge it into their wire kwargs with
the highest priority (above instance-level defaults).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from SimpleLLMFunc.context.ir._enums import ReasoningEffort
from SimpleLLMFunc.context.ir.messages import Conversation


class ToolFunction(BaseModel):
    """A tool function declaration. ``parameters`` is a JSON Schema dict."""

    name: str
    description: str | None = None
    parameters: dict[str, Any]


class Tool(BaseModel):
    """A tool declaration. Only the ``function`` type is defined."""

    type: Literal["function"] = "function"
    function: ToolFunction


class StreamOptions(BaseModel):
    """Options for a streaming request. ``include_usage`` requests a final
    chunk carrying token usage. Managed by the streaming adapter internally.
    """

    include_usage: bool = True


class Request(BaseModel):
    """A Chat-Completions-style request in the neutral IR.

    Notes:
        * ``model`` is informational at the IR level; a :class:`LLM_Interface`
          instance is bound to a specific model and uses ``self.model_name``
          for the actual call, so ``model`` may be ignored by adapters.
        * ``reasoning_effort`` selects reasoning intensity for
          reasoning-capable models; ``None`` leaves it to provider default.
        * ``extra`` carries provider-specific passthrough parameters; adapters
          merge it last so it overrides instance defaults.
    """

    model: str
    messages: Conversation
    tools: list[Tool] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    extra: dict[str, Any] | None = None
