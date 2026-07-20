"""Abstract base class for LLM provider interfaces.

The interface speaks the neutral context IR on its inner side (toward the
framework): :meth:`chat` consumes a :class:`Request` and returns a
:class:`Completion`, :meth:`chat_stream` consumes a :class:`Request` and
yields :class:`Chunk` items. Provider adapters translate between this IR
and their wire format, so the framework above never has to import
provider-specific SDK types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from SimpleLLMFunc.cancellation import CancellationToken
from SimpleLLMFunc.context.ir import Chunk, Completion, Request
from SimpleLLMFunc.interface.key_pool import APIKeyPool

DEFAULT_CONTEXT_WINDOW = 200_000


class LLM_Interface(ABC):
    """Provider-neutral interface for an LLM endpoint.

    An instance is bound to a single model (``model_name``) and an
    :class:`APIKeyPool`; per-call variation is carried by the
    :class:`Request` passed to ``chat`` / ``chat_stream``.
    """

    input_token_count: int
    output_token_count: int
    model_name: str
    base_url: str | None
    context_window: int

    @abstractmethod
    def __init__(
        self,
        api_key_pool: APIKeyPool,
        model_name: str,
        base_url: str | None = None,
        context_window: int | None = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        self.input_token_count = 0
        self.output_token_count = 0
        self.model_name = model_name
        self.base_url = base_url
        self.context_window = (
            DEFAULT_CONTEXT_WINDOW if context_window is None else context_window
        )

    @abstractmethod
    async def chat(
        self,
        request: Request,
        *,
        trace_id: str | None = None,
        timeout: int | None = 30,
        cancellation: CancellationToken | None = None,
    ) -> Completion:
        """Run a non-streaming request and return a :class:`Completion`."""

    @abstractmethod
    async def chat_stream(
        self,
        request: Request,
        *,
        trace_id: str | None = None,
        timeout: int | None = 30,
        cancellation: CancellationToken | None = None,
    ) -> AsyncGenerator[Chunk, None]:
        """Run a streaming request, yielding :class:`Chunk` items."""
        if False:
            yield Chunk(id="", created=0, model="", choices=[])

    async def aclose(self) -> None:
        """Release provider resources owned by this interface."""
