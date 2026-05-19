from abc import ABC, abstractmethod
from typing import Optional, Dict, Iterable, Literal, AsyncGenerator

from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.logger import get_current_trace_id
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion


DEFAULT_CONTEXT_WINDOW = 200_000


class LLM_Interface(ABC):
    @abstractmethod
    def __init__(
        self,
        api_key_pool: APIKeyPool,
        model_name: str,
        base_url: Optional[str] = None,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
    ):
        self.input_token_count = 0
        self.output_token_count = 0
        self.model_name = model_name
        self.context_window = (
            DEFAULT_CONTEXT_WINDOW if context_window is None else context_window
        )

    @abstractmethod
    async def chat(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[False] = False,
        messages: Iterable[Dict[str, str]] = [{"role": "user", "content": ""}],
        timeout: Optional[int] = None,
        *args,
        **kwargs,
    ) -> ChatCompletion:

        pass

    @abstractmethod
    async def chat_stream(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[True] = True,
        messages: Iterable[Dict[str, str]] = [{"role": "user", "content": ""}],
        timeout: Optional[int] = None,
        *args,
        **kwargs,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:

        if False:
            yield ChatCompletionChunk(
                id="", created=0, model="", object="chat.completion.chunk", choices=[]
            )
