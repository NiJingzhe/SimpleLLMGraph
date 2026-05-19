"""Response information extraction helpers."""

from __future__ import annotations

from typing import Optional, Union

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage


def extract_usage_from_response(
    response: Union[ChatCompletion, ChatCompletionChunk, None],
) -> Optional[CompletionUsage]:
    """从LLM响应中提取用量信息。

    Args:
        response: OpenAI API的ChatCompletion或ChatCompletionChunk响应对象

    Returns:
        CompletionUsage 对象（包含 prompt_tokens, completion_tokens, total_tokens），
        如果无法提取则返回None
    """
    if response is None:
        return None

    try:
        if hasattr(response, "usage") and response.usage:
            # 如果已经是 CompletionUsage 对象，直接返回
            if isinstance(response.usage, CompletionUsage):
                return response.usage
            # 如果是字典，转换为对象
            if isinstance(response.usage, dict):
                return CompletionUsage(
                    prompt_tokens=response.usage.get("prompt_tokens", 0),
                    completion_tokens=response.usage.get("completion_tokens", 0),
                    total_tokens=response.usage.get("total_tokens", 0),
                )
            # 如果有属性，尝试直接访问
            return CompletionUsage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                completion_tokens=getattr(response.usage, "completion_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
            )
    except (AttributeError, TypeError):
        pass
    return None
