"""Step 5: Process chat response stream."""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Literal, Tuple

from SimpleLLMFunc.base.post_process import (
    extract_content_from_response,
    extract_content_from_stream_response,
)
from SimpleLLMFunc.logger import app_log
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.type import MessageList


def extract_stream_response_content(chunk: Any, func_name: str) -> str:
    """从流式响应 chunk 中提取内容"""
    return extract_content_from_stream_response(chunk, func_name)


def process_single_chat_response(
    response: Any,
    return_mode: Literal["text", "raw"],
    stream: bool,
    func_name: str,
) -> Any:
    """处理单个响应"""
    if return_mode == "raw":
        return response

    # text 模式：提取内容
    if stream:
        return extract_stream_response_content(response, func_name)
    else:
        return extract_content_from_response(response, func_name) or ""


async def process_chat_response_stream(
    response_stream: AsyncGenerator[Tuple[Any, MessageList], None],
    return_mode: Literal["text", "raw"],
    messages: MessageList,  # 初始消息，用于兼容性
    func_name: str,
    stream: bool,
) -> AsyncGenerator[Tuple[Any, MessageList], None]:
    """处理流式响应的完整流程"""
    current_messages = messages.copy()  # 初始消息
    
    async for response, updated_messages in response_stream:
        # 更新当前消息为最新版本（包含工具调用结果）
        current_messages = updated_messages
        
        # 记录响应日志
        app_log(
            f"LLM Chat '{func_name}' received response:"
            f"\n{json.dumps(response, default=str, ensure_ascii=False, indent=2)}",
            location=get_location(),
        )

        # 处理单个响应
        content = process_single_chat_response(
            response,
            return_mode,
            stream,
            func_name,
        )

        # Yield 响应和更新后的历史（包含工具调用结果）
        yield content, current_messages.copy()

    # 流结束标记（text 模式）
    if return_mode == "text":
        yield "", current_messages.copy()
