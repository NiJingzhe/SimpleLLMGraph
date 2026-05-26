"""OpenAI API message type definitions.

直接使用 OpenAI SDK 定义的消息类型，确保类型安全。

消息类型结构是固定的：
- role: "system" | "user" | "assistant" | "tool" | "function"
- content: str | List[Dict[str, Any]] | None (取决于 role)
- reasoning_details: 可选字段，某些模型（如 Google Gemini）会返回 reasoning 信息

类型说明：
- MessageParam: 单个消息参数类型，可以是 OpenAI SDK 的 ChatCompletionMessage 或扩展的 ExtendedMessageParam
- MessageList: 消息列表类型，用于传递对话历史
- ReasoningDetail: 推理细节类型（用于 Google Gemini 等模型）
- ExtendedMessageParam: 扩展的消息参数类型，支持额外的字段如 reasoning_details

使用示例：
    ```python
    from SimpleLLMFunc.type.message import MessageList, MessageParam
    
    messages: MessageList = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    ```
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, NotRequired, TypeAlias, TypedDict

ChatImageDetail: TypeAlias = Literal["low", "high", "auto"]


class ChatTextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ChatImageUrl(TypedDict, total=False):
    url: str
    detail: ChatImageDetail


class ChatImageUrlContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: ChatImageUrl


ChatContentPart: TypeAlias = ChatTextContentPart | ChatImageUrlContentPart
ChatMessageContent: TypeAlias = str | List[ChatContentPart]

# Reasoning detail 的类型定义
class ReasoningDetail(TypedDict):
    """推理细节的类型定义（用于 Google Gemini 等模型）"""
    id: str
    format: str
    index: int
    type: Literal["reasoning.encrypted"]
    data: str

# 扩展的消息参数类型，包含所有可能的字段
class ExtendedMessageParam(TypedDict, total=False):
    """扩展的消息参数类型，支持额外的字段如 reasoning_details
    
    包含 ChatCompletionMessage 的所有字段，并添加 reasoning_details 支持
    """
    # 基础字段
    role: str
    content: ChatMessageContent | None
    
    # OpenAI 标准字段
    refusal: NotRequired[str | None]
    annotations: NotRequired[List[Dict[str, Any]] | None]
    audio: NotRequired[Dict[str, Any] | None]
    function_call: NotRequired[Dict[str, Any] | None]
    tool_calls: NotRequired[List[Dict[str, Any]] | None]
    
    # 扩展字段（如 Google Gemini 的 reasoning_details）
    reasoning_details: NotRequired[List[ReasoningDetail]]


class ToolMessageParam(TypedDict, total=False):
    role: Literal["tool"]
    content: str
    tool_call_id: str


class FunctionMessageParam(TypedDict, total=False):
    role: Literal["function"]
    content: str
    name: str

# 使用 dict-like 消息类型作为主要消息类型。
# SDK 的 ChatCompletionMessage 只应该出现在边界输入/输出，进入框架内部后
# 应通过 message_to_dict() 归一化为这些字典结构。
MessageParam: TypeAlias = ExtendedMessageParam | ToolMessageParam | FunctionMessageParam

# 消息列表类型
MessageList: TypeAlias = List[MessageParam]

# Internal normalized dict-like message shape used after boundary coercion.
NormalizedMessageParam: TypeAlias = ExtendedMessageParam | ToolMessageParam | FunctionMessageParam
NormalizedMessageList: TypeAlias = List[NormalizedMessageParam]
