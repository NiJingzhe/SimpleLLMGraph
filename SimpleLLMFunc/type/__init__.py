# ============================================================================
# 消息类型
# ============================================================================
from SimpleLLMFunc.type.message import (
    ChatContentPart,
    ChatImageDetail,
    ChatImageUrl,
    ChatImageUrlContentPart,
    ChatMessageContent,
    ChatTextContentPart,
    MessageList,
    MessageParam,
    NormalizedMessageList,
    NormalizedMessageParam,
    ReasoningDetail,
    ExtendedMessageParam,
)

# ============================================================================
# 多模态类型
# ============================================================================
from SimpleLLMFunc.type.multimodal import (
    ImgPath,
    ImgUrl,
    Text,
    MultimodalContent,
    MultimodalList,
)
from SimpleLLMFunc.type.chat_input import (
    UserChatContent,
    UserChatContentInput,
    UserChatContentPart,
    UserChatMessage,
    normalize_user_chat_content,
    normalize_user_chat_message,
)

# ============================================================================
# 工具调用类型
# ============================================================================
from SimpleLLMFunc.type.tool_call import (
    ToolCall,
    ToolCallFunction,
    ToolCallArguments,
    ToolDefinition,
    ToolFunctionDefinition,
    ToolDefinitionList,
    ToolCallFunctionInfo,
    AccumulatedToolCall,
    dict_to_tool_call,
    tool_call_to_dict,
)

# ============================================================================
# LLM 响应类型
# ============================================================================
from SimpleLLMFunc.type.llm import (
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
)

# ============================================================================
# Hook 系统类型
# ============================================================================
from SimpleLLMFunc.type.hooks import (
    HookContext,
    ReActPhase,
    ToolResult,
    ToolCallEvent,
    ToolCallEventList,
    Message,
    Messages,
    HistoryList,  # 向后兼容
)

# ============================================================================
# 接口类型
# ============================================================================
from SimpleLLMFunc.interface.llm_interface import LLM_Interface

__all__ = [
    # 消息类型
    "MessageParam",
    "MessageList",
    "NormalizedMessageParam",
    "NormalizedMessageList",
    "ChatContentPart",
    "ChatImageDetail",
    "ChatImageUrl",
    "ChatImageUrlContentPart",
    "ChatMessageContent",
    "ChatTextContentPart",
    "ReasoningDetail",
    "ExtendedMessageParam",
    # 多模态类型
    "Text",
    "ImgUrl",
    "ImgPath",
    "MultimodalContent",
    "MultimodalList",
    "UserChatContent",
    "UserChatContentInput",
    "UserChatContentPart",
    "UserChatMessage",
    "normalize_user_chat_content",
    "normalize_user_chat_message",
    # 工具调用类型
    "ToolCall",
    "ToolCallFunction",
    "ToolCallArguments",
    "ToolDefinition",
    "ToolFunctionDefinition",
    "ToolDefinitionList",
    "ToolCallFunctionInfo",
    "AccumulatedToolCall",
    "dict_to_tool_call",
    "tool_call_to_dict",
    # LLM 响应类型
    "LLMResponse",
    "LLMStreamChunk",
    "LLMUsage",
    # Hook 系统类型
    "HookContext",
    "ReActPhase",
    "ToolResult",
    "ToolCallEvent",
    "ToolCallEventList",
    "Message",
    "Messages",
    "HistoryList",
    # 接口类型
    "LLM_Interface",
]
