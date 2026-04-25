"""ReAct 事件类型定义

定义 ReAct 循环中的所有事件类型，使用类型系统确保类型安全。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, NotRequired

# 导入类型系统
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.type.tool_call import (
    ToolCall,
    ToolCallArguments,
    ToolDefinitionList,
)
from SimpleLLMFunc.type.llm import (
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
)
from SimpleLLMFunc.type.hooks import ToolResult


# ============================================================================
# 事件类型枚举
# ============================================================================


class ReActEventType(str, Enum):
    """事件类型枚举"""

    REACT_START = "react_start"
    REACT_ITERATION_START = "react_iteration_start"
    LLM_CALL_START = "llm_call_start"
    LLM_CHUNK_ARRIVE = "llm_chunk_arrive"
    LLM_CALL_END = "llm_call_end"
    LLM_CALL_ERROR = "llm_call_error"
    TOOL_CALLS_BATCH_START = "tool_calls_batch_start"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_ARGUMENTS_DELTA = "tool_call_arguments_delta"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"
    TOOL_CALLS_BATCH_END = "tool_calls_batch_end"
    REACT_ITERATION_END = "react_iteration_end"
    REACT_END = "react_end"
    CUSTOM_EVENT = "custom_event"


# ============================================================================
# 基础事件类
# ============================================================================


@dataclass(kw_only=True)
class ReActEvent:
    """ReAct 事件基类

    所有事件都继承自此类，包含通用字段。
    """

    # 事件基本信息
    event_type: ReActEventType  # 事件类型
    timestamp: datetime  # 事件发生时间

    # 执行上下文
    trace_id: str  # 追踪 ID
    func_name: str  # 函数名称
    iteration: int  # 当前迭代次数（从 0 开始，0 表示初始调用）
    extra: Dict[str, Any] = field(default_factory=dict, kw_only=True)


# ============================================================================
# 工具调用结果类型（用于批次结束事件）
# ============================================================================


class ToolCallResult(TypedDict):
    """单个工具调用结果"""

    tool_name: str
    tool_call_id: str
    result: ToolResult
    execution_time: float
    success: bool
    error: NotRequired[Optional[Exception]]  # 仅在失败时存在


# ============================================================================
# 具体事件类型
# ============================================================================


@dataclass
class ReactStartEvent(ReActEvent):
    """ReAct 循环开始事件

    触发时机：ReAct 循环开始时，包括用户任务提示
    """

    # 用户输入
    user_task_prompt: str  # 用户任务提示（函数调用时的输入）
    initial_messages: MessageList  # 初始消息列表
    available_tools: ToolDefinitionList  # 可用工具列表
    # extra 字段由 __init_subclass__ 动态添加


@dataclass
class ReactIterationStartEvent(ReActEvent):
    """ReAct 迭代开始事件

    触发时机：每次 ReAct 迭代开始时（工具调用批次 + 后续 LLM 调用的开始）
    """

    # 迭代信息
    current_messages: MessageList  # 当前消息历史


@dataclass
class LLMCallStartEvent(ReActEvent):
    """LLM 调用开始事件

    触发时机：LLM 调用开始前，包括调用参数和消息列表
    """

    # 调用参数
    messages: MessageList  # 消息列表
    tools: ToolDefinitionList  # 工具定义列表
    llm_kwargs: Dict[str, Any]  # LLM 调用参数（temperature, max_tokens 等）
    stream: bool  # 是否流式调用


@dataclass
class LLMChunkArriveEvent(ReActEvent):
    """LLM chunk 到达事件（仅 streaming）

    触发时机：流式调用时，每个 chunk 到达时（仅在 streaming 模式下触发）
    """

    # Chunk 数据
    chunk: LLMStreamChunk  # LLM 返回的 chunk 对象
    accumulated_content: str  # 累积的内容（从开始到当前 chunk）
    chunk_index: int  # Chunk 序号（从 0 开始）


@dataclass
class LLMCallEndEvent(ReActEvent):
    """LLM 调用结束事件

    触发时机：LLM 调用完成后，包括包含 assistant 消息的消息列表
    """

    # 响应数据
    response: LLMResponse  # LLM 响应对象
    messages: MessageList  # 更新后的消息列表（包含 assistant 消息）
    tool_calls: List[ToolCall]  # 提取的工具调用列表
    execution_time: float  # 执行耗时（秒）
    content: str = ""  # 聚合后的 assistant 文本内容
    reasoning_details: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[LLMUsage] = None  # Token 使用统计（如果可用，放在最后）


@dataclass
class LLMCallErrorEvent(ReActEvent):
    """LLM 调用错误事件

    触发时机：LLM 调用发生错误时
    """

    # 错误信息
    error: Exception  # 异常对象
    error_message: str  # 错误消息
    error_type: str  # 错误类型
    messages: MessageList  # 调用时的消息列表
    llm_kwargs: Dict[str, Any]  # 调用时的参数


@dataclass
class ToolCallsBatchStartEvent(ReActEvent):
    """工具调用批次开始事件

    触发时机：工具调用批次开始前（当 LLM 返回多个工具调用时）
    """

    # 批次信息
    tool_calls: List[ToolCall]  # 工具调用列表
    batch_size: int  # 批次大小


@dataclass
class ToolCallStartEvent(ReActEvent):
    """工具调用开始事件

    触发时机：单个工具调用开始前，包括工具调用映射
    """

    # 工具调用信息
    tool_name: str  # 工具名称
    tool_call_id: str  # 工具调用 ID
    arguments: ToolCallArguments  # 工具调用参数
    tool_call: ToolCall  # 完整的工具调用对象


@dataclass
class ToolCallArgumentsDeltaEvent(ReActEvent):
    """工具调用参数增量事件（仅 streaming）

    触发时机：流式响应接收 tool_call 参数片段时，实时解析并输出单参数增量。
    """

    tool_name: str  # 工具名称
    tool_call_id: str  # 工具调用 ID
    argname: str  # 参数名
    argcontent_delta: str  # 参数内容增量


@dataclass
class ToolCallEndEvent(ReActEvent):
    """工具调用结束事件

    触发时机：单个工具调用完成后，包括工具调用结果

    关键：立即获取工具调用结果（无需等待下一轮 LLM 调用）
    """

    # 工具调用信息
    tool_name: str  # 工具名称
    tool_call_id: str  # 工具调用 ID
    arguments: ToolCallArguments  # 工具调用参数

    # 执行结果
    result: ToolResult  # 工具执行结果
    execution_time: float  # 执行耗时（秒）
    success: bool  # 是否成功执行


@dataclass
class ToolCallErrorEvent(ReActEvent):
    """工具调用错误事件

    触发时机：工具调用发生错误时
    """

    # 工具调用信息
    tool_name: str  # 工具名称
    tool_call_id: str  # 工具调用 ID
    arguments: ToolCallArguments  # 工具调用参数

    # 错误信息
    error: Exception  # 异常对象
    error_message: str  # 错误消息
    error_type: str  # 错误类型
    execution_time: float = 0.0  # 执行耗时（秒，如果已开始执行，放在最后）


@dataclass
class ToolCallsBatchEndEvent(ReActEvent):
    """工具调用批次结束事件

    触发时机：工具调用批次全部完成后
    """

    # 批次结果
    tool_results: List[ToolCallResult]  # 所有工具调用结果列表
    batch_size: int  # 批次大小
    total_execution_time: float  # 总执行时间（秒）
    success_count: int  # 成功数量
    error_count: int  # 失败数量


@dataclass
class ReactIterationEndEvent(ReActEvent):
    """ReAct 迭代结束事件

    触发时机：每次 ReAct 迭代完成时
    """

    # 迭代信息
    messages: MessageList  # 更新后的消息历史（包含工具调用结果）
    iteration_time: float  # 迭代耗时（秒）
    tool_calls_count: int  # 本次迭代的工具调用数量


@dataclass
class ReactEndEvent(ReActEvent):
    """ReAct 循环结束事件

    触发时机：ReAct 循环结束时，包括最终结果内容
    """

    # 最终结果
    final_response: str  # 最终响应内容（文本）
    final_messages: MessageList  # 完整的消息历史
    total_iterations: int  # 总迭代次数
    total_execution_time: float  # 总执行时间（秒）
    total_tool_calls: int  # 总工具调用次数
    total_llm_calls: int  # 总 LLM 调用次数
    total_token_usage: Optional[LLMUsage] = None  # 总 token 使用统计（如果可用）


@dataclass
class CustomEvent(ReActEvent):
    """自定义事件

    触发时机：由用户代码主动发射，用于在 tool 内部报告进度、状态等

    使用方式：
        # 在 tool 函数中获取 event_emitter 并发射事件
        await event_emitter.emit("my_event", {"progress": 50})
    """

    # 自定义事件信息
    event_name: str  # 事件名称，由用户定义
    data: Any = None  # 事件数据，任意类型
    tool_name: Optional[str] = None  # 所属工具名称（如果可用）
    tool_call_id: Optional[str] = None  # 所属工具调用 ID（如果可用）


__all__ = [
    # 枚举
    "ReActEventType",
    # 基础类
    "ReActEvent",
    # 工具调用结果类型
    "ToolCallResult",
    # 具体事件类型
    "ReactStartEvent",
    "ReactIterationStartEvent",
    "LLMCallStartEvent",
    "LLMChunkArriveEvent",
    "LLMCallEndEvent",
    "LLMCallErrorEvent",
    "ToolCallsBatchStartEvent",
    "ToolCallStartEvent",
    "ToolCallArgumentsDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallErrorEvent",
    "ToolCallsBatchEndEvent",
    "ReactIterationEndEvent",
    "ReactEndEvent",
    "CustomEvent",
]
