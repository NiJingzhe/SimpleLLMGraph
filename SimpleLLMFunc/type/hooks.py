"""Hook 系统相关的类型定义

为 ReAct Hook 系统提供完整的类型支持。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict, NotRequired, TypeAlias
from datetime import datetime
from enum import Enum

# 复用框架内类型
from SimpleLLMFunc.type.message import MessageList, MessageParam, NormalizedMessageList
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text
from SimpleLLMFunc.type.tool_call import (
    ToolCallArguments,
)

# ============================================================================
# Hook 上下文类型
# ============================================================================

class ReActPhase(str, Enum):
    """
    ReAct 循环的执行阶段枚举
    
    提供了细粒度的阶段标识，覆盖整个 ReAct 循环的所有关键步骤。
    """
    # ========================================================================
    # 循环级别阶段
    # ========================================================================
    REACT_LOOP_START = "react_loop_start"
    """ReAct 循环开始"""
    
    REACT_LOOP_COMPLETE = "react_loop_complete"
    """ReAct 循环完成"""
    
    # ========================================================================
    # 初始 LLM 调用阶段
    # ========================================================================
    INITIAL_LLM_CALL_START = "initial_llm_call_start"
    """初始 LLM 调用开始"""
    
    INITIAL_LLM_CALL_COMPLETE = "initial_llm_call_complete"
    """初始 LLM 调用完成"""
    
    # ========================================================================
    # 工具调用阶段
    # ========================================================================
    TOOL_CALLS_BATCH_START = "tool_calls_batch_start"
    """工具调用批次开始（一批工具调用开始执行）"""
    
    TOOL_CALL_START = "tool_call_start"
    """单个工具调用开始"""
    
    TOOL_CALL_COMPLETE = "tool_call_complete"
    """单个工具调用完成"""
    
    TOOL_CALLS_BATCH_COMPLETE = "tool_calls_batch_complete"
    """工具调用批次完成（所有工具调用执行完毕）"""
    
    # ========================================================================
    # 迭代阶段
    # ========================================================================
    ITERATION_START = "iteration_start"
    """迭代开始（第 N 次迭代）"""
    
    ITERATION_LLM_CALL_START = "iteration_llm_call_start"
    """迭代中的 LLM 调用开始"""
    
    ITERATION_LLM_CALL_COMPLETE = "iteration_llm_call_complete"
    """迭代中的 LLM 调用完成"""
    
    ITERATION_COMPLETE = "iteration_complete"
    """迭代完成"""
    
    # ========================================================================
    # 最终阶段
    # ========================================================================
    FINAL_LLM_CALL_START = "final_llm_call_start"
    """最终 LLM 调用开始（达到 max_tool_calls 时）"""
    
    FINAL_LLM_CALL_COMPLETE = "final_llm_call_complete"
    """最终 LLM 调用完成"""
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def is_llm_call_phase(self) -> bool:
        """判断是否为 LLM 调用阶段"""
        return self in (
            ReActPhase.INITIAL_LLM_CALL_START,
            ReActPhase.INITIAL_LLM_CALL_COMPLETE,
            ReActPhase.ITERATION_LLM_CALL_START,
            ReActPhase.ITERATION_LLM_CALL_COMPLETE,
            ReActPhase.FINAL_LLM_CALL_START,
            ReActPhase.FINAL_LLM_CALL_COMPLETE,
        )
    
    def is_tool_call_phase(self) -> bool:
        """判断是否为工具调用阶段"""
        return self in (
            ReActPhase.TOOL_CALLS_BATCH_START,
            ReActPhase.TOOL_CALL_START,
            ReActPhase.TOOL_CALL_COMPLETE,
            ReActPhase.TOOL_CALLS_BATCH_COMPLETE,
        )
    
    def is_iteration_phase(self) -> bool:
        """判断是否为迭代阶段"""
        return self in (
            ReActPhase.ITERATION_START,
            ReActPhase.ITERATION_LLM_CALL_START,
            ReActPhase.ITERATION_LLM_CALL_COMPLETE,
            ReActPhase.ITERATION_COMPLETE,
        )

class HookContext(TypedDict):
    """Hook 调用的上下文信息"""
    # 基本信息
    func_name: str
    trace_id: str
    
    # 执行状态
    iteration: int  # 当前迭代次数（从 0 开始）
    phase: ReActPhase
    
    # 时间信息
    start_time: datetime
    current_time: datetime
    
    # 扩展信息（可选）
    extra: NotRequired[Dict[str, Any]]

# ============================================================================
# 工具结果类型
# ============================================================================

ToolResult = Union[
    str,                      # 文本结果
    Dict[str, Any],           # JSON 对象
    List[Any],                # 数组
    ImgPath,                  # 本地图片
    ImgUrl,                   # 图片 URL
    Tuple[str, ImgPath],      # 文本 + 本地图片
    Tuple[str, ImgUrl],       # 文本 + 图片 URL
    Text,                     # 文本包装类
]
"""
工具函数的返回值类型

支持的类型：
- 基本类型：str, dict, list
- 多模态类型：ImgPath, ImgUrl, Text
- 组合类型：(str, ImgPath), (str, ImgUrl)
"""

# ============================================================================
# 工具调用事件类型
# ============================================================================

class ToolCallEvent(TypedDict):
    """单个工具调用的完整事件信息"""
    tool_name: str
    tool_call_id: str
    arguments: ToolCallArguments
    result: ToolResult
    execution_time: float  # 执行耗时（秒）
    error: NotRequired[Optional[Exception]]
    context: HookContext

ToolCallEventList = List[ToolCallEvent]
"""工具调用事件列表"""

# ============================================================================
# 消息类型别名（复用现有类型）
# ============================================================================

# 直接使用已定义的类型
Message: TypeAlias = MessageParam
"""消息类型（复用 MessageParam）"""

Messages: TypeAlias = MessageList
"""消息列表类型（复用 MessageList）"""

# 历史消息类型（统一使用归一化后的 dict-like 消息列表）
HistoryList: TypeAlias = NormalizedMessageList
"""
历史消息列表类型

注意：这是为了向后兼容而保留的别名，实际类型与 MessageList 相同。
"""


__all__ = [
    # Hook 上下文
    "ReActPhase",
    "HookContext",
    # 工具结果
    "ToolResult",
    # 工具调用事件
    "ToolCallEvent",
    "ToolCallEventList",
    # 消息类型别名
    "Message",
    "Messages",
    "HistoryList",
]
