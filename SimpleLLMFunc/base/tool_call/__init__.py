"""Tool call extraction and execution helpers."""

from SimpleLLMFunc.base.tool_call.execution import (
    ExecutedToolCallResult,
    execute_single_tool_call_result,
)
from SimpleLLMFunc.base.tool_call.extraction import (
    AccumulatedToolCall,
    ToolCallFunctionInfo,
    accumulate_tool_calls_from_chunks,
    extract_reasoning_details,
    extract_reasoning_details_from_stream,
    parse_tool_call_arguments,
    repair_tool_call_arguments,
    extract_tool_calls,
    extract_tool_calls_from_stream_response,
)
from SimpleLLMFunc.base.tool_call.streaming import (
    StreamingToolCallState,
    collect_stream_argument_deltas,
    collect_tool_argument_delta_payloads,
    stringify_argument_value,
)

# 从统一类型系统导入 ReasoningDetail（向后兼容）
from SimpleLLMFunc.type.message import ReasoningDetail
from SimpleLLMFunc.base.tool_call.validation import (
    is_valid_tool_result,
    serialize_tool_output_for_langfuse,
)

__all__ = [
    "serialize_tool_output_for_langfuse",
    "is_valid_tool_result",
    "ExecutedToolCallResult",
    "execute_single_tool_call_result",
    "extract_tool_calls",
    "accumulate_tool_calls_from_chunks",
    "parse_tool_call_arguments",
    "repair_tool_call_arguments",
    "extract_tool_calls_from_stream_response",
    "extract_reasoning_details",
    "extract_reasoning_details_from_stream",
    "StreamingToolCallState",
    "collect_stream_argument_deltas",
    "collect_tool_argument_delta_payloads",
    "stringify_argument_value",
    "ToolCallFunctionInfo",
    "AccumulatedToolCall",
    "ReasoningDetail",
]
