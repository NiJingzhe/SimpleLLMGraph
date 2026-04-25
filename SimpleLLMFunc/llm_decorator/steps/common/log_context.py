"""Step 2: Setup log context."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Dict, AsyncGenerator, Optional
from typing import cast
from langfuse.types import TraceContext

from SimpleLLMFunc.observability.langfuse_client import (
    get_langfuse_trace_context,
    langfuse_client,
    reset_langfuse_trace_context,
    set_langfuse_trace_context,
)

from SimpleLLMFunc.logger import app_log, async_log_context
from SimpleLLMFunc.logger.logger import get_location


def log_function_call(func_name: str, arguments: Dict[str, Any]) -> None:
    """记录函数调用日志"""
    args_str = json.dumps(arguments, default=str, ensure_ascii=False, indent=4)
    app_log(
        f"Async LLM function '{func_name}' called with arguments: {args_str}",
        location=get_location(),
    )


def create_log_context_manager(
    func_name: str, trace_id: str
) -> AsyncContextManager[None]:
    """创建日志上下文管理器"""
    return async_log_context(
        trace_id=trace_id,
        function_name=func_name,
        input_tokens=0,
        output_tokens=0,
    )


@asynccontextmanager
async def _log_context_manager(
    base_manager: AsyncContextManager[None],
) -> AsyncGenerator[None, None]:
    trace_context = get_langfuse_trace_context()
    trace_token: Optional[object] = None
    if trace_context is None:
        trace_context = cast(
            TraceContext,
            {"trace_id": langfuse_client.create_trace_id()},
        )
        trace_token = set_langfuse_trace_context(trace_context)

    async with base_manager:
        try:
            yield
        finally:
            if trace_token is not None:
                reset_langfuse_trace_context(trace_token)


def setup_log_context(
    func_name: str,
    trace_id: str,
    arguments: Dict[str, Any],
) -> AsyncContextManager[None]:
    """设置日志上下文的完整流程"""
    log_function_call(func_name, arguments)
    base_manager = create_log_context_manager(func_name, trace_id)
    return _log_context_manager(base_manager)
