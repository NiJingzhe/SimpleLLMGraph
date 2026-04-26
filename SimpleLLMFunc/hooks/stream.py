"""ReAct Event Stream 实现

提供 Tagged Union 类型定义和辅助工具函数，用于处理事件流。
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Literal,
    Set,
    TypeGuard,
    Union,
)

from SimpleLLMFunc.hooks.events import ReActEvent, ReActEventType
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.type.llm import LLMResponse, LLMStreamChunk


# ============================================================================
# Tagged Union 类型定义
# ============================================================================

from dataclasses import asdict, dataclass, field


@dataclass
class EventOrigin:
    """Normalized origin metadata for one emitted event."""

    session_id: str
    agent_call_id: str
    event_seq: int
    parent_agent_call_id: str | None = None
    fork_id: str | None = None
    fork_depth: int = 0
    fork_seq: int | None = None
    selfref_instance_id: str | None = None
    source_memory_key: str | None = None
    memory_key: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_origin() -> EventOrigin:
    return EventOrigin(
        session_id="legacy",
        agent_call_id="legacy",
        event_seq=0,
        fork_depth=0,
    )


@dataclass
class ResponseYield:
    """Response item emitted by the event stream."""

    response: Union[LLMResponse, LLMStreamChunk, str]  # 根据 return_mode 决定
    messages: MessageList
    type: Literal["response"] = "response"  # 放在最后，因为有默认值


@dataclass
class EventYield:
    """Event item emitted by the event stream."""

    event: ReActEvent
    origin: EventOrigin = field(default_factory=_default_origin)
    type: Literal["event"] = "event"  # 放在最后，因为有默认值


# 联合类型
ReactOutput = Union[ResponseYield, EventYield]
"""ReAct 输出类型：响应或事件"""


# ============================================================================
# 类型守卫函数
# ============================================================================


def is_response_yield(output: ReactOutput) -> TypeGuard[ResponseYield]:
    """类型守卫：判断是否为响应 yield

    Args:
        output: ReactOutput 对象

    Returns:
        如果是 ResponseYield，返回 True
    """
    return output.type == "response"


def is_event_yield(output: ReactOutput) -> TypeGuard[EventYield]:
    """类型守卫：判断是否为事件 yield

    Args:
        output: ReactOutput 对象

    Returns:
        如果是 EventYield，返回 True
    """
    return output.type == "event"


# ============================================================================
# 过滤器函数
# ============================================================================


async def responses_only(
    generator: AsyncGenerator[ReactOutput, None],
) -> AsyncGenerator[tuple[Any, MessageList], None]:
    """只保留响应，忽略事件

    用于向后兼容：将 ReactOutput 转换为 (response, messages) 元组。

    Args:
        generator: 产生 ReactOutput 的异步生成器

    Yields:
        (response, messages) 元组，与现有 API 兼容
    """
    async for output in generator:
        if is_response_yield(output):
            yield output.response, output.messages


async def events_only(
    generator: AsyncGenerator[ReactOutput, None],
) -> AsyncGenerator[ReActEvent, None]:
    """只保留事件，忽略响应

    Args:
        generator: 产生 ReactOutput 的异步生成器

    Yields:
        ReActEvent 对象
    """
    async for output in generator:
        if is_event_yield(output):
            yield output.event


async def filter_events(
    generator: AsyncGenerator[ReactOutput, None],
    event_types: Set[ReActEventType],
) -> AsyncGenerator[ReActEvent, None]:
    """过滤特定类型的事件

    Args:
        generator: 产生 ReactOutput 的异步生成器
        event_types: 要保留的事件类型集合

    Yields:
        匹配的事件对象
    """
    async for output in generator:
        if is_event_yield(output) and output.event.event_type in event_types:
            yield output.event


# ============================================================================
# 装饰器函数
# ============================================================================


def with_event_observer(
    observer: Callable[[ReActEvent], Awaitable[None]],
):
    """装饰器：为 generator 添加事件观测器

    使用示例：
        @with_event_observer(my_observer)
        @llm_chat(llm_interface=my_llm)
        async def my_chat(message: str):
            pass

    Args:
        observer: 异步事件处理函数

    Returns:
        装饰器函数
    """

    def decorator(
        generator_func: Callable[..., AsyncGenerator[ReactOutput, None]],
    ) -> Callable[..., AsyncGenerator[ReactOutput, None]]:
        async def wrapper(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[ReactOutput, None]:
            async for output in generator_func(*args, **kwargs):
                if is_event_yield(output):
                    try:
                        await observer(output.event)
                    except Exception:
                        # 事件处理失败不应影响主流程
                        pass
                yield output

        return wrapper

    return decorator


__all__ = [
    # Tagged Union 类型
    "ResponseYield",
    "EventOrigin",
    "EventYield",
    "ReactOutput",
    # 类型守卫
    "is_response_yield",
    "is_event_yield",
    # 过滤器
    "responses_only",
    "events_only",
    "filter_events",
    # 装饰器
    "with_event_observer",
]
