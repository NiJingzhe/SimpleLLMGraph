"""
日志上下文管理模块

本模块提供异步安全的日志上下文管理功能，使用contextvars实现。
支持嵌套上下文和跨协程的日志上下文传递。
"""

import contextvars
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Generator, AsyncGenerator, Optional
import threading

# 使用 contextvars 来管理日志上下文，支持异步和多线程环境
_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)
_context_lock = threading.RLock()  # 保留锁用于向后兼容，但主要逻辑会使用 contextvars

# 用于表示默认trace_id的常量
DEFAULT_TRACE_ID = ""


def _merge_context(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    合并当前上下文和额外参数

    Args:
        extra: 额外参数字典

    Returns:
        合并后的字典，包含当前上下文和额外参数

    Example:
        >>> current_context = _merge_context({"user_id": "123"})
        >>> # current_context 包含了当前上下文 + user_id
    """
    result = {}

    # 获取当前上下文
    current_context = _log_context.get({})
    result.update(current_context)

    # 添加额外参数（如果有）
    if extra:
        result.update(extra)

    return result


def get_current_trace_id() -> str:
    """
    获取当前上下文中的trace_id

    Returns:
        当前上下文中的trace_id，如果不存在则返回空字符串

    Example:
        >>> trace_id = get_current_trace_id()
        >>> if trace_id:
        ...     print(f"Current trace: {trace_id}")
    """
    current_context = _log_context.get({})
    return current_context.get("trace_id", DEFAULT_TRACE_ID)


def get_current_context_attribute(key: str) -> Any:
    """
    获取当前上下文中的指定属性值

    Args:
        key: 属性名称

    Returns:
        属性值，如果不存在则返回None

    Example:
        >>> user_id = get_current_context_attribute("user_id")
        >>> if user_id:
        ...     print(f"User: {user_id}")
    """
    current_context = _log_context.get({})
    return current_context.get(key, None)


def set_current_context_attribute(key: str, value: Any) -> None:
    """
    设置当前log上下文中某个属性的值

    Args:
        key: 属性名称
        value: 属性值

    Note:
        对于已知的系统属性不会产生警告，对于新的属性会产生警告提示

    Example:
        >>> set_current_context_attribute("user_id", "12345")
        >>> set_current_context_attribute("execution_time", 0.123)
    """
    current_context = _log_context.get({})

    # 系统已知的属性，不需要警告
    KNOWN_SYSTEM_ATTRIBUTES = {
        "input_tokens",
        "output_tokens",
        "trace_id",
        "location",
        "execution_time",
        "model_name",
        "function_name",
    }

    if key not in current_context and key not in KNOWN_SYSTEM_ATTRIBUTES:
        from .core import push_warning
        push_warning(
            f"You are changing a never seen attribute in current log context: {key}"
        )

    # 创建新的上下文字典
    new_context = current_context.copy()
    new_context[key] = value
    _log_context.set(new_context)


@asynccontextmanager
async def async_log_context(**kwargs: Any) -> AsyncGenerator[None, None]:
    """
    创建异步日志上下文，在上下文中的所有日志都会包含指定的字段

    可以通过提供一些参数来指定在一层上下文中统一的属性值，并会被自动添加到log中
    当context发生嵌套时，外层的属性并不会继承到内层，嵌套的上下文会以栈的形式被管理

    Args:
        **kwargs: 要添加到上下文的键值对

    Example:
        >>> async with async_log_context(trace_id="my_function_123", user_id="456"):
        ...     push_info("Processing user request")  # Logs automatically include trace_id and user_id

    Note:
        - 支持异步环境
        - 上下文是栈式的，后进先出
        - 支持GeneratorExit异常处理
    """
    # 获取当前上下文
    current_context = _log_context.get({})

    # 创建新的上下文，合并新的属性
    new_context = current_context.copy()
    new_context.update(kwargs)

    # 设置新的上下文
    token = _log_context.set(new_context)

    try:
        yield
    except GeneratorExit:
        # 处理异步生成器被提前关闭的情况
        # 直接重置上下文并重新抛出异常
        try:
            _log_context.reset(token)
        except (ValueError, RuntimeError):
            # 忽略上下文重置错误
            pass
        raise
    except Exception:
        # 处理其他异常
        try:
            _log_context.reset(token)
        except (ValueError, RuntimeError):
            # 忽略上下文重置错误
            pass
        raise
    else:
        # 正常完成时重置上下文
        try:
            _log_context.reset(token)
        except (ValueError, RuntimeError):
            # 忽略上下文重置错误
            pass


@contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """
    创建日志上下文，在上下文中的所有日志都会包含指定的字段

    可以通过提供一些参数来指定在一层上下文中统一的属性值，并会被自动添加到log中
    当context发生嵌套时，外层的属性并不会继承到内层，嵌套的上下文会以栈的形式被管理

    Args:
        **kwargs: 要添加到上下文的键值对

    Example:
        >>> with log_context(trace_id="my_function_123", user_id="456"):
        ...     push_info("Processing user request")  # Logs automatically include trace_id and user_id

    Note:
        - 支持同步环境
        - 上下文是栈式的，后进先出
        - 异常安全的上下文管理
    """
    # 获取当前上下文
    current_context = _log_context.get({})

    # 创建新的上下文，合并新的属性
    new_context = current_context.copy()
    new_context.update(kwargs)

    # 设置新的上下文
    token = _log_context.set(new_context)

    try:
        yield
    finally:
        # 恢复原始上下文
        try:
            _log_context.reset(token)
        except ValueError:
            # 在某些边缘情况下，Context 可能在不同的任务中被重置
            # 这种情况下忽略 ValueError 是安全的
            pass
