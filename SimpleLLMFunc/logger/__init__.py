"""
初始化全局日志系统单例并导出日志函数
"""

from .logger_config import logger_config

from .logger import (
    setup_logger,
    app_log,
    push_warning,
    push_error,
    push_critical,
    push_info,
    push_debug,
    get_location,
    LogLevel,
    get_logger,
    log_context,
    async_log_context,
    ConsoleFormatter,
    get_current_trace_id,
    get_current_context_attribute,
    set_current_context_attribute,
)


_log_level = logger_config.LOG_LEVEL.upper()


# 日志级别映射
_log_level_map = {
    "DEBUG": LogLevel.DEBUG,
    "INFO": LogLevel.INFO,
    "WARNING": LogLevel.WARNING,
    "ERROR": LogLevel.ERROR,
    "CRITICAL": LogLevel.CRITICAL,
}

# 将字符串级别转换为枚举
console_level = _log_level_map.get(_log_level, LogLevel.INFO)

# 初始化全局单例日志器
GLOBAL_LOGGER = setup_logger(
    console_level=console_level,
    use_color=True,
    logger_name="SimpleLLMFunc",
)

# 记录日志系统初始化完成
push_info(f"Global logging system initialized, console log level: {_log_level}")

# 确保启动时打印一条测试日志
push_debug("Test DEBUG level log")
app_log("Test INFO level log (app_log)")
push_info("Test INFO level log (push_info)")

__all__ = [
    "app_log",
    "push_warning",
    "push_error",
    "push_critical",
    "push_info",
    "push_debug",
    "get_location",
    "log_context",
    "async_log_context",
    "LogLevel",
    "get_logger",
    "setup_logger",
    "ConsoleFormatter",
    "get_current_trace_id",
    "get_current_context_attribute",
    "set_current_context_attribute",
]
