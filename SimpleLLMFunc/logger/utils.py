"""
日志系统工具函数模块

本模块包含了日志系统使用的各种工具函数，包括位置获取、时间转换等。
"""

import inspect
import os
from typing import Any


def _is_logger_module_frame(frame) -> bool:
    """
    检查给定的帧是否属于 SimpleLLMFunc.logger 模块

    使用严格的检测条件，确保只检测 SimpleLLMFunc.logger 模块，
    不会误判其他框架的 logger 模块。

    Args:
        frame: 要检查的帧对象

    Returns:
        如果帧属于 SimpleLLMFunc.logger 模块返回 True，否则返回 False
    """
    if frame is None:
        return False
    
    try:
        # 方法1: 优先使用模块名检测（最可靠）
        # 检查模块名是否以 SimpleLLMFunc.logger 开头
        try:
            module = inspect.getmodule(frame)
            if module and hasattr(module, '__name__'):
                module_name = module.__name__
                if module_name.startswith('SimpleLLMFunc.logger'):
                    return True
        except (AttributeError, TypeError):
            # 如果无法获取模块信息，继续使用路径检测
            pass
        
        # 方法2: 检查文件路径中是否包含 SimpleLLMFunc/logger/ 目录
        # 使用更严格的路径匹配，确保只匹配 SimpleLLMFunc 包的 logger 模块
        frame_info = inspect.getframeinfo(frame)
        filepath = frame_info.filename
        
        # 标准化路径分隔符
        normalized_path = filepath.replace('\\', '/')
        
        # 严格检查：路径中必须包含 SimpleLLMFunc/logger/ 或 SimpleLLMFunc\logger\
        # 这样可以避免误判其他项目的 logger 模块
        if '/SimpleLLMFunc/logger/' in normalized_path:
            return True
        
        return False
    except Exception:
        # 如果检查过程中出现任何错误，保守地返回 False
        return False


def get_location(depth: int = 2) -> str:
    """
    获取调用者的代码位置信息

    此函数通过检查调用栈来获取调用者的位置信息，用于在日志中标识代码位置。
    当在 logger 模块内部调用时，会自动跳过 logger 模块内的调用，直到找到用户代码。

    Args:
        depth: 调用栈深度，默认为2（调用者的调用者）
              当在 logger 模块内部调用时，会自动跳过 logger 模块内的调用

    Returns:
        格式化的位置字符串，如 "module.py:function:42"

    Example:
        >>> location = get_location()
        >>> print(location)  # "main.py:main:10"

    Note:
        - depth=1: 当前函数
        - depth=2: 调用当前函数的函数（默认）
        - 如果无法获取位置信息，返回"unknown"
        - 当在 logger 模块内部调用时，会自动跳过 logger 模块内的调用栈
    """
    frame = inspect.currentframe()
    try:
        # 向上追溯调用栈
        for _ in range(depth):
            if frame is None:
                break
            frame = frame.f_back

        # 如果是在 logger 模块内部调用（比如从 _log_message 调用），
        # 需要继续向上追溯，跳过所有 logger 模块内的调用
        if frame:
            # 如果当前帧还在 logger 模块内，继续向上追溯
            while frame and _is_logger_module_frame(frame):
                frame = frame.f_back
            
            if frame:
                frame_info = inspect.getframeinfo(frame)
                filename = os.path.basename(frame_info.filename)
                return f"{filename}:{frame_info.function}:{frame_info.lineno}"
            else:
                return "unknown"
        else:
            return "unknown"
    finally:
        # 删除引用，避免循环引用
        del frame

