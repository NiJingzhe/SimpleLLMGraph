from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional, Union

from SimpleLLMFunc.tool import Tool

ToolkitList = List[Union[Tool, Callable[..., Awaitable[Any]]]]

DEFAULT_MAX_TOOL_CALLS: Optional[int] = None

AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR = "__simplellmfunc_accepts_template_params__"
AGENT_FORK_TOOLKIT_FACTORY_ATTR = "__simplellmfunc_fork_toolkit_factory__"
FORK_CLONED_PYREPL_ATTR = "__simplellmfunc_fork_cloned_pyrepl__"

__all__ = [
    "AGENT_FORK_TOOLKIT_FACTORY_ATTR",
    "AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR",
    "DEFAULT_MAX_TOOL_CALLS",
    "FORK_CLONED_PYREPL_ATTR",
    "ToolkitList",
]
