"""
工具处理公用模块

此模块提供统一的工具解析和处理逻辑，供 llm_chat_decorator 和 llm_function_decorator 使用。
"""

import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from SimpleLLMFunc.logger import get_location, push_debug, push_warning
from SimpleLLMFunc.tool import Tool


TOOL_PROMPT_BLOCK_START = "<tool_best_practices>"
TOOL_PROMPT_BLOCK_END = "</tool_best_practices>"


def _normalize_prompt_text_for_dedupe(text: str) -> str:
    collapsed = " ".join(text.replace("`", "").split())
    normalized = collapsed.strip().lower()
    return normalized.rstrip(".")


def process_tools(
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]] = None,
    func_name: str = "unknown_function",
) -> Tuple[
    Optional[List[Dict[str, Any]]],
    Dict[str, Callable[..., Awaitable[Any]]],
]:
    """
    处理工具列表，返回 API 所需的工具参数和工具映射。

    此函数是 llm_chat_decorator 和 llm_function_decorator 中工具处理逻辑的统一实现。
    统一将所有 @tool 装饰的函数映射到 tool_obj.run 方法。

    ## 工具类型支持
    - `Tool` 对象：直接使用，要求 `run` 方法为 `async` 函数
    - `@tool` 装饰的异步函数：会被转换为相应的工具映射

    ## 处理流程
    1. 验证输入工具列表
    2. 遍历工具，区分 Tool 对象和 @tool 装饰的函数
    3. 检查类型合法性（必须为 async）
    4. 构建工具对象列表和工具名称到函数的映射
    5. 序列化工具以供 LLM API 使用
    6. 返回序列化的工具参数和工具映射字典

    Args:
        toolkit: 工具列表，可以是 Tool 对象或被 @tool 装饰的异步函数，为 None 或空列表时返回 (None, {})
        func_name: 函数名称，用于日志记录和错误信息

    Returns:
        (tool_param_for_api, tool_map) 元组：
            - tool_param_for_api: 序列化后的工具参数列表，供 LLM API 使用，如果无工具则为 None
            - tool_map: 工具名称到异步函数的映射字典，用于工具调用时的查找

    Raises:
        TypeError: 当工具的 run 方法或被装饰的函数不是 async 时抛出

    Examples:
        ```python
        from SimpleLLMFunc.tool import Tool

        # 示例1：使用 Tool 对象
        my_tool = Tool(name="get_weather", ...)
        tool_param, tool_map = process_tools([my_tool], "my_func")

        # 示例2：使用 @tool 装饰的函数
        @tool(name="calculate")
        async def calculate(a: int, b: int) -> int:
            return a + b

        tool_param, tool_map = process_tools([calculate], "my_func")

        # 示例3：混合使用
        tools = [my_tool, calculate]
        tool_param, tool_map = process_tools(tools, "my_func")
        ```

    Note:
        - 所有工具的 run 方法或函数本体必须是异步的（async）
        - 工具名称通过 Tool.name 属性获取
        - 序列化使用 Tool.serialize_tools() 方法
        - 所有 @tool 装饰的函数都统一映射到其 tool_obj.run 方法
    """
    if not toolkit:
        return None, {}

    tool_objects: List[Union[Tool, Callable[..., Awaitable[Any]]]] = []
    tool_map: Dict[str, Callable[..., Awaitable[Any]]] = {}

    for tool in toolkit:
        if isinstance(tool, Tool):
            # 处理 Tool 对象
            _process_tool_object(tool, func_name, tool_objects, tool_map)
        elif callable(tool) and hasattr(tool, "_tool"):
            # 处理 @tool 装饰的函数
            _process_decorated_function(tool, func_name, tool_objects, tool_map)
        else:
            push_warning(
                f"LLM function '{func_name}': unsupported tool type {type(tool)}; "
                "tools must be Tool instances or functions decorated with @tool",
                location=get_location(),
            )

    # 序列化工具以供 LLM API 使用
    tool_param_for_api: Optional[List[Dict[str, Any]]] = (
        Tool.serialize_tools(tool_objects) if tool_objects else None
    )

    push_debug(
        f"LLM function '{func_name}' loaded {len(tool_objects)} tools",
        location=get_location(),
    )

    return tool_param_for_api, tool_map


def collect_tool_prompt_specs(
    toolkit: Optional[List[Union[Tool, Callable[..., Awaitable[Any]]]]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Collect deduplicated ToolSpec metadata for system-prompt injection."""

    if not toolkit:
        return []

    collected: Dict[str, Dict[str, Any]] = {}

    for item in toolkit:
        tool_obj: Optional[Tool] = None
        if isinstance(item, Tool):
            tool_obj = item
        elif callable(item) and hasattr(item, "_tool"):
            maybe_tool_obj = getattr(item, "_tool", None)
            if isinstance(maybe_tool_obj, Tool):
                tool_obj = maybe_tool_obj

        if tool_obj is None:
            continue

        raw_spec = tool_obj.tool_spec
        raw_name = raw_spec.get("name")
        if not isinstance(raw_name, str):
            continue

        name = raw_name.strip()
        if not name:
            continue

        raw_description = raw_spec.get("description")
        description = (
            raw_description.strip() if isinstance(raw_description, str) else ""
        )

        raw_best_practices = raw_spec.get("best_practices")
        normalized_best_practices: List[str] = []
        if isinstance(raw_best_practices, list):
            for practice in raw_best_practices:
                if not isinstance(practice, str):
                    continue
                text = practice.strip()
                if text and text not in normalized_best_practices:
                    normalized_best_practices.append(text)

        normalized_prompt_injections: List[str] = []
        try:
            prompt_injection = tool_obj.build_system_prompt_injection(context=context)
        except Exception:
            prompt_injection = None

        if isinstance(prompt_injection, str):
            normalized_injection = prompt_injection.strip()
            if normalized_injection:
                normalized_prompt_injections.append(normalized_injection)

        existing = collected.get(name)
        if existing is None:
            collected[name] = {
                "name": name,
                "description": description,
                "best_practices": normalized_best_practices,
                "prompt_injections": normalized_prompt_injections,
            }
            continue

        if not existing.get("description") and description:
            existing["description"] = description

        existing_best_practices = existing.get("best_practices")
        if not isinstance(existing_best_practices, list):
            existing_best_practices = []
            existing["best_practices"] = existing_best_practices

        for practice in normalized_best_practices:
            if practice not in existing_best_practices:
                existing_best_practices.append(practice)

        existing_prompt_injections = existing.get("prompt_injections")
        if not isinstance(existing_prompt_injections, list):
            existing_prompt_injections = []
            existing["prompt_injections"] = existing_prompt_injections

        for injection in normalized_prompt_injections:
            if injection not in existing_prompt_injections:
                existing_prompt_injections.append(injection)

    return [collected[name] for name in sorted(collected.keys())]


def build_tool_best_practices_prompt_block(
    tool_specs: List[Dict[str, Any]],
) -> Optional[str]:
    """Render a stable ToolSpec guidance block for system prompt."""

    if not tool_specs:
        return None

    lines: List[str] = [
        TOOL_PROMPT_BLOCK_START,
        "<instruction>Follow each tool contract and prefer tool-specific best practices.</instruction>",
    ]

    has_any_spec = False
    for spec in tool_specs:
        raw_name = spec.get("name")
        if not isinstance(raw_name, str):
            continue

        name = raw_name.strip()
        if not name:
            continue

        has_any_spec = True
        lines.append(f'<tool name="{name}">')
        raw_description = spec.get("description")
        description = (
            raw_description.strip() if isinstance(raw_description, str) else ""
        )
        if description:
            lines.append(f"<use>{description}</use>")

        normalized_description = _normalize_prompt_text_for_dedupe(description)

        best_practices = spec.get("best_practices")
        if isinstance(best_practices, list):
            normalized_best_practices: List[str] = []
            seen_best_practices: set[str] = set()
            for practice in best_practices:
                if not isinstance(practice, str):
                    continue
                text = practice.strip()
                if not text:
                    continue

                normalized_text = _normalize_prompt_text_for_dedupe(text)
                if not normalized_text or normalized_text in seen_best_practices:
                    continue

                if normalized_description and (
                    normalized_text == normalized_description
                    or normalized_text in normalized_description
                ):
                    continue

                normalized_best_practices.append(text)
                seen_best_practices.add(normalized_text)

            for index, practice in enumerate(normalized_best_practices, start=1):
                lines.append(
                    f'<best_practice index="{index}">{practice}</best_practice>'
                )

        prompt_injections = spec.get("prompt_injections")
        if isinstance(prompt_injections, list):
            normalized_prompt_injections: List[str] = []
            for injection in prompt_injections:
                if not isinstance(injection, str):
                    continue
                text = injection.strip()
                if text and text not in normalized_prompt_injections:
                    normalized_prompt_injections.append(text)

            lines.extend(normalized_prompt_injections)

        lines.append("</tool>")

    if not has_any_spec:
        return None

    lines.append(TOOL_PROMPT_BLOCK_END)
    return "\n".join(lines)


def remove_tool_best_practices_prompt_block(system_prompt: str) -> str:
    """Remove previously injected ToolSpec guidance blocks."""

    cleaned_prompt = system_prompt
    while True:
        start_index = cleaned_prompt.find(TOOL_PROMPT_BLOCK_START)
        if start_index < 0:
            break

        end_index = cleaned_prompt.find(TOOL_PROMPT_BLOCK_END, start_index)
        if end_index < 0:
            cleaned_prompt = cleaned_prompt[:start_index]
            break

        cleaned_prompt = (
            cleaned_prompt[:start_index]
            + cleaned_prompt[end_index + len(TOOL_PROMPT_BLOCK_END) :]
        )

    return cleaned_prompt.strip()


def append_tool_best_practices_prompt_to_messages(
    messages: List[Any],
    tool_specs: List[Dict[str, Any]],
) -> None:
    """Inject one deduplicated ToolSpec guidance block at prompt head."""

    prompt_block = build_tool_best_practices_prompt_block(tool_specs)
    if not prompt_block:
        return

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        if message.get("role") != "system":
            continue

        content = message.get("content")
        base_prompt = ""
        if isinstance(content, str):
            base_prompt = remove_tool_best_practices_prompt_block(content)

        if base_prompt:
            merged_prompt = f"{prompt_block}\n\n{base_prompt}"
        else:
            merged_prompt = prompt_block

        messages[index] = {**message, "content": merged_prompt}
        return

    messages.insert(0, {"role": "system", "content": prompt_block})


def _process_tool_object(
    tool: Tool,
    func_name: str,
    tool_objects: List[Union[Tool, Callable[..., Awaitable[Any]]]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
) -> None:
    """
    处理 Tool 对象。

    Args:
        tool: Tool 实例
        func_name: 函数名，用于日志记录
        tool_objects: 工具对象列表（会被修改）
        tool_map: 工具名称到函数的映射（会被修改）

    Raises:
        TypeError: 当工具的 run 方法不是 async 时抛出
    """
    if not inspect.iscoroutinefunction(tool.run):
        raise TypeError(
            f"LLM 函数 '{func_name}': Tool '{tool.name}' 必须实现 async run 方法"
        )
    tool_objects.append(tool)
    tool_map[tool.name] = tool.run


def _process_decorated_function(
    tool: Callable[..., Awaitable[Any]],
    func_name: str,
    tool_objects: List[Union[Tool, Callable[..., Awaitable[Any]]]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
) -> None:
    """
    处理被 @tool 装饰的函数。

    统一将被 @tool 装饰的函数映射到其 tool_obj.run 方法。

    Args:
        tool: 被 @tool 装饰的异步函数
        func_name: 函数名，用于日志记录
        tool_objects: 工具对象列表（会被修改）
        tool_map: 工具名称到函数的映射（会被修改）

    Raises:
        TypeError: 当函数不是 async 时抛出
    """
    if not inspect.iscoroutinefunction(tool):
        raise TypeError(
            f"LLM 函数 '{func_name}': 被 @tool 装饰的函数 '{tool.__name__}' 必须是 async 函数"
        )

    tool_obj = getattr(tool, "_tool", None)
    assert isinstance(tool_obj, Tool), "这一定是一个Tool对象，不会是None！是None我赤石"

    # 添加 Tool 对象到列表（用于序列化）
    tool_objects.append(tool_obj)

    # 统一映射到 tool_obj.run
    tool_map[tool_obj.name] = tool_obj.run
