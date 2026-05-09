"""Tool call execution helpers."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    get_type_hints,
    get_origin,
    get_args,
    Union as TypingUnion,
)
from langfuse.types import TraceContext

from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text
from SimpleLLMFunc.observability.langfuse_client import (
    coerce_langfuse_metadata,
    langfuse_client,
)
from SimpleLLMFunc.base.tool_call.extraction import (
    parse_tool_call_arguments,
    repair_tool_call_arguments,
)


@dataclass
class ExecutedToolCallResult:
    tool_call: Dict[str, Any]
    tool_call_id: str
    tool_name: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    result: Any = None
    is_multimodal: bool = False
    success: bool = True
    error: Optional[Exception] = None


def _convert_tool_arguments(
    arguments: Dict[str, Any],
    tool_func: Callable[..., Awaitable[Any]],
) -> Dict[str, Any]:
    """转换工具参数，将字符串列表转换为多模态对象列表。

    根据工具函数的类型注解，自动将 LLM 传递的字符串数组转换为对应的多模态对象数组。
    支持的类型：
    - List[ImgPath] -> List[ImgPath对象]
    - List[ImgUrl] -> List[ImgUrl对象]
    - List[Text] -> List[Text对象]
    - Optional[List[...]] -> 处理 None 值
    - Union 类型 -> 提取非 None 类型

    Args:
        arguments: LLM 传递的原始参数字典（JSON 解析后）
        tool_func: 工具函数对象

    Returns:
        转换后的参数字典
    """
    try:
        # 获取函数签名和类型注解
        signature = inspect.signature(tool_func)
        type_hints = get_type_hints(tool_func)

        converted_args = {}

        for param_name, param_value in arguments.items():
            if param_name not in signature.parameters:
                # 参数不在签名中，保持原样
                converted_args[param_name] = param_value
                continue

            param_type = type_hints.get(param_name, Any)

            # 处理 None 值
            if param_value is None:
                converted_args[param_name] = None
                continue

            # 处理 Optional 类型
            origin = get_origin(param_type)
            if origin is TypingUnion:
                args = get_args(param_type)
                # 提取非 None 类型
                non_none_types = [t for t in args if t is not type(None)]
                if non_none_types:
                    param_type = non_none_types[0]
                    origin = get_origin(param_type)

            # 处理列表类型
            if origin is list:
                args = get_args(param_type)
                if not args:
                    # List 没有类型参数，保持原样
                    converted_args[param_name] = param_value
                    continue

                element_type = args[0]

                # 检查是否为多模态列表类型
                if element_type is ImgPath:
                    if isinstance(param_value, list):
                        try:
                            converted_args[param_name] = [
                                ImgPath(path) for path in param_value
                            ]
                        except Exception as e:
                            push_warning(
                                f"Failed to convert tool argument '{param_name}' to List[ImgPath]: {e}; using original value",
                                location=get_location(),
                            )
                            converted_args[param_name] = param_value
                    else:
                        converted_args[param_name] = param_value
                elif element_type is ImgUrl:
                    if isinstance(param_value, list):
                        try:
                            converted_args[param_name] = [
                                ImgUrl(url) for url in param_value
                            ]
                        except Exception as e:
                            push_warning(
                                f"Failed to convert tool argument '{param_name}' to List[ImgUrl]: {e}; using original value",
                                location=get_location(),
                            )
                            converted_args[param_name] = param_value
                    else:
                        converted_args[param_name] = param_value
                elif element_type is Text:
                    if isinstance(param_value, list):
                        try:
                            converted_args[param_name] = [
                                Text(text) for text in param_value
                            ]
                        except Exception as e:
                            push_warning(
                                f"Failed to convert tool argument '{param_name}' to List[Text]: {e}; using original value",
                                location=get_location(),
                            )
                            converted_args[param_name] = param_value
                    else:
                        converted_args[param_name] = param_value
                else:
                    # 非多模态列表，保持原样
                    converted_args[param_name] = param_value
            # 处理单个多模态类型
            elif param_type is ImgPath:
                if isinstance(param_value, str):
                    try:
                        converted_args[param_name] = ImgPath(param_value)
                    except Exception as e:
                        push_warning(
                            f"Failed to convert tool argument '{param_name}' to ImgPath: {e}; using original value",
                            location=get_location(),
                        )
                        converted_args[param_name] = param_value
                else:
                    converted_args[param_name] = param_value
            elif param_type is ImgUrl:
                if isinstance(param_value, str):
                    try:
                        converted_args[param_name] = ImgUrl(param_value)
                    except Exception as e:
                        push_warning(
                            f"Failed to convert tool argument '{param_name}' to ImgUrl: {e}; using original value",
                            location=get_location(),
                        )
                        converted_args[param_name] = param_value
                else:
                    converted_args[param_name] = param_value
            elif param_type is Text:
                if isinstance(param_value, str):
                    try:
                        converted_args[param_name] = Text(param_value)
                    except Exception as e:
                        push_warning(
                            f"Failed to convert tool argument '{param_name}' to Text: {e}; using original value",
                            location=get_location(),
                        )
                        converted_args[param_name] = param_value
                else:
                    converted_args[param_name] = param_value
            else:
                # 其他类型，保持原样
                converted_args[param_name] = param_value

        return converted_args
    except Exception as e:
        push_warning(
            f"Error while converting tool arguments: {e}; using original arguments",
            location=get_location(),
        )
        return arguments


async def execute_single_tool_call_result(
    tool_call: Dict[str, Any],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    event_emitter: Any = None,
    trace_context: Optional[TraceContext] = None,
) -> ExecutedToolCallResult:
    """Execute a single tool call and return its results.

    Returns a structured tool call execution result.

    The returned object carries both the raw execution result and the generated
    message patches, with a dedicated multimodal flag for higher layers.
    """

    tool_call_id = tool_call.get("id")
    function_call = tool_call.get("function", {})
    tool_name = function_call.get("name")
    arguments_str = function_call.get("arguments", "{}")
    messages_to_append: List[Dict[str, Any]] = []

    if tool_name not in tool_map:
        push_error(f"Tool '{tool_name}' is not available in the tool map")
        tool_error_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(
                {"error": f"Tool '{tool_name}' was not found"}, ensure_ascii=False, indent=2
            ),
        }
        messages_to_append.append(tool_error_message)
        return ExecutedToolCallResult(
            tool_call=tool_call,
            tool_call_id=str(tool_call_id or ""),
            tool_name=str(tool_name or ""),
            messages=messages_to_append,
            result={"error": f"Tool '{tool_name}' was not found"},
            success=False,
        )

    # 使用 Langfuse 观测工具调用
    with langfuse_client.start_as_current_observation(
        as_type="tool",
        name=tool_name,
        input={"raw_arguments": arguments_str},
        metadata=coerce_langfuse_metadata({"tool_call_id": tool_call_id}),
        trace_context=trace_context,
    ) as tool_span:
        try:
            repaired_arguments_str = repair_tool_call_arguments(arguments_str)
            if repaired_arguments_str != arguments_str:
                push_warning(
                    f"Tool '{tool_name}' argument JSON was auto-repaired",
                    location=get_location(),
                )
                arguments_str = repaired_arguments_str

            arguments = parse_tool_call_arguments(arguments_str)
            if arguments is None:
                raise ValueError("Tool arguments are not a valid JSON object")

            tool_span.update(input=arguments)

            push_debug(f"Executing tool '{tool_name}' with arguments: {arguments_str}")

            tool_func = tool_map[tool_name]

            # 获取原始函数和 Tool 对象（用于检查参数）
            # tool_func 是 tool.run，是一个绑定方法，__self__ 就是 Tool 对象
            original_func = None
            tool_obj = None
            has_event_emitter_param = False

            # 尝试从绑定方法获取 Tool 对象
            bound_self = getattr(tool_func, "__self__", None)
            if bound_self is not None:
                tool_obj = bound_self
                original_func = getattr(tool_obj, "func", None)
                if tool_obj and hasattr(tool_obj, "parameters"):
                    has_event_emitter_param = "event_emitter" in [
                        p.name for p in tool_obj.parameters
                    ]

            if original_func is None:
                original_func = tool_func

            # 转换参数：将字符串列表转换为多模态对象列表
            converted_arguments = _convert_tool_arguments(arguments, original_func)

            # 注入 event_emitter：如果工具函数有 event_emitter 参数
            if event_emitter is not None and has_event_emitter_param:
                converted_arguments["event_emitter"] = event_emitter

            tool_result = await tool_func(**converted_arguments)

            # 更新工具调用观测数据，序列化输出以便langfuse记录
            from SimpleLLMFunc.base.tool_call.validation import (
                is_valid_tool_result,
                serialize_tool_output_for_langfuse,
            )

            serialized_output = serialize_tool_output_for_langfuse(tool_result)
            tool_span.update(output=serialized_output)

            if not is_valid_tool_result(tool_result):
                push_warning(
                    f"工具 '{tool_name}' 返回了不支持的格式: {type(tool_result)}。支持的返回格式包括: str, JSON可序列化对象, ImgPath, ImgUrl, Tuple[str, ImgPath], Tuple[str, ImgUrl]",
                    f"Tool '{tool_name}' returned unsupported type {type(tool_result)}. Supported return formats: str, JSON-serializable objects, ImgPath, ImgUrl, Tuple[str, ImgPath], Tuple[str, ImgUrl]",
                    location=get_location(),
                )
                tool_result_content_json: str = json.dumps(
                    str(tool_result), ensure_ascii=False, indent=2
                )
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }
                messages_to_append.append(tool_message)
                return ExecutedToolCallResult(
                    tool_call=tool_call,
                    tool_call_id=str(tool_call_id or ""),
                    tool_name=str(tool_name or ""),
                    messages=messages_to_append,
                    result=str(tool_result),
                )

            if isinstance(tool_result, ImgUrl):
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": tool_result.url,
                        "detail": tool_result.detail,
                    },
                }

                user_multimodal_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"This is an image returned by tool '{tool_name}':",
                        },
                        image_content,
                    ],
                }
                messages_to_append.append(user_multimodal_message)
                return ExecutedToolCallResult(
                    tool_call=tool_call,
                    tool_call_id=str(tool_call_id or ""),
                    tool_name=str(tool_name or ""),
                    messages=messages_to_append,
                    result=tool_result,
                    is_multimodal=True,
                )

            if isinstance(tool_result, ImgPath):
                base64_img = tool_result.to_base64()
                mime_type = tool_result.get_mime_type()
                data_url = f"data:{mime_type};base64,{base64_img}"

                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": tool_result.detail,
                    },
                }

                user_multimodal_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"This is an image file returned by tool '{tool_name}':",
                        },
                        image_content,
                    ],
                }
                messages_to_append.append(user_multimodal_message)
                return ExecutedToolCallResult(
                    tool_call=tool_call,
                    tool_call_id=str(tool_call_id or ""),
                    tool_name=str(tool_name or ""),
                    messages=messages_to_append,
                    result=tool_result,
                    is_multimodal=True,
                )

            if isinstance(tool_result, tuple) and len(tool_result) == 2:
                text_part, img_part = tool_result
                if isinstance(text_part, str) and isinstance(img_part, ImgUrl):
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": img_part.url,
                            "detail": img_part.detail,
                        },
                    }

                    user_multimodal_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"This is an image and description returned by tool '{tool_name}': {text_part}",
                            },
                            image_content,
                        ],
                    }
                    messages_to_append.append(user_multimodal_message)
                    return ExecutedToolCallResult(
                        tool_call=tool_call,
                        tool_call_id=str(tool_call_id or ""),
                        tool_name=str(tool_name or ""),
                        messages=messages_to_append,
                        result=tool_result,
                        is_multimodal=True,
                    )

                if isinstance(text_part, str) and isinstance(img_part, ImgPath):
                    base64_img = img_part.to_base64()
                    mime_type = img_part.get_mime_type()
                    data_url = f"data:{mime_type};base64,{base64_img}"

                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": img_part.detail,
                        },
                    }

                    user_multimodal_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"This is an image file and description returned by tool '{tool_name}': {text_part}",
                            },
                            image_content,
                        ],
                    }
                    messages_to_append.append(user_multimodal_message)
                    return ExecutedToolCallResult(
                        tool_call=tool_call,
                        tool_call_id=str(tool_call_id or ""),
                        tool_name=str(tool_name or ""),
                        messages=messages_to_append,
                        result=tool_result,
                        is_multimodal=True,
                    )

                tool_result_content_json = json.dumps(
                    tool_result, ensure_ascii=False, indent=2
                )
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }
                messages_to_append.append(tool_message)
                push_debug(
                    f"Tool '{tool_name}' completed: {tool_result_content_json}"
                )
                return ExecutedToolCallResult(
                    tool_call=tool_call,
                    tool_call_id=str(tool_call_id or ""),
                    tool_name=str(tool_name or ""),
                    messages=messages_to_append,
                    result=tool_result,
                )

            if isinstance(tool_result, (Text, str)):
                tool_result_content_json = json.dumps(
                    tool_result, ensure_ascii=False, indent=2
                )

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }
            else:
                tool_result_content_json = json.dumps(
                    tool_result, ensure_ascii=False, indent=2
                )

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_content_json,
                }

            messages_to_append.append(tool_message)

            if isinstance(tool_result, (ImgUrl, ImgPath)):
                push_debug(
                    f"Tool '{tool_name}' completed: image payload",
                    location=get_location(),
                )
            else:
                push_debug(
                    f"Tool '{tool_name}' completed: {json.dumps(tool_result, ensure_ascii=False)}"
                )

        except Exception as exc:
            error_message = (
                f"Tool '{tool_name}' failed during execution or result parsing with "
                f"arguments {arguments_str}: {str(exc)}"
            )
            push_error(error_message)

            # 记录错误到langfuse
            tool_span.update(
                output={"error": error_message, "exception_type": type(exc).__name__},
                level="ERROR",
            )

            tool_error_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {"error": error_message}, ensure_ascii=False, indent=2
                ),
            }
            messages_to_append.append(tool_error_message)
            return ExecutedToolCallResult(
                tool_call=tool_call,
                tool_call_id=str(tool_call_id or ""),
                tool_name=str(tool_name or ""),
                messages=messages_to_append,
                result={"error": error_message},
                success=False,
                error=exc,
            )

    return ExecutedToolCallResult(
        tool_call=tool_call,
        tool_call_id=str(tool_call_id or ""),
        tool_name=str(tool_name or ""),
        messages=messages_to_append,
        result=tool_result,
    )


__all__ = [
    "ExecutedToolCallResult",
    "execute_single_tool_call_result",
]
