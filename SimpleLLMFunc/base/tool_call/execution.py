"""Tool call execution helpers."""

from __future__ import annotations

import asyncio
import inspect
import json
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

from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.observability.langfuse_client import (
    coerce_langfuse_metadata,
    get_langfuse_trace_context,
    langfuse_client,
)
from SimpleLLMFunc.base.tool_call.extraction import (
    parse_tool_call_arguments,
    repair_tool_call_arguments,
)


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


async def _execute_single_tool_call(
    tool_call: Dict[str, Any],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    event_emitter: Any = None,
    trace_context: Optional[dict[str, str]] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    """Execute a single tool call and return its results.

    处理两类工具调用结果：
    1. 普通工具调用（返回 JSON 可序列化的文本/对象）
       - 返回 is_multimodal=False
       - 消息列表包含标准的 tool role message
       - 这些结果会直接添加到消息历史中

    2. 多模态工具调用（返回图像、文件等）
       - 返回 is_multimodal=True
       - 消息列表包含 user role message，带有多模态内容（图像、文件等）
       - 这些结果不能通过标准的 OpenAI tool_call 机制传输
       - 因此需要特殊处理：移除原始 assistant message 中的 tool_call，
         用 assistant + user 消息对替代

    Returns:
        Tuple of (tool_call_dict, list_of_messages_to_append, is_multimodal)
        其中 is_multimodal 指示是否为多模态结果
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
        return (tool_call, messages_to_append, False)

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

            # 更新为解析后的参数
            tool_span.update(input=arguments)

            push_debug(f"Executing tool '{tool_name}' with arguments: {arguments_str}")

            tool_func = tool_map[tool_name]

            # 获取原始函数和 Tool 对象（用于检查参数）
            # tool_func 是 tool.run，是一个绑定方法，__self__ 就是 Tool 对象
            original_func = None
            tool_obj = None
            has_event_emitter_param = False

            # 尝试从绑定方法获取 Tool 对象
            if hasattr(tool_func, "__self__"):
                tool_obj = tool_func.__self__
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
                return (tool_call, messages_to_append, False)

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
                            "text": f"这是工具 '{tool_name}' 返回的图像：",
                        },
                        image_content,
                    ],
                }
                messages_to_append.append(user_multimodal_message)
                return (tool_call, messages_to_append, True)

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
                            "text": f"这是工具 '{tool_name}' 返回的图像文件：",
                        },
                        image_content,
                    ],
                }
                messages_to_append.append(user_multimodal_message)
                return (tool_call, messages_to_append, True)

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
                                "text": f"这是工具 '{tool_name}' 返回的图像和说明：{text_part}",
                            },
                            image_content,
                        ],
                    }
                    messages_to_append.append(user_multimodal_message)
                    return (tool_call, messages_to_append, True)

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
                                "text": f"这是工具 '{tool_name}' 返回的图像文件和说明：{text_part}",
                            },
                            image_content,
                        ],
                    }
                    messages_to_append.append(user_multimodal_message)
                    return (tool_call, messages_to_append, True)

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
                return (tool_call, messages_to_append, False)

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

    return (tool_call, messages_to_append, False)


async def process_tool_calls(
    tool_calls: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    tool_map: Dict[str, Callable[..., Awaitable[Any]]],
    event_emitter: Any = None,
    abort_signal: Optional[AbortSignal] = None,
) -> List[Dict[str, Any]]:
    """Execute tool calls concurrently and append results to the message history.

    All tool calls are executed in parallel using structured concurrency with asyncio.gather(),
    then results are appended to messages in the original order.

    对于多模态工具调用，会先插入一个 assistant message 说明将使用该工具，
    然后再插入工具结果的 user message。

    IMPORTANT: 此函数会修改 `messages` 参数中的原始字典对象（特别是 assistant message），
    这是**必需的行为**，而非 bug，原因如下：

    1. 多模态工具调用处理：
       - OpenAI API 的 tool_call 机制无法传输图像等多模态内容
       - 对于多模态工具调用（返回图片、文件等），我们需要：
         a) 从原始 assistant message 中移除该工具的 tool_call 定义
         b) 用自定义的 assistant + user 消息对替代（用户在 user message 中提供多模态内容）
       - 这就是为什么需要修改原始 messages 中的 assistant message 对象

    2. 为什么不用 deep copy：
       - deep copy 会增加内存开销
       - 业务逻辑本身需要改变消息结构
       - 调用者最终收到的 messages 就包含了这些必要的修改

    Args:
        tool_calls: 要执行的工具调用列表
        messages: 消息历史列表。**会被就地修改**（仅修改 assistant message，不改变列表本身）
        tool_map: 工具名称到函数的映射字典

    Returns:
        修改后的完整消息列表，包含原始消息、工具调用结果和多模态替代消息
    """

    if not tool_calls:
        return messages

    if abort_signal is not None and abort_signal.is_aborted:
        return messages

    # Execute all tool calls concurrently
    trace_context = get_langfuse_trace_context()
    tasks = [
        asyncio.create_task(
            _execute_single_tool_call(
                tool_call,
                tool_map,
                event_emitter,
                trace_context=trace_context,
            )
        )
        for tool_call in tool_calls
    ]

    if abort_signal is None:
        results = await asyncio.gather(*tasks)
    else:
        abort_task = asyncio.create_task(abort_signal.wait())
        gather_future = asyncio.gather(*tasks)
        done, _ = await asyncio.wait(
            {abort_task, gather_future},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            gather_future.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(gather_future, return_exceptions=True)
            return messages
        abort_task.cancel()
        results = await gather_future

    # 分类结果：普通工具调用和多模态工具调用
    normal_results: List[List[Dict[str, Any]]] = []
    multimodal_results: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    multimodal_tool_call_ids: set[str] = set()

    for tool_call_dict, messages_to_append, is_multimodal in results:
        if is_multimodal:
            multimodal_results.append((tool_call_dict, messages_to_append))
            tool_call_id = tool_call_dict.get("id")
            if tool_call_id:
                multimodal_tool_call_ids.add(tool_call_id)
        else:
            normal_results.append(messages_to_append)

    # =========================================================================
    # 阶段 1: 从原始 messages 中移除多模态工具调用的 tool_calls
    # =========================================================================
    # 这一步是**必需的**，原因如下：
    # 1. 多模态工具调用无法通过标准的 OpenAI tool_call 机制传输（OpenAI API 不支持）
    # 2. 因此我们需要从消息历史中移除这些 tool_calls
    # 3. 后续会用 assistant + user 消息对替代（user message 中包含多模态内容）
    # 4. 修改原始 messages 中的字典对象是合理的，因为这些改动必须被反映到最终结果中
    # =========================================================================
    if multimodal_tool_call_ids:
        # 找到最后一个包含tool_calls的assistant message
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                original_tool_calls = msg["tool_calls"]
                # 过滤掉多模态工具调用，保留普通工具调用
                filtered_tool_calls = [
                    tc
                    for tc in original_tool_calls
                    if tc.get("id") not in multimodal_tool_call_ids
                ]

                if not filtered_tool_calls:
                    # 如果所有tool_calls都是多模态的，移除tool_calls字段
                    # 并用空字符串替代content（不能是None，因为那是tool_call专用格式）
                    del msg["tool_calls"]
                    if msg.get("content") is None:
                        msg["content"] = ""
                else:
                    # 否则更新为过滤后的tool_calls（只保留普通工具调用）
                    msg["tool_calls"] = filtered_tool_calls
                break

    # =========================================================================
    # 阶段 2: 构建最终消息列表
    # =========================================================================
    # 从修改后的 messages 开始（已移除多模态工具调用），然后追加结果
    # 注：这里的 .copy() 只是为了创建新列表对象，不是 defensive copy
    #    因为已经在上面修改了原始 messages 中的字典内容（assistant message）
    #    这些修改是必需的，不需要"防守"
    current_messages = messages.copy()
    for msgs in normal_results:
        current_messages.extend(msgs)

    # =========================================================================
    # 阶段 3: 处理多模态工具调用结果
    # =========================================================================
    # 多模态工具调用（返回图像、文件等）需要特殊处理：
    # 1. 创建一个 assistant message 说明将使用该工具
    # 2. 然后添加用户提供的 user message（包含多模态内容）
    # 这样做是因为 OpenAI API 的标准 tool_call 机制无法处理多模态结果
    # 所以我们用消息对的方式来模拟工具调用的交互过程
    for tool_call_dict, user_messages in multimodal_results:
        tool_name = tool_call_dict.get("function", {}).get("name", "unknown")
        arguments = tool_call_dict.get("function", {}).get("arguments", "{}")

        # 创建assistant message说明将使用该工具
        assistant_message = {
            "role": "assistant",
            "content": f"我将求助用户使用 {tool_name} 工具来获取结果，使用参数为：{arguments}，请用户按照工具的描述和参数要求，提供符合要求的结果。",
        }
        current_messages.append(assistant_message)

        # 添加工具返回的user message（包含多模态内容）
        current_messages.extend(user_messages)

    return current_messages
