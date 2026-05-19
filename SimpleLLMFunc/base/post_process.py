"""LLM response post-processing helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Type, TypeVar, cast, get_origin, get_args

from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_current_context_attribute, get_location

T = TypeVar("T")


def process_response(response: Any, return_type: Optional[Type[T]]) -> T:
    """Convert an LLM response into the expected return type."""

    func_name = get_current_context_attribute("function_name") or "Unknown Function"
    content = extract_content_from_response(response, func_name)

    if content is None:
        content = ""

    if return_type is None or return_type is str:
        return cast(T, content)

    if return_type in (int, float, bool):
        return cast(T, _convert_to_primitive_type(content, return_type))

    # 检查是否为 List 类型
    origin = getattr(return_type, "__origin__", None) or get_origin(return_type)
    if origin is list or origin is List:
        return cast(T, _convert_xml_to_list(content, return_type, func_name))

    if return_type is dict or getattr(return_type, "__origin__", None) is dict:
        return cast(T, _convert_from_xml(content, func_name))

    if return_type and hasattr(return_type, "model_validate"):
        return cast(T, _convert_xml_to_pydantic(content, return_type, func_name))

    try:
        return cast(T, content)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot parse LLM output as expected type: {content}") from exc


def extract_content_from_response(response: Any, func_name: str) -> str:
    """Extract textual content from a normal LLM response."""

    content = ""
    try:
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            # 检查是否是流式响应的 Choice（有 delta 属性）
            if hasattr(choice, "delta") and choice.delta:
                # 这是流式响应的 Choice，使用 delta
                delta = choice.delta
                if hasattr(delta, "content") and delta.content is not None:
                    content = delta.content
                else:
                    content = ""
            elif hasattr(choice, "message") and choice.message:
                # 这是非流式响应的 Choice，使用 message
                message = choice.message
                if hasattr(message, "content") and message.content is not None:
                    content = message.content
                else:
                    content = ""
            else:
                content = ""
        else:
            push_error(
                f"LLM function '{func_name}': unknown response format {type(response)}; converting directly to string",
                location=get_location(),
            )
            content = ""
    except Exception as exc:
        push_error(f"Error extracting response content: {str(exc)}")
        content = ""

    push_debug(f"LLM function '{func_name}' extracted content:\n{content}")
    return content


def extract_content_from_stream_response(chunk: Any, func_name: str) -> str:
    """Extract textual content from a streaming LLM chunk."""

    content = ""
    if not chunk:
        push_warning(
            f"LLM function '{func_name}': detected empty stream chunk; returning empty string",
            location=get_location(),
        )
        return content
    try:
        if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta
                if hasattr(delta, "content") and delta.content is not None:
                    content = delta.content
                else:
                    content = ""
            else:
                content = ""
        else:
            push_debug(
                f"LLM function '{func_name}': detected stream response format {type(chunk)} with content {chunk}; expected no content field, returning empty string",
                location=get_location(),
            )
            content = ""
    except Exception as exc:
        push_error(f"Error extracting stream response content: {str(exc)}")
        content = ""

    return content


def _convert_to_primitive_type(content: str, return_type: Type) -> Any:
    """Cast textual content to primitive Python types."""

    try:
        if return_type is int:
            return int(content.strip())
        if return_type is float:
            return float(content.strip())
        if return_type is bool:
            return content.strip().lower() in ("true", "yes", "1")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot convert LLM response: '{content}' into type: {return_type.__name__}"
        ) from exc
    raise ValueError(f"Unsupported type convertion: {return_type}")


def _extract_xml_content(content: str) -> str:
    """从内容中提取 XML（处理代码块包装）"""
    # 处理 ```xml ... ``` 包装
    xml_pattern = r"```xml\s*([\s\S]*?)\s*```"
    match = re.search(xml_pattern, content)
    if match:
        return match.group(1).strip()

    # 处理纯 XML
    cleaned_content = content.strip()
    if cleaned_content.startswith("<"):
        # 移除可能的代码块标记
        if cleaned_content.startswith("```") and cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[3:-3].strip()
        return cleaned_content

    # 如果没有找到 XML，返回原始内容（让解析器处理错误）
    return cleaned_content


def _convert_from_xml(content: str, func_name: str) -> Dict[str, Any]:
    """从 XML 字符串解析为字典"""
    from SimpleLLMFunc.base.type_resolve.xml_utils import xml_to_dict

    try:
        xml_content = _extract_xml_content(content)
        return xml_to_dict(xml_content)
    except Exception as exc:
        push_error(
            f"LLM function '{func_name}': XML parsing failed: {str(exc)}, content: {content[:200]}",
            location=get_location(),
        )
        raise ValueError(f"Cannot parse LLM response as valid XML: {str(exc)}") from exc


def _convert_xml_to_list(content: str, list_type: Any, func_name: str) -> List[Any]:
    """从 XML 字符串解析为 List"""
    from SimpleLLMFunc.base.type_resolve.xml_utils import xml_to_dict

    try:
        if not content.strip():
            raise ValueError("Received empty response")

        xml_content = _extract_xml_content(content)
        data = xml_to_dict(xml_content)

        # 处理根元素为 result 的情况：<result><item>...</item><item>...</item></result>
        if isinstance(data, dict) and "item" in data and isinstance(data["item"], list):
            item_list = data["item"]
        elif isinstance(data, list):
            # 如果直接是列表（例如 <result> 包含多个 <item> 时可能返回列表）
            item_list = data
        elif isinstance(data, dict) and len(data) == 1:
            # 如果只有一个键，尝试使用其值
            item_list = list(data.values())[0]
            if not isinstance(item_list, list):
                item_list = [item_list]
        else:
            raise ValueError(f"Cannot extract list data from XML: {data}")

        # 获取列表元素的类型
        args = get_args(list_type)
        item_type = args[0] if args else Any

        # 转换列表中的每个元素
        converted_list = []
        for item in item_list:
            if item_type is str:
                converted_list.append(str(item))
            elif item_type is int:
                converted_list.append(int(item) if isinstance(item, (int, str)) else int(str(item)))
            elif item_type is float:
                converted_list.append(float(item) if isinstance(item, (int, float, str)) else float(str(item)))
            elif item_type is bool:
                if isinstance(item, bool):
                    converted_list.append(item)
                elif isinstance(item, str):
                    converted_list.append(item.lower() in ("true", "yes", "1"))
                else:
                    converted_list.append(bool(item))
            else:
                # 其他类型直接使用
                converted_list.append(item)

        return converted_list
    except Exception as exc:
        push_error(f"Parsing error details: {str(exc)}, content: {content[:200]}")
        raise ValueError(f"Cannot parse as List: {str(exc)}") from exc


def _convert_xml_to_pydantic(content: str, model_class: Type, func_name: str) -> Any:
    """从 XML 字符串解析为 Pydantic 模型"""
    from SimpleLLMFunc.base.type_resolve.xml_utils import xml_to_dict, dict_to_pydantic

    try:
        if not content.strip():
            raise ValueError("Received empty response")

        xml_content = _extract_xml_content(content)
        data_dict = xml_to_dict(xml_content)

        # 处理根元素：如果根元素是模型类名，提取其内容
        model_name = model_class.__name__
        if isinstance(data_dict, dict) and len(data_dict) == 1 and model_name in data_dict:
            data_dict = data_dict[model_name]

        return dict_to_pydantic(data_dict, model_class)
    except Exception as exc:
        push_error(f"Parsing error details: {str(exc)}, content: {content[:200]}")
        raise ValueError(f"Cannot parse as Pydantic model: {str(exc)}") from exc


__all__ = [
    "process_response",
    "extract_content_from_response",
    "extract_content_from_stream_response",
]
