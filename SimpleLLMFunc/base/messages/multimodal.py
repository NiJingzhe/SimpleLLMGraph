"""Multimodal content construction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from SimpleLLMFunc.logger import push_debug, push_error, push_warning
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


def handle_union_type(value: Any, args: tuple, param_name: str) -> List[Dict[str, Any]]:
    """Handle Union annotations containing multimodal payload combinations."""

    content: List[Dict[str, Any]] = []

    if isinstance(value, (Text, ImgUrl, ImgPath, str)):
        if isinstance(value, (Text, str)):
            content.append(create_text_content(value, param_name))
        elif isinstance(value, ImgUrl):
            content.append(create_image_url_content(value, param_name))
        elif isinstance(value, ImgPath):
            content.append(create_image_path_content(value, param_name))
        return content

    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            if isinstance(item, (Text, ImgUrl, ImgPath, str)):
                if isinstance(item, (Text, str)):
                    content.append(create_text_content(item, f"{param_name}[{i}]"))
                elif isinstance(item, ImgUrl):
                    content.append(create_image_url_content(item, f"{param_name}[{i}]"))
                elif isinstance(item, ImgPath):
                    content.append(create_image_path_content(item, f"{param_name}[{i}]"))
            else:
                push_error(
                    "多模态参数只能被标注为Optional[List[Text/ImgUrl/ImgPath]] 或 Optional[Text/ImgUrl/ImgPath] 或 List[Text/ImgUrl/ImgPath] 或 Text/ImgUrl/ImgPath",
                    location=get_location(),
                )
                content.append(create_text_content(item, f"{param_name}[{i}]"))
        return content

    return [create_text_content(value, param_name)]


def build_multimodal_content(
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    exclude_params: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Build multimodal payloads based on function arguments and annotations."""

    exclude_params = exclude_params or []
    content: List[Dict[str, Any]] = []

    for param_name, param_value in arguments.items():
        if param_name in exclude_params:
            continue

        if param_name in type_hints:
            annotation = type_hints[param_name]
            parsed_content = parse_multimodal_parameter(
                param_value, annotation, param_name
            )
            content.extend(parsed_content)
        else:
            content.append(create_text_content(param_value, param_name))

    return content


def parse_multimodal_parameter(
    value: Any, annotation: Any, param_name: str
) -> List[Dict[str, Any]]:
    """Recursively parse annotated parameters into OpenAI content payloads."""

    from typing import List as TypingList, Union, get_args, get_origin

    if value is None:
        return []

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        return handle_union_type(value, args, param_name)

    if origin in (list, TypingList):
        if not isinstance(value, (list, tuple)):
            push_warning(
                f"参数 {param_name} 应为列表类型，但获得 {type(value)}",
                location=get_location(),
            )
            return [create_text_content(value, param_name)]

        if not args:
            push_error(
                f"参数 {param_name} 的List类型缺少元素类型注解",
                location=get_location(),
            )
            return [create_text_content(value, param_name)]

        element_type = args[0]

        if element_type not in (Text, ImgUrl, ImgPath, str):
            push_error(
                f"参数 {param_name} 的List类型必须直接包裹基础类型（Text, ImgUrl, ImgPath, str），但获得 {element_type}",
                location=get_location(),
            )
            return [create_text_content(value, param_name)]

        content: List[Dict[str, Any]] = []
        for i, item in enumerate(value):
            item_content = parse_multimodal_parameter(
                item, element_type, f"{param_name}[{i}]"
            )
            content.extend(item_content)
        return content

    if annotation in (Text, str):
        return [create_text_content(value, param_name)]
    if annotation is ImgUrl:
        return [create_image_url_content(value, param_name)]
    if annotation is ImgPath:
        return [create_image_path_content(value, param_name)]

    return [create_text_content(value, param_name)]


def create_text_content(value: Any, param_name: str) -> Dict[str, Any]:
    """Build a text content payload."""

    if isinstance(value, Text):
        text = value.content
    else:
        text = str(value)

    return {"type": "text", "text": f"{param_name}: {text}"}


def create_image_url_content(value: Any, param_name: str) -> Dict[str, Any]:
    """Build an image-url content payload."""

    if value is None:
        return create_text_content("None", param_name)

    if isinstance(value, ImgUrl):
        url = value.url
        detail = value.detail
    else:
        url = str(value)
        detail = "auto"

    push_debug(
        f"Added image URL: {param_name} = {url} (detail: {detail})",
        location=get_location(),
    )

    image_url_data = {"url": url}
    if detail != "auto":
        image_url_data["detail"] = detail

    return {"type": "image_url", "image_url": image_url_data}


def create_image_path_content(value: Any, param_name: str) -> Dict[str, Any]:
    """Build an image-path content payload encoded as base64."""

    if value is None:
        return create_text_content("None", param_name)

    if isinstance(value, ImgPath):
        img_path = value
        detail = value.detail
    else:
        img_path = ImgPath(value)
        detail = "auto"

    base64_img = img_path.to_base64()
    mime_type = img_path.get_mime_type()
    data_url = f"data:{mime_type};base64,{base64_img}"

    push_debug(
        f"Added local image: {param_name} = {img_path.path} (detail: {detail})",
        location=get_location(),
    )

    image_url_data = {"url": data_url}
    if detail != "auto":
        image_url_data["detail"] = detail

    return {"type": "image_url", "image_url": image_url_data}
