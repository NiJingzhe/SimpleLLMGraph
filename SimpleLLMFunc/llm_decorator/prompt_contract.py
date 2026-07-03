"""Prompt contract and transcript-seed helpers for decorators.

This module contains real prompt/seed rules used by InvocationSpec builders. It
replaces the historical decorator step package without introducing a
runner/adapter layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast, get_origin

from SimpleLLMFunc.base.messages import build_multimodal_content
from SimpleLLMFunc.base.type_resolve.description import (
    build_type_description_xml,
    generate_example_xml,
    get_detailed_type_description,
)
from SimpleLLMFunc.base.type_resolve.multimodal import has_multimodal_content
from SimpleLLMFunc.logger import push_debug, push_warning
from SimpleLLMFunc.logger.logger import get_location
from SimpleLLMFunc.runtime.selfref.state import MemoryHistory
from SimpleLLMFunc.type import HistoryList
from SimpleLLMFunc.type.chat_input import (
    UserChatMessage,
    normalize_user_chat_message,
)
from SimpleLLMFunc.type.message import MessageList, MessageParam, NormalizedMessageList

HISTORY_PARAM_NAMES: List[str] = ["history", "chat_history"]

DEFAULT_SYSTEM_PROMPT_TEMPLATE_PLAIN = """
Your task is to provide results that meet the requirements based on the **function description**
and the user's request.

- Function Description:
    {function_description}

- You will receive the following parameters:
    {parameters_description}

- The type of content you need to return:
    {return_type_description}

Execution Requirements:
1. Return the result in plain text
2. Use minimal formatting and add extra structure when the request calls for it
"""

DEFAULT_SYSTEM_PROMPT_TEMPLATE_XML = """
Your task is to provide results that meet the requirements based on the **function description** 
and the user's request.

- Function Description:
    {function_description}

- You will receive the following parameters:
    {parameters_description}

- The type of content you need to return:
    {return_type_description}

Execution Requirements:
1. Use available tools to assist in completing the task if needed
2. Return the result as well-formed XML
3. Ensure all XML tags are properly closed
"""

DEFAULT_USER_PROMPT_TEMPLATE = """
The parameters provided are:
    {parameters}

Return the result directly as the final output.
"""


def process_docstring_template(
    docstring: str,
    template_params: Optional[Dict[str, Any]],
) -> str:
    if not template_params:
        return docstring

    try:
        return docstring.format(**template_params)
    except KeyError as exc:
        push_warning(
            f"DocString template parameter substitution failed: missing parameter {exc}. "
            "Using original DocString.",
            location=get_location(),
        )
        return docstring
    except Exception as exc:
        push_warning(
            f"Error during DocString template parameter substitution: {str(exc)}. "
            "Using original DocString.",
            location=get_location(),
        )
        return docstring


def extract_parameter_type_hints(type_hints: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in type_hints.items() if key != "return"}


def is_complex_return_type(return_type: Any) -> bool:
    from typing import Union as TypingUnion

    from pydantic import BaseModel

    if return_type is None:
        return False

    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return True

    origin = getattr(return_type, "__origin__", None) or get_origin(return_type)
    return origin in (list, List, dict, Dict, TypingUnion)


def build_parameter_type_descriptions(
    param_type_hints: Dict[str, Any],
) -> List[str]:
    descriptions = []
    for param_name, param_type in param_type_hints.items():
        type_str = (
            get_detailed_type_description(param_type) if param_type else "Unknown Type"
        )
        descriptions.append(f"  - {param_name}: {type_str}")
    return descriptions


def build_return_type_description(return_type: Any) -> str:
    from typing import Union as TypingUnion

    from pydantic import BaseModel

    if return_type is None:
        return "未知类型"

    if return_type in (str, int, float, bool, type(None)):
        return get_detailed_type_description(return_type)

    complex_type = False
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        complex_type = True
    else:
        origin = getattr(return_type, "__origin__", None) or get_origin(return_type)
        if origin in (list, List, dict, Dict, TypingUnion):
            complex_type = True

    if not complex_type:
        return get_detailed_type_description(return_type)

    try:
        type_xml_schema = build_type_description_xml(return_type)
        example_xml = generate_example_xml(return_type)
        return "XML Schema:\n" + type_xml_schema + "\n\nExample XML:\n" + example_xml
    except Exception as exc:
        push_warning(
            "Failed to generate structured XML type description, "
            f"falling back to text format: {str(exc)}",
            location=get_location(),
        )
        return get_detailed_type_description(return_type)


def build_text_messages(
    processed_docstring: str,
    param_type_descriptions: List[str],
    return_type_description: str,
    arguments: Dict[str, Any],
    system_template: str,
    user_template: str,
) -> MessageList:
    system_prompt = system_template.format(
        function_description=processed_docstring,
        parameters_description="\n".join(param_type_descriptions),
        return_type_description=return_type_description,
    )
    user_param_values = [
        f"  - {param_name}: {param_value}"
        for param_name, param_value in arguments.items()
    ]
    user_prompt = user_template.format(parameters="\n".join(user_param_values))

    messages: MessageList = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    push_debug(f"System prompt: {system_prompt}", location=get_location())
    push_debug(f"User prompt: {user_prompt}", location=get_location())
    return messages


def build_function_transcript_seed(
    *,
    processed_docstring: str,
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    return_type: Any,
    system_prompt_template: Optional[str],
    user_prompt_template: Optional[str],
) -> tuple[NormalizedMessageList, str, str]:
    param_type_hints = extract_parameter_type_hints(type_hints)
    param_type_descriptions = build_parameter_type_descriptions(param_type_hints)
    return_type_description = build_return_type_description(return_type)
    system_template = system_prompt_template or (
        DEFAULT_SYSTEM_PROMPT_TEMPLATE_XML
        if is_complex_return_type(return_type)
        else DEFAULT_SYSTEM_PROMPT_TEMPLATE_PLAIN
    )
    user_template = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
    text_messages = build_text_messages(
        processed_docstring,
        param_type_descriptions,
        return_type_description,
        arguments,
        system_template,
        user_template,
    )
    system_prompt = cast(str, text_messages[0].get("content") or "")

    if has_multimodal_content(arguments, type_hints):
        user_content = build_multimodal_content(arguments, type_hints)
        return (
            cast(
                NormalizedMessageList,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            ),
            system_prompt,
            return_type_description,
        )

    return cast(NormalizedMessageList, text_messages), system_prompt, return_type_description


def extract_conversation_history(
    arguments: Dict[str, Any],
    func_name: str,
    history_param_names: Optional[List[str]] = None,
) -> Optional[HistoryList]:
    history_param_names = history_param_names or HISTORY_PARAM_NAMES
    history_param_name = next(
        (param_name for param_name in history_param_names if param_name in arguments),
        None,
    )
    if not history_param_name:
        push_warning(
            f"LLM Chat '{func_name}' missing history parameter "
            f"(parameter name should be one of {history_param_names}). "
            "History will not be passed.",
            location=get_location(),
        )
        return None

    custom_history = arguments[history_param_name]
    if not (
        isinstance(custom_history, list)
        and all(isinstance(item, dict) for item in custom_history)
    ):
        push_warning(
            f"LLM Chat '{func_name}' history parameter should be List[Dict[str, str]] type. "
            "History will not be passed.",
            location=get_location(),
        )
        return None

    return custom_history


def _extract_explicit_chat_user_message(
    arguments: Dict[str, Any],
    exclude_params: List[str],
) -> Optional[Dict[str, Any]]:
    if "message" not in arguments or "message" in exclude_params:
        return None

    value = arguments["message"]
    if isinstance(value, UserChatMessage):
        return value.to_message()
    if isinstance(value, dict) and value.get("role") == "user" and "content" in value:
        return normalize_user_chat_message(value)
    return None


def build_chat_user_message_content(
    arguments: Dict[str, Any],
    exclude_params: List[str],
) -> Union[str, List[Dict[str, Any]]]:
    explicit_message = _extract_explicit_chat_user_message(arguments, exclude_params)
    if explicit_message is not None:
        return cast(Union[str, List[Dict[str, Any]]], explicit_message["content"])

    if "message" not in arguments or "message" in exclude_params:
        return ""

    message_value = arguments["message"]
    if message_value is None:
        return ""
    if isinstance(message_value, str):
        return message_value
    return str(message_value)


def build_chat_system_prompt(
    docstring: str,
    history_system_prompt: Optional[str] = None,
) -> Optional[str]:
    base_prompt = history_system_prompt if history_system_prompt else docstring
    return base_prompt or None


def extract_history_system_prompt(history: Optional[HistoryList]) -> Optional[str]:
    if not history:
        return None

    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
    return None


def filter_history_messages(
    history: HistoryList,
    func_name: str,
) -> HistoryList:
    _ = func_name
    filtered = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            if msg["role"] not in ["system"]:
                filtered.append(msg)
        else:
            push_warning(
                f"Skipping malformed history item: {msg}",
                location=get_location(),
            )
    return filtered


def build_chat_messages(
    *,
    docstring: str,
    func_name: str,
    arguments: Dict[str, Any],
    type_hints: Dict[str, Any],
    exclude_params: Optional[List[str]] = None,
    template_params: Optional[Dict[str, Any]] = None,
) -> MemoryHistory:
    exclude_params = exclude_params or HISTORY_PARAM_NAMES
    messages: MemoryHistory = []
    processed_docstring = process_docstring_template(docstring, template_params)
    custom_history = extract_conversation_history(arguments, func_name)
    system_content = build_chat_system_prompt(
        processed_docstring,
        history_system_prompt=extract_history_system_prompt(custom_history),
    )
    if system_content:
        messages.append({"role": "system", "content": system_content})
    if custom_history:
        messages.extend(filter_history_messages(custom_history, func_name))

    explicit_message = _extract_explicit_chat_user_message(arguments, exclude_params)
    if explicit_message is not None:
        messages.append(cast(MessageParam, explicit_message))
        return messages

    user_message_content = build_chat_user_message_content(
        arguments,
        exclude_params,
    )
    if user_message_content:
        messages.append(
            cast(MessageParam, {"role": "user", "content": user_message_content})
        )
    return messages


__all__ = [
    "DEFAULT_SYSTEM_PROMPT_TEMPLATE_PLAIN",
    "DEFAULT_SYSTEM_PROMPT_TEMPLATE_XML",
    "DEFAULT_USER_PROMPT_TEMPLATE",
    "HISTORY_PARAM_NAMES",
    "build_chat_messages",
    "build_chat_system_prompt",
    "build_chat_user_message_content",
    "_extract_explicit_chat_user_message",
    "build_function_transcript_seed",
    "build_parameter_type_descriptions",
    "build_return_type_description",
    "build_text_messages",
    "extract_conversation_history",
    "extract_history_system_prompt",
    "extract_parameter_type_hints",
    "filter_history_messages",
    "is_complex_return_type",
    "process_docstring_template",
]
