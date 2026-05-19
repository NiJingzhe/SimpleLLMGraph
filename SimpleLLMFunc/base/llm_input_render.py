"""Ephemeral LLM-input rendering on top of durable compiled context."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from SimpleLLMFunc.base.context_compile import clone_messages
from SimpleLLMFunc.llm_decorator.utils.tools import build_tool_best_practices_prompt_block
from SimpleLLMFunc.type.message import NormalizedMessageList, NormalizedMessageParam

_MUST_PRINCIPLES_PROMPT_BLOCK_START = "<must_principles>"
_MUST_PRINCIPLES_PROMPT_BLOCK_END = "</must_principles>"


def _remove_prompt_block(text: str, start_tag: str, end_tag: str) -> str:
    cleaned = text
    while True:
        start_index = cleaned.find(start_tag)
        if start_index < 0:
            break

        end_index = cleaned.find(end_tag, start_index)
        if end_index < 0:
            cleaned = cleaned[:start_index]
            break

        cleaned = cleaned[:start_index] + cleaned[end_index + len(end_tag) :]

    return cleaned.strip()


def _build_must_principles_prompt_block() -> str:
    lines = [
        _MUST_PRINCIPLES_PROMPT_BLOCK_START,
        "<rule>Invoke tools through native structured tool_calls / function-calling fields.</rule>",
        "<rule>Use assistant content for natural-language reasoning and final responses.</rule>",
        "<rule>Keep tool invocation payloads in the native tool channel.</rule>",
        _MUST_PRINCIPLES_PROMPT_BLOCK_END,
    ]
    return "\n".join(lines)


def _remove_must_principles_prompt_block(system_prompt: str) -> str:
    return _remove_prompt_block(
        system_prompt,
        _MUST_PRINCIPLES_PROMPT_BLOCK_START,
        _MUST_PRINCIPLES_PROMPT_BLOCK_END,
    )


def _render_system_prompt(
    base_prompt: str,
    *,
    tool_prompt_specs: Optional[List[Dict[str, Any]]] = None,
    include_must_principles: bool = False,
) -> str:
    cleaned_base = _remove_must_principles_prompt_block(base_prompt)
    sections: List[str] = []

    tool_prompt_block = build_tool_best_practices_prompt_block(tool_prompt_specs or [])
    if tool_prompt_block:
        sections.append(tool_prompt_block)

    if cleaned_base:
        sections.append(cleaned_base)

    if include_must_principles:
        sections.append(_build_must_principles_prompt_block())

    return "\n\n".join(section for section in sections if section).strip()


def render_llm_input_messages(
    messages: NormalizedMessageList,
    *,
    tool_prompt_specs: Optional[List[Dict[str, Any]]] = None,
    include_must_principles: bool = False,
) -> NormalizedMessageList:
    rendered = clone_messages(messages)
    if not rendered:
        return rendered

    for index, message in enumerate(rendered):
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        rendered[index] = cast(
            NormalizedMessageParam,
            {
                **message,
                "content": _render_system_prompt(
                    content,
                    tool_prompt_specs=tool_prompt_specs,
                    include_must_principles=include_must_principles,
                ),
            },
        )
        return rendered

    rendered.insert(
        0,
        cast(
            NormalizedMessageParam,
            {
                "role": "system",
                "content": _render_system_prompt(
                    "",
                    tool_prompt_specs=tool_prompt_specs,
                    include_must_principles=include_must_principles,
                ),
            },
        ),
    )
    return rendered


__all__ = ["render_llm_input_messages"]
