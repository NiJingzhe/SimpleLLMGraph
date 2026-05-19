from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, cast

from SimpleLLMFunc.base.types import DataFromSelfRef
from SimpleLLMFunc.type.message import NormalizedMessageList

MemoryHistory = List[Dict[str, Any]]

_EXPERIENCE_BLOCK_START = "<experience>"
_EXPERIENCE_BLOCK_END = "</experience>"
_CONTEXT_COMPACTION_SUMMARY_START = "<context_compaction_summary>"
_CONTEXT_COMPACTION_SUMMARY_END = "</context_compaction_summary>"
_TOOL_BEST_PRACTICES_START = "<tool_best_practices>"
_TOOL_BEST_PRACTICES_END = "</tool_best_practices>"
_RUNTIME_PRIMITIVE_CONTRACT_START = "<runtime_primitive_contract>"
_RUNTIME_PRIMITIVE_CONTRACT_END = "</runtime_primitive_contract>"
_LEGACY_SELFREF_CONTRACT_START = "[SelfReference Memory Contract]"
_LEGACY_SELFREF_CONTRACT_END = "[/SelfReference Memory Contract]"
_MUST_PRINCIPLES_START = "<must_principles>"
_MUST_PRINCIPLES_END = "</must_principles>"
_SUMMARY_SECTION_TITLES = {
    "goal": "Goal",
    "instruction": "Instruction",
    "discoveries": "Discoveries",
    "completed": "Completed",
    "current_status": "Current Status",
    "likely_next_work": "Likely next work",
    "relevant_files_directories": "Relevant files/directories",
}
_LIST_SUMMARY_KEYS = {
    "discoveries",
    "completed",
    "likely_next_work",
    "relevant_files_directories",
}
_TRANSIENT_MESSAGE_FIELDS = {"reasoning_details"}


def clone_messages(messages: MemoryHistory) -> MemoryHistory:
    cloned: MemoryHistory = []
    for message in messages:
        sanitized = copy.deepcopy(message)
        for field in _TRANSIENT_MESSAGE_FIELDS:
            sanitized.pop(field, None)
        cloned.append(sanitized)
    return cloned


def extract_latest_system_prompt(messages: MemoryHistory) -> Optional[str]:
    for message in reversed(messages):
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None


def remove_tagged_block(text: str, start_tag: str, end_tag: str) -> str:
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


def normalize_experience_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("experience text must be a string")

    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("experience text must be a non-empty string")
    return normalized


def remove_framework_injected_prompt_blocks(system_prompt: str) -> str:
    cleaned = system_prompt
    for start_tag, end_tag in (
        (_TOOL_BEST_PRACTICES_START, _TOOL_BEST_PRACTICES_END),
        (_RUNTIME_PRIMITIVE_CONTRACT_START, _RUNTIME_PRIMITIVE_CONTRACT_END),
        (_LEGACY_SELFREF_CONTRACT_START, _LEGACY_SELFREF_CONTRACT_END),
        (_MUST_PRINCIPLES_START, _MUST_PRINCIPLES_END),
    ):
        cleaned = remove_tagged_block(cleaned, start_tag, end_tag)
    return cleaned.strip()


def parse_experience_block(block_text: str) -> List[Dict[str, str]]:
    experiences: List[Dict[str, str]] = []

    for raw_line in block_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^-\s*\[(?P<id>[^\]]+)\]\s+(?P<text>.+)$", line)
        if match is None:
            continue
        experience_id = match.group("id").strip()
        experience_text = normalize_experience_text(match.group("text"))
        if not experience_id:
            continue
        experiences.append({"id": experience_id, "text": experience_text})

    return experiences


def split_system_prompt_experiences(
    system_prompt: str,
) -> tuple[str, List[Dict[str, str]]]:
    if not isinstance(system_prompt, str):
        return "", []

    start_index = system_prompt.find(_EXPERIENCE_BLOCK_START)
    if start_index < 0:
        return system_prompt.strip(), []

    end_index = system_prompt.find(_EXPERIENCE_BLOCK_END, start_index)
    if end_index < 0:
        return remove_tagged_block(
            system_prompt,
            _EXPERIENCE_BLOCK_START,
            _EXPERIENCE_BLOCK_END,
        ), []

    before = system_prompt[:start_index].strip()
    after = system_prompt[end_index + len(_EXPERIENCE_BLOCK_END) :].strip()
    block_text = system_prompt[
        start_index + len(_EXPERIENCE_BLOCK_START) : end_index
    ].strip()

    base_parts = [part for part in (before, after) if part]
    base_prompt = "\n\n".join(base_parts).strip()
    return base_prompt, parse_experience_block(block_text)


def render_system_prompt_with_experiences(
    base_prompt: str,
    experiences: List[Dict[str, str]],
) -> str:
    base = base_prompt.strip()
    if not experiences:
        return base

    experience_lines = [
        _EXPERIENCE_BLOCK_START,
        *[
            f"- [{item['id']}] {normalize_experience_text(item['text'])}"
            for item in experiences
        ],
        _EXPERIENCE_BLOCK_END,
    ]
    experience_block = "\n".join(experience_lines)

    if not base:
        return experience_block
    return f"{base}\n\n{experience_block}"


def normalize_summary_text_field(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def normalize_summary_list_field(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list[str]")

    normalized_items: List[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string")
        normalized = item.strip()
        if not normalized:
            continue
        normalized_items.append(normalized)

    return normalized_items


def normalize_context_summary_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}

    for key in (
        "goal",
        "instruction",
        "current_status",
    ):
        normalized[key] = normalize_summary_text_field(payload.get(key), key)

    for key in (
        "discoveries",
        "completed",
        "likely_next_work",
        "relevant_files_directories",
    ):
        normalized[key] = normalize_summary_list_field(payload.get(key), key)

    return normalized


def render_context_compaction_summary(summary: Dict[str, Any]) -> str:
    normalized = normalize_context_summary_payload(summary)
    lines = [_CONTEXT_COMPACTION_SUMMARY_START]

    for key in (
        "goal",
        "instruction",
        "discoveries",
        "completed",
        "current_status",
        "likely_next_work",
        "relevant_files_directories",
    ):
        lines.append(f"## {_SUMMARY_SECTION_TITLES[key]}")
        value = normalized[key]
        if key in _LIST_SUMMARY_KEYS:
            for item in cast(List[str], value):
                lines.append(f"- {item}")
        else:
            lines.append(cast(str, value))
        lines.append("")

    while lines and not lines[-1]:
        lines.pop()

    lines.append(_CONTEXT_COMPACTION_SUMMARY_END)
    return "\n".join(lines)


def parse_context_compaction_summary(content: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(content, str):
        return None

    stripped = content.strip()
    if not stripped.startswith(_CONTEXT_COMPACTION_SUMMARY_START):
        return None
    if not stripped.endswith(_CONTEXT_COMPACTION_SUMMARY_END):
        return None

    body = stripped[
        len(_CONTEXT_COMPACTION_SUMMARY_START) : -len(_CONTEXT_COMPACTION_SUMMARY_END)
    ].strip()
    sections: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    title_to_key = {value: key for key, value in _SUMMARY_SECTION_TITLES.items()}

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith("## "):
            title = stripped_line[3:].strip()
            current_key = title_to_key.get(title)
            if current_key is not None:
                sections[current_key] = []
            continue
        if current_key is None:
            continue
        sections[current_key].append(stripped_line)

    required_keys = set(_SUMMARY_SECTION_TITLES.keys())
    if not required_keys.issubset(sections.keys()):
        return None

    parsed: Dict[str, Any] = {}
    for key, lines in sections.items():
        if key in _LIST_SUMMARY_KEYS:
            parsed[key] = [
                line[2:].strip()
                for line in lines
                if line.startswith("- ") and line[2:].strip()
            ]
        else:
            parsed[key] = "\n".join(lines).strip()

    try:
        return normalize_context_summary_payload(parsed)
    except ValueError:
        return None


def parse_context_messages(messages: MemoryHistory) -> Dict[str, Any]:
    system_prompt = remove_framework_injected_prompt_blocks(
        extract_latest_system_prompt(messages) or ""
    )
    base_system_prompt, experiences = split_system_prompt_experiences(system_prompt)

    summary: Optional[Dict[str, Any]] = None
    summary_message: Optional[Dict[str, Any]] = None
    working_messages: MemoryHistory = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue

        if summary is None and role == "assistant":
            parsed_summary = parse_context_compaction_summary(message.get("content"))
            if parsed_summary is not None:
                summary = parsed_summary
                summary_message = copy.deepcopy(message)
                continue

        working_messages.append(copy.deepcopy(message))

    return {
        "base_system_prompt": base_system_prompt,
        "experiences": experiences,
        "summary": summary,
        "summary_message": summary_message,
        "working_messages": working_messages,
    }


def parse_data_from_selfref(messages: MemoryHistory) -> DataFromSelfRef:
    parsed = parse_context_messages(messages)
    return DataFromSelfRef(
        base_system_prompt=cast(str, parsed["base_system_prompt"]),
        experiences=cast(List[Dict[str, str]], parsed["experiences"]),
        summary=cast(Optional[Dict[str, Any]], parsed["summary"]),
        summary_message=cast(Optional[Dict[str, Any]], parsed["summary_message"]),
        working_messages=cast(NormalizedMessageList, parsed["working_messages"]),
    )


def build_context_messages_from_state_data(
    context_state: Dict[str, Any],
) -> MemoryHistory:
    compiled: MemoryHistory = []
    rendered_system_prompt = render_system_prompt_with_experiences(
        cast(str, context_state.get("base_system_prompt") or ""),
        cast(List[Dict[str, str]], context_state.get("experiences") or []),
    )
    if rendered_system_prompt:
        compiled.append({"role": "system", "content": rendered_system_prompt})

    summary = context_state.get("summary")
    if isinstance(summary, dict):
        compiled.append(
            {
                "role": "assistant",
                "content": render_context_compaction_summary(summary),
            }
        )
    elif isinstance(context_state.get("summary_message"), dict):
        compiled.append(
            copy.deepcopy(cast(Dict[str, Any], context_state["summary_message"]))
        )

    compiled.extend(
        clone_messages(cast(MemoryHistory, context_state.get("working_messages") or []))
    )
    return compiled


def build_context_messages_from_selfref_data(data: DataFromSelfRef) -> MemoryHistory:
    return build_context_messages_from_state_data(
        {
            "base_system_prompt": data.base_system_prompt,
            "experiences": copy.deepcopy(data.experiences),
            "summary": copy.deepcopy(data.summary),
            "summary_message": copy.deepcopy(data.summary_message),
            "working_messages": clone_messages(cast(MemoryHistory, data.working_messages)),
        }
    )


def canonicalize_context_messages(messages: MemoryHistory) -> MemoryHistory:
    return build_context_messages_from_selfref_data(parse_data_from_selfref(messages))


__all__ = [
    "MemoryHistory",
    "build_context_messages_from_selfref_data",
    "build_context_messages_from_state_data",
    "canonicalize_context_messages",
    "clone_messages",
    "extract_latest_system_prompt",
    "normalize_experience_text",
    "parse_data_from_selfref",
    "parse_context_compaction_summary",
    "parse_context_messages",
    "remove_framework_injected_prompt_blocks",
    "render_context_compaction_summary",
    "render_system_prompt_with_experiences",
    "split_system_prompt_experiences",
]
