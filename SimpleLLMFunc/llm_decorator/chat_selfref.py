from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from SimpleLLMFunc.hooks.events import ReactEndEvent
from SimpleLLMFunc.llm_decorator.chat_types import ToolkitList
from SimpleLLMFunc.llm_decorator.prompt_contract import HISTORY_PARAM_NAMES
from SimpleLLMFunc.llm_decorator.utils import remove_tool_best_practices_prompt_block
from SimpleLLMFunc.runtime.selfref.context_ops import parse_data_from_selfref
from SimpleLLMFunc.runtime.selfref.session import SelfRefSession
from SimpleLLMFunc.runtime.selfref.state import (
    MemoryHistory,
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SelfReference,
    _coerce_history_list,
)
from SimpleLLMFunc.type import MessageList

_RUNTIME_PRIMITIVE_PROMPT_BLOCK_START = "<runtime_primitive_contract>"
_RUNTIME_PRIMITIVE_PROMPT_BLOCK_END = "</runtime_primitive_contract>"
_LEGACY_SELF_REFERENCE_PROMPT_BLOCK_START = "[SelfReference Memory Contract]"
_LEGACY_SELF_REFERENCE_PROMPT_BLOCK_END = "[/SelfReference Memory Contract]"
_MUST_PRINCIPLES_PROMPT_BLOCK_START = "<must_principles>"
_MUST_PRINCIPLES_PROMPT_BLOCK_END = "</must_principles>"


def extract_raw_history_reference(
    arguments: dict[str, Any],
) -> Optional[MemoryHistory]:
    """Extract the original history list object from bound call arguments."""

    for history_param_name in HISTORY_PARAM_NAMES:
        if history_param_name not in arguments:
            continue

        history = arguments[history_param_name]
        if history is None:
            return None

        if isinstance(history, list) and all(
            isinstance(item, dict) for item in history
        ):
            return cast(MemoryHistory, history)

        return None

    return None


def set_history_argument(
    arguments: dict[str, Any],
    history: MemoryHistory,
) -> bool:
    """Inject a history snapshot into bound arguments when history param exists."""

    for history_param_name in HISTORY_PARAM_NAMES:
        if history_param_name in arguments:
            arguments[history_param_name] = list(history)
            return True
    return False


def resolve_self_reference_key(
    explicit_key: Optional[str],
    func_name: str,
) -> str:
    if explicit_key is None:
        return func_name

    normalized = explicit_key.strip()
    if not normalized:
        raise ValueError("self_reference_key must be a non-empty string")
    return normalized


def resolve_runtime_self_reference_key(
    *,
    explicit_key: Optional[str],
    func_name: str,
    template_params: Optional[Dict[str, Any]],
) -> str:
    runtime_self_reference_key = explicit_key
    if template_params is not None:
        override_key = template_params.get(SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM)
        if override_key is not None:
            if not isinstance(override_key, str):
                raise ValueError("self_reference key override must be a non-empty string")
            runtime_self_reference_key = override_key

    return resolve_self_reference_key(runtime_self_reference_key, func_name)


def remove_prompt_block(system_prompt: str, start_marker: str, end_marker: str) -> str:
    cleaned_prompt = system_prompt

    while True:
        start_index = cleaned_prompt.find(start_marker)
        if start_index < 0:
            break

        end_index = cleaned_prompt.find(end_marker, start_index)
        if end_index < 0:
            cleaned_prompt = cleaned_prompt[:start_index]
            break

        cleaned_prompt = (
            cleaned_prompt[:start_index] + cleaned_prompt[end_index + len(end_marker) :]
        )

    return cleaned_prompt


def remove_runtime_primitive_prompt_block(system_prompt: str) -> str:
    cleaned_prompt = system_prompt
    for start_marker, end_marker in (
        (
            _RUNTIME_PRIMITIVE_PROMPT_BLOCK_START,
            _RUNTIME_PRIMITIVE_PROMPT_BLOCK_END,
        ),
        (
            _LEGACY_SELF_REFERENCE_PROMPT_BLOCK_START,
            _LEGACY_SELF_REFERENCE_PROMPT_BLOCK_END,
        ),
    ):
        cleaned_prompt = remove_prompt_block(
            cleaned_prompt,
            start_marker,
            end_marker,
        )

    return cleaned_prompt.strip()


def remove_must_principles_prompt_block(system_prompt: str) -> str:
    cleaned_prompt = remove_prompt_block(
        system_prompt,
        _MUST_PRINCIPLES_PROMPT_BLOCK_START,
        _MUST_PRINCIPLES_PROMPT_BLOCK_END,
    )
    return cleaned_prompt.strip()


def remove_injected_prompt_blocks(system_prompt: str) -> str:
    cleaned_prompt = remove_runtime_primitive_prompt_block(system_prompt)
    cleaned_prompt = remove_tool_best_practices_prompt_block(cleaned_prompt)
    cleaned_prompt = remove_must_principles_prompt_block(cleaned_prompt)
    return cleaned_prompt.strip()


def build_must_principles_prompt_block() -> str:
    lines = [
        _MUST_PRINCIPLES_PROMPT_BLOCK_START,
        "<rule>Invoke tools through native structured tool_calls / function-calling fields.</rule>",
        "<rule>Use assistant content for natural-language reasoning and final responses.</rule>",
        "<rule>Keep tool invocation payloads in the native tool channel.</rule>",
        _MUST_PRINCIPLES_PROMPT_BLOCK_END,
    ]
    return "\n".join(lines)


def extract_first_system_prompt_from_messages(messages: MessageList) -> Optional[str]:
    for message in messages:
        if message.get("role") != "system":
            continue

        content = message.get("content")
        if isinstance(content, str):
            return content

    return None


def seed_self_reference_system_prompt_if_missing(
    self_reference: SelfReference,
    memory_key: str,
    messages: MessageList,
) -> None:
    if self_reference.get_system_prompt(memory_key) is not None:
        return

    system_prompt = extract_first_system_prompt_from_messages(messages)
    if system_prompt is None:
        return

    # Keep memory store focused on durable prompt content. If the system prompt
    # already contains auto-injected tool/runtime guidance blocks, store a
    # cleaned version and let llm_chat inject guidance again per turn.
    cleaned_system_prompt = remove_injected_prompt_blocks(system_prompt)
    system_prompt_for_store = cleaned_system_prompt or system_prompt

    current_history: MemoryHistory = self_reference.snapshot_history(memory_key)
    seeded_history: MemoryHistory = [
        {"role": "system", "content": system_prompt_for_store},
        *current_history,
    ]
    self_reference.replace_history(
        key=memory_key,
        messages=seeded_history,
        strict=False,
    )


def react_end_event_has_fork_origin(event: ReactEndEvent, origin: Any) -> bool:
    origin_fork_id = getattr(origin, "fork_id", None)
    if isinstance(origin_fork_id, str) and origin_fork_id:
        return True

    event_extra = getattr(event, "extra", None)
    if not isinstance(event_extra, dict):
        return False

    raw_origin = event_extra.get("origin")
    if not isinstance(raw_origin, dict):
        return False

    raw_fork_id = raw_origin.get("fork_id")
    return isinstance(raw_fork_id, str) and bool(raw_fork_id)


def finalize_self_reference_history(
    self_reference: SelfReference,
    memory_key: str,
    history: MemoryHistory,
    *,
    baseline_history_count: int,
    base_system_prompt: str,
) -> MemoryHistory:
    _ = base_system_prompt
    resolved_history = cast(MemoryHistory, _coerce_history_list(history))
    resolved_source = parse_data_from_selfref(resolved_history)
    if (
        resolved_source.experiences
        or resolved_source.summary is not None
        or resolved_source.summary_message is not None
    ):
        return self_reference.store_history(memory_key, resolved_history)

    self_reference.consume_destructive_history_mutation(memory_key)
    resolved_history = self_reference.merge_turn_history(
        key=memory_key,
        baseline_history_count=baseline_history_count,
        updated_history=resolved_history,
        commit=True,
    )
    return self_reference.store_history(memory_key, resolved_history)


def create_selfref_session(
    *,
    backend: SelfReference,
    memory_key: str,
    template_params: Optional[Dict[str, Any]],
    runtime_toolkit: Optional[ToolkitList],
    raw_history_reference: Optional[MemoryHistory],
    agent_instance: Any,
    baseline_history_count: int,
) -> SelfRefSession:
    session = SelfRefSession(
        backend=backend,
        memory_key=memory_key,
        template_params=template_params,
        runtime_toolkit=runtime_toolkit,
        raw_history_reference=raw_history_reference,
        agent_instance=agent_instance,
    )
    session.history_authority = (
        "external"
        if raw_history_reference is not None
        else ("selfref" if backend.has_history(memory_key) else "seed")
    )
    session.baseline_history_count = baseline_history_count
    return session


# Backward-compatible internal aliases for existing tests/users that imported
# these helpers from llm_chat_decorator.
_extract_raw_history_reference = extract_raw_history_reference
_set_history_argument = set_history_argument
_resolve_self_reference_key = resolve_self_reference_key
_remove_prompt_block = remove_prompt_block
_remove_runtime_primitive_prompt_block = remove_runtime_primitive_prompt_block
_remove_injected_prompt_blocks = remove_injected_prompt_blocks
_build_must_principles_prompt_block = build_must_principles_prompt_block
_remove_must_principles_prompt_block = remove_must_principles_prompt_block
_extract_first_system_prompt_from_messages = extract_first_system_prompt_from_messages
_seed_self_reference_system_prompt_if_missing = seed_self_reference_system_prompt_if_missing
_react_end_event_has_fork_origin = react_end_event_has_fork_origin
_finalize_self_reference_history = finalize_self_reference_history

__all__ = [
    "create_selfref_session",
    "extract_raw_history_reference",
    "finalize_self_reference_history",
    "react_end_event_has_fork_origin",
    "resolve_runtime_self_reference_key",
    "resolve_self_reference_key",
    "seed_self_reference_system_prompt_if_missing",
    "set_history_argument",
    "_build_must_principles_prompt_block",
    "_extract_first_system_prompt_from_messages",
    "_extract_raw_history_reference",
    "_finalize_self_reference_history",
    "_react_end_event_has_fork_origin",
    "_remove_injected_prompt_blocks",
    "_remove_must_principles_prompt_block",
    "_remove_prompt_block",
    "_remove_runtime_primitive_prompt_block",
    "_resolve_self_reference_key",
    "_seed_self_reference_system_prompt_if_missing",
    "_set_history_argument",
]
