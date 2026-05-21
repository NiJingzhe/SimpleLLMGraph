from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, cast

from SimpleLLMFunc.base.messages import (
    validate_message_shape,
    validate_tool_linkage,
)
from SimpleLLMFunc.base.types import DataFromSelfRef
from SimpleLLMFunc.type.message import NormalizedMessageList, NormalizedMessageParam
from SimpleLLMFunc.runtime.selfref.context_ops import (
    build_context_messages_from_selfref_data,
    build_context_messages_from_state_data,
    canonicalize_context_messages,
    clone_messages,
    extract_latest_system_prompt,
    normalize_context_summary_payload,
    normalize_experience_text,
    parse_context_messages,
    parse_data_from_selfref,
    render_context_compaction_summary,
)

MemoryHistory = List[Dict[str, Any]]
HISTORY_PARAM_NAMES = ("history", "chat_history")


def normalize_key(key: str) -> str:
    if not isinstance(key, str):
        raise ValueError("key must be a non-empty string")
    normalized = key.strip()
    if not normalized:
        raise ValueError("key must be a non-empty string")
    return normalized


def validate_history_for_memory_methods(messages: MemoryHistory) -> None:
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"history item at index {index} must be a dict")
        validate_message_shape(cast(NormalizedMessageParam, message), index)

    validate_tool_linkage(cast(NormalizedMessageList, messages))


def coerce_history_list(history: List[Any]) -> MemoryHistory:
    if not isinstance(history, list):
        raise ValueError("history must be List[Dict[str, Any]]")
    if not all(isinstance(item, dict) for item in history):
        raise ValueError("history must be List[Dict[str, Any]]")
    return clone_messages(cast(MemoryHistory, history))


def filter_non_system_messages(messages: MemoryHistory) -> MemoryHistory:
    filtered: MemoryHistory = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if "role" in message and "content" in message:
            filtered.append(copy.deepcopy(message))
    return filtered


class SelfReferenceContextMemoryMixin:
    """Context-message and direct memory editing API for ``SelfReference``."""

    def _next_experience_id(self) -> str:
        with self._lock:
            self._experience_id_counter += 1
            return f"exp_{self._experience_id_counter}"

    def parse_context_state(self, key: str) -> Dict[str, Any]:
        normalized_key = normalize_key(key)
        source = self.snapshot_selfref_source(normalized_key)
        context_state = {
            "base_system_prompt": source.base_system_prompt,
            "experiences": copy.deepcopy(source.experiences),
            "summary": copy.deepcopy(source.summary),
            "summary_message": copy.deepcopy(source.summary_message),
            "working_messages": clone_messages(
                cast(MemoryHistory, source.working_messages)
            ),
        }
        context_state["messages"] = build_context_messages_from_selfref_data(source)
        return context_state

    def compile_context_messages(
        self,
        key: str,
        *,
        include_summary: bool = True,
    ) -> MemoryHistory:
        normalized_key = normalize_key(key)
        source = self.snapshot_selfref_source(normalized_key)
        if include_summary:
            return build_context_messages_from_selfref_data(source)

        return build_context_messages_from_selfref_data(
            DataFromSelfRef(
                base_system_prompt=source.base_system_prompt,
                experiences=copy.deepcopy(source.experiences),
                summary=None,
                summary_message=None,
                working_messages=cast(
                    NormalizedMessageList,
                    clone_messages(cast(MemoryHistory, source.working_messages)),
                ),
            )
        )

    def set_context_messages(self, key: str, messages: List[Dict[str, Any]]) -> None:
        normalized_key = normalize_key(key)
        compiled = self._coerce_compiled_context_messages(messages)
        active_messages = self._get_active_history_target(normalized_key)
        if active_messages is not None:
            active_messages.clear()
            active_messages.extend(clone_messages(compiled))
            self._store.update_existing(
                normalized_key,
                cast(MemoryHistory, active_messages),
            )
            return

        self._store.update_existing(normalized_key, compiled)

    def _coerce_compiled_context_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        validate_working_linkage: bool = True,
    ) -> MemoryHistory:
        normalized_messages = coerce_history_list(messages)
        for index, message in enumerate(normalized_messages):
            if not isinstance(message, dict):
                raise ValueError(f"context message at index {index} must be a dict")
            validate_message_shape(cast(NormalizedMessageParam, message), index)

        working_messages = cast(
            MemoryHistory,
            parse_context_messages(normalized_messages)["working_messages"],
        )
        if validate_working_linkage:
            validate_tool_linkage(cast(NormalizedMessageList, working_messages))
        return canonicalize_context_messages(normalized_messages)

    def snapshot_context_messages(self, key: str) -> MemoryHistory:
        normalized_key = normalize_key(key)
        active_messages = self._snapshot_active_history(normalized_key)
        if active_messages is not None:
            return active_messages
        return self.compile_context_messages(normalized_key)

    def list_context_experiences(self, key: str) -> List[Dict[str, str]]:
        context_state = self.parse_context_state(key)
        return copy.deepcopy(cast(List[Dict[str, str]], context_state["experiences"]))

    def remember_experience(self, key: str, text: str) -> Dict[str, str]:
        normalized_key = normalize_key(key)
        normalized_text = normalize_experience_text(text)
        active_messages = self._get_active_history_target(normalized_key)
        if active_messages is not None:
            self.queue_context_mutation(
                normalized_key,
                {"type": "experience_remember", "text": normalized_text},
            )
            return {"id": f"pending::{normalized_text}", "text": normalized_text}

        context_state = self.parse_context_state(normalized_key)
        experiences = cast(List[Dict[str, str]], context_state["experiences"])

        for item in experiences:
            if normalize_experience_text(item["text"]) == normalized_text:
                return copy.deepcopy(item)

        new_item = {"id": self._next_experience_id(), "text": normalized_text}
        experiences.append(new_item)
        self.set_context_messages(
            normalized_key,
            self._build_context_messages_from_state(context_state),
        )
        return copy.deepcopy(new_item)

    def forget_experience(self, key: str, experience_id: str) -> bool:
        normalized_key = normalize_key(key)
        if not isinstance(experience_id, str) or not experience_id.strip():
            raise ValueError("experience_id must be a non-empty string")

        active_messages = self._get_active_history_target(normalized_key)
        if active_messages is not None:
            self.queue_context_mutation(
                normalized_key,
                {"type": "experience_forget", "experience_id": experience_id.strip()},
            )
            return True

        context_state = self.parse_context_state(normalized_key)
        experiences = cast(List[Dict[str, str]], context_state["experiences"])
        retained = [
            item for item in experiences if item.get("id") != experience_id.strip()
        ]
        if len(retained) == len(experiences):
            return False

        context_state["experiences"] = retained
        self.set_context_messages(
            normalized_key,
            self._build_context_messages_from_state(context_state),
        )
        return True

    def queue_context_compaction(
        self,
        key: str,
        summary: Dict[str, Any],
        *,
        remember: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        normalized_key = normalize_key(key)
        normalized_summary = normalize_context_summary_payload(summary)
        normalized_remember = [
            normalize_experience_text(item)
            for item in (remember or [])
            if str(item).strip()
        ]
        payload = {
            "summary": normalized_summary,
            "remember": normalized_remember,
            "rendered_summary": render_context_compaction_summary(normalized_summary),
        }
        return self._mutations.queue_context_compaction(normalized_key, payload)

    def commit_pending_compaction(
        self,
        key: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[MemoryHistory]:
        normalized_key = normalize_key(key)
        pending = self.consume_pending_compaction(normalized_key)
        if pending is None:
            return None

        base_messages = (
            self._coerce_compiled_context_messages(messages)
            if messages is not None
            else self.snapshot_context_messages(normalized_key)
        )
        context_state = parse_context_messages(base_messages)
        for item in cast(List[str], pending["remember"]):
            existing_texts = {
                normalize_experience_text(experience["text"])
                for experience in cast(
                    List[Dict[str, str]], context_state["experiences"]
                )
            }
            normalized_text = normalize_experience_text(item)
            if normalized_text in existing_texts:
                continue
            cast(List[Dict[str, str]], context_state["experiences"]).append(
                {"id": self._next_experience_id(), "text": normalized_text}
            )

        context_state["summary"] = copy.deepcopy(pending["summary"])
        context_state["summary_message"] = {
            "role": "assistant",
            "content": cast(str, pending["rendered_summary"]),
        }
        context_state["working_messages"] = []

        compiled = self._build_context_messages_from_state(context_state)
        self.mark_destructive_history_mutation(normalized_key)
        self.set_context_messages(normalized_key, compiled)
        return compiled

    def _build_context_messages_from_state(
        self, context_state: Dict[str, Any]
    ) -> MemoryHistory:
        return build_context_messages_from_state_data(context_state)

    def filtered_history_count(self, key: str) -> int:
        messages = self.snapshot_history(key)
        return len(filter_non_system_messages(messages))

    def merge_turn_history(
        self,
        key: str,
        baseline_history_count: int,
        updated_history: List[Dict[str, Any]],
        commit: bool = False,
    ) -> MemoryHistory:
        normalized_key = normalize_key(key)
        updated_history_dicts = coerce_history_list(updated_history)

        runtime_messages = self._store.get_history_or_empty(normalized_key)

        runtime_non_system = filter_non_system_messages(runtime_messages)
        runtime_system_prompt = extract_latest_system_prompt(runtime_messages)

        if baseline_history_count < 0:
            baseline_history_count = 0

        merged: MemoryHistory = []

        updated_system_prompt: Optional[str] = None
        if updated_history_dicts:
            first = updated_history_dicts[0]
            first_content = first.get("content")
            if first.get("role") == "system" and isinstance(first_content, str):
                updated_system_prompt = first_content

        effective_system_prompt = runtime_system_prompt or updated_system_prompt
        if effective_system_prompt is not None:
            merged.append({"role": "system", "content": effective_system_prompt})

        tail_start = baseline_history_count
        if updated_system_prompt is not None:
            tail_start += 1

        tail_start = min(tail_start, len(updated_history_dicts))
        merged.extend(runtime_non_system)
        merged.extend(clone_messages(updated_history_dicts[tail_start:]))

        if commit:
            canonicalized = coerce_history_list(canonicalize_context_messages(merged))
            self._store.update_existing(normalized_key, canonicalized)

        return merged

    def replace_history(
        self,
        key: str,
        messages: List[Dict[str, Any]],
        strict: bool = False,
    ) -> None:
        normalized_key = normalize_key(key)
        normalized_messages = coerce_history_list(messages)
        if strict:
            validate_history_for_memory_methods(normalized_messages)
        normalized_messages = self._coerce_compiled_context_messages(
            normalized_messages
        )
        active_messages = self._get_active_history_target(normalized_key)
        if active_messages is not None:
            self.mark_destructive_history_mutation(normalized_key)
            active_messages.clear()
            active_messages.extend(clone_messages(normalized_messages))
            self._store.update_existing(
                normalized_key,
                cast(MemoryHistory, active_messages),
            )
            return

        self._store.update_existing(normalized_key, normalized_messages)

    def append_message(self, key: str, message: Dict[str, Any]) -> None:
        self._mutate_messages(key, lambda msgs: msgs.append(copy.deepcopy(message)))

    def insert_message(self, key: str, index: int, message: Dict[str, Any]) -> None:
        self._mutate_messages(
            key,
            lambda msgs: msgs.insert(index, copy.deepcopy(message)),
        )

    def update_message(self, key: str, index: int, message: Dict[str, Any]) -> None:
        def mutate(messages: MemoryHistory) -> None:
            messages[index] = copy.deepcopy(message)

        self._mutate_messages(key, mutate)

    def delete_message(self, key: str, index: int) -> None:
        self.mark_destructive_history_mutation(key)

        def mutate(messages: MemoryHistory) -> None:
            messages.pop(index)

        self._mutate_messages(key, mutate)

    def get_system_prompt(self, key: str) -> Optional[str]:
        messages = self.snapshot_history(key)
        return extract_latest_system_prompt(messages)

    def set_system_prompt(self, key: str, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("system prompt text must be a string")

        self.mark_destructive_history_mutation(key)

        def mutate(messages: MemoryHistory) -> None:
            non_system = [msg for msg in messages if msg.get("role") != "system"]
            messages.clear()
            messages.append({"role": "system", "content": text})
            messages.extend(non_system)

        self._mutate_messages(key, mutate)

    def append_system_prompt(self, key: str, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("system prompt text must be a string")

        current = self.get_system_prompt(key)
        if current:
            updated = f"{current}\n{text}"
        else:
            updated = text
        self.set_system_prompt(key, updated)

    def _mutate_messages(
        self,
        key: str,
        mutator: Callable[[MemoryHistory], None],
    ) -> None:
        normalized_key = normalize_key(key)

        active_messages = self._get_active_history_target(normalized_key)
        if active_messages is not None:
            working_messages = clone_messages(active_messages)
            mutator(working_messages)
            working_messages = clone_messages(working_messages)
            validate_history_for_memory_methods(working_messages)
            active_messages.clear()
            active_messages.extend(working_messages)
            self._store.update_existing(
                normalized_key,
                cast(MemoryHistory, active_messages),
            )
            return

        messages = self._store.get_history(normalized_key)

        mutator(messages)
        messages = clone_messages(messages)
        validate_history_for_memory_methods(messages)
        self._store.update_existing(normalized_key, messages)


__all__ = [
    "HISTORY_PARAM_NAMES",
    "MemoryHistory",
    "SelfReferenceContextMemoryMixin",
    "coerce_history_list",
    "filter_non_system_messages",
    "normalize_key",
    "validate_history_for_memory_methods",
]
