"""Context state and compile helpers for the new ReAct core."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Dict, List, Optional, cast

from SimpleLLMFunc.base.messages.assistant import (
    build_assistant_response_message,
    build_assistant_tool_message,
)
from SimpleLLMFunc.base.messages.validation import validate_tool_linkage
from SimpleLLMFunc.base.mutation import (
    AssistantMessageMutation,
    AssistantTruncatedMutation,
    ContextMutation,
    ContextReplaceMutation,
    ContextSummaryMutation,
    ExperienceForgetMutation,
    ExperienceRememberMutation,
    ToolCancelledMutation,
    ToolResultMutation,
    UserMessageMutation,
)
from SimpleLLMFunc.type.message import NormalizedMessageList, NormalizedMessageParam


@dataclass
class ContextState:
    messages: NormalizedMessageList
    pending_mutations: List[ContextMutation] = field(default_factory=list)


@dataclass
class CompiledContext:
    llm_messages: NormalizedMessageList
    semantic_messages: NormalizedMessageList


def clone_messages(messages: NormalizedMessageList) -> NormalizedMessageList:
    return [copy.deepcopy(message) for message in messages]


def assistant_tool_calls_are_resolved(
    messages: NormalizedMessageList,
    start_index: int,
    tool_call_ids: List[str],
) -> bool:
    pending = set(tool_call_ids)
    for message in messages[start_index + 1 :]:
        if not pending:
            return True
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str):
                pending.discard(tool_call_id)
            continue
        if role == "assistant":
            break
    return not pending


def compile_semantic_messages(messages: NormalizedMessageList) -> NormalizedMessageList:
    compiled: NormalizedMessageList = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            compiled.append(copy.deepcopy(message))
            continue
        if message.get("role") != "assistant":
            compiled.append(copy.deepcopy(message))
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            compiled.append(copy.deepcopy(message))
            continue
        tool_call_ids = [
            tool_call.get("id")
            for tool_call in tool_calls
            if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str)
        ]
        if tool_call_ids and assistant_tool_calls_are_resolved(
            messages,
            index,
            cast(List[str], tool_call_ids),
        ):
            continue
        compiled.append(copy.deepcopy(message))
    return compiled


def _build_cancelled_assistant_tool_message(
    mutation: ToolCancelledMutation,
) -> NormalizedMessageParam:
    return cast(
        NormalizedMessageParam,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": mutation.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": mutation.tool_name,
                        "arguments": "{}",
                    },
                }
            ],
        },
    )


def _build_tool_result_message(
    tool_call_id: str,
    content: str,
) -> NormalizedMessageParam:
    return cast(
        NormalizedMessageParam,
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        },
    )


def _append_tool_cancelled_mutation(
    messages: NormalizedMessageList,
    mutation: ToolCancelledMutation,
) -> None:
    reason_suffix = (
        f" Reason: {mutation.abort_reason}" if mutation.abort_reason else ""
    )
    messages.append(_build_cancelled_assistant_tool_message(mutation))
    messages.append(
        _build_tool_result_message(
            mutation.tool_call_id,
            (
                "<Tool execution cancelled by user."
                f" Tool: {mutation.tool_name}.{reason_suffix}>"
            ),
        )
    )


def _append_assistant_truncated_mutation(
    messages: NormalizedMessageList,
    mutation: AssistantTruncatedMutation,
) -> None:
    reason_suffix = (
        f" Reason: {mutation.abort_reason}" if mutation.abort_reason else ""
    )
    truncated_notice = (
        "<Truncated due to cancel from user."
        f"{reason_suffix}>"
    )
    content = mutation.partial_content + truncated_notice
    messages.append(build_assistant_response_message(content))


def apply_mutations(
    messages: NormalizedMessageList,
    mutations: List[ContextMutation],
) -> NormalizedMessageList:
    current = clone_messages(messages)
    pending_experience_remembers: List[str] = []
    pending_experience_forgets: List[str] = []

    def _apply_pending_experience_mutations() -> None:
        nonlocal current
        if not pending_experience_remembers and not pending_experience_forgets:
            return

        from SimpleLLMFunc.runtime.selfref.context_ops import (
            build_context_messages_from_state_data,
            normalize_experience_text,
            parse_context_messages,
        )

        context_state = parse_context_messages(cast(List[Dict[str, Any]], current))
        experiences = cast(List[Dict[str, str]], context_state["experiences"])

        if pending_experience_forgets:
            forget_ids = {item.strip() for item in pending_experience_forgets if item.strip()}
            experiences = [item for item in experiences if item.get("id") not in forget_ids]

        existing_texts = {
            normalize_experience_text(experience["text"]) for experience in experiences
        }
        next_exp_id = 1
        for item in experiences:
            exp_id = str(item.get("id", ""))
            if exp_id.startswith("exp_"):
                try:
                    next_exp_id = max(next_exp_id, int(exp_id.split("_", 1)[1]) + 1)
                except Exception:
                    pass

        for text in pending_experience_remembers:
            normalized = normalize_experience_text(text)
            if normalized in existing_texts:
                continue
            experiences.append({"id": f"exp_{next_exp_id}", "text": normalized})
            existing_texts.add(normalized)
            next_exp_id += 1

        context_state["experiences"] = experiences
        current = build_context_messages_from_state_data(context_state)
        pending_experience_remembers.clear()
        pending_experience_forgets.clear()

    for mutation in mutations:
        if isinstance(mutation, ContextReplaceMutation):
            _apply_pending_experience_mutations()
            current = clone_messages(cast(NormalizedMessageList, mutation.messages))
            continue
        if isinstance(mutation, ContextSummaryMutation):
            _apply_pending_experience_mutations()
            base_system = None
            if current and isinstance(current[0], dict) and current[0].get("role") == "system":
                base_system = copy.deepcopy(current[0])
            current = []
            if base_system is not None:
                current.append(base_system)
            current.append(copy.deepcopy(cast(NormalizedMessageParam, mutation.summary_message)))
            for remembered in mutation.remember:
                text = remembered.get("text") if isinstance(remembered, dict) else None
                if isinstance(text, str) and text.strip():
                    pending_experience_remembers.append(text)
            _apply_pending_experience_mutations()
            continue
        if isinstance(mutation, ExperienceRememberMutation):
            pending_experience_remembers.append(mutation.text)
            continue
        if isinstance(mutation, ExperienceForgetMutation):
            pending_experience_forgets.append(mutation.experience_id)
            continue
        if isinstance(mutation, AssistantMessageMutation):
            _apply_pending_experience_mutations()
            if mutation.tool_calls:
                current.append(
                    build_assistant_tool_message(
                        mutation.tool_calls,
                        mutation.reasoning_details or None,
                    )
                )
            elif mutation.content is not None:
                current.append(build_assistant_response_message(mutation.content))
            continue
        if isinstance(mutation, ToolResultMutation):
            _apply_pending_experience_mutations()
            current.append(
                _build_tool_result_message(
                    mutation.tool_call_id,
                    mutation.content,
                )
            )
            continue
        if isinstance(mutation, UserMessageMutation):
            _apply_pending_experience_mutations()
            current.append(copy.deepcopy(cast(NormalizedMessageParam, mutation.message)))
            continue
        if isinstance(mutation, AssistantTruncatedMutation):
            _apply_pending_experience_mutations()
            _append_assistant_truncated_mutation(current, mutation)
            continue
        if isinstance(mutation, ToolCancelledMutation):
            _apply_pending_experience_mutations()
            _append_tool_cancelled_mutation(current, mutation)
            continue

    _apply_pending_experience_mutations()
    validate_tool_linkage(current)
    return current


def compile_context(
    state: ContextState,
    mutations: Optional[List[ContextMutation]] = None,
) -> CompiledContext:
    applied_messages = apply_mutations(state.messages, mutations or state.pending_mutations)
    return CompiledContext(
        llm_messages=clone_messages(applied_messages),
        semantic_messages=compile_semantic_messages(applied_messages),
    )


__all__ = [
    "CompiledContext",
    "ContextState",
    "apply_mutations",
    "assistant_tool_calls_are_resolved",
    "clone_messages",
    "compile_context",
    "compile_semantic_messages",
]
