from __future__ import annotations

import copy
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

from SimpleLLMFunc.hooks.stream import EventOrigin

MemoryHistory = List[Dict[str, Any]]
HISTORY_PARAM_NAMES = ("history", "chat_history")
SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM = "__self_reference_key_override"
SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM = "__self_reference_toolkit_override"
SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM = "__self_reference_fork_task"
AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR = "__simplellmfunc_accepts_template_params__"
AGENT_FORK_TOOLKIT_FACTORY_ATTR = "__simplellmfunc_fork_toolkit_factory__"


def clone_messages(messages: MemoryHistory) -> MemoryHistory:
    return copy.deepcopy(messages)


def strip_terminal_pending_tool_calls_message(
    history: MemoryHistory,
) -> tuple[MemoryHistory, bool]:
    if not history:
        return history, False

    last_message = history[-1]
    if not isinstance(last_message, dict):
        return history, False
    if last_message.get("role") != "assistant":
        return history, False

    tool_calls = last_message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return history, False

    return clone_messages(history[:-1]), True


def append_fork_task_user_message(
    history: MemoryHistory,
    *,
    task_message: str,
) -> MemoryHistory:
    child_task_instruction = (
        "You are now already a forked subagent. "
        "So you have no need to care about whether the previous fork was "
        "correct or not, because it has already succeeded. "
        "And now, the only thing you are required to do is:\n\n"
        f"{task_message}\n\n"
        "Follow the instructions above to finish this task."
    )
    updated_history = clone_messages(history)
    updated_history.append(
        {
            "role": "user",
            "content": child_task_instruction,
        }
    )
    return updated_history


def extract_history_from_any(value: Any) -> Optional[MemoryHistory]:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, dict) for item in value):
        return None
    return clone_messages(cast(MemoryHistory, value))


def extract_response_and_history_from_output(
    output: Any,
) -> tuple[Any, Optional[MemoryHistory]]:
    if isinstance(output, tuple) and len(output) == 2:
        maybe_history = extract_history_from_any(output[1])
        if maybe_history is not None:
            return output[0], maybe_history

    event = getattr(output, "event", None)
    if event is not None:
        maybe_history = extract_history_from_any(
            getattr(event, "final_messages", None)
        )
        if maybe_history is not None:
            return getattr(event, "final_response", None), maybe_history

    return output, None


def normalize_fork_ids(value: Optional[Any], normalize_key: Callable[[str], str]) -> Optional[List[str]]:
    if value is None:
        return None

    if isinstance(value, str):
        return [normalize_key(value)]

    if isinstance(value, dict):
        fork_id = value.get("fork_id")
        if isinstance(fork_id, str) and fork_id.strip():
            return [normalize_key(fork_id)]
        raise ValueError("fork_ids dict must include fork_id")

    if not isinstance(value, list):
        raise ValueError(
            "fork_ids must be a fork_id string, fork handle dict, or list of either"
        )

    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized.append(normalize_key(item))
            continue

        if isinstance(item, dict):
            fork_id = item.get("fork_id")
            if isinstance(fork_id, str) and fork_id.strip():
                normalized.append(normalize_key(fork_id))
                continue

        raise ValueError("fork_ids must contain fork_id strings or dicts with fork_id")

    return normalized


def extract_event_and_origin_from_agent_output(output: Any) -> tuple[Any, Any]:
    if getattr(output, "type", None) != "event":
        return None, None
    return getattr(output, "event", None), getattr(output, "origin", None)


def extract_history_param_name(agent_instance: Any) -> Optional[str]:
    try:
        signature = inspect.signature(agent_instance)
    except (TypeError, ValueError):
        return None

    for candidate in HISTORY_PARAM_NAMES:
        if candidate in signature.parameters:
            return candidate

    return None


def agent_supports_template_params(agent_instance: Any) -> bool:
    if bool(getattr(agent_instance, AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR, False)):
        return True

    try:
        signature = inspect.signature(agent_instance)
    except (TypeError, ValueError):
        return False

    if "_template_params" in signature.parameters:
        return True

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return False


def get_agent_fork_toolkit_factory(
    agent_instance: Any,
) -> Optional[Callable[[Any], Any]]:
    maybe_factory = getattr(agent_instance, AGENT_FORK_TOOLKIT_FACTORY_ATTR, None)
    if callable(maybe_factory):
        return cast(Optional[Callable[[Any], Any]], maybe_factory)
    return None


def build_fork_error_result(
    *,
    fork_id: str,
    source_memory_key: str,
    memory_key: str,
    parent_fork_id: Optional[str],
    depth: int,
    error: BaseException,
) -> Dict[str, Any]:
    return {
        "fork_id": fork_id,
        "parent_fork_id": parent_fork_id,
        "depth": depth,
        "source_memory_key": source_memory_key,
        "memory_key": memory_key,
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "response": None,
        "result": None,
        "history": [],
        "history_count": 0,
        "history_included": True,
    }


async def emit_fork_custom_event(
    event_emitter: Any,
    event_name: str,
    payload: Dict[str, Any],
) -> None:
    if event_emitter is None:
        return

    emit = getattr(event_emitter, "emit", None)
    if not callable(emit):
        return

    maybe_awaitable = emit(event_name, payload)
    if inspect.isawaitable(maybe_awaitable):
        await cast(Awaitable[Any], maybe_awaitable)


async def emit_fork_agent_event(
    event_emitter: Any,
    event: Any,
    *,
    fork_id: str,
    depth: int,
    source_memory_key: str,
    memory_key: str,
    forwarded_origin: Any = None,
) -> None:
    if event_emitter is None:
        return

    emit_event = getattr(event_emitter, "emit_event", None)
    if not callable(emit_event):
        return

    resolved_origin_overrides: Dict[str, Any]
    forwarded_fork_id = getattr(forwarded_origin, "fork_id", None)
    if isinstance(forwarded_fork_id, str) and forwarded_fork_id:
        resolved_origin_overrides = {
            "fork_id": forwarded_fork_id,
        }

        forwarded_depth = getattr(forwarded_origin, "fork_depth", None)
        if isinstance(forwarded_depth, int):
            resolved_origin_overrides["fork_depth"] = forwarded_depth
        else:
            resolved_origin_overrides["fork_depth"] = depth

        forwarded_source_memory_key = getattr(
            forwarded_origin,
            "source_memory_key",
            None,
        )
        if isinstance(forwarded_source_memory_key, str) and forwarded_source_memory_key:
            resolved_origin_overrides["source_memory_key"] = forwarded_source_memory_key
        else:
            resolved_origin_overrides["source_memory_key"] = source_memory_key

        forwarded_memory_key = getattr(forwarded_origin, "memory_key", None)
        if isinstance(forwarded_memory_key, str) and forwarded_memory_key:
            resolved_origin_overrides["memory_key"] = forwarded_memory_key
        else:
            resolved_origin_overrides["memory_key"] = memory_key

        forwarded_tool_name = getattr(forwarded_origin, "tool_name", None)
        if isinstance(forwarded_tool_name, str) and forwarded_tool_name:
            resolved_origin_overrides["tool_name"] = forwarded_tool_name

        forwarded_tool_call_id = getattr(forwarded_origin, "tool_call_id", None)
        if isinstance(forwarded_tool_call_id, str) and forwarded_tool_call_id:
            resolved_origin_overrides["tool_call_id"] = forwarded_tool_call_id
    else:
        resolved_origin_overrides = {
            "fork_id": fork_id,
            "fork_depth": depth,
            "source_memory_key": source_memory_key,
            "memory_key": memory_key,
        }

    maybe_awaitable = emit_event(
        event,
        origin_overrides=resolved_origin_overrides,
    )
    if inspect.isawaitable(maybe_awaitable):
        await cast(Awaitable[Any], maybe_awaitable)


async def consume_agent_call_output(
    call_output: Any,
    *,
    event_emitter: Any = None,
    fork_id: Optional[str] = None,
    parent_fork_id: Optional[str] = None,
    depth: Optional[int] = None,
    source_memory_key: Optional[str] = None,
    memory_key: Optional[str] = None,
) -> tuple[Any, Optional[MemoryHistory]]:
    _ = parent_fork_id
    if inspect.isawaitable(call_output):
        awaited_output = await cast(Awaitable[Any], call_output)
        return extract_response_and_history_from_output(awaited_output)

    if hasattr(call_output, "__aiter__"):
        last_response: Any = None
        last_history: Optional[MemoryHistory] = None
        stream_forwarding_enabled = (
            fork_id is not None
            and depth is not None
            and source_memory_key is not None
            and memory_key is not None
        )

        async for output in call_output:
            if stream_forwarding_enabled:
                forwarded_event, forwarded_origin = (
                    extract_event_and_origin_from_agent_output(output)
                )
                if forwarded_event is not None:
                    await emit_fork_agent_event(
                        event_emitter,
                        forwarded_event,
                        fork_id=cast(str, fork_id),
                        depth=cast(int, depth),
                        source_memory_key=cast(str, source_memory_key),
                        memory_key=cast(str, memory_key),
                        forwarded_origin=forwarded_origin,
                    )

            response, history = extract_response_and_history_from_output(output)
            last_response = response
            if history is not None:
                last_history = history

        return last_response, last_history

    return extract_response_and_history_from_output(call_output)


__all__ = [
    "AGENT_FORK_TOOLKIT_FACTORY_ATTR",
    "AGENT_TEMPLATE_PARAMS_SUPPORT_ATTR",
    "HISTORY_PARAM_NAMES",
    "SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM",
    "SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM",
    "SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM",
    "agent_supports_template_params",
    "append_fork_task_user_message",
    "build_fork_error_result",
    "clone_messages",
    "consume_agent_call_output",
    "emit_fork_agent_event",
    "emit_fork_custom_event",
    "extract_event_and_origin_from_agent_output",
    "extract_history_from_any",
    "extract_history_param_name",
    "extract_response_and_history_from_output",
    "get_agent_fork_toolkit_factory",
    "normalize_fork_ids",
    "strip_terminal_pending_tool_calls_message",
]
