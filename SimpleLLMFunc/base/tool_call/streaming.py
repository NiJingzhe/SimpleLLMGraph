"""Streaming tool-call protocol helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Tuple

from SimpleLLMFunc.type.tool_call import ToolCallArguments
from SimpleLLMFunc.base.tool_call.extraction import parse_tool_call_arguments


@dataclass
class StreamingToolCallState:
    """Streaming state per tool-call index while parsing argument deltas."""

    tool_call_id: str = ""
    tool_name: str = ""
    arguments: str = ""
    emitted_argument_values: Dict[str, str] = field(default_factory=dict)


def stringify_argument_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def collect_stream_argument_deltas(
    state: StreamingToolCallState,
    parsed_arguments: ToolCallArguments,
) -> List[Tuple[str, str]]:
    """Compute per-argument text deltas from parsed arguments snapshot."""

    deltas: List[Tuple[str, str]] = []

    for argname, argvalue in parsed_arguments.items():
        current_value = stringify_argument_value(argvalue)
        previous_value = state.emitted_argument_values.get(argname, "")

        if current_value == previous_value:
            continue

        if previous_value and previous_value.startswith(current_value):
            # Ignore temporary parser regression caused by partial payloads.
            continue

        if previous_value and current_value.startswith(previous_value):
            delta = current_value[len(previous_value) :]
        else:
            delta = current_value

        if delta:
            deltas.append((argname, delta))

        state.emitted_argument_values[argname] = current_value

    return deltas


def collect_tool_argument_delta_payloads(
    tool_call_chunks: List[Dict[str, Any]],
    stream_states: Dict[int, StreamingToolCallState],
) -> List[Tuple[str, str, str, str]]:
    """Collect (tool_name, tool_call_id, argname, arg_delta) payloads from chunks."""

    payloads: List[Tuple[str, str, str, str]] = []

    for chunk in tool_call_chunks:
        index = chunk.get("index")
        if index is None:
            continue

        state = stream_states.setdefault(index, StreamingToolCallState())

        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str) and chunk_id:
            state.tool_call_id = chunk_id

        function_chunk = chunk.get("function")
        if isinstance(function_chunk, dict):
            chunk_name = function_chunk.get("name")
            if isinstance(chunk_name, str) and chunk_name:
                state.tool_name = chunk_name

            argument_delta = function_chunk.get("arguments")
            if isinstance(argument_delta, str) and argument_delta:
                state.arguments += argument_delta

        if not state.arguments or not state.tool_call_id:
            continue

        parsed_arguments = parse_tool_call_arguments(
            state.arguments,
            allow_closure=True,
        )
        if parsed_arguments is None:
            continue

        deltas = collect_stream_argument_deltas(state, parsed_arguments)
        for argname, argcontent_delta in deltas:
            payloads.append(
                (
                    state.tool_name,
                    state.tool_call_id,
                    argname,
                    argcontent_delta,
                )
            )

    return payloads


__all__ = [
    "StreamingToolCallState",
    "collect_stream_argument_deltas",
    "collect_tool_argument_delta_payloads",
    "stringify_argument_value",
]
