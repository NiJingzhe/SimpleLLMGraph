"""Core stream-consumption logic for the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import inspect
from typing import AsyncGenerator, Optional, Protocol, Sequence, Any, Awaitable, cast

from SimpleLLMFunc.hooks.events import (
    CustomEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    ReactEndEvent,
    ReActEvent,
    ToolCallArgumentsDeltaEvent,
    ToolCallEndEvent,
    ToolCallErrorEvent,
    ToolCallStartEvent,
)
from SimpleLLMFunc.hooks.stream import EventOrigin, ReactOutput, is_event_yield
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.utils.tui.formatters import (
    extract_reasoning_delta,
    extract_stream_text,
    extract_text_from_response,
    format_custom_event_fallback,
    format_model_stats,
    format_tool_result_markdown,
    format_tool_stats,
)
from SimpleLLMFunc.utils.tui.hooks import (
    ToolCustomEventHook,
    ToolRenderSnapshot,
    apply_tool_event_hooks,
)


class TUIStreamAdapter(Protocol):
    """Adapter protocol for rendering stream events."""

    async def start_model_response(self, model_call_id: str) -> None: ...

    async def append_model_content(
        self, model_call_id: str, content_delta: str
    ) -> None: ...

    async def append_model_reasoning(
        self,
        model_call_id: str,
        reasoning_delta: str,
    ) -> None: ...

    async def finish_model_response(
        self, model_call_id: str, stats_line: str
    ) -> None: ...

    async def start_tool_call(
        self,
        model_call_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
    ) -> None: ...

    async def append_tool_output(
        self, tool_call_id: str, output_delta: str
    ) -> None: ...

    async def append_tool_argument(
        self,
        tool_call_id: str,
        argname: str,
        argcontent_delta: str,
    ) -> None: ...

    async def set_tool_status(self, tool_call_id: str, status: str) -> None: ...

    async def request_tool_input(
        self,
        tool_call_id: str,
        request_id: str,
        prompt: str,
    ) -> None: ...

    async def clear_tool_input(self, tool_call_id: str) -> None: ...

    async def finish_tool_call(
        self,
        tool_call_id: str,
        result_markdown: str,
        stats_line: str,
        success: bool,
    ) -> None: ...


@dataclass
class _StreamConsumeState:
    """Mutable internal state while consuming one agent turn."""

    llm_call_count: int = 0
    active_model_call_id: Optional[str] = None
    model_has_chunk_text: dict[str, bool] = field(default_factory=dict)
    tool_snapshots: dict[str, ToolRenderSnapshot] = field(default_factory=dict)
    running_tool_call_ids: list[str] = field(default_factory=list)
    fork_sessions: dict[str, "_ForkSessionState"] = field(default_factory=dict)
    final_history: MessageList = field(default_factory=list)
    top_level_react_end_seen: bool = False


@dataclass
class _ForkSessionState:
    """Per-fork render state for fork UI cards."""

    fork_id: str
    model_call_id: str
    parent_fork_id: Optional[str] = None
    depth: Optional[int] = None
    memory_key: str = ""
    metadata_rendered: bool = False
    llm_call_count: int = 0
    running_tool_call_ids: list[str] = field(default_factory=list)
    tool_call_id_map: dict[str, str] = field(default_factory=dict)


_FORK_LIFECYCLE_EVENT_NAMES = {
    "selfref_fork_start",
    "selfref_fork_spawned",
    "selfref_fork_end",
    "selfref_fork_error",
}


async def _ensure_active_model_call(
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
) -> str:
    if state.active_model_call_id:
        return state.active_model_call_id

    state.llm_call_count += 1
    model_call_id = f"llm_call_{state.llm_call_count}"
    state.active_model_call_id = model_call_id
    state.model_has_chunk_text[model_call_id] = False
    await adapter.start_model_response(model_call_id)
    return model_call_id


def _extract_input_request(data: object) -> tuple[Optional[str], str]:
    if not isinstance(data, dict):
        return None, ""

    request_id = data.get("request_id")
    prompt = data.get("prompt")

    if not isinstance(request_id, str) or not request_id:
        return None, ""

    if not isinstance(prompt, str):
        prompt = ""

    return request_id, prompt


def _parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None

    return None


def _compose_fork_model_call_id(fork_id: str) -> str:
    return f"fork::{fork_id}"


def _compose_fork_tool_call_id(fork_id: str, tool_call_id: str) -> str:
    return f"fork::{fork_id}::tool::{tool_call_id}"


def _format_execute_code_result_markdown(result: object) -> str:
    if not isinstance(result, dict):
        return format_tool_result_markdown(result)

    blocks: list[str] = []

    return_value = result.get("return_value")
    if return_value is not None:
        rendered_return_value = format_tool_result_markdown(return_value)
        if rendered_return_value:
            blocks.append(f"Return value:\n{rendered_return_value}")

    success = bool(result.get("success", False))
    error = result.get("error")
    if (not success) and isinstance(error, str) and error:
        blocks.append(f"Error: {error}")

    return "\n\n".join(blocks)


def _format_tool_result_for_display(tool_name: str, result: object) -> str:
    if tool_name == "execute_code":
        return _format_execute_code_result_markdown(result)
    return format_tool_result_markdown(result)


async def _ensure_fork_session(
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
    *,
    fork_id: str,
    parent_fork_id: Optional[str],
    depth: Optional[int],
    memory_key: str,
) -> _ForkSessionState:
    existing = state.fork_sessions.get(fork_id)
    if existing is not None:
        if parent_fork_id is not None:
            existing.parent_fork_id = parent_fork_id
        if depth is not None:
            existing.depth = depth
        if memory_key:
            existing.memory_key = memory_key
        return existing

    model_call_id = _compose_fork_model_call_id(fork_id)
    await adapter.start_model_response(model_call_id)
    session = _ForkSessionState(
        fork_id=fork_id,
        model_call_id=model_call_id,
        parent_fork_id=parent_fork_id,
        depth=depth,
        memory_key=memory_key,
    )
    state.fork_sessions[fork_id] = session
    return session


def _resolve_fork_session_tool_call_id(
    session: _ForkSessionState,
    child_tool_call_id: str,
) -> str:
    mapped = session.tool_call_id_map.get(child_tool_call_id)
    if mapped is not None:
        return mapped

    mapped = _compose_fork_tool_call_id(session.fork_id, child_tool_call_id)
    session.tool_call_id_map[child_tool_call_id] = mapped
    return mapped


def _append_argument_delta_to_snapshot(
    snapshot: ToolRenderSnapshot,
    *,
    argname: str,
    argcontent_delta: str,
) -> None:
    previous = snapshot.arguments.get(argname, "")
    if not isinstance(previous, str):
        previous = str(previous)
    snapshot.arguments[argname] = previous + argcontent_delta


async def _ensure_tool_snapshot(
    *,
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
    model_call_id: str,
    tool_call_id: str,
    tool_name: str,
) -> ToolRenderSnapshot:
    snapshot = state.tool_snapshots.get(tool_call_id)
    if snapshot is not None:
        if tool_name and snapshot.tool_name != tool_name:
            snapshot.tool_name = tool_name
        return snapshot

    await adapter.start_tool_call(
        model_call_id=model_call_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments={},
    )
    snapshot = ToolRenderSnapshot(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        arguments={},
    )
    state.tool_snapshots[tool_call_id] = snapshot
    return snapshot


async def _apply_custom_tool_event(
    *,
    event: CustomEvent,
    resolved_tool_call_id: Optional[str],
    running_tool_call_ids: list[str],
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
    custom_hooks: Sequence[ToolCustomEventHook],
) -> None:
    tool_call_id = resolved_tool_call_id
    if not tool_call_id and running_tool_call_ids:
        tool_call_id = running_tool_call_ids[-1]

    if not tool_call_id:
        return

    snapshot = state.tool_snapshots.get(tool_call_id)
    if snapshot is None:
        return

    if event.event_name == "kernel_input_request":
        request_id, prompt = _extract_input_request(event.data)
        if request_id:
            await adapter.request_tool_input(
                tool_call_id=tool_call_id,
                request_id=request_id,
                prompt=prompt,
            )

    update = apply_tool_event_hooks(
        event=event,
        snapshot=snapshot,
        custom_hooks=custom_hooks,
    )

    if update is None:
        update_text = format_custom_event_fallback(event.event_name, event.data)
        await adapter.append_tool_output(tool_call_id, update_text)
        snapshot.output += update_text
        return

    if update.replace_output is not None:
        snapshot.output = update.replace_output
        await adapter.append_tool_output(tool_call_id, update.replace_output)
    elif update.append_output:
        snapshot.output += update.append_output
        await adapter.append_tool_output(tool_call_id, update.append_output)

    if update.status:
        snapshot.status = update.status
        await adapter.set_tool_status(tool_call_id, update.status)


async def _handle_fork_lifecycle_custom_event(
    event: CustomEvent,
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
) -> bool:
    if event.event_name not in _FORK_LIFECYCLE_EVENT_NAMES:
        return False

    if not isinstance(event.data, dict):
        return True

    fork_id_raw = event.data.get("fork_id")
    if not isinstance(fork_id_raw, str) or not fork_id_raw:
        return True

    parent_fork_id = event.data.get("parent_fork_id")
    if not isinstance(parent_fork_id, str):
        parent_fork_id = None

    depth = _parse_optional_int(event.data.get("depth"))
    memory_key = event.data.get("memory_key")
    if not isinstance(memory_key, str):
        memory_key = ""

    session = await _ensure_fork_session(
        adapter=adapter,
        state=state,
        fork_id=fork_id_raw,
        parent_fork_id=parent_fork_id,
        depth=depth,
        memory_key=memory_key,
    )

    if event.event_name == "selfref_fork_error":
        error_type = event.data.get("error_type")
        error_message = event.data.get("error_message")
        if isinstance(error_type, str) and isinstance(error_message, str):
            await adapter.append_model_reasoning(
                session.model_call_id,
                f"{error_type}: {error_message}",
            )
        await adapter.finish_model_response(session.model_call_id, "fork | error")
        return True

    if event.event_name == "selfref_fork_end":
        await adapter.finish_model_response(session.model_call_id, "fork | completed")
        return True

    return True


def _extract_origin_payload(
    *,
    event: ReActEvent,
    origin: Optional[EventOrigin],
) -> dict[str, object]:
    if origin is not None:
        return origin.as_dict()

    event_extra = getattr(event, "extra", None)
    if isinstance(event_extra, dict):
        raw_origin = event_extra.get("origin")
        if isinstance(raw_origin, dict):
            return raw_origin

    return {}


def _resolve_fork_origin_context(
    *,
    event: ReActEvent,
    origin: Optional[EventOrigin],
) -> tuple[Optional[str], Optional[str], Optional[int], str]:
    if origin is not None and origin.fork_id:
        return (
            origin.fork_id,
            None,
            origin.fork_depth,
            origin.memory_key or "",
        )

    origin_payload = _extract_origin_payload(event=event, origin=origin)
    fork_id_raw = origin_payload.get("fork_id")
    if not isinstance(fork_id_raw, str) or not fork_id_raw:
        return None, None, None, ""

    parent_fork_id_raw = origin_payload.get("parent_fork_id")
    parent_fork_id = parent_fork_id_raw if isinstance(parent_fork_id_raw, str) else None

    depth = _parse_optional_int(origin_payload.get("fork_depth"))
    memory_key_raw = origin_payload.get("memory_key")
    memory_key = memory_key_raw if isinstance(memory_key_raw, str) else ""
    return fork_id_raw, parent_fork_id, depth, memory_key


async def _handle_origin_scoped_fork_event(
    *,
    event: ReActEvent,
    origin: Optional[EventOrigin],
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
    custom_hooks: Sequence[ToolCustomEventHook],
) -> bool:
    if isinstance(event, CustomEvent) and (
        event.event_name in _FORK_LIFECYCLE_EVENT_NAMES
    ):
        return False

    fork_id, parent_fork_id, depth, memory_key = _resolve_fork_origin_context(
        event=event,
        origin=origin,
    )
    if fork_id is None:
        return False

    session = await _ensure_fork_session(
        adapter=adapter,
        state=state,
        fork_id=fork_id,
        parent_fork_id=parent_fork_id,
        depth=depth,
        memory_key=memory_key,
    )

    if isinstance(event, LLMCallStartEvent):
        session.llm_call_count += 1
        if session.llm_call_count > 1:
            await adapter.start_model_response(session.model_call_id)
        state.model_has_chunk_text[session.model_call_id] = False
        return True

    if isinstance(event, LLMChunkArriveEvent):
        content_delta = extract_stream_text(event.chunk)
        if content_delta:
            state.model_has_chunk_text[session.model_call_id] = True
            await adapter.append_model_content(session.model_call_id, content_delta)

        reasoning_delta = extract_reasoning_delta(event.chunk)
        if reasoning_delta:
            await adapter.append_model_reasoning(session.model_call_id, reasoning_delta)
        return True

    if isinstance(event, LLMCallEndEvent):
        if not state.model_has_chunk_text.get(session.model_call_id, False):
            text = extract_text_from_response(event.response)
            if text:
                await adapter.append_model_content(session.model_call_id, text)

        model_name = getattr(event.response, "model", None)
        stats_line = format_model_stats(
            execution_time=event.execution_time,
            usage=event.usage,
            model_name=model_name,
        )
        await adapter.finish_model_response(session.model_call_id, stats_line)
        return True

    if isinstance(event, ToolCallArgumentsDeltaEvent):
        mapped_tool_call_id = _resolve_fork_session_tool_call_id(
            session,
            event.tool_call_id,
        )
        snapshot = await _ensure_tool_snapshot(
            adapter=adapter,
            state=state,
            model_call_id=session.model_call_id,
            tool_call_id=mapped_tool_call_id,
            tool_name=event.tool_name,
        )
        _append_argument_delta_to_snapshot(
            snapshot,
            argname=event.argname,
            argcontent_delta=event.argcontent_delta,
        )
        await adapter.append_tool_argument(
            mapped_tool_call_id,
            event.argname,
            event.argcontent_delta,
        )
        if mapped_tool_call_id not in session.running_tool_call_ids:
            session.running_tool_call_ids.append(mapped_tool_call_id)
        return True

    if isinstance(event, ToolCallStartEvent):
        mapped_tool_call_id = _resolve_fork_session_tool_call_id(
            session,
            event.tool_call_id,
        )
        existing_snapshot = state.tool_snapshots.get(mapped_tool_call_id)
        if existing_snapshot is None:
            await adapter.start_tool_call(
                model_call_id=session.model_call_id,
                tool_call_id=mapped_tool_call_id,
                tool_name=event.tool_name,
                arguments=event.arguments,
            )
            state.tool_snapshots[mapped_tool_call_id] = ToolRenderSnapshot(
                tool_name=event.tool_name,
                tool_call_id=mapped_tool_call_id,
                arguments=dict(event.arguments),
            )
        else:
            if event.tool_name:
                existing_snapshot.tool_name = event.tool_name
            existing_snapshot.arguments = dict(event.arguments)
            await adapter.set_tool_status(mapped_tool_call_id, "running")

        if mapped_tool_call_id not in session.running_tool_call_ids:
            session.running_tool_call_ids.append(mapped_tool_call_id)
        return True

    if isinstance(event, CustomEvent):
        resolved_tool_call_id: Optional[str] = None
        if event.tool_call_id:
            resolved_tool_call_id = _resolve_fork_session_tool_call_id(
                session,
                event.tool_call_id,
            )

        await _apply_custom_tool_event(
            event=event,
            resolved_tool_call_id=resolved_tool_call_id,
            running_tool_call_ids=session.running_tool_call_ids,
            adapter=adapter,
            state=state,
            custom_hooks=custom_hooks,
        )
        return True

    if isinstance(event, ToolCallEndEvent):
        mapped_tool_call_id = _resolve_fork_session_tool_call_id(
            session,
            event.tool_call_id,
        )
        snapshot = state.tool_snapshots.get(mapped_tool_call_id)
        tool_name = snapshot.tool_name if snapshot is not None else event.tool_name
        result_markdown = _format_tool_result_for_display(tool_name, event.result)
        stats_line = format_tool_stats(event.execution_time, event.success)
        await adapter.finish_tool_call(
            tool_call_id=mapped_tool_call_id,
            result_markdown=result_markdown,
            stats_line=stats_line,
            success=event.success,
        )
        await adapter.clear_tool_input(mapped_tool_call_id)
        if mapped_tool_call_id in session.running_tool_call_ids:
            session.running_tool_call_ids.remove(mapped_tool_call_id)
        if snapshot is not None:
            snapshot.status = "success" if event.success else "error"
        return True

    if isinstance(event, ToolCallErrorEvent):
        mapped_tool_call_id = _resolve_fork_session_tool_call_id(
            session,
            event.tool_call_id,
        )
        result_markdown = format_tool_result_markdown(
            {
                "error": event.error_message,
                "error_type": event.error_type,
            }
        )
        stats_line = format_tool_stats(event.execution_time, False)
        await adapter.finish_tool_call(
            tool_call_id=mapped_tool_call_id,
            result_markdown=result_markdown,
            stats_line=stats_line,
            success=False,
        )
        await adapter.clear_tool_input(mapped_tool_call_id)
        if mapped_tool_call_id in session.running_tool_call_ids:
            session.running_tool_call_ids.remove(mapped_tool_call_id)
        snapshot = state.tool_snapshots.get(mapped_tool_call_id)
        if snapshot is not None:
            snapshot.status = "error"
        return True

    return True


async def _handle_event(
    event: ReActEvent,
    origin: Optional[EventOrigin],
    adapter: TUIStreamAdapter,
    state: _StreamConsumeState,
    custom_hooks: Sequence[ToolCustomEventHook],
) -> None:
    handled_origin_scoped_fork_event = await _handle_origin_scoped_fork_event(
        event=event,
        origin=origin,
        adapter=adapter,
        state=state,
        custom_hooks=custom_hooks,
    )
    if handled_origin_scoped_fork_event:
        return

    if isinstance(event, LLMCallStartEvent):
        state.llm_call_count += 1
        state.active_model_call_id = f"llm_call_{state.llm_call_count}"
        state.model_has_chunk_text[state.active_model_call_id] = False
        await adapter.start_model_response(state.active_model_call_id)
        return

    if isinstance(event, LLMChunkArriveEvent):
        model_call_id = await _ensure_active_model_call(adapter, state)

        content_delta = extract_stream_text(event.chunk)
        if content_delta:
            state.model_has_chunk_text[model_call_id] = True
            await adapter.append_model_content(model_call_id, content_delta)

        reasoning_delta = extract_reasoning_delta(event.chunk)
        if reasoning_delta:
            await adapter.append_model_reasoning(model_call_id, reasoning_delta)
        return

    if isinstance(event, LLMCallEndEvent):
        model_call_id = await _ensure_active_model_call(adapter, state)

        if not state.model_has_chunk_text.get(model_call_id, False):
            text = extract_text_from_response(event.response)
            if text:
                await adapter.append_model_content(model_call_id, text)

        model_name = getattr(event.response, "model", None)
        stats_line = format_model_stats(
            execution_time=event.execution_time,
            usage=event.usage,
            model_name=model_name,
        )
        await adapter.finish_model_response(model_call_id, stats_line)
        return

    if isinstance(event, ToolCallArgumentsDeltaEvent):
        model_call_id = await _ensure_active_model_call(adapter, state)
        snapshot = await _ensure_tool_snapshot(
            adapter=adapter,
            state=state,
            model_call_id=model_call_id,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
        )
        _append_argument_delta_to_snapshot(
            snapshot,
            argname=event.argname,
            argcontent_delta=event.argcontent_delta,
        )
        await adapter.append_tool_argument(
            event.tool_call_id,
            event.argname,
            event.argcontent_delta,
        )
        if event.tool_call_id not in state.running_tool_call_ids:
            state.running_tool_call_ids.append(event.tool_call_id)
        return

    if isinstance(event, ToolCallStartEvent):
        model_call_id = await _ensure_active_model_call(adapter, state)
        existing_snapshot = state.tool_snapshots.get(event.tool_call_id)
        if existing_snapshot is None:
            await adapter.start_tool_call(
                model_call_id=model_call_id,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                arguments=event.arguments,
            )
            state.tool_snapshots[event.tool_call_id] = ToolRenderSnapshot(
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                arguments=dict(event.arguments),
            )
        else:
            if event.tool_name:
                existing_snapshot.tool_name = event.tool_name
            existing_snapshot.arguments = dict(event.arguments)
            await adapter.set_tool_status(event.tool_call_id, "running")

        if event.tool_call_id not in state.running_tool_call_ids:
            state.running_tool_call_ids.append(event.tool_call_id)
        return

    if isinstance(event, CustomEvent):
        handled_fork_lifecycle = await _handle_fork_lifecycle_custom_event(
            event=event,
            adapter=adapter,
            state=state,
        )
        if handled_fork_lifecycle:
            return

        await _apply_custom_tool_event(
            event=event,
            resolved_tool_call_id=event.tool_call_id,
            running_tool_call_ids=state.running_tool_call_ids,
            adapter=adapter,
            state=state,
            custom_hooks=custom_hooks,
        )
        return

    if isinstance(event, ToolCallEndEvent):
        snapshot = state.tool_snapshots.get(event.tool_call_id)
        tool_name = snapshot.tool_name if snapshot is not None else event.tool_name
        result_markdown = _format_tool_result_for_display(tool_name, event.result)
        stats_line = format_tool_stats(event.execution_time, event.success)
        await adapter.finish_tool_call(
            tool_call_id=event.tool_call_id,
            result_markdown=result_markdown,
            stats_line=stats_line,
            success=event.success,
        )
        await adapter.clear_tool_input(event.tool_call_id)
        if event.tool_call_id in state.running_tool_call_ids:
            state.running_tool_call_ids.remove(event.tool_call_id)
        if snapshot is not None:
            snapshot.status = "success" if event.success else "error"
        return

    if isinstance(event, ToolCallErrorEvent):
        result_markdown = format_tool_result_markdown(
            {
                "error": event.error_message,
                "error_type": event.error_type,
            }
        )
        stats_line = format_tool_stats(event.execution_time, False)
        await adapter.finish_tool_call(
            tool_call_id=event.tool_call_id,
            result_markdown=result_markdown,
            stats_line=stats_line,
            success=False,
        )
        await adapter.clear_tool_input(event.tool_call_id)
        if event.tool_call_id in state.running_tool_call_ids:
            state.running_tool_call_ids.remove(event.tool_call_id)
        snapshot = state.tool_snapshots.get(event.tool_call_id)
        if snapshot is not None:
            snapshot.status = "error"
        return

    if isinstance(event, ReactEndEvent):
        fork_id, _, _, _ = _resolve_fork_origin_context(event=event, origin=origin)
        if fork_id is None:
            state.top_level_react_end_seen = True
        state.final_history = event.final_messages


async def consume_react_stream(
    stream: AsyncGenerator[ReactOutput, None],
    adapter: TUIStreamAdapter,
    custom_hooks: Optional[Sequence[ToolCustomEventHook]] = None,
    abort_signal: Optional[AbortSignal] = None,
) -> MessageList:
    """Consume one react event stream and update the adapter in real-time."""

    async def _close_stream(stream_obj: Any) -> None:
        close_method = getattr(stream_obj, "aclose", None)
        if callable(close_method):
            try:
                result = close_method()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)
            except Exception:
                pass

    stream_queue: asyncio.Queue[object] = asyncio.Queue()
    done_marker = object()

    async def _drain_stream() -> None:
        try:
            async for output in stream:
                await stream_queue.put(output)
        except Exception as exc:
            await stream_queue.put(exc)
        finally:
            await stream_queue.put(done_marker)

    stream_task = asyncio.create_task(_drain_stream())

    async def _next_output() -> object:
        if abort_signal is None:
            return await stream_queue.get()

        abort_task = asyncio.create_task(abort_signal.wait())
        next_task = asyncio.create_task(stream_queue.get())
        done, _ = await asyncio.wait(
            {abort_task, next_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
            abort_task.cancel()
            raise asyncio.CancelledError
        abort_task.cancel()
        return next_task.result()

    hooks = list(custom_hooks or [])
    state = _StreamConsumeState()
    saw_event = False
    aborted = False

    while True:
        try:
            payload = await _next_output()
        except StopAsyncIteration:
            break
        except asyncio.CancelledError:
            aborted = True
            await _close_stream(stream)
            break

        if payload is done_marker:
            break

        if isinstance(payload, Exception):
            raise payload

        output = cast(ReactOutput, payload)

        if not is_event_yield(output):
            continue

        saw_event = True

        await _handle_event(
            event=output.event,
            origin=output.origin,
            adapter=adapter,
            state=state,
            custom_hooks=hooks,
        )

        if state.top_level_react_end_seen and not state.running_tool_call_ids:
            has_pending_fork_tools = any(
                bool(session.running_tool_call_ids)
                for session in state.fork_sessions.values()
            )
            if (
                not has_pending_fork_tools
                and stream_task.done()
                and stream_queue.empty()
            ):
                break

    if not stream_task.done():
        stream_task.cancel()
        await asyncio.gather(stream_task, return_exceptions=True)

    if not saw_event and not aborted:
        raise ValueError(
            "TUI requires a ReactOutput event stream. No EventYield output was received."
        )

    return state.final_history


__all__ = [
    "TUIStreamAdapter",
    "consume_react_stream",
]
