"""Textual app implementation for SimpleLLMFunc chat TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Literal, Optional, Sequence

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import (
    Horizontal,
    HorizontalScroll,
    VerticalGroup,
    VerticalScroll,
)
from textual.widgets import Input, Static

from SimpleLLMFunc.hooks.input_stream import AgentInputRouter, UserInputEvent
from SimpleLLMFunc.hooks.stream import ReactOutput
from SimpleLLMFunc.hooks.abort import AbortSignal, ABORT_SIGNAL_PARAM
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.utils.tui.core import consume_react_stream
from SimpleLLMFunc.utils.tui.formatters import format_tool_arguments_markdown
from SimpleLLMFunc.utils.tui.hooks import ToolCustomEventHook
from SimpleLLMFunc.utils.tui.tool_cards.base import ToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.factory import build_tool_call_card
from SimpleLLMFunc.utils.tui.widgets import DotPulse


@dataclass
class _ModelWidgets:
    model_call_id: str
    agent_key: str
    root: VerticalGroup
    loading_row: Horizontal
    loading_widget: DotPulse
    loading_label_widget: Static
    reasoning_widget: Static
    content_widget: Static
    tool_list_widget: VerticalGroup
    stats_widget: Static
    content: str = ""
    reasoning: str = ""
    stats_line: str = ""
    tool_call_ids: list[str] = field(default_factory=list)
    running_tool_call_ids: set[str] = field(default_factory=set)
    finished: bool = False
    content_dirty: bool = False
    reasoning_dirty: bool = False
    render_pending: bool = False


@dataclass
class _RenderBlock:
    block_id: str
    agent_key: str
    kind: Literal["user", "model"]
    root: VerticalGroup
    model_call_id: Optional[str] = None
    user_text: str = ""
    archived: bool = False


@dataclass
class _AgentColumnWidgets:
    root: VerticalScroll
    header_widget: Static
    message_list_widget: VerticalGroup


class AgentTUIApp(App[None]):
    """Terminal chat UI for llm_chat event streams."""

    DEFAULT_INPUT_PLACEHOLDER = "Type a message and press Enter"
    MAIN_AGENT_KEY = "main"
    VIRTUAL_KEEP_LIVE_BLOCKS = 36
    VIRTUALIZE_NEAR_BOTTOM_MARGIN = 4
    ARCHIVE_PREVIEW_CHARS = 160

    CSS = """
    Screen {
        layout: vertical;
        background: #0f1115;
    }

    #chat-log {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
    }

    #chat-input {
        dock: bottom;
        margin: 0 1 1 1;
        border: tall #6f87a8;
    }

    #render-stats {
        dock: bottom;
        margin: 0 1;
        padding: 0 1;
        border: round #2f3a4e;
        color: #7f8798;
        background: #121722;
    }

    .bubble {
        height: auto;
        margin: 0 0 1 0;
        padding: 1;
        border: round #3a4252;
        background: #151922;
    }

    .tool-list {
        height: auto;
    }

    .user-bubble {
        border: round #5f8d5a;
        background: #152019;
    }

    .model-bubble {
        border: round #5978a1;
        background: #151d2a;
    }

    .agent-board {
        layout: horizontal;
        width: 100%;
        height: 1fr;
        margin: 0 0 1 0;
        overflow-x: auto;
        overflow-y: hidden;
    }

    .agent-column {
        width: auto;
        min-width: 33%;
        height: 1fr;
        margin: 0;
        overflow-y: auto;
    }

    #agent-column-main {
        margin: 0;
    }

    .agent-column-header {
        margin: 0 0 1 0;
        padding: 0 1;
        border: round #3a4252;
        background: #121722;
        color: #a8b8d0;
        text-style: bold;
    }

    .agent-column-messages {
        height: auto;
    }

    .role {
        text-style: bold;
        color: #a8b8d0;
        margin: 0 0 1 0;
    }

    .reasoning {
        color: #7f8798;
        margin: 0 0 1 0;
    }

    .assistant-loading {
        width: 3;
        min-width: 1;
        max-width: 3;
        height: 1;
        min-height: 1;
        max-height: 1;
        margin: 0;
        color: #7f8798;
    }

    .assistant-loading-row {
        layout: horizontal;
        height: 1;
        width: auto;
        margin: 1 0 0 0;
        align-horizontal: left;
    }

    .assistant-loading-label {
        width: auto;
        margin: 0 1 0 0;
        color: #7f8798;
    }

    .stats {
        color: #7f8798;
        margin: 1 0 0 0;
    }

    .tool-call {
        height: auto;
        margin: 1 0 0 0;
        padding: 1;
        border: round #8f7a4d;
        background: #221f16;
    }

    .tool-status {
        color: #d4c9a2;
    }

    .archived-block {
        color: #7f8798;
        text-style: italic;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+y", "copy_all_text", "Copy All"),
    ]

    def __init__(
        self,
        agent_func: Callable[..., AsyncGenerator[ReactOutput, None]],
        input_param: str,
        history_param: Optional[str],
        static_kwargs: dict[str, Any],
        custom_hooks: Optional[Sequence[ToolCustomEventHook]] = None,
        title_text: str = "SimpleLLMFunc TUI",
        initial_history: Optional[MessageList] = None,
        input_router: Optional[AgentInputRouter] = None,
    ):
        super().__init__()
        self.title = title_text
        self.agent_func = agent_func
        self.input_param = input_param
        self.history_param = history_param
        self.static_kwargs = static_kwargs
        self.custom_hooks = list(custom_hooks or [])

        self.history: MessageList = list(initial_history or [])
        self._busy = False
        self._active_abort_signal: Optional[AbortSignal] = None
        self._pending_user_text: Optional[str] = None

        if input_router is not None:
            self._input_router = input_router
        else:
            from SimpleLLMFunc.builtin import PyRepl

            self._input_router = AgentInputRouter(submit_tool_input=PyRepl.submit_input)

        self._models: dict[str, _ModelWidgets] = {}
        self._tools: dict[str, ToolCallCard] = {}
        self._agent_board: Optional[HorizontalScroll] = None
        self._agent_columns: dict[str, _AgentColumnWidgets] = {}
        self._model_agent_key: dict[str, str] = {}
        self._agent_blocks: dict[str, list[_RenderBlock]] = {}
        self._system_hints: list[str] = []
        self._detached_transcript_sections: list[str] = []
        self._next_user_block_index = 1
        self._archive_count = 0
        self._render_flush_count = 0
        self._total_blocks = 0
        self._archived_blocks = 0
        self._active_models = 0
        self._active_tools = 0
        self._virtual_keep_live_blocks = self.VIRTUAL_KEEP_LIVE_BLOCKS
        self._virtualize_near_bottom_margin = self.VIRTUALIZE_NEAR_BOTTOM_MARGIN
        self._scroll_after_refresh_pending = False
        self._chat_scroll_expected_y: Optional[float] = None
        self._agent_scroll_pending: set[str] = set()
        self._agent_scroll_expected_y: dict[str, float] = {}
        self._agent_layout_pending = False

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-log")
        yield Static("", id="render-stats")
        yield Input(placeholder=self.DEFAULT_INPUT_PLACEHOLDER, id="chat-input")

    async def on_mount(self) -> None:
        await self._append_system_hint(
            "TUI ready. Type a message to start. Use /copy or Ctrl+Y to copy all text; use /chat <message> to bypass pending tool input; use /exit, /quit, /q, or Ctrl+Q to quit."
        )
        self._update_render_stats_widget()
        self.query_one("#chat-input", Input).focus()

    def _scroll_chat_log_to_end(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.scroll_end(animate=False, force=True, immediate=True)

    def _auto_scroll_chat_log(self) -> None:
        self._scroll_chat_log_to_end()
        chat_log = self.query_one("#chat-log", VerticalScroll)
        self._chat_scroll_expected_y = chat_log.scroll_y

        if self._scroll_after_refresh_pending:
            return

        self._scroll_after_refresh_pending = True

        def _after_refresh_scroll() -> None:
            self._scroll_after_refresh_pending = False
            expected_y = self._chat_scroll_expected_y
            self._chat_scroll_expected_y = None
            chat_log_widget = self.query_one("#chat-log", VerticalScroll)
            if expected_y is not None and chat_log_widget.scroll_y >= max(
                expected_y - 1,
                0,
            ):
                self._scroll_chat_log_to_end()

        self.call_after_refresh(_after_refresh_scroll)

    def _scroll_agent_column_to_end(self, agent_key: str) -> None:
        column = self._agent_columns.get(agent_key)
        if column is None:
            return
        column.root.scroll_end(animate=False, force=True, immediate=True)

    def _auto_scroll_agent_column(self, agent_key: str) -> None:
        self._scroll_agent_column_to_end(agent_key)
        column = self._agent_columns.get(agent_key)
        if column is not None:
            self._agent_scroll_expected_y[agent_key] = column.root.scroll_y

        if agent_key in self._agent_scroll_pending:
            return

        self._agent_scroll_pending.add(agent_key)

        def _after_refresh_scroll() -> None:
            self._agent_scroll_pending.discard(agent_key)
            expected_y = self._agent_scroll_expected_y.pop(agent_key, None)
            column_widget = self._agent_columns.get(agent_key)
            if column_widget is None:
                return
            if expected_y is not None and column_widget.root.scroll_y >= max(
                expected_y - 1,
                0,
            ):
                self._scroll_agent_column_to_end(agent_key)

        self.call_after_refresh(_after_refresh_scroll)

    def _auto_scroll_for_model(self, model_call_id: str) -> None:
        agent_key = self._model_agent_key.get(
            model_call_id,
            self._resolve_agent_key(model_call_id),
        )
        self._auto_scroll_agent_column(agent_key)

    def _auto_scroll_for_tool(self, tool_call_id: str) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return
        self._auto_scroll_for_model(tool.model_call_id)

    def _schedule_agent_column_layout(self) -> None:
        if self._agent_layout_pending:
            return

        self._agent_layout_pending = True

        def _after_refresh_layout() -> None:
            self._agent_layout_pending = False
            self._reflow_agent_column_widths()

        self.call_after_refresh(_after_refresh_layout)

    def _reflow_agent_column_widths(self) -> None:
        if not self._agent_columns:
            return

        column_count = len(self._agent_columns)

        if column_count <= 1:
            target_width = "100%"
        elif column_count == 2:
            target_width = "50%"
        else:
            target_width = "33%"

        for column in self._agent_columns.values():
            column.root.styles.min_width = "33%"
            column.root.styles.width = target_width

    def _update_render_stats_widget(self) -> None:
        stats_widget = self.query_one("#render-stats", Static)
        total_blocks = self._total_blocks
        archived_blocks = self._archived_blocks
        active_models = self._active_models
        active_tools = self._active_tools
        live_blocks = max(total_blocks - archived_blocks, 0)
        stats_widget.update(
            "Render "
            f"| live {live_blocks}/{total_blocks} "
            f"| archived {archived_blocks} "
            f"| active model {active_models} "
            f"| active tool {active_tools} "
            f"| flush {self._render_flush_count} "
            f"| prune {self._archive_count}"
        )

    def _is_agent_column_near_bottom(self, agent_key: str) -> bool:
        column = self._agent_columns.get(agent_key)
        if column is None:
            return False

        if column.root.max_scroll_y <= 0:
            return True

        return (
            column.root.scroll_y
            >= column.root.max_scroll_y - self._virtualize_near_bottom_margin
        )

    def _register_render_block(self, block: _RenderBlock) -> None:
        blocks = self._agent_blocks.setdefault(block.agent_key, [])
        blocks.append(block)
        self._total_blocks += 1
        self._update_render_stats_widget()

    def _is_block_pinned(self, block: _RenderBlock) -> bool:
        if block.kind != "model" or block.model_call_id is None:
            return False

        model = self._models.get(block.model_call_id)
        if model is None:
            return False

        return (not model.finished) or bool(model.running_tool_call_ids)

    def _compose_tool_plain_text(self, tool: ToolCallCard) -> str:
        lines: list[str] = [f"Tool: {tool.tool_name}", f"Status: {tool.status}"]

        if tool.arguments:
            lines.append("Arguments:")
            lines.append(json.dumps(tool.arguments, ensure_ascii=False, indent=2))

        if tool.output.strip():
            lines.append("Output:")
            lines.append(tool.output.rstrip())

        if tool.result_markdown.strip():
            lines.append("Result:")
            lines.append(tool.result_markdown.strip())

        if tool.stats_line.strip():
            lines.append(tool.stats_line.strip())

        return "\n".join(lines)

    def _compose_model_plain_text(self, model_call_id: str) -> str:
        model = self._models.get(model_call_id)
        if model is None:
            return "Assistant"

        lines: list[str] = ["Assistant"]

        if model.reasoning.strip():
            lines.append("Reasoning:")
            lines.append(model.reasoning.strip())

        if model.content.strip():
            lines.append("Content:")
            lines.append(model.content.strip())

        for tool_call_id in model.tool_call_ids:
            tool = self._tools.get(tool_call_id)
            if tool is None:
                continue
            lines.append(self._compose_tool_plain_text(tool))

        if model.stats_line.strip():
            lines.append(model.stats_line.strip())

        return "\n\n".join(lines)

    def _compose_block_plain_text(self, block: _RenderBlock) -> str:
        if block.kind == "user":
            return f"User\n{block.user_text.strip()}"

        if block.model_call_id is None:
            return "Assistant"

        return self._compose_model_plain_text(block.model_call_id)

    def _summarize_for_archive(self, block: _RenderBlock) -> str:
        role = "User" if block.kind == "user" else "Assistant"
        full_text = self._compose_block_plain_text(block).replace("\n", " ").strip()
        if len(full_text) > self.ARCHIVE_PREVIEW_CHARS:
            full_text = f"{full_text[: self.ARCHIVE_PREVIEW_CHARS - 3]}..."
        return f"[archived {role.lower()} block] {full_text}"

    async def _archive_render_block(self, block: _RenderBlock) -> bool:
        if block.archived or self._is_block_pinned(block):
            return False

        for child in list(block.root.children):
            await child.remove()

        await block.root.mount(
            Static(
                self._summarize_for_archive(block),
                classes="archived-block",
            )
        )
        block.archived = True
        self._archive_count += 1
        self._archived_blocks += 1
        self._update_render_stats_widget()
        return True

    async def _virtualize_agent_column_if_needed(self, agent_key: str) -> None:
        blocks = self._agent_blocks.get(agent_key, [])
        live_blocks = [block for block in blocks if not block.archived]
        if len(live_blocks) <= self._virtual_keep_live_blocks:
            return

        if not self._is_agent_column_near_bottom(agent_key):
            return

        overflow = len(live_blocks) - self._virtual_keep_live_blocks
        for block in live_blocks:
            if overflow <= 0:
                break
            archived = await self._archive_render_block(block)
            if archived:
                overflow -= 1

    def _compose_agent_transcript(
        self, agent_key: str, blocks: list[_RenderBlock]
    ) -> str:
        if not blocks:
            return ""

        header = self._build_agent_column_label(agent_key)
        rendered_blocks: list[str] = []
        for block in blocks:
            text = self._compose_block_plain_text(block)
            if text.strip():
                rendered_blocks.append(text)

        if not rendered_blocks:
            return ""
        return f"[{header}]\n\n" + "\n\n".join(rendered_blocks)

    def _build_copyable_transcript(self) -> str:
        sections: list[str] = []

        if self._system_hints:
            sections.append("[System]\n\n" + "\n".join(self._system_hints))

        for agent_key, blocks in self._agent_blocks.items():
            section = self._compose_agent_transcript(agent_key, blocks)
            if section:
                sections.append(section)

        sections.extend(self._detached_transcript_sections)
        return "\n\n---\n\n".join(section for section in sections if section.strip())

    async def action_copy_all_text(self) -> None:
        transcript = self._build_copyable_transcript()
        if not transcript.strip():
            await self._append_system_hint("Nothing to copy yet.")
            return

        try:
            self.copy_to_clipboard(transcript)
        except Exception as exc:
            await self._append_system_hint(f"Clipboard copy failed: {exc}")
            return

        await self._append_system_hint(
            f"Copied transcript ({len(transcript)} chars) to clipboard."
        )

    def _set_chat_placeholder(self) -> None:
        input_widget = self.query_one("#chat-input", Input)
        input_widget.placeholder = self.DEFAULT_INPUT_PLACEHOLDER

    def _set_tool_input_placeholder(self, prompt: str) -> None:
        input_widget = self.query_one("#chat-input", Input)
        prompt_text = prompt.strip()
        if prompt_text:
            input_widget.placeholder = f"Tool input: {prompt_text}"
        else:
            input_widget.placeholder = (
                "Tool input required; type response and press Enter"
            )

    def _refresh_input_widget_state(self) -> None:
        input_widget = self.query_one("#chat-input", Input)
        pending_request = self._input_router.peek_pending_tool_request()

        if pending_request is not None:
            input_widget.disabled = False
            self._set_tool_input_placeholder(pending_request.prompt)
            input_widget.focus()
            return

        self._set_chat_placeholder()
        if self._busy:
            input_widget.disabled = True
            return
        input_widget.disabled = False
        input_widget.focus()

    def _prepend_abort_marker(self, text: str) -> str:
        marker = "我要打断你的回复。"
        if not text:
            return marker
        stripped = text.lstrip()
        if stripped.startswith(marker):
            return text
        return f"{marker}\n{text}"

    def _request_abort_current_turn(self, reason: str = "steering") -> None:
        if self._active_abort_signal is None:
            return
        if self._active_abort_signal.is_aborted:
            return
        self._active_abort_signal.abort(reason)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        input_widget = self.query_one("#chat-input", Input)
        stripped_value = event.value.strip()
        lowered_value = stripped_value.lower()

        if lowered_value in {"/exit", "/quit", "/q"}:
            self.exit()
            return

        if lowered_value in {"/copy", "/copyall"}:
            input_widget.value = ""
            await self.action_copy_all_text()
            self._refresh_input_widget_state()
            return

        force_chat = False
        input_payload = event.value
        if lowered_value.startswith("/chat"):
            force_chat = True
            input_payload = stripped_value[5:].lstrip()

        route_result = self._input_router.route_input(
            UserInputEvent(
                text=input_payload,
                force_chat=force_chat,
            )
        )

        if route_result.route == "tool":
            if route_result.request is not None:
                await self.append_tool_output(
                    route_result.request.tool_call_id,
                    f"> {route_result.submitted_text}\n",
                )
                await self.set_tool_status(route_result.request.tool_call_id, "running")

            input_widget.value = ""
            self._refresh_input_widget_state()
            return

        if route_result.route == "rejected":
            await self._append_system_hint(
                route_result.reason or "Tool input request expired."
            )
            input_widget.value = ""
            self._refresh_input_widget_state()
            return

        if route_result.route == "noop":
            return

        user_text = route_result.chat_text

        if self._busy:
            input_widget.value = ""
            steered_text = self._prepend_abort_marker(user_text)
            await self._append_user_message(steered_text)
            self._pending_user_text = steered_text
            self._request_abort_current_turn()
            self._refresh_input_widget_state()
            return

        input_widget.value = ""

        await self._append_user_message(user_text)
        self._busy = True

        self.run_worker(self._run_turn(user_text), thread=False)

    async def _run_turn(self, user_text: str) -> None:
        input_widget = self.query_one("#chat-input", Input)
        abort_signal = AbortSignal()
        self._active_abort_signal = abort_signal
        pending_text: Optional[str] = None
        try:
            call_kwargs = dict(self.static_kwargs)
            call_kwargs[self.input_param] = user_text
            if self.history_param:
                call_kwargs[self.history_param] = self.history

            call_kwargs[ABORT_SIGNAL_PARAM] = abort_signal
            stream = self.agent_func(**call_kwargs)
            new_history = await consume_react_stream(
                stream,
                adapter=self,
                custom_hooks=self.custom_hooks,
                abort_signal=abort_signal,
            )
            if new_history:
                self.history = new_history
        except Exception as exc:
            await self._append_system_hint(f"Agent error: {exc}")
        finally:
            self._busy = False
            self._input_router.clear_all_requests()
            if self._active_abort_signal is abort_signal:
                self._active_abort_signal = None
            pending_text = self._pending_user_text
            self._pending_user_text = None

        if pending_text:
            self._busy = True
            self.run_worker(self._run_turn(pending_text), thread=False)
        else:
            self._refresh_input_widget_state()

    async def _append_user_message(self, text: str) -> None:
        bubble = VerticalGroup(classes="bubble user-bubble")
        role = Static("User", classes="role")
        body = Static(classes="body")
        body.update(Markdown(text))

        await self._mount_into_agent_column(self.MAIN_AGENT_KEY, bubble)
        await bubble.mount(role)
        await bubble.mount(body)

        block_id = f"user::{self._next_user_block_index}"
        self._next_user_block_index += 1
        self._register_render_block(
            _RenderBlock(
                block_id=block_id,
                agent_key=self.MAIN_AGENT_KEY,
                kind="user",
                root=bubble,
                user_text=text,
            )
        )

        await self._virtualize_agent_column_if_needed(self.MAIN_AGENT_KEY)
        self._auto_scroll_agent_column(self.MAIN_AGENT_KEY)

    async def _append_system_hint(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        hint = Static(Text(text, style="#7f8798"))
        await chat_log.mount(hint)
        self._system_hints.append(text)
        self._auto_scroll_chat_log()

    def _is_fork_model_call_id(self, model_call_id: str) -> bool:
        return model_call_id.startswith("fork::")

    def _extract_fork_id(self, model_call_id: str) -> str:
        if not self._is_fork_model_call_id(model_call_id):
            return model_call_id
        return model_call_id.split("::", 1)[1]

    def _resolve_agent_key(self, model_call_id: str) -> str:
        if self._is_fork_model_call_id(model_call_id):
            return self._extract_fork_id(model_call_id)
        return self.MAIN_AGENT_KEY

    def _normalize_dom_id_suffix(self, value: str) -> str:
        normalized_chars: list[str] = []
        for char in value:
            if char.isalnum() or char == "_":
                normalized_chars.append(char)
            else:
                normalized_chars.append("_")

        normalized = "".join(normalized_chars).strip("_")
        if normalized:
            return normalized
        return "agent"

    def _build_agent_column_dom_id(self, agent_key: str) -> str:
        if agent_key == self.MAIN_AGENT_KEY:
            return "agent-column-main"
        return f"agent-column-fork_{self._normalize_dom_id_suffix(agent_key)}"

    def _build_agent_column_label(self, agent_key: str) -> str:
        if agent_key == self.MAIN_AGENT_KEY:
            return "Main Agent"
        return f"Fork {agent_key}"

    def _build_loading_label(self, model_call_id: str) -> str:
        if self._is_fork_model_call_id(model_call_id):
            return "running fork"
        return "typing"

    @staticmethod
    def _sync_model_loading_label(model: _ModelWidgets) -> None:
        if model.content.strip():
            model.loading_label_widget.update("")
            model.loading_label_widget.display = False
            return

        if not model.loading_label_widget.display:
            model.loading_label_widget.display = True

    async def _ensure_agent_board(self) -> HorizontalScroll:
        if self._agent_board is not None:
            return self._agent_board

        chat_log = self.query_one("#chat-log", VerticalScroll)
        board = HorizontalScroll(classes="agent-board", id="agent-board")
        await chat_log.mount(board)
        self._agent_board = board
        self._schedule_agent_column_layout()
        return board

    async def _ensure_agent_column(self, agent_key: str) -> _AgentColumnWidgets:
        existing = self._agent_columns.get(agent_key)
        if existing is not None:
            return existing

        if (
            agent_key != self.MAIN_AGENT_KEY
            and self.MAIN_AGENT_KEY not in self._agent_columns
        ):
            await self._ensure_agent_column(self.MAIN_AGENT_KEY)

        board = await self._ensure_agent_board()
        column = VerticalScroll(
            classes="agent-column",
            id=self._build_agent_column_dom_id(agent_key),
        )
        header = Static(
            self._build_agent_column_label(agent_key),
            classes="agent-column-header",
        )
        message_list = VerticalGroup(classes="agent-column-messages")

        await board.mount(column)
        await column.mount(header)
        await column.mount(message_list)

        created = _AgentColumnWidgets(
            root=column,
            header_widget=header,
            message_list_widget=message_list,
        )
        self._agent_columns[agent_key] = created
        self._schedule_agent_column_layout()
        return created

    async def _mount_into_agent_column(
        self, agent_key: str, widget: VerticalGroup
    ) -> None:
        column = await self._ensure_agent_column(agent_key)
        await column.message_list_widget.mount(widget)

    async def _teardown_agent_column(self, agent_key: str) -> None:
        if agent_key == self.MAIN_AGENT_KEY:
            return

        self._agent_scroll_pending.discard(agent_key)
        self._agent_scroll_expected_y.pop(agent_key, None)

        column = self._agent_columns.pop(agent_key, None)
        if column is None:
            return

        removed_model_call_ids = [
            model_call_id
            for model_call_id, mapped_agent_key in self._model_agent_key.items()
            if mapped_agent_key == agent_key
        ]

        removed_blocks = self._agent_blocks.pop(agent_key, [])
        if removed_blocks:
            removed_archived = sum(1 for block in removed_blocks if block.archived)
            self._total_blocks = max(0, self._total_blocks - len(removed_blocks))
            self._archived_blocks = max(0, self._archived_blocks - removed_archived)
            transcript = self._compose_agent_transcript(agent_key, removed_blocks)
            if transcript:
                self._detached_transcript_sections.append(transcript)

        for model_call_id in removed_model_call_ids:
            model = self._models.get(model_call_id)
            if model is not None and not model.finished:
                self._active_models = max(0, self._active_models - 1)
            self._model_agent_key.pop(model_call_id, None)
            self._models.pop(model_call_id, None)

        removed_tool_call_ids = [
            tool_call_id
            for tool_call_id, tool in self._tools.items()
            if tool.model_call_id in removed_model_call_ids
        ]
        for tool_call_id in removed_tool_call_ids:
            tool = self._tools.get(tool_call_id)
            if tool is not None and tool.status == "running":
                self._active_tools = max(0, self._active_tools - 1)
            self._tools.pop(tool_call_id, None)
            self._input_router.clear_tool_requests(tool_call_id)

        await column.root.remove()
        self._schedule_agent_column_layout()
        self._update_render_stats_widget()

    async def _mount_model_root(
        self,
        model_call_id: str,
        root: VerticalGroup,
    ) -> None:
        agent_key = self._resolve_agent_key(model_call_id)
        self._model_agent_key[model_call_id] = agent_key
        await self._mount_into_agent_column(agent_key, root)

    def _schedule_model_render(self, model_call_id: str) -> None:
        model = self._models.get(model_call_id)
        if model is None or model.render_pending:
            return

        model.render_pending = True

        def _after_refresh_render() -> None:
            self.run_worker(self._flush_model_render(model_call_id), thread=False)

        self.call_after_refresh(_after_refresh_render)

    async def _flush_model_render(self, model_call_id: str) -> None:
        model = self._models.get(model_call_id)
        if model is None:
            return

        model.render_pending = False
        updated = False

        if model.content_dirty:
            model.content_dirty = False
            try:
                model.content_widget.update(Markdown(model.content or " "))
                self._sync_model_loading_label(model)
                updated = True
            except Exception:
                pass

        if model.reasoning_dirty:
            model.reasoning_dirty = False
            try:
                model.reasoning_widget.update(model.reasoning)
                updated = True
            except Exception:
                pass

        if updated:
            self._render_flush_count += 1
            self._update_render_stats_widget()

        self._auto_scroll_for_model(model_call_id)

    def _schedule_tool_render(self, tool_call_id: str) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None or tool.render_pending:
            return

        tool.render_pending = True

        def _after_refresh_render() -> None:
            self.run_worker(self._flush_tool_render(tool_call_id), thread=False)

        self.call_after_refresh(_after_refresh_render)

    async def _flush_tool_render(self, tool_call_id: str) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        tool.render_pending = False
        try:
            updated = tool.flush_render()
        except Exception:
            updated = False

        if updated:
            self._render_flush_count += 1
            self._update_render_stats_widget()

        self._auto_scroll_for_tool(tool_call_id)

    # ------------------------------------------------------------------
    # Adapter methods used by consume_react_stream
    # ------------------------------------------------------------------

    async def start_model_response(self, model_call_id: str) -> None:
        existing_model = self._models.get(model_call_id)
        if existing_model is not None and not existing_model.finished:
            self._active_models = max(0, self._active_models - 1)

        root = VerticalGroup(classes="bubble model-bubble")
        role = Static("Assistant", classes="role")
        loading_row = Horizontal(classes="assistant-loading-row")
        loading_label_widget = Static(
            self._build_loading_label(model_call_id),
            classes="assistant-loading-label",
        )
        loading_widget = DotPulse(classes="assistant-loading")
        reasoning_widget = Static("", classes="reasoning")
        content_widget = Static("", classes="body")
        tool_list_widget = VerticalGroup(classes="tool-list")
        stats_widget = Static("", classes="stats")

        await self._mount_model_root(model_call_id, root)
        await root.mount(role)
        await root.mount(reasoning_widget)
        await root.mount(content_widget)
        await root.mount(tool_list_widget)
        await root.mount(loading_row)
        await loading_row.mount(loading_label_widget)
        await loading_row.mount(loading_widget)
        await root.mount(stats_widget)

        agent_key = self._model_agent_key.get(model_call_id, self.MAIN_AGENT_KEY)

        model_widgets = _ModelWidgets(
            model_call_id=model_call_id,
            agent_key=agent_key,
            root=root,
            loading_row=loading_row,
            loading_widget=loading_widget,
            loading_label_widget=loading_label_widget,
            reasoning_widget=reasoning_widget,
            content_widget=content_widget,
            tool_list_widget=tool_list_widget,
            stats_widget=stats_widget,
        )
        self._models[model_call_id] = model_widgets
        self._sync_model_loading_label(model_widgets)
        self._active_models += 1

        self._register_render_block(
            _RenderBlock(
                block_id=model_call_id,
                agent_key=agent_key,
                kind="model",
                root=root,
                model_call_id=model_call_id,
            )
        )

        await self._virtualize_agent_column_if_needed(agent_key)
        self._auto_scroll_for_model(model_call_id)

    async def append_model_content(
        self, model_call_id: str, content_delta: str
    ) -> None:
        model = self._models.get(model_call_id)
        if model is None:
            return

        model.content += content_delta
        model.content_dirty = True
        self._sync_model_loading_label(model)
        self._schedule_model_render(model_call_id)

    async def append_model_reasoning(
        self,
        model_call_id: str,
        reasoning_delta: str,
    ) -> None:
        model = self._models.get(model_call_id)
        if model is None:
            return

        model.reasoning += reasoning_delta
        model.reasoning_dirty = True
        self._schedule_model_render(model_call_id)

    async def finish_model_response(self, model_call_id: str, stats_line: str) -> None:
        model = self._models.get(model_call_id)
        if model is None:
            return

        model.stats_line = stats_line
        if not model.finished:
            model.finished = True
            self._active_models = max(0, self._active_models - 1)
        else:
            model.finished = True

        if model.loading_row.parent is not None:
            await model.loading_row.remove()

        model.stats_widget.update(stats_line)
        self._update_render_stats_widget()

        if self._is_fork_model_call_id(model_call_id) and stats_line in {
            "fork | completed",
            "fork | error",
        }:
            agent_key = self._model_agent_key.get(model_call_id)
            if agent_key is not None:
                await self._teardown_agent_column(agent_key)
                return

        await self._virtualize_agent_column_if_needed(model.agent_key)

        self._auto_scroll_for_model(model_call_id)

    async def start_tool_call(
        self,
        model_call_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        existing_tool = self._tools.get(tool_call_id)
        if existing_tool is not None:
            previous_status = existing_tool.status
            existing_tool.set_tool_name(tool_name)
            existing_tool.set_arguments(dict(arguments))
            existing_tool.set_status("running")
            if previous_status != "running":
                self._active_tools += 1
            self._schedule_tool_render(tool_call_id)

            model_for_existing_tool = self._models.get(existing_tool.model_call_id)
            if model_for_existing_tool is not None:
                model_for_existing_tool.running_tool_call_ids.add(tool_call_id)
                if tool_call_id not in model_for_existing_tool.tool_call_ids:
                    model_for_existing_tool.tool_call_ids.append(tool_call_id)

            self._update_render_stats_widget()
            self._auto_scroll_for_tool(tool_call_id)
            return

        model = self._models.get(model_call_id)
        if model is None:
            return

        tool_card = build_tool_call_card(
            tool_call_id=tool_call_id,
            model_call_id=model_call_id,
            tool_name=tool_name,
            arguments=dict(arguments),
            argument_formatter=format_tool_arguments_markdown,
        )
        self._tools[tool_call_id] = tool_card
        self._active_tools += 1

        await model.tool_list_widget.mount(tool_card)

        model.tool_call_ids.append(tool_call_id)
        model.running_tool_call_ids.add(tool_call_id)
        self._update_render_stats_widget()
        self._auto_scroll_for_tool(tool_call_id)

    async def append_tool_argument(
        self,
        tool_call_id: str,
        argname: str,
        argcontent_delta: str,
    ) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        tool.append_argument(argname, argcontent_delta)
        self._schedule_tool_render(tool_call_id)

    async def append_tool_output(self, tool_call_id: str, output_delta: str) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        tool.append_output(output_delta)
        self._schedule_tool_render(tool_call_id)

    async def set_tool_status(self, tool_call_id: str, status: str) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        previous_status = tool.status
        tool.set_status(status)
        if previous_status == "running" and status != "running":
            self._active_tools = max(0, self._active_tools - 1)
        elif previous_status != "running" and status == "running":
            self._active_tools += 1
        self._schedule_tool_render(tool_call_id)
        self._update_render_stats_widget()

    async def request_tool_input(
        self,
        tool_call_id: str,
        request_id: str,
        prompt: str,
    ) -> None:
        self._input_router.register_tool_request(
            tool_call_id=tool_call_id,
            request_id=request_id,
            prompt=prompt,
        )
        self._refresh_input_widget_state()

    async def clear_tool_input(self, tool_call_id: str) -> None:
        self._input_router.clear_tool_requests(tool_call_id)
        self._refresh_input_widget_state()

    async def finish_tool_call(
        self,
        tool_call_id: str,
        result_markdown: str,
        stats_line: str,
        success: bool,
    ) -> None:
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return

        previous_status = tool.status
        tool.set_status("success" if success else "error")
        if previous_status == "running":
            self._active_tools = max(0, self._active_tools - 1)
        tool.set_result_markdown(result_markdown)
        tool.set_stats(stats_line)
        self._schedule_tool_render(tool_call_id)

        model = self._models.get(tool.model_call_id)
        if model is not None:
            model.running_tool_call_ids.discard(tool_call_id)
            if model.finished:
                await self._virtualize_agent_column_if_needed(model.agent_key)

        self._update_render_stats_widget()
        self._auto_scroll_for_tool(tool_call_id)


__all__ = ["AgentTUIApp"]
