"""Regression tests for Textual TUI widget mounting."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import Mock

import pytest
from textual.css.query import NoMatches
from textual.containers import HorizontalScroll, VerticalScroll
from textual.widgets import Input, Static

from SimpleLLMFunc.hooks.input_stream import AgentInputRouter
from SimpleLLMFunc.hooks.stream import ReactOutput
from SimpleLLMFunc.utils.tui import app as app_module
from SimpleLLMFunc.utils.tui.app import AgentTUIApp
from SimpleLLMFunc.utils.tui.tool_cards.base import ToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.default import DefaultToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.execute_code import ExecuteCodeToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.file_tools import (
    ReadFileToolCallCard,
    ReadImageToolCallCard,
    SedToolCallCard,
)


async def _unused_agent(_message: str) -> AsyncGenerator[ReactOutput, None]:
    if False:
        yield None


def _make_app(input_router: AgentInputRouter | None = None) -> AgentTUIApp:
    return AgentTUIApp(
        agent_func=_unused_agent,
        input_param="message",
        history_param=None,
        static_kwargs={},
        input_router=input_router,
    )


@pytest.mark.asyncio
async def test_append_user_message_mounts_children_after_parent() -> None:
    """User message mount order should not raise Textual MountError."""
    app = _make_app()

    async with app.run_test():
        await app._append_user_message("hello")

        bubble = app.query_one(".user-bubble")
        bubble.query_one(".role", Static)
        bubble.query_one(".body", Static)


@pytest.mark.asyncio
async def test_model_and_tool_blocks_mount_children_after_parent() -> None:
    """Model and tool sections should mount without detached-parent errors."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-1",
            tool_name="execute_code",
            arguments={"code": "print(1)"},
        )

        app.query_one(".model-bubble")
        tool_block = app.query_one(".tool-call")
        tool_block.query_one(".tool-status", Static)


@pytest.mark.asyncio
async def test_assistant_bubble_shows_loading_indicator_while_running() -> None:
    """Assistant bubble should show loading indicator before model finishes."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")

        bubble = app.query_one(".model-bubble")
        loading = bubble.query_one(".assistant-loading", Static)
        label = bubble.query_one(".assistant-loading-label", Static)
        assert str(loading.content) in {".", "..", "..."}
        assert str(label.content) == "typing"


@pytest.mark.asyncio
async def test_assistant_bubble_hides_loading_indicator_when_finished() -> None:
    """Assistant bubble loading indicator should disappear after completion."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.finish_model_response("llm_call_1", "model | 0.01s")

        bubble = app.query_one(".model-bubble")
        with pytest.raises(NoMatches):
            bubble.query_one(".assistant-loading", Static)


@pytest.mark.asyncio
async def test_assistant_streaming_content_remains_visible_while_loading() -> None:
    """Streaming content should stay visible while the loading indicator is active."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "streaming text")
        await pilot.pause(0.05)

        bubble = app.query_one(".model-bubble")
        loading = bubble.query_one(".assistant-loading", Static)
        content = app._models["llm_call_1"].content_widget

        assert app._models["llm_call_1"].content == "streaming text"
        assert content.region.height > 0
        assert content.region.y < loading.region.y


@pytest.mark.asyncio
async def test_assistant_loading_indicator_is_compact_bottom_status() -> None:
    """Loading indicator should render as a compact bottom status, not a full overlay."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "hello")
        await pilot.pause(0.05)

        bubble = app.query_one(".model-bubble")
        loading = bubble.query_one(".assistant-loading", Static)

        assert loading.region.height == 1
        assert loading.region.width <= 3


@pytest.mark.asyncio
async def test_assistant_typing_label_hides_on_first_content_delta() -> None:
    """typing label should disappear immediately when first content arrives."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "h")

        label = app._models["llm_call_1"].loading_label_widget
        assert label.display is False


@pytest.mark.asyncio
async def test_assistant_loading_label_hides_after_content_starts_streaming() -> None:
    """Once content starts streaming, only the spinner should remain visible."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "hello")
        await pilot.pause(0.05)

        bubble = app.query_one(".model-bubble")
        loading = bubble.query_one(".assistant-loading", Static)
        label = bubble.query_one(".assistant-loading-label", Static)

        assert loading.region.width > 0
        assert label.region.width == 0


@pytest.mark.asyncio
async def test_fork_bubble_uses_running_fork_loading_label() -> None:
    """Fork model bubbles should use a fork-specific loading label."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("fork::fork_1")

        bubble = app.query_one(".model-bubble")
        label = bubble.query_one(".assistant-loading-label", Static)

        assert str(label.content) == "running fork"


@pytest.mark.asyncio
async def test_default_tool_card_factory_uses_base_tool_card_subclass() -> None:
    """Unknown tools should use the default ToolCallCard implementation."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-default",
            tool_name="lookup",
            arguments={"query": "hello"},
        )

        tool = app._tools["call-default"]
        assert isinstance(tool, ToolCallCard)
        assert isinstance(tool, DefaultToolCallCard)
        assert "- `query`: `hello`" in tool.arguments_markdown


@pytest.mark.asyncio
async def test_execute_code_uses_specialized_tool_card() -> None:
    """execute_code should render with its dedicated card implementation."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-exec",
            tool_name="execute_code",
            arguments={"code": "print(1)", "timeout_seconds": 30},
        )

        tool = app._tools["call-exec"]
        assert isinstance(tool, ExecuteCodeToolCallCard)
        assert "## Code" in tool.arguments_markdown
        assert "```python" in tool.arguments_markdown
        assert "## Runtime Controls" in tool.arguments_markdown
        assert "timeout_seconds" in tool.arguments_markdown


@pytest.mark.asyncio
async def test_read_file_uses_specialized_file_tool_card() -> None:
    """read_file should render path and line range with file-focused layout."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-read",
            tool_name="read_file",
            arguments={"path": "src/app.py", "start_line": 10, "end_line": 25},
        )

        tool = app._tools["call-read"]
        assert isinstance(tool, ReadFileToolCallCard)
        assert "## File" in tool.arguments_markdown
        assert "`src/app.py`" in tool.arguments_markdown
        assert "## Line Range" in tool.arguments_markdown
        assert "10-25" in tool.arguments_markdown


@pytest.mark.asyncio
async def test_read_image_uses_specialized_file_tool_card() -> None:
    """read_image should render path with the file-focused layout."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-read-image",
            tool_name="read_image",
            arguments={"path": "img/chart.png"},
        )

        tool = app._tools["call-read-image"]
        assert isinstance(tool, ReadImageToolCallCard)
        assert "## File" in tool.arguments_markdown
        assert "`img/chart.png`" in tool.arguments_markdown
        assert "## Line Range" not in tool.arguments_markdown


@pytest.mark.asyncio
async def test_sed_uses_specialized_file_tool_card() -> None:
    """sed should render a diff-like regex replacement preview."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-sed",
            tool_name="sed",
            arguments={
                "path": "src/app.py",
                "start_line": 10,
                "end_line": 25,
                "pattern_to_be_replaced": "foo.+bar",
                "new_string": "baz",
            },
        )

        tool = app._tools["call-sed"]
        assert isinstance(tool, SedToolCallCard)
        assert "## Edit Preview" in tool.arguments_markdown
        assert "```diff" in tool.arguments_markdown
        assert "- /foo.+bar/" in tool.arguments_markdown
        assert "+ baz" in tool.arguments_markdown
        assert "foo.+bar" in tool.arguments_markdown


@pytest.mark.asyncio
async def test_short_message_height_is_not_viewport_sized() -> None:
    """Short message bubbles should size to content instead of filling viewport."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app._append_user_message("hello")
        await pilot.pause(0.05)

        bubble = app.query_one(".user-bubble")
        chat_log = app.query_one("#chat-log", VerticalScroll)

        assert bubble.region.height < (chat_log.region.height // 2)


@pytest.mark.asyncio
async def test_ctrl_q_quits_app() -> None:
    """Ctrl+Q should provide a direct exit keybinding."""
    app = _make_app()

    async with app.run_test() as pilot:
        assert app.is_running
        await pilot.press("ctrl+q")
        await pilot.pause(0.05)
        assert not app.is_running


@pytest.mark.asyncio
async def test_streaming_model_content_auto_scrolls_to_bottom() -> None:
    """Model streaming updates should keep model column pinned to latest content."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "word " * 2200)
        await pilot.pause(0.05)

        main_column = app.query_one("#agent-column-main", VerticalScroll)
        assert main_column.max_scroll_y > 0

        main_column.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.02)
        assert main_column.scroll_y == 0

        await app.append_model_content("llm_call_1", "tail " * 200)
        await pilot.pause(0.05)

        assert main_column.scroll_y >= main_column.max_scroll_y - 0.5


@pytest.mark.asyncio
async def test_streaming_tool_output_auto_scrolls_to_bottom() -> None:
    """Tool streaming output should keep owning agent column pinned."""
    app = _make_app()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-1",
            tool_name="execute_code",
            arguments={"code": "print(1)"},
        )
        await app.append_tool_output(
            "call-1", "".join(f"line {i}\n" for i in range(160))
        )
        await pilot.pause(0.05)

        main_column = app.query_one("#agent-column-main", VerticalScroll)
        assert main_column.max_scroll_y > 0

        main_column.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.02)
        assert main_column.scroll_y == 0

        await app.append_tool_output("call-1", "tail line\n")
        await pilot.pause(0.05)

        assert main_column.scroll_y >= main_column.max_scroll_y - 0.5


@pytest.mark.asyncio
async def test_pending_tool_input_submits_to_pyrepl() -> None:
    """When PyRepl requests input, Enter should submit to request id."""
    submit_mock = Mock(return_value=True)
    app = _make_app(input_router=AgentInputRouter(submit_tool_input=submit_mock))

    async with app.run_test() as pilot:
        app._busy = True
        await app.request_tool_input(
            tool_call_id="call-1",
            request_id="req-1",
            prompt="Name: ",
        )

        input_widget = app.query_one("#chat-input", Input)
        assert not input_widget.disabled
        assert "Tool input" in input_widget.placeholder

        await pilot.click("#chat-input")
        await pilot.press("A", "l", "i", "c", "e", "enter")
        await pilot.pause(0.05)

        submit_mock.assert_called_once_with("req-1", "Alice")
        assert not app._input_router.has_pending_tool_requests()
        assert input_widget.disabled


@pytest.mark.asyncio
async def test_chat_command_bypasses_pending_tool_input() -> None:
    """/chat should bypass pending tool-input routing."""
    submit_mock = Mock(return_value=True)
    app = _make_app(input_router=AgentInputRouter(submit_tool_input=submit_mock))

    async with app.run_test() as pilot:
        app._busy = True
        await app.request_tool_input(
            tool_call_id="call-2",
            request_id="req-2",
            prompt="Age: ",
        )

        await pilot.click("#chat-input")
        await pilot.press("/", "c", "h", "a", "t", "space", "h", "i", "enter")
        await pilot.pause(0.05)

        submit_mock.assert_not_called()
        assert app._input_router.has_pending_tool_requests()


@pytest.mark.asyncio
async def test_main_stream_uses_single_agent_column() -> None:
    """Without forks, the message board should stay in single-column mode."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("llm_call_1")

        board = app.query_one("#agent-board")
        columns = list(board.query(".agent-column"))

        assert len(columns) == 1
        app.query_one("#agent-column-main", VerticalScroll)


@pytest.mark.asyncio
async def test_fork_stream_adds_peer_column_and_unloads_on_finish() -> None:
    """Fork agent should get its own column and unload after completion."""
    app = _make_app()

    async with app.run_test() as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")

        board = app.query_one("#agent-board")
        columns = list(board.query(".agent-column"))
        assert len(columns) == 2
        app.query_one("#agent-column-main", VerticalScroll)
        app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        await app.finish_model_response("fork::fork_1", "fork | completed")
        await pilot.pause(0.05)

        with pytest.raises(NoMatches):
            app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        board = app.query_one("#agent-board")
        columns = list(board.query(".agent-column"))

        assert len(columns) == 1
        app.query_one("#agent-column-main", VerticalScroll)


@pytest.mark.asyncio
async def test_fork_error_column_unloads_after_failure() -> None:
    """Errored fork column should unload after failure."""
    app = _make_app()

    async with app.run_test() as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.append_model_reasoning("fork::fork_1", "TypeError: missing argument")
        await app.finish_model_response("fork::fork_1", "fork | error")
        await pilot.pause(0.05)

        await pilot.pause(0.05)

        with pytest.raises(NoMatches):
            app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        board = app.query_one("#agent-board")
        columns = list(board.query(".agent-column"))
        assert len(columns) == 1
        app.query_one("#agent-column-main", VerticalScroll)


@pytest.mark.asyncio
async def test_agent_board_scrolls_horizontally_when_more_than_three_columns() -> None:
    """Agent board should enable horizontal scrolling when columns exceed width."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.start_model_response("fork::fork_2")
        await app.start_model_response("fork::fork_3")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board", HorizontalScroll)
        main_column = app.query_one("#agent-column-main", VerticalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        one_third = max(1, board.size.width // 3)
        assert main_column.region.width >= one_third - 1
        assert fork_column.region.width >= one_third - 1
        assert main_column.region.width <= one_third + 1
        assert fork_column.region.width <= one_third + 1
        assert main_column.region.width < board.size.width
        assert board.max_scroll_x > 0


@pytest.mark.asyncio
async def test_two_agent_columns_use_half_width_without_horizontal_scroll() -> None:
    """Two columns should split viewport in half with no horizontal overflow."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board", HorizontalScroll)
        main_column = app.query_one("#agent-column-main", VerticalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        one_half = max(1, board.size.width // 2)
        assert abs(main_column.region.width - one_half) <= 1
        assert abs(fork_column.region.width - one_half) <= 1
        assert board.max_scroll_x == 0


@pytest.mark.asyncio
async def test_three_agent_columns_fit_without_horizontal_scroll() -> None:
    """Exactly three columns should fit viewport without horizontal scroll."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.start_model_response("fork::fork_2")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board", HorizontalScroll)
        main_column = app.query_one("#agent-column-main", VerticalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)

        one_third = max(1, board.size.width // 3)
        assert abs(main_column.region.width - one_third) <= 1
        assert abs(fork_column.region.width - one_third) <= 1
        assert board.max_scroll_x == 0


@pytest.mark.asyncio
async def test_fork_streaming_remains_visible_with_horizontal_scroll_layout() -> None:
    """Fork streaming should still render in >3-column horizontal layout."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.start_model_response("fork::fork_2")
        await app.start_model_response("fork::fork_3")
        await app.append_model_content("fork::fork_3", "streaming token")
        await pilot.pause(0.05)

        board = app.query_one("#agent-board", HorizontalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_3", VerticalScroll)

        one_third = max(1, board.size.width // 3)
        assert fork_column.region.width >= one_third - 1
        assert fork_column.region.width <= one_third + 1
        assert app._models["fork::fork_3"].content == "streaming token"


@pytest.mark.asyncio
async def test_same_fork_model_call_id_creates_new_assistant_bubble() -> None:
    """Repeated fork model start should append a new assistant bubble."""
    app = _make_app()

    async with app.run_test():
        await app.start_model_response("fork::fork_1")
        await app.append_model_content("fork::fork_1", "first")

        await app.start_model_response("fork::fork_1")
        await app.append_model_content("fork::fork_1", "second")

        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)
        bubbles = list(fork_column.query(".model-bubble"))

        assert len(bubbles) == 2


@pytest.mark.asyncio
async def test_agent_columns_keep_independent_scroll_positions() -> None:
    """Appending to one column should not move sibling column scroll."""
    app = _make_app()

    async with app.run_test(size=(120, 28)) as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_model_response("fork::fork_1")
        await app.append_model_content("llm_call_1", "main " * 2000)
        await app.append_model_content("fork::fork_1", "fork " * 2000)
        await pilot.pause(0.05)

        main_column = app.query_one("#agent-column-main", VerticalScroll)
        fork_column = app.query_one("#agent-column-fork_fork_1", VerticalScroll)
        assert main_column.max_scroll_y > 0
        assert fork_column.max_scroll_y > 0

        main_column.scroll_home(animate=False, immediate=True)
        fork_column.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.02)

        await app.append_model_content("llm_call_1", "tail " * 300)
        await pilot.pause(0.05)

        assert main_column.scroll_y >= main_column.max_scroll_y - 0.5
        assert fork_column.scroll_y == 0


@pytest.mark.asyncio
async def test_copy_all_text_contains_user_model_and_tool_content() -> None:
    """Copy-all action should include key rendered transcript sections."""
    app = _make_app()
    copied: list[str] = []

    def _copy_to_clipboard(text: str) -> None:
        copied.append(text)

    app.copy_to_clipboard = _copy_to_clipboard  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await app._append_user_message("hello from user")
        await app.start_model_response("llm_call_1")
        await app.append_model_content("llm_call_1", "assistant says hi")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-1",
            tool_name="execute_code",
            arguments={"code": "print('ok')"},
        )
        await app.append_tool_output("call-1", "ok\n")
        await app.finish_tool_call(
            tool_call_id="call-1",
            result_markdown="done",
            stats_line="Tool | 0.01s | success",
            success=True,
        )
        await app.finish_model_response("llm_call_1", "model | 0.02s")
        await pilot.pause(0.05)

        await app.action_copy_all_text()

    assert copied
    transcript = copied[-1]
    assert "hello from user" in transcript
    assert "assistant says hi" in transcript
    assert "Tool: execute_code" in transcript
    assert "ok" in transcript


@pytest.mark.asyncio
async def test_copy_command_routes_to_clipboard_action() -> None:
    """/copy command should trigger copy-all without sending chat input."""
    copy_mock = Mock()
    app = _make_app()
    app.copy_to_clipboard = copy_mock  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await app._append_user_message("copy me")
        await pilot.click("#chat-input")
        await pilot.press("/", "c", "o", "p", "y", "enter")
        await pilot.pause(0.05)

    copy_mock.assert_called_once()


@pytest.mark.asyncio
async def test_virtualization_archives_old_blocks_but_copy_keeps_full_text() -> None:
    """Old off-screen bubbles should archive while transcript copy remains complete."""
    app = _make_app()
    app._virtual_keep_live_blocks = 6
    copied: list[str] = []
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]

    async with app.run_test(size=(100, 28)) as pilot:
        for idx in range(18):
            await app._append_user_message(f"msg-{idx}")
        await pilot.pause(0.05)

        assert app._archive_count > 0
        assert len(list(app.query(".archived-block"))) > 0

        await app.action_copy_all_text()

    assert copied
    transcript = copied[-1]
    assert "msg-0" in transcript
    assert "msg-17" in transcript


@pytest.mark.asyncio
async def test_render_stats_counters_track_lifecycle_changes() -> None:
    """Render stats counters should update on model/tool lifecycle."""
    app = _make_app()

    async with app.run_test():
        assert app._total_blocks == 0
        assert app._archived_blocks == 0
        assert app._active_models == 0
        assert app._active_tools == 0

        await app._append_user_message("hello")
        assert app._total_blocks == 1
        assert app._archived_blocks == 0

        await app.start_model_response("llm_call_1")
        assert app._total_blocks == 2
        assert app._active_models == 1

        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-1",
            tool_name="search",
            arguments={"query": "hi"},
        )
        assert app._active_tools == 1

        await app.finish_tool_call(
            tool_call_id="call-1",
            result_markdown="ok",
            stats_line="Tool | 0.01s | success",
            success=True,
        )
        assert app._active_tools == 0

        await app.finish_model_response("llm_call_1", "model | 0.01s")
        assert app._active_models == 0

        block = app._agent_blocks[app.MAIN_AGENT_KEY][0]
        archived = await app._archive_render_block(block)
        assert archived is True
        assert app._archived_blocks == 1


@pytest.mark.asyncio
async def test_tool_argument_markdown_cache_updates_changed_key_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool argument rendering should only reformat changed arguments."""
    app = _make_app()
    calls: list[list[str]] = []
    original = app_module.format_tool_arguments_markdown

    def _spy(arguments: dict, tool_name: str | None = None) -> str:
        calls.append(list(arguments.keys()))
        return original(arguments, tool_name=tool_name)

    monkeypatch.setattr(app_module, "format_tool_arguments_markdown", _spy)

    async with app.run_test() as pilot:
        await app.start_model_response("llm_call_1")
        await app.start_tool_call(
            model_call_id="llm_call_1",
            tool_call_id="call-1",
            tool_name="lookup",
            arguments={"alpha": "1", "beta": "2"},
        )
        await pilot.pause(0.05)

        assert {tuple(keys) for keys in calls} == {("alpha",), ("beta",)}

        calls.clear()
        await app.append_tool_argument("call-1", "alpha", "3")
        await pilot.pause(0.05)

        assert calls == [["alpha"]]
        tool = app._tools["call-1"]
        assert "- `alpha`: `13`" in tool.arguments_markdown
        assert "- `beta`: `2`" in tool.arguments_markdown
