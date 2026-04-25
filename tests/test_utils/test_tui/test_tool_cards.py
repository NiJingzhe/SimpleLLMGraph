"""Tests for standalone tool-card widgets and factory selection."""

from __future__ import annotations

from SimpleLLMFunc.utils.tui.tool_cards.default import DefaultToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.execute_code import ExecuteCodeToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.factory import build_tool_call_card
from SimpleLLMFunc.utils.tui.tool_cards.file_tools import (
    EchoIntoToolCallCard,
    GrepToolCallCard,
    ReadFileToolCallCard,
    ReadImageToolCallCard,
    SedToolCallCard,
)


def test_factory_returns_builtin_specialized_cards() -> None:
    """Factory should map builtin tools to their dedicated card types."""

    execute_card = build_tool_call_card(
        tool_call_id="call-1",
        model_call_id="llm-1",
        tool_name="execute_code",
    )
    read_card = build_tool_call_card(
        tool_call_id="call-2",
        model_call_id="llm-1",
        tool_name="read_file",
    )
    read_image_card = build_tool_call_card(
        tool_call_id="call-2b",
        model_call_id="llm-1",
        tool_name="read_image",
    )
    grep_card = build_tool_call_card(
        tool_call_id="call-3",
        model_call_id="llm-1",
        tool_name="grep",
    )
    sed_card = build_tool_call_card(
        tool_call_id="call-4",
        model_call_id="llm-1",
        tool_name="sed",
    )
    echo_card = build_tool_call_card(
        tool_call_id="call-5",
        model_call_id="llm-1",
        tool_name="echo_into",
    )
    default_card = build_tool_call_card(
        tool_call_id="call-6",
        model_call_id="llm-1",
        tool_name="lookup",
    )

    assert isinstance(execute_card, ExecuteCodeToolCallCard)
    assert isinstance(read_card, ReadFileToolCallCard)
    assert isinstance(read_image_card, ReadImageToolCallCard)
    assert isinstance(grep_card, GrepToolCallCard)
    assert isinstance(sed_card, SedToolCallCard)
    assert isinstance(echo_card, EchoIntoToolCallCard)
    assert isinstance(default_card, DefaultToolCallCard)


def test_execute_code_card_builds_dedicated_sections() -> None:
    """execute_code card should separate code, live output, and summary."""

    card = ExecuteCodeToolCallCard(
        tool_call_id="call-1",
        model_call_id="llm-1",
        tool_name="execute_code",
        arguments={"code": "print(1)", "timeout_seconds": 30},
    )
    card.append_output("1\n")
    card.set_result_markdown("Return value:\n`None`")
    card.refresh_arguments_markdown()

    assert "## Code" in card.arguments_markdown
    assert "```python" in card.arguments_markdown
    assert "## Runtime Controls" in card.arguments_markdown
    assert "timeout_seconds" in card.arguments_markdown
    assert "## Live Output" in card.build_output_markdown()
    assert "## Execution Summary" in card.build_result_markdown()


def test_read_file_card_groups_path_and_line_range() -> None:
    """read_file card should emphasize file path and requested line span."""

    card = ReadFileToolCallCard(
        tool_call_id="call-2",
        model_call_id="llm-1",
        tool_name="read_file",
        arguments={"path": "src/main.py", "start_line": 10, "end_line": 20},
    )
    card.refresh_arguments_markdown()

    assert "## File" in card.arguments_markdown
    assert "`src/main.py`" in card.arguments_markdown
    assert "## Line Range" in card.arguments_markdown
    assert "10-20" in card.arguments_markdown


def test_sed_card_renders_diff_like_edit_preview() -> None:
    """sed card should render a git-diff-like edit intent preview."""

    card = SedToolCallCard(
        tool_call_id="call-3",
        model_call_id="llm-1",
        tool_name="sed",
        arguments={
            "path": "src/main.py",
            "start_line": 3,
            "end_line": 7,
            "pattern_to_be_replaced": "foo.+bar",
            "new_string": "baz",
        },
    )
    card.refresh_arguments_markdown()

    assert "## File" in card.arguments_markdown
    assert "## Line Range" in card.arguments_markdown
    assert "## Edit Preview" in card.arguments_markdown
    assert "```diff" in card.arguments_markdown
    assert "- /foo.+bar/" in card.arguments_markdown
    assert "+ baz" in card.arguments_markdown
    assert "regex replacement preview" in card.arguments_markdown.lower()
    assert "foo.+bar" in card.arguments_markdown
