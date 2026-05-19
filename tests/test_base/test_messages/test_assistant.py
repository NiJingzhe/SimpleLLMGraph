"""Tests for base.messages.assistant module."""

from __future__ import annotations

from SimpleLLMFunc.base.messages.assistant import (
    build_assistant_response_message,
    build_assistant_tool_message,
)


class TestBuildAssistantResponseMessage:
    """Tests for build_assistant_response_message function."""

    def test_build_with_content(self) -> None:
        """Test building assistant message with content."""
        result = build_assistant_response_message("Hello, world!")
        assert result == {
            "role": "assistant",
            "content": "Hello, world!",
        }

    def test_build_with_empty_content(self) -> None:
        """Test building assistant message with empty content."""
        result = build_assistant_response_message("")
        assert result == {
            "role": "assistant",
            "content": "",
        }


class TestBuildAssistantToolMessage:
    """Tests for build_assistant_tool_message function."""

    def test_build_with_tool_calls(self) -> None:
        """Test building assistant message with tool calls."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        result = build_assistant_tool_message(tool_calls)
        assert result == {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }

    def test_build_with_tool_calls_and_content(self) -> None:
        """Assistant tool-call messages may also carry textual content."""

        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        result = build_assistant_tool_message(tool_calls, content="Let me check that.")
        assert result == {
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": tool_calls,
        }

    def test_build_with_empty_tool_calls(self) -> None:
        """Test building assistant message with empty tool calls."""
        result = build_assistant_tool_message([])
        assert result == {}

    def test_build_with_multiple_tool_calls(self) -> None:
        """Test building assistant message with multiple tool calls."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "tool1", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "tool2", "arguments": "{}"},
            },
        ]
        result = build_assistant_tool_message(tool_calls)
        assert result["tool_calls"] == tool_calls
        assert len(result["tool_calls"]) == 2

    def test_build_drops_reasoning_details(self) -> None:
        """Reasoning details should be consumed transiently, not persisted."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]

        result = build_assistant_tool_message(
            tool_calls,
            reasoning_details=[
                {
                    "id": "r1",
                    "type": "reasoning.text",
                    "data": "transient",
                }
            ],
        )

        assert "reasoning_details" not in result
