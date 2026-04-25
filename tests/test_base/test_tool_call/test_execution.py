"""Tests for base.tool_call.execution module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from SimpleLLMFunc.builtin.file_tools import FileToolset
from SimpleLLMFunc.base.tool_call.execution import (
    _execute_single_tool_call,
    process_tool_calls,
)
from SimpleLLMFunc.hooks.abort import AbortSignal
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


class TestExecuteSingleToolCall:
    """Tests for _execute_single_tool_call function."""

    @pytest.mark.asyncio
    async def test_execute_string_result(self) -> None:
        """Test executing tool call with string result."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
        }
        tool_map = {"test_tool": AsyncMock(return_value="result")}

        tool_call_dict, messages, is_multimodal = await _execute_single_tool_call(
            tool_call, tool_map
        )

        assert tool_call_dict == tool_call
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert is_multimodal is False

    @pytest.mark.asyncio
    async def test_execute_dict_result(self) -> None:
        """Test executing tool call with dict result."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
        }
        tool_map = {"test_tool": AsyncMock(return_value={"key": "value"})}

        tool_call_dict, messages, is_multimodal = await _execute_single_tool_call(
            tool_call, tool_map
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert is_multimodal is False

    @pytest.mark.asyncio
    async def test_execute_img_url_result(self, img_url: ImgUrl) -> None:
        """Test executing tool call with ImgUrl result."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(return_value=img_url)}

        tool_call_dict, messages, is_multimodal = await _execute_single_tool_call(
            tool_call, tool_map
        )

        assert is_multimodal is True
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_execute_img_path_result(self, img_path: ImgPath) -> None:
        """Test executing tool call with ImgPath result."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(return_value=img_path)}

        tool_call_dict, messages, is_multimodal = await _execute_single_tool_call(
            tool_call, tool_map
        )

        assert is_multimodal is True
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_execute_builtin_read_image_result(self, tmp_path) -> None:
        """read_image builtin should surface as multimodal user content."""
        image_path = tmp_path / "chart.png"
        image_path.write_bytes(b"")

        tool = next(
            tool for tool in FileToolset(tmp_path).toolset if tool.name == "read_image"
        )
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "read_image", "arguments": '{"path": "chart.png"}'},
        }
        tool_map = {"read_image": tool.run}

        _, messages, is_multimodal = await _execute_single_tool_call(tool_call, tool_map)

        assert is_multimodal is True
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["detail"] == "high"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_execute_tuple_result(self, img_url: ImgUrl) -> None:
        """Test executing tool call with tuple result."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(return_value=("text", img_url))}

        tool_call_dict, messages, is_multimodal = await _execute_single_tool_call(
            tool_call, tool_map
        )

        assert is_multimodal is True
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self) -> None:
        """Test executing tool call when tool not found."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "unknown_tool", "arguments": "{}"},
        }
        tool_map = {}

        tool_call_dict, messages, is_multimodal = await _execute_single_tool_call(
            tool_call, tool_map
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "error" in json.loads(messages[0]["content"])
        assert is_multimodal is False

    @pytest.mark.asyncio
    async def test_execute_tool_error(self) -> None:
        """Test executing tool call when tool raises error."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(side_effect=ValueError("Tool error"))}

        tool_call_dict, messages, is_multimodal = await _execute_single_tool_call(
            tool_call, tool_map
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "error" in json.loads(messages[0]["content"])
        assert is_multimodal is False

    @pytest.mark.asyncio
    async def test_execute_repair_malformed_arguments(self) -> None:
        """Test executing tool call with repairable malformed arguments."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{{"arg": "value"}'},
        }
        tool_func = AsyncMock(return_value="result")
        tool_map = {"test_tool": tool_func}

        _, messages, is_multimodal = await _execute_single_tool_call(
            tool_call, tool_map
        )

        tool_func.assert_awaited_once_with(arg="value")
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert is_multimodal is False


class TestProcessToolCalls:
    """Tests for process_tool_calls function."""

    @pytest.mark.asyncio
    async def test_process_single_tool_call(self) -> None:
        """Test processing single tool call."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        ]
        messages = [{"role": "user", "content": "test"}]
        tool_map = {"test_tool": AsyncMock(return_value="result")}

        result = await process_tool_calls(tool_calls, messages, tool_map)

        assert len(result) > len(messages)
        assert any(msg["role"] == "tool" for msg in result)

    @pytest.mark.asyncio
    async def test_process_multiple_tool_calls(self) -> None:
        """Test processing multiple tool calls."""
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
        messages = [{"role": "user", "content": "test"}]
        tool_map = {
            "tool1": AsyncMock(return_value="result1"),
            "tool2": AsyncMock(return_value="result2"),
        }

        result = await process_tool_calls(tool_calls, messages, tool_map)

        assert len(result) > len(messages)
        # Should have tool messages for both calls
        tool_messages = [msg for msg in result if msg["role"] == "tool"]
        assert len(tool_messages) == 2

    @pytest.mark.asyncio
    async def test_process_empty_tool_calls(self) -> None:
        """Test processing empty tool calls."""
        messages = [{"role": "user", "content": "test"}]
        result = await process_tool_calls([], messages, {})
        assert result == messages

    @pytest.mark.asyncio
    async def test_process_multimodal_tool_call(self, img_url: ImgUrl) -> None:
        """Test processing multimodal tool call."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }
        ]
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
        ]
        tool_map = {"test_tool": AsyncMock(return_value=img_url)}

        result = await process_tool_calls(tool_calls, messages, tool_map)

        # Should have assistant message and user message for multimodal result
        assert len(result) > len(messages)
        user_messages = [msg for msg in result if msg["role"] == "user"]
        assert len(user_messages) >= 1

    @pytest.mark.asyncio
    async def test_process_tool_calls_aborts_running_tasks(self) -> None:
        """Abort signal should cancel running tool tasks."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "slow_tool", "arguments": "{}"},
            }
        ]
        messages = [{"role": "user", "content": "test"}]
        cancel_seen = asyncio.Event()
        ready = asyncio.Event()

        async def slow_tool() -> str:
            ready.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                raise
            return "done"

        tool_map = {"slow_tool": slow_tool}
        abort_signal = AbortSignal()

        task = asyncio.create_task(
            process_tool_calls(
                tool_calls,
                messages,
                tool_map,
                abort_signal=abort_signal,
            )
        )

        await ready.wait()
        abort_signal.abort("user_interrupt")
        result = await task

        assert result == messages
        await asyncio.wait_for(cancel_seen.wait(), timeout=1.0)
