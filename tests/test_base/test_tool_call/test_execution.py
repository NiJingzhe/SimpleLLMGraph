"""Tests for base.tool_call.execution module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from SimpleLLMFunc.builtin.file_tools import FileToolset
from SimpleLLMFunc.base.tool_call.execution import (
    ExecutedToolCallResult,
    execute_single_tool_call_result,
)
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl


class TestExecuteSingleToolCallResult:
    """Tests for execute_single_tool_call_result."""

    @pytest.mark.asyncio
    async def test_execute_string_result(self) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
        }
        tool_map = {"test_tool": AsyncMock(return_value="result")}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert isinstance(result, ExecutedToolCallResult)
        assert result.tool_call == tool_call
        assert result.tool_name == "test_tool"
        assert result.tool_call_id == "call_123"
        assert result.is_multimodal is False
        assert result.success is True
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "tool"

    @pytest.mark.asyncio
    async def test_execute_dict_result(self) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
        }
        tool_map = {"test_tool": AsyncMock(return_value={"key": "value"})}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert result.is_multimodal is False
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "tool"

    @pytest.mark.asyncio
    async def test_execute_img_url_result(self, img_url: ImgUrl) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(return_value=img_url)}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert result.is_multimodal is True
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_execute_img_path_result(self, img_path: ImgPath) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(return_value=img_path)}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert result.is_multimodal is True
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_execute_builtin_read_image_result(self, tmp_path) -> None:
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

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert result.is_multimodal is True
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"
        content = result.messages[0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["detail"] == "high"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_execute_tuple_result(self, img_url: ImgUrl) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(return_value=("text", img_url))}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert result.is_multimodal is True
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_execute_tuple_with_multiple_images_result(
        self,
        img_path: ImgPath,
        img_url: ImgUrl,
    ) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(return_value=("text", [img_url, img_path]))}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert result.is_multimodal is True
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"
        content = result.messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {
            "type": "text",
            "text": "This is images and description returned by tool 'test_tool': text",
        }
        image_parts = [part for part in content if part.get("type") == "image_url"]
        assert len(image_parts) == 2
        assert image_parts[0]["image_url"]["url"] == img_url.url
        assert image_parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "unknown_tool", "arguments": "{}"},
        }
        tool_map = {}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert result.success is False
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "tool"
        assert "error" in json.loads(result.messages[0]["content"])

    @pytest.mark.asyncio
    async def test_execute_tool_error(self) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        tool_map = {"test_tool": AsyncMock(side_effect=ValueError("Tool error"))}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        assert result.success is False
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "tool"
        assert "error" in json.loads(result.messages[0]["content"])

    @pytest.mark.asyncio
    async def test_execute_repair_malformed_arguments(self) -> None:
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{{"arg": "value"}'},
        }
        tool_func = AsyncMock(return_value="result")
        tool_map = {"test_tool": tool_func}

        result = await execute_single_tool_call_result(tool_call, tool_map)

        tool_func.assert_awaited_once_with(arg="value")
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "tool"
        assert result.is_multimodal is False
