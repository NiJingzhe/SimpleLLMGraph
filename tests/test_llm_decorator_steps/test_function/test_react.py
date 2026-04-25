"""Tests for llm_decorator.steps.function.react module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.function.react import (
    check_response_content_empty,
    execute_llm_call,
    execute_react_loop,
    get_final_response,
    prepare_tools_for_execution,
    retry_llm_call,
)


class TestPrepareToolsForExecution:
    """Tests for prepare_tools_for_execution function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.process_tools")
    def test_prepare_tools(self, mock_process_tools: MagicMock) -> None:
        """Test preparing tools for execution."""
        mock_process_tools.return_value = ([{"name": "tool1"}], {"tool1": AsyncMock()})
        result = prepare_tools_for_execution([], "test_func")
        mock_process_tools.assert_called_once_with([], "test_func")


class TestExecuteLLMCall:
    """Tests for execute_llm_call function."""

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.ReAct_loop")
    async def test_execute_call(
        self, mock_react_loop: AsyncMock, mock_llm_interface: Any, sample_messages: list
    ) -> None:
        """Test executing LLM call."""
        async def mock_generator():
            yield "response1", sample_messages.copy()
            yield "response2", sample_messages.copy()
        
        mock_react_loop.return_value = mock_generator()
        
        result = execute_llm_call(
            mock_llm_interface,
            sample_messages,
            None,
            {},
            5,
            stream=False,
        )
        
        responses = []
        async for r in result:
            responses.append(r)
        
        assert len(responses) >= 1


class TestGetFinalResponse:
    """Tests for get_final_response function."""

    @pytest.mark.asyncio
    async def test_get_final_response(self) -> None:
        """Test getting final response from stream."""
        async def mock_generator():
            yield "response1"
            yield "response2"
            yield "final_response"
        
        result = await get_final_response(mock_generator())
        assert result == "final_response"


class TestCheckResponseContentEmpty:
    """Tests for check_response_content_empty function."""

    def test_check_empty_content(self, mock_chat_completion: Any) -> None:
        """Test checking empty content."""
        # Modify mock to have empty content
        mock_chat_completion.choices[0].message.content = ""
        result = check_response_content_empty(mock_chat_completion, "test_func")
        assert result is True

    def test_check_non_empty_content(self, mock_chat_completion: Any) -> None:
        """Test checking non-empty content."""
        result = check_response_content_empty(mock_chat_completion, "test_func")
        assert result is False


class TestRetryLLMCall:
    """Tests for retry_llm_call function."""

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.execute_llm_call")
    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.get_final_response")
    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.extract_content_from_response")
    async def test_retry_success(
        self,
        mock_extract: MagicMock,
        mock_get_final: AsyncMock,
        mock_execute: AsyncMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        """Test retrying LLM call successfully."""
        # extract_content_from_response 被调用3次：
        # 1. 第一次尝试（返回空）
        # 2. 第二次尝试（返回成功，触发break）
        # 3. 最终检查（返回成功）
        mock_extract.side_effect = ["", "success", "success"]
        mock_get_final.return_value = MagicMock()
        mock_execute.return_value = AsyncMock()
        
        result = await retry_llm_call(
            mock_llm_interface,
            sample_messages,
            None,
            {},
            5,
            2,
            "test_func",
        )
        
        assert mock_execute.call_count <= 3  # Initial + retries


class TestExecuteReactLoop:
    """Tests for execute_react_loop function."""

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.prepare_tools_for_execution")
    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.execute_llm_call")
    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.get_final_response")
    @patch("SimpleLLMFunc.llm_decorator.steps.function.react.check_response_content_empty")
    async def test_execute_react_loop(
        self,
        mock_check_empty: MagicMock,
        mock_get_final: AsyncMock,
        mock_execute: AsyncMock,
        mock_prepare: MagicMock,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        """Test executing ReAct loop."""
        mock_prepare.return_value = (None, {})
        mock_check_empty.return_value = False
        mock_get_final.return_value = MagicMock()
        mock_execute.return_value = AsyncMock()
        
        result = await execute_react_loop(
            mock_llm_interface,
            sample_messages,
            None,
            5,
            {},
            "test_func",
        )
        
        assert result is not None
