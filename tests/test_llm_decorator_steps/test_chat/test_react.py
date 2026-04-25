"""Tests for llm_decorator.steps.chat.react module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from SimpleLLMFunc.llm_decorator.steps.chat.react import (
    execute_llm_call,
    execute_react_loop_streaming,
    prepare_tools_for_execution,
)


class TestPrepareToolsForExecution:
    """Tests for prepare_tools_for_execution function."""

    @patch("SimpleLLMFunc.llm_decorator.steps.chat.react.process_tools")
    def test_prepare_tools(self, mock_process_tools: Any) -> None:
        """Test preparing tools for execution."""
        mock_process_tools.return_value = ([{"name": "tool1"}], {"tool1": AsyncMock()})
        result = prepare_tools_for_execution([], "test_func")
        mock_process_tools.assert_called_once_with([], "test_func")


class TestExecuteLLMCall:
    """Tests for execute_llm_call function."""

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.llm_decorator.steps.chat.react.ReAct_loop")
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
            stream=True,
        )
        
        responses = []
        async for r, _ in result:
            responses.append(r)
        
        assert len(responses) >= 1


class TestExecuteReactLoopStreaming:
    """Tests for execute_react_loop_streaming function."""

    @pytest.mark.asyncio
    @patch("SimpleLLMFunc.llm_decorator.steps.chat.react.prepare_tools_for_execution")
    @patch("SimpleLLMFunc.llm_decorator.steps.chat.react.execute_llm_call")
    async def test_execute_streaming(
        self,
        mock_execute: AsyncMock,
        mock_prepare: Any,
        mock_llm_interface: Any,
        sample_messages: list,
    ) -> None:
        """Test executing streaming ReAct loop."""
        mock_prepare.return_value = (None, {})
        
        async def mock_generator():
            yield "response1", sample_messages.copy()
            yield "response2", sample_messages.copy()
        
        mock_execute.return_value = mock_generator()
        
        responses = []
        async for response, _ in execute_react_loop_streaming(
            mock_llm_interface,
            sample_messages,
            None,
            5,
            True,
            {},
            "test_func",
        ):
            responses.append(response)
        
        assert len(responses) >= 1
        mock_prepare.assert_called_once()
        mock_execute.assert_called_once()
