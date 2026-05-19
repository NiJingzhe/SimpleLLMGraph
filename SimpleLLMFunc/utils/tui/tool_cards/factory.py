"""Factory for selecting specialized ToolCallCard implementations."""

from __future__ import annotations

from typing import Any, Optional

from SimpleLLMFunc.utils.tui.tool_cards.base import ToolArgumentFormatter, ToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.default import DefaultToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.execute_code import ExecuteCodeToolCallCard
from SimpleLLMFunc.utils.tui.tool_cards.file_tools import (
    EchoIntoToolCallCard,
    GrepToolCallCard,
    ReadFileToolCallCard,
    ReadImageToolCallCard,
    SedToolCallCard,
)


_CARD_TYPES = {
    "execute_code": ExecuteCodeToolCallCard,
    "read_file": ReadFileToolCallCard,
    "read_image": ReadImageToolCallCard,
    "grep": GrepToolCallCard,
    "sed": SedToolCallCard,
    "echo_into": EchoIntoToolCallCard,
}


def build_tool_call_card(
    *,
    tool_call_id: str,
    model_call_id: str,
    tool_name: str,
    arguments: Optional[dict[str, Any]] = None,
    argument_formatter: Optional[ToolArgumentFormatter] = None,
) -> ToolCallCard:
    card_type = _CARD_TYPES.get(tool_name, DefaultToolCallCard)
    return card_type(
        tool_call_id=tool_call_id,
        model_call_id=model_call_id,
        tool_name=tool_name,
        arguments=arguments,
        argument_formatter=argument_formatter,
    )


__all__ = ["build_tool_call_card"]
