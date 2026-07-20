import SimpleLLMFunc.loop.tool_contract as tool_contract
from SimpleLLMFunc.loop import (
    EventSnapshot,
    FunctionTool,
    ModelCallEffect,
    ModelCallResolver,
    ToolCallEffect,
    ToolCallResolver,
    ToolResultCompiler,
    tool,
)
from SimpleLLMFunc.loop.tool_contract import ToolReturnContract


def test_loop_types_are_owned_by_semantic_modules() -> None:
    assert ModelCallEffect.__module__ == "SimpleLLMFunc.loop.model_call"
    assert ModelCallResolver.__module__ == "SimpleLLMFunc.loop.model_call"
    assert EventSnapshot.__module__ == "SimpleLLMFunc.loop.tool_call"
    assert ToolCallEffect.__module__ == "SimpleLLMFunc.loop.tool_call"
    assert FunctionTool.__module__ == "SimpleLLMFunc.loop.tool"
    assert tool.__module__ == "SimpleLLMFunc.loop.tool"
    assert ToolCallResolver.__module__ == "SimpleLLMFunc.loop.tool_runtime"
    assert ToolReturnContract.__module__ == "SimpleLLMFunc.loop.tool_contract"
    assert ToolResultCompiler is tool_contract.ToolResultCompiler
