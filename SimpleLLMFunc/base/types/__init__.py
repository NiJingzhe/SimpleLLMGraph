"""Shared dataclass/type contracts for SimpleLLMFunc.base."""

from SimpleLLMFunc.base.types.compile import CompiledTurnContext, ReducedTurnContext
from SimpleLLMFunc.base.types.context import CompiledContext, ContextState
from SimpleLLMFunc.base.types.llm import SingleLLMCallResult, SingleLLMPhaseResultYield
from SimpleLLMFunc.base.types.mutation import (
    AssistantMessageMutation,
    AssistantTruncatedMutation,
    ContextMutation,
    ContextReplaceMutation,
    ContextSummaryMutation,
    ExperienceForgetMutation,
    ExperienceRememberMutation,
    MultimodalToolResultMutation,
    ToolCancelledMutation,
    ToolResultMutation,
    UserMessageMutation,
)
from SimpleLLMFunc.base.types.react import ReactLoopState
from SimpleLLMFunc.base.types.scheduler import ToolSchedulerResult
from SimpleLLMFunc.base.types.source import CompileSource, DataFromAgentConfig, DataFromSelfRef

__all__ = [
    "AssistantMessageMutation",
    "AssistantTruncatedMutation",
    "CompileSource",
    "CompiledContext",
    "CompiledTurnContext",
    "ContextMutation",
    "ContextReplaceMutation",
    "ContextState",
    "ContextSummaryMutation",
    "DataFromAgentConfig",
    "DataFromSelfRef",
    "ExperienceForgetMutation",
    "ExperienceRememberMutation",
    "MultimodalToolResultMutation",
    "ReactLoopState",
    "ReducedTurnContext",
    "SingleLLMCallResult",
    "SingleLLMPhaseResultYield",
    "ToolCancelledMutation",
    "ToolResultMutation",
    "ToolSchedulerResult",
    "UserMessageMutation",
]
