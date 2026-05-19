from __future__ import annotations

from dataclasses import is_dataclass


def test_base_types_package_reexports_core_dataclasses() -> None:
    from SimpleLLMFunc.base.types import (
        AssistantMessageMutation,
        AssistantTruncatedMutation,
        CompileSource,
        CompiledContext,
        CompiledTurnContext,
        ContextMutation,
        ContextReplaceMutation,
        ContextState,
        ContextSummaryMutation,
        DataFromAgentConfig,
        DataFromSelfRef,
        ExperienceForgetMutation,
        ExperienceRememberMutation,
        ReactLoopState,
        ReducedTurnContext,
        SingleLLMCallResult,
        SingleLLMPhaseResultYield,
        ToolCancelledMutation,
        ToolResultMutation,
        ToolSchedulerResult,
        UserMessageMutation,
    )

    exported = [
        AssistantMessageMutation,
        AssistantTruncatedMutation,
        CompileSource,
        CompiledContext,
        CompiledTurnContext,
        ContextReplaceMutation,
        ContextState,
        ContextSummaryMutation,
        DataFromAgentConfig,
        DataFromSelfRef,
        ExperienceForgetMutation,
        ExperienceRememberMutation,
        ReactLoopState,
        ReducedTurnContext,
        SingleLLMCallResult,
        SingleLLMPhaseResultYield,
        ToolCancelledMutation,
        ToolResultMutation,
        ToolSchedulerResult,
        UserMessageMutation,
    ]

    assert all(is_dataclass(item) for item in exported)
    assert ContextMutation is not None
