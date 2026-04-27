"""Compatibility re-exports for structured context mutation contracts."""

from SimpleLLMFunc.base.types.mutation import (
    AssistantMessageMutation,
    AssistantTruncatedMutation,
    ContextMutation,
    ContextReplaceMutation,
    ContextSummaryMutation,
    ExperienceForgetMutation,
    ExperienceRememberMutation,
    ToolCancelledMutation,
    ToolResultMutation,
    UserMessageMutation,
)

__all__ = [
    "AssistantMessageMutation",
    "AssistantTruncatedMutation",
    "ContextMutation",
    "ContextReplaceMutation",
    "ContextSummaryMutation",
    "ExperienceRememberMutation",
    "ExperienceForgetMutation",
    "ToolCancelledMutation",
    "ToolResultMutation",
    "UserMessageMutation",
]
