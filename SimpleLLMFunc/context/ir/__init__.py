"""Provider-neutral context entry IR.

A minimal, pydantic-typed intermediate representation for LLM conversation
entries, requests and streaming deltas. Modelled on the OpenAI Chat
Completions streaming spec, extended with first-class reasoning support
(an effort parameter plus reasoning content parts/deltas).

A higher-level, more semantic Event layer is expected to sit above this IR
in future revisions; the IR itself deliberately stays close to the wire
spec so that provider adapters can translate to and from it cheaply.
"""

from SimpleLLMFunc.context.ir._enums import (
    AudioFormat,
    FinishReason,
    ImageDetail,
    ReasoningEffort,
    Role,
    StreamObject,
    ToolCallType,
)
from SimpleLLMFunc.context.ir.messages import (
    AssistantMessage,
    Conversation,
    ConversationEntry,
    DeveloperMessage,
    SystemMessage,
    ToolCall,
    ToolCallFunction,
    ToolMessage,
    UserMessage,
)
from SimpleLLMFunc.context.ir.parts import (
    ContentPart,
    Image,
    InputAudioData,
    InputAudioPart,
    InputImagePart,
    InputImageURL,
    InputTextPart,
    OutputAudioData,
    OutputAudioPart,
    OutputTextPart,
    ReasoningPart,
)
from SimpleLLMFunc.context.ir.tool_result import ToolResult
from SimpleLLMFunc.context.ir.request import (
    Request,
    StreamOptions,
    Tool,
    ToolFunction,
)
from SimpleLLMFunc.context.ir.response import Completion, CompletionChoice
from SimpleLLMFunc.context.ir.streaming import (
    Choice,
    Chunk,
    CompletionTokensDetails,
    Delta,
    ToolCallDelta,
    ToolCallDeltaFunction,
    Usage,
)

__all__ = [
    # enums
    "AudioFormat",
    "FinishReason",
    "ImageDetail",
    "ReasoningEffort",
    "Role",
    "StreamObject",
    "ToolCallType",
    # parts
    "ContentPart",
    "Image",
    "InputAudioData",
    "InputAudioPart",
    "InputImagePart",
    "InputImageURL",
    "InputTextPart",
    "OutputAudioData",
    "OutputAudioPart",
    "OutputTextPart",
    "ReasoningPart",
    "ToolResult",
    # messages
    "AssistantMessage",
    "Conversation",
    "ConversationEntry",
    "DeveloperMessage",
    "SystemMessage",
    "ToolCall",
    "ToolCallFunction",
    "ToolMessage",
    "UserMessage",
    # request
    "Request",
    "StreamOptions",
    "Tool",
    "ToolFunction",
    # response (non-streaming)
    "Completion",
    "CompletionChoice",
    # streaming
    "Choice",
    "Chunk",
    "CompletionTokensDetails",
    "Delta",
    "ToolCallDelta",
    "ToolCallDeltaFunction",
    "Usage",
]
