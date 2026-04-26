from __future__ import annotations

from SimpleLLMFunc.base.context_compile import (
    ContextState,
    apply_mutations,
    compile_context,
)
from SimpleLLMFunc.base.context_source import CompileSource, DataFromAgentConfig, DataFromSelfRef
from SimpleLLMFunc.base.llm_call import SingleLLMCallResult
from SimpleLLMFunc.base.mutation import (
    AssistantMessageMutation,
    AssistantTruncatedMutation,
    ContextReplaceMutation,
    ContextSummaryMutation,
    ToolCancelledMutation,
    ToolResultMutation,
    UserMessageMutation,
)
from SimpleLLMFunc.base.react_loop import ReactLoopState
from SimpleLLMFunc.base.tool_scheduler import ToolSchedulerResult


def test_compile_context_applies_assistant_and_tool_mutations() -> None:
    state = ContextState(messages=[{"role": "user", "content": "hello"}])
    compiled = compile_context(
        state,
        [
            AssistantMessageMutation(
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ]
            ),
            ToolResultMutation(tool_call_id="call_1", content='{"ok": true}'),
            AssistantMessageMutation(content="done"),
        ],
    )

    assert compiled.messages[0] == {"role": "user", "content": "hello"}
    assert compiled.messages[1]["role"] == "assistant"
    assert compiled.messages[1]["tool_calls"][0]["id"] == "call_1"
    assert compiled.messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"ok": true}',
    }
    assert compiled.messages[3] == {"role": "assistant", "content": "done"}
    assert compiled.messages == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "done"},
    ]


def test_compile_context_keeps_assistant_content_when_tool_calls_resolve() -> None:
    state = ContextState(messages=[{"role": "user", "content": "hello"}])
    compiled = compile_context(
        state,
        [
            AssistantMessageMutation(
                content="Let me check that.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ],
            ),
            ToolResultMutation(tool_call_id="call_1", content='{"ok": true}'),
        ],
    )

    assert compiled.messages == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]


def test_compile_context_replaces_history_when_context_replace_arrives() -> None:
    state = ContextState(messages=[{"role": "user", "content": "old"}])
    compiled = compile_context(
        state,
        [
            ContextReplaceMutation(
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "assistant", "content": "summary"},
                ]
            ),
            AssistantMessageMutation(content="fresh"),
        ],
    )

    assert compiled.messages == [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "summary"},
        {"role": "assistant", "content": "fresh"},
    ]


def test_compile_context_summary_rebuilds_context() -> None:
    state = ContextState(
        messages=[
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "seed"},
        ]
    )
    compiled = compile_context(
        state,
        [
            ContextSummaryMutation(
                summary_message={
                    "role": "assistant",
                    "content": "<context_compaction_summary>Goal</context_compaction_summary>",
                }
            ),
            AssistantMessageMutation(content="done"),
        ],
    )

    assert compiled.messages == [
        {"role": "system", "content": "agent"},
        {
            "role": "assistant",
            "content": "<context_compaction_summary>Goal</context_compaction_summary>",
        },
        {"role": "assistant", "content": "done"},
    ]


def test_apply_mutations_builds_abort_mutations_into_messages() -> None:
    messages = apply_mutations(
        [{"role": "user", "content": "hello"}],
        [
            AssistantTruncatedMutation(
                partial_content="partial ",
                abort_reason="user_interrupt",
            ),
            ToolCancelledMutation(
                tool_call_id="call_1",
                tool_name="execute_code",
                abort_reason="user_interrupt",
            ),
        ],
    )

    assert messages[1]["role"] == "assistant"
    assert "partial" in messages[1]["content"]
    assert "user_interrupt" in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call_1"
    assert "execute_code" in messages[3]["content"]
    assert "user_interrupt" in messages[3]["content"]


def test_new_core_contract_types_hold_mutations_and_counts() -> None:
    llm_result = SingleLLMCallResult(
        content="done",
        mutations=[AssistantMessageMutation(content="done")],
    )
    scheduler_result = ToolSchedulerResult(
        mutations=[ToolResultMutation(tool_call_id="call_1", content="ok")],
        total_tool_calls=1,
    )
    loop_state = ReactLoopState(
        context_state=ContextState(messages=[{"role": "user", "content": "x"}]),
        pending_mutations=llm_result.mutations + scheduler_result.mutations,
        total_llm_calls=1,
        total_tool_calls=1,
    )

    assert llm_result.mutations[0].content == "done"
    assert scheduler_result.total_tool_calls == 1
    assert len(loop_state.pending_mutations) == 2


def test_compile_context_appends_user_message_mutation_for_multimodal_or_prompted_input() -> None:
    state = ContextState(messages=[{"role": "user", "content": "hello"}])
    compiled = compile_context(
        state,
        [
            UserMessageMutation(
                message={
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "image from tool"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        },
                    ],
                }
            )
        ],
    )

    assert compiled.messages[-1]["role"] == "user"
    assert compiled.messages[-1]["content"][0]["text"] == "image from tool"


def test_compile_source_can_join_agent_config_and_selfref_source() -> None:
    source = CompileSource(
        data_from_agent_config=DataFromAgentConfig(
            base_system_prompt="agent docstring",
            tool_prompt_specs=[{"tool_name": "execute_code"}],
            include_must_principles=True,
        ),
        data_from_selfref=DataFromSelfRef(
            base_system_prompt="agent docstring",
            working_messages=[{"role": "user", "content": "seed"}],
        ),
        input_messages=[{"role": "user", "content": "seed"}],
    )

    assert source.data_from_agent_config.base_system_prompt == "agent docstring"
    assert source.data_from_agent_config.include_must_principles is True
    assert source.data_from_selfref is not None
    assert source.data_from_selfref.working_messages == [{"role": "user", "content": "seed"}]
