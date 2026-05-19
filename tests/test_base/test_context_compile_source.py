from __future__ import annotations

from typing import cast

from SimpleLLMFunc.base.compile_pipeline import build_compiled_messages_from_source
from SimpleLLMFunc.base.types import CompileSource, DataFromAgentConfig, DataFromSelfRef
from SimpleLLMFunc.base.llm_input_render import render_llm_input_messages
from SimpleLLMFunc.runtime.selfref.context_ops import build_context_messages_from_selfref_data
from SimpleLLMFunc.type.message import NormalizedMessageList


def test_compile_source_can_build_base_messages_from_selfref_source() -> None:
    source = CompileSource(
        data_from_agent_config=DataFromAgentConfig(
            base_system_prompt="Agent docstring",
            template_params={"target": "demo"},
        ),
        data_from_selfref=DataFromSelfRef(
            base_system_prompt="Agent docstring",
            experiences=[{"id": "exp_1", "text": "Preference A"}],
            summary={
                "goal": "Goal A",
                "instruction": "Instruction A",
                "discoveries": ["Discovery A"],
                "completed": ["Completed A"],
                "current_status": "Status A",
                "likely_next_work": ["Next A"],
                "relevant_files_directories": ["src/a.py"],
            },
            working_messages=[{"role": "user", "content": "follow-up"}],
        ),
        input_messages=[{"role": "user", "content": "follow-up"}],
    )

    assert source.data_from_selfref is not None
    compiled_messages = build_context_messages_from_selfref_data(source.data_from_selfref)

    assert compiled_messages[0]["role"] == "system"
    assert "Preference A" in str(compiled_messages[0]["content"])
    assert compiled_messages[1]["role"] == "assistant"
    assert "Goal A" in str(compiled_messages[1]["content"])
    assert compiled_messages[2] == {"role": "user", "content": "follow-up"}


def test_render_llm_input_messages_injects_tool_and_must_blocks_on_compiled_messages() -> None:
    compiled_messages = cast(
        NormalizedMessageList,
        [
            {"role": "system", "content": "Agent docstring"},
            {"role": "user", "content": "hello"},
        ],
    )

    rendered = render_llm_input_messages(
        compiled_messages,
        tool_prompt_specs=[
                {
                "name": "execute_code",
                "description": "Execute Python code when concrete runtime inspection is required.",
                "best_practices": ["Use only when concrete execution is needed."],
                }
            ],
            include_must_principles=True,
    )

    assert rendered[0]["role"] == "system"
    assert "<tool_best_practices>" in str(rendered[0]["content"])
    assert "execute_code" in str(rendered[0]["content"])
    assert "Agent docstring" in str(rendered[0]["content"])
    assert "<must_principles>" in str(rendered[0]["content"])
    assert rendered[1] == {"role": "user", "content": "hello"}


def test_build_compiled_messages_from_source_uses_selfref_source_then_injects_prompt_blocks() -> None:
    source = CompileSource(
        data_from_agent_config=DataFromAgentConfig(
            base_system_prompt="Agent docstring",
            tool_prompt_specs=[
                {
                    "name": "execute_code",
                    "description": "Execute Python code when concrete runtime inspection is required.",
                    "best_practices": ["Use only when concrete execution is needed."],
                }
            ],
            include_must_principles=True,
        ),
        data_from_selfref=DataFromSelfRef(
            base_system_prompt="Agent docstring",
            experiences=[{"id": "exp_1", "text": "Preference A"}],
            working_messages=[{"role": "user", "content": "follow-up"}],
        ),
        input_messages=cast(
            NormalizedMessageList,
            [{"role": "user", "content": "follow-up"}],
        ),
    )

    compiled = build_compiled_messages_from_source(source)

    assert compiled[0]["role"] == "system"
    assert "Preference A" in str(compiled[0]["content"])
    assert "<tool_best_practices>" in str(compiled[0]["content"])
    assert "<must_principles>" in str(compiled[0]["content"])
    assert compiled[1] == {"role": "user", "content": "follow-up"}


def test_build_compiled_messages_from_source_uses_input_messages_for_non_system_content() -> None:
    source = CompileSource(
        data_from_agent_config=DataFromAgentConfig(
            base_system_prompt="Agent docstring",
        ),
        data_from_selfref=DataFromSelfRef(
            base_system_prompt="Runtime system",
            experiences=[{"id": "exp_1", "text": "Preference A"}],
            working_messages=[{"role": "user", "content": "stale selfref message"}],
        ),
        input_messages=cast(
            NormalizedMessageList,
            [
                {"role": "system", "content": "Agent docstring"},
                {"role": "user", "content": "seed"},
                {"role": "user", "content": "message: hello"},
            ],
        ),
    )

    compiled = build_compiled_messages_from_source(source)

    assert compiled[0]["role"] == "system"
    assert "Runtime system" in str(compiled[0]["content"])
    assert "Preference A" in str(compiled[0]["content"])
    assert compiled[1:] == [
        {"role": "user", "content": "seed"},
        {"role": "user", "content": "message: hello"},
    ]
