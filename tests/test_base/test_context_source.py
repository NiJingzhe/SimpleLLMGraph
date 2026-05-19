from __future__ import annotations

from SimpleLLMFunc.base.types import CompileSource, DataFromAgentConfig, DataFromSelfRef


def test_context_source_types_hold_agent_and_selfref_source_data() -> None:
    agent_data = DataFromAgentConfig(
        base_system_prompt="You are an agent.",
        template_params={"name": "demo"},
        tool_prompt_specs=[{"tool_name": "execute_code", "best_practices": "Use only when needed."}],
        include_must_principles=True,
    )
    selfref_data = DataFromSelfRef(
        base_system_prompt="You are an agent.",
        experiences=[{"id": "exp_1", "text": "Prefer concise output"}],
        summary={
            "goal": "Goal A",
            "instruction": "Instruction A",
            "discoveries": ["Discovery A"],
            "completed": ["Completed A"],
            "current_status": "Status A",
            "likely_next_work": ["Next A"],
            "relevant_files_directories": ["src/a.py"],
        },
        summary_message={"role": "assistant", "content": "summary"},
        working_messages=[{"role": "user", "content": "hello"}],
    )

    source = CompileSource(
        data_from_agent_config=agent_data,
        data_from_selfref=selfref_data,
        input_messages=[{"role": "user", "content": "hello"}],
    )

    assert source.data_from_agent_config.base_system_prompt == "You are an agent."
    assert source.data_from_agent_config.template_params == {"name": "demo"}
    assert source.data_from_agent_config.tool_prompt_specs[0]["tool_name"] == "execute_code"
    assert source.data_from_agent_config.include_must_principles is True
    assert source.data_from_selfref is not None
    assert source.data_from_selfref.experiences == [{"id": "exp_1", "text": "Prefer concise output"}]
    assert source.data_from_selfref.working_messages == [{"role": "user", "content": "hello"}]
