from __future__ import annotations

from SimpleLLMFunc.base.context_source import DataFromSelfRef
from SimpleLLMFunc.runtime.selfref.context_ops import (
    build_context_messages_from_selfref_data,
    build_context_messages_from_state_data,
    parse_data_from_selfref,
    parse_context_messages,
    parse_context_compaction_summary,
    render_context_compaction_summary,
    render_system_prompt_with_experiences,
)


def test_render_and_parse_context_compaction_summary_round_trip() -> None:
    summary = {
        "goal": "Goal A",
        "instruction": "Instruction A",
        "discoveries": ["Discovery A"],
        "completed": ["Completed A"],
        "current_status": "Status A",
        "likely_next_work": ["Next A"],
        "relevant_files_directories": ["src/a.py"],
    }

    rendered = render_context_compaction_summary(summary)

    assert parse_context_compaction_summary(rendered) == summary


def test_parse_context_messages_extracts_system_experiences_summary_and_working_messages() -> (
    None
):
    messages = [
        {
            "role": "system",
            "content": "Base rules\n\n<experience>\n- [exp_1] Preference A\n</experience>",
        },
        {
            "role": "assistant",
            "content": render_context_compaction_summary(
                {
                    "goal": "Goal A",
                    "instruction": "Instruction A",
                    "discoveries": ["Discovery A"],
                    "completed": ["Completed A"],
                    "current_status": "Status A",
                    "likely_next_work": ["Next A"],
                    "relevant_files_directories": ["src/a.py"],
                }
            ),
        },
        {"role": "assistant", "content": "fresh output"},
        {"role": "user", "content": "follow-up"},
    ]

    parsed = parse_context_messages(messages)

    assert parsed["base_system_prompt"] == "Base rules"
    assert parsed["experiences"] == [{"id": "exp_1", "text": "Preference A"}]
    assert parsed["summary"] == {
        "goal": "Goal A",
        "instruction": "Instruction A",
        "discoveries": ["Discovery A"],
        "completed": ["Completed A"],
        "current_status": "Status A",
        "likely_next_work": ["Next A"],
        "relevant_files_directories": ["src/a.py"],
    }
    assert parsed["working_messages"] == [
        {"role": "assistant", "content": "fresh output"},
        {"role": "user", "content": "follow-up"},
    ]


def test_build_context_messages_rehydrates_system_summary_and_working_messages() -> (
    None
):
    state_data = {
        "base_system_prompt": "Base rules",
        "experiences": [{"id": "exp_1", "text": "Preference A"}],
        "summary": {
            "goal": "Goal A",
            "instruction": "Instruction A",
            "discoveries": ["Discovery A"],
            "completed": ["Completed A"],
            "current_status": "Status A",
            "likely_next_work": ["Next A"],
            "relevant_files_directories": ["src/a.py"],
        },
        "summary_message": None,
        "working_messages": [
            {"role": "assistant", "content": "fresh output"},
            {"role": "user", "content": "follow-up"},
        ],
    }

    compiled = build_context_messages_from_state_data(state_data)

    assert compiled[0] == {
        "role": "system",
        "content": render_system_prompt_with_experiences(
            "Base rules",
            [{"id": "exp_1", "text": "Preference A"}],
        ),
    }
    assert (
        parse_context_compaction_summary(compiled[1]["content"])
        == state_data["summary"]
    )
    assert compiled[2:] == state_data["working_messages"]


def test_parse_and_build_context_messages_round_trip() -> None:
    original_messages = [
        {
            "role": "system",
            "content": "Base rules\n\n<experience>\n- [exp_1] Preference A\n</experience>",
        },
        {
            "role": "assistant",
            "content": render_context_compaction_summary(
                {
                    "goal": "Goal A",
                    "instruction": "Instruction A",
                    "discoveries": ["Discovery A"],
                    "completed": ["Completed A"],
                    "current_status": "Status A",
                    "likely_next_work": ["Next A"],
                    "relevant_files_directories": ["src/a.py"],
                }
            ),
        },
        {"role": "assistant", "content": "fresh output"},
        {"role": "user", "content": "follow-up"},
    ]

    parsed = parse_context_messages(original_messages)
    rebuilt = build_context_messages_from_state_data(parsed)

    assert rebuilt == original_messages


def test_parsed_context_messages_map_cleanly_to_data_from_selfref() -> None:
    messages = [
        {
            "role": "system",
            "content": "Base rules\n\n<experience>\n- [exp_1] Preference A\n</experience>",
        },
        {
            "role": "assistant",
            "content": render_context_compaction_summary(
                {
                    "goal": "Goal A",
                    "instruction": "Instruction A",
                    "discoveries": ["Discovery A"],
                    "completed": ["Completed A"],
                    "current_status": "Status A",
                    "likely_next_work": ["Next A"],
                    "relevant_files_directories": ["src/a.py"],
                }
            ),
        },
        {"role": "user", "content": "follow-up"},
    ]

    parsed = parse_context_messages(messages)
    data = DataFromSelfRef(
        base_system_prompt=parsed["base_system_prompt"],
        experiences=parsed["experiences"],
        summary=parsed["summary"],
        summary_message=parsed["summary_message"],
        working_messages=parsed["working_messages"],
    )

    assert data.base_system_prompt == "Base rules"
    assert data.experiences == [{"id": "exp_1", "text": "Preference A"}]
    assert data.summary is not None
    assert data.working_messages == [{"role": "user", "content": "follow-up"}]


def test_parse_and_build_data_from_selfref_round_trip() -> None:
    original_messages = [
        {
            "role": "system",
            "content": "Base rules\n\n<experience>\n- [exp_1] Preference A\n</experience>",
        },
        {
            "role": "assistant",
            "content": render_context_compaction_summary(
                {
                    "goal": "Goal A",
                    "instruction": "Instruction A",
                    "discoveries": ["Discovery A"],
                    "completed": ["Completed A"],
                    "current_status": "Status A",
                    "likely_next_work": ["Next A"],
                    "relevant_files_directories": ["src/a.py"],
                }
            ),
        },
        {"role": "assistant", "content": "fresh output"},
    ]

    data = parse_data_from_selfref(original_messages)
    rebuilt = build_context_messages_from_selfref_data(data)

    assert isinstance(data, DataFromSelfRef)
    assert rebuilt == original_messages
