"""Unified compile / convert-to-LLM pipeline.

This module is the formal Stage 2 compile boundary. Both ``llm_function`` and
``llm_chat`` should reach provider-facing messages through
``compile_invocation_turn`` rather than rendering prompts in decorator code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from SimpleLLMFunc.base.context_compile import apply_mutations, clone_messages
from SimpleLLMFunc.base.context_source import CompileSource, DataFromSelfRef
from SimpleLLMFunc.base.llm_input_render import render_llm_input_messages
from SimpleLLMFunc.base.mutation import ContextMutation
from SimpleLLMFunc.llm_decorator.invocation_spec import InvocationSpec, PromptContract
from SimpleLLMFunc.runtime.selfref.context_ops import render_system_prompt_with_experiences
from SimpleLLMFunc.type.message import NormalizedMessageList, NormalizedMessageParam


@dataclass(frozen=True)
class ReducedTurnContext:
    transcript: NormalizedMessageList
    selfref_snapshot: Optional[DataFromSelfRef] = None


@dataclass(frozen=True)
class CompiledTurnContext:
    transcript: NormalizedMessageList
    system_prompt: Optional[str]
    llm_messages: NormalizedMessageList
    selfref_snapshot: Optional[DataFromSelfRef] = None


@dataclass(frozen=True)
class LLMRequest:
    messages: NormalizedMessageList


def _resolve_system_prompt(
    transcript: NormalizedMessageList,
    prompt_contract: PromptContract,
    selfref_snapshot: Optional[DataFromSelfRef],
) -> Optional[str]:
    if selfref_snapshot is not None:
        return render_system_prompt_with_experiences(
            selfref_snapshot.base_system_prompt,
            selfref_snapshot.experiences,
        )

    if prompt_contract.system_prompt is not None:
        return prompt_contract.system_prompt

    for message in transcript:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content

    if prompt_contract.base_instruction:
        return prompt_contract.base_instruction
    return None


def reduce_turn_context(
    transcript: NormalizedMessageList,
    pending_mutations: List[ContextMutation],
    selfref_snapshot: Optional[DataFromSelfRef] = None,
) -> ReducedTurnContext:
    applied_transcript = apply_mutations(transcript, pending_mutations)
    next_selfref_snapshot = selfref_snapshot
    if selfref_snapshot is not None:
        from SimpleLLMFunc.runtime.selfref.context_ops import (
            parse_context_compaction_summary,
            parse_data_from_selfref,
            split_system_prompt_experiences,
        )

        parsed_source = parse_data_from_selfref(
            cast(List[Dict[str, Any]], applied_transcript)
        )
        _base_prompt, parsed_experiences = split_system_prompt_experiences(
            parsed_source.base_system_prompt
        )
        has_selfref_markers = bool(
            parsed_experiences
            or parsed_source.summary is not None
            or parsed_source.summary_message is not None
            or any(
                isinstance(message, dict)
                and parse_context_compaction_summary(message.get("content")) is not None
                for message in applied_transcript
            )
        )
        if has_selfref_markers:
            next_selfref_snapshot = parsed_source

    return ReducedTurnContext(
        transcript=clone_messages(applied_transcript),
        selfref_snapshot=next_selfref_snapshot,
    )


def convert_to_llm_request(
    reduced: ReducedTurnContext,
    prompt_contract: PromptContract,
) -> CompiledTurnContext:
    transcript = clone_messages(reduced.transcript)
    system_prompt = _resolve_system_prompt(
        transcript,
        prompt_contract,
        reduced.selfref_snapshot,
    )

    if transcript and isinstance(transcript[0], dict) and transcript[0].get("role") == "system":
        if system_prompt:
            transcript[0] = cast(
                NormalizedMessageParam,
                {**transcript[0], "content": system_prompt},
            )
        else:
            transcript = cast(NormalizedMessageList, transcript[1:])
    elif system_prompt:
        transcript = cast(
            NormalizedMessageList,
            [{"role": "system", "content": system_prompt}, *transcript],
        )

    llm_messages = render_llm_input_messages(
        transcript,
        tool_prompt_specs=prompt_contract.tool_prompt_specs,
        include_must_principles=prompt_contract.include_must_principles,
    )

    return CompiledTurnContext(
        transcript=clone_messages(transcript),
        system_prompt=system_prompt,
        llm_messages=llm_messages,
        selfref_snapshot=reduced.selfref_snapshot,
    )


def compile_invocation_turn(
    spec: InvocationSpec,
    transcript: NormalizedMessageList,
    pending_mutations: Optional[List[ContextMutation]] = None,
    selfref_snapshot: Optional[DataFromSelfRef] = None,
) -> CompiledTurnContext:
    reduced = reduce_turn_context(
        transcript,
        list(pending_mutations or []),
        selfref_snapshot if selfref_snapshot is not None else spec.data_from_selfref,
    )
    return convert_to_llm_request(reduced, spec.prompt_contract)


def build_compiled_messages_from_source(source: CompileSource) -> NormalizedMessageList:
    """Compatibility adapter from Stage 1 ``CompileSource`` to Stage 2 pipeline."""

    from SimpleLLMFunc.llm_decorator.invocation_spec import PromptContract, TranscriptSeed

    spec = InvocationSpec(
        mode="chat",
        func_name="compile_source",
        trace_id="compile_source",
        docstring=source.data_from_agent_config.base_system_prompt,
        bound_args={},
        type_hints={},
        return_type=None,
        template_params=source.data_from_agent_config.template_params,
        llm_kwargs={},
        stream=False,
        return_mode="text",
        prompt_contract=PromptContract(
            base_instruction=source.data_from_agent_config.base_system_prompt,
            tool_prompt_specs=list(source.data_from_agent_config.tool_prompt_specs),
            include_must_principles=source.data_from_agent_config.include_must_principles,
        ),
        transcript_seed=TranscriptSeed(
            initial_messages=cast(NormalizedMessageList, source.input_messages),
        ),
        data_from_selfref=source.data_from_selfref,
    )
    compiled = compile_invocation_turn(
        spec,
        cast(NormalizedMessageList, source.input_messages),
        [],
        source.data_from_selfref,
    )
    return compiled.llm_messages


__all__ = [
    "CompiledTurnContext",
    "LLMRequest",
    "ReducedTurnContext",
    "build_compiled_messages_from_source",
    "compile_invocation_turn",
    "convert_to_llm_request",
    "reduce_turn_context",
]
