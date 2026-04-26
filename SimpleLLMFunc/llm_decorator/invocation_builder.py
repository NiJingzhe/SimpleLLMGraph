"""Builders that turn decorator call data into InvocationSpec."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union, cast

from SimpleLLMFunc.llm_decorator.invocation_spec import (
    InvocationSpec,
    ParameterContract,
    PromptContract,
    ReturnContract,
    TranscriptSeed,
)
from SimpleLLMFunc.llm_decorator.prompt_contract import (
    HISTORY_PARAM_NAMES,
    build_chat_messages,
    build_function_transcript_seed,
    build_return_type_description,
    extract_parameter_type_hints,
    is_complex_return_type,
    process_docstring_template,
)
from SimpleLLMFunc.llm_decorator.signature import FunctionSignature
from SimpleLLMFunc.llm_decorator.utils import collect_tool_prompt_specs
from SimpleLLMFunc.runtime.selfref.session import SelfRefSession
from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.type.message import NormalizedMessageList


def _parameter_contracts(type_hints: Dict[str, Any]) -> List[ParameterContract]:
    param_type_hints = extract_parameter_type_hints(type_hints)
    return [
        ParameterContract(name=name, type_hint=hint)
        for name, hint in param_type_hints.items()
    ]


def build_function_invocation_spec(
    *,
    signature: FunctionSignature,
    template_params: Optional[Dict[str, Any]],
    llm_kwargs: Dict[str, Any],
    system_prompt_template: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    toolkit: Optional[List[Union[Tool, Any]]] = None,
) -> InvocationSpec:
    processed_docstring = process_docstring_template(
        signature.docstring,
        template_params,
    )
    initial_messages, system_prompt, return_type_description = build_function_transcript_seed(
        processed_docstring=processed_docstring,
        arguments=dict(signature.bound_args.arguments),
        type_hints=signature.type_hints,
        return_type=signature.return_type,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
    )

    prompt_contract = PromptContract(
        base_instruction=processed_docstring,
        parameter_contract=_parameter_contracts(signature.type_hints),
        return_contract=ReturnContract(
            return_type=signature.return_type,
            description=return_type_description,
            structured=is_complex_return_type(signature.return_type),
        ),
        tool_prompt_specs=collect_tool_prompt_specs(toolkit),
        include_must_principles=False,
        system_prompt=system_prompt,
    )

    return InvocationSpec(
        mode="function",
        func_name=signature.func_name,
        trace_id=signature.trace_id,
        docstring=signature.docstring,
        bound_args=dict(signature.bound_args.arguments),
        type_hints=dict(signature.type_hints),
        return_type=signature.return_type,
        template_params=template_params,
        llm_kwargs=dict(llm_kwargs),
        stream=False,
        return_mode="typed",
        prompt_contract=prompt_contract,
        transcript_seed=TranscriptSeed(initial_messages=initial_messages),
    )


def build_chat_invocation_spec(
    *,
    signature: FunctionSignature,
    template_params: Optional[Dict[str, Any]],
    llm_kwargs: Dict[str, Any],
    stream: bool,
    return_mode: Literal["text", "raw"],
    runtime_toolkit: Optional[List[Union[Tool, Any]]],
    selfref_session: Optional[SelfRefSession] = None,
    raw_history_reference: Optional[List[Dict[str, Any]]] = None,
) -> InvocationSpec:
    processed_docstring = process_docstring_template(
        signature.docstring,
        template_params,
    )
    messages = build_chat_messages(
        docstring=signature.docstring,
        func_name=signature.func_name,
        arguments=dict(signature.bound_args.arguments),
        type_hints=signature.type_hints,
        exclude_params=HISTORY_PARAM_NAMES,
        template_params=template_params,
    )
    data_from_selfref = (
        selfref_session.snapshot_source() if selfref_session is not None else None
    )
    history_authority: Literal["external", "selfref", "seed"] = "seed"
    if selfref_session is not None:
        history_authority = selfref_session.history_authority
    elif raw_history_reference is not None:
        history_authority = "external"

    prompt_contract = PromptContract(
        base_instruction=processed_docstring,
        parameter_contract=_parameter_contracts(signature.type_hints),
        return_contract=None,
        tool_prompt_specs=collect_tool_prompt_specs(runtime_toolkit),
        include_must_principles=True,
        system_prompt=processed_docstring,
    )

    return InvocationSpec(
        mode="chat",
        func_name=signature.func_name,
        trace_id=signature.trace_id,
        docstring=signature.docstring,
        bound_args=dict(signature.bound_args.arguments),
        type_hints=dict(signature.type_hints),
        return_type=signature.return_type,
        template_params=template_params,
        llm_kwargs=dict(llm_kwargs),
        stream=stream,
        return_mode=return_mode,
        prompt_contract=prompt_contract,
        transcript_seed=TranscriptSeed(
            initial_messages=cast(NormalizedMessageList, messages),
            external_history_ref=raw_history_reference,
            history_authority=history_authority,
        ),
        data_from_selfref=data_from_selfref,
    )


__all__ = ["build_chat_invocation_spec", "build_function_invocation_spec"]
