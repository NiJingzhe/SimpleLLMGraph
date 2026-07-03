from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from SimpleLLMFunc.hooks.abort import ABORT_SIGNAL_PARAM, AbortSignal
from SimpleLLMFunc.llm_decorator.chat_toolkit import (
    resolve_effective_self_reference,
    resolve_runtime_toolkit,
)
from SimpleLLMFunc.llm_decorator.chat_types import ToolkitList
from SimpleLLMFunc.llm_decorator.signature import FunctionSignature, parse_function_signature
from SimpleLLMFunc.runtime.selfref.state import SelfReference
from SimpleLLMFunc.type.chat_input import normalize_user_chat_message


@dataclass(frozen=True)
class ChatCallContext:
    signature: FunctionSignature
    template_params: Optional[Dict[str, Any]]
    runtime_toolkit: Optional[ToolkitList]
    effective_self_reference: Optional[SelfReference]
    user_task_prompt: str
    abort_signal: Optional[AbortSignal]


def build_chat_call_context(
    *,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_toolkit: Optional[ToolkitList],
    explicit_self_reference: Optional[SelfReference],
) -> ChatCallContext:
    abort_signal = kwargs.pop(ABORT_SIGNAL_PARAM, None)
    if not isinstance(abort_signal, AbortSignal):
        abort_signal = None

    function_signature, template_params = parse_function_signature(
        func,
        args,
        kwargs,
    )
    runtime_toolkit = resolve_runtime_toolkit(default_toolkit, template_params)
    effective_self_reference = resolve_effective_self_reference(
        explicit_self_reference,
        runtime_toolkit,
    )

    task_arguments = dict(function_signature.bound_args.arguments)
    message_value = task_arguments.get("message")
    if "message" in task_arguments:
        try:
            user_task_payload: Any = normalize_user_chat_message(message_value)
        except ValueError:
            user_task_payload = "" if message_value is None else str(message_value)
    else:
        user_task_payload = task_arguments

    if isinstance(user_task_payload, str):
        user_task_prompt = user_task_payload
    else:
        user_task_prompt = json.dumps(
            user_task_payload,
            default=str,
            ensure_ascii=False,
        )

    return ChatCallContext(
        signature=function_signature,
        template_params=template_params,
        runtime_toolkit=runtime_toolkit,
        effective_self_reference=effective_self_reference,
        user_task_prompt=user_task_prompt,
        abort_signal=abort_signal,
    )


__all__ = ["ChatCallContext", "build_chat_call_context"]
