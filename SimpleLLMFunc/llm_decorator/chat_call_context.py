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

    user_task_prompt = json.dumps(
        function_signature.bound_args.arguments,
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
