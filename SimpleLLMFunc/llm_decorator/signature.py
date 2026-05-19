"""Decorator boundary helpers: signature binding and log context."""

from __future__ import annotations

import inspect
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, AsyncGenerator, Callable, Dict, NamedTuple, Optional, Tuple, cast, get_type_hints

from langfuse.types import TraceContext

from SimpleLLMFunc.logger import app_log, async_log_context
from SimpleLLMFunc.logger.logger import get_current_trace_id, get_location
from SimpleLLMFunc.observability.langfuse_client import (
    get_langfuse_trace_context,
    langfuse_client,
    reset_langfuse_trace_context,
    set_langfuse_trace_context,
)


class FunctionSignature(NamedTuple):
    """Bound Python call metadata used to build an InvocationSpec."""

    func_name: str
    trace_id: str
    bound_args: inspect.BoundArguments
    signature: inspect.Signature
    type_hints: Dict[str, Any]
    return_type: Any
    docstring: str


def extract_template_params(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return kwargs.pop("_template_params", None)


def extract_function_metadata(
    func: Callable[..., Any],
) -> Tuple[inspect.Signature, Dict[str, Any], Any, str, str]:
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return")
    docstring = func.__doc__ or ""
    func_name = func.__name__
    return signature, type_hints, return_type, docstring, func_name


def generate_trace_id(func_name: str) -> str:
    context_trace_id = get_current_trace_id()
    current_trace_id = f"{func_name}_{uuid.uuid4()}"
    if context_trace_id:
        current_trace_id += f"_{context_trace_id}"
    return current_trace_id


def bind_function_arguments(
    signature: inspect.Signature,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> inspect.BoundArguments:
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args


def build_function_signature(
    func_name: str,
    trace_id: str,
    bound_args: inspect.BoundArguments,
    signature: inspect.Signature,
    type_hints: Dict[str, Any],
    return_type: Any,
    docstring: str,
) -> FunctionSignature:
    return FunctionSignature(
        func_name=func_name,
        trace_id=trace_id,
        bound_args=bound_args,
        signature=signature,
        type_hints=type_hints,
        return_type=return_type,
        docstring=docstring,
    )


def parse_function_signature(
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[FunctionSignature, Optional[Dict[str, Any]]]:
    template_params = extract_template_params(kwargs)
    signature, type_hints, return_type, docstring, func_name = extract_function_metadata(
        func
    )
    trace_id = generate_trace_id(func_name)
    bound_args = bind_function_arguments(signature, args, kwargs)
    return (
        build_function_signature(
            func_name=func_name,
            trace_id=trace_id,
            bound_args=bound_args,
            signature=signature,
            type_hints=type_hints,
            return_type=return_type,
            docstring=docstring,
        ),
        template_params,
    )


def log_function_call(func_name: str, arguments: Dict[str, Any]) -> None:
    args_str = json.dumps(arguments, default=str, ensure_ascii=False, indent=4)
    app_log(
        f"Async LLM function '{func_name}' called with arguments: {args_str}",
        location=get_location(),
    )


def create_log_context_manager(
    func_name: str,
    trace_id: str,
) -> AsyncContextManager[None]:
    return async_log_context(
        trace_id=trace_id,
        function_name=func_name,
        input_tokens=0,
        output_tokens=0,
    )


@asynccontextmanager
async def _log_context_manager(
    base_manager: AsyncContextManager[None],
) -> AsyncGenerator[None, None]:
    trace_context = get_langfuse_trace_context()
    trace_token: Optional[object] = None
    if trace_context is None:
        trace_context = cast(
            TraceContext,
            {"trace_id": langfuse_client.create_trace_id()},
        )
        trace_token = set_langfuse_trace_context(trace_context)

    async with base_manager:
        try:
            yield
        finally:
            if trace_token is not None:
                reset_langfuse_trace_context(trace_token)


def setup_log_context(
    func_name: str,
    trace_id: str,
    arguments: Dict[str, Any],
) -> AsyncContextManager[None]:
    log_function_call(func_name, arguments)
    return _log_context_manager(create_log_context_manager(func_name, trace_id))


__all__ = [
    "FunctionSignature",
    "bind_function_arguments",
    "build_function_signature",
    "create_log_context_manager",
    "extract_function_metadata",
    "extract_template_params",
    "generate_trace_id",
    "log_function_call",
    "parse_function_signature",
    "setup_log_context",
]
