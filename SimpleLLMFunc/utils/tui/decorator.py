"""Decorator that launches a Textual chat UI for llm_chat agents."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Optional, Sequence

from SimpleLLMFunc.hooks.stream import ReactOutput
from SimpleLLMFunc.llm_decorator.prompt_contract import HISTORY_PARAM_NAMES
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.utils.tui.hooks import ToolCustomEventHook

try:
    from SimpleLLMFunc.utils.tui.app import AgentTUIApp
except Exception:  # pragma: no cover - tested via patched symbol
    AgentTUIApp = None


def _resolve_chat_parameters(
    agent_func: Callable[..., AsyncGenerator[ReactOutput, None]],
) -> tuple[str, Optional[str]]:
    """Resolve the user-input parameter and optional history parameter."""

    signature = inspect.signature(agent_func)
    param_names = list(signature.parameters.keys())

    history_param = next(
        (name for name in HISTORY_PARAM_NAMES if name in signature.parameters),
        None,
    )
    input_candidates = [name for name in param_names if name not in HISTORY_PARAM_NAMES]

    if not input_candidates:
        raise ValueError(
            "tui requires at least one non-history parameter for user input"
        )

    return input_candidates[0], history_param


def _build_runtime_kwargs(
    agent_func: Callable[..., AsyncGenerator[ReactOutput, None]],
    input_param: str,
    history_param: Optional[str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], MessageList]:
    """Build static kwargs and initial history from launcher args."""

    signature = inspect.signature(agent_func)
    bound = signature.bind_partial(*args, **kwargs)

    static_kwargs: dict[str, Any] = {}
    initial_history: MessageList = []

    for name, parameter in signature.parameters.items():
        if name == input_param:
            continue

        if name == history_param:
            value = bound.arguments.get(name, parameter.default)
            if value is inspect.Signature.empty or value is None:
                initial_history = []
            else:
                if not isinstance(value, list):
                    raise ValueError(f"history parameter '{name}' must be a list")
                initial_history = list(value)
            continue

        if name in bound.arguments:
            static_kwargs[name] = bound.arguments[name]
            continue

        if parameter.default is inspect.Signature.empty:
            raise ValueError(f"tui launcher missing required parameter: {name}")

        static_kwargs[name] = parameter.default

    return static_kwargs, initial_history


def tui(
    custom_event_hook: Optional[Sequence[ToolCustomEventHook]] = None,
    title: str = "SimpleLLMFunc TUI",
) -> Callable[
    [Callable[..., AsyncGenerator[ReactOutput, None]]],
    Callable[..., None],
]:
    """Wrap an event-streaming llm_chat function with an interactive TUI."""

    def decorator(
        agent_func: Callable[..., AsyncGenerator[ReactOutput, None]],
    ) -> Callable[..., None]:
        input_param, history_param = _resolve_chat_parameters(agent_func)

        @wraps(agent_func)
        def launcher(*args: Any, **kwargs: Any) -> None:
            if AgentTUIApp is None:
                raise ImportError(
                    "Textual is required for @tui. Install dependency 'textual'."
                )

            static_kwargs, initial_history = _build_runtime_kwargs(
                agent_func=agent_func,
                input_param=input_param,
                history_param=history_param,
                args=args,
                kwargs=kwargs,
            )

            app = AgentTUIApp(
                agent_func=agent_func,
                input_param=input_param,
                history_param=history_param,
                static_kwargs=static_kwargs,
                custom_hooks=custom_event_hook,
                title_text=title,
                initial_history=initial_history,
            )
            app.run()

        return launcher

    return decorator


__all__ = [
    "tui",
    "_resolve_chat_parameters",
]
