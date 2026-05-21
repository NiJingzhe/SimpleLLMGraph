from __future__ import annotations

from typing import Any, Dict, Optional, cast

from SimpleLLMFunc.llm_decorator.chat_types import FORK_CLONED_PYREPL_ATTR, ToolkitList
from SimpleLLMFunc.runtime.selfref.state import (
    SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM,
    SelfReference,
)
from SimpleLLMFunc.tool import Tool


def clone_toolkit_for_fork(
    base_toolkit: Optional[ToolkitList],
    self_reference_override: Optional[SelfReference],
) -> Optional[ToolkitList]:
    if base_toolkit is None:
        return None

    from SimpleLLMFunc.builtin import PyRepl

    cloned_toolkit: ToolkitList = []
    repl_clones: Dict[int, PyRepl] = {}
    backend_overrides: Optional[Dict[str, Any]] = None
    if self_reference_override is not None:
        backend_overrides = {
            PyRepl.DEFAULT_SELF_REFERENCE_BACKEND_NAME: self_reference_override,
        }

    for item in base_toolkit:
        if isinstance(item, Tool):
            bound_instance = getattr(item.func, "__self__", None)
            if isinstance(bound_instance, PyRepl):
                original_repl_id = id(bound_instance)
                if original_repl_id not in repl_clones:
                    replacement_repl = bound_instance._clone_for_fork(
                        backend_overrides=backend_overrides,
                    )
                    setattr(replacement_repl, FORK_CLONED_PYREPL_ATTR, True)

                    repl_clones[original_repl_id] = replacement_repl

                replacement_repl = repl_clones[original_repl_id]
                replacement_tool = next(
                    tool for tool in replacement_repl.toolset if tool.name == item.name
                )
                cloned_toolkit.append(replacement_tool)
                continue

        cloned_toolkit.append(item)

    return cloned_toolkit


def close_fork_cloned_pyrepls(toolkit: Optional[ToolkitList]) -> None:
    if not toolkit:
        return

    from SimpleLLMFunc.builtin import PyRepl

    closed_repl_ids: set[int] = set()

    for item in toolkit:
        if not isinstance(item, Tool):
            continue

        bound_instance = getattr(item.func, "__self__", None)
        if not isinstance(bound_instance, PyRepl):
            continue
        if not bool(getattr(bound_instance, FORK_CLONED_PYREPL_ATTR, False)):
            continue

        repl_id = id(bound_instance)
        if repl_id in closed_repl_ids:
            continue
        closed_repl_ids.add(repl_id)

        try:
            bound_instance.close()
        except Exception:
            continue


def extract_self_reference_from_toolkit(
    toolkit: Optional[ToolkitList],
) -> Optional[SelfReference]:
    if not toolkit:
        return None

    from SimpleLLMFunc.builtin import PyRepl

    discovered: Dict[int, SelfReference] = {}

    for item in toolkit:
        if not isinstance(item, Tool):
            continue

        bound_instance = getattr(item.func, "__self__", None)
        if not isinstance(bound_instance, PyRepl):
            continue

        default_backend = bound_instance.get_runtime_backend(
            PyRepl.DEFAULT_SELF_REFERENCE_BACKEND_NAME
        )
        if isinstance(default_backend, SelfReference):
            discovered[id(default_backend)] = default_backend
            continue

        for backend_name in bound_instance.list_runtime_backends():
            backend_value = bound_instance.get_runtime_backend(backend_name)
            if isinstance(backend_value, SelfReference):
                discovered[id(backend_value)] = backend_value

    if not discovered:
        return None

    return next(iter(discovered.values()))


def resolve_effective_self_reference(
    explicit_self_reference: Optional[SelfReference],
    toolkit: Optional[ToolkitList],
) -> Optional[SelfReference]:
    if explicit_self_reference is not None:
        return explicit_self_reference
    return extract_self_reference_from_toolkit(toolkit)


def resolve_runtime_toolkit(
    default_toolkit: Optional[ToolkitList],
    template_params: Optional[Dict[str, Any]],
) -> Optional[ToolkitList]:
    if template_params is None:
        return default_toolkit

    override_toolkit = template_params.get(
        SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM
    )
    if override_toolkit is None:
        return default_toolkit

    if not isinstance(override_toolkit, list):
        raise ValueError("self_reference toolkit override must be a list")

    if not default_toolkit:
        return cast(Optional[ToolkitList], override_toolkit)

    seen_names: set[str] = set()
    merged: ToolkitList = []

    for item in override_toolkit:
        if isinstance(item, Tool):
            seen_names.add(item.name)
        elif callable(item):
            seen_names.add(getattr(item, "__name__", ""))

    for item in default_toolkit:
        if isinstance(item, Tool):
            if item.name not in seen_names:
                merged.append(item)
        elif callable(item):
            name = getattr(item, "__name__", "")
            if name not in seen_names:
                merged.append(item)

    merged.extend(override_toolkit)

    return merged


# Backward-compatible internal aliases for existing tests/users that imported
# these helpers from llm_chat_decorator.
_clone_toolkit_for_fork = clone_toolkit_for_fork
_close_fork_cloned_pyrepls = close_fork_cloned_pyrepls
_extract_self_reference_from_toolkit = extract_self_reference_from_toolkit
_resolve_effective_self_reference = resolve_effective_self_reference
_resolve_runtime_toolkit = resolve_runtime_toolkit

__all__ = [
    "clone_toolkit_for_fork",
    "close_fork_cloned_pyrepls",
    "extract_self_reference_from_toolkit",
    "resolve_effective_self_reference",
    "resolve_runtime_toolkit",
    "_clone_toolkit_for_fork",
    "_close_fork_cloned_pyrepls",
    "_extract_self_reference_from_toolkit",
    "_resolve_effective_self_reference",
    "_resolve_runtime_toolkit",
]
