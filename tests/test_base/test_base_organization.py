from __future__ import annotations

import importlib
import inspect
from pathlib import Path


BASE_ROOT = Path(__file__).resolve().parents[2] / "SimpleLLMFunc" / "base"


def test_base_does_not_keep_context_source_compile_shim() -> None:
    """Base should not keep module-level shims that only re-export one function."""

    assert not (BASE_ROOT / "context_source_compile.py").exists()


def test_compile_pipeline_has_no_unused_llm_request_wrapper() -> None:
    """Provider-facing output is CompiledTurnContext; no one-field request wrapper."""

    module = importlib.import_module("SimpleLLMFunc.base.compile_pipeline")

    assert not hasattr(module, "LLMRequest")
    assert "LLMRequest" not in getattr(module, "__all__", [])


def test_react_loop_entrypoint_lives_in_react_loop_module() -> None:
    """Keep base.ReAct as compatibility only; main loop lives with core loop code."""

    react_loop = importlib.import_module("SimpleLLMFunc.base.react_loop")
    compat_react = importlib.import_module("SimpleLLMFunc.base.ReAct")

    assert hasattr(react_loop, "ReAct_loop")
    assert compat_react.ReAct_loop is react_loop.ReAct_loop
    assert "async def ReAct_loop" not in inspect.getsource(compat_react)


def test_mutation_module_is_compatibility_only() -> None:
    mutation_module = importlib.import_module("SimpleLLMFunc.base.mutation")

    assert "@dataclass" not in inspect.getsource(mutation_module)


def test_base_init_does_not_advertise_removed_compatibility_modules() -> None:
    import SimpleLLMFunc.base as base

    assert "ReAct" not in getattr(base, "__all__", [])
