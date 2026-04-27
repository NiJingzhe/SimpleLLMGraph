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


def test_base_does_not_keep_name_only_compatibility_modules() -> None:
    """If the source module can export directly, base should not keep alias-only files."""

    forbidden = [
        BASE_ROOT / "ReAct.py",
        BASE_ROOT / "context_source.py",
        BASE_ROOT / "mutation.py",
    ]

    assert [path for path in forbidden if path.exists()] == []


def test_internal_modules_use_direct_source_modules() -> None:
    import SimpleLLMFunc.llm_decorator.llm_chat_decorator as chat_module
    import SimpleLLMFunc.llm_decorator.llm_function_decorator as function_module
    import SimpleLLMFunc as package_root
    import SimpleLLMFunc.runtime.selfref.context_ops as selfref_context_ops
    import SimpleLLMFunc.runtime.selfref.session as selfref_session
    import SimpleLLMFunc.runtime.selfref.state as selfref_state

    for module in [
        chat_module,
        function_module,
        package_root,
        selfref_context_ops,
        selfref_session,
        selfref_state,
    ]:
        source = inspect.getsource(module)
        assert "base.ReAct" not in source
        assert "base.context_source" not in source
        assert "base.mutation" not in source


def test_base_init_advertises_types_not_removed_compatibility_modules() -> None:
    import SimpleLLMFunc.base as base

    assert "ReAct" not in getattr(base, "__all__", [])
    assert "types" in getattr(base, "__all__", [])
