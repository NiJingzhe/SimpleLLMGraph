"""Test that _resolve_runtime_toolkit merges toolkits instead of replacing."""

import pytest
from SimpleLLMFunc.llm_decorator.llm_chat_decorator import _resolve_runtime_toolkit
from SimpleLLMFunc.runtime.selfref.state import SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM


class TestResolveRuntimeToolkitMerge:
    """_resolve_runtime_toolkit should merge default and override toolkits."""

    def test_no_template_params_returns_default(self):
        default = ["default_tool"]
        result = _resolve_runtime_toolkit(default, None)
        assert result == default

    def test_no_override_key_returns_default(self):
        default = ["default_tool"]
        result = _resolve_runtime_toolkit(default, {"other_key": "value"})
        assert result == default

    def test_no_default_returns_override(self):
        override = ["override_tool"]
        template_params = {SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM: override}
        result = _resolve_runtime_toolkit(None, template_params)
        assert result == override

    def test_empty_default_returns_override(self):
        override = ["override_tool"]
        template_params = {SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM: override}
        result = _resolve_runtime_toolkit([], template_params)
        assert result == override

    def test_merges_when_both_exist(self):
        def tool_a(): pass
        def tool_b(): pass
        def tool_c(): pass
        def tool_x(): pass
        def tool_y(): pass

        default = [tool_a, tool_b, tool_c]
        override = [tool_x, tool_y]
        template_params = {SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM: override}
        result = _resolve_runtime_toolkit(default, template_params)
        # All default tools should be present (none clash)
        assert tool_a in result
        assert tool_b in result
        assert tool_c in result
        # All override tools should be present
        assert tool_x in result
        assert tool_y in result
        assert len(result) == 5

    def test_override_takes_priority_on_name_clash(self):
        """When a tool name exists in both, the override version should be used."""

        def default_shared():
            return "default"

        def override_shared():
            return "override"

        default_shared.__name__ = "shared_tool"
        override_shared.__name__ = "shared_tool"

        default = [default_shared]
        override = [override_shared]
        template_params = {SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM: override}
        result = _resolve_runtime_toolkit(default, template_params)

        # Should have exactly one tool named "shared_tool"
        shared_tools = [t for t in result if callable(t) and getattr(t, "__name__", "") == "shared_tool"]
        assert len(shared_tools) == 1
        # It should be the override version
        assert shared_tools[0] is override_shared

    def test_callable_tools_also_dedup_by_name(self):
        def default_func():
            pass

        def override_func():
            pass

        default_func.__name__ = "my_tool"
        override_func.__name__ = "my_tool"

        default = [default_func]
        override = [override_func]
        template_params = {SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM: override}
        result = _resolve_runtime_toolkit(default, template_params)

        # Should have exactly one tool
        assert len(result) == 1
        assert result[0] is override_func

    def test_tool_objects_dedup_by_name(self):
        """Tool objects should be deduplicated by their .name attribute."""
        from SimpleLLMFunc.tool import Tool

        async def default_execute(): pass
        async def override_execute(): pass

        default_tool = Tool(name="execute_code", description="Default", func=default_execute)
        override_tool = Tool(name="execute_code", description="Override", func=override_execute)
        extra_tool = Tool(name="reset_repl", description="Extra", func=override_execute)

        default = [default_tool]
        override = [override_tool, extra_tool]
        template_params = {SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM: override}
        result = _resolve_runtime_toolkit(default, template_params)

        # Should have exactly 2 tools (default deduped, override kept)
        assert len(result) == 2
        execute_tools = [t for t in result if isinstance(t, Tool) and t.name == "execute_code"]
        assert len(execute_tools) == 1
        # The override version should win
        assert execute_tools[0] is override_tool
