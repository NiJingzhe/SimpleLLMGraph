"""Tests for PyRepl builtin tool."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import sys
from typing import Any, cast
from pathlib import Path

import pytest

from SimpleLLMFunc.hooks.events import CustomEvent


async def _wait_for_input_request(
    emitter,
    seen_request_ids: set[str] | None = None,
    timeout: float = 15.0,
) -> tuple[str, str]:
    """Wait until one unseen kernel_input_request event is emitted."""

    seen = seen_request_ids if seen_request_ids is not None else set()
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        events = await emitter.get_events()
        for event_yield in events:
            event = event_yield.event
            if not isinstance(event, CustomEvent):
                continue
            if event.event_name != "kernel_input_request":
                continue

            data = getattr(event, "data", None)
            if not isinstance(data, dict):
                continue

            request_id = data.get("request_id")
            prompt = data.get("prompt", "")
            if not isinstance(request_id, str) or not request_id:
                continue
            if request_id in seen:
                continue
            if not isinstance(prompt, str):
                prompt = ""

            return request_id, prompt

        await asyncio.sleep(0.01)

    raise AssertionError("Timed out waiting for kernel_input_request event")


def _builtin_self_reference(repl: Any):
    from SimpleLLMFunc.runtime.selfref import SelfReference

    self_reference = repl.get_runtime_backend("selfref")
    assert isinstance(self_reference, SelfReference)
    return self_reference


class TestPyReplCreation:
    """Test PyRepl class creation."""

    def test_repl_creation(self):
        """Test creating a PyRepl instance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        assert repl is not None
        assert repl.namespace == {}

    def test_repl_has_lock(self):
        """Test that PyRepl has a lock for thread safety."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        assert repl._lock is not None

    def test_close_releases_worker_queue_handles(self):
        """close() should release multiprocessing queue handles promptly."""
        from unittest.mock import MagicMock

        from SimpleLLMFunc.builtin import PyRepl

        class _DeadProcess:
            def is_alive(self) -> bool:
                return False

            def close(self) -> None:
                return None

        repl = PyRepl()
        command_queue = MagicMock()
        event_queue = MagicMock()
        process = _DeadProcess()
        process.close = MagicMock()

        repl._process = process
        repl._command_queue = command_queue
        repl._event_queue = event_queue

        repl.close()

        process.close.assert_called_once()
        command_queue.close.assert_called_once()
        command_queue.join_thread.assert_called_once()
        event_queue.close.assert_called_once()
        event_queue.join_thread.assert_called_once()

    def test_self_reference_exported_from_builtin_primitive(self):
        """SelfReference should be importable from builtin.primitive."""
        from SimpleLLMFunc.builtin.primitive import (
            SelfReference as BuiltinSelfReference,
        )
        from SimpleLLMFunc.runtime.selfref import SelfReference

        assert BuiltinSelfReference is SelfReference

    def test_self_reference_exported_from_builtin_package(self):
        """SelfReference should be re-exported from builtin package."""
        from SimpleLLMFunc.builtin import SelfReference as BuiltinSelfReference
        from SimpleLLMFunc.runtime.selfref import SelfReference

        assert BuiltinSelfReference is SelfReference

    def test_repl_timeout_defaults(self):
        """PyRepl should expose documented timeout defaults."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        assert (
            repl.execution_timeout_seconds == PyRepl.DEFAULT_EXECUTION_TIMEOUT_SECONDS
        )
        assert (
            repl.input_idle_timeout_seconds == PyRepl.DEFAULT_INPUT_IDLE_TIMEOUT_SECONDS
        )

    def test_repl_rejects_non_positive_timeouts(self):
        """Timeout values should be validated at construction time."""
        from SimpleLLMFunc.builtin import PyRepl

        with pytest.raises(ValueError, match="execution_timeout_seconds"):
            PyRepl(execution_timeout_seconds=0)

        with pytest.raises(ValueError, match="input_idle_timeout_seconds"):
            PyRepl(input_idle_timeout_seconds=0)

    def test_repl_rejects_invalid_working_directory(self, tmp_path: Path) -> None:
        """PyRepl should reject non-directory working_directory values."""
        from SimpleLLMFunc.builtin import PyRepl

        missing = tmp_path / "missing"
        file_path = tmp_path / "file.txt"
        file_path.write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="working_directory"):
            PyRepl(working_directory=missing)

        with pytest.raises(ValueError, match="working_directory"):
            PyRepl(working_directory=file_path)

        with pytest.raises(ValueError, match="working_directory"):
            PyRepl(working_directory=cast(Any, 123))

    @pytest.mark.asyncio
    async def test_repl_uses_working_directory(self, tmp_path: Path) -> None:
        """PyRepl should start worker in the provided working directory."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(working_directory=tmp_path)
        result = await repl.execute("import os\nprint(os.getcwd())")

        assert result["success"] is True, result["error"] or result["stderr"]
        assert Path(result["stdout"].strip()).resolve() == tmp_path.resolve()


class TestPyReplToolset:
    """Test PyRepl.toolset property."""

    def test_toolset_returns_list(self):
        """Test that toolset returns a list."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        toolset = repl.toolset
        assert isinstance(toolset, list)

    def test_toolset_contains_expected_tools(self):
        """Test that toolset contains expected tool names."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        toolset = repl.toolset

        tool_names = [tool.name for tool in toolset]
        assert "execute_code" in tool_names
        assert "reset_repl" in tool_names
        assert "list_variables" not in tool_names

    def test_list_variables_api_removed(self):
        """list_variables API should not be exposed on PyRepl."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        assert not hasattr(repl, "list_variables")

    def test_execute_tool_description_has_repl_guidance(self):
        """execute_code description should stay focused on REPL execution itself."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        description = execute_tool.description
        assert "persistent REPL session" in description
        assert "top-level executable code" in description
        assert "input()" in description
        assert "timeout_seconds" in description
        assert "runtime.list_primitives()" not in description
        assert "runtime.get_primitive_spec(name)" not in description
        assert "selfref = your agent state" not in description
        assert "runtime memory is unchanged" not in description

    def test_execute_tool_prompt_includes_working_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """Prompt injection should include working_directory guidance when configured."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(working_directory=tmp_path)
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        prompt = execute_tool.build_system_prompt_injection()

        assert isinstance(prompt, str)
        assert "Working directory:" in prompt
        assert tmp_path.resolve().as_posix() in prompt

    def test_execute_tool_prompt_omits_working_directory_when_unset(self) -> None:
        """Prompt injection should skip working_directory guidance when unset."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        prompt = execute_tool.build_system_prompt_injection()

        assert isinstance(prompt, str)
        assert "Working directory:" not in prompt

    def test_execute_tool_prompt_includes_installed_pack_guidance(self) -> None:
        """Prompt injection should include installed pack guidance as plain text."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        prompt = execute_tool.build_system_prompt_injection()

        assert isinstance(prompt, str)
        assert "<runtime_primitive_contract>" in prompt
        assert "Use this block for orientation" in prompt
        assert "Installed primitive packs:" in prompt
        assert "- selfref:" in prompt
        assert "selfref = your agent context" in prompt
        assert "parallel sub-agent decomposition" in prompt

    def test_execute_tool_prompt_explains_primitive_vs_tool_boundary(self) -> None:
        """Runtime guidance should say primitives run inside execute_code, not as standalone tools."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        prompt = execute_tool.build_system_prompt_injection()

        assert isinstance(prompt, str)
        assert "Runtime primitives are not standalone tool calls." in prompt
        assert "Call them inside execute_code as runtime.namespace.name(...)." in prompt
        assert (
            'For example: execute_code(code="runtime.selfref.fork.spawn(...)")'
            in prompt
        )

    def test_execute_tool_prompt_includes_custom_pack_guidance(self) -> None:
        """Prompt injection should render generic pack guidance, not just selfref."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        pack = repl.pack(
            "demo",
            backend=object(),
            guidance="demo = a scratch runtime namespace for small host-side helpers.",
        )

        @pack.primitive("ping")
        def demo_ping(ctx):
            """
            Use: Health check for the demo backend.
            Output: `str`.
            Best Practices:
            - Use for smoke tests only.
            """

            _ = ctx
            return "pong"

        repl.install_pack(pack)
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        prompt = execute_tool.build_system_prompt_injection()

        assert isinstance(prompt, str)
        assert "Installed primitive packs:" in prompt
        assert (
            "- demo: demo = a scratch runtime namespace for small host-side helpers."
            in prompt
        )

    def test_execute_tool_schema_exposes_timeout_seconds(self):
        """execute_code tool schema should expose per-call timeout controls."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        function_schema = execute_tool.to_openai_tool()["function"]
        params_schema = function_schema["parameters"]
        properties = params_schema["properties"]

        assert "timeout_seconds" in properties
        timeout_type = properties["timeout_seconds"].get("type")
        if isinstance(timeout_type, list):
            assert "number" in timeout_type
        else:
            assert timeout_type == "number"

        required = params_schema.get("required", [])
        assert "timeout_seconds" not in required

    def test_all_tool_descriptions_are_english_guidance(self):
        """Builtin tool descriptions should be explicit English guidance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        descriptions = {tool.name: tool.description for tool in repl.toolset}

        assert "Reset REPL runtime variables" in descriptions["reset_repl"]
        assert (
            "preserving registered runtime primitive backends"
            in descriptions["reset_repl"]
        )

    def test_tool_best_practices_stay_tool_scoped(self) -> None:
        """Tool best practices should stay generic and leave selfref guidance to selfref."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )
        reset_tool = next(tool for tool in repl.toolset if tool.name == "reset_repl")

        execute_text = " ".join(execute_tool.best_practices)
        reset_text = " ".join(reset_tool.best_practices)

        assert "contains='<namespace>.'" in execute_text
        assert "selfref" not in execute_text
        assert "fresh REPL variable namespace" in reset_text
        assert "selfref" not in reset_text


class TestPyReplToolsetOutput:
    """Toolset outputs should be natural language strings."""

    @pytest.mark.asyncio
    async def test_execute_tool_returns_text_summary(self):
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        result = await execute_tool.run("print('hello')")

        assert isinstance(result, str)
        assert "Execution succeeded" in result
        assert "stdout:" in result
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_tool_returns_multimodal_output_for_image_artifact(self):
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.type import ImgPath

        repl = PyRepl()
        execute_tool = next(
            tool for tool in repl.toolset if tool.name == "execute_code"
        )

        result = await execute_tool.run(
            """
from IPython.display import Image
Image(data=b'fake image data', format='png')
"""
        )

        assert isinstance(result, tuple)
        summary, images = result
        assert isinstance(summary, str)
        assert "Execution succeeded" in summary
        assert "image artifact" in summary
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], ImgPath)


class TestPyReplExecute:
    """Test PyRepl execute functionality."""

    @pytest.mark.asyncio
    async def test_execute_simple_print(self):
        """Test basic print execution."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("print('hello')")

        assert result["success"] is True, result["error"] or result["stderr"]
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_variable_assignment(self):
        """Test variable assignment."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("x = 100")

        assert result["success"] is True, result["error"] or result["stderr"]

    @pytest.mark.asyncio
    async def test_variable_persistence(self):
        """Test that variables persist across execute calls."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        await repl.execute("x = 100")
        result = await repl.execute("print(x * 2)")

        assert result["success"] is True
        assert "200" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test error handling."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("1/0")

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_expression_result(self):
        """Test expression evaluation returns result."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("1 + 1")

        assert result["success"] is True
        assert result["return_value"] == "2"

    @pytest.mark.asyncio
    async def test_execute_expression_image_result_records_artifact(self):
        """Expression display images should be returned as structured artifacts."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            """
from IPython.display import Image
Image(data=b'fake image data', format='png')
"""
        )

        assert result["success"] is True, result["error"] or result["stderr"]
        assert result["return_value"] is None
        artifacts = result["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "image"
        assert artifacts[0]["mime_type"] == "image/png"
        assert Path(artifacts[0]["path"]).exists()

    @pytest.mark.asyncio
    async def test_execute_display_image_call_records_artifact(self):
        """Explicit display(Image(...)) calls should be returned as artifacts."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            """
from IPython.display import Image, display
display(Image(data=b'fake image data', format='png'))
"""
        )

        assert result["success"] is True, result["error"] or result["stderr"]
        artifacts = result["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "image"
        assert artifacts[0]["mime_type"] == "image/png"
        assert Path(artifacts[0]["path"]).exists()

    @pytest.mark.asyncio
    async def test_execute_display_mixed_content_preserves_text_output(self):
        """display(Image(...), text) should capture images and keep text display."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            """
from IPython.display import Image, display
display(Image(data=b'fake image data', format='png'), "keep text")
"""
        )

        assert result["success"] is True, result["error"] or result["stderr"]
        assert "keep text" in result["stdout"]
        artifacts = result["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_execute_runtime_error_includes_structured_details(self):
        """Runtime errors should provide line-aware structured diagnostics."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("x = 1\ny = 0\nx / y")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert "ZeroDivisionError" in result["error"]

        details = result["error_details"]
        assert isinstance(details, dict)
        assert details["error_type"] == "ZeroDivisionError"
        assert details["line"] == 3
        assert details["snippet"] == "x / y"

    @pytest.mark.asyncio
    async def test_execute_syntax_error_includes_snippet_and_pointer(self):
        """Syntax errors should expose exact snippet location information."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("for i in range(2)\n    print(i)")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert "SyntaxError" in result["error"]

        details = result["error_details"]
        assert isinstance(details, dict)
        assert details["error_type"] == "SyntaxError"
        assert details["line"] == 1
        assert details["column"] == 18
        assert details["snippet"] == "for i in range(2)"
        assert details["pointer"] == " " * 17 + "^"

    @pytest.mark.asyncio
    async def test_execute_import_runtime_includes_hint(self):
        """Importing runtime should return a guidance hint."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("import runtime")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert "runtime" in result["error"]
        assert "cannot be imported" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_handles_stderr_with_invalid_fileno(self, monkeypatch):
        """Worker startup should survive environments where stderr has fileno=-1."""
        from SimpleLLMFunc.builtin import PyRepl

        class _InvalidStderr:
            def fileno(self) -> int:
                return -1

            def write(self, _text: str) -> int:
                return 0

            def flush(self) -> None:
                return None

        monkeypatch.setattr(sys, "stderr", _InvalidStderr())

        repl = PyRepl()
        try:
            result = await repl.execute("print('ok')")
        finally:
            repl.close()

        assert result["success"] is True
        assert "ok" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_worker_standard_fds_are_isolated(self):
        """Worker fd 0/1/2 should not inherit a host tty by default."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        try:
            result = await repl.execute(
                "import os\n"
                "for fd in (0, 1, 2):\n"
                "    print(f'{fd}:{os.isatty(fd)}')"
            )
        finally:
            repl.close()

        assert result["success"] is True, result["error"] or result["stderr"]
        assert result["stdout"].splitlines() == ["0:False", "1:False", "2:False"]

    @pytest.mark.asyncio
    async def test_execute_captures_direct_fd_writes(self):
        """Direct os.write output should be captured into execute results."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        try:
            result = await repl.execute(
                "import os\n"
                "os.write(1, b'DIRECT_FD1_MARKER\\n')\n"
                "os.write(2, b'DIRECT_FD2_MARKER\\n')"
            )
        finally:
            repl.close()

        assert result["success"] is True, result["error"] or result["stderr"]
        assert "DIRECT_FD1_MARKER" in result["stdout"]
        assert "DIRECT_FD2_MARKER" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_captures_child_process_fd_writes(self):
        """Subprocess fd output should use the PyRepl output channel."""
        from SimpleLLMFunc.builtin import PyRepl

        child_code = (
            "import os; "
            "os.write(1, b'CHILD_FD1_MARKER\\n'); "
            "os.write(2, b'CHILD_FD2_MARKER\\n')"
        )
        repl = PyRepl()
        try:
            result = await repl.execute(
                "import subprocess, sys\n"
                f"subprocess.run([sys.executable, '-c', {child_code!r}], check=False)"
            )
        finally:
            repl.close()

        assert result["success"] is True, result["error"] or result["stderr"]
        assert "CHILD_FD1_MARKER" in result["stdout"]
        assert "CHILD_FD2_MARKER" in result["stderr"]

    @pytest.mark.asyncio
    async def test_late_child_fd_output_does_not_contaminate_next_execute(self):
        """Long-lived child output should not be attributed to later snippets."""
        from SimpleLLMFunc.builtin import PyRepl

        child_code = (
            "import os, time; "
            "time.sleep(0.15); "
            "os.write(1, b'LATE_CHILD_FD_MARKER\\n')"
        )
        repl = PyRepl()
        try:
            spawn_result = await repl.execute(
                "import subprocess, sys\n"
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
                "print('child spawned')"
            )
            next_result = await repl.execute(
                "import time\n"
                "print('NEXT_EXECUTE_START')\n"
                "time.sleep(0.4)\n"
                "print('NEXT_EXECUTE_END')"
            )
        finally:
            repl.close()

        assert spawn_result["success"] is True, spawn_result["error"] or spawn_result["stderr"]
        assert next_result["success"] is True, next_result["error"] or next_result["stderr"]
        assert "LATE_CHILD_FD_MARKER" not in next_result["stdout"]
        assert next_result["stdout"].splitlines() == [
            "NEXT_EXECUTE_START",
            "NEXT_EXECUTE_END",
        ]

    @pytest.mark.asyncio
    async def test_execute_python_print_is_not_duplicated_by_fd_capture(self):
        """Python print output should still be emitted once via _LineCapture."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        try:
            result = await repl.execute("print('PRINT_MARKER')")
        finally:
            repl.close()

        assert result["success"] is True, result["error"] or result["stderr"]
        assert result["stdout"].splitlines().count("PRINT_MARKER") == 1


class TestPyReplAudit:
    """Test per-instance audit log persistence behavior."""

    @pytest.mark.asyncio
    async def test_each_instance_writes_isolated_audit_log(self, monkeypatch, tmp_path):
        """Each PyRepl instance should persist execution history separately."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.logger.logger_config import logger_config

        monkeypatch.setattr(logger_config, "LOG_DIR", str(tmp_path))

        repl_a = PyRepl()
        repl_b = PyRepl()

        try:
            result_a = await repl_a.execute("print('A')")
            result_b = await repl_b.execute("print('B')")
        finally:
            repl_a.close()
            repl_b.close()

        assert result_a["success"] is True
        assert result_b["success"] is True

        audit_dir_a = Path(repl_a.audit_log_dir)
        audit_dir_b = Path(repl_b.audit_log_dir)
        assert audit_dir_a != audit_dir_b
        assert audit_dir_a.parent == tmp_path / "pyrepl"
        assert audit_dir_b.parent == tmp_path / "pyrepl"

        audit_file_a = Path(repl_a.audit_log_file)
        audit_file_b = Path(repl_b.audit_log_file)
        assert audit_file_a.exists()
        assert audit_file_b.exists()

        records_a = [
            json.loads(line)
            for line in audit_file_a.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records_b = [
            json.loads(line)
            for line in audit_file_b.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert records_a[-1]["code"] == "print('A')"
        assert records_b[-1]["code"] == "print('B')"
        assert records_a[-1]["result"]["success"] is True
        assert records_b[-1]["result"]["success"] is True


class TestPyReplReset:
    """Test PyRepl reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_variables(self):
        """Test that reset clears all variables."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        await repl.execute("x = 100")
        await repl.execute("y = 200")

        result = await repl.reset()
        assert "已重置" in result

        # Verify variables are cleared by attempting to access a cleared name
        post_reset = await repl.execute("x")
        assert post_reset["success"] is False


class TestPyReplPrimitivePacks:
    """Test primitive-pack installation and runtime backend behavior."""

    def test_repl_auto_installs_selfref_pack_registers_backend_and_primitives(self):
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.runtime.selfref import SelfReference

        repl = PyRepl()
        self_reference = repl.get_runtime_backend("selfref")

        assert isinstance(self_reference, SelfReference)
        assert repl.list_runtime_backends() == ["selfref"]
        assert repl.list_installed_packs() == ["selfref"]
        assert "selfref.context.inspect" in repl.list_primitives()
        assert "selfref.fork.spawn" in repl.list_primitives()
        assert "selfref.fork.gather_all" in repl.list_primitives()
        assert "memory.keys" not in repl.list_primitives()
        assert "fork.run" not in repl.list_primitives()
        assert "selfref.fork.run" not in repl.list_primitives()
        assert "selfref.fork.run_chat" not in repl.list_primitives()
        assert "selfref.fork.spawn_chat" not in repl.list_primitives()
        assert "selfref.fork.wait_all" not in repl.list_primitives()
        assert "selfref.fork.wait" not in repl.list_primitives()

    def test_install_unknown_primitive_pack_raises(self):
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        with pytest.raises(KeyError, match="primitive pack"):
            repl.install_primitive_pack("unknown_pack")

    def test_install_legacy_self_reference_pack_name_raises(self):
        """Hard-cut migration: old pack name must be rejected."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.runtime.selfref import SelfReference

        repl = PyRepl()
        with pytest.raises(KeyError, match="primitive pack"):
            repl.install_primitive_pack("self_reference", backend=SelfReference())

    @pytest.mark.asyncio
    async def test_install_pack_supports_backend_aware_custom_primitives(self):
        """First-class PrimitivePack should install backend-aware custom primitives."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        constants = repl.pack(
            "constants",
            backend={
                "app_name": "SimpleLLMFunc",
                "memory_key": "agent_main",
            },
        )

        @constants.primitive(
            "get",
            description="Read one value from constants backend.",
        )
        def constants_get(ctx, key: str):
            """
            Use: Read one value from constants backend.
            Input: `key: str`.
            Output: `str | None`.
            Best Practices:
            - Keep lookups to single keys.
            """
            backend = ctx.backend
            if not isinstance(backend, dict):
                raise RuntimeError("constants backend must be a dict")
            return backend.get(key)

        @constants.primitive("lookup_via_context")
        def constants_lookup_via_context(ctx, key: str):
            """
            Use: Read one value from constants backend via ctx.get_backend.
            Input: `key: str`.
            Output: `str | None`.
            Best Practices:
            - Prefer ctx.backend when available.
            """
            backend = ctx.get_backend("constants")
            if not isinstance(backend, dict):
                raise RuntimeError("constants backend must be a dict")
            return backend.get(key)

        repl.install_pack(constants)

        assert repl.list_installed_packs() == ["constants", "selfref"]
        assert repl.get_runtime_backend("constants") == {
            "app_name": "SimpleLLMFunc",
            "memory_key": "agent_main",
        }
        assert "constants.get" in repl.list_primitives()
        assert "constants.lookup_via_context" in repl.list_primitives()

        result = await repl.execute(
            "print(runtime.constants.get('app_name'))\n"
            "print(runtime.constants.lookup_via_context('memory_key'))\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["SimpleLLMFunc", "agent_main"]

    @pytest.mark.asyncio
    async def test_repl_primitive_decorator_registers_backend_aware_handler(self):
        """PyRepl.primitive decorator should provide low-friction backend binding."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        repl.register_runtime_backend(
            "constants",
            {"project": "SimpleLLMFunc", "version": "dev"},
            replace=True,
        )

        @repl.primitive(
            "constants.get",
            backend="constants",
            description="Read one constant value.",
            replace=True,
        )
        def constants_get(ctx, key: str):
            """
            Use: Read one constant value.
            Input: `key: str`.
            Output: `str | None`.
            Best Practices:
            - Keep calls narrow and avoid bulk reads.
            """
            backend = ctx.backend
            if not isinstance(backend, dict):
                raise RuntimeError("constants backend must be a dict")
            return backend.get(key)

        @repl.primitive(
            "constants.lookup_via_context",
            backend="constants",
            replace=True,
        )
        def constants_lookup_via_context(ctx, key: str):
            """
            Use: Read one constant value via ctx.get_backend.
            Input: `key: str`.
            Output: `str | None`.
            Best Practices:
            - Prefer ctx.backend when available.
            """
            backend = ctx.get_backend("constants")
            if not isinstance(backend, dict):
                raise RuntimeError("constants backend must be a dict")
            return backend.get(key)

        contract = repl.get_primitive_contract("constants.get")
        assert contract["name"] == "constants.get"
        assert any(item["name"] == "key" for item in contract["parameters"])

        result = await repl.execute(
            "print(runtime.constants.get('project'))\n"
            "print(runtime.constants.lookup_via_context('version'))\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["SimpleLLMFunc", "dev"]

    @pytest.mark.asyncio
    async def test_execute_can_mutate_memory_via_runtime_primitives(self):
        """execute_code should mutate context through runtime primitives."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        result = await repl.execute("runtime.selfref.context.remember('ok')\n_ = 1")

        assert result["success"] is True
        persisted = self_reference.snapshot_history("agent_main")
        assert persisted[0]["role"] == "system"
        assert "ok" in str(persisted[0]["content"])
        assert persisted[1:] == [{"role": "user", "content": "seed"}]

    @pytest.mark.asyncio
    async def test_reset_keeps_registered_self_reference_backend(self):
        """reset_repl should preserve installed runtime backend registration."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)

        await repl.execute("x = 1")
        await repl.reset()

        assert repl.get_runtime_backend("selfref") is self_reference

    @pytest.mark.asyncio
    async def test_reset_does_not_delete_self_reference_memory(self):
        """reset_repl should not clear SelfReference history store."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history(
            "agent_main",
            [{"role": "user", "content": "remember me"}],
        )
        await repl.execute("x = 1")
        await repl.reset()

        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "remember me"}
        ]

    @pytest.mark.asyncio
    async def test_execute_can_fork_bound_agent_instance_with_memory_snapshot(self):
        """REPL runtime.selfref.fork.spawn should inherit memory as child context."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        observed_calls: list[dict[str, object]] = []

        async def fake_agent(message: str, history=None):
            observed_calls.append(
                {
                    "message": message,
                    "history": list(history or []),
                }
            )
            yield (
                f"forked:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "handle = runtime.selfref.fork.spawn('sub-task')\n"
            "results = runtime.selfref.fork.gather_all(handle)\n"
            "fork_result = results[handle['fork_id']]\n"
            "print(fork_result['source_memory_key'])\n"
            "print(fork_result['memory_key'])\n"
            "print(fork_result['response'])\n"
        )

        assert result["success"] is True
        assert "agent_main" in result["stdout"]
        assert "forked:sub-task" in result["stdout"]
        assert observed_calls[0]["history"] == [{"role": "user", "content": "seed"}]

        fork_keys = [
            key
            for key in self_reference.list_history_keys()
            if key.startswith("agent_main::fork::")
        ]
        assert len(fork_keys) == 1
        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "seed"}
        ]
        assert self_reference.snapshot_history(fork_keys[0]) == [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "child done"},
        ]

    @pytest.mark.asyncio
    async def test_execute_fork_spawn_uses_pre_fork_history_for_child(
        self,
    ):
        """Forking from an active assistant tool_call should hide the pending parent tool-call context from the child."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history(
            "agent_main",
            [
                {"role": "user", "content": "seed"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fork_1",
                            "type": "function",
                            "function": {
                                "name": "execute_code",
                                "arguments": "runtime.selfref.fork.spawn('sub-task')",
                            },
                        },
                        {
                            "id": "call_read_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "README.md"}',
                            },
                        },
                    ],
                },
            ],
        )

        observed_calls: list[dict[str, object]] = []

        async def fake_agent(message: str, history=None):
            observed_calls.append(
                {
                    "message": message,
                    "history": list(history or []),
                }
            )
            return (
                "forked",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "handle = runtime.selfref.fork.spawn('sub-task')\n"
            "results = runtime.selfref.fork.gather_all(handle, include_history=True)\n"
            "fork_result = results[handle['fork_id']]\n"
            "print(fork_result['history'])\n"
        )

        assert result["success"] is True, result["error"] or result["stderr"]
        assert observed_calls[0]["message"] == "sub-task"
        assert observed_calls[0]["history"] == [{"role": "user", "content": "seed"}]
        assert (
            result["stdout"].strip()
            == "[{'role': 'user', 'content': 'seed'}, {'role': 'assistant', 'content': 'child done'}]"
        )

    @pytest.mark.asyncio
    async def test_execute_can_spawn_and_gather_fork_from_code_act(self):
        """Code-act fork should be runtime-hooked and support spawn/gather APIs."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                {"task": message, "runtime_pid": os.getpid()},
                [
                    *(history or []),
                    {"role": "assistant", "content": f"done:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "spawned = runtime.selfref.fork.spawn('task-a')\n"
            "print(spawned['status'])\n"
            "results = runtime.selfref.fork.gather_all(spawned['fork_id'])\n"
            "final = results[spawned['fork_id']]\n"
            "print(final['status'])\n"
            "print(final['response']['runtime_pid'])\n"
        )

        assert result["success"] is True
        assert "running" in result["stdout"]
        assert "completed" in result["stdout"]
        assert str(os.getpid()) in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_gather_all_can_include_history_on_demand(self):
        """runtime.selfref.fork.gather_all should keep compact default and support include_history."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "spawned = runtime.selfref.fork.spawn('task-a')\n"
            "compact_results = runtime.selfref.fork.gather_all(spawned['fork_id'])\n"
            "compact = compact_results[spawned['fork_id']]\n"
            "print('history' in compact)\n"
            "print(compact['history_included'])\n"
            "hydrated_results = runtime.selfref.fork.gather_all(spawned['fork_id'], include_history=True)\n"
            "hydrated = hydrated_results[spawned['fork_id']]\n"
            "print('history' in hydrated)\n"
            "print(hydrated['history_included'])\n"
            "print(hydrated['history_count'])\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["False", "False", "True", "True", "2"]

    @pytest.mark.asyncio
    async def test_execute_can_gather_all_spawned_forks(self):
        """Code-act runtime.selfref.fork.gather_all should collect spawned forks."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                message,
                [
                    *(history or []),
                    {"role": "assistant", "content": f"done:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "handles = [\n"
            "    runtime.selfref.fork.spawn('task-a'),\n"
            "    runtime.selfref.fork.spawn('task-b'),\n"
            "]\n"
            "ids = [item['fork_id'] for item in handles]\n"
            "all_results = runtime.selfref.fork.gather_all(ids)\n"
            "all_handle_results = runtime.selfref.fork.gather_all(handles)\n"
            "print(len(all_results))\n"
            "print(sorted(all_results.keys()) == sorted(ids))\n"
            "print(all(v['status'] == 'completed' for v in all_results.values()))\n"
            "print(sorted(all_handle_results.keys()) == sorted(ids))\n"
        )

        assert result["success"] is True
        assert "2" in result["stdout"]
        assert "True" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_emits_fork_lifecycle_events(self):
        """Code-act fork should emit structured lifecycle custom events."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = ToolEventEmitter()

        result = await repl.execute(
            "handle = runtime.selfref.fork.spawn('task-a')\n"
            "_ = runtime.selfref.fork.gather_all(handle)\n",
            event_emitter=emitter,
        )

        assert result["success"] is True

        events = await emitter.get_events()
        event_names = [
            event.event.event_name
            for event in events
            if isinstance(event.event, CustomEvent)
        ]

        assert "selfref_fork_spawned" in event_names
        assert "selfref_fork_start" in event_names
        assert "selfref_fork_end" in event_names

    @pytest.mark.asyncio
    async def test_execute_does_not_emit_fork_stream_events(self):
        """Fork spawn/gather should not emit fork stream events (use lifecycle only)."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            yield (
                "child-a ",
                list(history or []),
            )
            await asyncio.sleep(0)
            yield (
                "child-b\n",
                list(history or []),
            )
            yield (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = ToolEventEmitter()

        result = await repl.execute(
            "handle = runtime.selfref.fork.spawn('task-a')\n"
            "results = runtime.selfref.fork.gather_all(handle)\n"
            "fork_result = results[handle['fork_id']]\n"
            "print(fork_result['status'])\n",
            event_emitter=emitter,
        )

        assert result["success"] is True
        assert "completed" in result["stdout"]

        events = await emitter.get_events()
        custom_events = [
            event.event for event in events if isinstance(event.event, CustomEvent)
        ]
        event_names = [event.event_name for event in custom_events]

        assert "selfref_fork_start" in event_names
        assert "selfref_fork_end" in event_names
        assert "selfref_fork_stream_open" not in event_names
        assert "selfref_fork_stream_delta" not in event_names
        assert "selfref_fork_stream_close" not in event_names


class TestPyReplRuntimePrimitives:
    """Test direct runtime primitive access inside execute_code."""

    def test_registered_primitive_specs_match_handler_signatures(self):
        """Declared primitive parameter kinds should match handler signatures."""
        from SimpleLLMFunc.builtin import PyRepl

        kind_map = {
            inspect.Parameter.POSITIONAL_ONLY: "positional_only",
            inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
            inspect.Parameter.KEYWORD_ONLY: "keyword_only",
            inspect.Parameter.VAR_POSITIONAL: "var_positional",
            inspect.Parameter.VAR_KEYWORD: "var_keyword",
        }

        repl = PyRepl()

        for primitive_name in [
            "runtime.list_primitive_specs",
            "selfref.context.inspect",
            "selfref.context.remember",
            "selfref.context.forget",
            "selfref.context.compact",
            "runtime.list_primitives",
            "selfref.fork.spawn",
            "selfref.fork.gather_all",
        ]:
            spec = repl.get_primitive_spec(primitive_name, format="dict")
            assert isinstance(spec, dict)
            handler = repl._primitive_registry._specs[primitive_name].handler
            signature = inspect.signature(handler)
            actual_params = []
            for param in list(signature.parameters.values())[1:]:
                name = param.name
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    name = f"*{name}"
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    name = f"**{name}"
                actual_params.append((name, kind_map[param.kind]))

            declared_params = [
                (item["name"], item["kind"]) for item in spec.get("parameters", [])
            ]
            assert actual_params == declared_params

    @pytest.mark.asyncio
    async def test_execute_exposes_runtime_list_primitives(self):
        """runtime.list_primitives should list baseline runtime primitives."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            "names = runtime.list_primitives()\n"
            "filtered = runtime.list_primitives(prefix='runtime.list_')\n"
            "fork_contains = runtime.list_primitives(contains='selfref.fork.')\n"
            "print('runtime.list_primitives' in names)\n"
            "print('runtime.list_backends' in names)\n"
            "print('selfref.context.inspect' in names)\n"
            "print(all(item.startswith('runtime.list_') for item in filtered))\n"
            "print('runtime.list_primitives' in filtered)\n"
            "print(all('selfref.fork.' in item for item in fork_contains))\n"
        )

        assert result["success"] is True
        stdout_lines = [
            line
            for line in result["stdout"].splitlines()
            if not line.startswith("After calling runtime.list_primitives")
        ]
        assert stdout_lines == [
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
        ]

    @pytest.mark.asyncio
    async def test_runtime_list_primitives_prints_next_steps(self):
        """runtime.list_primitives should emit next-step guidance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("runtime.list_primitives()")

        assert result["success"] is True
        assert "After calling runtime.list_primitives" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_exposes_runtime_list_primitive_specs(self):
        """runtime.list_primitive_specs should include structured contract metadata."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            "specs = runtime.list_primitive_specs(format='dict')\n"
            "core = next(item for item in specs if item.get('name') == 'runtime.list_primitive_specs')\n"
            "exact = runtime.get_primitive_spec('runtime.list_primitives', format='dict')\n"
            "filtered = runtime.list_primitive_specs(names=['runtime.list_primitives'], format='dict')\n"
            "prefixed = runtime.list_primitive_specs(prefix='runtime.list_', format='dict')\n"
            "name_spec = runtime.get_primitive_spec('runtime.list_primitives', format='dict')\n"
            "param_names = [item.get('name') for item in core.get('parameters', [])]\n"
            "name_param_names = [item.get('name') for item in name_spec.get('parameters', [])]\n"
            "print(isinstance(specs, list))\n"
            "print(any(item.get('name') == 'runtime.list_primitives' for item in specs))\n"
            "print(any(item.get('name') == 'runtime.list_primitive_specs' for item in specs))\n"
            "print(any(item.get('name') == 'runtime.get_primitive_spec' for item in specs))\n"
            "print(any(item.get('name') == 'runtime.list_backends' for item in specs))\n"
            "print(isinstance(core.get('input_type'), str))\n"
            "print(isinstance(core.get('output_type'), str))\n"
            "print(isinstance(core.get('output_parsing'), str))\n"
            "print(isinstance(core.get('parameters'), list))\n"
            "print(isinstance(core.get('best_practices'), list))\n"
            "print('structured specs' in str(core.get('description')).lower())\n"
            "print('iterate the list' in str(core.get('output_parsing')).lower())\n"
            "print('names' in param_names)\n"
            "print('prefix' in param_names)\n"
            "print('prefix' in name_param_names)\n"
            "print(len(filtered) == 1 and filtered[0].get('name') == 'runtime.list_primitives')\n"
            "print(all(str(item.get('name')).startswith('runtime.list_') for item in prefixed))\n"
            "print(exact.get('name') == 'runtime.list_primitives')\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == [
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
        ]

    @pytest.mark.asyncio
    async def test_runtime_primitive_param_error_includes_hint(self):
        """Primitive parameter errors should include parameter guidance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute("runtime.get_primitive_spec()")

        assert result["success"] is False
        assert isinstance(result["error"], str)
        assert "Parameter requirements" in result["error"]
        assert "name" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_runtime_spec_queries_support_xml_format(self):
        """runtime list/get primitive spec calls should default to XML output."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            "xml_list = runtime.list_primitive_specs(names=['runtime.list_primitives'])\n"
            "xml_one = runtime.get_primitive_spec('runtime.list_primitives')\n"
            "dict_list = runtime.list_primitive_specs(names=['runtime.list_primitives'], format='dict')\n"
            "dict_one = runtime.get_primitive_spec('runtime.list_primitives', format='dict')\n"
            "print(isinstance(xml_list, str))\n"
            "print(isinstance(xml_one, str))\n"
            "print(isinstance(dict_list, list))\n"
            "print(isinstance(dict_one, dict))\n"
            "print('<primitive_specs>' in xml_list)\n"
            "print('<primitive>' in xml_list)\n"
            "print('<name>runtime.list_primitives</name>' in xml_list)\n"
            "print('<primitive_spec>' in xml_one)\n"
            "print('<name>runtime.list_primitives</name>' in xml_one)\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == [
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
        ]

    @pytest.mark.asyncio
    async def test_execute_exposes_selfref_guide_and_best_practices(self):
        """Selfref pack should expose namespace guide with fork/memory best practices."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            "guide = runtime.selfref.guide()\n"
            "specs = runtime.list_primitive_specs(format='dict')\n"
            "guide_spec = next(item for item in specs if item.get('name') == 'selfref.guide')\n"
            "print('best_practices' in guide)\n"
            "print(len(guide.get('best_practices', [])) >= 5)\n"
            "guide_best_text = ' '.join(str(item) for item in guide.get('best_practices', []))\n"
            "print('status/response/result/memory_key/history_count' in guide_best_text)\n"
            "print('error_type/error_message before retrying' in guide_best_text)\n"
            "print(isinstance(guide_spec.get('parameters'), list))\n"
            "print(isinstance(guide_spec.get('best_practices'), list))\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == [
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
        ]

    @pytest.mark.asyncio
    async def test_execute_selfref_fork_spec_declares_result_contract_and_safe_read(
        self,
    ):
        """selfref fork specs should define compact contract and no-raw-print guidance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        result = await repl.execute(
            "specs = runtime.list_primitive_specs(format='dict')\n"
            "spawn_spec = next(item for item in specs if item.get('name') == 'selfref.fork.spawn')\n"
            "gather_all_spec = next(item for item in specs if item.get('name') == 'selfref.fork.gather_all')\n"
            "best_text = ' '.join(str(item) for item in spawn_spec.get('best_practices', []))\n"
            "print(\"status:'running'\" in str(spawn_spec.get('output_type')))\n"
            "spawn_params = [item.get('name') for item in spawn_spec.get('parameters', [])]\n"
            "print('message' in spawn_params)\n"
            "gather_output = str(gather_all_spec.get('output_type')).lower()\n"
            "gather_parse = str(gather_all_spec.get('output_parsing')).lower()\n"
            'print("status:\'completed\'" in gather_output and "error_message:str" in gather_output)\n'
            "print('check `status` first' in gather_parse)\n"
            "print('history_count' in gather_output)\n"
            "print('history_included' in gather_output)\n"
            "print('keyed by' in gather_output and 'fork_id' in gather_output)\n"
            "print('.items()' in gather_parse)\n"
            "print('read the fields you need from fork results' in best_text.lower())\n"
            "print('status/response/result/memory_key' in best_text)\n"
            "print('include_history=True' in best_text)\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == [
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
            "True",
        ]

    @pytest.mark.asyncio
    async def test_execute_can_mutate_context_via_runtime_primitive_calls(self):
        """runtime.selfref.context.* should proxy host self-reference operations."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])
        result = await repl.execute(
            "exp = runtime.selfref.context.remember('ok')\n"
            "snapshot = runtime.selfref.context.inspect()\n"
            "print(snapshot['active_key'])\n"
            "print(snapshot['experiences'][0]['text'])\n"
            "print(exp['id'].startswith('exp_'))\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["agent_main", "ok", "True"]
        assert "ok" in str(self_reference.snapshot_history("agent_main")[0]["content"])

    @pytest.mark.asyncio
    async def test_execute_runtime_context_compact_replaces_working_messages(self):
        """runtime.selfref.context.compact should rewrite context immediately."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])
        result = await repl.execute(
            "payload = runtime.selfref.context.compact(\n"
            "    goal='Goal A',\n"
            "    instruction='Instruction A',\n"
            "    discoveries=['Discovery A'],\n"
            "    completed=['Completed A'],\n"
            "    current_status='Status A',\n"
            "    likely_next_work=['Next A'],\n"
            "    relevant_files_directories=['src/a.py'],\n"
            ")\n"
            "print(payload['status'])\n"
            "print('## Goal' in payload['assistant_message'])\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["queued", "True"]

        committed = self_reference.snapshot_history("agent_main")
        assert committed[0]["role"] == "assistant"
        assert "Goal A" in committed[0]["content"]
        assert self_reference.commit_pending_compaction("agent_main") is None

    @pytest.mark.asyncio
    async def test_execute_context_remember_accepts_keyword_text(self):
        """runtime.selfref.context.remember should accept model-natural keyword usage."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        result = await repl.execute(
            "runtime.selfref.context.remember(text='Rule B')\n"
            "snapshot = runtime.selfref.context.inspect()\n"
            "print(snapshot['experiences'][0]['text'])\n"
        )

        assert result["success"] is True
        assert result["stdout"].strip() == "Rule B"
        assert "Rule B" in str(
            self_reference.snapshot_history("agent_main")[0]["content"]
        )

    @pytest.mark.asyncio
    async def test_execute_runtime_argument_error_stays_actionable(self):
        """Primitive argument errors should surface a direct fix, not unrelated history-shape noise."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        result = await repl.execute("runtime.selfref.context.remember()")

        assert result["success"] is False
        assert "Parameter requirements:" in str(result["error"])
        assert "text(str, required)" in str(result["error"])
        assert "Unmatched assistant tool_calls without tool results" not in str(
            result["error"]
        )

    @pytest.mark.asyncio
    async def test_execute_can_run_fork_via_runtime_primitive_calls(self):
        """runtime.selfref.fork.spawn should fork bound agent instance."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                f"runtime:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "handle = runtime.selfref.fork.spawn('sub-task')\n"
            "results = runtime.selfref.fork.gather_all(handle)\n"
            "fork_result = results[handle['fork_id']]\n"
            "print(fork_result['source_memory_key'])\n"
            "print(fork_result['response'])\n"
            "print(fork_result['result'])\n"
        )

        assert result["success"] is True
        assert "agent_main" in result["stdout"]
        assert "runtime:sub-task" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_gather_all_include_history_on_demand(self):
        """runtime.selfref.fork.gather_all should honor include_history per spec."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "handle = runtime.selfref.fork.spawn('task-a')\n"
            "results = runtime.selfref.fork.gather_all(handle, include_history=True)\n"
            "fork_result = results[handle['fork_id']]\n"
            "print('history' in fork_result)\n"
            "print(fork_result['history_included'])\n"
            "print(fork_result['history_count'])\n"
            "print(fork_result['history'][-1]['content'])\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["True", "True", "2", "child:task-a"]

    @pytest.mark.asyncio
    async def test_execute_spawn_rejects_include_history_argument(self):
        """runtime.selfref.fork.spawn should reject unsupported include_history."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return message, list(history or [])

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "runtime.selfref.fork.spawn('task-a', include_history=True)"
        )

        assert result["success"] is False
        assert "does not accept include_history" in str(result["error"])

    @pytest.mark.asyncio
    async def test_execute_gather_all_can_include_history_on_demand(self):
        """runtime.selfref.fork.gather_all should hydrate histories when requested."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        self_reference = _builtin_self_reference(repl)
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                message,
                [
                    *(history or []),
                    {"role": "assistant", "content": f"done:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        result = await repl.execute(
            "handles = [runtime.selfref.fork.spawn('task-a'), runtime.selfref.fork.spawn('task-b')]\n"
            "ids = [item['fork_id'] for item in handles]\n"
            "all_results = runtime.selfref.fork.gather_all(ids, include_history=True)\n"
            "print(len(all_results))\n"
            "print(sorted(all_results.keys()) == sorted(ids))\n"
            "print(all(v['history_included'] for v in all_results.values()))\n"
            "print(all(v['history_count'] == 2 for v in all_results.values()))\n"
            "print(all('history' in v for v in all_results.values()))\n"
        )

        assert result["success"] is True
        assert result["stdout"].splitlines() == ["2", "True", "True", "True", "True"]


class TestPyReplStreaming:
    """Test PyRepl streaming with event_emitter."""

    @pytest.mark.asyncio
    async def test_execute_with_event_emitter(self):
        """Test that event_emitter receives stdout events."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl()
        emitter = ToolEventEmitter()

        result = await repl.execute("print('hello')", event_emitter=emitter)

        assert result["success"] is True

        await asyncio.sleep(0.1)

        events = await emitter.get_events()
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_streaming_multiple_lines(self):
        """Test streaming with multiple print statements."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl()
        emitter = ToolEventEmitter()

        result = await repl.execute(
            "import time\nfor i in range(3):\n    print(f'line {i}')",
            event_emitter=emitter,
        )

        assert result["success"] is True

        await asyncio.sleep(0.1)

        events = await emitter.get_events()
        assert len(events) >= 3

    @pytest.mark.asyncio
    async def test_event_contains_correct_data(self):
        """Test that emitted events contain correct data."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
        from SimpleLLMFunc.hooks.events import CustomEvent

        repl = PyRepl()
        emitter = ToolEventEmitter()

        await repl.execute("print('test')", event_emitter=emitter)

        await asyncio.sleep(0.1)

        events = await emitter.get_events()

        stdout_events = []
        for event_yield in events:
            event = event_yield.event
            if isinstance(event, CustomEvent) and event.event_name == "kernel_stdout":
                stdout_events.append(event)
        assert len(stdout_events) > 0
        assert "test" in str(stdout_events[0].data)


class TestPyReplEventLoopSafety:
    """Test PyRepl does not block asyncio event loop."""

    @pytest.mark.asyncio
    async def test_execute_does_not_block_event_loop(self):
        """execute_code should not freeze the loop during long-running code."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        warmup = await repl.execute("0")
        assert warmup["success"] is True, warmup["error"] or warmup["stderr"]

        tick_count = 0
        running = True

        async def ticker() -> None:
            nonlocal tick_count
            while running:
                tick_count += 1
                await asyncio.sleep(0.01)

        ticker_task = asyncio.create_task(ticker())
        try:
            result = await repl.execute("import time\ntime.sleep(0.12)")
            assert result["success"] is True
        finally:
            running = False
            await ticker_task

        assert tick_count >= 3


class TestPyReplTimeout:
    """Test PyRepl timeout policy for execution and interactive input."""

    @pytest.mark.asyncio
    async def test_execute_timeout_is_configurable(self):
        """Execution should honor configured timeout duration."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(execution_timeout_seconds=0.2)
        result = await repl.execute("import time\ntime.sleep(0.5)")

        assert result["success"] is False
        assert result["error"] is not None
        assert "0.2 seconds" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_timeout_can_be_overridden_per_call(self):
        """Per-call timeout argument should override the REPL default timeout."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(execution_timeout_seconds=2.0)
        result = await repl.execute(
            "import time\ntime.sleep(0.5)",
            timeout_seconds=0.1,
        )

        assert result["success"] is False
        assert result["error"] is not None
        assert "0.1 seconds" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_rejects_non_positive_per_call_timeout(self):
        """Per-call timeout should reject non-positive values."""
        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl()
        with pytest.raises(ValueError, match="timeout_seconds"):
            await repl.execute("print('hello')", timeout_seconds=0)

    @pytest.mark.asyncio
    async def test_waiting_for_input_does_not_consume_timeout(self):
        """input() waiting time should be excluded from timeout budget."""
        import time
        from unittest.mock import patch

        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(execution_timeout_seconds=1.0, input_idle_timeout_seconds=10)

        def delayed_input(prompt: str = "") -> str:
            assert prompt == "Name: "
            time.sleep(1.3)
            return "Alice"

        with patch("builtins.input", side_effect=delayed_input):
            result = await repl.execute(
                "name = input('Name: ')\nprint(f'Hello, {name}!')"
            )

        assert result["success"] is True
        assert "Hello, Alice!" in result["stdout"]

    @pytest.mark.asyncio
    async def test_timeout_is_reset_after_each_input_submission(self):
        """Accepted input should reset execution timeout window."""
        import time
        from unittest.mock import patch

        from SimpleLLMFunc.builtin import PyRepl

        repl = PyRepl(execution_timeout_seconds=1.0, input_idle_timeout_seconds=10)

        prompts_seen: list[str] = []
        values = iter(["A", "B"])

        def delayed_input(prompt: str = "") -> str:
            prompts_seen.append(prompt)
            time.sleep(1.3)
            return next(values)

        with patch("builtins.input", side_effect=delayed_input):
            result = await repl.execute(
                """
first = input('First: ')
second = input('Second: ')
import time
time.sleep(0.4)
print(first + second)
"""
            )

        assert result["success"] is True
        assert prompts_seen == ["First: ", "Second: "]
        assert "AB" in result["stdout"]

    @pytest.mark.asyncio
    async def test_input_idle_timeout_is_enforced(self):
        """Tool-input requests should fail after configured idle timeout."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.events import CustomEvent
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl(execution_timeout_seconds=10, input_idle_timeout_seconds=0.2)
        emitter = ToolEventEmitter()

        run_task = asyncio.create_task(
            repl.execute("value = input('Value: ')", event_emitter=emitter)
        )

        result = await asyncio.wait_for(run_task, timeout=10)

        assert result["success"] is False
        assert result["error"] == "Input request timed out after 0.2 seconds"
        assert "Input request timed out after 0.2 seconds" in result["stderr"]

        request_id: str | None = None
        request_prompt: str | None = None
        for event_yield in await emitter.get_events():
            event = event_yield.event
            if not isinstance(event, CustomEvent):
                continue
            if event.event_name != "kernel_input_request":
                continue

            data = getattr(event, "data", None)
            if not isinstance(data, dict):
                continue

            maybe_id = data.get("request_id")
            maybe_prompt = data.get("prompt")
            if isinstance(maybe_id, str) and maybe_id:
                request_id = maybe_id
            if isinstance(maybe_prompt, str):
                request_prompt = maybe_prompt
            break

        if request_prompt is not None:
            assert request_prompt == "Value: "

        assert PyRepl.submit_input(request_id or "late-request-id", "late") is False


class TestPyReplInputHook:
    """Test PyRepl interactive input() bridge."""

    def test_submit_input_returns_false_for_unknown_request(self):
        """Submitting to an unknown request id should fail gracefully."""
        from SimpleLLMFunc.builtin import PyRepl

        assert PyRepl.submit_input("unknown-request", "value") is False

    @pytest.mark.asyncio
    async def test_execute_supports_input_roundtrip_via_events(self):
        """execute should emit input request and accept UI-provided response."""
        from SimpleLLMFunc.builtin import PyRepl
        from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

        repl = PyRepl()
        emitter = ToolEventEmitter()

        run_task = asyncio.create_task(
            repl.execute(
                "name = input('Name: ')\nprint(f'Hello, {name}!')",
                event_emitter=emitter,
            )
        )

        try:
            request_id, prompt = await _wait_for_input_request(emitter)
            assert prompt == "Name: "
            assert PyRepl.submit_input(request_id, "Alice") is True

            result = await asyncio.wait_for(run_task, timeout=5)
            assert result["success"] is True
            assert "Hello, Alice!" in result["stdout"]
        finally:
            if not run_task.done():
                run_task.cancel()
                with contextlib.suppress(Exception):
                    await run_task
            repl.close()
