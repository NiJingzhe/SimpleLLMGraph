from __future__ import annotations

from typing import Any, Dict, List, Optional

from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.tool import TOO_LONG_TO_FILE_MAX_TOKENS, Tool


EXECUTE_TOOL_DESCRIPTION = (
    "Run Python code in a persistent REPL session (state persists across "
    "calls). Write direct executable snippets for the active REPL session. "
    "Use top-level executable code. Interactive "
    "`input()` is supported. Use `timeout_seconds` to control per-call "
    "timeout (default 600)."
    f"By the way, if a code snippet produces output more than {TOO_LONG_TO_FILE_MAX_TOKENS} tokens, "
    f"we truncate the tool result to no more than {TOO_LONG_TO_FILE_MAX_TOKENS} tokens, "
    "store the full result in a temporary file, and tell you its path."
)
RESET_TOOL_DESCRIPTION = (
    "Reset REPL runtime variables in the current session while preserving "
    "registered runtime primitive backends."
)
EXECUTE_TOOL_BEST_PRACTICES = [
    "Primitive = host-registered callable; use runtime.namespace.name(...). Use contains='<namespace>.' for namespace discovery.",
    "Runtime primitives are not standalone tool calls; call them inside execute_code as runtime.namespace.name(...).",
    "Spec lookups return XML by default; use format='dict' for direct field access in code.",
    "Inspect the contracts that support the current step and keep prompt context focused on the selected primitives.",
]
RESET_TOOL_BEST_PRACTICES = [
    "Use reset_repl for REPL variable cleanup while continuing with the same runtime backend state.",
    "Continue the next execution step from a fresh REPL variable namespace after reset.",
]


def format_execute_tool_output(result: Dict[str, Any]) -> str:
    success = bool(result.get("success"))
    execution_time_ms = result.get("execution_time_ms")
    if isinstance(execution_time_ms, (int, float)):
        duration_text = f"{execution_time_ms:.0f} ms"
    else:
        duration_text = "unknown duration"

    status = "succeeded" if success else "failed"
    lines = [f"Execution {status} in {duration_text}."]

    stdout = result.get("stdout")
    if isinstance(stdout, str) and stdout:
        lines.append("stdout:\n" + stdout)
    else:
        lines.append("stdout: (empty)")

    stderr = result.get("stderr")
    if isinstance(stderr, str) and stderr:
        lines.append("stderr:\n" + stderr)
    else:
        lines.append("stderr: (empty)")

    return_value = result.get("return_value")
    if isinstance(return_value, str) and return_value:
        lines.append(f"return_value: {return_value}")
    else:
        lines.append("return_value: (none)")

    error = result.get("error")
    if isinstance(error, str) and error:
        lines.append(f"error: {error}")

    error_details = result.get("error_details")
    if isinstance(error_details, dict):
        summary = error_details.get("summary")
        if isinstance(summary, str) and summary:
            lines.append(f"error_summary: {summary}")

    return "\n".join(lines)


async def execute_tool_adapter(
    repl: Any,
    code: str,
    timeout_seconds: Optional[float] = None,
    event_emitter: Optional[ToolEventEmitter] = None,
) -> str:
    result = await repl.execute(
        code,
        timeout_seconds=timeout_seconds,
        event_emitter=event_emitter,
    )
    return format_execute_tool_output(result)


def create_pyrepl_tools(repl: Any) -> List[Tool]:
    return [
        Tool(
            name="execute_code",
            description=EXECUTE_TOOL_DESCRIPTION,
            func=repl._execute_tool,
            best_practices=EXECUTE_TOOL_BEST_PRACTICES,
            prompt_injection_builder=repl._build_execute_tool_prompt_injection,
            too_long_to_file=True,
        ),
        Tool(
            name="reset_repl",
            description=RESET_TOOL_DESCRIPTION,
            func=repl.reset,
            best_practices=RESET_TOOL_BEST_PRACTICES,
        ),
    ]


__all__ = [
    "EXECUTE_TOOL_BEST_PRACTICES",
    "EXECUTE_TOOL_DESCRIPTION",
    "RESET_TOOL_BEST_PRACTICES",
    "RESET_TOOL_DESCRIPTION",
    "create_pyrepl_tools",
    "execute_tool_adapter",
    "format_execute_tool_output",
]
