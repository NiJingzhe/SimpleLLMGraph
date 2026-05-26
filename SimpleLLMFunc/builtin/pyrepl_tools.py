from __future__ import annotations

from typing import Any, Dict, List, Optional

from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter
from SimpleLLMFunc.tool import TOO_LONG_TO_FILE_MAX_TOKENS, Tool
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl


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

    artifacts = result.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        image_count = sum(
            1
            for artifact in artifacts
            if isinstance(artifact, dict) and artifact.get("type") == "image"
        )
        if image_count:
            suffix = "s" if image_count != 1 else ""
            lines.append(f"image artifacts: {image_count} image artifact{suffix}")

    return "\n".join(lines)


def _image_payload_from_artifact(artifact: Dict[str, Any]) -> ImgPath | ImgUrl | None:
    if artifact.get("type") != "image":
        return None

    detail = artifact.get("detail")
    if detail not in {"low", "high", "auto"}:
        detail = "auto"

    url = artifact.get("url")
    if isinstance(url, str) and url:
        try:
            return ImgUrl(url, detail=detail)
        except ValueError:
            return None

    path = artifact.get("path")
    if isinstance(path, str) and path:
        try:
            return ImgPath(path, detail=detail)
        except (FileNotFoundError, ValueError):
            return None

    return None


def build_execute_tool_return(result: Dict[str, Any]) -> str | tuple[str, List[ImgPath | ImgUrl]]:
    summary = format_execute_tool_output(result)
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        return summary

    images: List[ImgPath | ImgUrl] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        image = _image_payload_from_artifact(artifact)
        if image is not None:
            images.append(image)

    if not images:
        return summary

    return summary, images


async def execute_tool_adapter(
    repl: Any,
    code: str,
    timeout_seconds: Optional[float] = None,
    event_emitter: Optional[ToolEventEmitter] = None,
) -> str | tuple[str, List[ImgPath | ImgUrl]]:
    result = await repl.execute(
        code,
        timeout_seconds=timeout_seconds,
        event_emitter=event_emitter,
    )
    return build_execute_tool_return(result)


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
    "build_execute_tool_return",
    "execute_tool_adapter",
    "format_execute_tool_output",
]
