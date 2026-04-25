"""Specialized tool-call cards for builtin file tools."""

from __future__ import annotations

from typing import Any

from SimpleLLMFunc.utils.tui.tool_cards.base import ToolCallCard


def _render_fenced_block(content: str, language: str = "") -> str:
    normalized = content.rstrip("\n")
    fence = "```"
    if "```" in normalized:
        fence = "````"
    header = f"{fence}{language}" if language else fence
    return f"{header}\n{normalized}\n{fence}"


def _render_section(title: str, body: str) -> str:
    return f"## {title}\n{body}" if body else ""


def _format_line_range(start_line: Any, end_line: Any) -> str:
    if start_line is None and end_line is None:
        return ""
    if start_line is not None and end_line is not None:
        return f"`{start_line}-{end_line}`"
    if start_line is not None:
        return f"`from {start_line}`"
    return f"`through {end_line}`"


def _build_sed_diff_preview(pattern: str, replacement: str) -> str:
    lines = [
        "# Regex replacement preview",
        f"- /{pattern}/",
        f"+ {replacement}",
    ]
    return _render_fenced_block("\n".join(lines), "diff")


class FileToolCallCard(ToolCallCard):
    """Shared base class for file-oriented builtin tools."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_class("file-tool-call")

    def _path_section(self) -> str:
        path = self.arguments.get("path")
        if not isinstance(path, str) or not path:
            return ""
        return _render_section("File", f"`{path}`")

    def _line_range_section(self) -> str:
        line_range = _format_line_range(
            self.arguments.get("start_line"),
            self.arguments.get("end_line"),
        )
        if not line_range:
            return ""
        return _render_section("Line Range", line_range)

    def _render_argument_block(self, key: str, value: Any) -> str:
        return self.render_single_argument_markdown(key, value)

    def _render_remaining_arguments(self, excluded_keys: set[str]) -> str:
        lines: list[str] = []
        for key, value in self.arguments.items():
            if key in excluded_keys:
                continue
            lines.append(self._render_argument_block(key, value))
        return "\n".join(line for line in lines if line)

    def refresh_arguments_markdown(self) -> None:
        sections = [self._path_section(), self._line_range_section()]
        remaining = self._render_remaining_arguments(
            excluded_keys={"path", "start_line", "end_line"}
        )
        if remaining:
            sections.append(_render_section("Arguments", remaining))
        self.arguments_markdown = "\n\n".join(
            section for section in sections if section
        )
        if not self.arguments_markdown.strip():
            self.arguments_markdown = "_No arguments_"
        self.last_argument_changed = None


class ReadFileToolCallCard(FileToolCallCard):
    """Card for read_file."""


class ReadImageToolCallCard(FileToolCallCard):
    """Card for read_image."""


class GrepToolCallCard(FileToolCallCard):
    """Card for grep with explicit pattern/scope sections."""

    def refresh_arguments_markdown(self) -> None:
        sections = []
        pattern = self.arguments.get("pattern")
        if isinstance(pattern, str) and pattern:
            sections.append(
                _render_section(
                    "Content Pattern", _render_fenced_block(pattern, "text")
                )
            )
        path_pattern = self.arguments.get("path_pattern")
        if isinstance(path_pattern, str) and path_pattern:
            sections.append(
                _render_section(
                    "Path Scope", _render_fenced_block(path_pattern, "text")
                )
            )
        remaining = self._render_remaining_arguments(
            excluded_keys={"pattern", "path_pattern"}
        )
        if remaining:
            sections.append(_render_section("Arguments", remaining))
        self.arguments_markdown = "\n\n".join(
            section for section in sections if section
        )
        if not self.arguments_markdown.strip():
            self.arguments_markdown = "_No arguments_"
        self.last_argument_changed = None


class SedToolCallCard(FileToolCallCard):
    """Card for sed with regex and replacement emphasis."""

    def refresh_arguments_markdown(self) -> None:
        sections = [self._path_section(), self._line_range_section()]
        pattern = self.arguments.get("pattern_to_be_replaced")
        replacement = self.arguments.get("new_string")
        if isinstance(pattern, str) and pattern and isinstance(replacement, str):
            sections.append(
                _render_section(
                    "Edit Preview",
                    _build_sed_diff_preview(pattern, replacement),
                )
            )
        else:
            if isinstance(pattern, str) and pattern:
                sections.append(
                    _render_section(
                        "Regex Pattern",
                        _render_fenced_block(pattern, "text"),
                    )
                )
            if isinstance(replacement, str):
                sections.append(
                    _render_section(
                        "Replacement",
                        _render_fenced_block(replacement, "text"),
                    )
                )
        remaining = self._render_remaining_arguments(
            excluded_keys={
                "path",
                "start_line",
                "end_line",
                "pattern_to_be_replaced",
                "new_string",
            }
        )
        if remaining:
            sections.append(_render_section("Arguments", remaining))
        self.arguments_markdown = "\n\n".join(
            section for section in sections if section
        )
        if not self.arguments_markdown.strip():
            self.arguments_markdown = "_No arguments_"
        self.last_argument_changed = None


class EchoIntoToolCallCard(FileToolCallCard):
    """Card for echo_into with full-file content emphasis."""

    def refresh_arguments_markdown(self) -> None:
        sections = [self._path_section()]
        content = self.arguments.get("content")
        if isinstance(content, str):
            sections.append(
                _render_section("Content", _render_fenced_block(content, "text"))
            )
        remaining = self._render_remaining_arguments(excluded_keys={"path", "content"})
        if remaining:
            sections.append(_render_section("Arguments", remaining))
        self.arguments_markdown = "\n\n".join(
            section for section in sections if section
        )
        if not self.arguments_markdown.strip():
            self.arguments_markdown = "_No arguments_"
        self.last_argument_changed = None


__all__ = [
    "FileToolCallCard",
    "ReadFileToolCallCard",
    "ReadImageToolCallCard",
    "GrepToolCallCard",
    "SedToolCallCard",
    "EchoIntoToolCallCard",
]
