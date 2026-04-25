"""Base tool-call card widgets for the Textual TUI."""

from __future__ import annotations

from typing import Any, Callable, Optional

from rich.markdown import Markdown
from textual.app import ComposeResult
from textual.containers import VerticalGroup
from textual.widgets import Static

from SimpleLLMFunc.utils.tui.formatters import format_tool_arguments_markdown

ToolArgumentFormatter = Callable[[dict[str, Any], Optional[str]], str]


class ToolCallCard(VerticalGroup):
    """Base class for tool-call cards rendered under one model bubble."""

    def __init__(
        self,
        *,
        tool_call_id: str,
        model_call_id: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        argument_formatter: Optional[ToolArgumentFormatter] = None,
    ) -> None:
        super().__init__(classes="tool-call")
        self.tool_call_id = tool_call_id
        self.model_call_id = model_call_id
        self.tool_name = tool_name
        self.arguments: dict[str, Any] = dict(arguments or {})
        self.arguments_markdown = ""
        self.argument_markdown_cache: dict[str, str] = {}
        self.argument_order: list[str] = []
        self.last_argument_changed: Optional[str] = None
        self.arguments_tool_name = tool_name
        self.output = ""
        self.status = "running"
        self.result_markdown = ""
        self.stats_line = ""
        self.render_pending = False
        self.title_dirty = False
        self.arguments_dirty = False
        self.output_dirty = False
        self.status_dirty = False
        self.result_dirty = False
        self.stats_dirty = False
        self._argument_formatter = argument_formatter or format_tool_arguments_markdown

        self.title_widget = Static(f"Tool: {tool_name}", classes="role tool-title")
        self.arguments_widget = Static(classes="tool-arguments")
        self.status_widget = Static(self.status, classes="tool-status")
        self.output_widget = Static(classes="tool-output")
        self.result_widget = Static(classes="tool-result")
        self.stats_widget = Static("", classes="stats tool-stats")

        self.refresh_arguments_markdown()
        self.arguments_widget.update(Markdown(self.arguments_markdown))

    def compose(self) -> ComposeResult:
        yield self.title_widget
        yield self.arguments_widget
        yield self.status_widget
        yield self.output_widget
        yield self.result_widget
        yield self.stats_widget

    def set_tool_name(self, tool_name: str) -> None:
        if tool_name == self.tool_name:
            return
        self.tool_name = tool_name
        self.title_dirty = True
        self.arguments_dirty = True

    def set_arguments(self, arguments: dict[str, Any]) -> None:
        self.arguments = dict(arguments)
        self.last_argument_changed = None
        self.arguments_dirty = True

    def append_argument(self, argname: str, argcontent_delta: str) -> None:
        previous = self.arguments.get(argname, "")
        if not isinstance(previous, str):
            previous = str(previous)
        self.arguments[argname] = previous + argcontent_delta
        self.last_argument_changed = argname
        self.arguments_dirty = True

    def append_output(self, output_delta: str) -> None:
        self.output += output_delta
        self.output_dirty = True

    def set_status(self, status: str) -> None:
        self.status = status
        self.status_dirty = True

    def set_result_markdown(self, result_markdown: str) -> None:
        self.result_markdown = result_markdown or ""
        self.result_dirty = True

    def set_stats(self, stats_line: str) -> None:
        self.stats_line = stats_line
        self.stats_dirty = True

    def render_single_argument_markdown(self, key: str, value: Any) -> str:
        return self._argument_formatter({key: value}, self.tool_name)

    def refresh_arguments_markdown(self) -> None:
        if not self.arguments:
            self.argument_order = []
            self.argument_markdown_cache = {}
            self.arguments_markdown = "_No arguments_"
            self.last_argument_changed = None
            self.arguments_tool_name = self.tool_name
            return

        tool_name_changed = self.arguments_tool_name != self.tool_name
        if tool_name_changed:
            self.argument_markdown_cache = {}
            self.argument_order = []
        self.arguments_tool_name = self.tool_name

        changed_key = self.last_argument_changed
        if (
            changed_key is None
            or tool_name_changed
            or changed_key not in self.arguments
        ):
            self.argument_order = list(self.arguments.keys())
            self.argument_markdown_cache = {
                key: self.render_single_argument_markdown(key, self.arguments[key])
                for key in self.argument_order
            }
        else:
            if changed_key not in self.argument_order:
                self.argument_order.append(changed_key)
            self.argument_markdown_cache[changed_key] = (
                self.render_single_argument_markdown(
                    changed_key,
                    self.arguments[changed_key],
                )
            )
            missing_keys = [
                key
                for key in self.argument_order
                if key not in self.argument_markdown_cache
            ]
            if missing_keys:
                self.argument_order = list(self.arguments.keys())
                self.argument_markdown_cache = {
                    key: self.render_single_argument_markdown(key, self.arguments[key])
                    for key in self.argument_order
                }

        self.arguments_markdown = "\n".join(
            self.argument_markdown_cache[key]
            for key in self.argument_order
            if key in self.argument_markdown_cache
        )
        if not self.arguments_markdown.strip():
            self.arguments_markdown = "_No arguments_"
        self.last_argument_changed = None

    def build_output_markdown(self) -> str:
        if not self.output:
            return ""
        return f"```text\n{self.output}\n```"

    def build_result_markdown(self) -> str:
        return self.result_markdown

    def flush_render(self) -> bool:
        updated = False

        if self.title_dirty:
            self.title_dirty = False
            self.title_widget.update(f"Tool: {self.tool_name}")
            updated = True

        if self.arguments_dirty:
            self.arguments_dirty = False
            self.refresh_arguments_markdown()
            self.arguments_widget.update(Markdown(self.arguments_markdown))
            updated = True

        if self.output_dirty:
            self.output_dirty = False
            output_markdown = self.build_output_markdown()
            if output_markdown:
                self.output_widget.update(Markdown(output_markdown))
            else:
                self.output_widget.update("")
            updated = True

        if self.status_dirty:
            self.status_dirty = False
            self.status_widget.update(self.status)
            updated = True

        if self.result_dirty:
            self.result_dirty = False
            result_markdown = self.build_result_markdown()
            if result_markdown:
                self.result_widget.update(Markdown(result_markdown))
            else:
                self.result_widget.update("")
            updated = True

        if self.stats_dirty:
            self.stats_dirty = False
            self.stats_widget.update(self.stats_line)
            updated = True

        return updated


__all__ = ["ToolCallCard", "ToolArgumentFormatter"]
