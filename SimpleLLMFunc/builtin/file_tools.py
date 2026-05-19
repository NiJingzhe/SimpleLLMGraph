from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from SimpleLLMFunc.tool import Tool
from SimpleLLMFunc.type.multimodal import ImgPath


STALE_FILE_MESSAGE = (
    "file has been changed since last read, you should read it now first "
    "before continue to do any modifications."
)


LANGUAGE_MAP = {
    ".py": "python",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".sh": "bash",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".txt": "text",
}


class FileToolset:
    """Builtin file tools with per-instance hash tracking."""

    READ_FILE_DESCRIPTION = (
        "Read a file with optional line range; each output line is prefixed with "
        "`<lineno> |` and the response keeps original formatting."
    )
    READ_IMAGE_DESCRIPTION = (
        "Read one local image file from the workspace and return it as multimodal "
        "image content with `detail=\"high\"`."
    )
    GREP_DESCRIPTION = (
        "Regex search over workspace files (non-hidden, non-binary). "
        "Requires a path regex to scope the search."
    )
    SED_DESCRIPTION = (
        "Regex replace within a file and line range. Edits are rejected if the "
        "file has changed since the last read."
    )
    ECHO_DESCRIPTION = (
        "Overwrite an entire file with provided content. Edits are rejected if the "
        "file has changed since the last read."
    )

    READ_FILE_BEST_PRACTICES = [
        "Use start_line/end_line to limit context; line numbers are 1-based.",
        "Read the file before any modification to refresh the hash state.",
        "Hidden files or folders are not accessible.",
    ]
    READ_IMAGE_BEST_PRACTICES = [
        "Use this when the model needs to inspect a workspace-local image.",
        "The returned image is always sent with detail='high'.",
        "Hidden files or folders are not accessible.",
    ]
    GREP_BEST_PRACTICES = [
        "Always provide path_pattern to scope the search.",
        "Hidden files and binary files are skipped.",
        "Use read_file for detailed context around matches.",
    ]
    SED_BEST_PRACTICES = [
        "pattern_to_be_replaced is a regex string.",
        "Use start_line/end_line to bound replacements.",
        "Read the file first to avoid stale-hash edits.",
    ]
    ECHO_BEST_PRACTICES = [
        "Use only for full-file replacement.",
        "Parent directory must exist; create it first.",
        "Read the file before overwriting to avoid stale-hash edits.",
    ]

    def __init__(self, workspace: Path | str) -> None:
        base = Path(workspace).expanduser().resolve()
        if not base.exists() or not base.is_dir():
            raise ValueError("workspace must be an existing directory")
        self.workspace = base
        self._file_hashes: Dict[Path, str] = {}
        self._tools: Optional[List[Tool]] = None

    @property
    def toolset(self) -> List[Tool]:
        if self._tools is None:
            self._tools = self._create_tools()
        return self._tools

    def _create_tools(self) -> List[Tool]:
        prompt_injection_builder = self._build_workspace_prompt_injection
        return [
            Tool(
                name="read_file",
                description=self.READ_FILE_DESCRIPTION,
                func=self.read_file,
                best_practices=self.READ_FILE_BEST_PRACTICES,
                prompt_injection_builder=prompt_injection_builder,
            ),
            Tool(
                name="read_image",
                description=self.READ_IMAGE_DESCRIPTION,
                func=self.read_image,
                best_practices=self.READ_IMAGE_BEST_PRACTICES,
                prompt_injection_builder=prompt_injection_builder,
            ),
            Tool(
                name="grep",
                description=self.GREP_DESCRIPTION,
                func=self.grep,
                best_practices=self.GREP_BEST_PRACTICES,
                prompt_injection_builder=prompt_injection_builder,
            ),
            Tool(
                name="sed",
                description=self.SED_DESCRIPTION,
                func=self.sed,
                best_practices=self.SED_BEST_PRACTICES,
                prompt_injection_builder=prompt_injection_builder,
            ),
            Tool(
                name="echo_into",
                description=self.ECHO_DESCRIPTION,
                func=self.echo_into,
                best_practices=self.ECHO_BEST_PRACTICES,
                prompt_injection_builder=prompt_injection_builder,
            ),
        ]

    def _build_workspace_prompt_injection(self, context: Dict[str, Any]) -> str:
        return (
            "File tool workspace root: "
            + self.workspace.as_posix()
            + ". Paths must stay within this workspace; hidden files are not accessible."
        )

    def _resolve_path(self, path: str) -> Tuple[Optional[Path], Optional[str]]:
        if not isinstance(path, str) or not path.strip():
            return None, "path must be a non-empty string"

        raw = Path(path)
        resolved = (self.workspace / raw).resolve()
        try:
            relative = resolved.relative_to(self.workspace)
        except ValueError:
            return None, "path must be within workspace"

        for part in relative.parts:
            if part.startswith("."):
                return None, "hidden files are not allowed"

        return resolved, None

    def _display_path(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.workspace)
            return relative.as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def _read_text(self, path: Path) -> str:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read()

    @staticmethod
    def _is_binary(data: bytes) -> bool:
        return b"\x00" in data

    def _ensure_hash_matches(self, path: Path) -> Optional[str]:
        if not path.exists():
            return "file does not exist"
        current_hash = self._compute_hash(self._read_bytes(path))
        recorded = self._file_hashes.get(path)
        if recorded is None or recorded != current_hash:
            return STALE_FILE_MESSAGE
        return None

    def _update_hash(self, path: Path, data: bytes) -> None:
        self._file_hashes[path] = self._compute_hash(data)

    def _format_language(self, path: Path) -> str:
        return LANGUAGE_MAP.get(path.suffix.lower(), "text")

    def _validate_line_number(self, value: Optional[int], name: str) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, int):
            return f"{name} must be an integer"
        if value < 1:
            return f"{name} must be >= 1"
        return None

    @staticmethod
    def _is_full_wildcard_regex(pattern: str) -> bool:
        normalized = pattern.strip()
        if normalized.startswith("(?s)"):
            normalized = normalized[4:]
        while normalized.startswith("^"):
            normalized = normalized[1:]
        while normalized.endswith("$"):
            normalized = normalized[:-1]
        return normalized in {".", ".*", ".+"}

    async def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        resolved, error = self._resolve_path(path)
        if error:
            return error
        if resolved is None:
            return "invalid path"
        if not resolved.exists() or not resolved.is_file():
            return "file does not exist"

        start_error = self._validate_line_number(start_line, "start_line")
        if start_error:
            return start_error
        end_error = self._validate_line_number(end_line, "end_line")
        if end_error:
            return end_error

        warning = ""
        if start_line is not None and end_line is not None and end_line < start_line:
            warning = (
                "Note: start_line must be <= end_line; "
                "please check the range next time.\n"
            )

        data = self._read_bytes(resolved)
        self._update_hash(resolved, data)
        content = data.decode("utf-8", errors="replace")
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        if total_lines == 0:
            start_lineno = 1
            end_lineno = 0
            language = self._format_language(resolved)
            display = self._display_path(resolved)
            return (
                warning + f"This is file {display} from line {start_lineno} to line "
                f"{end_lineno}, we have shown the line number as `<lineno> |` "
                "at the beginning of each line:\n"
                f"```{language}\n```"
            )

        start_lineno = start_line if start_line is not None else 1
        end_lineno = end_line if end_line is not None else total_lines
        start_lineno = max(1, min(start_lineno, total_lines))
        end_lineno = max(1, min(end_lineno, total_lines))
        if end_lineno < start_lineno:
            return (
                "end_line must be >= start_line; please swap them or adjust the range."
            )

        snippet = []
        for index, line in enumerate(lines[start_lineno - 1 : end_lineno], start=0):
            lineno = start_lineno + index
            snippet.append(f"{lineno} | {line}")

        language = self._format_language(resolved)
        display = self._display_path(resolved)
        body = "".join(snippet)
        return (
            warning + f"This is file {display} from line {start_lineno} to line "
            f"{end_lineno}, we have shown the line number as `<lineno> |` "
            "at the beginning of each line:\n"
            f"```{language}\n{body}```"
        )

    async def read_image(self, path: str) -> ImgPath | str:
        resolved, error = self._resolve_path(path)
        if error:
            return error
        if resolved is None:
            return "invalid path"
        if not resolved.exists() or not resolved.is_file():
            return "file does not exist"

        try:
            image = ImgPath(resolved, detail="high")
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)

        self._update_hash(resolved, self._read_bytes(resolved))
        return image

    async def grep(self, pattern: str, path_pattern: str) -> str:
        if not isinstance(pattern, str) or not pattern.strip():
            return "pattern must be a non-empty string"
        if not isinstance(path_pattern, str) or not path_pattern.strip():
            return (
                "path_pattern is required; please provide a regex to scope the search"
            )

        try:
            content_regex = re.compile(pattern)
        except re.error as exc:
            return f"invalid pattern regex: {exc}"

        try:
            path_regex = re.compile(path_pattern)
        except re.error as exc:
            return f"invalid path_pattern regex: {exc}"

        if self._is_full_wildcard_regex(pattern):
            return (
                "pattern cannot be a full wildcard regex such as '.', '.*', or '.+'; "
                "use a more specific content pattern instead."
            )
        if self._is_full_wildcard_regex(path_pattern):
            return (
                "path_pattern cannot be a full wildcard regex such as '.', '.*', or '.+'; "
                "use a scoped path regex such as '.*\\.py$'."
            )

        matches: List[str] = []

        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in files:
                if filename.startswith("."):
                    continue
                file_path = Path(root) / filename
                try:
                    relative = file_path.relative_to(self.workspace).as_posix()
                except ValueError:
                    continue
                if path_regex.search(relative) is None:
                    continue

                try:
                    data = self._read_bytes(file_path)
                except Exception:
                    continue
                if self._is_binary(data):
                    continue
                text = data.decode("utf-8", errors="replace")
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if content_regex.search(line) is None:
                        continue
                    matches.append(f"{relative}:{line_number} | {line}")

        recommendation = (
            (
                "You are recommended to use read_file to check more detailed context "
                "by reading about 10 lines around these positions."
            )
            if matches
            else ("Nothing matched provided pattern.")
        )

        if matches:
            return "\n".join(matches) + "\n" + recommendation
        return recommendation

    @staticmethod
    def _line_number_from_offset(text: str, offset: int) -> int:
        return text.count("\n", 0, offset) + 1

    async def sed(
        self,
        path: str,
        start_line: Optional[int],
        end_line: Optional[int],
        pattern_to_be_replaced: str,
        new_string: str,
    ) -> str:
        resolved, error = self._resolve_path(path)
        if error:
            return error
        if resolved is None:
            return "invalid path"
        if not resolved.exists() or not resolved.is_file():
            return "file does not exist"

        start_error = self._validate_line_number(start_line, "start_line")
        if start_error:
            return start_error
        end_error = self._validate_line_number(end_line, "end_line")
        if end_error:
            return end_error

        if not isinstance(pattern_to_be_replaced, str) or not pattern_to_be_replaced:
            return "pattern_to_be_replaced must be a non-empty string"
        if not isinstance(new_string, str):
            return "new_string must be a string"

        stale_error = self._ensure_hash_matches(resolved)
        if stale_error:
            return stale_error

        try:
            regex = re.compile(pattern_to_be_replaced)
        except re.error as exc:
            return f"invalid pattern_to_be_replaced regex: {exc}"

        content = self._read_text(resolved)
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        if total_lines == 0:
            return "file is empty"

        start_lineno = start_line if start_line is not None else 1
        end_lineno = end_line if end_line is not None else total_lines
        start_lineno = max(1, min(start_lineno, total_lines))
        end_lineno = max(1, min(end_lineno, total_lines))
        if end_lineno < start_lineno:
            return (
                "end_line must be >= start_line; please swap them or adjust the range."
            )

        start_index = start_lineno - 1
        end_index = end_lineno
        segment = "".join(lines[start_index:end_index])
        matches = list(regex.finditer(segment))
        if not matches:
            return (
                "pattern_to_be_replaced was not found within the specified line "
                "range; please adjust the pattern or range."
            )

        min_line = min(
            self._line_number_from_offset(segment, match.start()) for match in matches
        )
        max_line = max(
            self._line_number_from_offset(segment, match.end()) for match in matches
        )

        replaced_segment = regex.sub(new_string, segment)
        new_segment_lines = replaced_segment.splitlines(keepends=True)
        new_lines = lines[:start_index] + new_segment_lines + lines[end_index:]
        new_content = "".join(new_lines)
        with resolved.open("w", encoding="utf-8", newline="") as handle:
            handle.write(new_content)
        self._update_hash(resolved, new_content.encode("utf-8"))

        absolute_start = start_lineno + min_line - 1
        absolute_end = start_lineno + max_line - 1
        snippet_start = max(1, absolute_start - 3)
        snippet_end = min(len(new_lines), absolute_end + 3)
        snippet = "".join(new_lines[snippet_start - 1 : snippet_end])
        return f"after edition: we have:\n{snippet}"

    async def echo_into(self, path: str, content: str) -> str:
        resolved, error = self._resolve_path(path)
        if error:
            return error
        if resolved is None:
            return "invalid path"
        if not isinstance(content, str):
            return "content must be a string"

        parent = resolved.parent
        if not parent.exists():
            return "parent directory does not exist; please create the folder first"
        if resolved.exists():
            stale_error = self._ensure_hash_matches(resolved)
            if stale_error:
                return stale_error
        if resolved.exists() and not resolved.is_file():
            return "path is not a file"

        with resolved.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        self._update_hash(resolved, content.encode("utf-8"))
        display = self._display_path(resolved)
        return f"write success: {display}"


__all__ = ["FileToolset"]
