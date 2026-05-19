"""Tests for FileToolset builtin tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from SimpleLLMFunc.builtin.file_tools import FileToolset, STALE_FILE_MESSAGE
from SimpleLLMFunc.type.multimodal import ImgPath


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_workspace_must_be_existing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError):
        FileToolset(missing)

    file_path = tmp_path / "file.txt"
    _write(file_path, "hello")
    with pytest.raises(ValueError):
        FileToolset(file_path)


def test_toolset_exposes_expected_tools(tmp_path: Path) -> None:
    toolset = FileToolset(tmp_path).toolset
    names = [tool.name for tool in toolset]
    assert names == ["read_file", "read_image", "grep", "sed", "echo_into"]


def test_toolset_injects_workspace_prompt(tmp_path: Path) -> None:
    toolset = FileToolset(tmp_path).toolset
    workspace_text = tmp_path.resolve().as_posix()

    for tool in toolset:
        prompt = tool.build_system_prompt_injection()
        assert isinstance(prompt, str)
        assert workspace_text in prompt
        assert "workspace" in prompt


@pytest.mark.asyncio
async def test_read_file_formats_output_and_tracks_hash(tmp_path: Path) -> None:
    path = tmp_path / "script.py"
    _write(path, "print('a')\nprint('b')\n")

    toolset = FileToolset(tmp_path)
    output = await toolset.read_file("script.py", start_line=1, end_line=1)

    assert "This is file script.py from line 1 to line 1" in output
    assert "```python" in output
    assert "1 | print('a')" in output
    assert path in toolset._file_hashes


@pytest.mark.asyncio
async def test_read_file_rejects_hidden_and_outside_paths(tmp_path: Path) -> None:
    hidden = tmp_path / ".secret"
    _write(hidden, "secret")

    toolset = FileToolset(tmp_path)
    assert await toolset.read_file(".secret") == "hidden files are not allowed"

    outside = tmp_path.parent / "outside.txt"
    _write(outside, "outside")
    assert await toolset.read_file(str(outside)) == "path must be within workspace"


@pytest.mark.asyncio
async def test_read_file_validates_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    _write(path, "one\nTwo\n")

    toolset = FileToolset(tmp_path)
    assert (
        await toolset.read_file("notes.txt", start_line=2, end_line=1)
        == "end_line must be >= start_line; please swap them or adjust the range."
    )
    assert (
        await toolset.read_file("notes.txt", start_line=0) == "start_line must be >= 1"
    )
    assert await toolset.read_file("notes.txt", end_line=0) == "end_line must be >= 1"
    assert (
        await toolset.read_file("notes.txt", start_line=cast(Any, "1"))
        == "start_line must be an integer"
    )

    _write(path, "one\n")
    output = await toolset.read_file("notes.txt", start_line=2, end_line=1)
    assert "start_line must be <= end_line" in output


@pytest.mark.asyncio
async def test_read_file_handles_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    _write(path, "")

    toolset = FileToolset(tmp_path)
    output = await toolset.read_file("empty.txt")

    assert "from line 1 to line 0" in output
    assert "```text\n```" in output


@pytest.mark.asyncio
async def test_read_image_returns_high_detail_img_path(tmp_path: Path) -> None:
    path = tmp_path / "chart.png"
    path.write_bytes(b"")

    toolset = FileToolset(tmp_path)
    result = await toolset.read_image("chart.png")

    assert isinstance(result, ImgPath)
    assert result.path == path
    assert result.detail == "high"
    assert path in toolset._file_hashes


@pytest.mark.asyncio
async def test_read_image_rejects_non_image_files(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    _write(path, "hello")

    toolset = FileToolset(tmp_path)
    result = await toolset.read_image("notes.txt")

    assert result == "Unsupported image format: .txt"


@pytest.mark.asyncio
async def test_read_image_reuses_workspace_path_guards(tmp_path: Path) -> None:
    hidden = tmp_path / ".secret.png"
    hidden.write_bytes(b"")

    toolset = FileToolset(tmp_path)
    assert await toolset.read_image(".secret.png") == "hidden files are not allowed"


@pytest.mark.asyncio
async def test_grep_returns_matches_and_recommendation(tmp_path: Path) -> None:
    _write(tmp_path / "alpha.txt", "first\nneedle here\nlast\n")
    _write(tmp_path / "beta.txt", "no match\n")

    toolset = FileToolset(tmp_path)
    output = await toolset.grep("needle", path_pattern=r".*\.txt$")

    assert "alpha.txt:2 | needle here" in output
    assert "You are recommended to use read_file" in output


@pytest.mark.asyncio
async def test_grep_validates_inputs(tmp_path: Path) -> None:
    toolset = FileToolset(tmp_path)
    assert (
        await toolset.grep("", path_pattern=r".*")
        == "pattern must be a non-empty string"
    )
    assert (
        await toolset.grep("x", path_pattern="")
        == "path_pattern is required; please provide a regex to scope the search"
    )
    assert (await toolset.grep("[", path_pattern=r".*")).startswith(
        "invalid pattern regex:"
    )
    assert (await toolset.grep("x", path_pattern="[")).startswith(
        "invalid path_pattern regex:"
    )


@pytest.mark.asyncio
async def test_grep_rejects_full_wildcard_regexes(tmp_path: Path) -> None:
    toolset = FileToolset(tmp_path)

    assert "pattern cannot be a full wildcard regex" in await toolset.grep(
        ".",
        path_pattern=r".*\.py$",
    )
    assert "pattern cannot be a full wildcard regex" in await toolset.grep(
        ".*",
        path_pattern=r".*\.py$",
    )
    assert "pattern cannot be a full wildcard regex" in await toolset.grep(
        ".+",
        path_pattern=r".*\.py$",
    )
    assert "path_pattern cannot be a full wildcard regex" in await toolset.grep(
        "needle",
        path_pattern=".",
    )
    assert "path_pattern cannot be a full wildcard regex" in await toolset.grep(
        "needle",
        path_pattern=r"^.*$",
    )
    assert "path_pattern cannot be a full wildcard regex" in await toolset.grep(
        "needle",
        path_pattern=r"(?s).*",
    )


@pytest.mark.asyncio
async def test_grep_returns_recommendation_when_no_matches(tmp_path: Path) -> None:
    _write(tmp_path / "alpha.txt", "nothing here\n")
    toolset = FileToolset(tmp_path)

    output = await toolset.grep("needle", path_pattern=r".*\.txt$")
    assert output == "Nothing matched provided pattern."


@pytest.mark.asyncio
async def test_sed_requires_read_and_detects_stale_hash(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    _write(path, "alpha\nbeta\n")

    toolset = FileToolset(tmp_path)
    result = await toolset.sed(
        "data.txt",
        start_line=1,
        end_line=2,
        pattern_to_be_replaced="alpha",
        new_string="omega",
    )
    assert result == STALE_FILE_MESSAGE

    await toolset.read_file("data.txt")
    _write(path, "changed\n")
    result = await toolset.sed(
        "data.txt",
        start_line=1,
        end_line=1,
        pattern_to_be_replaced="changed",
        new_string="done",
    )
    assert result == STALE_FILE_MESSAGE


@pytest.mark.asyncio
async def test_sed_replaces_content_and_updates_hash(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    _write(path, "alpha\nbeta\n")

    toolset = FileToolset(tmp_path)
    await toolset.read_file("data.txt")
    result = await toolset.sed(
        "data.txt",
        start_line=1,
        end_line=1,
        pattern_to_be_replaced="alpha",
        new_string="omega",
    )

    assert result.startswith("after edition: we have:\n")
    assert path.read_text(encoding="utf-8") == "omega\nbeta\n"
    assert path in toolset._file_hashes


@pytest.mark.asyncio
async def test_sed_reports_pattern_missing(tmp_path: Path) -> None:
    _write(tmp_path / "data.txt", "alpha\nbeta\n")

    toolset = FileToolset(tmp_path)
    await toolset.read_file("data.txt")
    result = await toolset.sed(
        "data.txt",
        start_line=1,
        end_line=2,
        pattern_to_be_replaced="gamma",
        new_string="omega",
    )

    assert "pattern_to_be_replaced was not found" in result


@pytest.mark.asyncio
async def test_echo_into_creates_file_and_tracks_hash(tmp_path: Path) -> None:
    toolset = FileToolset(tmp_path)
    result = await toolset.echo_into("new.txt", "hello")

    assert result == "write success: new.txt"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello"
    assert (tmp_path / "new.txt") in toolset._file_hashes


@pytest.mark.asyncio
async def test_echo_into_requires_read_for_existing_files(tmp_path: Path) -> None:
    path = tmp_path / "existing.txt"
    _write(path, "original")

    toolset = FileToolset(tmp_path)
    result = await toolset.echo_into("existing.txt", "updated")
    assert result == STALE_FILE_MESSAGE

    await toolset.read_file("existing.txt")
    result = await toolset.echo_into("existing.txt", "updated")
    assert result == "write success: existing.txt"
    assert path.read_text(encoding="utf-8") == "updated"


@pytest.mark.asyncio
async def test_echo_into_rejects_missing_parent(tmp_path: Path) -> None:
    toolset = FileToolset(tmp_path)
    result = await toolset.echo_into("missing/child.txt", "data")

    assert result == "parent directory does not exist; please create the folder first"
