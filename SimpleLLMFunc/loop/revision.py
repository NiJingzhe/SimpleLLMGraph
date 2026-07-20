"""Deterministic implementation and dependency revision hashing."""

from __future__ import annotations

import hashlib
import inspect
import json
import marshal
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import cast


def module_revision(path: str) -> str:
    """Hash the source file that owns a runtime implementation."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combine_revisions(*revisions: str) -> str:
    """Combine semantic dependency revisions into one stable digest."""

    encoded = json.dumps(
        revisions,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dependency_revision(value: object) -> object:
    revision = getattr(value, "revision", None)
    if isinstance(revision, str) and revision:
        return {"revision": revision, "type": type(value).__qualname__}
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Path):
        return {"path": str(value.resolve())}
    if isinstance(value, (list, tuple)):
        items = cast(list[object] | tuple[object, ...], value)
        return [_dependency_revision(item) for item in items]
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        return {
            str(key): _dependency_revision(item)
            for key, item in sorted(
                items.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def function_revision(function: Callable[..., object]) -> str:
    """Hash a function implementation and its revision-aware closure."""

    implementation = bytearray(marshal.dumps(function.__code__))
    source_path = inspect.getsourcefile(function)
    if source_path is not None and Path(source_path).is_file():
        implementation.extend(Path(source_path).read_bytes())
    closure = [
        _dependency_revision(cell.cell_contents)
        for cell in function.__closure__ or ()
    ]
    implementation.extend(
        json.dumps(
            closure,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return f"function@sha256:{hashlib.sha256(implementation).hexdigest()}"
