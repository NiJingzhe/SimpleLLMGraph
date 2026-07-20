"""Canonical JSON helpers shared by durable Loop artifacts."""

from __future__ import annotations

import json

from SimpleLLMFunc.loop.revision import module_revision


JSON_CODEC_REVISION = module_revision(__file__)


def canonical_json(value: object) -> str:
    """Encode a value with the deterministic JSON format used by the Loop."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def reject_json_constant(value: str) -> None:
    """Reject NaN and Infinity while decoding provider or journal JSON."""

    raise ValueError(f"tool arguments contain non-JSON constant {value}")
