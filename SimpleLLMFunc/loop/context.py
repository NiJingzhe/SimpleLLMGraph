"""Compiled, inspectable context artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from SimpleLLMFunc.context.ir import Request


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class ContextProvenance(BaseModel):
    """Declared lineage from a Request path to source semantic Events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    event_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_event_ids(self) -> Self:
        if len(self.event_ids) != len(set(self.event_ids)):
            raise ValueError("duplicate event id in context provenance")
        if any(not event_id for event_id in self.event_ids):
            raise ValueError("context provenance event ids must not be empty")
        return self


class CompiledContext(BaseModel):
    """The exact provider Request plus compilation evidence.

    The Request is stored as canonical JSON and reparsed on access so callers
    cannot mutate the artifact after its digest has been calculated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_json: str
    request_digest: str
    source_revision: int = Field(ge=0)
    compiler_revision: str = Field(min_length=1)
    provenance: tuple[ContextProvenance, ...] = ()
    parent_digest: str | None = None
    transformation: str | None = None
    digest: str

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        try:
            request = Request.model_validate_json(self.request_json)
        except ValueError as exc:
            raise ValueError("request_json must contain a valid Request") from exc
        request_data = request.model_dump(mode="json")
        if self.request_json != _canonical_json(request_data):
            raise ValueError("request_json must be canonical JSON")
        if self.request_digest != _digest(request_data):
            raise ValueError("request_digest does not match request_json")
        paths = [entry.path for entry in self.provenance]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate provenance path")
        for path in paths:
            prefix, _, index_text = path.partition(".")
            if prefix != "messages" or not index_text.isdigit():
                raise ValueError("context provenance path must target a Request message")
            if int(index_text) >= len(request.messages):
                raise ValueError("context provenance path is outside the Request")
        if (self.parent_digest is None) != (self.transformation is None):
            raise ValueError(
                "parent_digest and transformation must be provided together"
            )
        if self.parent_digest == "" or self.transformation == "":
            raise ValueError("derivation metadata must not be empty")
        context_data = {
            "request_digest": self.request_digest,
            "source_revision": self.source_revision,
            "compiler_revision": self.compiler_revision,
            "provenance": [
                entry.model_dump(mode="json") for entry in self.provenance
            ],
            "parent_digest": self.parent_digest,
            "transformation": self.transformation,
        }
        if self.digest != _digest(context_data):
            raise ValueError("context digest does not match compilation evidence")
        return self

    @classmethod
    def create(
        cls,
        *,
        request: Request,
        source_revision: int,
        compiler_revision: str,
        provenance: tuple[ContextProvenance, ...] = (),
        parent_digest: str | None = None,
        transformation: str | None = None,
    ) -> Self:
        if source_revision < 0:
            raise ValueError("source_revision must be greater than or equal to 0")
        if not compiler_revision:
            raise ValueError("compiler_revision must not be empty")
        paths = [entry.path for entry in provenance]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate provenance path")
        if (parent_digest is None) != (transformation is None):
            raise ValueError(
                "parent_digest and transformation must be provided together"
            )

        request_data = request.model_dump(mode="json")
        request_json = _canonical_json(request_data)
        request_digest = _digest(request_data)
        context_data = {
            "request_digest": request_digest,
            "source_revision": source_revision,
            "compiler_revision": compiler_revision,
            "provenance": [entry.model_dump(mode="json") for entry in provenance],
            "parent_digest": parent_digest,
            "transformation": transformation,
        }
        return cls(
            request_json=request_json,
            request_digest=request_digest,
            source_revision=source_revision,
            compiler_revision=compiler_revision,
            provenance=provenance,
            parent_digest=parent_digest,
            transformation=transformation,
            digest=_digest(context_data),
        )

    @property
    def request(self) -> Request:
        return Request.model_validate_json(self.request_json)

    def derive(self, *, request: Request, transformation: str) -> Self:
        if not transformation:
            raise ValueError("transformation must not be empty")
        if request.messages != self.request.messages:
            raise ValueError(
                "CompiledContext.derive cannot change messages without new provenance"
            )
        return type(self).create(
            request=request,
            source_revision=self.source_revision,
            compiler_revision=self.compiler_revision,
            provenance=self.provenance,
            parent_digest=self.digest,
            transformation=transformation,
        )
