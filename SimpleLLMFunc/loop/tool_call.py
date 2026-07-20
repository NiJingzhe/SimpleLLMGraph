"""Durable artifacts for one function-tool call."""

from __future__ import annotations

import json
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)
from pydantic.types import JsonValue

from SimpleLLMFunc.context.ir import ToolResult
from SimpleLLMFunc.event import BaseEvent, EventView
from SimpleLLMFunc.loop.core import (
    Effect,
    EffectResolution,
    ResolutionStatus,
    restore_artifact_subtype,
)
from SimpleLLMFunc.loop.json_codec import (
    JSON_CODEC_REVISION,
    canonical_json,
    reject_json_constant,
)
from SimpleLLMFunc.loop.revision import combine_revisions, module_revision


TOOL_CALL_ARTIFACT_REVISION = combine_revisions(
    module_revision(__file__),
    JSON_CODEC_REVISION,
)


class EventSnapshot(BaseModel):
    """Serializable copy of the semantic events visible before an Effect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    events: tuple[SerializeAsAny[BaseEvent], ...]
    digest: str = Field(min_length=1)

    @field_validator("events", mode="before")
    @classmethod
    def restore_events(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        items = cast(list[object] | tuple[object, ...], value)
        return tuple(restore_artifact_subtype(BaseEvent, item) for item in items)

    @model_validator(mode="after")
    def validate_digest(self) -> "EventSnapshot":
        if self.view.digest != self.digest:
            raise ValueError("event snapshot digest does not match its events")
        return self

    @classmethod
    def create(cls, view: EventView) -> "EventSnapshot":
        return cls(
            run_id=view.run_id,
            revision=view.revision,
            events=view.events,
            digest=view.digest,
        )

    @property
    def view(self) -> EventView:
        return EventView.create(self.run_id, self.revision, self.events)


class ToolCallEffect(Effect):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str
    event_snapshot: EventSnapshot

    @model_validator(mode="after")
    def validate_arguments(self) -> "ToolCallEffect":
        try:
            value = json.loads(
                self.arguments_json,
                parse_constant=reject_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be a JSON object")
        return self

    @classmethod
    def from_call(
        cls,
        *,
        id: str,
        attempt: int,
        idempotency_key: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, object],
        events: EventView,
    ) -> "ToolCallEffect":
        return cls(
            id=id,
            attempt=attempt,
            idempotency_key=idempotency_key,
            tool_call_id=tool_call_id,
            name=name,
            event_snapshot=EventSnapshot.create(events),
            arguments_json=canonical_json(arguments),
        )

    @property
    def arguments(self) -> dict[str, object]:
        value = json.loads(
            self.arguments_json,
            parse_constant=reject_json_constant,
        )
        if not isinstance(value, dict):
            raise AssertionError("validated tool arguments are not an object")
        parsed = cast(dict[str, object], value)
        return dict(parsed)


class ToolCallResolution(EffectResolution):
    type: Literal["tool_call_resolution"] = "tool_call_resolution"
    tool_call_id: str
    name: str
    value: JsonValue = None
    result: ToolResult | None = None
    tool_revision: str | None = None
    event_digest: str | None = None

    @field_validator("value")
    @classmethod
    def validate_json_value(cls, value: JsonValue) -> JsonValue:
        canonical_json(value)
        return value

    @model_validator(mode="after")
    def validate_completed_result(self) -> "ToolCallResolution":
        if self.status is ResolutionStatus.COMPLETED and (
            self.result is None
            or not self.tool_revision
            or not self.event_digest
        ):
            raise ValueError(
                "completed tool Resolution requires result compilation evidence"
            )
        return self
