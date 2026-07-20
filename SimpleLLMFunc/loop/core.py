"""Pure Loop protocol and immutable Step/Resolution models."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Generic, Self, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from SimpleLLMFunc.event import BaseEvent, EventView


def _subclasses(model: type[BaseModel]) -> tuple[type[BaseModel], ...]:
    direct = tuple(model.__subclasses__())
    return direct + tuple(child for item in direct for child in _subclasses(item))


def restore_artifact_subtype(model: type[BaseModel], value: object) -> object:
    """Rehydrate a registered concrete model from its literal ``type`` tag."""

    if isinstance(value, model) or not isinstance(value, dict):
        return value
    payload = cast(dict[str, object], value)
    type_tag = payload.get("type")
    if type_tag is None:
        return cast(object, payload)
    matches: list[type[BaseModel]] = []
    for candidate in _subclasses(model):
        field = candidate.model_fields.get("type")
        if field is not None and field.default == type_tag:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous {model.__name__} type {type_tag!r}")
    return matches[0].model_validate(payload)


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ResolutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class RunStatus(str, Enum):
    READY = "ready"
    WAITING_FOR_EFFECT = "waiting_for_effect"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StructuredError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    message: str


class Effect(BaseModel):
    """An external operation requested by a pure Step calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1)


class EffectResolution(BaseModel):
    """The terminal outcome of one Effect attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effect_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    status: ResolutionStatus
    error: StructuredError | None = None

    @classmethod
    def from_effect(cls, effect: Effect, **data: object) -> Self:
        return cls.model_validate(
            {
                "effect_id": effect.id,
                "attempt": effect.attempt,
                "status": ResolutionStatus.COMPLETED,
                **data,
            }
        )

    @classmethod
    def completed(cls, effect: Effect) -> Self:
        return cls(
            effect_id=effect.id,
            attempt=effect.attempt,
            status=ResolutionStatus.COMPLETED,
        )

    @classmethod
    def failed(cls, effect: Effect, error: BaseException) -> Self:
        return cls(
            effect_id=effect.id,
            attempt=effect.attempt,
            status=ResolutionStatus.FAILED,
            error=StructuredError(type=type(error).__name__, message=str(error)),
        )

    @classmethod
    def denied(cls, effect: Effect, message: str) -> Self:
        return cls(
            effect_id=effect.id,
            attempt=effect.attempt,
            status=ResolutionStatus.DENIED,
            error=StructuredError(type="PolicyDenied", message=message),
        )

    @classmethod
    def cancelled(cls, effect: Effect) -> Self:
        return cls(
            effect_id=effect.id,
            attempt=effect.attempt,
            status=ResolutionStatus.CANCELLED,
        )


class Step(BaseModel):
    """A pure, inspectable plan for the next external operations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    source_revision: int = Field(ge=0)
    policy_revision: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    effects: tuple[SerializeAsAny[Effect], ...]
    execution: ExecutionMode = ExecutionMode.SEQUENTIAL

    @field_validator("effects", mode="before")
    @classmethod
    def restore_effects(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        items = cast(list[object] | tuple[object, ...], value)
        return tuple(restore_artifact_subtype(Effect, item) for item in items)

    @model_validator(mode="after")
    def validate_effects(self) -> Self:
        if not self.effects:
            raise ValueError("a Step must contain at least one Effect")
        effect_ids = [effect.id for effect in self.effects]
        if len(effect_ids) != len(set(effect_ids)):
            raise ValueError("duplicate effect id in Step")
        return self

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json", serialize_as_any=True),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class LoopState(BaseModel):
    """Revisioned semantic state from which a Loop calculates Steps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    status: RunStatus
    settled: bool
    events: tuple[SerializeAsAny[BaseEvent], ...] = ()

    @field_validator("events", mode="before")
    @classmethod
    def restore_events(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        items = cast(list[object] | tuple[object, ...], value)
        return tuple(restore_artifact_subtype(BaseEvent, item) for item in items)

    @property
    def event_view(self) -> EventView:
        return EventView.create(self.run_id, self.revision, self.events)


StateT = TypeVar("StateT", bound=LoopState)
CycleStateT = TypeVar("CycleStateT", bound=LoopState)
InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


class Reduction(BaseModel, Generic[StateT]):
    """A pure candidate state transition and the Events it appends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_step_id: str = Field(min_length=1)
    source_step_digest: str = Field(min_length=1)
    source_revision: int = Field(ge=0)
    resolution_digest: str = Field(min_length=1)
    next_state: StateT
    events: tuple[SerializeAsAny[BaseEvent], ...] = ()

    @field_validator("events", mode="before")
    @classmethod
    def restore_events(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        items = cast(list[object] | tuple[object, ...], value)
        return tuple(restore_artifact_subtype(BaseEvent, item) for item in items)

    @classmethod
    def from_cycle(
        cls,
        *,
        step: Step,
        resolutions: Sequence[EffectResolution],
        next_state: CycleStateT,
        events: tuple[BaseEvent, ...] = (),
    ) -> Reduction[CycleStateT]:
        """Bind a candidate transition to the exact cycle that produced it."""

        payload = [resolution.model_dump(mode="json") for resolution in resolutions]
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return Reduction[CycleStateT](
            source_step_id=step.id,
            source_step_digest=step.digest,
            source_revision=step.source_revision,
            resolution_digest=hashlib.sha256(encoded).hexdigest(),
            next_state=next_state,
            events=events,
        )

    def matches_cycle(
        self,
        step: Step,
        resolutions: Sequence[EffectResolution],
    ) -> bool:
        expected = type(self).from_cycle(
            step=step,
            resolutions=resolutions,
            next_state=self.next_state,
            events=self.events,
        )
        return (
            self.source_step_id == expected.source_step_id
            and self.source_step_digest == expected.source_step_digest
            and self.source_revision == expected.source_revision
            and self.resolution_digest == expected.resolution_digest
        )


class LoopPolicy(ABC, Generic[StateT, InputT, ResultT]):
    """Low-level pure policy for Step calculation and State reduction."""

    policy_revision: str

    @abstractmethod
    def initialize(self, value: InputT, *, run_id: str) -> StateT:
        """Create revision-zero State without external I/O."""

    @abstractmethod
    def can_step(self, state: StateT) -> bool:
        """Return whether another Step can be calculated."""

    @abstractmethod
    def step(self, state: StateT) -> Step:
        """Calculate the next Step without external I/O or State mutation."""

    @abstractmethod
    def reduce(
        self,
        state: StateT,
        step: Step,
        resolutions: Sequence[EffectResolution],
    ) -> Reduction[StateT]:
        """Deterministically calculate the next semantic State."""

    @abstractmethod
    def result(self, state: StateT) -> ResultT:
        """Materialize the result of a terminal State."""

    def validate_step(self, state: StateT, step: Step) -> None:
        """Reject a Step that cannot execute against the current policy State."""

        if step.run_id != state.run_id:
            raise ValueError("Step run_id does not match LoopState")
        if step.source_revision != state.revision:
            raise ValueError("Step source revision does not match LoopState")
        if step.policy_revision != self.policy_revision:
            raise ValueError("Step policy revision does not match Loop")

    def validate_cycle(
        self,
        state: StateT,
        step: Step,
        resolutions: Sequence[EffectResolution],
    ) -> None:
        self.validate_step(state, step)
        expected = [(effect.id, effect.attempt) for effect in step.effects]
        actual = [
            (resolution.effect_id, resolution.attempt)
            for resolution in resolutions
        ]
        if expected != actual:
            raise ValueError("Resolutions do not match Step Effects")
