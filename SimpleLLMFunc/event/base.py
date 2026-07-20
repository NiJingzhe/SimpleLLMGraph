"""Immutable semantic event envelopes and revisioned event views."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from typing import ClassVar, Self, TypeVar, overload

from pydantic import BaseModel, ConfigDict, Field


class BaseEvent(BaseModel):
    """A fact that has already happened during a Loop run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    correlation_id: str | None = None
    causation_id: str | None = None


class CtxMixin(BaseModel):
    """Marker for events that a context compiler may project."""

    model_config = ConfigDict(extra="forbid", frozen=True)


EventT = TypeVar("EventT", bound=BaseEvent)


class EventView(Sequence[BaseEvent]):
    """An immutable, revisioned snapshot of semantic events.

    Event payloads may contain mutable containers, so the view deep-copies on
    both ingress and egress. Callers can never mutate the stored snapshot.
    """

    __slots__: ClassVar[tuple[str, ...]] = (
        "_digest",
        "_events",
        "_revision",
        "_run_id",
    )

    def __init__(
        self,
        run_id: str,
        revision: int,
        events: tuple[BaseEvent, ...],
        digest: str,
    ) -> None:
        self._run_id = run_id
        self._revision = revision
        self._events = events
        self._digest = digest

    @classmethod
    def create(
        cls,
        run_id: str,
        revision: int,
        events: Sequence[BaseEvent],
    ) -> Self:
        if not run_id:
            raise ValueError("run_id must not be empty")
        if revision < 0:
            raise ValueError("revision must be greater than or equal to 0")

        copied = tuple(event.model_copy(deep=True) for event in events)
        event_ids = [event.id for event in copied]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate event id in EventView")

        payload = {
            "run_id": run_id,
            "revision": revision,
            "events": [event.model_dump(mode="json") for event in copied],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return cls(run_id, revision, copied, hashlib.sha256(encoded).hexdigest())

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def events(self) -> tuple[BaseEvent, ...]:
        return tuple(event.model_copy(deep=True) for event in self._events)

    def all(self, event_type: type[EventT]) -> tuple[EventT, ...]:
        return tuple(
            event.model_copy(deep=True)
            for event in self._events
            if isinstance(event, event_type)
        )

    def latest(self, event_type: type[EventT]) -> EventT:
        event = self.latest_or_none(event_type)
        if event is None:
            raise LookupError(f"EventView contains no {event_type.__name__}")
        return event

    def latest_or_none(self, event_type: type[EventT]) -> EventT | None:
        for event in reversed(self._events):
            if isinstance(event, event_type):
                return event.model_copy(deep=True)
        return None

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[BaseEvent]:
        return iter(self.events)

    @overload
    def __getitem__(self, index: int) -> BaseEvent: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[BaseEvent, ...]: ...

    def __getitem__(self, index: int | slice) -> BaseEvent | tuple[BaseEvent, ...]:
        if isinstance(index, slice):
            return tuple(event.model_copy(deep=True) for event in self._events[index])
        return self._events[index].model_copy(deep=True)

    def __repr__(self) -> str:
        return (
            f"EventView(run_id={self.run_id!r}, revision={self.revision}, "
            f"events={len(self)})"
        )
