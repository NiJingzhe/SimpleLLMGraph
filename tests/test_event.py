from typing import Literal

import pytest
from pydantic import ValidationError

from SimpleLLMFunc.event import BaseEvent, EventView, UserMessageEvent


class TagsEvent(BaseEvent):
    type: Literal["tags"] = "tags"
    tags: list[str]


def test_event_view_is_ordered_queryable_and_defensively_copied() -> None:
    source = TagsEvent(id="event-1", tags=["original"])
    user = UserMessageEvent(id="event-2", content="hello")
    view = EventView.create("run-1", 3, [source, user])

    source.tags.append("source-mutated")
    first_read = view.events
    assert isinstance(first_read[0], TagsEvent)
    first_read[0].tags.append("read-mutated")

    assert view.run_id == "run-1"
    assert view.revision == 3
    assert [event.id for event in view] == ["event-1", "event-2"]
    assert view.all(TagsEvent)[0].tags == ["original"]
    assert view.latest(UserMessageEvent).content == "hello"
    assert view.latest_or_none(TagsEvent) is not None
    assert view.digest == EventView.create("run-1", 3, [
        TagsEvent(id="event-1", tags=["original"]),
        user,
    ]).digest


def test_event_view_rejects_duplicate_ids_and_invalid_revision() -> None:
    event = UserMessageEvent(id="same", content="hello")

    with pytest.raises(ValueError, match="duplicate event id"):
        EventView.create("run-1", 0, [event, event])

    with pytest.raises(ValueError, match="revision"):
        EventView.create("run-1", -1, [])

    with pytest.raises(ValueError, match="run_id"):
        EventView.create("", 0, [])


def test_event_is_frozen_and_latest_requires_a_match() -> None:
    event = UserMessageEvent(id="event-1", content="hello")
    view = EventView.create("run-1", 0, [event])

    with pytest.raises(ValidationError):
        event.content = "changed"  # type: ignore[misc]

    with pytest.raises(LookupError, match="TagsEvent"):
        view.latest(TagsEvent)

    assert view.latest_or_none(TagsEvent) is None
    assert len(view) == 1
    assert view[0].id == "event-1"
    assert view[:1] == (event,)
    assert repr(view) == "EventView(run_id='run-1', revision=0, events=1)"
