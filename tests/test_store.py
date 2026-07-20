from typing import Literal, cast

import pytest

from SimpleLLMFunc.event import BaseEvent, UserMessageEvent
from SimpleLLMFunc.loop import (
    Effect,
    EffectResolution,
    InMemoryRunStore,
    InvalidReductionError,
    LoopState,
    Reduction,
    RevisionConflictError,
    RunJournal,
    RunStatus,
    Step,
)


class NoopEffect(Effect):
    type: Literal["noop"] = "noop"
    payload: str = "original"


class MutableEvent(BaseEvent):
    type: Literal["mutable"] = "mutable"
    values: list[str]


def make_state(revision: int = 0) -> LoopState:
    return LoopState(
        run_id="run-1",
        revision=revision,
        status=RunStatus.READY,
        settled=False,
        events=(UserMessageEvent(id="event-1", content="hello"),),
    )


def make_step() -> Step:
    return Step(
        id="step-1",
        run_id="run-1",
        source_revision=0,
        policy_revision="policy@1",
        kind="noop",
        effects=(
            NoopEffect(
                id="effect-1",
                attempt=1,
                idempotency_key="run-1:effect-1:1",
            ),
        ),
    )


def make_reduction(
    next_state: LoopState,
    *,
    events: tuple[BaseEvent, ...] = (),
    step: Step | None = None,
    resolutions: tuple[EffectResolution, ...] | None = None,
) -> Reduction[LoopState]:
    source_step = step or make_step()
    source_resolutions = resolutions or (
        EffectResolution.completed(source_step.effects[0]),
    )
    return Reduction.from_cycle(
        step=source_step,
        resolutions=source_resolutions,
        next_state=next_state,
        events=events,
    )


@pytest.mark.asyncio
async def test_store_records_and_commits_one_complete_cycle() -> None:
    store = InMemoryRunStore[LoopState]()
    initial = make_state()
    await store.create(initial)
    step = make_step()
    await store.record_step(step, expected_revision=0)
    resolutions = (EffectResolution.completed(step.effects[0]),)
    await store.record_resolutions(step, resolutions)
    event = UserMessageEvent(
        id="event-2",
        content="next",
        causation_id="effect-1",
    )
    next_state = initial.model_copy(
        update={
            "revision": 1,
            "events": (*initial.events, event),
        }
    )
    reduction = make_reduction(
        next_state,
        events=(event,),
        step=step,
        resolutions=resolutions,
    )

    committed = await store.commit(
        step,
        resolutions,
        reduction,
        expected_revision=0,
    )

    assert committed.revision == 1
    assert [item.id for item in committed.events] == ["event-1", "event-2"]
    assert committed.event_view.revision == 1
    assert (await store.load("run-1")) == committed
    journal = await store.journal("run-1")
    assert journal.steps == (step,)
    assert journal.resolutions == resolutions


@pytest.mark.asyncio
async def test_store_uses_defensive_copies() -> None:
    store = InMemoryRunStore[LoopState]()
    mutable = MutableEvent(id="event-mutable", values=["original"])
    initial = make_state().model_copy(update={"events": (mutable,)})
    await store.create(initial)
    loaded = await store.load("run-1")
    loaded_event = cast(MutableEvent, loaded.events[0])
    loaded_event.values.append("bad")

    stored_event = cast(MutableEvent, (await store.load("run-1")).events[0])
    assert stored_event.values == ["original"]


@pytest.mark.asyncio
async def test_store_rejects_stale_steps_and_commits() -> None:
    store = InMemoryRunStore[LoopState]()
    await store.create(make_state())

    with pytest.raises(RevisionConflictError):
        await store.record_step(make_step(), expected_revision=1)

    invalid_next = make_state().model_copy(update={"revision": 1})
    reduction = make_reduction(invalid_next)
    with pytest.raises(RevisionConflictError):
        await store.commit(
            make_step(),
            (EffectResolution.completed(make_step().effects[0]),),
            reduction,
            expected_revision=1,
        )


@pytest.mark.asyncio
async def test_store_rejects_invalid_reduction_and_resolution_mapping() -> None:
    store = InMemoryRunStore[LoopState]()
    initial = make_state()
    await store.create(initial)
    step = make_step()
    await store.record_step(step, expected_revision=0)

    wrong_effect = NoopEffect(
        id="wrong",
        attempt=1,
        idempotency_key="wrong:1",
    )
    with pytest.raises(ValueError, match="do not match"):
        await store.record_resolutions(
            step,
            (EffectResolution.completed(wrong_effect),),
        )

    non_appended = UserMessageEvent(id="event-2", content="next")
    bad_state = initial.model_copy(
        update={
            "revision": 1,
            "events": (non_appended,),
        }
    )
    with pytest.raises(InvalidReductionError, match="append"):
        await store.commit(
            step,
            (EffectResolution.completed(step.effects[0]),),
            make_reduction(
                bad_state,
                events=(non_appended,),
                step=step,
            ),
            expected_revision=0,
        )


@pytest.mark.asyncio
async def test_store_rejects_duplicate_runs_steps_and_missing_runs() -> None:
    store = InMemoryRunStore[LoopState]()
    initial = make_state()
    await store.create(initial)

    with pytest.raises(ValueError, match="already exists"):
        await store.create(initial)

    with pytest.raises(KeyError, match="missing"):
        await store.load("missing")

    with pytest.raises(KeyError, match="missing"):
        await store.journal("missing")

    step = make_step()
    await store.record_step(step, expected_revision=0)
    with pytest.raises(ValueError, match="already recorded"):
        await store.record_step(step, expected_revision=0)


@pytest.mark.asyncio
async def test_store_requires_revision_zero_and_one_step_per_revision() -> None:
    store = InMemoryRunStore[LoopState]()
    with pytest.raises(ValueError, match="revision 0"):
        await store.create(make_state(revision=1))

    await store.create(make_state())
    mismatched = make_step().model_copy(update={"source_revision": 1})
    with pytest.raises(RevisionConflictError, match="source revision"):
        await store.record_step(mismatched, expected_revision=0)

    step = make_step()
    await store.record_step(step, expected_revision=0)
    second = step.model_copy(update={"id": "step-2"})
    with pytest.raises(ValueError, match="already has Step"):
        await store.record_step(second, expected_revision=0)


@pytest.mark.asyncio
async def test_store_requires_recorded_step_and_single_resolution_batch() -> None:
    store = InMemoryRunStore[LoopState]()
    initial = make_state()
    await store.create(initial)
    step = make_step()
    resolutions = (EffectResolution.completed(step.effects[0]),)

    with pytest.raises(ValueError, match="Step must be recorded"):
        await store.record_resolutions(step, resolutions)

    await store.record_step(step, expected_revision=0)
    await store.record_resolutions(step, resolutions)
    with pytest.raises(ValueError, match="already recorded"):
        await store.record_resolutions(step, resolutions)


@pytest.mark.asyncio
async def test_store_loads_pending_cycle_artifacts_defensively() -> None:
    store = InMemoryRunStore[LoopState]()
    await store.create(make_state())
    assert await store.pending_step("run-1", expected_revision=0) is None

    step = make_step()
    await store.record_step(step, expected_revision=0)
    pending = await store.pending_step("run-1", expected_revision=0)
    assert pending == step
    assert await store.recorded_resolutions(step) is None

    unrecorded = step.model_copy(update={"id": "unrecorded"})
    with pytest.raises(ValueError, match="Step must be recorded"):
        await store.recorded_resolutions(unrecorded)

    resolutions = (EffectResolution.completed(step.effects[0]),)
    await store.record_resolutions(step, resolutions)
    assert await store.recorded_resolutions(step) == resolutions


def test_cycle_artifacts_round_trip_concrete_types_through_json() -> None:
    state = make_state()
    step = make_step()
    resolutions = (EffectResolution.completed(step.effects[0]),)
    journal = RunJournal(steps=(step,), resolutions=resolutions)

    restored_state = LoopState.model_validate_json(state.model_dump_json())
    restored_step = Step.model_validate_json(step.model_dump_json())
    restored_journal = RunJournal.model_validate_json(journal.model_dump_json())

    assert isinstance(restored_state.events[0], UserMessageEvent)
    assert isinstance(restored_step.effects[0], NoopEffect)
    assert restored_step == step
    assert restored_journal == journal

    with pytest.raises(ValueError):
        RunJournal.model_validate({"steps": (), "resolutions": "invalid"})


@pytest.mark.asyncio
async def test_store_binds_commit_to_exact_cycle_evidence() -> None:
    store = InMemoryRunStore[LoopState]()
    initial = make_state()
    await store.create(initial)
    step = make_step()
    completed = (EffectResolution.completed(step.effects[0]),)
    await store.record_step(step, expected_revision=0)
    await store.record_resolutions(step, completed)
    next_state = initial.model_copy(update={"revision": 1})

    stale_step = step.model_copy(update={"source_revision": 1})
    stale_reduction = make_reduction(
        next_state,
        step=stale_step,
        resolutions=completed,
    )
    with pytest.raises(RevisionConflictError, match="source revision"):
        await store.commit(
            stale_step,
            completed,
            stale_reduction,
            expected_revision=0,
        )

    unrelated_step = step.model_copy(update={"id": "unrelated"})
    unrelated = make_reduction(
        next_state,
        step=unrelated_step,
        resolutions=completed,
    )
    with pytest.raises(InvalidReductionError, match="not bound"):
        await store.commit(step, completed, unrelated, expected_revision=0)

    altered_step = step.model_copy(update={"kind": "altered"})
    altered = make_reduction(
        next_state,
        step=altered_step,
        resolutions=completed,
    )
    with pytest.raises(InvalidReductionError, match="not bound"):
        await store.commit(step, completed, altered, expected_revision=0)

    altered_effect = step.effects[0].model_copy(update={"payload": "changed"})
    altered_payload_step = step.model_copy(update={"effects": (altered_effect,)})
    altered_payload = make_reduction(
        next_state,
        step=altered_payload_step,
        resolutions=completed,
    )
    assert altered_payload_step.digest != step.digest
    with pytest.raises(InvalidReductionError, match="not bound"):
        await store.commit(
            step,
            completed,
            altered_payload,
            expected_revision=0,
        )

    denied = (EffectResolution.denied(step.effects[0], "denied"),)
    denied_reduction = make_reduction(
        next_state,
        step=step,
        resolutions=denied,
    )
    with pytest.raises(InvalidReductionError, match="recorded batch"):
        await store.commit(step, denied, denied_reduction, expected_revision=0)


@pytest.mark.asyncio
async def test_store_requires_complete_cycle_before_commit() -> None:
    initial = make_state()
    next_state = initial.model_copy(update={"revision": 1})
    reduction = make_reduction(next_state)

    no_step = InMemoryRunStore[LoopState]()
    await no_step.create(initial)
    with pytest.raises(InvalidReductionError, match="Step must be recorded"):
        step = make_step()
        await no_step.commit(
            step,
            (EffectResolution.completed(step.effects[0]),),
            reduction,
            expected_revision=0,
        )

    no_resolutions = InMemoryRunStore[LoopState]()
    await no_resolutions.create(initial)
    await no_resolutions.record_step(make_step(), expected_revision=0)
    with pytest.raises(InvalidReductionError, match="Resolutions must be recorded"):
        step = make_step()
        await no_resolutions.commit(
            step,
            (EffectResolution.completed(step.effects[0]),),
            reduction,
            expected_revision=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("next_state", "events", "message"),
    [
        (make_state().model_copy(update={"run_id": "run-2", "revision": 1}), (), "run_id"),
        (make_state().model_copy(update={"revision": 2}), (), "exactly 1"),
        (
            make_state().model_copy(
                update={
                    "revision": 1,
                    "events": (
                        *make_state().events,
                        UserMessageEvent(id="event-2", content="next"),
                    ),
                }
            ),
            (),
            "equal the appended",
        ),
        (
            make_state().model_copy(
                update={
                    "revision": 1,
                    "events": (
                        *make_state().events,
                        UserMessageEvent(id="event-1", content="duplicate"),
                    ),
                }
            ),
            (UserMessageEvent(id="event-1", content="duplicate"),),
            "duplicate Event id",
        ),
    ],
)
async def test_store_rejects_all_invalid_reduction_shapes(
    next_state: LoopState,
    events: tuple[BaseEvent, ...],
    message: str,
) -> None:
    store = InMemoryRunStore[LoopState]()
    await store.create(make_state())

    with pytest.raises(InvalidReductionError, match=message):
        await store.commit(
            make_step(),
            (EffectResolution.completed(make_step().effects[0]),),
            make_reduction(next_state, events=events),
            expected_revision=0,
        )
