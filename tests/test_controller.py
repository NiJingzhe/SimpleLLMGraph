import asyncio
from collections.abc import Sequence
from typing import Literal

import pytest

from SimpleLLMFunc.event import BaseEvent, UserMessageEvent
from SimpleLLMFunc.loop import (
    CancellationToken,
    DefaultController,
    Effect,
    EffectResolution,
    InMemoryRunStore,
    LoopPolicy,
    LoopState,
    Reduction,
    ResolutionStatus,
    Resolver,
    RunStatus,
    Runtime,
    Step,
)


class IncrementEffect(Effect):
    type: Literal["increment"] = "increment"
    amount: int


class IncrementedEvent(BaseEvent):
    type: Literal["incremented"] = "incremented"
    amount: int


class CounterState(LoopState):
    value: int = 0
    target: int = 0


class CounterResolver(Resolver):
    effect_type = IncrementEffect

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        assert isinstance(effect, IncrementEffect)
        if cancellation.cancelled:
            return EffectResolution.cancelled(effect)
        return EffectResolution.completed(effect)


class TrackingCounterResolver(CounterResolver):
    def __init__(self) -> None:
        self.calls = 0

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        self.calls += 1
        return await super().resolve(effect, cancellation)


class CounterLoop(LoopPolicy[CounterState, int, int]):
    policy_revision = "counter@1"

    def initialize(self, value: int, *, run_id: str) -> CounterState:
        return CounterState(
            run_id=run_id,
            revision=0,
            status=RunStatus.READY,
            settled=False,
            target=value,
        )

    def can_step(self, state: CounterState) -> bool:
        return state.status is RunStatus.READY

    def step(self, state: CounterState) -> Step:
        return Step(
            id=f"{state.run_id}:step:{state.revision}",
            run_id=state.run_id,
            source_revision=state.revision,
            policy_revision=self.policy_revision,
            kind="increment",
            effects=(
                IncrementEffect(
                    id=f"{state.run_id}:effect:{state.revision}",
                    attempt=1,
                    idempotency_key=f"{state.run_id}:{state.revision}:1",
                    amount=1,
                ),
            ),
        )

    def reduce(
        self,
        state: CounterState,
        step: Step,
        resolutions: Sequence[EffectResolution],
    ) -> Reduction[CounterState]:
        self.validate_cycle(state, step, resolutions)
        resolution = resolutions[0]
        if resolution.status is not ResolutionStatus.COMPLETED:
            next_state = state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "status": RunStatus.FAILED,
                    "settled": True,
                }
            )
            return Reduction.from_cycle(
                step=step,
                resolutions=resolutions,
                next_state=next_state,
            )

        effect = step.effects[0]
        assert isinstance(effect, IncrementEffect)
        value = state.value + effect.amount
        event = IncrementedEvent(
            id=f"{step.id}:event",
            causation_id=effect.id,
            amount=effect.amount,
        )
        done = value >= state.target
        next_state = state.model_copy(
            update={
                "revision": state.revision + 1,
                "value": value,
                "status": RunStatus.COMPLETED if done else RunStatus.READY,
                "settled": done,
                "events": (*state.events, event),
            }
        )
        return Reduction.from_cycle(
            step=step,
            resolutions=resolutions,
            next_state=next_state,
            events=(event,),
        )

    def result(self, state: CounterState) -> int:
        if state.status is not RunStatus.COMPLETED:
            raise RuntimeError("loop is not complete")
        return state.value


async def drive_manually(
    loop: CounterLoop,
    state: CounterState,
    runtime: Runtime,
    store: InMemoryRunStore[CounterState],
) -> CounterState:
    await store.create(state)
    while loop.can_step(state):
        step = loop.step(state)
        loop.validate_step(state, step)
        await store.record_step(step, expected_revision=state.revision)
        resolutions = await runtime.resolve_step(step)
        await store.record_resolutions(step, resolutions)
        reduction = loop.reduce(state, step, resolutions)
        state = await store.commit(
            step,
            resolutions,
            reduction,
            expected_revision=state.revision,
        )
    return state


@pytest.mark.asyncio
async def test_default_controller_matches_manual_drive() -> None:
    loop = CounterLoop()
    runtime = Runtime([CounterResolver()])
    packaged_store = InMemoryRunStore[CounterState]()
    manual_store = InMemoryRunStore[CounterState]()

    packaged = await DefaultController().run(
        loop,
        loop.initialize(3, run_id="equivalent"),
        runtime,
        packaged_store,
    )
    manual = await drive_manually(
        loop,
        loop.initialize(3, run_id="equivalent"),
        runtime,
        manual_store,
    )

    assert packaged == manual
    assert await packaged_store.journal("equivalent") == await manual_store.journal(
        "equivalent"
    )
    assert packaged.value == 3
    assert packaged.status is RunStatus.COMPLETED
    assert loop.result(packaged) == loop.result(manual) == 3


@pytest.mark.asyncio
async def test_controller_stops_on_cancellation() -> None:
    loop = CounterLoop()
    cancellation = CancellationToken()
    cancellation.cancel()

    state = await DefaultController().run(
        loop,
        loop.initialize(3, run_id="cancelled"),
        Runtime([CounterResolver()]),
        InMemoryRunStore[CounterState](),
        cancellation=cancellation,
    )

    assert state.status is RunStatus.FAILED
    assert state.settled


@pytest.mark.asyncio
async def test_controller_cancels_an_active_prepare_step_callback() -> None:
    loop = CounterLoop()
    cancellation = CancellationToken()
    started = asyncio.Event()
    callback_cancelled = asyncio.Event()

    async def prepare(step: Step) -> Step:
        started.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            callback_cancelled.set()
            raise
        return step

    task = asyncio.create_task(
        DefaultController().run(
            loop,
            loop.initialize(3, run_id="cancel-prepare"),
            Runtime([CounterResolver()]),
            InMemoryRunStore[CounterState](),
            cancellation=cancellation,
            prepare_step=prepare,
        )
    )
    await started.wait()
    cancellation.cancel()
    state = await task

    assert callback_cancelled.is_set()
    assert state.status is RunStatus.FAILED
    assert state.settled


@pytest.mark.asyncio
async def test_controller_propagates_direct_task_cancellation() -> None:
    loop = CounterLoop()
    started = asyncio.Event()
    callback_cancelled = asyncio.Event()

    async def prepare(step: Step) -> Step:
        started.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            callback_cancelled.set()
            raise
        return step

    task = asyncio.create_task(
        DefaultController().run(
            loop,
            loop.initialize(3, run_id="force-cancel-prepare"),
            Runtime([CounterResolver()]),
            InMemoryRunStore[CounterState](),
            prepare_step=prepare,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert callback_cancelled.is_set()


@pytest.mark.asyncio
async def test_controller_resumes_a_fully_recorded_uncommitted_cycle() -> None:
    loop = CounterLoop()
    store = InMemoryRunStore[CounterState]()
    state = loop.initialize(1, run_id="resume")
    await store.create(state)
    step = loop.step(state)
    resolutions = (EffectResolution.completed(step.effects[0]),)
    await store.record_step(step, expected_revision=0)
    await store.record_resolutions(step, resolutions)

    resumed = await DefaultController().resume(
        loop,
        "resume",
        Runtime([CounterResolver()]),
        store,
    )

    assert resumed.status is RunStatus.COMPLETED
    assert resumed.value == 1
    journal = await store.journal("resume")
    assert journal.steps == (step,)
    assert journal.resolutions == resolutions


@pytest.mark.asyncio
async def test_controller_rejects_stale_policy_before_resolution() -> None:
    loop = CounterLoop()
    store = InMemoryRunStore[CounterState]()
    state = loop.initialize(1, run_id="stale-policy")
    await store.create(state)
    step = loop.step(state).model_copy(update={"policy_revision": "counter@0"})
    await store.record_step(step, expected_revision=0)
    resolver = TrackingCounterResolver()

    with pytest.raises(ValueError, match="policy revision"):
        await DefaultController().resume(
            loop,
            state.run_id,
            Runtime([resolver]),
            store,
        )

    assert resolver.calls == 0
    journal = await store.journal(state.run_id)
    assert journal.steps == (step,)
    assert journal.resolutions == ()


@pytest.mark.asyncio
async def test_controller_prepares_a_new_step_before_recording() -> None:
    loop = CounterLoop()
    store = InMemoryRunStore[CounterState]()

    async def prepare(step: Step) -> Step:
        effect = step.effects[0]
        assert isinstance(effect, IncrementEffect)
        return step.model_copy(
            update={"effects": (effect.model_copy(update={"amount": 2}),)}
        )

    state = await DefaultController().run(
        loop,
        loop.initialize(2, run_id="prepared"),
        Runtime([CounterResolver()]),
        store,
        prepare_step=prepare,
    )

    assert state.value == 2
    journal = await store.journal("prepared")
    effect = journal.steps[0].effects[0]
    assert isinstance(effect, IncrementEffect)
    assert effect.amount == 2


def test_loop_cycle_validation_rejects_mismatches() -> None:
    loop = CounterLoop()
    state = loop.initialize(1, run_id="run-1")
    step = loop.step(state)
    resolution = EffectResolution.completed(step.effects[0])

    loop.validate_cycle(state, step, (resolution,))

    stale_state = state.model_copy(update={"revision": 1})
    with pytest.raises(ValueError, match="source revision"):
        loop.validate_cycle(stale_state, step, (resolution,))

    wrong_run_step = step.model_copy(update={"run_id": "run-2"})
    with pytest.raises(ValueError, match="run_id"):
        loop.validate_cycle(state, wrong_run_step, (resolution,))

    wrong_policy_step = step.model_copy(update={"policy_revision": "counter@2"})
    with pytest.raises(ValueError, match="policy revision"):
        loop.validate_cycle(state, wrong_policy_step, (resolution,))

    wrong = Effect(
        id="wrong",
        attempt=1,
        idempotency_key="wrong",
    )
    with pytest.raises(ValueError, match="do not match"):
        loop.validate_cycle(
            state,
            step,
            (EffectResolution.completed(wrong),),
        )


def test_counter_result_requires_completion() -> None:
    loop = CounterLoop()
    state = loop.initialize(1, run_id="run-1")
    state = state.model_copy(
        update={
            "events": (
                UserMessageEvent(id="event-1", content="still running"),
            )
        }
    )
    with pytest.raises(RuntimeError, match="not complete"):
        loop.result(state)
