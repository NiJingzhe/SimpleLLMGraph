import asyncio
from typing import ClassVar, Literal

import pytest

from SimpleLLMFunc.loop import (
    CancellationToken,
    Effect,
    EffectResolution,
    ExecutionMode,
    InvalidResolutionError,
    LoopState,
    MissingResolverError,
    Reduction,
    ResolutionStatus,
    Resolver,
    RunStatus,
    Runtime,
    Step,
)


class ValueEffect(Effect):
    type: Literal["value"] = "value"
    value: int


class OtherEffect(Effect):
    type: Literal["other"] = "other"


class ValueResolution(EffectResolution):
    type: Literal["value_resolution"] = "value_resolution"
    value: int


class RecordingResolver(Resolver):
    effect_type: ClassVar[type[Effect]] = ValueEffect

    def __init__(self, order: list[int], gate: asyncio.Event | None = None) -> None:
        self.order = order
        self.gate = gate

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        assert isinstance(effect, ValueEffect)
        if self.gate is not None:
            await self.gate.wait()
        if cancellation.cancelled:
            return EffectResolution.cancelled(effect)
        self.order.append(effect.value)
        if effect.value == 99:
            raise RuntimeError("boom")
        return ValueResolution.from_effect(effect, value=effect.value * 2)


class CancellingResolver(Resolver):
    effect_type: ClassVar[type[Effect]] = ValueEffect

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        cancellation.cancel()
        return EffectResolution.completed(effect)


class InvalidResolver(Resolver):
    effect_type: ClassVar[type[Effect]] = ValueEffect

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        del cancellation
        return EffectResolution(
            effect_id=effect.id,
            attempt=effect.attempt + 1,
            status=ResolutionStatus.COMPLETED,
        )


class TaskCancellingResolver(Resolver):
    effect_type: ClassVar[type[Effect]] = ValueEffect

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        del effect, cancellation
        raise asyncio.CancelledError


class BlockingResolver(Resolver):
    effect_type: ClassVar[type[Effect]] = ValueEffect

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        del cancellation
        self.started.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return EffectResolution.completed(effect)


def make_step(mode: ExecutionMode, values: tuple[int, ...]) -> Step:
    return Step(
        id=f"step-{mode.value}",
        run_id="run-1",
        source_revision=0,
        policy_revision="policy@1",
        kind="test",
        effects=tuple(
            ValueEffect(
                id=f"effect-{value}",
                attempt=1,
                idempotency_key=f"run-1:{value}:1",
                value=value,
            )
            for value in values
        ),
        execution=mode,
    )


@pytest.mark.asyncio
async def test_runtime_resolves_sequential_step_in_order() -> None:
    order: list[int] = []
    runtime = Runtime([RecordingResolver(order)])

    resolutions = await runtime.resolve_step(
        make_step(ExecutionMode.SEQUENTIAL, (1, 2, 3))
    )

    assert order == [1, 2, 3]
    assert [resolution.status for resolution in resolutions] == [
        ResolutionStatus.COMPLETED,
        ResolutionStatus.COMPLETED,
        ResolutionStatus.COMPLETED,
    ]
    assert [resolution.effect_id for resolution in resolutions] == [
        "effect-1",
        "effect-2",
        "effect-3",
    ]


@pytest.mark.asyncio
async def test_runtime_resolves_parallel_step_concurrently() -> None:
    order: list[int] = []
    gate = asyncio.Event()
    runtime = Runtime([RecordingResolver(order, gate)])
    task = asyncio.create_task(
        runtime.resolve_step(make_step(ExecutionMode.PARALLEL, (1, 2)))
    )

    await asyncio.sleep(0)
    assert not task.done()
    gate.set()
    resolutions = await task

    assert sorted(order) == [1, 2]
    assert [resolution.effect_id for resolution in resolutions] == [
        "effect-1",
        "effect-2",
    ]


@pytest.mark.asyncio
async def test_runtime_converts_resolver_error_to_failed_resolution() -> None:
    runtime = Runtime([RecordingResolver([])])

    resolution = (
        await runtime.resolve_step(
            make_step(ExecutionMode.SEQUENTIAL, (99,))
        )
    )[0]

    assert resolution.status is ResolutionStatus.FAILED
    assert resolution.effect_id == "effect-99"
    assert resolution.attempt == 1
    assert resolution.error is not None
    assert resolution.error.type == "RuntimeError"
    assert resolution.error.message == "boom"


@pytest.mark.asyncio
async def test_runtime_fails_closed_for_missing_or_duplicate_resolvers() -> None:
    runtime = Runtime([RecordingResolver([])])
    effect = OtherEffect(
        id="effect-other",
        attempt=1,
        idempotency_key="other:1",
    )
    step = Step(
        id="step-other",
        run_id="run-1",
        source_revision=0,
        policy_revision="policy@1",
        kind="other",
        effects=(effect,),
    )

    with pytest.raises(MissingResolverError, match="OtherEffect"):
        await runtime.resolve_step(step)

    with pytest.raises(ValueError, match="duplicate resolver"):
        Runtime([RecordingResolver([]), RecordingResolver([])])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [ExecutionMode.SEQUENTIAL, ExecutionMode.PARALLEL],
)
async def test_runtime_preflights_all_resolvers_before_side_effects(
    mode: ExecutionMode,
) -> None:
    order: list[int] = []
    runtime = Runtime([RecordingResolver(order)])
    first = ValueEffect(
        id="first",
        attempt=1,
        idempotency_key="first:1",
        value=1,
    )
    missing = OtherEffect(
        id="missing",
        attempt=1,
        idempotency_key="missing:1",
    )
    step = Step(
        id=f"preflight-{mode.value}",
        run_id="run-1",
        source_revision=0,
        policy_revision="policy@1",
        kind="test",
        effects=(first, missing),
        execution=mode,
    )

    with pytest.raises(MissingResolverError, match="OtherEffect"):
        await runtime.resolve_step(step)

    assert order == []


@pytest.mark.asyncio
async def test_runtime_honors_pre_cancelled_token() -> None:
    runtime = Runtime([RecordingResolver([])])
    cancellation = CancellationToken()
    cancellation.cancel()

    resolutions = await runtime.resolve_step(
        make_step(ExecutionMode.SEQUENTIAL, (1, 2)),
        cancellation=cancellation,
    )

    assert all(
        resolution.status is ResolutionStatus.CANCELLED
        for resolution in resolutions
    )


@pytest.mark.asyncio
async def test_runtime_honors_cancellation_between_sequential_effects() -> None:
    runtime = Runtime([CancellingResolver()])

    resolutions = await runtime.resolve_step(
        make_step(ExecutionMode.SEQUENTIAL, (1, 2))
    )

    assert resolutions[0].status is ResolutionStatus.COMPLETED
    assert resolutions[1].status is ResolutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancellation_wait_and_task_cancellation() -> None:
    cancellation = CancellationToken()
    waiting = asyncio.create_task(cancellation.wait())
    await asyncio.sleep(0)
    assert not waiting.done()
    cancellation.cancel()
    await waiting

    runtime = Runtime([TaskCancellingResolver()])
    with pytest.raises(asyncio.CancelledError):
        await runtime.resolve_step(make_step(ExecutionMode.SEQUENTIAL, (1,)))


@pytest.mark.asyncio
async def test_runtime_cancels_an_active_resolver_task() -> None:
    resolver = BlockingResolver()
    cancellation = CancellationToken()
    task = asyncio.create_task(
        Runtime([resolver]).resolve_step(
            make_step(ExecutionMode.SEQUENTIAL, (1,)),
            cancellation=cancellation,
        )
    )
    await resolver.started.wait()

    cancellation.cancel()
    resolution = (await task)[0]

    assert resolution.status is ResolutionStatus.CANCELLED
    assert resolver.cancelled.is_set()


@pytest.mark.asyncio
async def test_parallel_failure_cancels_and_awaits_sibling_resolvers() -> None:
    resolver = BlockingResolver()
    runtime = Runtime([resolver])
    blocking = ValueEffect(
        id="blocking",
        attempt=1,
        idempotency_key="blocking:1",
        value=1,
    )
    missing = OtherEffect(
        id="missing",
        attempt=1,
        idempotency_key="missing:1",
    )
    step = Step(
        id="parallel-failure",
        run_id="run-1",
        source_revision=0,
        policy_revision="policy@1",
        kind="test",
        effects=(blocking, missing),
        execution=ExecutionMode.PARALLEL,
    )

    with pytest.raises(MissingResolverError, match="OtherEffect"):
        await runtime.resolve_step(step)

    assert not resolver.cancelled.is_set()


@pytest.mark.asyncio
async def test_parallel_resolver_cancellation_cleans_up_all_tasks() -> None:
    runtime = Runtime([TaskCancellingResolver()])

    with pytest.raises(asyncio.CancelledError):
        await runtime.resolve_step(make_step(ExecutionMode.PARALLEL, (1, 2)))


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_resolution_identity_and_mode() -> None:
    runtime = Runtime([InvalidResolver()])
    with pytest.raises(InvalidResolutionError, match="different Effect attempt"):
        await runtime.resolve_step(make_step(ExecutionMode.SEQUENTIAL, (1,)))

    invalid_step = make_step(ExecutionMode.SEQUENTIAL, (1,)).model_copy(
        update={"execution": "invalid"}
    )
    with pytest.raises(AssertionError, match="unsupported execution mode"):
        await Runtime([RecordingResolver([])]).resolve_step(invalid_step)


def test_step_rejects_duplicate_effect_ids_and_invalid_attempt() -> None:
    effect = ValueEffect(
        id="effect-1",
        attempt=1,
        idempotency_key="key",
        value=1,
    )
    with pytest.raises(ValueError, match="duplicate effect id"):
        Step(
            id="step-1",
            run_id="run-1",
            source_revision=0,
            policy_revision="policy@1",
            kind="test",
            effects=(effect, effect),
        )

    with pytest.raises(ValueError, match="attempt"):
        ValueEffect(
            id="effect-2",
            attempt=0,
            idempotency_key="key",
            value=1,
        )

    with pytest.raises(ValueError, match="at least one Effect"):
        Step(
            id="step-empty",
            run_id="run-1",
            source_revision=0,
            policy_revision="policy@1",
            kind="test",
            effects=(),
        )

    step_payload = make_step(ExecutionMode.SEQUENTIAL, (1,)).model_dump(mode="json")
    step_payload["effects"] = [
        {
            "type": "unknown",
            "id": "effect",
            "attempt": 1,
            "idempotency_key": "key",
        }
    ]
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        Step.model_validate(step_payload)

    with pytest.raises(ValueError):
        Step.model_validate({**step_payload, "effects": "not-a-sequence"})


def test_resolution_helpers_and_state_event_view() -> None:
    effect = ValueEffect(
        id="effect-1",
        attempt=1,
        idempotency_key="key",
        value=1,
    )
    denied = EffectResolution.denied(effect, "approval required")

    assert denied.status is ResolutionStatus.DENIED
    assert denied.error is not None
    assert denied.error.message == "approval required"

    with pytest.raises(ValueError):
        LoopState.model_validate(
            {
                "run_id": "run-1",
                "revision": 0,
                "status": RunStatus.READY,
                "settled": False,
                "events": "invalid",
            }
        )

    reduction = Reduction.from_cycle(
        step=make_step(ExecutionMode.SEQUENTIAL, (1,)),
        resolutions=(EffectResolution.completed(effect),),
        next_state=LoopState(
            run_id="run-1",
            revision=1,
            status=RunStatus.COMPLETED,
            settled=True,
        ),
    )
    payload = reduction.model_dump(mode="python")
    payload["events"] = "invalid"
    with pytest.raises(ValueError):
        Reduction[LoopState].model_validate(payload)
