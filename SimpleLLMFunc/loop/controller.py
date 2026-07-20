"""The single packaged driving path for Loop execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from SimpleLLMFunc.cancellation import await_with_cancellation
from SimpleLLMFunc.loop.core import InputT, LoopPolicy, ResultT, StateT, Step
from SimpleLLMFunc.loop.runtime import CancellationToken, Runtime
from SimpleLLMFunc.loop.store import InMemoryRunStore


PrepareStep = Callable[[Step], Awaitable[Step]]


class DefaultController:
    """Drive the same public cycle protocol available to application code."""

    async def run(
        self,
        loop: LoopPolicy[StateT, InputT, ResultT],
        initial_state: StateT,
        runtime: Runtime,
        store: InMemoryRunStore[StateT],
        *,
        cancellation: CancellationToken | None = None,
        prepare_step: PrepareStep | None = None,
    ) -> StateT:
        await store.create(initial_state)
        return await self.resume(
            loop,
            initial_state.run_id,
            runtime,
            store,
            cancellation=cancellation,
            prepare_step=prepare_step,
        )

    async def resume(
        self,
        loop: LoopPolicy[StateT, InputT, ResultT],
        run_id: str,
        runtime: Runtime,
        store: InMemoryRunStore[StateT],
        *,
        cancellation: CancellationToken | None = None,
        prepare_step: PrepareStep | None = None,
    ) -> StateT:
        token = cancellation or CancellationToken()
        state = await store.load(run_id)
        while loop.can_step(state):
            step = await store.pending_step(
                run_id,
                expected_revision=state.revision,
            )
            should_record = step is None
            if step is None:
                step = loop.step(state)
                loop.validate_step(state, step)
                if prepare_step is not None:
                    current_step = step
                    try:
                        step = await await_with_cancellation(
                            lambda: prepare_step(current_step),
                            token,
                        )
                    except asyncio.CancelledError:
                        if not token.cancelled:
                            raise
            loop.validate_step(state, step)
            if should_record:
                await store.record_step(step, expected_revision=state.revision)
            resolutions = await store.recorded_resolutions(step)
            if resolutions is None:
                resolutions = await runtime.resolve_step(
                    step,
                    cancellation=token,
                )
                await store.record_resolutions(step, resolutions)
            reduction = loop.reduce(state, step, resolutions)
            state = await store.commit(
                step,
                resolutions,
                reduction,
                expected_revision=state.revision,
            )
        return state
