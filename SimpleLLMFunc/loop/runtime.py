"""Effect resolver registry and Step-aware scheduling."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import ClassVar

from SimpleLLMFunc.cancellation import CancellationToken
from SimpleLLMFunc.loop.core import (
    Effect,
    EffectResolution,
    ExecutionMode,
    Step,
)


class MissingResolverError(LookupError):
    """Raised when no resolver is registered for an Effect type."""


class InvalidResolutionError(ValueError):
    """Raised when a resolver returns an outcome for another Effect attempt."""


class Resolver(ABC):
    """Resolve one concrete Effect type."""

    effect_type: ClassVar[type[Effect]]

    @abstractmethod
    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        """Return one terminal Resolution for the supplied Effect attempt."""


class Runtime:
    """Resolve every Effect in a Step according to its execution semantics."""

    def __init__(self, resolvers: list[Resolver]) -> None:
        self._resolvers: dict[type[Effect], Resolver] = {}
        for resolver in resolvers:
            effect_type = resolver.effect_type
            if effect_type in self._resolvers:
                raise ValueError(f"duplicate resolver for {effect_type.__name__}")
            self._resolvers[effect_type] = resolver

    async def resolve_step(
        self,
        step: Step,
        *,
        cancellation: CancellationToken | None = None,
    ) -> tuple[EffectResolution, ...]:
        token = cancellation or CancellationToken()
        missing = next(
            (
                effect
                for effect in step.effects
                if type(effect) not in self._resolvers
            ),
            None,
        )
        if missing is not None:
            raise MissingResolverError(
                f"no resolver registered for {type(missing).__name__}"
            )
        if token.cancelled:
            return tuple(
                EffectResolution.cancelled(effect) for effect in step.effects
            )

        if step.execution is ExecutionMode.SEQUENTIAL:
            resolutions: list[EffectResolution] = []
            for effect in step.effects:
                if token.cancelled:
                    resolutions.append(EffectResolution.cancelled(effect))
                else:
                    resolutions.append(await self._resolve_one(effect, token))
            return tuple(resolutions)

        if step.execution is ExecutionMode.PARALLEL:
            tasks = [
                asyncio.create_task(self._resolve_one(effect, token))
                for effect in step.effects
            ]
            try:
                return tuple(await asyncio.gather(*tasks))
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        raise AssertionError(f"unsupported execution mode: {step.execution}")

    async def _resolve_one(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        resolver = self._resolvers[type(effect)]
        resolve_task = asyncio.create_task(resolver.resolve(effect, cancellation))
        cancellation_task = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {resolve_task, cancellation_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if resolve_task not in done:
                # Give token-aware resolvers one turn to publish a result that
                # completed concurrently with the cancellation request.
                await asyncio.sleep(0)
            if not resolve_task.done():
                resolve_task.cancel()
                await asyncio.gather(resolve_task, return_exceptions=True)
            try:
                resolution = resolve_task.result()
            except asyncio.CancelledError:
                if cancellation.cancelled:
                    return EffectResolution.cancelled(effect)
                raise
            except Exception as exc:
                return EffectResolution.failed(effect, exc)
        finally:
            cancellation_task.cancel()
            resolve_task.cancel()
            await asyncio.gather(
                resolve_task,
                cancellation_task,
                return_exceptions=True,
            )

        if (
            resolution.effect_id != effect.id
            or resolution.attempt != effect.attempt
        ):
            raise InvalidResolutionError(
                "resolver returned a Resolution for a different Effect attempt"
            )
        return resolution
