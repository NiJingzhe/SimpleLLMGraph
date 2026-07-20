"""Revisioned in-memory Store for complete Loop cycle persistence."""

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, SerializeAsAny, field_validator

from SimpleLLMFunc.loop.core import (
    EffectResolution,
    LoopState,
    Reduction,
    Step,
    restore_artifact_subtype,
)


class RevisionConflictError(RuntimeError):
    """Raised when a Step or Reduction targets a stale State revision."""


class InvalidReductionError(ValueError):
    """Raised when a Reduction violates append-only State invariants."""


class RunJournal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: tuple[Step, ...] = ()
    resolutions: tuple[SerializeAsAny[EffectResolution], ...] = ()

    @field_validator("resolutions", mode="before")
    @classmethod
    def restore_resolutions(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        items = cast(list[object] | tuple[object, ...], value)
        return tuple(
            restore_artifact_subtype(EffectResolution, item) for item in items
        )


StateT = TypeVar("StateT", bound=LoopState)


class InMemoryRunStore(Generic[StateT]):
    """A complete reference Store with compare-and-swap commit semantics."""

    def __init__(self) -> None:
        self._states: dict[str, StateT] = {}
        self._steps: dict[str, dict[str, Step]] = {}
        self._step_by_revision: dict[str, dict[int, str]] = {}
        self._resolutions: dict[str, dict[str, tuple[EffectResolution, ...]]] = {}
        self._lock = asyncio.Lock()

    async def create(self, state: StateT) -> None:
        async with self._lock:
            if state.run_id in self._states:
                raise ValueError(f"run {state.run_id} already exists")
            if state.revision != 0:
                raise ValueError("a new run must start at revision 0")
            self._states[state.run_id] = state.model_copy(deep=True)
            self._steps[state.run_id] = {}
            self._step_by_revision[state.run_id] = {}
            self._resolutions[state.run_id] = {}

    async def load(self, run_id: str) -> StateT:
        async with self._lock:
            state = self._states.get(run_id)
            if state is None:
                raise KeyError(f"run {run_id} is missing")
            return state.model_copy(deep=True)

    async def record_step(self, step: Step, *, expected_revision: int) -> None:
        async with self._lock:
            state = self._require_state(step.run_id)
            self._require_revision(state, expected_revision)
            if step.source_revision != expected_revision:
                raise RevisionConflictError(
                    "Step source revision does not match expected revision"
                )
            if step.id in self._steps[step.run_id]:
                raise ValueError(f"Step {step.id} is already recorded")
            previous = self._step_by_revision[step.run_id].get(expected_revision)
            if previous is not None:
                raise ValueError(
                    f"revision {expected_revision} already has Step {previous}"
                )
            self._steps[step.run_id][step.id] = step.model_copy(deep=True)
            self._step_by_revision[step.run_id][expected_revision] = step.id

    async def record_resolutions(
        self,
        step: Step,
        resolutions: tuple[EffectResolution, ...],
    ) -> None:
        async with self._lock:
            recorded = self._steps.get(step.run_id, {}).get(step.id)
            if recorded is None or recorded != step:
                raise ValueError("Step must be recorded before its Resolutions")
            if step.id in self._resolutions[step.run_id]:
                raise ValueError(f"Resolutions for Step {step.id} are already recorded")
            expected = [(effect.id, effect.attempt) for effect in step.effects]
            actual = [
                (resolution.effect_id, resolution.attempt)
                for resolution in resolutions
            ]
            if expected != actual:
                raise ValueError("Resolutions do not match Step Effects")
            self._resolutions[step.run_id][step.id] = tuple(
                resolution.model_copy(deep=True) for resolution in resolutions
            )

    async def pending_step(
        self,
        run_id: str,
        *,
        expected_revision: int,
    ) -> Step | None:
        """Return the Step already recorded for the current State revision."""

        async with self._lock:
            state = self._require_state(run_id)
            self._require_revision(state, expected_revision)
            step_id = self._step_by_revision[run_id].get(expected_revision)
            if step_id is None:
                return None
            return self._steps[run_id][step_id].model_copy(deep=True)

    async def recorded_resolutions(
        self,
        step: Step,
    ) -> tuple[EffectResolution, ...] | None:
        """Return a recorded terminal Resolution batch for an exact Step."""

        async with self._lock:
            recorded = self._steps.get(step.run_id, {}).get(step.id)
            if recorded is None or recorded != step:
                raise ValueError("Step must be recorded before loading Resolutions")
            resolutions = self._resolutions[step.run_id].get(step.id)
            if resolutions is None:
                return None
            return tuple(
                resolution.model_copy(deep=True) for resolution in resolutions
            )

    async def commit(
        self,
        step: Step,
        resolutions: tuple[EffectResolution, ...],
        reduction: Reduction[StateT],
        *,
        expected_revision: int,
    ) -> StateT:
        async with self._lock:
            next_state = reduction.next_state
            current = self._require_state(step.run_id)
            self._require_revision(current, expected_revision)
            if step.source_revision != expected_revision:
                raise RevisionConflictError(
                    "Step source revision does not match expected revision"
                )
            if not reduction.matches_cycle(step, resolutions):
                raise InvalidReductionError(
                    "Reduction is not bound to the supplied Step and Resolutions"
                )
            self._validate_reduction(current, reduction, expected_revision)

            recorded_step = self._steps[step.run_id].get(step.id)
            if recorded_step != step:
                raise InvalidReductionError(
                    "the exact Step must be recorded before committing a Reduction"
                )
            recorded_resolutions = self._resolutions[step.run_id].get(step.id)
            if recorded_resolutions is None:
                raise InvalidReductionError(
                    "Resolutions must be recorded before committing a Reduction"
                )
            if recorded_resolutions != resolutions:
                raise InvalidReductionError(
                    "commit Resolutions do not match the recorded batch"
                )

            stored = next_state.model_copy(deep=True)
            self._states[step.run_id] = stored
            return stored.model_copy(deep=True)

    async def journal(self, run_id: str) -> RunJournal:
        async with self._lock:
            self._require_state(run_id)
            steps = tuple(self._steps[run_id].values())
            resolutions = tuple(
                resolution
                for step in steps
                for resolution in self._resolutions[run_id].get(step.id, ())
            )
            return RunJournal(
                steps=tuple(step.model_copy(deep=True) for step in steps),
                resolutions=tuple(
                    resolution.model_copy(deep=True)
                    for resolution in resolutions
                ),
            )

    def _require_state(self, run_id: str) -> StateT:
        state = self._states.get(run_id)
        if state is None:
            raise KeyError(f"run {run_id} is missing")
        return state

    @staticmethod
    def _require_revision(state: LoopState, expected_revision: int) -> None:
        if state.revision != expected_revision:
            raise RevisionConflictError(
                f"expected revision {expected_revision}, found {state.revision}"
            )

    @staticmethod
    def _validate_reduction(
        current: StateT,
        reduction: Reduction[StateT],
        expected_revision: int,
    ) -> None:
        next_state = reduction.next_state
        if next_state.run_id != current.run_id:
            raise InvalidReductionError("Reduction changed run_id")
        if next_state.revision != expected_revision + 1:
            raise InvalidReductionError("Reduction must advance revision by exactly 1")
        prefix = next_state.events[: len(current.events)]
        if prefix != current.events:
            raise InvalidReductionError("Reduction must append to existing Events")
        appended = next_state.events[len(current.events) :]
        if appended != reduction.events:
            raise InvalidReductionError(
                "Reduction Events must equal the appended State Events"
            )
        event_ids = [event.id for event in next_state.events]
        if len(event_ids) != len(set(event_ids)):
            raise InvalidReductionError("Reduction introduced a duplicate Event id")
