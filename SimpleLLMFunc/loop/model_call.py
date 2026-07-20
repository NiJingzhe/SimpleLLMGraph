"""Model-call artifacts and their external resolver."""

from __future__ import annotations

import asyncio
import time
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from SimpleLLMFunc.cancellation import CancellationToken
from SimpleLLMFunc.context.ir import Completion
from SimpleLLMFunc.interface import LLM_Interface
from SimpleLLMFunc.loop.context import CompiledContext
from SimpleLLMFunc.loop.core import (
    Effect,
    EffectResolution,
    ResolutionStatus,
)
from SimpleLLMFunc.loop.revision import module_revision
from SimpleLLMFunc.loop.runtime import Resolver


class ModelCallEffect(Effect):
    type: Literal["model_call"] = "model_call"
    context: CompiledContext
    timeout: int | None = 30


class CallTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actual_model: str
    context_digest: str
    duration_seconds: float = Field(ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ModelCallResolution(EffectResolution):
    type: Literal["model_call_resolution"] = "model_call_resolution"
    completion: Completion
    trace: CallTrace


class ModelCallResolver(Resolver):
    effect_type: ClassVar[type[Effect]] = ModelCallEffect
    revision: ClassVar[str] = (
        f"model-call-resolver@sha256:{module_revision(__file__)}"
    )

    def __init__(self, llm: LLM_Interface) -> None:
        self.llm = llm

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        if not isinstance(effect, ModelCallEffect):
            raise TypeError("ModelCallResolver requires ModelCallEffect")
        if cancellation.cancelled:
            return EffectResolution.cancelled(effect)

        context = CompiledContext.model_validate(
            effect.context.model_dump(mode="python")
        )
        request = context.request
        if request.model != self.llm.model_name:
            raise ValueError(
                "Request model assertion does not match the bound LLM interface"
            )

        started = time.monotonic()
        try:
            completion = await self.llm.chat(
                request,
                trace_id=effect.id,
                timeout=effect.timeout,
                cancellation=cancellation,
            )
        except asyncio.CancelledError:
            if cancellation.cancelled:
                return EffectResolution.cancelled(effect)
            raise
        duration = time.monotonic() - started
        usage = completion.usage
        trace = CallTrace(
            actual_model=completion.model or self.llm.model_name,
            context_digest=context.digest,
            duration_seconds=duration,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
        return ModelCallResolution(
            effect_id=effect.id,
            attempt=effect.attempt,
            status=ResolutionStatus.COMPLETED,
            completion=completion,
            trace=trace,
        )
