"""External execution of function-tool call Effects."""

from __future__ import annotations

import asyncio
import inspect
from typing import ClassVar, cast

from pydantic.types import JsonValue

from SimpleLLMFunc.cancellation import CancellationToken
from SimpleLLMFunc.loop.core import (
    Effect,
    EffectResolution,
    ResolutionStatus,
    StructuredError,
)
from SimpleLLMFunc.loop.revision import combine_revisions, module_revision
from SimpleLLMFunc.loop.runtime import Resolver
from SimpleLLMFunc.loop.tool import FunctionTool, TOOL_DECLARATION_REVISION
from SimpleLLMFunc.loop.tool_call import (
    TOOL_CALL_ARTIFACT_REVISION,
    ToolCallEffect,
    ToolCallResolution,
)
from SimpleLLMFunc.loop.tool_contract import TOOL_CONTRACT_REVISION


_TOOL_RUNTIME_REVISION = combine_revisions(
    module_revision(__file__),
    TOOL_DECLARATION_REVISION,
    TOOL_CONTRACT_REVISION,
    TOOL_CALL_ARTIFACT_REVISION,
)


class ToolCallResolver(Resolver):
    effect_type: ClassVar[type[Effect]] = ToolCallEffect
    revision: ClassVar[str] = (
        f"function-tool-resolver@sha256:{_TOOL_RUNTIME_REVISION}"
    )

    def __init__(self, tools: list[FunctionTool[..., object]]) -> None:
        self._tools: dict[str, FunctionTool[..., object]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool {tool.name}")
            self._tools[tool.name] = tool

    async def resolve(
        self,
        effect: Effect,
        cancellation: CancellationToken,
    ) -> EffectResolution:
        if not isinstance(effect, ToolCallEffect):
            raise TypeError("ToolCallResolver requires ToolCallEffect")
        if cancellation.cancelled:
            return EffectResolution.cancelled(effect)

        tool = self._tools.get(effect.name)
        if tool is None:
            return ToolCallResolution(
                effect_id=effect.id,
                attempt=effect.attempt,
                status=ResolutionStatus.DENIED,
                error=StructuredError(
                    type="UnknownTool",
                    message=f"tool {effect.name} is not registered",
                ),
                tool_call_id=effect.tool_call_id,
                name=effect.name,
            )

        try:
            interrupted = False
            if tool.is_async:
                value = tool.invoke(effect.arguments)
            else:
                thread = asyncio.create_task(
                    asyncio.to_thread(tool.invoke, effect.arguments)
                )
                try:
                    value = await asyncio.shield(thread)
                except asyncio.CancelledError:
                    interrupted = True
                    value = await thread
            if inspect.isawaitable(value):
                if interrupted:
                    pending = asyncio.ensure_future(value)
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                    raise asyncio.CancelledError
                value = await value
            normalized, result = tool.prepare_result(
                value,
                effect.event_snapshot.view,
            )
        except Exception as exc:
            return ToolCallResolution(
                effect_id=effect.id,
                attempt=effect.attempt,
                status=ResolutionStatus.FAILED,
                error=StructuredError(
                    type=type(exc).__name__,
                    message=str(exc),
                ),
                tool_call_id=effect.tool_call_id,
                name=effect.name,
            )

        return ToolCallResolution(
            effect_id=effect.id,
            attempt=effect.attempt,
            status=ResolutionStatus.COMPLETED,
            tool_call_id=effect.tool_call_id,
            name=effect.name,
            value=cast(JsonValue, normalized),
            result=result,
            tool_revision=tool.revision,
            event_digest=effect.event_snapshot.digest,
        )
