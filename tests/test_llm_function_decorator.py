"""Tests for llm_function decorator defaults and orchestration."""

from __future__ import annotations

import contextvars
import inspect
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from SimpleLLMFunc import llm_function
from SimpleLLMFunc.hooks.stream import ResponseYield
from SimpleLLMFunc.observability.langfuse_client import (
    langfuse_client as shared_langfuse_client,
)


class _DummyObservation:
    """Simple context manager used to stub Langfuse observations."""

    def __enter__(self) -> "_DummyObservation":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        return None


class _TrackingObservation:
    def __init__(
        self,
        tracker: "_TrackingLangfuseClient",
        record: dict[str, Any],
    ) -> None:
        self._tracker = tracker
        self._record = record
        self._trace_token: Optional[contextvars.Token[Optional[str]]] = None
        self._observation_token: Optional[contextvars.Token[Optional[str]]] = None

    def __enter__(self) -> "_TrackingObservation":
        self._trace_token = self._tracker._trace_id_var.set(self._record["trace_id"])
        self._observation_token = self._tracker._observation_id_var.set(
            self._record["span_id"]
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._observation_token is not None:
            self._tracker._observation_id_var.reset(self._observation_token)
        if self._trace_token is not None:
            self._tracker._trace_id_var.reset(self._trace_token)
        return None

    def update(self, **kwargs: Any) -> None:
        self._record.setdefault("updates", []).append(kwargs)

    def set_attribute(self, key: str, value: Any) -> None:
        self._record.setdefault("attributes", {})[key] = value

    def is_recording(self) -> bool:
        return True


class _TrackingLangfuseClient:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._counter = 0
        self.trace_names_by_trace_id: dict[str, str] = {}
        self._trace_id_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar("test_langfuse_trace_id_function", default=None)
        )
        self._observation_id_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar(
                "test_langfuse_observation_id_function", default=None
            )
        )
        self._trace_name_var: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar("test_langfuse_trace_name_function", default=None)
        )

    def start_as_current_observation(self, **kwargs: Any) -> _TrackingObservation:
        self._counter += 1
        trace_context = kwargs.get("trace_context")
        current_trace_id = self._trace_id_var.get()
        trace_id = ""
        if isinstance(trace_context, dict):
            raw_trace_id = trace_context.get("trace_id")
            if isinstance(raw_trace_id, str):
                trace_id = raw_trace_id
        if not trace_id:
            trace_id = current_trace_id or f"trace_{self._counter}"

        parent_span_id: Optional[str] = None
        if isinstance(trace_context, dict):
            raw_parent_span_id = trace_context.get("parent_span_id")
            if isinstance(raw_parent_span_id, str) and raw_parent_span_id:
                parent_span_id = raw_parent_span_id
        if parent_span_id is None:
            parent_span_id = self._observation_id_var.get()

        record = {
            "span_id": f"obs_{self._counter}",
            "as_type": kwargs.get("as_type"),
            "name": kwargs.get("name"),
            "input": kwargs.get("input"),
            "metadata": kwargs.get("metadata"),
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "trace_context": trace_context,
            "trace_name": self._trace_name_var.get(),
        }
        self.records.append(record)
        return _TrackingObservation(self, record)

    def propagate_attributes(self, *, trace_name: Optional[str] = None, **kwargs: Any):
        _ = kwargs
        tracker = self

        class _Context:
            def __enter__(self_nonlocal):
                self_nonlocal._token = tracker._trace_name_var.set(trace_name)
                trace_id = tracker._trace_id_var.get()
                if trace_id and trace_name:
                    tracker.trace_names_by_trace_id[trace_id] = trace_name
                return None

            def __exit__(self_nonlocal, exc_type: Any, exc: Any, tb: Any) -> None:
                tracker._trace_name_var.reset(self_nonlocal._token)
                return None

        return _Context()

    def get_current_observation_id(self) -> str:
        return self._observation_id_var.get() or ""

    def _active_observation(self) -> Optional[_TrackingObservation]:
        current_observation_id = self._observation_id_var.get()
        if not current_observation_id:
            return None

        for record in reversed(self.records):
            if record.get("span_id") == current_observation_id:
                return _TrackingObservation(self, record)

        return None


def test_llm_function_default_max_tool_calls_is_none() -> None:
    """llm_function should not impose a default tool-call limit."""

    signature = inspect.signature(llm_function)

    assert signature.parameters["max_tool_calls"].default is None


@pytest.mark.asyncio
async def test_llm_function_passes_none_max_tool_calls_to_react_loop() -> None:
    """llm_function should forward the unbounded default into ReAct orchestration."""

    captured: dict[str, Any] = {}
    raw_response = object()

    async def fake_react_loop(*args: Any, **kwargs: Any):
        _ = args
        captured["max_tool_calls"] = kwargs.get("max_tool_calls")
        yield ResponseYield(type="response", response=raw_response, messages=[])

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.ReAct_loop",
            new=fake_react_loop,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.process_response",
            return_value="parsed-result",
        ) as mock_parse,
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.langfuse_client.start_as_current_observation",
            return_value=_DummyObservation(),
        ),
    ):

        @llm_function(llm_interface=mock_llm)
        async def summarize(text: str) -> str:
            """Summarize input text."""

        result = await summarize("hello")

    assert captured["max_tool_calls"] is None
    assert result == "parsed-result"
    mock_parse.assert_called_once_with(raw_response, str)


@pytest.mark.asyncio
async def test_llm_function_sets_explicit_langfuse_trace_name() -> None:
    tracker = _TrackingLangfuseClient()
    raw_response = object()

    async def fake_react_loop(*args: Any, **kwargs: Any):
        _ = args, kwargs
        yield ResponseYield(type="response", response=raw_response, messages=[])

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.ReAct_loop",
            new=fake_react_loop,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.process_response",
            return_value="parsed-result",
        ),
        patch.object(
            shared_langfuse_client,
            "start_as_current_observation",
            side_effect=tracker.start_as_current_observation,
        ),
        patch.object(
            shared_langfuse_client,
            "get_current_observation_id",
            side_effect=tracker.get_current_observation_id,
        ),
        patch(
            "SimpleLLMFunc.observability.langfuse_client.otel_trace_api.get_current_span",
            side_effect=tracker._active_observation,
        ),
    ):

        @llm_function(llm_interface=mock_llm)
        async def summarize(text: str) -> str:
            """Summarize input text."""

        result = await summarize("hello")

    function_span = next(
        record
        for record in tracker.records
        if record.get("name") == "summarize_function_call"
    )

    assert result == "parsed-result"
    assert function_span["attributes"]["langfuse.trace.name"] == "summarize"


@pytest.mark.asyncio
async def test_llm_function_propagates_trace_name_to_child_observations() -> None:
    tracker = _TrackingLangfuseClient()
    raw_response = object()

    async def fake_react_loop(*args: Any, **kwargs: Any):
        _ = args, kwargs
        with tracker.start_as_current_observation(
            as_type="generation",
            name="downstream_generation",
        ):
            yield ResponseYield(type="response", response=raw_response, messages=[])

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"

    with (
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.ReAct_loop",
            new=fake_react_loop,
        ),
        patch(
            "SimpleLLMFunc.llm_decorator.llm_function_decorator.process_response",
            return_value="parsed-result",
        ),
        patch.object(
            shared_langfuse_client,
            "start_as_current_observation",
            side_effect=tracker.start_as_current_observation,
        ),
        patch.object(
            shared_langfuse_client,
            "get_current_observation_id",
            side_effect=tracker.get_current_observation_id,
        ),
        patch(
            "SimpleLLMFunc.observability.langfuse_client.otel_trace_api.get_current_span",
            side_effect=tracker._active_observation,
        ),
        patch(
            "SimpleLLMFunc.observability.langfuse_client.langfuse_propagate_attributes",
            side_effect=tracker.propagate_attributes,
        ),
    ):

        @llm_function(llm_interface=mock_llm)
        async def summarize(text: str) -> str:
            """Summarize input text."""

        result = await summarize("hello")

    downstream_generation = next(
        record
        for record in tracker.records
        if record.get("name") == "downstream_generation"
    )

    assert result == "parsed-result"
    assert downstream_generation["trace_name"] == "summarize"
    assert (
        tracker.trace_names_by_trace_id[downstream_generation["trace_id"]]
        == "summarize"
    )
