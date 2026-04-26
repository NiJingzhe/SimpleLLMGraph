"""Tests for SelfReference memory proxy behavior."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from SimpleLLMFunc.base.react_hooks import ReActHookExecutionContext
from SimpleLLMFunc.base.context_source import DataFromSelfRef
from SimpleLLMFunc.hooks.events import CustomEvent, ReActEventType
from SimpleLLMFunc.hooks.stream import EventOrigin, EventYield
from SimpleLLMFunc.runtime.selfref import SelfReference


class _CollectingEmitter:
    """Test helper that records emitted custom events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.react_events: list[EventYield] = []

    async def emit(self, event_name: str, payload: dict[str, Any]) -> None:
        self.events.append((event_name, payload))

    async def emit_event(
        self,
        event: Any,
        *,
        origin: EventOrigin | None = None,
        origin_overrides: dict[str, Any] | None = None,
    ) -> None:
        resolved_origin = origin or EventOrigin(
            session_id="test-session",
            agent_call_id="test-agent",
            event_seq=len(self.react_events) + 1,
            fork_depth=0,
        )
        if origin_overrides:
            resolved_origin = replace(resolved_origin, **origin_overrides)
        self.react_events.append(EventYield(event=event, origin=resolved_origin))


class TestSelfReferenceMemoryProxy:
    """Validate memory-handle CRUD and helper operations."""

    def test_legacy_self_reference_module_reexports_selfreference(self) -> None:
        from SimpleLLMFunc.runtime.selfref import SelfReference as RuntimeSelfReference
        from SimpleLLMFunc.self_reference import SelfReference as LegacySelfReference

        assert LegacySelfReference is RuntimeSelfReference

    def test_memory_handle_requires_bound_key(self) -> None:
        self_reference = SelfReference()

        with pytest.raises(KeyError, match="is not bound"):
            _ = self_reference.memory["agent_main"]

    def test_memory_handle_crud(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main", [{"role": "user", "content": "hello"}]
        )

        handle = self_reference.memory["agent_main"]
        handle.append({"role": "assistant", "content": "hi"})
        handle.insert(1, {"role": "user", "content": "follow up"})
        handle.update(0, {"role": "user", "content": "hello updated"})
        handle.delete(1)

        assert handle.count() == 2
        assert handle.get(0)["content"] == "hello updated"
        assert handle.get(1)["role"] == "assistant"

    def test_system_prompt_helpers(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main", [{"role": "user", "content": "hello"}]
        )

        handle = self_reference.memory["agent_main"]
        handle.set_system_prompt("Rule A")
        handle.append_system_prompt("Rule B")

        assert handle.get_system_prompt() == "Rule A\nRule B"
        assert handle.all()[0] == {"role": "system", "content": "Rule A\nRule B"}

    def test_clear_preserves_system_prompt(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main",
            [
                {"role": "system", "content": "Rule A"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

        handle = self_reference.memory["agent_main"]
        handle.clear()

        assert handle.count() == 1
        assert handle.all() == [{"role": "system", "content": "Rule A"}]

    def test_replace_validates_tool_linkage(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [])

        handle = self_reference.memory["agent_main"]
        with pytest.raises(ValueError, match="without preceding assistant tool_calls"):
            handle.replace(
                [
                    {"role": "tool", "tool_call_id": "tc_bad", "content": "{}"},
                ]
            )

    def test_resolve_history_key_with_single_bound_key(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [])

        assert self_reference.resolve_history_key() == "agent_main"

    def test_resolve_history_key_requires_explicit_key_when_ambiguous(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [])
        self_reference.bind_history("agent_other", [])

        with pytest.raises(ValueError, match="history key is required"):
            self_reference.resolve_history_key()


class TestSelfReferenceTurnMerge:
    """Validate turn-merge behavior used by llm_chat integration."""

    def test_merge_turn_history_preserves_memory_edits(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        self_reference.memory["agent_main"].append(
            {"role": "user", "content": "[plan] keep me"}
        )

        merged = self_reference.merge_turn_history(
            key="agent_main",
            baseline_history_count=1,
            updated_history=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "seed"},
                {"role": "user", "content": "message: hello"},
                {"role": "assistant", "content": "done"},
            ],
            commit=True,
        )

        assert merged == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "seed"},
            {"role": "user", "content": "[plan] keep me"},
            {"role": "user", "content": "message: hello"},
            {"role": "assistant", "content": "done"},
        ]
        assert self_reference.snapshot_history("agent_main") == merged

    def test_merge_turn_history_prefers_runtime_system_prompt(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])
        self_reference.memory["agent_main"].set_system_prompt("runtime system")

        merged = self_reference.merge_turn_history(
            key="agent_main",
            baseline_history_count=1,
            updated_history=[
                {"role": "system", "content": "docstring system"},
                {"role": "user", "content": "seed"},
                {"role": "assistant", "content": "done"},
            ],
            commit=False,
        )

        assert merged[0] == {"role": "system", "content": "runtime system"}

    def test_merge_turn_history_drops_reasoning_details(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        merged = self_reference.merge_turn_history(
            key="agent_main",
            baseline_history_count=1,
            updated_history=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "seed"},
                {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_details": [
                        {
                            "id": "r1",
                            "type": "reasoning.text",
                            "data": "should not persist",
                        }
                    ],
                },
            ],
            commit=True,
        )

        assert merged[-1] == {"role": "assistant", "content": "done"}
        assert (
            "reasoning_details" not in self_reference.snapshot_history("agent_main")[-1]
        )

    def test_snapshot_history_reads_active_messages(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        state = ReActHookExecutionContext(
            trace_id="trace-1",
            func_name="agent",
            user_task_prompt="hello",
            messages=[
                {"role": "user", "content": "seed"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "test_tool", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '"ok"'},
            ],
            llm_kwargs={},
            stream=False,
        )

        memory_token = self_reference._set_active_memory_key("agent_main")
        state_token = self_reference._set_active_react_state(state)
        try:
            assert self_reference.snapshot_history("agent_main") == state.messages
            assert self_reference.snapshot_context_messages("agent_main") == state.messages
        finally:
            self_reference._reset_active_react_state(state_token)
            self_reference._reset_active_memory_key(memory_token)

    def test_snapshot_selfref_source_returns_parsed_source_state(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main",
            [
                {
                    "role": "system",
                    "content": "Base rules\n\n<experience>\n- [exp_1] Preference A\n</experience>",
                },
                {
                    "role": "assistant",
                    "content": (
                        "<context_compaction_summary>\n"
                        "## Goal\nGoal A\n\n"
                        "## Instruction\nInstruction A\n\n"
                        "## Discoveries\n- Discovery A\n\n"
                        "## Completed\n- Completed A\n\n"
                        "## Current Status\nStatus A\n\n"
                        "## Likely next work\n- Next A\n\n"
                        "## Relevant files/directories\n- src/a.py\n"
                        "</context_compaction_summary>"
                    ),
                },
                {"role": "user", "content": "follow-up"},
            ],
        )

        source = self_reference.snapshot_selfref_source("agent_main")

        assert isinstance(source, DataFromSelfRef)
        assert source.base_system_prompt == "Base rules"
        assert source.experiences == [{"id": "exp_1", "text": "Preference A"}]
        assert source.summary is not None
        assert source.summary["goal"] == "Goal A"
        assert source.working_messages == [{"role": "user", "content": "follow-up"}]

    def test_replace_history_updates_snapshot_selfref_source(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        self_reference.replace_history(
            "agent_main",
            [
                {
                    "role": "system",
                    "content": "Base rules\n\n<experience>\n- [exp_1] Preference B\n</experience>",
                },
                {"role": "assistant", "content": "kept summary"},
            ],
        )

        source = self_reference.snapshot_selfref_source("agent_main")

        assert source.base_system_prompt == "Base rules"
        assert source.experiences == [{"id": "exp_1", "text": "Preference B"}]
        assert source.working_messages == [{"role": "assistant", "content": "kept summary"}]


class TestSelfReferenceInstanceProxy:
    """Validate self instance binding and fork behavior."""

    @pytest.mark.asyncio
    async def test_instance_fork_requires_bound_agent_instance(self) -> None:
        self_reference = SelfReference()

        with pytest.raises(RuntimeError, match="No agent instance"):
            await self_reference.instance.fork("sub-task")

    @pytest.mark.asyncio
    async def test_instance_fork_inherits_memory_snapshot_to_child_key(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        observed_calls: list[dict[str, Any]] = []

        async def fake_agent(message: str, history=None):
            observed_calls.append(
                {
                    "message": message,
                    "history": list(history or []),
                }
            )
            yield (
                f"forked:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        fork_result = await self_reference.instance.fork("sub-task")

        assert fork_result["source_memory_key"] == "agent_main"
        assert fork_result["response"] == "forked:sub-task"
        assert fork_result["history_included"] is False
        assert "history" not in fork_result
        assert fork_result["history_count"] == 2
        assert observed_calls[0]["history"] == [{"role": "user", "content": "seed"}]

        fork_memory_key = fork_result["memory_key"]
        assert isinstance(fork_memory_key, str)
        assert fork_memory_key.startswith("agent_main::fork::")

        assert self_reference.snapshot_history("agent_main") == [
            {"role": "user", "content": "seed"}
        ]
        assert self_reference.snapshot_history(fork_memory_key) == [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "child done"},
        ]

    @pytest.mark.asyncio
    async def test_nested_fork_defaults_to_current_fork_memory_key(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history(
            "agent_main",
            [{"role": "user", "content": "root seed"}],
        )

        async def fake_agent(message: str, history=None):
            if message == "spawn-child":
                nested = await self_reference.instance.fork("leaf-task")
                yield (
                    nested,
                    list(history or []),
                )
                return

            yield (
                "leaf-done",
                [
                    *(history or []),
                    {"role": "assistant", "content": "leaf done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        parent_fork = await self_reference.instance.fork("spawn-child")
        child_fork = parent_fork["response"]

        assert parent_fork["source_memory_key"] == "agent_main"
        assert parent_fork["memory_key"].startswith("agent_main::fork::")

        assert isinstance(child_fork, dict)
        assert child_fork["source_memory_key"] == parent_fork["memory_key"]
        assert child_fork["history_included"] is False
        assert "history" not in child_fork
        assert child_fork["memory_key"].startswith(
            f"{parent_fork['memory_key']}::fork::"
        )
        assert self_reference.snapshot_history(child_fork["memory_key"])[-1] == {
            "role": "assistant",
            "content": "leaf done",
        }

    @pytest.mark.asyncio
    async def test_instance_fork_spawn_and_gather(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            await asyncio.sleep(0.01)
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        spawned = await self_reference.instance.fork_spawn("task-a")
        assert spawned["status"] == "running"
        assert spawned["fork_id"].startswith("fork_")

        completed_results = await self_reference.instance.fork_gather_all(
            spawned["fork_id"]
        )
        completed = completed_results[spawned["fork_id"]]
        assert completed["status"] == "completed"
        assert completed["response"] == "done:task-a"
        assert completed["memory_key"] == spawned["memory_key"]
        assert completed["history_included"] is False
        assert "history" not in completed
        assert completed["history_count"] == 2
        assert self_reference.snapshot_history(completed["memory_key"])[-1] == {
            "role": "assistant",
            "content": "child:task-a",
        }

    @pytest.mark.asyncio
    async def test_instance_fork_from_tool_call_context_uses_pre_fork_history(
        self,
    ) -> None:
        self_reference = SelfReference()
        parent_history = [
            {"role": "user", "content": "seed"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_fork_1",
                        "type": "function",
                        "function": {
                            "name": "execute_code",
                            "arguments": "runtime.selfref.fork.spawn('task-a')",
                        },
                    },
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "README.md"}',
                        },
                    },
                ],
            },
        ]
        self_reference.bind_history("agent_main", parent_history)

        observed_calls: list[dict[str, Any]] = []

        async def fake_agent(message: str, history=None):
            observed_calls.append(
                {
                    "message": message,
                    "history": list(history or []),
                }
            )
            return (
                "done",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        completed = await self_reference.instance.fork("task-a", include_history=True)

        assert completed["status"] == "completed"
        assert observed_calls[0]["message"] == "task-a"
        child_history = observed_calls[0]["history"]
        assert child_history == [{"role": "user", "content": "seed"}]
        assert completed["history"] == [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "child done"},
        ]

    @pytest.mark.asyncio
    async def test_instance_fork_include_history_can_be_enabled(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        completed = await self_reference.instance.fork(
            "task-a",
            include_history=True,
        )

        assert completed["status"] == "completed"
        assert completed["history_included"] is True
        assert isinstance(completed.get("history"), list)
        assert completed["history_count"] == 2
        assert completed["history"][-1] == {
            "role": "assistant",
            "content": "child:task-a",
        }

    @pytest.mark.asyncio
    async def test_instance_fork_gather_single_can_hydrate_history_on_demand(
        self,
    ) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            await asyncio.sleep(0.01)
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        spawned = await self_reference.instance.fork_spawn("task-a")

        compact_results = await self_reference.instance.fork_gather_all(
            spawned["fork_id"]
        )
        compact = compact_results[spawned["fork_id"]]
        assert compact["history_included"] is False
        assert "history" not in compact
        assert compact["history_count"] == 2
        assert compact["response"] == "done:task-a"
        assert compact["result"] == "done:task-a"

        hydrated_results = await self_reference.instance.fork_gather_all(
            spawned["fork_id"],
            include_history=True,
        )
        hydrated = hydrated_results[spawned["fork_id"]]
        assert hydrated["history_included"] is True
        assert isinstance(hydrated.get("history"), list)
        assert hydrated["history_count"] == 2
        assert hydrated["history"][-1] == {
            "role": "assistant",
            "content": "child:task-a",
        }

    @pytest.mark.asyncio
    async def test_instance_fork_gather_all_can_hydrate_history_on_demand(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            await asyncio.sleep(0.01)
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        first = await self_reference.instance.fork_spawn("task-a")
        second = await self_reference.instance.fork_spawn("task-b")
        all_results = await self_reference.instance.fork_gather_all(
            [first["fork_id"], second["fork_id"]],
            include_history=True,
        )

        assert set(all_results.keys()) == {first["fork_id"], second["fork_id"]}
        assert all(
            result["history_included"] is True for result in all_results.values()
        )
        assert all(result["history_count"] == 2 for result in all_results.values())

    @pytest.mark.asyncio
    async def test_instance_fork_gather_all_collects_spawned_children(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            await asyncio.sleep(0.01)
            return (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": f"child:{message}"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        first = await self_reference.instance.fork_spawn("task-a")
        second = await self_reference.instance.fork_spawn("task-b")

        all_results = await self_reference.instance.fork_gather_all(
            [first["fork_id"], second["fork_id"]]
        )
        assert set(all_results.keys()) == {first["fork_id"], second["fork_id"]}
        assert all(result["status"] == "completed" for result in all_results.values())
        assert all(
            result["result"] == result["response"] for result in all_results.values()
        )

    @pytest.mark.asyncio
    async def test_instance_fork_gather_all_error_payload_exposes_result_alias(
        self,
    ) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def failing_agent(message: str, history=None):
            _ = history
            raise RuntimeError(f"boom:{message}")

        self_reference.bind_agent_instance(
            failing_agent, default_memory_key="agent_main"
        )

        spawned = await self_reference.instance.fork_spawn("task-a")
        gathered = await self_reference.instance.fork_gather_all(spawned["fork_id"])
        result = gathered[spawned["fork_id"]]

        assert result["status"] == "error"
        assert result["response"] is None
        assert result["result"] is None
        assert result["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_instance_fork_emits_lifecycle_events_only(self) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            yield (
                "alpha ",
                list(history or []),
            )
            yield (
                "beta\n",
                list(history or []),
            )
            yield (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = _CollectingEmitter()

        result = await self_reference.instance.fork(
            "task-a",
            _event_emitter=emitter,
        )

        event_names = [name for name, _ in emitter.events]
        assert "selfref_fork_start" in event_names
        assert "selfref_fork_end" in event_names
        assert "selfref_fork_stream_open" not in event_names
        assert "selfref_fork_stream_delta" not in event_names
        assert "selfref_fork_stream_close" not in event_names

    @pytest.mark.asyncio
    async def test_instance_fork_emits_error_when_child_fails(
        self,
    ) -> None:
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(_message: str, history=None):
            yield (
                "partial",
                list(history or []),
            )
            raise RuntimeError("boom")

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = _CollectingEmitter()

        with pytest.raises(RuntimeError, match="boom"):
            await self_reference.instance.fork(
                "task-a",
                _event_emitter=emitter,
            )

        error_events = [
            payload for name, payload in emitter.events if name == "selfref_fork_error"
        ]
        assert len(error_events) == 1
        assert error_events[0].get("error_type") == "RuntimeError"
        assert error_events[0].get("error_message") == "boom"

    @pytest.mark.asyncio
    async def test_instance_fork_forwards_child_events(self) -> None:
        """Fork should forward child EventYield payloads as direct events."""
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        async def fake_agent(message: str, history=None):
            yield EventYield(
                event=CustomEvent(
                    event_type=ReActEventType.CUSTOM_EVENT,
                    timestamp=datetime.now(timezone.utc),
                    trace_id="trace-child",
                    func_name="agent",
                    iteration=1,
                    event_name="child_progress",
                    data={"message": message},
                )
            )
            yield (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = _CollectingEmitter()

        result = await self_reference.instance.fork(
            "task-a",
            _event_emitter=emitter,
        )

        forwarded_events = [
            event_yield.event
            for event_yield in emitter.react_events
            if isinstance(event_yield.event, CustomEvent)
        ]
        assert len(forwarded_events) == 1
        assert forwarded_events[0].event_name == "child_progress"

        forwarded_origin = emitter.react_events[0].origin
        assert forwarded_origin.fork_id == result["fork_id"]
        assert forwarded_origin.memory_key == result["memory_key"]
        assert forwarded_origin.fork_depth == result["depth"]

    @pytest.mark.asyncio
    async def test_instance_fork_preserves_nested_fork_origin_metadata(self) -> None:
        """Nested fork-origin events should keep their own fork scope metadata."""
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        nested_origin = EventOrigin(
            session_id="trace-nested",
            agent_call_id="agent-nested",
            event_seq=1,
            fork_id="fork_nested",
            fork_depth=2,
            source_memory_key="agent_main::fork::1",
            memory_key="agent_main::fork::1::fork::2",
            tool_name="execute_code",
            tool_call_id="nested-tool-call",
        )

        async def fake_agent(message: str, history=None):
            yield EventYield(
                event=CustomEvent(
                    event_type=ReActEventType.CUSTOM_EVENT,
                    timestamp=datetime.now(timezone.utc),
                    trace_id="trace-child",
                    func_name="agent",
                    iteration=1,
                    event_name="nested_progress",
                    data={"message": message},
                ),
                origin=nested_origin,
            )
            yield (
                f"done:{message}",
                [
                    *(history or []),
                    {"role": "assistant", "content": "child done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")
        emitter = _CollectingEmitter()

        result = await self_reference.instance.fork(
            "task-a",
            _event_emitter=emitter,
        )

        nested_progress_events = [
            event_yield
            for event_yield in emitter.react_events
            if isinstance(event_yield.event, CustomEvent)
            and event_yield.event.event_name == "nested_progress"
        ]
        assert len(nested_progress_events) == 1

        forwarded_origin = nested_progress_events[0].origin
        assert forwarded_origin.fork_id == nested_origin.fork_id
        assert forwarded_origin.fork_depth == nested_origin.fork_depth
        assert forwarded_origin.source_memory_key == nested_origin.source_memory_key
        assert forwarded_origin.memory_key == nested_origin.memory_key
        assert forwarded_origin.tool_call_id == nested_origin.tool_call_id
        assert forwarded_origin.fork_id != result["fork_id"]

    @pytest.mark.asyncio
    async def test_instance_fork_preserves_deep_nested_origin_through_multi_hops(
        self,
    ) -> None:
        """Deep nested fork-origin events should survive multi-hop forwarding."""
        self_reference = SelfReference()
        self_reference.bind_history("agent_main", [{"role": "user", "content": "seed"}])

        target_depth = 5
        emitter = _CollectingEmitter()

        async def fake_agent(message: str, history=None):
            current_depth = self_reference._get_active_fork_depth()
            if current_depth < target_depth:
                await self_reference.instance.fork(message, _event_emitter=emitter)
                yield (
                    f"pass-depth-{current_depth}",
                    list(history or []),
                )
                return

            active_fork_id = self_reference._get_active_fork_id()
            active_memory_key = self_reference._get_active_memory_key() or ""
            if "::fork::" in active_memory_key:
                source_memory_key = active_memory_key.rsplit("::fork::", 1)[0]
            else:
                source_memory_key = "agent_main"

            yield EventYield(
                event=CustomEvent(
                    event_type=ReActEventType.CUSTOM_EVENT,
                    timestamp=datetime.now(timezone.utc),
                    trace_id="trace-deep",
                    func_name="agent",
                    iteration=1,
                    event_name="deep_progress",
                    data={"depth": current_depth},
                ),
                origin=EventOrigin(
                    session_id="trace-deep",
                    agent_call_id="agent-deep",
                    event_seq=1,
                    fork_id=active_fork_id,
                    fork_depth=current_depth,
                    source_memory_key=source_memory_key,
                    memory_key=active_memory_key,
                ),
            )
            yield (
                "leaf-done",
                [
                    *(history or []),
                    {"role": "assistant", "content": "leaf done"},
                ],
            )

        self_reference.bind_agent_instance(fake_agent, default_memory_key="agent_main")

        top_level_result = await self_reference.instance.fork(
            "descend",
            _event_emitter=emitter,
        )

        deep_events = [
            event_yield
            for event_yield in emitter.react_events
            if isinstance(event_yield.event, CustomEvent)
            and event_yield.event.event_name == "deep_progress"
        ]
        assert len(deep_events) == 1

        forwarded_origin = deep_events[0].origin
        assert forwarded_origin.fork_depth == target_depth
        assert forwarded_origin.fork_id is not None
        assert forwarded_origin.fork_id != top_level_result["fork_id"]
        assert isinstance(forwarded_origin.memory_key, str)
        assert forwarded_origin.memory_key.startswith("agent_main::fork::")
