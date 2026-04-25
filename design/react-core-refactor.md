# ReAct Core Refactor Boundary Plan

## Status

- Branch: `refactor/react-core`
- Status: design document before large-scale refactor
- Scope: `SimpleLLMFunc/base/*`, `llm_decorator/*`, and runtime integration points that currently couple ReAct orchestration with selfref, tool execution, events, and cancellation

## Motivation

The current ReAct implementation has grown too large and too entangled.

The main problems are:

1. `SimpleLLMFunc/base/ReAct.py` is doing too many jobs at once.
2. Context mutation currently happens in more than one place.
3. `selfref` currently couples tightly to internal ReAct execution state.
4. Tool execution, primitive execution, event emission, and context mutation are not separated cleanly enough.
5. Abort handling exists, but the correct boundary for abort-driven partial results and mutations is not yet encoded as a first-class architecture rule.

This refactor is intended to reduce complexity by making the core loop small, explicit, and boundary-driven.

## Current Codebase Pain Points

The current implementation has several practical problems beyond architectural aesthetics.

1. `ReAct.py` has grown too large to serve as a reliable place for further iteration.
2. many internal responsibilities are currently implemented as closure functions whose names already suggest module-level ownership rather than local ownership.
3. some lower-level capabilities are already encapsulated elsewhere, but `ReAct.py` still duplicates orchestration-adjacent logic around them.
4. this makes it too easy for each refactor step to reduce one kind of complexity while introducing another.

This document therefore treats file-size reduction and boundary clarification as linked goals, not separate cleanups.

## Stable Compatibility Boundaries

This refactor is intended to be a full internal rewrite of the ReAct core, but it must preserve two external contract layers.

### 1. LLM Interface Layer Stays Stable

The existing `LLM_Interface` contract is a fixed lower boundary for this refactor.

That means the refactor must continue to work against the current interface shape:

1. `LLM_Interface.chat(...)`
2. `LLM_Interface.chat_stream(...)`
3. current message payload shape expected by these methods
4. current stream versus non-stream behavior at that boundary

The refactor must not require changes to provider adapters or the lower LLM transport layer.

### 2. Decorator Step Layer Stays Stable

`SimpleLLMFunc/llm_decorator/steps/` is the fixed upper boundary for this refactor.

That means the refactor must preserve the behavior expected by the current step layer, including:

1. `llm_decorator/steps/chat/react.py`
2. `llm_decorator/steps/function/react.py`
3. current step-level ability to derive legacy return values from the underlying stream when needed
4. current `ReactOutput`-based behavior as the only core output mode
5. current compatibility expectations around `ReAct_loop(...)` or equivalent entrypoints used by the step layer

The ReAct core can be rewritten internally, but it must continue to present the same usable contract to the step layer.

### 3. Hooks, Event Stream, And Langfuse Stay Supported

Even if the core is fully rewritten, the following integration surfaces must continue to work.

1. internal ReAct hooks
2. event stream support
3. Langfuse tracing support

This means the refactor is not allowed to simplify itself by dropping these capabilities.

## Core Boundary

The most important architecture rule is:

> Context must only be modified during context compilation.

This means:

1. `single_llm_call` must not directly mutate context.
2. `tool_scheduler` must not directly mutate context.
3. primitives must not directly mutate context.
4. tools and primitives should return mutations, not directly edit message history.
5. the ReAct loop must only carry pending mutations between phases.
6. `compile_context(...)` is the only place allowed to apply mutations and produce the next context view.

If any component directly edits the live message list outside compile, that violates the intended boundary.

## Target ReAct Loop

The intended ReAct loop should conceptually be:

```python
while True:
    compiled = compile_context(base_state, pending_mutations)

    single_call_result = await single_llm_call(compiled.llm_messages)
    pending_mutations.extend(single_call_result.mutations)

    if not single_call_result.tool_requests:
        break

    tool_result = await tool_scheduler(single_call_result.tool_requests)
    pending_mutations.extend(tool_result.mutations)
```

Expanded in words:

1. compile context and apply mutations from the last round
2. call the LLM once
3. collect pending mutations from the assistant side
4. schedule and execute tools in parallel if needed
5. collect pending mutations from tool results
6. repeat until no tool work remains

This loop must stay small.

## Explicit Non-Goals For The Loop

The loop itself should not do any of the following directly:

1. manually splice message lists together
2. directly edit context because a tool or primitive decided to do so
3. own selfref-specific compaction logic
4. own provider-specific protocol parsing details
5. own event formatting details beyond pushing events into the event queue
6. own UI-facing rendering concerns

## Architectural Components

The refactor should separate the current large implementation into the following conceptual modules.

### 1. `react_loop.py`

Responsibilities:

1. orchestrate the loop
2. call `compile_context(...)`
3. call `single_llm_call(...)`
4. call `tool_scheduler(...)`
5. push loop-level lifecycle events
6. stop on completion or abort

Non-responsibilities:

1. protocol parsing
2. direct context mutation
3. selfref-specific state mutation
4. tool result formatting

### 2. `context_compile.py`

Responsibilities:

1. define the canonical context state
2. apply pending mutations
3. produce the next LLM-safe message list
4. produce semantic and final display views if needed
5. become the only legal place where context is changed

This is where selfref should integrate in a looser way.

### 3. `mutation.py`

Responsibilities:

1. define mutation types as first-class structured objects
2. provide a stable boundary between producers and context compilation

Example categories:

1. `AssistantContentMutation`
2. `ToolCallRequestMutation`
3. `ToolResultMutation`
4. `ToolCancelledMutation`
5. `AssistantTruncatedMutation`
6. `ContextSummaryMutation`
7. `ExperienceRememberMutation`
8. `ExperienceForgetMutation`
9. `ReplaceHistoryMutation`

The exact set may evolve, but the important rule is that these are structured mutations, not ad hoc message list edits.

### 4. `llm_call.py`

Responsibilities:

1. execute one model call
2. parse streamed or non-streamed model output
3. emit LLM-call-level events
4. return structured Python results and assistant-side mutations

It should not mutate context directly.

### 5. `tool_scheduler.py`

Responsibilities:

1. schedule tool calls, including parallel execution
2. emit tool lifecycle events
3. aggregate tool results into mutations
4. handle tool-stage cancellation correctly

It should not mutate context directly.

### 6. `event_queue.py`

Responsibilities:

1. provide one event queue abstraction for the loop and tools to push into
2. expose a single async consumer stream outward

The desired model is:

1. ReAct loop pushes iteration and lifecycle events
2. single-call execution pushes model events
3. tool scheduler pushes tool events
4. outer consumers pull events with `async for`

## Context State Versus Messages

One of the root issues in the current implementation is that message history is treated as both:

1. the source of truth
2. the transport format sent to the model

The refactor should separate these.

Proposed direction:

```python
@dataclass
class ContextState:
    ...

@dataclass
class CompiledContext:
    llm_messages: MessageList
    semantic_messages: MessageList
```

Where:

1. `ContextState` is the source of truth
2. `CompiledContext.llm_messages` is the protocol-safe model input
3. `CompiledContext.semantic_messages` is the semantic view used for hooks or finalization when needed

The implementation does not have to use exactly these names, but this separation should exist.

## Selfref Integration Direction

### Current Problem

`selfref` is currently too aware of live ReAct execution state. It reaches into active history synchronization through hooks and runtime coordination.

This works, but it makes the coupling too strong.

### Desired Direction

`selfref` should integrate by contributing mutations and by participating in context compilation, instead of directly mutating live ReAct message history.

In other words:

1. selfref operations should become mutation producers
2. compile-time logic should apply those mutations
3. the ReAct loop should not need deep awareness of selfref internals

Examples:

1. `context.compact(...)` should become a summary mutation producer
2. `remember(...)` should become an experience mutation producer
3. history replacement should become a structured mutation rather than a direct live edit wherever possible

This does not mean selfref becomes passive. It means selfref moves to a cleaner boundary: mutation production plus compile participation.

## Tool And Primitive Boundary

Tools and primitives should follow the same architectural rule.

They are both effect producers.

That means:

1. a tool should return mutations, not directly patch context
2. a primitive should return mutations, not directly patch context
3. their outputs may still include normal result payloads, but any context-changing effect must be encoded as mutation

This is especially important for runtime primitives like selfref operations.

## Event Model

The desired event model is queue-based.

Requirements:

1. tools should push events into the event queue
2. the ReAct loop should push events into the same event queue
3. outer consumers should consume by pulling with `async for`
4. event producers should not own outer consumption logic

This should allow a cleaner and more explicit split between event production and event consumption.

## Must Preserve Integration Surfaces

The following behaviors are required compatibility targets during the rewrite.

### Hooks

The internal hook lifecycle must remain supported.

Current hook points include:

1. `on_run_start`
2. `before_llm_call`
3. `after_llm_call`
4. `before_tool_batch`
5. `after_tool_batch`
6. `before_finalize`

The refactor may change the internal implementation, but it should preserve the existence and usefulness of these lifecycle hooks unless there is a deliberate, separately documented migration plan.

### Event Stream

The new core is event-stream-only.

That means the core should expose `ReactOutput` to outer consumers through async iteration, and it should not preserve an internal split between event mode and non-event mode.

If upper layers still need legacy final-value behavior, they should derive that behavior by consuming the `ReactOutput` stream rather than by asking the core to run in a different output mode.

This includes preserving support for:

1. `EventYield`
2. `ResponseYield`
3. current outer-consumer usage through `async for`
4. compatibility with existing TUI and event-stream consumers

Internally, the event production model may be rewritten around a queue, but externally it must still behave like the current stream contract.

### Langfuse Tracing

Langfuse support must remain first-class.

The refactor must continue to support:

1. top-level loop tracing
2. LLM call observations
3. tool execution observations
4. trace-context propagation across nested operations
5. compatibility with current `langfuse_client.start_as_current_observation(...)` usage style

The rewrite should preserve tracing fidelity while moving responsibility into cleaner modules.

## Abort Semantics

Abort handling must follow the same mutation boundary rule.

### Rule

Abort should not just stop execution. Abort should produce mutations that can be compiled into the next context.

### Abort Checkpoints

There are two required checkpoints:

1. `single_llm_call` must be cancellable
2. `tool_scheduler` must be cancellable

### Single Call Abort

If cancellation happens during a single LLM call:

1. partial assistant content already received must not be dropped
2. the result should produce a truncation mutation
3. the truncation mutation must include the abort reason

Conceptually:

```text
<Truncated due to cancel from user. Reason: ...>
```

But this should be carried as a structured mutation, not hardcoded into the loop.

### Tool Scheduler Abort

If cancellation happens during tool execution:

1. tools that already completed should produce normal tool-result mutations
2. tools that have not completed should produce cancellation mutations
3. cancellation mutations must include the abort reason

Conceptually:

```text
<Tool execution cancelled by user. Reason: ...>
```

Again, this should be a structured mutation, not a raw text shortcut inside the loop.

### Compile Responsibility

Compile is responsible for deciding how these abort mutations become visible context.

That keeps the boundary consistent:

1. execution units stop and produce mutations
2. compile transforms mutations into the next context

## Planned Migration Strategy

This refactor should not proceed by continuing to grow the current `ReAct.py` file.

The migration strategy should be:

1. define the new contracts first
2. introduce new modules alongside the old implementation
3. move behavior incrementally behind the new contracts
4. reduce the old `ReAct.py` into a compatibility wrapper as the last step

### Phase 1: Contracts

Introduce:

1. mutation types
2. context state types
3. compiled context types
4. single-call result types
5. tool-scheduler result types

### Phase 2: New Core Skeleton

Introduce new internal modules, likely:

1. `base/mutation.py`
2. `base/context_compile.py`
3. `base/llm_call.py`
4. `base/tool_scheduler.py`
5. `base/react_loop.py`

These can initially be thin wrappers while contracts stabilize.

### Phase 3: Single Call Migration

Move single-call logic out of `ReAct.py` so that it only returns structured parsed results plus mutations.

### Phase 4: Tool Scheduler Migration

Move tool scheduling and tool-result aggregation out of `ReAct.py` so it returns tool mutations only.

### Phase 5: Context Compiler Migration

Move all context mutation application and view generation into `compile_context(...)`.

### Phase 6: Selfref Integration Cleanup

Move selfref to mutation-producing and compile-time participation wherever feasible, reducing direct live-state coupling.

### Phase 7: Compatibility Cleanup

Reduce the old `ReAct.py` file into:

1. compatibility entrypoints
2. imports of the new implementation
3. minimal transition glue only where necessary

## Immediate Refactor Principles

During implementation, the following rules should be enforced:

1. do not continue adding new closure-heavy local helpers to `ReAct.py`
2. do not introduce more live message list mutation outside compile
3. prefer creating new modules over adding new architectural responsibilities into the current file
4. keep temporary compatibility shims small and clearly marked
5. move toward fewer sources of truth, not more

## Success Criteria

This refactor is successful when the following statements are true:

1. the core loop is short and easy to read
2. context is only changed inside compile
3. tools and primitives return mutations instead of editing history directly
4. selfref participates through a looser compile-oriented boundary
5. abort produces structured mutations instead of special-case ad hoc behavior
6. event production is queue-based and consistent across loop and tools
7. `ReAct.py` is no longer the main container for all orchestration details

## Open Questions To Resolve During Implementation

These are implementation questions, not boundary questions.

1. exact mutation taxonomy and naming
2. whether one or more compiled context views are needed beyond `llm_messages` and `semantic_messages`
3. how much legacy hook compatibility should be preserved during migration
4. whether any selfref operations must remain direct state edits temporarily during transition
5. how event queue buffering and shutdown should work under heavy parallel tool execution

These should be answered in follow-up implementation notes, but they do not change the high-level boundary defined above.

## Summary

The refactor direction is:

1. shrink the ReAct loop into a clean orchestrator
2. make compile the only context mutation boundary
3. turn tools, primitives, aborts, and selfref operations into mutation producers
4. unify event production behind one queue
5. move complexity out of `ReAct.py` and into explicit modules with clear responsibilities

This document defines the intended boundary before the code migration proceeds.
