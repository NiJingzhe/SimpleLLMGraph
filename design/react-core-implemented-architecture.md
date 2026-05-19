# ReAct Core Implemented Architecture

## Purpose

This document describes the architecture that is now actually implemented in the codebase after the ReAct core rewrite.

It is different from the refactor plan document.

- `design/react-core-refactor.md` explains the target boundary and migration plan.
- this document explains the concrete module layout, responsibilities, and data flow that now exist in the repository.

## Current High-Level Shape

The core is now structured around five main internal modules.

1. `SimpleLLMFunc/base/mutation.py`
2. `SimpleLLMFunc/base/context_compile.py`
3. `SimpleLLMFunc/base/llm_call.py`
4. `SimpleLLMFunc/base/tool_scheduler.py`
5. `SimpleLLMFunc/base/react_loop.py`

`SimpleLLMFunc/base/ReAct.py` is now a thin compatibility wrapper.

## Main Rule

The implemented core follows one main rule:

> context changes are applied through compile-time mutation application, not by letting tools or selfref directly rewrite the live ReAct loop context.

This rule is fully implemented for the main ReAct turn flow, including selfref context compaction.

## Module Responsibilities

### `base/mutation.py`

This module defines structured mutation objects.

The current mutation taxonomy includes:

1. `AssistantMessageMutation`
2. `ToolResultMutation`
3. `UserMessageMutation`
4. `ContextReplaceMutation`
5. `ContextSummaryMutation`
6. `ExperienceRememberMutation`
7. `ExperienceForgetMutation`
8. `AssistantTruncatedMutation`
9. `ToolCancelledMutation`

These mutations are the data boundary between producers and context compilation.

### `base/context_compile.py`

This module owns compile-time context transformation.

Its responsibilities are:

1. define `ContextState`
2. define `CompiledContext`
3. apply mutations to message history
4. produce model-facing messages
5. preserve one protocol-valid compiled history for hooks and finalize

Important exported pieces:

1. `ContextState`
2. `CompiledContext`
3. `compile_context(...)`
4. `apply_mutations(...)`

Current compile-source clarification:

1. compile still owns construction of the full final message list sent into the next LLM call
2. compile does not require system and non-system messages to come from the same source
3. in the current chat path:
   - system content is assembled from source-level data such as `data_from_agent_config` and `data_from_selfref`
   - non-system messages come from the already-built `input_messages` / turn history
4. prompt injection such as tool best practices and must-principles is applied after that full list is assembled

This means selfref is no longer treated as the default source for reconstructing the whole non-system transcript on each turn.

### `base/llm_call.py`

This module owns single-call execution.

Its responsibilities are:

1. call `LLM_Interface.chat(...)` or `LLM_Interface.chat_stream(...)`
2. parse content, tool calls, reasoning details, and usage
3. emit LLM call events
4. convert single-call results into assistant-side mutations
5. convert abort during a single call into truncation mutation

Important exported pieces:

1. `SingleLLMCallResult`
2. `SingleLLMPhaseResultYield`
3. `execute_single_llm_phase(...)`

### `base/tool_scheduler.py`

This module owns tool execution scheduling.

Its responsibilities are:

1. execute tool calls, including parallel execution
2. emit tool lifecycle events
3. convert tool outputs into mutations
4. convert multimodal/user-inserted tool outputs into `UserMessageMutation`
5. convert tool-stage cancellation into `ToolCancelledMutation`

Important exported pieces:

1. `ToolSchedulerResult`
2. `schedule_tool_batch(...)`

### `base/react_loop.py`

This module owns the main ReAct orchestration loop.

Its responsibilities are:

1. maintain `ReactLoopState`
2. collect pending mutations before each compile boundary
3. compile the next context
4. execute one model call
5. execute one tool batch when required
6. emit loop-level lifecycle events
7. call internal hooks
8. finalize the turn

Important exported pieces:

1. `ReactLoopState`
2. `run_react_loop(...)`

### `base/ReAct.py`

This module is no longer the main implementation container.

It now exists only to preserve public entrypoints expected by upper layers and tests:

1. `execute_single_llm_call(...)`
2. `ReAct_loop(...)`

Internally it delegates to the new core modules.

## Event Model

The implemented core is event-only.

That means:

1. the core always produces `ReactOutput`
2. the core no longer owns an event/non-event dual mode internally
3. upper layers can still derive old-style return values if needed, but they do that by consuming the event stream

Core event production sources are:

1. `base/react_loop.py` for loop lifecycle events
2. `base/llm_call.py` for LLM call and chunk events
3. `base/tool_scheduler.py` for tool lifecycle events

These events are carried through `EventBus` and emitted outward as `EventYield`.

## Step-Layer Compatibility Strategy

The lower core is event-only, but the step layer still preserves some older call patterns.

Current strategy:

1. `base.ReAct_loop(...)` always emits `ReactOutput`
2. `llm_decorator/steps/chat/react.py` derives legacy `(response, messages)` pairs only when needed
3. `llm_decorator/steps/function/react.py` derives legacy final raw responses only when needed

This keeps the core clean while containing compatibility logic in the step layer.

## Hook Model

The hook lifecycle is still supported.

Current hook points:

1. `on_run_start`
2. `before_llm_call`
3. `after_llm_call`
4. `before_tool_batch`
5. `after_tool_batch`
6. `before_finalize`
7. `collect_context_mutations`

`collect_context_mutations` is the important addition for the new boundary model.

It allows integrations such as selfref to contribute structured mutations before a compile boundary, instead of rewriting live context directly.

## Selfref Integration Model

The current selfref boundary is source-oriented.

That means:

1. selfref stores durable source-level context such as base system prompt, experiences, summary, and parsed working state
2. selfref contributes mutations at compile boundaries
3. final compiled histories can be parsed back into selfref source form after mutation application and finalize
4. selfref is not the default source for rebuilding every non-system message when `llm_chat` already has an explicit `history + current input` message list

In practice, the implemented chat compile path treats:

1. `data_from_agent_config` as the agent-config side of system construction
2. `data_from_selfref` as the durable selfref side of system construction
3. `input_messages` as the default source for non-system messages

### What Is Mutation-Only Now

The ReAct turn path for selfref context compaction is now mutation-driven.

Current flow:

1. runtime primitive queues compaction intent with `queue_context_compaction(...)`
2. `SelfReferenceReActSyncHooks.collect_context_mutations(...)` converts that intent into `ContextSummaryMutation`
3. `react_loop` collects those mutations before compile
4. `compile_context(...)` applies them
5. finalize writes the resulting compiled context back to selfref store

This means same-turn compaction now affects the next compile boundary without mutating the live loop context directly.

### Remember/Forget During Active ReAct Turns

During an active ReAct turn:

1. `remember_experience(...)` now queues a pending context mutation intent
2. `forget_experience(...)` now queues a pending context mutation intent
3. `SelfReferenceReActSyncHooks.collect_context_mutations(...)` converts them into:
   - `ExperienceRememberMutation`
   - `ExperienceForgetMutation`
4. `compile_context(...)` applies them before the next model-visible context is produced

Outside an active ReAct turn, the existing direct store update behavior is still kept for the memory API and direct PyRepl usage.

This is an intentional hybrid compatibility stage.

### What Still Remains More Direct Than Ideal

Not every selfref API has been fully moved to mutation-only semantics yet.

Examples:

1. direct memory handle CRUD operations
2. direct system-prompt mutation helpers outside active ReAct turns
3. explicit history replacement APIs

These remain as direct store operations for now because they are broader runtime APIs, not only ReAct-turn orchestration hooks.

Within active ReAct loop execution, the important context-bearing paths have been shifted toward mutation collection.

## Abort Behavior

Abort is implemented as mutation-producing behavior.

### Single LLM Call Abort

When a single call is aborted during streaming:

1. already received assistant partial content is preserved
2. `AssistantTruncatedMutation` is produced
3. compile turns that mutation into an assistant message with truncation notice

### Tool Scheduler Abort

When tool execution is aborted:

1. completed tools keep their normal mutations
2. incomplete tools produce `ToolCancelledMutation`
3. compile turns those into protocol-safe assistant/tool history

## Langfuse

Langfuse remains supported in the new architecture.

Observations currently exist at least in:

1. loop-managed LLM call boundaries
2. tool execution boundaries

Compatibility patch points still exist through `base/ReAct.py` for tests and upper-layer patching.

## Actual Data Flow

The current effective loop shape is:

1. collect pending context mutations from hooks
2. compile context
3. run one LLM phase
4. turn single-call result into mutations
5. if tool calls exist, run one tool batch
6. turn tool outputs into mutations
7. repeat
8. finalize with the compiled protocol history

Conceptually:

```python
while True:
    pending_mutations += collect_context_mutations(...)
    compiled = compile_context(context_state, pending_mutations)
    pending_mutations = []

    # compile_source can assemble the full LLM input here,
    # using source-level system data plus non-system input messages

    llm_result = single_llm_call(compiled.messages)
    llm_mutations = llm_result.mutations

    if no_tool_calls:
        final = compile_context(context_state, llm_mutations)
        break

    tool_result = schedule_tool_batch(llm_result.tool_calls)
    pending_mutations = llm_mutations + tool_result.mutations
    context_state = ContextState(messages=compile_context(context_state, pending_mutations).messages)
    pending_mutations = []
```

The real code includes hooks, event emission, Langfuse, and compatibility wrappers, but the implemented shape now follows this model much more closely than the previous monolithic `ReAct.py` design.

## Current Compatibility Layers

These layers still intentionally contain compatibility logic.

1. `base/ReAct.py`
2. `llm_decorator/steps/chat/react.py`
3. `llm_decorator/steps/function/react.py`

The important property is that the new core implementation no longer depends on the old compatibility behavior. Compatibility is layered on top of the new core, not the other way around.

## Known Transitional Areas

The architecture is much cleaner now, but some transition-layer behavior still exists.

1. step-layer derivation of legacy return forms from event streams
2. direct selfref store mutation APIs that remain available outside active ReAct turns
3. compatibility patch points retained for tests and upper-layer patching

These are considered acceptable transition costs because they no longer define the core boundary.

## Summary

The implemented ReAct architecture is now:

1. event-only at the core
2. mutation-driven at the compile boundary
3. modularized across dedicated core files
4. compatible upward through thin wrappers
5. no longer centered around a monolithic `base/ReAct.py`

This document describes the architecture that is actually running in the repository today.
