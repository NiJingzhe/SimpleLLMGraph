---
name: simplellmfunc-developer
description: "Develop and maintain the SimpleLLMFunc framework itself. Use when changing framework internals, tests, docs, specs, runtime primitives, decorator behavior, tool plumbing, event streams, PyRepl integration, provider adapters such as OpenAICompatible/OpenAIResponsesCompatible, or contributor-facing project structure and conventions."
license: MIT
compatibility: "Python 3.12+ repo with pytest, Poetry, Mintlify docs, and SimpleLLMFunc source tree available."
metadata:
  project: SimpleLLMFunc
  version: "0.8.1"
---

# SimpleLLMFunc Framework Development

## When to use this skill
- Use this skill when the task changes the framework itself, not just an app built on it.
- Typical triggers: editing `SimpleLLMFunc/`, `tests/`, `docs/`, `spec/`, built-in tools, runtime primitives, decorator semantics, event-stream behavior, provider adapters, TUI utilities, or contributor docs.

## Core development philosophy
- **LLM is Function**: preserve the framework's function-first design — LLM calls are indistinguishable from Python function calls.
- **Prompt as Code**: docstrings are system prompts. Code and prompts are never separated.
- **Context-Centric**: each provider request is compiled from invocation configuration, a base transcript/history, and runtime transcript patches. `ContextMutation` is an internal typed patch protocol, not the whole context source of truth. No component may directly modify the live ReAct transcript.
- Keep public behavior explicit and typed.
- Favor small, composable modules over hidden orchestration.
- Prefer explicit boundaries between pure transforms, state mutation, and orchestration side effects. Recent selfref/ReAct work depends on keeping those lines sharp.
- Follow repo-grounded conventions instead of generic framework habits.
- When docs and code disagree, source and tests are the final authority.

## Default implementation workflow
1. Read the relevant docs, tests, and source before changing behavior.
2. Map the affected layer: decorator, compile boundary, ReAct runtime, tool system, runtime primitives, interface, hooks, or docs/spec.
3. Write or update tests first for the behavior you are changing.
4. Make the smallest coherent implementation change.
5. Run targeted tests, then broader tests if the change touches shared behavior.
6. Update docs/examples/specs when user-facing behavior or architecture changed.

## TDD and validation loop
- Start with a failing or missing test that captures the new behavior.
- Use red -> green -> refactor.
- Prefer focused unit tests in the mirrored `tests/` location.
- Add or update a runnable example when the feature is user-facing.
- If behavior changes affect docs or spec, update them in the same change.

## Project map

### L1. Decorator Layer (`llm_decorator/`)
Public entry points and invocation contract building.
- `llm_function_decorator.py`: `@llm_function` — returns an `LLMFunction` callable instance; `await instance(...)` gives typed result, `instance.stream(...)` yields `ReactOutput`
- `llm_chat_decorator.py`: `@llm_chat` — returns an `LLMChat` callable instance; calling it yields a `ReactOutput` async stream and provides a stable SelfRef agent identity
- `invocation_spec.py`: `InvocationSpec`, `PromptContract`, `TranscriptSeed`
- `invocation_builder.py`: `build_function_invocation_spec()`, `build_chat_invocation_spec()`
- `prompt_contract.py`: prompt templates, XML Schema generation, type descriptions
- `signature.py`: `parse_function_signature()`, trace_id, log context
- `utils/tools.py`: `process_tools()`, `collect_tool_prompt_specs()`
- `selfref_sync.py`: legacy compatibility shim around `SelfRefSession`; active `llm_chat` binding/session orchestration lives in `LLMChat` plus `runtime/selfref/session.py`

### L2+L3. Compile Boundary + ReAct Runtime (`base/`)
Mutation-driven context evolution and event-only ReAct loop.

Core loop modules:
- `react_loop.py`: `run_react_loop()` — main async generator, event-only
- `llm_call.py`: `execute_single_llm_phase()` — single LLM call, streaming/non-streaming
- `tool_scheduler.py`: `schedule_tool_batch()` — concurrent asyncio.Task execution
- `react_hooks.py`: `ReActHookExecutionContext`, hook lifecycle execution

Compile pipeline:
- `compile_pipeline.py`: **single compile entry** — `compile_invocation_turn()`, `reduce_turn_context()`, `convert_to_llm_request()`
- `context_compile.py`: `apply_mutations()`, `compile_context()` — mutation apply engine
- `llm_input_render.py`: `render_llm_input_messages()` — ephemeral system prompt rendering

Type contracts (`base/types/`):
- `source.py`: `CompileSource`, `DataFromAgentConfig`, `DataFromSelfRef`
- `context.py`: `ContextState`, `CompiledContext`
- `compile.py`: `ReducedTurnContext`, `CompiledTurnContext`
- `mutation.py`: `ContextMutation` union type (10 variants)
- `react.py`: `ReactLoopState`
- `llm.py`: `SingleLLMCallResult`
- `scheduler.py`: `ToolSchedulerResult`

Message and tool sub-modules:
- `messages/`: assistant message building, usage extraction, multimodal content, validation
- `tool_call/`: extraction, execution, streaming state, validation
- `type_resolve/`: type description, XML round-trip, multimodal detection
- `post_process.py`: response → typed result (XML → Pydantic)

### Runtime Primitive System (`runtime/`)
- `primitives.py`: `PrimitiveRegistry`, `PrimitivePack`, `PrimitiveCallContext`, `@primitive()`
- `worker_proxy.py`: `WorkerRuntimeProxy`, `WorkerRuntimeNamespace`, `PrimitiveTransport`
- `selfref/state.py`: `SelfReference` public facade, lifecycle, compatibility exports
- `selfref/store.py`: durable history/source store
- `selfref/active_turn.py`: active memory key, fork context, runtime toolkit, template params, active ReAct state contextvars
- `selfref/mutations.py`: pending compaction/context/destructive mutation queues
- `selfref/memory_api.py`: `self_reference.memory[...]` proxy/handle API
- `selfref/context_memory.py`: context snapshots, experience CRUD, compaction commit, direct memory editing API
- `selfref/agent_binding.py`: recursive agent callable binding state
- `selfref/fork_manager.py`: fork/spawn/gather lifecycle and result materialization
- `selfref/fork_utils.py`: fork helper functions and compatibility constants
- `selfref/session.py`: `SelfRefSession` — invocation-scoped plugin implementing ReAct hooks
- `selfref/context_ops.py`: `parse_context_messages()`, `build_context_messages_from_state_data()`, `canonicalize_context_messages()`
- `selfref/primitives.py`: selfref primitives (guide, context.inspect/remember/forget/compact, fork.is_bound/spawn/gather_all)

### Built-in Tools (`builtin/`)
- `pyrepl.py`: `PyRepl` public facade
- `pyrepl_execution.py`: execute/reset worker protocol orchestration
- `pyrepl_worker_client.py`: subprocess and multiprocessing queue lifecycle
- `pyrepl_worker_mixin.py`: facade-compatible worker lifecycle wrappers
- `pyrepl_primitive_host.py`: runtime backend, primitive registry, primitive RPC, fork clone integration
- `pyrepl_tools.py`: execute/reset tool creation and output formatting
- `pyrepl_audit.py`: audit log writer
- `pyrepl_input_bridge.py`: process-wide input request/reply bridge
- `pyrepl_input_mixin.py`: PyRepl input submission API
- `file_tools.py`: `FileToolset` — workspace-scoped file tools with stale-write protection

### Event Stream (`hooks/`)
- `events.py`: 14 `ReActEvent` subtypes
- `stream.py`: `ReactOutput`, `ResponseYield`, `EventYield`, type guards
- `event_bus.py`: `EventBus` — event ingress with origin metadata
- `event_emitter.py`: `ToolEventEmitter`, `NoOpEventEmitter`

### Interface Layer (`interface/`)
- `llm_interface.py`: `LLM_Interface` abstract base class
- `openai_compatible.py`: `OpenAICompatible` adapter
- `openai_responses_compatible.py`: `OpenAIResponsesCompatible` adapter
- `key_pool.py`: `APIKeyPool` — multi-key rotation
- `token_bucket.py`: token-bucket rate limiting

### Infrastructure
- `logger/`: structured logging, trace_id, async context manager
- `observability/`: Langfuse trace/span integration
- `type/`: multimodal types (`Text`, `ImgUrl`, `ImgPath`)
- `utils/tui/`: Textual TUI integration

### Tests (`tests/`)
Mirror of behavior and architecture; often the fastest place to infer conventions.

## Naming and style rules
- File names: `snake_case`.
- Functions: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Public APIs should carry type annotations.
- Public user-facing callables should have useful docstrings.
- Follow the repo's existing formatting and PEP 8 style.

## Framework-specific development rules

### Core rule: runtime transcript patch boundary
LLM-visible messages are compiled from invocation config, a base transcript/history, and runtime patches:
1. Producers (LLM call, tool exec, selfref hooks) emit `ContextMutation` objects for runtime transcript edits.
2. `react_loop` collects mutations before each compile boundary.
3. `compile_context()` applies mutations in order via `apply_mutations()`.
4. The compile pipeline combines that patched transcript with invocation config and SelfRef source data before rendering provider messages.

Do not introduce code that directly mutates `ContextState.messages` outside of `apply_mutations()`. Do not let tools or primitives write directly to the live transcript. Also do not describe mutations as the source of all context: docstrings, template params, tool schemas, and initial history are separate compile inputs.

### SelfRef rules
- `SelfReference` (`runtime/selfref/state.py`) is the public facade for the durable backend; do not re-expand it into a god object.
- `SelfRefSession` (`runtime/selfref/session.py`) is the invocation-scoped plugin that implements ReAct hooks (`collect_context_mutations`, `finalize`, etc.).
- Keep pure context parsing/rendering in `runtime/selfref/context_ops.py`.
- Keep durable history/source storage in `runtime/selfref/store.py`.
- Keep active turn contextvars and active ReAct state lookup in `runtime/selfref/active_turn.py`.
- Keep pending mutation/compaction queues in `runtime/selfref/mutations.py`.
- Keep context memory operations, experience CRUD, compaction commit, and direct memory editing in `runtime/selfref/context_memory.py`.
- Keep fork/spawn/gather lifecycle behavior in `runtime/selfref/fork_manager.py` and pure-ish helpers/constants in `runtime/selfref/fork_utils.py`.
- `LLMChat` is the stable callable agent instance bound into `SelfReference`; keep per-call mutable state inside invocation locals / `SelfRefSession`, not on shared instance fields.
- `llm_decorator/selfref_sync.py` is a legacy compatibility shim; do not move new lifecycle logic there unless intentionally reviving that layer.
- SelfRef can only affect the system through: source snapshot, pending intents → mutations, finalize side effects.
- `runtime.selfref.fork.spawn(...)` children inherit the pre-fork context snapshot. Do not reintroduce the parent's pending assistant tool-call message into child-visible history.

### ReAct runtime rules
- The core is event-only: `run_react_loop()` always yields `ReactOutput`.
- There is no `enable_event` parameter; event mode is the only mode at the core level.
- New terminal behavior should flow through the shared finalize path so `before_finalize` stays consistent across event, abort, and max-tool-cap exits.
- Abort is mutation-producing behavior: `AssistantTruncatedMutation` for LLM abort, `ToolCancelledMutation` for tool abort.
- Do not add dual-mode (event/non-event) logic inside `react_loop.py`, `llm_call.py`, or `tool_scheduler.py`.

### Decorator rules
- `@llm_function` returns `LLMFunction`; `@llm_chat` returns `LLMChat`. Preserve function-like metadata (`__wrapped__`, `__name__`, `__doc__`, `__annotations__`, `__signature__`) when changing them.
- Do not reintroduce closure-only wrapper implementations for SelfRef binding; SelfRef should bind the stable `LLMChat` instance.
- There is no `enable_event` or `return_mode` decorator parameter; `llm_chat` always yields `ReactOutput`, and `llm_function.stream(...)` yields `ReactOutput`.
- Prefer `async def` for decorated public patterns and tool implementations.
- Keep docstring-parsed contracts in sync with behavior. This matters for `@tool` and runtime primitives.
- Runtime primitive docstrings must include `Best Practices`; registration fails without them.
- Preserve history semantics in `llm_chat`: `history` and `chat_history` are special names.
- Preserve structured output parsing behavior unless the task explicitly changes it.

### Provider adapter rules
- Keep wire-format differences in the adapter layer under `SimpleLLMFunc/interface/`.
- Do not leak Responses-specific request/stream contracts into `ReAct` or decorator code unless the public framework contract is intentionally changing.
- `OpenAIResponsesCompatible` should remain a first-class adapter, not a special case hidden inside `ReAct`. System prompts map to Responses `instructions`, and Responses-specific reasoning/tool-stream handling belongs in the adapter.

### Test rules
- Treat tests as executable API documentation for subtle cases like self-reference, event mode, and provider compatibility.
- Add architecture tests (invocation isolation, compile single-entry, history authority) when changing the invocation/compile/finalize pipeline.

## Documentation and spec rules
- Update `mintlify_docs/` when user-facing behavior changes.
- Update `spec/` when module responsibilities, architecture map, or repo-wide guidance changes.
- Keep examples runnable and aligned with current behavior.
- Use progressive disclosure in skills and docs: concise guidance in the main file, details in reference docs.
- Keep `provider.json` format docs and `.env` / environment-variable docs aligned with actual loader and observability behavior.
- Keep the packaged `skills/` directory and the `simplellmfunc-skill` export CLI aligned so installed users can export the current skill contents correctly.
- When Responses adapter behavior or selfref fork behavior changes, update packaged `skills/` docs in the same change, not only `mintlify_docs/`.
- Treat `AGENTS.md` as a feedback-loop artifact: when recurring agent mistakes reveal missing environmental guidance, update the file so the fix lives in the system instead of only in maintainer memory.

## Read these reference docs as needed
- Architecture and contributor map: `reference/project-map.md`
- TDD, tests, and validation expectations: `reference/testing-and-tdd.md`
- Coding conventions and naming rules: `reference/style-and-spec.md`
- Framework-specific gotchas: `reference/framework-gotchas.md`
- Docs and examples workflow: `reference/docs-and-examples.md`
- Maintainer workflow notes: `reference/AGENT.md`
- Contributor guide: `reference/contributing.md`
- Mirrored repo spec: `reference/spec/project-map.md`, `reference/spec/overall-spec.md`, `reference/spec/primitive-dev-api-plan.md`, `reference/spec/meta.md`
- Developer examples: `examples/add_runtime_primitive_pattern.py`, `examples/test_first_decorator_change.md`, `examples/update_docs_checklist.md`
