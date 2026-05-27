# Change log for SimpleLLMFunc

## Unreleased

## 0.8.3 (2026-05-27) - Dependency Constraint Relaxation

### 🔧 Improvements

- Relaxed runtime, development, and build-system package dependency constraints to use lower-bound-only `>=...` specifiers instead of Poetry caret constraints, fixed pins, or explicit upper bounds.
- Dropped the tracked `uv.lock` file because the project release workflow no longer supports uv lock synchronization.
- Regenerated `poetry.lock` after the constraint cleanup; `rich` now resolves to `15.0.0`.

### 🧪 Testing

- Verified the Poetry metadata and sync workflow with `poetry check`, `poetry lock`, and `poetry sync`.
- Re-ran TUI dependency regression coverage: `19 passed`.

## 0.8.2 (2026-05-27) - Multimodal Inputs and PyRepl Image Outputs

### ✨ New Features

1. **Multimodal input support**:
   - `llm_function` now supports typed image input through explicit `ImgUrl` / `ImgPath` parameters and lists/unions containing those types.
   - `llm_chat` now accepts `UserChatMessage` for OpenAI-compatible multimodal user messages containing text and `image_url` content parts.
   - Added `UserChatMessage.multimodal(...)`, the canonical `UserChatMessage` helper for ergonomic chat message construction.

2. **PyRepl multimodal image output**:
   - `PyRepl.execute(...)` now exposes captured image artifacts from `display(Image(...))`, image-rich last expressions, and explicit `ImgPath` / `ImgUrl` results.
   - The `execute_code` tool converts image artifacts into multimodal tool results so agents can inspect generated plots directly.
   - Tool results now support multiple images through `list[ImgPath | ImgUrl]` and `(text, list[ImgPath | ImgUrl])`.
   - `OpenAIResponsesCompatible` now maps chat-style multimodal user content to Responses API `input_text` / `input_image` parts.

### 📚 Documentation & Examples

1. **PyRepl image workflow docs**:
   - Added English and Chinese docs for PyRepl image artifact capture and multimodal tool return behavior.
   - Added `examples/pyrepl_seaborn_multimodal_images.py`, a runnable seaborn example that returns multiple generated plots to the model.
   - Added developer skill notes covering the split PyRepl architecture and image artifact capture flow.

### 🧪 Testing

- Added regression coverage for multimodal chat input, multiple-image tool results, PyRepl image artifact capture, scheduler event results, and Responses API content conversion.
- Verified targeted release suites with `143 passed` for PyRepl/tool/Responses paths.

## 0.8.1 (2026-05-20) - Architecture Split: PyRepl, SelfRef, and Chat Decorator Internals

### 🔧 Improvements

1. **PyRepl facade split**:
   - Slimmed `builtin/pyrepl.py` into a public facade while preserving `PyRepl` behavior and compatibility wrappers.
   - Extracted worker lifecycle, execute/reset orchestration, primitive host integration, tool creation, audit logging, input bridging, and worker alias compatibility into focused modules.
   - Kept `execute_code`, `reset_repl`, runtime primitive calls, subprocess cleanup, input handling, and timeout behavior stable.

2. **SelfReference component split**:
   - Reduced `runtime/selfref/state.py` to a facade/lifecycle module.
   - Extracted durable store, active turn contextvars, mutation queues, memory proxy API, context memory/editing, agent binding, fork lifecycle, and fork helper logic into dedicated modules.
   - Preserved public APIs such as `SelfReference.memory`, `SelfReference.instance`, `bind_history`, context compaction, and `runtime.selfref.fork.spawn/gather_all`.

3. **`llm_chat` internal split**:
   - Split chat decorator support code into call-context, toolkit, selfref, and shared-type helper modules.
   - Preserved `LLMChat`, `ReAct_loop` patch points, runtime toolkit override behavior, and SelfRef finalization semantics.

### 📚 Documentation & Skills

1. **Architecture map refresh**:
   - Updated README/README_ZH project maps, `spec/project-map.md`, and packaged developer skill guidance to describe the current module boundaries.
   - Added the new PyRepl and SelfRef component files to architecture documentation.

### 🧪 Testing

- Verified the full test suite after the refactor: `661 passed`.
- Re-ran focused PyRepl, SelfRef, llm_chat, runtime-toolkit, stage2 invocation, and ReAct core suites during the split.

## 0.8.0 (2026-05-19) - Callable Decorators, Unified Chat Events, and Provider Defaults

### ✨ New Features

1. **OpenAI-compatible per-model API defaults**:
   - Added optional `api_params` support to `OpenAICompatible` model configs loaded from `provider.json`.
   - Instance-level `api_params` are passed to chat and streaming calls.
   - Call-level kwargs override `api_params`, enabling stable model defaults with one-off overrides.

### 💥 Breaking Changes

1. **Removed legacy `llm_chat(return_mode=...)`**:
   - `llm_chat` no longer accepts the obsolete `return_mode` decorator option.
   - Chat invocations now consistently yield `ReactOutput` items (`ResponseYield` / `EventYield`) directly.
   - Update old `(chunk, history)` consumers to route with `is_response_yield(...)` / `is_event_yield(...)` and read `output.messages` for updated history.

### 🔧 Improvements

1. **Runtime toolkit override merging**:
   - SelfRef runtime toolkit overrides now merge with the default toolkit instead of replacing it outright.
   - Override tools take priority when a tool with the same name exists in both toolkits.

2. **Decorator callable instances**:
   - `@llm_function` now returns an `LLMFunction` callable instance while preserving `await fn(...)`, `fn.stream(...)`, and function-like metadata.
   - `@llm_chat` now returns an `LLMChat` callable instance whose calls yield `ReactOutput` streams.
   - SelfRef now binds to the stable `LLMChat` agent instance, making rebinding and fork toolkit resolution more explicit internally.
   - Exported `LLMFunction` and `LLMChat` from `SimpleLLMFunc.llm_decorator` and package-level imports.

### 📚 Documentation & Skills

1. **Context model wording clarification**:
   - Updated Mintlify docs and packaged usage skill to describe `ContextMutation` as an internal runtime transcript patch protocol.
   - Clarified that LLM requests are compiled from invocation configuration, a base transcript/history, and runtime patches.
   - Removed public-facing wording that implied mutations are the whole context source or primary user-facing customization surface.

2. **Configuration docs for `api_params`**:
   - Documented `api_params` in English/Chinese Mintlify config pages and packaged skill references.

### 🧪 Testing

- Added tests for `OpenAICompatible` `api_params` loading, forwarding, and call-level override behavior.
- Added tests for runtime toolkit override merging and tool-name precedence.

## 0.7.8 (2026-04-16) - Responses Adapter and Selfref Fork Context Refinement

### ✨ New Features

1. **OpenAI Responses API adapter**:
   - Added `OpenAIResponsesCompatible` as a first-class interface adapter.
   - Supports `provider.json` loading, direct construction with `APIKeyPool`, Responses tool schema translation, and reasoning passthrough.
   - Keeps decorator-facing authoring unchanged while mapping system prompts to Responses `instructions`.

2. **Responses API runnable example**:
   - Added `examples/response_api_example.py` as a TUI-first demo using `OpenAIResponsesCompatible`, runtime selfref, and file tools.

### 🔧 Improvements

1. **SelfReference fork context construction**:
   - Child forks now inherit the pre-fork context snapshot instead of the parent's in-flight fork tool-call scene.
   - Removed the old fake tool-result closure path from child history construction.
   - Isolated child fork execution from the parent's active ReAct state to prevent context contamination.

2. **Fork result ergonomics**:
   - `runtime.selfref.fork.gather_all(...)` now exposes both `response` and `result` for successful results.
   - Error payloads now consistently include `response: None` and `result: None`.
   - Updated selfref primitive guidance to tell models to read `status` first, then `response` or `result`.

3. **PyRepl runtime guidance**:
   - Strengthened `execute_code` prompt injection to explicitly explain that runtime primitives are not standalone tool calls and must be invoked as `runtime.namespace.name(...)` inside `execute_code`.

4. **Examples and prompts**:
   - Updated TUI agent prompts and tests to use explicit runtime-primitive wording for selfref fork/compaction actions.

### 📚 Documentation & Skills

1. **Docs sync for Responses + selfref**:
   - Updated README, README_ZH, Mintlify docs, packaged skills, and mirrored skill references to document `OpenAIResponsesCompatible`, Responses `instructions`, and the refined selfref fork semantics.
   - Added explicit guidance for parsing `gather_all()` results through `status`, `response`, and `result`.

### 🧪 Testing

- Added targeted adapter regression coverage for `OpenAIResponsesCompatible`.
- Added selfref fork history and fork-result alias tests across runtime, decorator, PyRepl, and example suites.
- Verified broader ReAct regressions with targeted lower-level test runs.

## 0.7.7 (2026-04-03) - Mintlify Docs, i18n, and Skills Workflow

### ✨ New Features

1. **Mintlify bilingual docs**:
   - Added a Mintlify-based English docs tree under `mintlify_docs/en`.
   - Switched Mintlify navigation to `navigation.languages` with Chinese as default and English under `/en`.

2. **Harness and provider guidance in packaged skills**:
   - Expanded the bundled usage skill with guidance on organizing `provider.json`.
   - Added stronger recommendations for typed contracts and Pydantic-first structured outputs.
   - Added Harness Engineering guidance centered on context planning and closed-loop environment design.

### 🔧 Improvements

1. **Quickstart improvements**:
   - Added packaged skill export guidance directly to Chinese and English quickstart pages.
   - Surfaced the `skills_cli` workflow earlier so coding-agent users see it immediately after installation.

2. **Mintlify i18n workflow**:
   - Added a Mintlify-specific i18n sync script and override-memory mechanism.
   - Added tests for MDX segmentation, translation placeholder handling, and locale route rewriting.

3. **Skill and contributor docs alignment**:
   - Updated developer-facing skills and reference docs to point at Mintlify docs instead of the old Sphinx tree.
   - Added explicit AGENTS.md feedback-loop guidance in the developer skill.

### 🧹 Cleanup

1. **Removed legacy Read the Docs / Sphinx pipeline**:
   - Deleted the old `docs/` source tree and `.readthedocs.yaml`.
   - Removed obsolete translation scripts tied to the old Read the Docs workflow.
   - Removed unused Sphinx-specific dev dependencies from `pyproject.toml`.

2. **Repository documentation cleanup**:
   - Updated README and README_ZH to stop pointing at Read the Docs.
   - Aligned docs/spec/skills references with the Mintlify documentation workflow.

## 0.7.6 (2026-04-02) - Packaged Skills and Safer File Tools

### ✨ New Features

1. **Bundled Agent Skills**:
   - Added packaged `simplellmfunc` and `simplellmfunc-developer` Agent Skills with framework philosophy, prompt-construction guidance, provider/env setup, mirrored docs, and real examples.

2. **Skill Export CLI**:
   - Packaged the repository `skills/` directory into wheel/sdist builds.
   - Added `simplellmfunc-skill` so installed users can export usage or developer skills into tool directories such as `~/.config/opencode/skills`.

### 🔧 Improvements

1. **Long Output Handling**:
   - Raised `too_long_to_file` truncation from 4000 to 20000 tokens.
   - Synced PyRepl/tool guidance and improved the long-output reminder text shown to agents.

2. **File Tools Guardrails**:
   - `grep` now rejects full wildcard regexes such as `.`, `.*`, `.+`, and simple anchored equivalents for both `pattern` and `path_pattern`.
   - Prevents accidental workspace-wide scans while keeping scoped regex search intact.

3. **Configuration and Example Cleanup**:
   - Added explicit `provider.json` and `.env` guidance to the bundled skills, including the recommended `LOG_LEVEL=WARNING` default.
   - Updated provider templates to use unique model names and aligned multimodal example docstrings with their actual signatures and return values.

### 📚 Documentation & Localization

1. **Release Refresh**:
   - Updated README.md and README_ZH.md to show only the latest release notes at the top.
   - Refreshed release metadata and synchronized the bundled skills with the 0.7.6 release.

### 🧪 Testing

- Verified skill export, truncation, PyRepl, and file tool guard changes with targeted pytest suites.

## 0.7.5 (2026-04-01) - Long Output Truncation and Tool Improvements

### ✨ New Features

1. **Long Output Truncation (`too_long_to_file`)**:
   - Added `too_long_to_file` parameter to `@tool` decorator and `Tool` class.
   - When enabled, tool outputs exceeding 20000 tokens are automatically saved to a temp file.
   - The returned content is truncated to the first 20000 tokens with a `<system-reminder>` hint appended.
   - Includes a fast heuristic tokenizer for token estimation (Chinese chars ≈ 2 tokens, English words ≈ 1.3 tokens).

2. **PyRepl Auto-Truncation**:
   - `execute_code` tool now has `too_long_to_file=True` enabled by default.
   - Prevents context overflow from large code execution outputs.

### 🔧 Improvements

1. **File Tools Guard**:
   - `grep`/`sed` now reject overly broad `.*` patterns for both `pattern` and `path_pattern`.
   - Returns a helpful message guiding users to use more specific patterns.

2. **File Tools Feedback**:
   - Improved `grep` output when no matches are found, providing clearer feedback.

### 📚 Documentation & Localization

1. **Tool System Docs**:
   - Added documentation for `too_long_to_file` feature in `tool.md`.
   - Updated `@tool` decorator parameter list and `Tool` class signature.
   - Added "Long Output Truncation" section with usage examples and best practices.

2. **PyRepl Docs**:
   - Updated `pyrepl.md` to document the auto-truncation feature for `execute_code`.

3. **Translation Pipeline**:
   - Regenerated gettext catalogs, rebuilt `po`/`mo` files, and refreshed English translations.
   - Updated README.md and README_ZH.md with version 0.7.5 release notes.

## 0.7.4 (2026-03-27) - Builtin selfref Only and Release Correction

### 🔧 Improvements

1. **Builtin selfref only**:
   - Removed the temporary `PyRepl(self_reference=...)` override path.
   - `PyRepl` now always installs and uses its builtin `selfref` backend.
   - Host-side access to selfref state now goes through `repl.get_runtime_backend("selfref")`.

2. **Runtime API alignment**:
   - Updated tests, examples, and guides to follow the builtin-only selfref model.
   - Kept runtime primitive pack guidance, backend lifecycle handling, and selfref discovery aligned with the default implementation.

### 📚 Documentation & Localization

1. **Release correction**:
   - Updated source docs and examples to remove the external selfref injection path.
   - Regenerated locale catalogs, refreshed translations, and rebuilt bilingual Sphinx HTML docs for the corrected release.

### 🧪 Testing

- Verified targeted PyRepl, runtime backend, llm_chat, and observability test suites.
- Rebuilt Chinese and English Sphinx HTML documentation successfully.

## 0.7.3 (2026-03-27) - Runtime Primitive Packs, Docs, and Observability Polish

### ✨ New Features

1. **Standardized Runtime Primitive Packs**:
   - Added pack-level `guidance` so each `PrimitivePack` can describe its mental model and usage scope.
   - Standardized builtin `selfref` installation so `PyRepl` mounts it through the same `PrimitivePack` path used by custom packs.

### 🔧 Improvements

1. **Prompt and Runtime Guidance Cleanup**:
   - Simplified injected runtime/tool guidance and removed duplicated tool-description text.
   - Generalized runtime primitive discovery examples to `runtime.list_primitives(contains="<namespace>.")`.
   - Moved selfref namespace guidance to pack-level guidance and kept tool best practices tool-scoped.

2. **Runtime Backend Lifecycle**:
   - Standardized backend-name normalization across primitive registration and backend registration.
   - `install_pack(..., replace=True)` now closes replaced backends, skips duplicate lifecycle transitions for the same backend object, and unregisters stale primitives.
   - Backend-bound primitive calls now fail fast with clearer backend-resolution errors.

3. **Tooling and Framework Defaults**:
   - Made framework tool-call limits opt-in with `max_tool_calls=None` as the default.
   - Removed redundant tool preprocessing during chat message construction.

4. **Observability**:
   - Trace-context lookup now avoids noisy Langfuse "No active span in current context" warnings when no span is active.
   - Nested span parenting remains intact for `llm_chat`, `execute_code`, and selfref fork chains.

### 📚 Documentation & Localization

1. **Runtime Primitive Docs**:
   - Expanded the primitive and PyRepl guides to document the new pack authoring path, spec/contract model, builtin selfref behavior, and runtime discovery flow.
   - Updated examples and Chinese guides to match the current runtime primitive API.

2. **Translation Pipeline**:
   - Regenerated gettext catalogs, rebuilt `po`/`mo` files, and refreshed English translations with batch translation.
   - Cleaned residual Chinese text from English locale outputs and restored successful bilingual Sphinx HTML builds.
   - Added the missing roman numeral packages needed by the current Sphinx build stack.

### 🧪 Testing

- Verified runtime, prompt, and observability changes with targeted pytest suites.
- Rebuilt both Chinese and English Sphinx HTML documentation successfully.

## 0.7.2 (2026-03-18) - OpenAI SDK 2.x Baseline

### ⚠️ Breaking Changes

1. **OpenAI Python SDK Requirement**:
   - Updated the supported dependency range to `openai >=2.0.0,<3.0.0`.
   - OpenAI Python SDK 1.x is no longer supported.

### 🔧 Improvements

1. **Tool Call Type Handling**:
   - Switched internal tool-call construction to the concrete function tool-call model used by OpenAI SDK 2.x.
   - Removed reliance on the older `chat_completion_message_tool_call` import path for runtime code and tests.

2. **Release Metadata**:
   - Bumped the package version to `0.7.2` and refreshed release references in the English and Chinese README files.

### 🧪 Testing

- Re-locked dependencies on OpenAI SDK 2.x and verified the full pytest suite against the upgraded stack.

## 0.7.1 (2026-03-15) - Runtime Polish & File Tools

### ⚠️ Breaking Changes

1. **SelfReference Forking**:
   - Removed `selfref.fork.run`, `selfref.fork.wait`, and `selfref.fork.wait_all` (plus chat variants).
   - Use `selfref.fork.spawn` + `selfref.fork.gather_all` for async fork collection; `gather_all` accepts a single fork id/handle or a list.

2. **PyRepl Toolset Changes**:
   - `execute_code` tool now returns a natural-language summary string (not JSON).
   - Removed `list_variables` from the toolset and from the PyRepl API.

### ✨ New Features

1. **FileToolset (builtin)**:
   - Added workspace-scoped file tools: `read_file`, `grep`, `sed`, `echo_into` with hash-based stale-write protection.

2. **AbortSignal Control**:
   - Added cooperative abort support for in-flight turns; event streams now mark abort metadata on `ReactEndEvent.extra`.

### 🔧 Improvements

1. **TUI/StdIO UX**:
   - Suppress primitive lifecycle marker events in TUI output.
   - Disable input while busy; stdio only injects `_abort_signal` when supported.

2. **Runtime & Tooling**:
   - `read_file` now warns when start/end range is inverted but clamped.
   - Runtime primitive specs include clearer parsing guidance.
   - PyRepl now supports an initial working directory (`working_directory`).

3. **Examples & Sandbox**:
   - Added `sandbox/` (tracked, contents ignored) and moved all file tool workspaces to this directory.
   - New general TUI agent demo combines selfref primitives + FileToolset.

4. **Observability (Langfuse)**:
   - Preserve trace context across async tool calls and TUI stream consumption to avoid split spans.
   - Added `scripts/langfuse_tool_debug.py` for tool-call trace debugging.

### 📚 Docs

1. **New Guides & Navigation**:
   - Added AbortSignal guide and reorganized documentation navigation.
   - Documented FileToolset and updated PyRepl/tool references.


## 0.7.0 (2026-03-11) - Runtime Primitives, Forking, and Docs Refresh

### 🎉 Major Features

1. **Runtime Primitive Registry**:
   - Added host-side primitive registry with `runtime.list_primitives`, `runtime.get_primitive_spec`, and `runtime.list_primitive_specs` for in-REPL discovery.
   - Added worker proxy plumbing so primitives are callable in PyRepl without imports.
   - Bundled SelfReference primitive pack under `runtime.selfref.history.*` and `runtime.selfref.fork.*` namespaces.

2. **SelfReference Forking**:
   - Added fork lifecycle primitives: `run`, `spawn`, `wait`, `wait_all` (plus chat variants) for parallel agent work.
   - Fork results expose status, response, memory key, history count, and structured error details.

3. **Event Origin Metadata & TUI Routing**:
   - Normalized event origin metadata (session/fork context + tool linkage) for deterministic routing.
   - Added fork-aware routing and visualization in the Textual TUI.

### 🔧 Improvements

1. **Tool Prompt Injection**:
   - Added best-practice prompts and tool-specific prompt injection hooks for safer tool usage.


## 0.6.0 (2026-02-24) - PyRepl, Textual TUI, and Durable Agent Memory

### 🎉 Major Features

1. **Builtin PyRepl Toolchain**: Added a production-ready Python REPL builtin based on a subprocess IPython runtime.
   - Introduced `SimpleLLMFunc.builtin.PyRepl` for persistent code execution across tool calls.
   - Added startup, active, and idle timeout controls for long-running agent workflows.
   - Improved execution reliability with worker supervision and richer runtime diagnostics.

2. **Textual TUI for `llm_chat`**: Added an out-of-the-box terminal UI powered by event stream updates.
   - New `@tui` integration with streaming markdown conversation rendering.
   - Added tool call arguments/results panels and model/tool usage statistics.
   - Added built-in quit controls and improved multi-turn interaction stability.

3. **Durable `SelfReference` Memory Contract**: Added self-reference memory controls for stateful agents.
   - Supports local durable memory semantics shared by chat loops and tools.
   - Enables safer prompt-level memory ownership and lifecycle control.

### ✨ New Features

1. **Tool Event Emission Pipeline**:
   - Added custom tool event emission support via event emitter hooks.
   - Improved event injection behavior in tool execution and ReAct orchestration.

2. **Input Stream Routing**:
   - Added tool input stream hooks to route pending tool input before normal chat turns.
   - Improved agent interactivity for tools that require follow-up user input.

3. **Examples and Developer Experience**:
   - Added dedicated examples for PyRepl, Textual TUI, custom tool events, and SelfReference.
   - Added translation and locale workflow scripts for documentation maintenance.

### 🔧 Improvements

1. **ReAct and Interface Reliability**:
   - Improved OpenAI-compatible interface behavior and response handling.
   - Refined ReAct message and tool-call execution flows to better support streaming and tool-event scenarios.

2. **Documentation Refresh**:
   - Added a dedicated PyRepl guide and expanded examples documentation.
   - Updated both English and Chinese docs for new runtime, hooks, and TUI capabilities.

### 🧪 Testing

- Added comprehensive test coverage for PyRepl runtime, TUI modules, event emitter/input stream hooks, self-reference behaviors, and OpenAI-compatible execution paths.

### ⚠️ Compatibility Notes

- The previous builtin `Kernel` workflow is superseded by `PyRepl`. If you referenced legacy kernel APIs, migrate imports and usage to `SimpleLLMFunc.builtin.PyRepl`.
- For full TUI observability, enable event streaming in `@llm_chat` (for example, `enable_event=True`).

## 0.5.0.beta1 (2025-01-09) - Event Stream & Type System Refactoring

> ⚠️ **Beta Release Notice**: This is a beta release. Optional breaking changes may be introduced. Please review the migration guide below if you encounter any issues.

### 🎉 Major Features

1. **Event Stream System**: A brand new observability system that supports real-time observation of ReAct execution cycles
   - New `enable_event` parameter (defaults to `False` for backward compatibility)
   - Supports 13 event types: ReAct start/end, LLM calls, tool calls, iterations, etc.
   - Tagged Union design, type-safe and flexible
   - Provides filter functions: `responses_only()`, `events_only()`, `filter_events()`
   - Provides decorator: `with_event_observer()` for event observation

2. **Type System Refactoring**: Unified type definitions, eliminated duplicates, improved type safety
   - New `type/tool_call.py`: Tool call related types
   - New `type/llm.py`: LLM response related types
   - New `type/hooks.py`: Hook system related types
   - Reuses OpenAI SDK types, reduces custom types
   - Unified export of all types to `type/__init__.py`

### ✨ New Features

1. **Event Type System**:
   - `ReactStartEvent`: ReAct cycle start
   - `LLMCallStartEvent` / `LLMCallEndEvent`: LLM call events
   - `LLMChunkArriveEvent`: Streaming chunk arrival (streaming mode only)
   - `ToolCallsBatchStartEvent` / `ToolCallsBatchEndEvent`: Tool call batch events
   - `ToolCallStartEvent` / `ToolCallEndEvent` / `ToolCallErrorEvent`: Individual tool call events
   - `ReactIterationStartEvent` / `ReactIterationEndEvent`: Iteration events
   - `ReactEndEvent`: ReAct cycle end

2. **Event Stream API**:
   ```python
   @llm_chat(llm_interface=llm, enable_event=True)
   async def my_chat(message: str):
       pass
   
   # Handle events and responses
   async for output in my_chat("Hello"):
       if output.type == "response":
           print(output.response)
       elif output.type == "event":
           print(output.event.event_type)
   ```

3. **Helper Utility Functions**:
   - `responses_only()`: Get only responses (backward compatible)
   - `events_only()`: Get only events
   - `filter_events()`: Filter specific event types
   - `with_event_observer()`: Add event observer decorator

### 🔧 Improvements

1. **Type System**:
   - Unified use of `MessageList` instead of `List[Dict[str, Any]]`
   - Unified use of `ToolDefinitionList` instead of `Optional[List[Dict[str, Any]]]`
   - Unified use of `ToolCall` type (directly reuses OpenAI SDK types)
   - Removed duplicate type definitions (`ReasoningDetail`, `ToolCallFunctionInfo`, `AccumulatedToolCall`)

2. **Code Organization**:
   - Removed `type/decorator.py`, migrated `HistoryList` to `type/hooks.py`
   - Updated all import paths to use unified type system

3. **Code Refactoring**:
   - Removed unnecessary dynamic imports in `ReAct.py`
   - Use module-level imports for better testability

### 📝 Documentation Updates

- Updated `llm_chat.md`: Added Event Stream usage instructions
- Updated `llm_function.md`: Added `enable_event` parameter documentation
- Updated `examples.md`: Added event stream example documentation
- Added new `event_stream.md`: Complete Event Stream guide

### ⚠️ Backward Compatibility & Breaking Changes

#### Fully Backward Compatible (Default Behavior)
- **Default behavior unchanged**: `enable_event=False` is the default, existing code requires no modifications
- All existing APIs remain unchanged
- Type system refactoring does not affect runtime behavior

#### Optional Breaking Changes (When Using New Features)

1. **Type Imports** (Optional):
   - If you were importing types from `SimpleLLMFunc.type.decorator`, you need to update imports:
     - `HistoryList` is now in `SimpleLLMFunc.type.hooks`
   - Most users are not affected as these are internal types

2. **Event Stream Return Type** (When `enable_event=True`):
   - When `enable_event=True`, the return type changes from `AsyncGenerator[Tuple[Any, MessageList], None]` to `AsyncGenerator[ReactOutput, None]`
   - Use `responses_only()` helper to maintain backward compatibility:
     ```python
     # Old way (still works with enable_event=False)
     async for response, messages in my_chat("Hello"):
         ...
     
     # New way (with enable_event=True)
     async for output in my_chat("Hello"):
         if output.type == "response":
             response, messages = output.response, output.messages
     ```

3. **Type Annotations** (For Type Checkers):
   - If you use type checkers (mypy, pyright), you may need to update type hints
   - The framework now uses more specific types from OpenAI SDK

### 🔮 Future Plans

- **v0.5.1**: `enable_event=True` will become the default
- **v0.5.2**: Remove `enable_event` parameter, always enable event stream

### Migration Guide

If you encounter any issues after upgrading:

1. **Check your imports**: If you import internal types, update them according to the breaking changes section
2. **Test with `enable_event=False`**: The default behavior is unchanged, so existing code should work
3. **Gradually adopt Event Stream**: Enable `enable_event=True` only when you need observability features
4. **Use helper functions**: `responses_only()` can help maintain compatibility when using event stream

---

## 0.4.2 Release Notes

### Refactoring

1. **ReAct Engine Return Type Enhancement**: Modified the ReAct loop entrypoint at that time to return both response and message history in streaming mode.
   - Changed return type from `AsyncGenerator[Any, None]` to `AsyncGenerator[Tuple[Any, List[Dict[str, Any]]], None]`
   - Now yields `(response, current_messages.copy())` instead of just `response`
   - Creates a copy of `current_messages` to avoid modifying the original list
   - Updated related test files to adapt to the new return type

---

## 0.4.1 Release Notes

### Features

1. **Gemini 3 Pro Preview Support**: Added `reasoning_details` field support to enable compatibility with Google Gemini 3 Pro Preview model under OpenAI-compatible interface.

2. **Reasoning Details Extraction**: 
   - Added `ReasoningDetail` type definition in `extraction.py`
   - Implemented extraction functions for both streaming and non-streaming responses
   - Support for extracting reasoning details from message objects (both dict and object formats)

3. **Message Type Enhancement**: Extended message type definitions in `message.py` to include `reasoning_details` field support.

4. **ReAct Engine Integration**: Integrated reasoning details extraction and propagation in the ReAct engine for tool call workflows.

### Examples

- Updated example files (`llm_function_pydantic_example.py`, `parallel_toolcall_example.py`, `llm_chat_raw_tooluse_example.py`) to use `gemini-3-pro-preview` model.

---

## 0.4.0 Release Notes

### Major Refactoring

1. **Modular Architecture Restructuring**: Completely refactored the base module, splitting messages, tool_call, and type_resolve into dedicated sub-modules for better code organization and maintainability.

2. **Decorator Logic Step-based Implementation**: Refactored decorator logic into a steps-based architecture within the `llm_decorator` module, improving code clarity and extensibility.

3. **Type System Enhancement**: Introduced new type support modules including decorator types and multimodal type support, expanding framework capabilities.

4. **Type Resolution System Refactoring**: Comprehensive refactoring of the type resolution system to enhance functionality support and improve type inference accuracy.

### Features

1. **Enhanced Tool Call Execution**: Improved tool call execution mechanism with extended support for multimodal interactions, enabling richer LLM interactions.

2. **Multimodal Type Support**: Added comprehensive multimodal type support throughout the framework for better handling of diverse content types.

### Bug Fixes

1. Fixed system prompt nesting issues when building multi-model content.

### Testing

Added extensive test coverage for refactored modules to ensure stability and reliability.

---

## 0.3.2.beta2 Release Notes

1. Remove dependence: `nest-asyncio`

2. Fix document error about `provider.json`

## 0.3.2.beta1 Release Notes

1. Better tool call tips in system prompt.

2. Better compound type annotations in prompt.

## 0.3.1 Release Notes

1. Added dynamic template parameter support: The `llm_function` decorator now supports passing `_template_params` to dynamically set DocString template parameters. This allows developers to create a single function that can adapt to various use cases, changing its behavior by passing different template parameters at call time.

2. Integrated Langfuse support: You can now configure `LANGFUSE_BASE_URL`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_PUBLIC_KEY` to send logs to Langfuse for tracing and analysis.

3. Added multilingual support: The English README has been updated, now supporting both Chinese and English.

4. Added parallel tool calling support.

5. Fully native async implementation: All decorators are now implemented with native async support, completely dropping any sync fallback.

## 0.2.13 Release Notes

1. Added the `return_mode` parameter (`Literal["text", "raw"]`) to the `llm_chat` decorator, allowing you to specify the return mode. You can now return either the raw response or text. This is designed to better display tool call information when developing Agents.

2. Improved code type annotations.

-----

## 0.2.12.2 Release Notes

1. Added a `py.typed` file to the framework package to support type checking.
