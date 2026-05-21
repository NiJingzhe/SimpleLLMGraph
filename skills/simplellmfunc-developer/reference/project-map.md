# Project Map

## Top-level structure

```text
SimpleLLMFunc/
├── SimpleLLMFunc/   # framework package
├── tests/           # pytest suite
├── examples/        # runnable examples
├── mintlify_docs/   # Mintlify documentation source
├── spec/            # repo-specific maps and conventions
└── skills/          # agent skills
```

## Package map

### `SimpleLLMFunc/llm_decorator/`

Primary user entrypoints:

- `llm_function_decorator.py`
- `llm_chat_decorator.py` (public facade / `LLMChat` callable)
- `chat_call_context.py`
- `chat_selfref.py`
- `chat_toolkit.py`
- `chat_types.py`
- `selfref_sync.py` (legacy compatibility shim; active SelfRef session binding is in `LLMChat` + `runtime/selfref/session.py`)
- `invocation_spec.py`
- `invocation_builder.py`
- `prompt_contract.py`
- `signature.py`
- `utils/tools.py`

This layer turns typed Python call signatures into prompt-building, ReAct execution, response parsing, and lifecycle-aware integrations such as `llm_chat` <-> `SelfReference` synchronization.

### `SimpleLLMFunc/base/`

Core engine and lower-level primitives:

- `ReAct.py`
- `post_process.py`
- `messages/`
- `tool_call/`
- `type_resolve/`

When behavior seems "framework magical," the truth usually lives here.

### `SimpleLLMFunc/runtime/`

Runtime primitive system:

- primitive registry
- primitive contracts and docstring parsing
- backend lifecycle and fork behavior
- selfref context state and pure context transforms

This layer powers the `runtime.*` namespace used inside `PyRepl`.

Key current selfref split:

- `runtime/selfref/state.py`: `SelfReference` public facade, lifecycle, compatibility exports
- `runtime/selfref/store.py`: durable history/source store
- `runtime/selfref/active_turn.py`: active key/fork/toolkit/template contextvars and active ReAct state lookup
- `runtime/selfref/mutations.py`: pending compaction/context/destructive mutation queues
- `runtime/selfref/memory_api.py`: `self_reference.memory[...]` handle/proxy API
- `runtime/selfref/context_memory.py`: context snapshots, experience CRUD, compaction commit, direct memory editing
- `runtime/selfref/agent_binding.py`: recursive agent callable binding
- `runtime/selfref/fork_manager.py`: fork/spawn/gather lifecycle and result materialization
- `runtime/selfref/fork_utils.py`: fork helpers/constants
- `runtime/selfref/context_ops.py`: pure parse/render/canonicalize helpers for context messages

### `SimpleLLMFunc/builtin/`

End-user built-ins:

- `pyrepl.py` (public facade)
- `pyrepl_execution.py`
- `pyrepl_worker_client.py`
- `pyrepl_worker_mixin.py`
- `pyrepl_primitive_host.py`
- `pyrepl_tools.py`
- `pyrepl_audit.py`
- `pyrepl_input_bridge.py`
- `pyrepl_input_mixin.py`
- `file_tools.py`

These builtins expose high-level behavior on top of lower-level runtime and tool plumbing.

### `SimpleLLMFunc/hooks/`

Streaming and event infrastructure:

- event types
- event emitters
- event wrappers such as `EventYield` and `ResponseYield`
- abort signaling

### `SimpleLLMFunc/interface/`

Provider-facing integration:

- `llm_interface.py`
- `openai_compatible.py`
- `openai_responses_compatible.py`
- `key_pool.py`
- `token_bucket.py`

### `SimpleLLMFunc/logger/` and `SimpleLLMFunc/observability/`

Logging, trace propagation, and Langfuse support.

### `SimpleLLMFunc/utils/`

Support layers such as:

- `utils/tui/`
- `utils/stdio/`

## How to navigate changes

- Decorator behavior bug: start in `llm_decorator/`, then trace into `base/`.
- `llm_chat` + selfref sync issue: start in `llm_chat_decorator.py` (`LLMChat`) and `runtime/selfref/session.py`, then inspect `runtime/selfref/state.py` and the relevant decorator entrypoint.
- Tool schema or prompt-injection issue: start in `tool/tool.py` and `llm_decorator/utils/tools.py`.
- Runtime primitive issue: start in `runtime/primitives.py`, then inspect `builtin/pyrepl_primitive_host.py` or the relevant builtin backend.
- Selfref context parse/render issue: start in `runtime/selfref/context_ops.py` before changing stateful code.
- Selfref context memory or compaction issue: start in `runtime/selfref/context_memory.py`, then inspect `runtime/selfref/session.py`.
- Selfref fork lifecycle issue: start in `runtime/selfref/fork_manager.py` and `runtime/selfref/fork_utils.py`.
- Event-stream issue: start in `hooks/`, then inspect `base/react_loop.py` and decorator facade modules.
- ReAct termination / hook-order issue: start in `base/react_loop.py` plus `base/react_hooks.py` and `before_finalize` call sites.
- Provider transport issue: start in the relevant adapter under `interface/` (`openai_compatible.py` or `openai_responses_compatible.py`) plus related tests.
- User-facing docs mismatch: source code and tests win; then patch `mintlify_docs/` and maybe `README.md`.
