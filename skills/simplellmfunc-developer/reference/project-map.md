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
- `llm_chat_decorator.py`
- `selfref_sync.py` (legacy compatibility shim; active SelfRef session binding is in `LLMChat` + `runtime/selfref/session.py`) (legacy compatibility shim; active SelfRef session binding is in `LLMChat` + `runtime/selfref/session.py`)
- `steps/common/`
- `steps/function/`
- `steps/chat/`
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

- `runtime/selfref/context_ops.py`: pure parse/render/canonicalize helpers for context messages
- `runtime/selfref/state.py`: persistent memory state, mutation, validation, compaction queueing, and fork-aware behavior

### `SimpleLLMFunc/builtin/`

End-user built-ins:

- `pyrepl.py`
- `file_tools.py`
- `self_reference.py`

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
- Runtime primitive issue: start in `runtime/primitives.py`, then inspect `builtin/pyrepl.py` or the relevant builtin backend.
- Selfref context parse/render issue: start in `runtime/selfref/context_ops.py` before changing stateful code.
- Event-stream issue: start in `hooks/`, then inspect decorator step modules.
- ReAct termination / hook-order issue: start in `base/ReAct.py` and follow the phase helpers plus `before_finalize` call sites.
- Provider transport issue: start in the relevant adapter under `interface/` (`openai_compatible.py` or `openai_responses_compatible.py`) plus related tests.
- Selfref fork context-construction issue: start in `runtime/selfref/state.py`; check how pre-fork history is stripped/materialized before touching decorator wrappers.
- User-facing docs mismatch: source code and tests win; then patch `mintlify_docs/` and maybe `README.md`.
