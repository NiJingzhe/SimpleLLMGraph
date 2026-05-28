# PyRepl Runtime

## When to use `PyRepl`

Use `PyRepl` when the model needs:

- persistent Python execution across turns
- runtime primitive discovery through `runtime.*`
- image-producing code whose plots or `IPython.display.Image` results should be returned to the model as multimodal tool output
- durable self-reference context
- a forkable execution environment for sub-agents or parallel work

`PyRepl` is not just a one-shot shell command wrapper. It keeps state in a dedicated Python subprocess.

## Core workflow

Use this sequence whenever the model needs runtime help:

1. Discover capabilities with `runtime.list_primitives()`.
2. Inspect one contract with `runtime.get_primitive_spec(name)`.
3. Execute small, focused code snippets with `execute_code`.
4. Manage durable context through `runtime.selfref.context.*` only when you truly need context cleanup or durable experience.

## Self-reference basics

`PyRepl()` installs the built-in `selfref` backend by default.

Useful calls:

- `runtime.selfref.context.inspect()`
- `runtime.selfref.context.remember(text)`
- `runtime.selfref.context.forget(experience_id)`
- `runtime.selfref.context.compact(...)`

Good pattern:

- Keep a stable memory key such as `agent_main`.
- Append durable experiences only when they should survive future turns.
- Keep normal per-turn reasoning out of durable context; compact it into one structured assistant summary when a milestone ends.
- `runtime.selfref.context.compact(...)` queues that summary first. If the current turn continues after the tool batch, the next same-turn LLM step sees the compacted context; if not, finalize still commits it into the returned history.

## Forking

Fork-related primitives let an agent split work into child contexts.

Typical pattern:

1. Discover fork contract details with `runtime.get_primitive_spec("selfref.fork.spawn")`.
2. Spawn isolated subtasks.
3. Gather results with `runtime.selfref.fork.gather_all()`.

Use forks only for concrete, isolated, verifiable subtasks. Do not fork tiny inline work.

Important fork context rules:

- A child fork inherits the pre-fork context snapshot, not the parent's pending fork tool-call scene.
- Write the child task as execution guidance for a sub-agent that already exists; the child does not need to re-decide whether the fork step was correct.
- `runtime.selfref.fork.gather_all()` returns `dict[fork_id -> ForkResult]`.
- Compact fork results include `status`, `response`, `result`, `memory_key`, `history_count`, and `history_included`.
- Check `status` first, then read `response` or `result`. Use `include_history=True` only when the full child history is actually needed.

## Safety rules

- Execute small code blocks, not giant scripts.
- Print checkpoints and short summaries instead of dumping huge objects.
- Prefer runtime discovery before assuming a primitive contract.
- Treat REPL state as persistent and cumulative across calls.
- PyRepl isolates worker stdio by default: child process stdout/stderr and direct fd writes are captured into tool output instead of inheriting the host terminal.

## Important reset behavior

`reset_repl` clears REPL variables, but it does not wipe installed runtime backends or self-reference context. If you need to forget durable experience or compact stale working transcript, use the `runtime.selfref.context.*` APIs directly.

When `llm_chat` is using a bound `SelfReference`, the decorator keeps the active turn state and stored context synchronized for you. Prefer `runtime.selfref.context.inspect()/remember()/forget()/compact()` over manual history surgery.
