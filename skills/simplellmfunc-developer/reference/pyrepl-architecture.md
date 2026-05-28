# PyRepl Architecture Notes

Use this reference when changing PyRepl, runtime primitives, tool execution, event streams, or multimodal tool outputs.

## Core mental model

`PyRepl` is a public facade over a persistent IPython worker process. The main process owns the tool protocol, ReAct events, primitive registry, SelfRef state, and transcript mutations. The worker process owns Python code execution and the persistent user namespace.

```text
LLM tool call: execute_code(...)
        |
        v
main process: PyRepl facade / ReAct tool scheduler / ToolEventEmitter
        |
        | multiprocessing queues: command_queue / event_queue
        v
worker process: IPython InteractiveShell + persistent namespace
        |
        v
runtime proxy: runtime.selfref.* / runtime.<pack>.* primitive RPC back to main process
```

Important boundaries:

- Worker code must not mutate ReAct transcript/history directly.
- Runtime primitives are not native LLM tool calls. They are host-registered callables invoked inside `execute_code` through the injected `runtime` proxy.
- Tool results enter model-visible history through `ContextMutation` objects, usually `ToolResultMutation` or `MultimodalToolResultMutation`.

## File map

- `SimpleLLMFunc/builtin/pyrepl.py`: public `PyRepl` facade composed from focused mixins.
- `SimpleLLMFunc/builtin/pyrepl_worker_client.py`: subprocess and multiprocessing queue lifecycle.
- `SimpleLLMFunc/builtin/pyrepl_worker_mixin.py`: facade-compatible wrappers around the worker client.
- `SimpleLLMFunc/builtin/pyrepl_execution.py`: main-process `execute()` / `reset()` orchestration, timeout handling, event polling, event-emitter forwarding, audit payload assembly.
- `SimpleLLMFunc/builtin/pyrepl_worker.py`: worker-process IPython shell, fd-level and Python-level stdout/stderr capture, `input()` hook, primitive RPC transport, image artifact capture.
- `SimpleLLMFunc/builtin/pyrepl_primitive_host.py`: main-process primitive registry/backend host and primitive call execution.
- `SimpleLLMFunc/builtin/pyrepl_tools.py`: `execute_code` / `reset_repl` tool definitions, output formatting, and artifact-to-multimodal return conversion.
- `SimpleLLMFunc/runtime/worker_proxy.py`: worker-side dynamic `runtime` proxy and dotted namespace builder.
- `SimpleLLMFunc/runtime/primitives.py`: `PrimitivePack`, `PrimitiveRegistry`, `PrimitiveCallContext`, primitive metadata/spec contracts.

## Worker lifecycle and IPC

The main process starts the worker through `PyReplWorkerClient.ensure_worker(...)` using `multiprocessing.get_context("spawn")`.

Startup sequence:

1. Create `command_queue` and `event_queue`.
2. Spawn a daemon worker process targeting `run_pyrepl_worker(command_queue, event_queue, working_directory)`.
3. Worker constructs `_PyReplWorker`, isolates fd 0/1/2, detaches from the controlling terminal on POSIX when possible, then emits `EVENT_WORKER_READY`.
4. Main process waits up to 10 seconds for `EVENT_WORKER_READY`, while preserving unexpected startup events in `prefetched_events`.

Main-to-worker command messages are plain dicts:

```python
{"type": "execute", "exec_id": "...", "code": "...", ...}
{"type": "reset", "request_id": "..."}
{"type": "input_reply", "request_id": "...", "value": "..."}
{"type": "primitive_result", "call_id": "...", "ok": True, "result": ...}
{"type": "shutdown"}
```

Worker-to-main event messages are also plain dicts:

```python
{"type": "stdout", "exec_id": "...", "text": "..."}
{"type": "stderr", "exec_id": "...", "text": "..."}
{"type": "input_request", "exec_id": "...", "request_id": "...", "prompt": "..."}
{"type": "primitive_call", "exec_id": "...", "call_id": "...", "name": "...", "args": [...], "kwargs": {...}}
{"type": "execute_result", "exec_id": "...", "return_value": ..., "artifacts": [...], ...}
```

`PyReplWorkerMixin._start_execute_worker_command(...)` is the start boundary for one execution. It ensures the worker is alive, drains stale events, then sends `COMMAND_EXECUTE`.

`PyReplExecutionMixin.execute(...)` is the main orchestration loop. It polls worker events with `_receive_worker_event(...)`, which uses `asyncio.to_thread(event_queue.get, ...)` so queue reads do not block the asyncio loop.

Each execution has a fresh `exec_id`. The main process ignores stdout/stderr/input/result events whose `exec_id` does not match the active execution.

## Code execution in the worker

The worker hosts an `IPython.core.interactiveshell.InteractiveShell` and keeps its `user_ns` as the persistent namespace.

`_PyReplWorker._execute_python_code(...)` implements IPython-like last-expression behavior:

1. Transform the cell with `InteractiveShell.transform_cell(...)`.
2. Parse as an AST module.
3. If the last node is an expression, split it from the body.
4. `exec(...)` the body nodes in the persistent namespace.
5. `eval(...)` the last expression in the same namespace.
6. Return `repr(result)` unless the result is `None` or recognized as an image artifact.

This preserves normal REPL state across calls: variables defined in one `execute_code` call remain available in the next call until `reset_repl` clears the namespace.

## stdout and stderr capture

stdout/stderr capture happens inside the worker process. It has two layers:

- fd-level capture redirects worker fd `0` to `/dev/null`. During each execution, fd `1`/`2` point to execution-scoped worker-owned pipes. Reader threads drain those pipes and emit `EVENT_STDOUT` / `EVENT_STDERR` with that execution's `exec_id`. This captures `os.write(...)`, C extension writes, shell commands, and subprocess stdout/stderr.
- Between executions, worker fd `1`/`2` point to `/dev/null`. Long-lived children keep their old execution pipe; late output is drained and dropped instead of contaminating later snippets or reaching the host terminal.
- Python-level capture replaces `sys.stdout` / `sys.stderr` during each execution so normal Python `print()` output still streams line-by-line and is not duplicated by fd capture.

On POSIX, worker startup also calls `os.setsid()` when possible before user code runs. This prevents the worker and descendants from keeping the host terminal as their controlling terminal.

Before executing user code, `_handle_execute(...)` temporarily replaces:

```python
sys.stdout = _LineCapture(on_stdout)
sys.stderr = _LineCapture(on_stderr)
```

`_LineCapture` is a line-buffered `io.TextIOBase` adapter. It accumulates writes until it sees `\n`, then emits complete lines through its callback. `flush()` emits any remaining partial line.

Callbacks emit worker events:

```python
def on_stdout(text: str) -> None:
    self._emit(EVENT_STDOUT, exec_id=exec_id, text=text)

def on_stderr(text: str) -> None:
    self._emit(EVENT_STDERR, exec_id=exec_id, text=text)
```

The main process receives these events in `PyReplExecutionMixin.execute(...)`, appends text to `stdout_parts` / `stderr_parts`, and forwards live custom events when an event emitter is available.

There are two output paths:

```text
worker output capture -> event_queue -> stdout_parts/stderr_parts -> final execute result / tool summary
worker output capture -> event_queue -> ToolEventEmitter -> ReAct CustomEvent stream
```

## ToolEventEmitter path

`PyRepl` does not own a global event stream. The ReAct tool scheduler creates one `ToolEventEmitter` per tool call.

Flow:

1. `schedule_tool_batch(...)` creates `ToolEventEmitter` with trace, function, iteration, tool name, and tool call id metadata.
2. `execute_single_tool_call_result(...)` detects that `execute_code` accepts an `event_emitter` parameter and injects it.
3. `PyReplExecutionMixin.execute(...)` forwards worker events by calling `event_emitter.emit(...)`.

Emitted PyRepl custom events:

- `kernel_stdout`: `{"text": "..."}`
- `kernel_stderr`: `{"text": "..."}`
- `kernel_input_request`: `{"request_id": "...", "prompt": "...", "idle_timeout_seconds": ...}`

These become `CustomEvent` values in the `ReactOutput` stream. The final tool result remains separate from live event streaming.

## `input()` bridge

The worker temporarily replaces `builtins.input` with `input_hook` during execution.

When user code calls `input(prompt)`:

1. Worker generates a `request_id`.
2. Worker emits `EVENT_INPUT_REQUEST` with the prompt and idle timeout.
3. Worker waits for a matching `COMMAND_INPUT_REPLY`.
4. Main process either reads from real stdin when no event emitter exists, or exposes the request through `kernel_input_request` so UI code can call `PyRepl.submit_input(request_id, value)`.

The main process resets the active execution deadline when input is accepted. This means waiting for user input does not consume the normal execution timeout window.

## Runtime primitive injection and RPC

The worker never directly imports or owns host primitive backends. Instead, the worker namespace receives a global `runtime` object from `WorkerRuntimeProxy`.

Injection point:

- `_PyReplWorker._sync_runtime_binding(...)` sets `self._namespace["runtime"] = self._runtime_proxy` when runtime support is enabled.

Dynamic call shape:

```python
runtime.selfref.context.inspect()
runtime.selfref.fork.spawn(...)
runtime.call("selfref.context.remember", "fact")
```

`WorkerRuntimeProxy.__getattr__` returns a `WorkerRuntimeNamespace`; each dotted attribute extends the path. Calling the namespace invokes `transport.call_primitive(name=path, args=list(args), kwargs=kwargs)`.

The worker transport is `_PyReplWorker.call_primitive(...)`:

1. Generate `call_id`.
2. Emit `EVENT_PRIMITIVE_CALL` with `name`, `args`, and `kwargs`.
3. Block inside the worker waiting for matching `COMMAND_PRIMITIVE_RESULT`.
4. Return `result` if `ok=True`, otherwise raise a reconstructed error.

The main process handles primitive calls in `PyReplExecutionMixin.execute(...)` when it sees `EVENT_PRIMITIVE_CALL`. It calls `PyReplPrimitiveHostMixin._execute_primitive_call(...)`, sends the response back as `COMMAND_PRIMITIVE_RESULT`, then continues polling.

Primitive handlers receive a `PrimitiveCallContext` containing:

- `primitive_name`
- `call_id`
- `execution_id`
- `event_emitter`
- metadata such as `pyrepl_instance_id`
- the host `repl`
- the primitive `registry`
- resolved backend fields when applicable

This lets primitives emit events, access host backends, and queue SelfRef changes without directly mutating the worker namespace or ReAct transcript.

## Primitive packs and discovery

`PrimitivePack` is the declarative extension unit. A pack has:

- `name`: namespace used in `runtime.<name>.*`
- `backend_name`: host backend lookup name
- `backend`: host-side object/state
- `guidance`: prompt-injection guidance for the model
- primitive entries with handler and public contract metadata

Install path:

```python
pack = repl.pack("demo", backend=object(), guidance="...")

@pack.primitive("ping")
def ping(ctx): ...

repl.install_pack(pack)
```

`PyReplPrimitiveHostMixin` registers pack entries into `PrimitiveRegistry` and stores the backend under the pack/backend namespace. Built-in SelfRef primitives are installed by default unless disabled for internal cloning/tests.

Discovery primitives are exposed through the same runtime proxy:

- `runtime.list_primitives()`
- `runtime.list_primitive_specs(...)`
- `runtime.get_primitive_spec(name)`
- `runtime.list_backends()`

`_build_execute_tool_prompt_injection(...)` renders the installed primitive pack guidance into the tool-owned system prompt block so the model knows how to discover and call runtime primitives.

## Image artifact capture

Image capture happens inside `_PyReplWorker._execute_python_code(...)` and returns structured artifacts in the final `execute_result` event.

Supported sources:

- explicit `display(Image(...))`
- image-rich last expressions such as `Image(filename="chart.png")`
- last expression returning `ImgPath(...)`
- last expression returning `ImgUrl(...)`
- rich repr methods such as `_repr_png_()` and `_repr_jpeg_()`
- IPython MIME bundles containing `image/png`, `image/jpeg`, `image/gif`, `image/bmp`, or `image/webp`

Worker artifact shape for local images:

```python
{
    "type": "image",
    "source": "display_data" | "return_value",
    "path": "/tmp/pyrepl_image_xxx.png",
    "mime_type": "image/png",
    "detail": "auto",
}
```

Worker artifact shape for URL images:

```python
{
    "type": "image",
    "source": "return_value",
    "url": "https://..." | "data:image/png;base64,...",
    "detail": "auto" | "low" | "high",
}
```

Local image data is written to a temp file by `_write_image_artifact(...)` instead of sending large binary payloads through the queue. The main process later wraps these paths as `ImgPath`.

`display(...)` handling has a special hook:

- Image objects are captured and suppressed from normal display output.
- Non-image objects are passed through to the original IPython `display`, preserving text/plain output.
- Mixed calls like `display(Image(...), "note")` capture the image and preserve the text output.

## Tool return conversion for images

`PyRepl.execute(...)` returns raw structured data including `artifacts`.

The `execute_code` tool path converts artifacts in `pyrepl_tools.build_execute_tool_return(...)`:

- no images: return the normal summary string
- images present: return `(summary, list[ImgPath | ImgUrl])`

The generic tool execution layer in `base/tool_call/execution.py` supports multimodal tool returns:

- `ImgPath`
- `ImgUrl`
- `list[ImgPath | ImgUrl]`
- `(str, ImgPath)`
- `(str, ImgUrl)`
- `(str, list[ImgPath | ImgUrl])`

These become `ExecutedToolCallResult(is_multimodal=True, messages=[user_multimodal_message])`.

Then `schedule_tool_batch(...)` converts that into `MultimodalToolResultMutation`. At compile time, `context_compile._append_multimodal_tool_result_mutation(...)` removes the original assistant tool call from the transcript and appends:

1. an assistant explanation message
2. a user message containing OpenAI-compatible multimodal content parts

This is intentional because normal OpenAI `tool` messages cannot directly carry image content.

## Provider adapter boundary

Chat Completions-style providers accept user content parts like:

```python
{"type": "text", "text": "..."}
{"type": "image_url", "image_url": {"url": "...", "detail": "high"}}
```

`OpenAIResponsesCompatible` must translate those into Responses input parts:

```python
{"type": "input_text", "text": "..."}
{"type": "input_image", "image_url": "...", "detail": "high"}
```

Keep this wire-format conversion inside `interface/openai_responses_compatible.py`; do not leak Responses-specific shapes into ReAct, PyRepl, or decorator code.

## Tests to run

For PyRepl worker/protocol changes:

```bash
uv run pytest tests/test_builtin/test_pyrepl.py
```

For tool multimodal return changes:

```bash
uv run pytest tests/test_base/test_tool_call tests/test_base/test_react_core_scheduler.py
```

For provider boundary changes:

```bash
uv run pytest tests/test_interface/test_openai_responses_compatible.py
```

For broader regression after PyRepl/tool changes:

```bash
uv run pytest tests/test_base/test_tool_call tests/test_base/test_react_core_scheduler.py tests/test_builtin/test_pyrepl.py tests/test_llm_function_decorator.py tests/test_llm_chat_decorator.py tests/test_interface/test_openai_responses_compatible.py
```

Run lint on touched Python files with:

```bash
uv run ruff check <changed-python-files>
```

## Common pitfalls

- Do not send image bytes directly through queues unless there is a strong reason. Prefer temp files plus artifact metadata.
- Do not let primitives edit ReAct transcript/history directly. Use host-side SelfRef APIs and mutations.
- Do not treat runtime primitives as native LLM tools. The model calls them inside `execute_code` through `runtime.*`.
- Keep worker event messages JSON-ish / pickle-safe. They cross a multiprocessing queue.
- Preserve `exec_id` filtering; otherwise stale worker events can corrupt the current execution.
- Preserve stdout/stderr final aggregation even when streaming custom events are emitted.
- Preserve non-image `display(...)` behavior when adding image capture hooks.
- Keep provider-specific multimodal wire-format conversions in provider adapters.
- When `too_long_to_file=True` and a tool returns `(summary, images)`, apply long-output truncation to the text summary as well as plain string results.
