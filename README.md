# SimpleLLMFunc

SimpleLLMFunc is being rebuilt around an explicit, context-centric execution
kernel. A model call is external work requested by a declarative `Loop`, not
hidden control flow inside an Agent object.

The only L1 cycle is:

```text
Step -> Resolve -> Reduce -> Store
```

- `LoopPolicy.step(state)` purely calculates an inspectable `Step` and its `Effect`s.
- `Runtime.resolve_step(step)` performs external model or tool work.
- `LoopPolicy.reduce(...)` deterministically calculates a `Reduction`.
- `Store.commit(step, resolutions, reduction)` atomically advances semantic state.

Semantic Events, pending Effects, terminal Resolutions, and operational UI
progress are separate types. Replaying an Event never executes a tool.

## Install

The project requires Python 3.12 or newer.

```bash
uv sync
```

Use `provider.example.json` as the shape for local provider configuration. Keep
real `provider.json` files untracked.

## L0 Provider Call

The lowest layer accepts an exact provider-neutral `Request`:

```python
from SimpleLLMFunc import OpenAICompatible
from SimpleLLMFunc.context.ir import Request, UserMessage

models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openai"]["gpt-4o-mini"]

request = Request(
    model=llm.model_name,
    messages=[UserMessage(content="Why use an append-only event log?")],
)
completion = await llm.chat(request)
print(completion.choices[0].message.content)
```

L0 has no Loop or Event log. The caller supplies the complete Request.

## Declarative Loop

Normal applications declare a standard model/tool `Loop`; they do not write a
context compiler or model/tool reducer:

```python
from SimpleLLMFunc import Loop, tool


@tool
def read_file(path: str) -> str:
    """Read one workspace file."""

    return workspace.read(path)


loop = Loop(
    model=llm.model_name,
    system_prompt="Inspect the project, make focused changes, and verify them.",
    tools=[read_file],
    max_model_calls=12,
)
```

`@tool` derives the provider schema from the function name, docstring, type
annotations, and defaults. Tool dependencies are captured with ordinary
closures. The standard `Loop` supplies Event projection, exact
`CompiledContext` construction, provenance, model/tool Steps, deterministic
reduction, result materialization, limits, and failure handling.

Tool returns are validated against their declared annotation, serialized to
JSON, restored through the same annotation, and rejected if that round trip is
not lossless. User-defined return types must be Pydantic models; nested fields
must also have bounded, JSON-serializable types rather than `Any` or `object`.
Each tool can define a synchronous result compiler that receives the exact
typed return value plus an isolated copy of the semantic Events that existed
before execution:

```python
from pydantic import BaseModel, ConfigDict

from SimpleLLMFunc import EventView, ToolResult, tool


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: tuple[str, ...]


def compile_search(result: SearchResult, events: EventView) -> ToolResult:
    prior_calls = len(events)
    return ToolResult(
        content=f"Observed {len(result.matches)} matches after {prior_calls} events."
    )


@tool(result_compiler=compile_search)
def search(query: str) -> SearchResult:
    """Search the project index."""

    return SearchResult(matches=index.search(query))
```

The pre-execution Event snapshot, compiled `ToolResult`, tool/compiler revision,
and JSON return value are all journaled. A resumed cycle therefore does not
execute the tool or its result compiler again.

### Multimodal Tool Results

Use the framework's `Image` type anywhere inside a Pydantic tool return model.
It supports public URLs, existing base64 image data, bytes, and immediate local
file encoding:

```python
from SimpleLLMFunc import Image


class Screenshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    caption: str
    image: Image


@tool
def screenshot() -> Screenshot:
    """Capture the current application view."""

    return Screenshot(
        caption="Current dashboard",
        image=Image.from_path("artifacts/dashboard.png"),
    )
```

Available constructors are `Image.from_url(...)`, `Image.from_base64(...)`,
and `Image.from_path(...)`. `from_path` reads the file immediately and stores a
`data:image/...;base64,...` URL, so later context compilation performs no file
I/O.

Native provider tool messages cannot carry images consistently. If any return
type in a tool-call batch contains `Image`, the context compiler removes that
batch's native tool-call projection and emits ordinary assistant/user pairs for
all results in the batch. This keeps both Chat Completions and Responses input
protocols valid. A custom result compiler may return `ToolResult(content=[...])`
with `InputTextPart` and `Image` values to control that user-side multimodal
message explicitly.

The declaration does not hide L1. Its `initialize`, `step`, `reduce`, and
`result` methods implement `LoopPolicy`, so a session or context editor can
inspect a pending `ModelCallEffect.context`, derive a modified
`CompiledContext`, approve or defer Effects, and persist the same public cycle.
Applications that need a different state machine can implement `LoopPolicy`
directly.

`DefaultController.run(..., prepare_step=...)` applies an async Step editor
before the first journal write. The returned Step is revalidated and becomes
the exact intent that is resolved and committed; already-recorded pending Steps
remain immutable during resume.

## Manual L1 Loop

Applications can own every execution boundary. The Store records intent before
external work and binds the committed Reduction to the exact Step and Resolution
batch:

```python
state = loop.initialize(value, run_id="run-1")
await store.create(state)

while loop.can_step(state):
    step = loop.step(state)
    loop.validate_step(state, step)
    await store.record_step(step, expected_revision=state.revision)

    resolutions = await runtime.resolve_step(step)
    await store.record_resolutions(step, resolutions)

    reduction = loop.reduce(state, step, resolutions)
    state = await store.commit(
        step,
        resolutions,
        reduction,
        expected_revision=state.revision,
    )

result = loop.result(state)
```

Use one `CancellationToken` as the abort signal for the whole cycle. It can
interrupt an operator wait, model request, provider stream, retry delay, or
tool-resolution boundary while the cancelled Resolution is still journaled and
reduced deterministically:

```python
from SimpleLLMFunc import CancellationToken

cancellation = CancellationToken()
run = asyncio.create_task(
    runtime.resolve_step(step, cancellation=cancellation)
)
cancellation.cancel()
resolutions = await run
```

Provider adapters accept the same token through `LLM_Interface.chat(...)` and
`chat_stream(...)`. Cancellation closes the local HTTP/stream operation; it is
not proof that a provider has stopped server-side work.

`DefaultController` drives this same `LoopPolicy` sequence. It is not a second
hidden runtime. If a process stops after recording a Step or Resolution batch,
`DefaultController.resume()` reuses those journaled artifacts instead of
re-executing completed work.

The reference runtime is at-least-once across the crash window between an
external side effect and Resolution journaling. Stable Effect IDs and
idempotency keys support deduplication where an external system provides it;
the framework does not claim exactly-once execution for arbitrary tools.

## Validation

```bash
uv run pytest
uv run pyright SimpleLLMFunc tests examples
uv run ruff check SimpleLLMFunc tests examples
```

The test suite includes manual/packaged equivalence, revision conflicts,
cycle-evidence binding, interrupted-cycle resume, active cancellation, context
provenance, and malformed provider output.
