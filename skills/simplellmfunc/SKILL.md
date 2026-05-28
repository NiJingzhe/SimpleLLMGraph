---
name: simplellmfunc
description: "Use SimpleLLMFunc to build typed async LLM functions, chat agents, tools, event-stream consumers, and PyRepl/selfref workflows. Use when writing or editing app code that imports SimpleLLMFunc, configures provider.json, adds @llm_function/@llm_chat/@tool, chooses OpenAICompatible or OpenAIResponsesCompatible, or mounts PyRepl/FileToolset."
license: MIT
compatibility: "Python 3.12+ repo with async execution and OpenAI-compatible chat endpoints or OpenAI Responses API endpoints configured through provider.json."
metadata:
  project: SimpleLLMFunc
  version: "0.8.4"
---

# SimpleLLMFunc Usage

## When to use this skill
- Use this skill for application-level work built on top of SimpleLLMFunc.
- Use it when the task mentions `llm_function`, `llm_chat`, `tool`, `OpenAICompatible`, `OpenAIResponsesCompatible`, `provider.json`, `reasoning`, `PyRepl`, `SelfReference`, `FileToolset`, or the built-in TUI.
- Do not use this skill for framework-internal refactors; use `simplellmfunc-developer` for that.

## Core philosophy
- **LLM is Function**: treat the LLM call like a normal Python function call — signature, type hints, return value.
- **Prompt as Code**: put the prompt in the function docstring. Code and prompt are never separated.
- **Context-Centric**: each LLM request is compiled from invocation configuration, a base transcript/history, and internal runtime patches. `ContextMutation` is the internal patch protocol that prevents tools and SelfRef from directly modifying the live ReAct transcript.
- Keep orchestration in Python instead of hiding it in giant prompt strings.
- Prefer small, typed, composable building blocks.

## How system prompts are really built

This is critical for writing good prompts in SimpleLLMFunc: your docstring is important, but it is usually not the whole final system prompt.

### `llm_function`
- Your docstring is first treated as `function_description`.
- If you pass `_template_params`, the docstring is formatted before prompt construction.
- The framework then wraps that docstring inside a system template that also adds:
  - parameter type descriptions
  - return-type instructions
  - plain-text or XML output constraints depending on the return type
- If tools are mounted, the framework prepends a deduplicated `<tool_best_practices>` block before the main system prompt.
- Runtime argument values are not put in the system prompt; they go into the user prompt.
- For image input, keep the function-like style: declare explicit parameters as `ImgUrl`, `ImgPath`, or lists/unions of those types. Use `ImgUrl` for web/data URLs and `ImgPath` for local files.

Write `llm_function` docstrings as task policy, quality bar, constraints, and style guidance. Do not waste docstring space restating parameter schemas or low-level output formatting that the framework already injects.

### `llm_chat`
- Base system prompt is assembled from two sources:
  - `DataFromAgentConfig` (docstring + template params + tool specs)
  - `DataFromSelfRef` (if `self_reference_key` is set: experiences, summary, working messages)
- Then the framework prepends `<tool_best_practices>` when tools exist.
- Then it appends a `<must_principles>` block that tells the model to use native structured tool calls instead of writing fake tool calls in assistant text.
- Current turn data is added as the user message, not merged into the system prompt.
- For multimodal user turns, prefer `message: UserChatMessage` and construct content with `UserChatMessage.multimodal("text", ImgUrl(...), ImgPath(...))`. This keeps `llm_chat` as an Agent abstraction over one explicit user message instead of many loosely named image parameters.

Write `llm_chat` docstrings as stable assistant policy and long-lived behavior. Put current task content in the function call arguments, not in the docstring.

### Prompt-writing implications
- For `llm_function`, think: function contract + execution strategy.
- For `llm_chat`, think: assistant identity + durable rules.
- Put tool-usage advice in tool `best_practices` when possible, not only in the main docstring.
- If you need to durably change a chat agent's context mid-run, use self-reference context helpers such as `runtime.selfref.context.remember(...)` / `runtime.selfref.context.compact(...)` instead of trying to mutate old docstrings.
- When `llm_chat` is bound to `SelfReference`, the framework syncs turn state automatically through the `SelfRefSession` ReAct lifecycle hooks. Treat `runtime.selfref.context.*` as the supported way to change durable context instead of editing old history messages in place.

## Context-Centric architecture

The framework compiles each LLM request from three inputs:

1. invocation configuration: docstring prompt, template params, tool guidance, output contract, and SelfRef snapshot;
2. base transcript/history: previous messages plus the current user input;
3. internal runtime patches: `ContextMutation` objects produced by LLM calls, tools, SelfRef primitives, compaction, and abort/cancel handling.

The main internal rule is: **runtime side effects must not directly edit the live transcript**. They produce typed patches, and the compile boundary applies those patches in order before rendering the provider-facing message list.

This means same-turn selfref changes (remember, forget, compact) take effect at the next compile boundary, not immediately. It does not mean that docstrings, template params, tool schemas, or initial history are produced by mutations.

### SelfRef: Meta Context Editing

SelfRef enables an agent to read and edit durable context at runtime, while respecting the internal transcript patch boundary:

| Operation | What it does | Internal patch produced |
|-----------|-------------|-------------------------|
| **Remember** | Add durable experience that survives across turns | `ExperienceRememberMutation` |
| **Forget** | Remove experience by ID | `ExperienceForgetMutation` |
| **Compact** | Replace working transcript with a structured summary | `ContextSummaryMutation` |
| **Fork** | Spawn a child agent with inherited context snapshot | (sub-agent runs independently) |

These operations take effect at the next compile boundary — SelfRef cannot bypass compile to modify the live transcript directly.

## Fast start modes
- Project mode: load models from `provider.json` with `OpenAICompatible.load_from_json_file(...)` or `OpenAIResponsesCompatible.load_from_json_file(...)` when you have shared config or multiple models.
- Instant mode: write a tiny `python - <<'PY'` snippet, construct `APIKeyPool` plus the right adapter (`OpenAICompatible` or `OpenAIResponsesCompatible`) directly, decorate one function, and call it immediately.
- Prefer instant mode for quick shell usage, generated scripts, demos, and one-off local agents.

## Interface choice
- Use `OpenAICompatible` for normal OpenAI-style chat/completions endpoints.
- Use `OpenAIResponsesCompatible` for OpenAI Responses API endpoints when you want the Responses transport while keeping the same decorator surface.
- `OpenAIResponsesCompatible` uses the same `provider.json` shape and direct-construction shape as `OpenAICompatible`.
- When you use `OpenAIResponsesCompatible`, the framework still builds normal chat/system messages first; the adapter maps the chosen system prompt to Responses `instructions` and forwards `reasoning={...}` kwargs.
- Keep prompt authoring the same across both adapters. Do not rewrite docstrings around raw Responses wire format.

## Export the packaged skill

After installing `SimpleLLMFunc`, export the bundled Agent Skills with:

```bash
simplellmfunc-skill usage ~/.config/opencode/skills
simplellmfunc-skill developer ~/.config/opencode/skills
```

- `usage` exports the `simplellmfunc` folder.
- `developer` exports the `simplellmfunc-developer` folder.
- The second argument is the parent directory that receives the exported skill folder.
- Add `--force` if you need to overwrite an existing exported copy.

## Configuration essentials

### Minimal `provider.json` shape

SimpleLLMFunc expects `provider.json` to be:

```json
{
  "openrouter": [
    {
      "model_name": "z-ai/glm-5",
      "api_keys": ["sk-key-1", "sk-key-2"],
      "base_url": "https://openrouter.ai/api/v1",
      "max_retries": 5,
      "retry_delay": 1.0,
      "rate_limit_capacity": 20,
      "rate_limit_refill_rate": 3.0
    }
  ]
}
```

- Top level = provider id -> model config list.
- Lookup shape after loading = `providers[provider_id][model_name]`.
- Start from `examples/provider_template.json` when possible.

### How to organize `provider.json`

Treat `provider.json` as the canonical project-level model routing table, not as a random dump of keys.

Recommended organization rules:

- Group by provider first, then keep a short list of model configs under that provider.
- Keep `model_name` values stable and unique within one provider.
- Put multiple keys under the same hot model in `api_keys` instead of duplicating the model entry.
- Tune `max_retries`, `retry_delay`, `rate_limit_capacity`, and `rate_limit_refill_rate` per model, not once globally.
- Keep the file focused on runtime model access concerns only: provider, model, keys, endpoint, retry, and rate limits.
- Use `OpenAICompatible.load_from_json_file(...)` once near application setup, then pass resolved model handles into decorators.

Recommended mental model:

- `provider.json` decides **what model surface is available**.
- typed decorators and tools decide **how tasks are expressed**.
- your harness decides **what context reaches the model for one task**.

### `.env` and environment variables

The framework mainly reads `.env` / environment variables for logging and optional Langfuse observability.

```bash
LOG_LEVEL=WARNING
LOG_DIR=logs

LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_EXPORT_ALL_SPANS=true
LANGFUSE_ENABLED=true
```

- Precedence is: runtime environment variables -> `.env` -> framework defaults.
- Recommended default: `LOG_LEVEL=WARNING` to reduce noisy framework logs during normal agent usage.
- `provider.json` is the main model/provider config file; `.env` is not the primary place for provider definitions in project mode.
- For shell-first one-offs, direct constructor snippets are still fine even without `provider.json`.

## Default workflow
1. Choose the right surface:
   - `@llm_function` for one typed call.
   - `@llm_chat` for multi-turn agent behavior.
   - `@tool` for external capabilities the model may call.
   - `PyRepl` when the model needs persistent code execution or runtime primitives.
2. Define everything as `async def`.
3. Write a precise docstring prompt and leave the body as `pass`.
4. Use explicit parameter types and a typed return value; prefer Pydantic for structured outputs.
5. Build the model either from `provider.json` or directly with `APIKeyPool` + `OpenAICompatible` / `OpenAIResponsesCompatible`.
6. Add `toolkit=[...]` only when the task truly needs tools.
7. Validate with a focused example or event consumer.

## Strong typing and Pydantic (Recommended default)

Prefer explicit, typed function contracts over loose string-in/string-out prompting.

Recommended rules:

- Use narrow, explicit parameter types.
- Use typed return values whenever the output shape matters.
- Prefer Pydantic models for structured outputs that must be stable, inspectable, or reusable.
- Let the framework derive structured parsing from the Python return type instead of asking the model to hand-roll JSON.
- Keep the docstring focused on task intent, quality bar, constraints, and edge cases, not schema duplication.

Good pattern:

```python
from pydantic import BaseModel, Field

from SimpleLLMFunc import llm_function


class SearchSummary(BaseModel):
    answer: str = Field(description="Direct answer to the user's question")
    evidence: list[str] = Field(description="Supporting evidence bullets")
    confidence: float = Field(description="0.0 to 1.0 confidence score")


@llm_function(llm_interface=llm)
async def summarize_search_results(query: str, snippets: list[str]) -> SearchSummary:
    """
    Answer the query using the provided snippets.

    Prefer concise claims backed by evidence.
    If the snippets are insufficient, lower confidence and say so explicitly.
    """
    pass
```

Prefer this over:

- returning a free-form string and parsing it manually later
- asking the model to invent ad hoc JSON output shapes in the docstring
- mixing many unrelated fields into one loose `dict[str, Any]` unless the shape is genuinely variable

Use plain `str` returns only when free-form text is truly the product.

## Harness Engineering: context planning comes first

When building on SimpleLLMFunc, think in terms of harness engineering.

Core philosophy:

- the main job is context planning
- an agent is not a person; it is a method for constructing the right context for each reasoning step
- at every step, the model should see the shortest clean context that is still complete for the current task
- the system should be designed so this context can be maintained in a durable closed loop

What this means:

- when an agent fails, first ask what was missing from the environment or context
- encode the fix into the environment itself: tools, checks, constraints, docs, files, or workflow structure
- do not rely on the operator remembering the lesson in their head

Core beliefs:

- model capability is usually not the bottleneck; context quality is
- broad context stuffing is usually worse than precise context planning
- agent failures are predictable and should map to concrete harness changes
- cross-session memory must be rebuilt from external state, not assumed

Practical implications:

- choose `@llm_function` when one typed transformation gives the cleanest context
- choose `@llm_chat` only when multi-turn state genuinely improves the context for the task
- decide between keeping history, forking, or using a sub-agent based only on which option yields the most accurate and compact context for the current reasoning step
- persist state, progress, and important decisions outside the model so a fresh session can reconstruct context deterministically
- prefer tools and checks that let the system verify itself instead of asking the model to self-certify completion
- treat noisy, stale, or weakly relevant context as a design bug

If you remember only one rule, remember this:

- always construct the cleanest, most task-relevant, shortest complete context the system can maintain in a closed loop

## Best-practice patterns

### Build a general agent (Recommended starting point)

```python
from SimpleLLMFunc import llm_chat, OpenAICompatible, tui
from SimpleLLMFunc.builtin import PyRepl, FileToolset

llm = OpenAICompatible.load_from_json_file("provider.json")["openrouter"]["gpt-5.4"]
repl = PyRepl()
file_tools = FileToolset("./sandbox").toolset

@tui
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset, *file_tools],
    stream=True,
    self_reference_key="agent_main",
)
async def agent(message: str, history=None):
    """You are a practical local coding agent.

    ## Rules
    - Read files before editing. Prefer small, local edits.
    - Use execute_code for Python. Use file tools for read/grep/sed.
    - When a milestone is done, compact your context via:
      runtime.selfref.context.compact(...)
    - For parallel subtasks, spawn forks via:
      runtime.selfref.fork.spawn(...)
      then gather with runtime.selfref.fork.gather_all(...)
    """

if __name__ == "__main__":
    agent()  # launches an interactive TUI
```

### Instant shell-first usage (Recommended for one-offs)

Use the direct constructor path when you want to turn SimpleLLMFunc into a shell ability with almost no setup besides pasting your literal model settings.

```bash
python - <<'PY'
import asyncio

from SimpleLLMFunc import APIKeyPool, OpenAICompatible, llm_function


llm = OpenAICompatible(
    api_key_pool=APIKeyPool(
        api_keys=["sk-your-key"],
        provider_id="openrouter-z-ai-glm-5",
    ),
    model_name="z-ai/glm-5",
    base_url="https://openrouter.ai/api/v1",
)


@llm_function(llm_interface=llm)
async def answer(question: str) -> str:
    """Answer the question in a compact, practical way."""
    pass


print(asyncio.run(answer("Give me three uses of SimpleLLMFunc.")))
PY
```

For a shell agent with file tools and REPL, see `reference/instant-use.md` and `examples/instant_chat_agent.py`.

### Typed `llm_function`

```python
import asyncio

from pydantic import BaseModel, Field

from SimpleLLMFunc import OpenAICompatible, llm_function


class SentimentReport(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0.0 to 1.0 confidence score")
    summary: str = Field(description="one-sentence explanation")


models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openrouter"]["z-ai/glm-5"]


@llm_function(llm_interface=llm)
async def classify_sentiment(text: str) -> SentimentReport:
    """
    Classify the sentiment of the input text.

    Args:
        text: The user text to analyze.

    Returns:
        A structured sentiment report.
    """
    pass


async def main() -> None:
    result = await classify_sentiment("The setup was rough, but the product is excellent.")
    print(result.model_dump())


asyncio.run(main())
```

### Chat agent with a tool

```python
import asyncio

from SimpleLLMFunc import OpenAICompatible, llm_chat, tool
from SimpleLLMFunc.hooks.stream import is_response_yield


@tool
async def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First factor.
        b: Second factor.

    Returns:
        Product of the two inputs.
    """
    return a * b


models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openrouter"]["z-ai/glm-5"]


@llm_chat(llm_interface=llm, toolkit=[multiply], stream=True)
async def tutor(message: str, history: list[dict[str, str]] | None = None):
    """
    You are a concise math tutor.
    Use the multiply tool when arithmetic is requested.
    """
    pass


async def main() -> None:
    history: list[dict[str, str]] = []
    async for output in tutor("What is 12.5 times 8?", history):
        if is_response_yield(output):
            print(output.response, end="")
            history = output.messages


asyncio.run(main())
```

### Persistent runtime with `PyRepl`

```python
import asyncio

from SimpleLLMFunc.builtin import PyRepl, SelfReference


MEMORY_KEY = "agent_main"


async def main() -> None:
    repl = PyRepl()
    selfref = repl.get_runtime_backend("selfref")
    assert isinstance(selfref, SelfReference)

    selfref.bind_history(
        MEMORY_KEY,
        [{"role": "system", "content": "Answer in bullet points."}],
    )

    result = await repl.execute(
        "runtime.selfref.context.remember('remember this')\n"
        "snapshot = runtime.selfref.context.inspect()\n"
        "print(len(snapshot['experiences']))"
    )
    print(result["stdout"])


asyncio.run(main())
```

## Hard rules and gotchas
- Default to `async def` for all decorators. `@tool` enforces async directly.
- The function body does not implement behavior; the docstring does.
- The docstring is usually only the base material for prompt construction, not the entire final system prompt.
- For `llm_chat`, name the history parameter `history` or `chat_history`.
- The current direct-construction path for `OpenAICompatible` requires `APIKeyPool`; do not assume a simplified `api_key=` constructor exists.
- Instant snippets can call the decorated function directly at top level with `asyncio.run(...)`; no `__main__` guard is required.
- `max_tool_calls=None` means no framework-imposed tool-call cap. Set an explicit integer if you need a guardrail.
- Complex structured outputs are parsed from XML-oriented contracts internally. Do not manually force JSON unless you intentionally want plain-text behavior.
- `@llm_function` returns an `LLMFunction` callable instance; use `await fn(...)` for the parsed result and `fn.stream(...)` for `ReactOutput`.
- `@llm_chat` returns an `LLMChat` callable instance; calling it always produces `ReactOutput` events/responses. Consume with `async for output in agent(...)` and use `is_event_yield(output)` / `is_response_yield(output)` to route.
- There is no `enable_event` or `return_mode` decorator option. Do not write old `(chunk, history)` consumers.
- `too_long_to_file=True` keeps roughly the first 20000 tokens in chat and writes the full tool result to a temp file.
- `PyRepl.reset()` clears REPL variables but keeps runtime backends and self-reference memory.
- `execute_code` returns image-producing code as multimodal tool output when code uses `display(Image(...))`, returns an image-rich last expression, or returns `ImgPath` / `ImgUrl`.
- In 0.8.1, PyRepl and SelfRef were internally split into facade/component modules, but application code should continue using the same public surfaces: `PyRepl`, `SelfReference`, `runtime.selfref.context.*`, and `runtime.selfref.fork.*`.
- `runtime.selfref.context.compact(...)` is queued first. When called from a tool run, the compacted context is applied before the next same-turn LLM step when possible, and finalize still commits any leftover queued compaction before the turn ends.
- `OpenAIResponsesCompatible` is a first-class adapter. It maps the selected system prompt to Responses `instructions`, supports `reasoning={...}`, and keeps Responses-specific request/stream behavior out of your decorator code.
- `runtime.selfref.fork.spawn(...)` children inherit the pre-fork context snapshot, not the parent's in-flight fork tool-call scene.
- `runtime.selfref.fork.gather_all(...)` returns `dict[fork_id -> ForkResult]`. Check `status` first, then read `response` or `result`; compact results omit child history unless you request `include_history=True`.
- `FileToolset` is workspace-scoped and read-before-write guarded.
- Runtime side effects must go through the internal patch boundary. Do not try to directly modify the message list of a running agent — use `runtime.selfref.context.*` primitives instead.

## Load more context only when needed
- Philosophy and core concepts: `reference/philosophy-and-concepts.md`
- Harness engineering guidance: `reference/harness-engineering.md`
- System prompt construction and prompt-writing rules: `reference/system-prompt-construction.md`
- Instant shell-first setup and constructor usage: `reference/instant-use.md`
- Provider and environment setup: `reference/configuration.md`
- Decorators, tools, file tools, and event streams: `reference/decorators-and-tools.md`
- PyRepl, runtime primitives, and selfref: `reference/pyrepl-runtime.md`
- Non-obvious behavior: `reference/gotchas.md`
- Mirrored repo docs: `reference/docs-source/quickstart.md`, `reference/docs-source/guide.md`, `reference/docs-source/detailed_guide/`
- Instant heredoc examples: `examples/instant_llm_function.py`, `examples/instant_chat_agent.py`
- Real repo examples: `examples/agent_as_tool_example.py`, `examples/llm_function_pydantic_example.py`, `examples/runtime_primitives_basic_example.py`, `examples/tui_general_agent_example.py`
