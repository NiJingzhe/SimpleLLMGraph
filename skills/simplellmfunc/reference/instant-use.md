# Instant Use

## Why this path matters

SimpleLLMFunc is not only for large agent projects. It is also good at tiny shell snippets where you:

- write a few lines of Python
- skip `provider.json`
- construct the interface directly
- decorate one function or one agent
- call it immediately inside `python - <<'PY'`

The important implementation detail is: the current source-of-truth constructor needs `APIKeyPool` plus the transport adapter you actually want, typically `OpenAICompatible` or `OpenAIResponsesCompatible`.

## Minimal direct-construction pattern

### Chat/completions-compatible adapter

```python
from SimpleLLMFunc import APIKeyPool, OpenAICompatible


key_pool = APIKeyPool(
    api_keys=["sk-your-key"],
    provider_id="openrouter-z-ai-glm-5",
)

llm = OpenAICompatible(
    api_key_pool=key_pool,
    model_name="z-ai/glm-5",
    base_url="https://openrouter.ai/api/v1",
)
```

After that, you can use `llm` with either `@llm_function` or `@llm_chat`.

### Responses API adapter

```python
from SimpleLLMFunc import APIKeyPool, OpenAIResponsesCompatible


key_pool = APIKeyPool(
    api_keys=["sk-your-key"],
    provider_id="openrouter-gpt-5.4-responses",
)

llm = OpenAIResponsesCompatible(
    api_key_pool=key_pool,
    model_name="gpt-5.4",
    base_url="https://openrouter.ai/api/v1",
)
```

Important notes for the Responses adapter:

- The constructor shape is still `APIKeyPool` + `model_name` + `base_url`.
- The same `provider.json` structure works with `OpenAIResponsesCompatible.load_from_json_file(...)`.
- Keep writing normal SimpleLLMFunc docstrings and `history`; the adapter maps the chosen system prompt to Responses `instructions` for you.
- Pass `reasoning={...}` on `@llm_function` or `@llm_chat` when the provider supports Responses reasoning controls.

## Instant `llm_function` from the shell

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
async def rewrite(text: str) -> str:
    """
    Complete the user's request in a concise, practical way.
    Keep the answer technically accurate and directly usable.
    If the request asks for multiple points, use a short bullet list.
    """
    pass


print(asyncio.run(rewrite("make this sentence better and shorter")))
PY
```

Why this prompt shape is good:

- the docstring defines stable execution policy
- the concrete one-shot task lives in the function argument
- the framework will separately inject parameter and return-type guidance

## Instant streaming agent from the shell

```bash
python - <<'PY'
import asyncio
from pathlib import Path

from SimpleLLMFunc import APIKeyPool, OpenAICompatible, llm_chat
from SimpleLLMFunc.builtin import FileToolset, PyRepl


workspace = Path(".").resolve()
llm = OpenAICompatible(
    api_key_pool=APIKeyPool(
        api_keys=["sk-your-key"],
        provider_id="openrouter-z-ai-glm-5-agent",
    ),
    model_name="z-ai/glm-5",
    base_url="https://openrouter.ai/api/v1",
)
repl = PyRepl(working_directory=workspace)
file_tools = FileToolset(workspace).toolset


@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset, *file_tools],
    stream=True,
)
async def shell_agent(message: str, history=None):
    """
    You are a local coding agent working in the current workspace.
    Treat the incoming message as the concrete task to complete.
    Read files before editing and keep changes minimal.
    Use Python execution when it helps you inspect or compute.
    Summarize results with concrete file paths when relevant.
    """
    pass


async def run() -> None:
    async for output in shell_agent(
        "Read README.md and give me a 5-bullet summary.",
        [],
    ):
        if is_response_yield(output):
            print(output.response, end="", flush=True)
    print()


asyncio.run(run())
PY
```

Why this prompt shape is good:

- the docstring holds durable agent policy
- the current task is passed as `message`
- tool-specific usage guidance can be injected by the framework from the mounted tools

## Same pattern as reusable files

If you want to keep these snippets around as copy-pasteable templates, use:

- `examples/instant_llm_function.py`
- `examples/instant_chat_agent.py`

Both are intentionally written as minimal top-level snippets without a `__main__` guard, so they translate directly into heredoc usage.

## Best practices

- Use direct construction for one-offs, demos, and scripts generated on the fly.
- Use `provider.json` when you need shared project config, multiple models, or cleaner team setup.
- Give `APIKeyPool` a stable `provider_id` per key set when the process is long-lived.
- For one-shot heredocs, top-level `print(asyncio.run(...))` or `asyncio.run(run())` is the simplest shape.
- For chat agents, `FileToolset` plus `PyRepl` is the smallest useful local-agent toolkit.
- Keep stable policy in the docstring and put the current task in function arguments.
- Choose `OpenAIResponsesCompatible` when you specifically want OpenAI Responses API behavior such as `reasoning={...}` support while keeping the same decorator-level authoring model.
- If docs conflict on constructor shape, trust source code and `reference/docs-source/detailed_guide/llm_interface.md`.
