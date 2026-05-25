# Decorators And Tools

## `@llm_function`

Use `@llm_function` for one-shot typed tasks. It returns an `LLMFunction` callable instance, not a plain function wrapper; normal `await decorated(...)` usage is unchanged. Use `decorated.stream(...)` when you need the event stream.

Best practices:

- Keep the signature narrow and explicit.
- For image input, declare explicit `ImgUrl` / `ImgPath` parameters or typed lists such as `list[ImgUrl]`. This is the correct multimodal style for `llm_function`.
- Use a typed return value instead of asking the model for hand-written JSON.
- Use `_template_params` only when one prompt pattern truly needs runtime role/style slots.
- Add a toolkit only for real external capability, not for tasks the model can do unaided.

## `@llm_chat`

Use `@llm_chat` for agent-like or conversational behavior. It returns an `LLMChat` callable instance whose `__call__` yields `ReactOutput` items. SelfRef binds to this stable agent instance for fork/rebinding flows.

Best practices:

- Use `stream=True` for chat UIs or incremental feedback.
- Name the history parameter `history` or `chat_history`.
- For multimodal input, use exactly one canonical `message: UserChatMessage` parameter and construct turns with `UserChatMessage.multimodal(...)`.
- Keep `history` outside the function and feed it back in on the next turn.
- Use `strict_signature=True` when you want a stable agent signature for self-reference or fork-heavy flows.
- Set an explicit `max_tool_calls` only if you want a hard loop cap.

## `@tool`

Use `@tool` for capabilities the model may call.

Best practices:

- The function must be `async def`.
- Write a good docstring `Args:` section because parameter descriptions are extracted from it.
- Keep tool outputs concise unless the full payload is truly required.
- Use `best_practices=[...]` when the model needs durable guidance about how to use the tool.
- Use `too_long_to_file=True` for tools that may return massive text, such as code execution or large search results; the framework keeps roughly the first 20000 tokens in-chat and writes the full text to a temp file.

## `FileToolset`

`FileToolset` provides safe file operations inside one workspace.

Rules that matter in practice:

- Read before you write; edits are hash-guarded.
- `grep` requires a `path_pattern` and rejects overly broad `.*` searches.
- Hidden files and out-of-workspace paths are rejected.
- `echo_into` is for full-file replacement, not patch-style edits.

## Event streams and TUI

Event output is now the normal runtime surface; there is no `enable_event` or `return_mode` decorator option.

Key differences:

- `decorated_llm_function.stream(...)` yields `ReactOutput`; the final `ResponseYield.response` is the parsed Python result.
- Calling an `@llm_chat` agent yields `ReactOutput`; response payloads stay closer to raw response or stream chunks.

Use the built-in TUI with the existing pattern:

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.utils.tui import tui


@tui()
@llm_chat(..., stream=True)
async def agent(message: str, history=None):
    """Your agent prompt."""
    pass
```

When handling streams manually, inspect `ResponseYield` and `EventYield` separately instead of assuming `(chunk, history)` tuple outputs.
