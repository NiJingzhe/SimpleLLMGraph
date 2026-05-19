# Philosophy And Concepts

## Core mental model

SimpleLLMFunc is built around three ideas:

- `LLM as Function`: call the model through normal Python functions.
- `Prompt as Code`: keep the prompt in the function docstring.
- `Code as Doc`: use the signature and return type as the executable contract.

This means the framework handles prompt construction, ReAct loops, parsing, logging, and tool orchestration, while your application code keeps normal Python control flow.

## Choose the right surface

| Need | API | Best fit |
| --- | --- | --- |
| One typed transformation or extraction | `@llm_function` | Stateless tasks with a clear return value |
| Multi-turn conversation or agent loop | `@llm_chat` | Chat-style assistants with external history |
| External capability callable by the model | `@tool` | Search, math, file IO, API access, wrappers |
| Persistent code execution | `PyRepl` | CodeAct-style workflows and runtime primitives |
| Safe workspace file access | `FileToolset` | Controlled read/search/edit inside one workspace |
| Live UI / observability | `ReactOutput` streams and `@tui` | Streaming interfaces and execution tracing |

## Authoring rules

- Use `async def` for decorated functions.
- Put behavior in the docstring and keep the body as `pass`.
- Put variable task data in parameters, not in hard-coded prompt text.
- Prefer typed inputs and typed outputs over manual string parsing.
- Use Pydantic models when the output shape matters.
- Keep branching, retries, loops, and orchestration in Python whenever possible.

## Structured outputs

- Simple return types such as `str`, `int`, `float`, and `bool` are handled as plain text outputs.
- Complex return types such as `list[...]`, `dict[...]`, unions, and Pydantic models use the framework's structured XML-backed contract internally.
- This is why the best default is: define the Python return type and let the framework parse it.

## Multimodal support

SimpleLLMFunc supports multimodal values through:

- `Text`
- `ImgPath`
- `ImgUrl`

You can use them in function parameters, tool parameters, and some tool return shapes. For user-facing application code, prefer explicit multimodal types instead of ad hoc strings that happen to contain paths or URLs.

## History and state

- `@llm_function` returns an `LLMFunction` callable instance; `@llm_chat` returns an `LLMChat` callable instance. Treat them like normal callables in app code unless you are debugging framework internals.
- `@llm_chat` keeps per-call state in invocation/session objects, not on shared mutable globals.
- State normally lives in the `history` you pass in and SelfRef durable memory when mounted.
- If you need durable runtime memory, mount `PyRepl` and use the built-in `selfref` runtime backend.
- If you want the framework to recognize conversation history automatically, use the parameter name `history` or `chat_history`.

## Good defaults

- Start with `@llm_function` unless the task truly needs multi-turn interaction.
- Add tools only when the model needs an external capability.
- Consume `ReactOutput` streams when you need live progress, debugging, or a UI.
- Reach for `PyRepl` only when you want persistent execution, runtime discovery, or forkable memory.
