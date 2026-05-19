# System Prompt Construction

This is one of the most important things to understand when using SimpleLLMFunc.

Your docstring matters a lot, but the framework does not always send it to the model unchanged.

## `llm_function`: docstring is wrapped into a prompt template

For `@llm_function`, the framework builds prompts in this order:

1. Read the function docstring.
2. If `_template_params` is provided, apply string formatting to the docstring first.
3. Extract parameter type information from the function signature.
4. Build a return-type description from the annotated return type.
5. Choose the system template:
   - simple return types such as `str`, `int`, `float`, `bool`: plain-text template
   - complex return types such as Pydantic, `list`, `dict`, `Union`: XML-constrained template
6. Render the system prompt with:
   - `{function_description}` = processed docstring
   - `{parameters_description}` = auto-generated parameter type descriptions
   - `{return_type_description}` = auto-generated return-type description
7. Build a separate user prompt from the runtime argument values.
8. If tools exist, prepend one deduplicated `<tool_best_practices>` block to the system prompt.

So the final system prompt is conceptually:

```text
<tool_best_practices>...optional...</tool_best_practices>

[system template]
  function description = your docstring
  parameter descriptions = generated from signature
  return-type instructions = generated from return annotation
```

The current user input values go to the user message, not the system prompt.

Responses adapter note:

- If `llm_interface` is `OpenAIResponsesCompatible`, the framework still builds normal system/user messages first.
- The adapter then extracts the chosen system prompt and sends it as Responses `instructions`.
- Keep authoring docstrings and history the normal SimpleLLMFunc way; do not rewrite prompts around raw Responses request schema.

### What to write in an `llm_function` docstring

Good content:

- the actual task objective
- role or perspective if important
- decision rules
- constraints
- quality criteria
- output style preferences

Usually unnecessary or redundant:

- restating every parameter name and type
- manually describing the output schema in low-level detail when the return type already does that
- asking for XML or JSON just because you think the model needs structure

### Good example

```python
@llm_function(llm_interface=llm)
async def summarize(text: str) -> str:
    """
    Summarize the text for an impatient technical reader.
    Keep only the core points.
    Prefer concrete statements over generic commentary.
    """
    pass
```

Why this is good:

- task is clear
- audience is clear
- compression rule is clear
- style rule is clear
- parameter and output structure are left to the framework

## `llm_chat`: docstring becomes the base prompt, then framework adds blocks around it

For `@llm_chat`, the framework builds messages in this order:

1. Read the incoming `history` or `chat_history` parameter if present and valid.
2. Look for the latest `system` message inside history.
3. Choose the base system prompt:
   - latest history `system` message if present
   - otherwise the function docstring
4. Filter all `system` messages out of the history before appending the rest of the conversation.
5. Build the current user message from non-history arguments.
6. If tools exist, prepend a deduplicated `<tool_best_practices>` block to the system prompt.
7. Append a `<must_principles>` block telling the model to use native structured tool calls.

So the final chat system prompt is conceptually:

```text
<tool_best_practices>...optional...</tool_best_practices>

[base system prompt]
  = latest history system message OR your docstring

<must_principles>
  use native structured tool calls
</must_principles>
```

This has major consequences:

- only the latest history `system` message wins
- older history `system` messages are removed
- the docstring is the default policy, not the only possible system prompt
- current-turn task data belongs in the user message, not in the docstring

### What to write in an `llm_chat` docstring

Good content:

- assistant identity
- stable operating rules
- default tone
- safety boundaries
- persistent priorities

Bad content:

- the current task of one single call
- one-off user data
- tool-call payloads
- repeated reminders of the current message content

### Good example

```python
@llm_chat(llm_interface=llm, toolkit=[*repl.toolset, *file_tools], stream=True)
async def coding_agent(message: str, history=None):
    """
    You are a practical local coding agent.
    Prefer reading before editing.
    Keep changes minimal and explain tradeoffs briefly.
    Use Python execution when inspection or calculation is easier in code.
    """
    pass
```

Why this is good:

- it defines stable behavior
- it leaves the actual task to the runtime user message
- it works well across many turns

## Tools change the system prompt too

If you attach tools, the framework may prepend a `<tool_best_practices>` block.

That block is built from each tool's:

- `description`
- `best_practices`
- optional `prompt_injection_builder(...)` output

This means some prompt guidance belongs on the tool, not in the main docstring.

Use tool-level guidance when the rule is specifically about how to use one tool.

Examples:

- read before write for file editing tools
- use `runtime.get_primitive_spec(...)` before assuming a primitive contract
- avoid dumping huge results directly into chat

## Self-reference memory stores a cleaned version of the prompt

When `llm_chat` works with `SelfReference`, the framework seeds memory with a cleaned version of the system prompt.

Auto-injected blocks such as:

- `<tool_best_practices>`
- runtime primitive prompt blocks
- `<must_principles>`

are stripped before storing durable prompt memory, then re-injected per turn.

This means durable prompt memory is intended to hold the human-authored base policy, not framework-generated scaffolding.

## Practical writing rules

- `llm_function`: write the docstring like a function contract plus reasoning policy.
- `llm_chat`: write the docstring like a long-lived assistant policy.
- Put dynamic user/task data in call arguments.
- Put tool-specific usage rules on the tool when possible.
- If you need a different system prompt for one chat session, pass a latest `system` message in history.
- If you need durable runtime context changes, use self-reference context helpers such as `runtime.selfref.context.remember(...)` or `runtime.selfref.context.compact(...)`; these go through the framework's internal patch boundary.
- If you use `OpenAIResponsesCompatible`, keep the same prompt-writing model and let the adapter translate the selected system prompt to `instructions`.

## Source-of-truth pointers

- `reference/docs-source/detailed_guide/llm_function.md`
- `reference/docs-source/detailed_guide/llm_chat.md`
- `reference/docs-source/detailed_guide/tool.md`
- Source implementation:
  - `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`
  - `SimpleLLMFunc/llm_decorator/steps/chat/message.py`
  - `SimpleLLMFunc/llm_decorator/utils/tools.py`
  - `SimpleLLMFunc/llm_decorator/llm_chat_decorator.py`
