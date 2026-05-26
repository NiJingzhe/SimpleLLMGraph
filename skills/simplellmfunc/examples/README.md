# Examples Guide

This directory mixes two kinds of examples:

- small instant-use examples added for this skill
- mirrored repo examples copied from `examples/`

## Instant Use (No `provider.json` required)

These two examples are intentionally minimal templates for `python - <<'PY'` usage.
Replace the literal API key, model, and base URL in the file, then either run the file or paste it into a heredoc.

- `instant_llm_function.py`
  - Smallest `APIKeyPool` + `OpenAICompatible` + `@llm_function` pattern.
  - Calls the function immediately with `print(asyncio.run(...))`.
  - Also demonstrates the right prompt split: docstring = stable policy, function argument = current task.
- `instant_chat_agent.py`
  - Smallest useful local agent pattern.
  - Mounts both `PyRepl` and `FileToolset`, defines one `@llm_chat`, and runs it immediately without a `__main__` guard.
  - Also demonstrates the right prompt split: docstring = durable agent policy, `message` = current task.

Typical usage shape:

```bash
python - <<'PY'
# paste one of the instant examples here
PY
```

## Quick Play (No API key required)

- `runtime_primitives_basic_example.py`
  - Run: `poetry run python skills/simplellmfunc/examples/runtime_primitives_basic_example.py`
  - Shows `PrimitivePack` authoring, backend-aware runtime primitives, and `runtime.selfref.context.*` usage.

## Mirrored Repo Examples

These are copied from the repo's main `examples/` directory to ground the skill in real project usage.

- `tui_general_agent_example.py`
- `response_api_example.py`
- `tui_chat_example.py`
- `pyrepl_example.py`
- `pyrepl_seaborn_multimodal_images.py`
- `event_stream_chatbot.py`
- `custom_tool_event_example.py`
- `agent_as_tool_example.py`
- `parallel_toolcall_example.py`
- `multi_modality_toolcall.py`
- `dynamic_template_demo.py`
- `llm_function_pydantic_example.py`
- `llm_function_event_pydantic.py`
- `llm_function_token_usage.py`

## Notes

- `provider_template.json` is included as the safe starter config.
- `provider.json` is intentionally not bundled here because checked-in example copies may contain secrets.
- Many mirrored repo examples expect a local `provider.json` next to the script, because they preserve the original repo behavior.
- `response_api_example.py` is the main packaged example for `OpenAIResponsesCompatible`, `reasoning={...}`, and `runtime.selfref.fork.gather_all(...)` result handling.
