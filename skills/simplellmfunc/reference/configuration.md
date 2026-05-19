# Configuration

## Provider loading

The canonical loading path is one of these two adapters, depending on the upstream API surface:

```python
from SimpleLLMFunc import OpenAICompatible, OpenAIResponsesCompatible

chat_models = OpenAICompatible.load_from_json_file("provider.json")
responses_models = OpenAIResponsesCompatible.load_from_json_file("provider.json")

chat_llm = chat_models["openrouter"]["z-ai/glm-5"]
responses_llm = responses_models["openrouter"]["gpt-5.4"]
```

`load_from_json_file(...)` returns a nested mapping:

```text
providers[provider_id][model_name]
```

## Choosing the adapter

- Use `OpenAICompatible` for normal chat/completions-style OpenAI-compatible endpoints.
- Use `OpenAIResponsesCompatible` for OpenAI Responses API endpoints.
- Both adapters use the same `provider.json` structure and the same lookup shape.
- With `OpenAIResponsesCompatible`, you still write normal SimpleLLMFunc docstrings and `history`; the adapter maps the selected system prompt to Responses `instructions`.
- `reasoning={...}` is especially relevant with `OpenAIResponsesCompatible`, because it forwards Responses reasoning configuration without changing decorator usage.

## Direct construction for instant use

When you want a shell-first or one-file workflow, build the interface directly instead of creating `provider.json` first.

```python
from SimpleLLMFunc import APIKeyPool, OpenAICompatible, OpenAIResponsesCompatible


key_pool = APIKeyPool(
    api_keys=["sk-your-key"],
    provider_id="openrouter-z-ai-glm-5",
)

llm = OpenAICompatible(
    api_key_pool=key_pool,
    model_name="z-ai/glm-5",
    base_url="https://openrouter.ai/api/v1",
)

responses_llm = OpenAIResponsesCompatible(
    api_key_pool=key_pool,
    model_name="gpt-5.4",
    base_url="https://openrouter.ai/api/v1",
)
```

Important notes:

- Both constructor shapes are current source-of-truth implementations.
- `APIKeyPool` is singleton-like per `provider_id`, so reuse the same `provider_id` only for the same key set.
- Replace the literal placeholders with your real key/model/base URL and paste the snippet into `python - <<'PY'` if you want truly instant shell usage.
- Choose one adapter per actual upstream API surface. Do not swap adapters casually if the working project already expects one transport path.
- For ready-to-copy minimal patterns, see `reference/instant-use.md` and `examples/instant_llm_function.py`.

## `provider.json` shape

The expected shape is provider name to model config list:

```json
{
  "openrouter": [
    {
      "model_name": "z-ai/glm-5",
      "api_keys": ["key-1", "key-2"],
      "base_url": "https://openrouter.ai/api/v1",
      "api_params": {"reasoning_effort": "high"},
      "max_retries": 5,
      "retry_delay": 1.0,
      "rate_limit_capacity": 20,
      "rate_limit_refill_rate": 3.0
    }
  ]
}
```

Important fields:

- `provider_id` is the top-level object key such as `openrouter`, `openai`, or `zhipu`.
- `model_name`: lookup key under one provider.
- `api_keys`: one or more keys for load balancing.
- `base_url`: OpenAI-compatible endpoint.
- `api_params`: optional per-model default API kwargs (for example `reasoning_effort`); call-level kwargs override these defaults.
- `max_retries` and `retry_delay`: transport retry behavior.
- `rate_limit_capacity` and `rate_limit_refill_rate`: token bucket smoothing.

Example load path:

```python
models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openrouter"]["z-ai/glm-5"]
```

## Best practices

- Start from `examples/provider_template.json` instead of copying a random local file.
- Prefer direct construction when you want a tiny script, heredoc snippet, or shell-first workflow without local config files.
- Keep `provider.json` local to your environment; do not assume the repo-root `provider.json` is the canonical example.
- Within one provider, keep `model_name` values unique.
- Prefer multiple API keys per hot model to reduce single-key throttling.
- Use `api_params` for stable model-specific defaults such as reasoning effort, and call-level kwargs for one-off overrides.
- Tune rate limits per model instead of reusing one configuration everywhere.
- Keep the adapter choice explicit in app code so it is obvious whether one path is using chat/completions transport or Responses transport.

## `.env` usage

The framework mainly reads `.env` for logging and optional Langfuse integration.

Common settings:

```bash
LOG_LEVEL=WARNING
LOG_DIR=logs

LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_EXPORT_ALL_SPANS=true
LANGFUSE_ENABLED=true
```

Common meanings:

- `LOG_LEVEL`: console/file log level.
- `LOG_DIR`: log output directory.
- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`: Langfuse credentials.
- `LANGFUSE_BASE_URL`: Langfuse server URL.
- `LANGFUSE_EXPORT_ALL_SPANS`: keep v3-like export-all behavior.
- `LANGFUSE_ENABLED`: application-level switch convention; current repo docs note it is not a hard automatic framework off-switch.

Priority order:

1. runtime environment variables such as `export LOG_LEVEL=INFO`
2. `.env`
3. framework defaults

Recommended default:

- Use `LOG_LEVEL=WARNING` for everyday agent work to reduce noisy logs.
- Raise it to `INFO` or `DEBUG` only when you are actively diagnosing behavior.

Langfuse is optional. If your task needs observability, configure the corresponding Langfuse environment variables in `.env` or your shell.

## Selection strategy

Choose the model key and adapter that already exist in the working project's setup. Do not hard-code a different provider or transport unless the task explicitly asks for one.
