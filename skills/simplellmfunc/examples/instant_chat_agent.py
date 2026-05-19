from __future__ import annotations

import asyncio
from pathlib import Path

from SimpleLLMFunc import APIKeyPool, OpenAICompatible, llm_chat
from SimpleLLMFunc.hooks.stream import is_response_yield
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
async def instant_agent(message: str, history: list[dict[str, str]] | None = None):
    """
    You are a local coding agent working in the current workspace.
    Treat the incoming message as the concrete task to complete.
    Read files before editing and keep changes minimal.
    Use Python execution when it helps you inspect or compute.
    Summarize results with concrete file paths when relevant.
    """
    pass


async def main() -> None:
    history: list[dict[str, str]] = []
    async for output in instant_agent(
        "Read README.md and give me a 5-bullet summary.",
        history,
    ):
        if is_response_yield(output):
            print(output.response, end="", flush=True)
            history = output.messages
    print()


asyncio.run(main())
