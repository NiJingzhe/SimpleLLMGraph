"""Agent-as-tool example using stacked decorators.

Run:
    poetry run python examples/agent_as_tool_example.py

What this example demonstrates:
1. Use `@tool` outside `@llm_function` to expose a specialist LLM callable as a tool.
2. Let a supervisor `@llm_chat` agent delegate focused work to that specialist.
3. Keep the child agent non-eventful so it behaves like a normal async tool.

Why this shape:
- `@llm_chat` returns an async generator, so it is not a direct fit for `@tool`.
- `@llm_function` returns a normal async callable, which can be registered as a tool.

Prerequisite:
- Configure `examples/provider.json` with a tool-calling model.
"""

from __future__ import annotations

import asyncio
import os

from SimpleLLMFunc import OpenAICompatible, llm_chat, llm_function, tool
from SimpleLLMFunc.hooks.stream import is_response_yield
from SimpleLLMFunc.type import HistoryList


def load_llm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")
    models = OpenAICompatible.load_from_json_file(provider_json_path)
    return models["openrouter"]["minimax/minimax-m2.5"]


llm = load_llm()


@tool(
    name="implementation_planner",
    description="Delegate focused implementation planning to a specialist agent",
)
@llm_function(llm_interface=llm, temperature=0.3)
async def implementation_planner(task: str, constraints: str = "") -> str:
    """You are a specialist implementation planner.

    Produce a compact plan with:
    1. the likely files or components to touch,
    2. the execution order,
    3. the main risks or open questions.

    Keep the answer under 180 words unless the task explicitly asks for more.

    Args:
        task: The subtask that needs focused planning.
        constraints: Extra constraints from the supervisor.
    """
    pass


@llm_chat(
    llm_interface=llm,
    toolkit=[implementation_planner],
    stream=True,
)
async def supervisor(message: str, history: HistoryList | None = None):
    """You are a supervisor agent.

    For requests about implementation planning, migration strategy, or task
    breakdown, call `implementation_planner` exactly once before writing the
    final answer.

    After the tool returns:
    - synthesize the plan for the user,
    - keep the final answer concise,
    - clearly separate recommended steps from risks.
    """
    pass


async def main() -> None:
    query = (
        "We need to add an agent-as-a-tool example to a Python LLM framework. "
        "Propose a minimal implementation plan, likely files to update, and "
        "the main caveats to document."
    )
    history: HistoryList = []

    print("=" * 70)
    print("Agent as a tool example")
    print("=" * 70)
    print(f"User: {query}")
    print("Assistant: ", end="", flush=True)

    async for output in supervisor(query, history):
        if is_response_yield(output):
            print(output.response, end="", flush=True)
            history = output.messages

    print()


if __name__ == "__main__":
    asyncio.run(main())
