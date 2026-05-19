"""SimpleLLMFunc Textual TUI example with persistent PyRepl.

Run:
    poetry run python examples/tui_chat_example.py

Notes:
    - ``toolkit=[*repl.toolset]`` provides a persistent Python environment,
      similar to IPython.
    - State is shared across calls (imports, variables, functions), so
      incremental code execution is recommended.
"""

from __future__ import annotations

import os

from SimpleLLMFunc import OpenAICompatible, llm_chat, tui
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.type import HistoryList


def load_llm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")
    models = OpenAICompatible.load_from_json_file(provider_json_path)
    return models["openrouter"]["minimax/minimax-m2.5"]


llm = load_llm()
repl = PyRepl()


@tui()  # type: ignore
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset],
    stream=True,
)
async def agent(message: str, history: HistoryList):
    """You are a practical coding assistant.

    You have a persistent Python REPL toolset (like IPython):
    - All execute_code calls run in one continuous session.
    - Imports, variables, and functions persist across tool calls.
    - Prefer reusing existing session state instead of re-running everything.

    Important execution rules:
    - Do not rely on ``if __name__ == "__main__":`` blocks; this environment
      is not a script main-entry runtime.
    - ``input()`` is supported. Use it when interactive user values are needed.
    - Keep tool calls focused and print intermediate results when useful.

    Response style:
    - Keep answers concise and practical.
    """


if __name__ == "__main__":
    agent()
