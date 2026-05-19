"""General TUI agent demo with runtime selfref + file tools.

Run:
    poetry run python examples/tui_general_agent_example.py

What this example demonstrates:
1. ``SelfReference`` mounted as a ``PyRepl`` runtime backend via ``selfref`` pack.
2. One agent can use both ``runtime.selfref.context.*`` and ``runtime.selfref.fork.*``.
3. ``FileToolset`` mounted for workspace-safe file operations.
4. ``llm_chat`` auto-appends runtime primitive guidance into system prompt.
5. Forked context inherits memory snapshot from current selfref key.

Workspace:
- File tools are scoped to ``./sandbox`` under the project root.

Try prompts:
- "Use execute_code to inspect runtime.get_primitive_spec('selfref.fork.gather_all')"
- "Append a durable preference with runtime.selfref.context.remember"
- "Remember a note in memory, then read it back"
- "Split this task into two forks and merge their results"
- "Use grep to search for 'selfref' in README.md, then read the file"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from SimpleLLMFunc import OpenAICompatible, llm_chat
from SimpleLLMFunc.builtin import FileToolset, PyRepl
from SimpleLLMFunc.hooks.events import CustomEvent
from SimpleLLMFunc.type import HistoryList
from SimpleLLMFunc.utils.tui import ToolRenderSnapshot
from SimpleLLMFunc.utils.tui import tui


MEMORY_KEY = "agent_main"
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SANDBOX_DIR = PROJECT_ROOT / "sandbox"
DEBUG_LOG_PATH = PROJECT_ROOT / "logs" / "tui_general_agent_debug.log"


def _build_local_debug_logger() -> logging.Logger:
    logger = logging.getLogger("simplellmfunc.examples.tui_general_agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(DEBUG_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info("=== general TUI agent session started ===")
    return logger


LOCAL_DEBUG_LOGGER = _build_local_debug_logger()


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)


def local_debug_event_hook(
    event: CustomEvent,
    snapshot: ToolRenderSnapshot,
):
    """Write local debug logs for runtime selfref fork events."""
    if snapshot.tool_name != "execute_code":
        return None

    if not (
        event.event_name.startswith("selfref_fork")
        or event.event_name.startswith("kernel_")
    ):
        return None

    LOCAL_DEBUG_LOGGER.info(
        "tool_call_id=%s event=%s snapshot_status=%s data=%s",
        event.tool_call_id or "",
        event.event_name,
        snapshot.status,
        _safe_json(event.data),
    )
    return None


def load_llm():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")
    models = OpenAICompatible.load_from_json_file(provider_json_path)
    return models["openrouter"]["z-ai/glm-5"]


llm = load_llm()
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
repl = PyRepl(working_directory=SANDBOX_DIR)
file_tools = FileToolset(SANDBOX_DIR).toolset


@tui(custom_event_hook=[local_debug_event_hook])  # type: ignore
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset, *file_tools],
    stream=True,
    self_reference_key=MEMORY_KEY,
    temperature=1.0,
)
async def agent(message: str, history: HistoryList):
    """
    <WHO ARE YOU>
    You are a practical coding assistant with one persistent Python REPL.
    </WHO ARE YOU>

    <HOW YOU SHOULD ACT>
    Planning and execution policy:
    - Start with a short plan: objective, assumptions, milestones, deliverables.
    - Separate tasks into dependency levels before coding.
    - Execute dependent steps sequentially; execute independent steps in parallel.
    - Re-plan after each milestone using latest evidence from files and tool output.

    Dependency analysis and parallelism:
    - Task A depends on Task B only if A needs B artifacts (files, symbols, outputs).
    - Tasks that touch disjoint files/modules and have no data dependency can run in parallel.
    - If uncertain about dependency, treat as dependent first, then relax to parallel when proven safe.

    Fork policy:
    - Fork only when a subtask is concrete, isolated, and verifiable.
    - Do not fork tiny trivial work that is faster inline.
    - Use ``fork.spawn`` for independent subtasks, then ``fork.gather_all`` to collect.
    - ``fork.gather_all`` returns ``dict[fork_id -> ForkResult]``; iterate with ``.items()``/``.values()``.

    Sub-agent task contract (must be explicit):
    - Always include: goal, scope, inputs, required outputs, and acceptance checks.
    - Define a strict stop boundary (what NOT to do) to prevent extra work.
    - Ask sub-agents to return concise summaries and file paths, not long transcripts.

    FS-first workflow:
    - Store intermediate findings/results in files when content is long or reusable.
    - Prefer passing file paths between steps/forks instead of copying long text into chat.
    - For large artifacts, write summaries plus pointers to exact files/sections.

    Partial reading policy:
    - Never read entire large content by default.
    - Use grep/rg to locate relevant regions first, then read focused slices.
    - Use small Python snippets for targeted extraction (line ranges, matched blocks, structured fields).

    Output discipline:
    - Use ``print`` for checkpoints, key metrics, and short verification outputs.
    - Do not print full dicts, full histories, or very long file contents.
    - Print only necessary fields and brief summaries (counts, statuses, short excerpts).

    Runtime safety constraints:
    - Your memory key is "agent_main"; do not read or write any other memory key.
    - Never reassign the ``runtime`` variable.
    - REPL state is persistent across calls.
    - Forgetting memory is not ``reset_repl``; ``reset_repl`` only clears REPL variables.

    Response style:
    - Clear, concise, action-oriented.
    </HOW YOU SHOULD ACT>
    """


if __name__ == "__main__":
    print(f"[selfref-debug] local event log: {DEBUG_LOG_PATH}")
    agent()
