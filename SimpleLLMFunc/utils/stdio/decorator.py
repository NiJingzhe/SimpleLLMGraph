"""Decorator that streams llm_chat sessions over stdin/stdout."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
import inspect
from functools import wraps
from typing import Any, Optional, Sequence, TextIO

from SimpleLLMFunc.hooks.input_stream import AgentInputRouter, UserInputEvent
from SimpleLLMFunc.hooks.abort import AbortSignal, ABORT_SIGNAL_PARAM
from SimpleLLMFunc.type.message import MessageList
from SimpleLLMFunc.utils.tui.core import consume_react_stream
from SimpleLLMFunc.utils.tui.decorator import (
    _build_runtime_kwargs,
    _resolve_chat_parameters,
)
from SimpleLLMFunc.utils.tui.hooks import ToolCustomEventHook


def _stream_label_for_model(model_call_id: str) -> str:
    if model_call_id.startswith("fork::"):
        return f"Assistant[{model_call_id.split('::', 1)[1]}]"
    return "Assistant"


def _tool_scope_for_model(model_call_id: str) -> str:
    if model_call_id.startswith("fork::"):
        return model_call_id.split("::", 1)[1]
    return "main"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@dataclass
class _StdIOModelState:
    label: str
    model_call_id: str
    content_started: bool = False
    reasoning_started: bool = False


@dataclass
class _StdIOToolState:
    tool_name: str
    model_call_id: str
    output_started: bool = False


class _StdIOStreamAdapter:
    def __init__(
        self,
        *,
        input_router: AgentInputRouter,
        input_stream: TextIO,
        output_stream: TextIO,
        error_stream: TextIO,
    ) -> None:
        self._input_router = input_router
        self._input_stream = input_stream
        self._output_stream = output_stream
        self._error_stream = error_stream
        self._models: dict[str, _StdIOModelState] = {}
        self._tools: dict[str, _StdIOToolState] = {}

    def _write(self, text: str) -> None:
        self._output_stream.write(text)
        self._output_stream.flush()

    def _write_err(self, text: str) -> None:
        self._error_stream.write(text)
        self._error_stream.flush()

    async def start_model_response(self, model_call_id: str) -> None:
        self._models[model_call_id] = _StdIOModelState(
            label=_stream_label_for_model(model_call_id),
            model_call_id=model_call_id,
        )

    async def append_model_content(
        self, model_call_id: str, content_delta: str
    ) -> None:
        if not content_delta:
            return
        model = self._models.setdefault(
            model_call_id,
            _StdIOModelState(
                label=_stream_label_for_model(model_call_id),
                model_call_id=model_call_id,
            ),
        )
        if not model.content_started:
            self._write(
                f'<assistant_message call_id="{model_call_id}" label="{model.label}">\n'
            )
            model.content_started = True
        self._write(content_delta)

    async def append_model_reasoning(
        self,
        model_call_id: str,
        reasoning_delta: str,
    ) -> None:
        if not reasoning_delta:
            return
        model = self._models.setdefault(
            model_call_id,
            _StdIOModelState(
                label=_stream_label_for_model(model_call_id),
                model_call_id=model_call_id,
            ),
        )
        if not model.reasoning_started:
            self._write(
                f'<assistant_reasoning call_id="{model_call_id}" label="{model.label}">\n'
            )
            model.reasoning_started = True
        self._write(reasoning_delta)

    async def finish_model_response(self, model_call_id: str, stats_line: str) -> None:
        model = self._models.setdefault(
            model_call_id,
            _StdIOModelState(
                label=_stream_label_for_model(model_call_id),
                model_call_id=model_call_id,
            ),
        )
        if model.reasoning_started:
            self._write("\n</assistant_reasoning>\n")
            model.reasoning_started = False
        if model.content_started:
            self._write("\n</assistant_message>\n")
            model.content_started = False
        self._write(
            f'<assistant_stats call_id="{model_call_id}" label="{model.label}">{stats_line}</assistant_stats>\n'
        )

    async def start_tool_call(
        self,
        model_call_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        self._tools[tool_call_id] = _StdIOToolState(
            tool_name=tool_name,
            model_call_id=model_call_id,
        )
        scope = _tool_scope_for_model(model_call_id)
        self._write(
            f'<tool_call scope="{scope}" tool="{tool_name}" call_id="{tool_call_id}">\n<tool_arguments>{_json_text(arguments)}</tool_arguments>\n</tool_call>\n'
        )

    async def append_tool_output(self, tool_call_id: str, output_delta: str) -> None:
        if not output_delta:
            return
        tool = self._tools.get(tool_call_id)
        if tool is None:
            return
        if not tool.output_started:
            self._write(f'<tool_output call_id="{tool_call_id}">\n')
            tool.output_started = True
        self._write(output_delta)
        if not output_delta.endswith("\n"):
            self._write("\n")

    async def append_tool_argument(
        self,
        tool_call_id: str,
        argname: str,
        argcontent_delta: str,
    ) -> None:
        if not argcontent_delta:
            return
        self._write(
            f'<tool_argument_delta call_id="{tool_call_id}" name="{argname}">{_json_text(argcontent_delta)}</tool_argument_delta>\n'
        )

    async def set_tool_status(self, tool_call_id: str, status: str) -> None:
        self._write(f'<tool_status call_id="{tool_call_id}">{status}</tool_status>\n')

    async def request_tool_input(
        self,
        tool_call_id: str,
        request_id: str,
        prompt: str,
    ) -> None:
        self._input_router.register_tool_request(
            tool_call_id=tool_call_id,
            request_id=request_id,
            prompt=prompt,
        )
        prompt_text = prompt or f"Input for tool {tool_call_id}: "
        self._write(
            f'<tool_input_request call_id="{tool_call_id}" request_id="{request_id}">{_json_text(prompt_text)}</tool_input_request>\n'
        )
        submitted = await asyncio.to_thread(self._input_stream.readline)
        if submitted == "":
            submitted_text = ""
        else:
            submitted_text = submitted.rstrip("\r\n")
        route = self._input_router.route_input(UserInputEvent(text=submitted_text))
        if route.route == "tool":
            self._write(
                f'<tool_input_response call_id="{tool_call_id}" request_id="{request_id}">{_json_text(submitted_text)}</tool_input_response>\n'
            )
            return
        self._write_err(
            route.reason or "Tool input request expired before it could be submitted.\n"
        )

    async def clear_tool_input(self, tool_call_id: str) -> None:
        self._input_router.clear_tool_requests(tool_call_id)

    async def finish_tool_call(
        self,
        tool_call_id: str,
        result_markdown: str,
        stats_line: str,
        success: bool,
    ) -> None:
        status = "success" if success else "error"
        rendered = result_markdown.strip()
        tool = self._tools.get(tool_call_id)
        if tool is not None and tool.output_started:
            self._write("</tool_output>\n")
            tool.output_started = False
        self._write(
            f'<tool_result call_id="{tool_call_id}" status="{status}" stats={_json_text(stats_line)}>\n'
        )
        if rendered:
            self._write(rendered + ("\n" if not rendered.endswith("\n") else ""))
        self._write("</tool_result>\n")


class _StdIOSession:
    def __init__(
        self,
        *,
        agent_func: Any,
        input_param: str,
        history_param: Optional[str],
        static_kwargs: dict[str, Any],
        custom_hooks: Optional[Sequence[ToolCustomEventHook]],
        input_stream: TextIO,
        output_stream: TextIO,
        error_stream: TextIO,
        initial_history: Optional[MessageList] = None,
    ) -> None:
        self.agent_func = agent_func
        self.input_param = input_param
        self.history_param = history_param
        self.static_kwargs = dict(static_kwargs)
        self.custom_hooks = list(custom_hooks or [])
        self.history: MessageList = list(initial_history or [])
        self._input_router = AgentInputRouter(self._submit_tool_input)
        self._adapter = _StdIOStreamAdapter(
            input_router=self._input_router,
            input_stream=input_stream,
            output_stream=output_stream,
            error_stream=error_stream,
        )
        self._pending_tool_inputs: dict[str, str] = {}
        self._output_stream = output_stream
        self._error_stream = error_stream
        self._supports_abort_signal = self._check_abort_signal_support(agent_func)

    @staticmethod
    def _check_abort_signal_support(agent_func: Any) -> bool:
        try:
            signature = inspect.signature(agent_func)
        except (TypeError, ValueError):
            return False

        if ABORT_SIGNAL_PARAM in signature.parameters:
            return True

        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _submit_tool_input(self, request_id: str, value: str) -> bool:
        self._pending_tool_inputs[request_id] = value
        return True

    async def run_turn(self, user_text: str) -> None:
        call_kwargs = dict(self.static_kwargs)
        call_kwargs[self.input_param] = user_text
        if self.history_param:
            call_kwargs[self.history_param] = self.history
        abort_signal = AbortSignal()
        if self._supports_abort_signal:
            call_kwargs[ABORT_SIGNAL_PARAM] = abort_signal

        stream = self.agent_func(**call_kwargs)
        new_history = await consume_react_stream(
            stream,
            adapter=self._adapter,
            custom_hooks=self.custom_hooks,
            abort_signal=abort_signal,
        )
        if new_history:
            self.history = new_history
        self._input_router.clear_all_requests()
        self._pending_tool_inputs.clear()

    def _write(self, text: str) -> None:
        self._output_stream.write(text)
        self._output_stream.flush()

    def _write_err(self, text: str) -> None:
        self._error_stream.write(text)
        self._error_stream.flush()

    def run(self, input_stream: TextIO) -> None:
        while True:
            raw = input_stream.readline()
            if raw == "":
                break
            user_text = raw.rstrip("\r\n")
            if not user_text.strip():
                continue
            self._write(f"<user_message>{_json_text(user_text)}</user_message>\n")
            try:
                asyncio.run(self.run_turn(user_text))
            except KeyboardInterrupt:
                self._write_err("Interrupted.\n")
                break
            except Exception as exc:
                self._write_err(f"Agent error: {exc}\n")


def stdio(
    custom_event_hook: Optional[Sequence[ToolCustomEventHook]] = None,
    *,
    input_stream: Optional[TextIO] = None,
    output_stream: Optional[TextIO] = None,
    error_stream: Optional[TextIO] = None,
) -> Any:
    """Wrap an event-streaming llm_chat function with stdin/stdout streaming."""

    def decorator(
        agent_func: Any,
    ) -> Any:
        input_param, history_param = _resolve_chat_parameters(agent_func)

        @wraps(agent_func)
        def launcher(*args: Any, **kwargs: Any) -> None:
            static_kwargs, initial_history = _build_runtime_kwargs(
                agent_func=agent_func,
                input_param=input_param,
                history_param=history_param,
                args=args,
                kwargs=kwargs,
            )
            session = _StdIOSession(
                agent_func=agent_func,
                input_param=input_param,
                history_param=history_param,
                static_kwargs=static_kwargs,
                custom_hooks=custom_event_hook,
                input_stream=input_stream or sys.stdin,
                output_stream=output_stream or sys.stdout,
                error_stream=error_stream or sys.stderr,
                initial_history=initial_history,
            )
            session.run(input_stream or sys.stdin)

        return launcher

    return decorator


__all__ = ["stdio"]
