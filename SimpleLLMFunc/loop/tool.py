"""Function-tool declaration and decorator API."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable as AwaitableABC
from collections.abc import Callable
from functools import update_wrapper
from typing import (
    Any,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

from pydantic import BaseModel, ConfigDict, create_model

from SimpleLLMFunc.context.ir import Tool, ToolFunction, ToolResult
from SimpleLLMFunc.event import EventView
from SimpleLLMFunc.loop.revision import function_revision, module_revision
from SimpleLLMFunc.loop.tool_contract import (
    TOOL_CONTRACT_REVISION,
    ToolResultCompiler,
    ToolReturnContract,
    function_type_hints,
    unwrap_return_annotation,
)


P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)
CompilerResultT = TypeVar("CompilerResultT")
TOOL_DECLARATION_REVISION = module_revision(__file__)


class FunctionTool(Generic[P, R_co]):
    """A typed Python function exposed as an LLM tool."""

    __name__: str
    __doc__: str | None
    _function: Callable[P, R_co]
    _schema: Tool
    _arguments_model: type[BaseModel]
    _parameter_names: tuple[str, ...]
    _return_contract: ToolReturnContract[Any]
    revision: str

    def __init__(
        self,
        function: Callable[P, R_co],
        *,
        result_compiler: ToolResultCompiler[R_co] | None = None,
    ) -> None:
        description = inspect.getdoc(function)
        if not description:
            raise ValueError(f"tool {function.__name__} must have a docstring")
        local_name = function.__qualname__.rsplit("<locals>.", 1)[-1]
        if inspect.ismethod(function) or "." in local_name:
            raise TypeError(
                "@tool accepts free functions; close over dependencies instead"
            )

        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        if parameters and parameters[0].name in {"self", "cls"}:
            raise TypeError(
                "@tool accepts free functions; close over dependencies instead"
            )
        hints = function_type_hints(cast(Callable[..., object], function))
        if "return" not in hints:
            raise TypeError(f"tool {function.__name__} must have a return annotation")
        return_annotation = unwrap_return_annotation(hints["return"])
        fields: dict[str, Any] = {}
        for parameter in parameters:
            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                raise TypeError(
                    f"tool parameter {parameter.name} must be expressible as a "
                    "JSON object field"
                )
            annotation = hints.get(parameter.name)
            if annotation is None:
                raise TypeError(
                    f"tool parameter {parameter.name} must have a type annotation"
                )
            default = (
                ...
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            fields[parameter.name] = (annotation, default)

        return_contract = ToolReturnContract[Any](
            return_annotation,
            function.__name__,
            cast(ToolResultCompiler[Any] | None, result_compiler),
        )
        arguments_model = create_model(
            f"{function.__name__}Arguments",
            __config__=ConfigDict(extra="forbid", strict=True),
            **fields,
        )
        parameters_schema = arguments_model.model_json_schema()
        parameters_schema.pop("title", None)
        self._function = function
        self._schema = Tool(
            function=ToolFunction(
                name=function.__name__,
                description=description,
                parameters=parameters_schema,
            )
        )
        self._arguments_model = arguments_model
        self._parameter_names = tuple(fields)
        self._return_contract = return_contract
        declaration = {
            "contract": TOOL_CONTRACT_REVISION,
            "declaration": TOOL_DECLARATION_REVISION,
            "function": function_revision(cast(Callable[..., object], function)),
            "result_compiler": (
                function_revision(cast(Callable[..., object], result_compiler))
                if result_compiler is not None
                else "default"
            ),
            "return_schema": return_contract.schema,
        }
        encoded = json.dumps(
            declaration,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.revision = f"tool@sha256:{hashlib.sha256(encoded).hexdigest()}"
        update_wrapper(self, function)

    @property
    def name(self) -> str:
        return self._schema.function.name

    @property
    def schema(self) -> Tool:
        return self._schema.model_copy(deep=True)

    @property
    def return_schema(self) -> dict[str, Any]:
        return self._return_contract.schema

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self._function)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co:
        return self._function(*args, **kwargs)

    def invoke(self, arguments: dict[str, object]) -> R_co:
        validated = self._arguments_model.model_validate_json(
            json.dumps(arguments, allow_nan=False),
            strict=True,
        )
        kwargs = {
            name: getattr(validated, name)
            for name in self._parameter_names
        }
        function = cast(Callable[..., R_co], self._function)
        return function(**kwargs)

    def serialize_result(self, value: object) -> object:
        """Validate a function result and convert it to durable JSON data."""

        return self._return_contract.serialize(value)

    def restore_result(self, value: object) -> R_co:
        """Restore durable JSON data as the function's declared return type."""

        return cast(R_co, self._return_contract.restore(value))

    def compile_result(self, value: object, events: EventView) -> ToolResult:
        """Compile one typed result with its pre-execution Event snapshot."""

        return self._return_contract.compile(value, events)

    def prepare_result(
        self,
        value: object,
        events: EventView,
    ) -> tuple[object, ToolResult]:
        """Validate, persist, and compile an actual function return value."""

        return self._return_contract.prepare(value, events)


class _ConfiguredToolDecorator(Generic[CompilerResultT]):
    def __init__(
        self,
        result_compiler: ToolResultCompiler[CompilerResultT],
    ) -> None:
        self._result_compiler = result_compiler

    @overload
    def __call__(
        self,
        function: Callable[P, AwaitableABC[CompilerResultT]],
        /,
    ) -> FunctionTool[P, AwaitableABC[CompilerResultT]]: ...

    @overload
    def __call__(
        self,
        function: Callable[P, CompilerResultT],
        /,
    ) -> FunctionTool[P, CompilerResultT]: ...

    def __call__(
        self,
        function: Callable[P, object],
        /,
    ) -> FunctionTool[P, object]:
        return FunctionTool(
            function,
            result_compiler=cast(
                ToolResultCompiler[object],
                self._result_compiler,
            ),
        )


@overload
def tool(
    function: Callable[P, AwaitableABC[R_co]],
    /,
) -> FunctionTool[P, AwaitableABC[R_co]]: ...


@overload
def tool(function: Callable[P, R_co], /) -> FunctionTool[P, R_co]: ...


@overload
def tool(
    *,
    result_compiler: ToolResultCompiler[CompilerResultT],
) -> _ConfiguredToolDecorator[CompilerResultT]: ...


def tool(
    function: Callable[P, R_co] | None = None,
    /,
    *,
    result_compiler: ToolResultCompiler[R_co] | None = None,
) -> FunctionTool[P, R_co] | _ConfiguredToolDecorator[R_co]:
    """Decorate a typed Python function as an LLM tool."""

    if function is not None:
        return FunctionTool(function, result_compiler=result_compiler)
    if result_compiler is None:
        raise TypeError("tool requires a function or result_compiler")
    return _ConfiguredToolDecorator(result_compiler)
