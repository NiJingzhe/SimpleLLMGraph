"""Typed, durable return contracts for function tools."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable as AwaitableABC
from collections.abc import Callable, Coroutine as CoroutineABC
from dataclasses import is_dataclass
from typing import (
    Annotated,
    Any,
    Generic,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from pydantic import BaseModel, TypeAdapter
from pydantic.errors import PydanticInvalidForJsonSchema

from SimpleLLMFunc.context.ir import Image, InputTextPart, ToolResult
from SimpleLLMFunc.event import EventView
from SimpleLLMFunc.loop.json_codec import (
    JSON_CODEC_REVISION,
    canonical_json,
    reject_json_constant,
)
from SimpleLLMFunc.loop.revision import combine_revisions, module_revision


ResultT = TypeVar("ResultT")
ToolResultCompiler = Callable[[ResultT, EventView], ToolResult]
TOOL_CONTRACT_REVISION = combine_revisions(
    module_revision(__file__),
    JSON_CODEC_REVISION,
)


def function_type_hints(
    function: Callable[..., object],
    localns: dict[str, object] | None = None,
) -> dict[str, object]:
    """Resolve annotations with closure locals available to nested tools."""

    closure = inspect.getclosurevars(function)
    namespace = {**closure.globals, **closure.nonlocals, **(localns or {})}
    return get_type_hints(
        function,
        globalns=function.__globals__,
        localns=namespace,
        include_extras=True,
    )


def unwrap_return_annotation(annotation: object) -> object:
    """Resolve metadata and async wrappers to the produced value type."""

    origin = get_origin(annotation)
    if origin is Annotated:
        return unwrap_return_annotation(get_args(annotation)[0])
    if origin is AwaitableABC:
        return unwrap_return_annotation(get_args(annotation)[0])
    if origin is CoroutineABC:
        return unwrap_return_annotation(get_args(annotation)[2])
    return annotation


def _annotation_contains_image(
    annotation: object,
    seen: set[type[BaseModel]] | None = None,
) -> bool:
    annotation = unwrap_return_annotation(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if issubclass(annotation, Image):
            return True
        visited: set[type[BaseModel]] = seen if seen is not None else set()
        if annotation in visited:
            return False
        visited.add(annotation)
        return any(
            _annotation_contains_image(field.annotation, visited)
            for field in annotation.model_fields.values()
        )
    return any(
        argument is not Ellipsis
        and _annotation_contains_image(argument, seen)
        for argument in get_args(annotation)
    )


def _reject_unbounded_return_types(
    annotation: object,
    seen: set[type[BaseModel]] | None = None,
) -> None:
    annotation = unwrap_return_annotation(annotation)
    if annotation is Any or annotation is object or annotation is BaseModel:
        raise TypeError("tool return annotations cannot contain Any or object")
    if isinstance(annotation, type):
        model_type = cast(type[object], annotation)
        if model_type in (dict, list, set, tuple):
            raise TypeError("tool return annotations must parameterize containers")
        if (
            is_dataclass(model_type)
            or is_typeddict(model_type)
            or (
                issubclass(model_type, tuple)
                and "_fields" in cast(dict[str, object], model_type.__dict__)
            )
        ):
            raise TypeError("tool UDT return annotations must use pydantic BaseModel")
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        visited: set[type[BaseModel]] = seen if seen is not None else set()
        if annotation in visited:
            return
        visited.add(annotation)
        for field in annotation.model_fields.values():
            _reject_unbounded_return_types(field.annotation, visited)
        return
    for argument in get_args(annotation):
        if argument is not Ellipsis:
            _reject_unbounded_return_types(argument, seen)


def _return_adapter(annotation: object, tool_name: str) -> TypeAdapter[Any]:
    resolved = unwrap_return_annotation(annotation)
    _reject_unbounded_return_types(resolved)
    adapter = TypeAdapter[Any](resolved)
    try:
        adapter.json_schema(mode="serialization")
    except PydanticInvalidForJsonSchema as exc:
        raise TypeError(
            f"tool {tool_name} return annotation is not JSON serializable"
        ) from exc
    return adapter


def _validate_result_compiler(
    compiler: Callable[..., object],
    return_annotation: object,
) -> None:
    if not inspect.isfunction(compiler):
        raise TypeError("tool result compiler must be a function")
    if inspect.iscoroutinefunction(compiler):
        raise TypeError("tool result compiler must be synchronous")
    signature = inspect.signature(compiler)
    parameters = list(signature.parameters.values())
    if len(parameters) != 2 or any(
        parameter.kind
        not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        for parameter in parameters
    ):
        raise TypeError(
            "tool result compiler must accept (result, events) positional arguments"
        )
    localns: dict[str, object] | None = (
        {return_annotation.__name__: return_annotation}
        if isinstance(return_annotation, type)
        else None
    )
    hints = function_type_hints(compiler, localns)
    if unwrap_return_annotation(hints.get(parameters[0].name)) != return_annotation:
        raise TypeError(
            "tool result compiler result parameter must match the tool return type"
        )
    if hints.get(parameters[1].name) is not EventView:
        raise TypeError("tool result compiler events parameter must be EventView")
    if hints.get("return") is not ToolResult:
        raise TypeError("tool result compiler must declare a ToolResult return type")


def _extract_images(value: object) -> tuple[object, tuple[Image, ...]]:
    images: list[Image] = []

    def visit(item: object) -> object:
        if isinstance(item, dict):
            payload = cast(dict[object, object], item)
            if payload.get("type") == "input_image" and "image_url" in payload:
                image = Image.model_validate(payload)
                images.append(image)
                return {"image": f"<image:{len(images)}>"}
            return {str(key): visit(child) for key, child in payload.items()}
        if isinstance(item, list):
            return [visit(child) for child in cast(list[object], item)]
        return item

    return visit(value), tuple(images)


class ToolReturnContract(Generic[ResultT]):
    """Validate, persist, restore, and compile one declared return type."""

    def __init__(
        self,
        annotation: object,
        tool_name: str,
        result_compiler: ToolResultCompiler[ResultT] | None,
    ) -> None:
        self._adapter = _return_adapter(annotation, tool_name)
        if result_compiler is not None:
            _validate_result_compiler(
                cast(Callable[..., object], result_compiler),
                annotation,
            )
        self._result_compiler = result_compiler
        self._contains_image = _annotation_contains_image(annotation)

    @property
    def schema(self) -> dict[str, Any]:
        return self._adapter.json_schema(mode="serialization")

    def serialize(self, value: object) -> object:
        """Validate a result and convert it to durable JSON data."""

        validated = self._adapter.validate_python(value, strict=True)
        encoded = self._adapter.dump_json(validated, warnings="error")
        normalized = json.loads(encoded, parse_constant=reject_json_constant)
        restored = self.restore(normalized)
        if restored != validated:
            raise ValueError("tool result is not losslessly JSON serializable")
        return normalized

    def restore(self, value: object) -> ResultT:
        """Restore durable JSON data as the declared return type."""

        restored = self._adapter.validate_json(
            canonical_json(value),
            strict=True,
        )
        return cast(ResultT, restored)

    def compile(self, value: object, events: EventView) -> ToolResult:
        """Compile durable JSON with its pre-execution Event snapshot."""

        return self._compile_validated(self.restore(value), value, events)

    def prepare(
        self,
        value: object,
        events: EventView,
    ) -> tuple[object, ToolResult]:
        """Validate, persist, and compile an actual function return value."""

        validated = self._adapter.validate_python(value, strict=True)
        serialized = self.serialize(validated)
        restored = self.restore(serialized)
        return serialized, self._compile_validated(restored, serialized, events)

    def _compile_validated(
        self,
        restored: ResultT,
        serialized: object,
        events: EventView,
    ) -> ToolResult:
        if self._result_compiler is not None:
            compiler = cast(
                Callable[[ResultT, EventView], object],
                self._result_compiler,
            )
            compiled_value = compiler(restored, events)
            if not isinstance(compiled_value, ToolResult):
                raise TypeError("tool result compiler must return ToolResult")
            compiled = compiled_value
        elif self._contains_image:
            text_value, images = _extract_images(serialized)
            compiled = ToolResult(
                content=[
                    InputTextPart(text=canonical_json(text_value)),
                    *images,
                ],
            )
        else:
            compiled = ToolResult(content=canonical_json(serialized))
        if self._contains_image and not compiled.requires_user_message:
            compiled = compiled.model_copy(update={"as_user_message": True})
        return compiled.model_copy(deep=True)
