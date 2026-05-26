from __future__ import annotations
from abc import ABC
from typing import (
    Any,
    Awaitable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    Callable,
    get_type_hints,
    Type,
    TypeVar,
    get_origin,
    get_args,
)
import re
import inspect
import tempfile
import os
from pydantic import BaseModel

from SimpleLLMFunc.logger.logger import push_error
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


TOO_LONG_TO_FILE_MAX_TOKENS = 20_000


def _estimate_token_count(text: str) -> int:
    """快速估算文本的 token 数量。

    使用简单的启发式方法估算 token 数量：
    - 中文字符：每个字符约 2 tokens
    - 英文单词：每个单词约 1.3 tokens
    - 其他字符：每个字符约 1 token

    Args:
        text: 要估算的文本

    Returns:
        估算的 token 数量
    """
    if not text:
        return 0

    # 统计中文字符数量
    chinese_char_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")

    # 统计英文单词数量（简单按空格分词）
    english_words = [word for word in text.split() if word.isascii() and word.isalpha()]
    english_word_count = len(english_words)

    # 统计其他字符数量
    other_char_count = (
        len(text) - chinese_char_count - sum(len(word) for word in english_words)
    )

    # 估算 token 数量
    estimated_tokens = int(
        chinese_char_count * 2 + english_word_count * 1.3 + other_char_count
    )

    return max(estimated_tokens, len(text) // 4)  # 确保至少有字符数/4 的 token 数


def _truncate_text_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    """将文本截断到指定的 token 数量。

    Args:
        text: 要截断的文本
        max_tokens: 最大 token 数量

    Returns:
        截断后的文本和实际 token 数量的元组
    """
    if not text:
        return "", 0

    # 如果文本本身就少于 max_tokens，直接返回
    current_tokens = _estimate_token_count(text)
    if current_tokens <= max_tokens:
        return text, current_tokens

    # 二分查找找到合适的截断位置
    left, right = 0, len(text)
    result_text = ""
    result_tokens = 0

    while left <= right:
        mid = (left + right) // 2
        truncated = text[:mid]
        tokens = _estimate_token_count(truncated)

        if tokens <= max_tokens:
            result_text = truncated
            result_tokens = tokens
            left = mid + 1
        else:
            right = mid - 1

    return result_text, result_tokens


def _truncate_tool_result_if_needed(tool_name: str, result: Any) -> Any:
    if not isinstance(result, str):
        return result

    token_count = _estimate_token_count(result)

    if token_count <= TOO_LONG_TO_FILE_MAX_TOKENS:
        return result

    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(
        temp_dir, f"tool_result_{tool_name}_{os.urandom(8).hex()}.txt"
    )

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(result)
    except Exception as e:
        push_error(f"Failed to save tool result to temporary file: {e}")
        return result

    truncated_text, _ = _truncate_text_to_tokens(
        result,
        TOO_LONG_TO_FILE_MAX_TOKENS,
    )

    reminder = (
        "\n\n<system-reminder>\n"
        "Tool output was too long, so this result has been truncated. "
        f"The full result was saved to {temp_file}. "
        "Use file-operation tools or PyRepl to read only the parts you need. "
        "If you do not have a way to read files, warn the user and ask for help.\n"
        "</system-reminder>"
    )

    return truncated_text + reminder


def _truncate_multimodal_text_if_needed(tool_name: str, result: Any) -> Any:
    if not (isinstance(result, tuple) and len(result) == 2):
        return result
    text_part, image_part = result
    if not isinstance(text_part, str):
        return result
    if isinstance(image_part, (ImgPath, ImgUrl)) or (
        isinstance(image_part, list)
        and image_part
        and all(isinstance(item, (ImgPath, ImgUrl)) for item in image_part)
    ):
        return (_truncate_tool_result_if_needed(tool_name, text_part), image_part)
    return result


class Parameter:
    """
    工具参数的简单包装类，仅用于存储信息，不作为主要API
    """

    def __init__(
        self,
        name: str,
        description: str,
        type_annotation: Type,
        required: bool,
        default: Any = None,
        example: Any = None,
    ):
        self.name = name
        self.description = description
        self.type_annotation = type_annotation  # 存储原生Python类型
        self.required = required
        self.default = default
        self.example = example


class Tool(ABC):
    """
    抽象工具基类，可以通过两种方式创建：
    1. 通过子类继承并实现异步 run 方法
    2. 通过@tool装饰器装饰一个 async 函数（推荐方式）
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Optional[Callable[..., Awaitable[Any]]] = None,
        best_practices: Optional[Sequence[str]] = None,
        prompt_injection_builder: Optional[
            Callable[[Dict[str, Any]], Optional[str]]
        ] = None,
        too_long_to_file: bool = False,
    ):
        self.name = name
        self.description = description
        self.best_practices = self._normalize_best_practices(best_practices)
        self._prompt_injection_builder = prompt_injection_builder
        self.too_long_to_file = too_long_to_file
        if func is not None and not inspect.iscoroutinefunction(func):
            func_name = getattr(func, "__name__", repr(func))
            raise TypeError(
                f"Tool '{name}' 的实现必须是 async 函数，检测到同步函数: {func_name}"
            )
        self.func = func
        self.parameters = self._extract_parameters() if func else []

    def build_system_prompt_injection(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return optional tool-owned system-prompt guidance text."""

        builder = self._prompt_injection_builder
        if builder is None:
            return None

        payload = context if isinstance(context, dict) else {}
        rendered = builder(payload)
        if not isinstance(rendered, str):
            return None

        normalized = rendered.strip()
        return normalized or None

    @staticmethod
    def _normalize_best_practices(
        best_practices: Optional[Sequence[str]],
    ) -> List[str]:
        if not best_practices:
            return []

        normalized: List[str] = []
        seen: set[str] = set()
        for item in best_practices:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
        return normalized

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """Structured prompt metadata used by decorator-level injection."""

        return {
            "name": self.name,
            "description": self.description,
            "best_practices": list(self.best_practices),
        }

    def _extract_parameters(self) -> List[Parameter]:
        """
        从函数签名中提取参数信息

        Returns:
            参数列表
        """
        if not self.func:
            return []

        signature = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        docstring = inspect.getdoc(self.func) or ""

        # 尝试解析函数文档字符串中的参数描述
        param_descriptions = self._parse_docstring_params(docstring)

        parameters = []

        for param_name, param in signature.parameters.items():
            # 跳过self参数
            if param_name == "self":
                continue

            # 获取参数类型
            param_type = type_hints.get(param_name, Any)

            # 确定参数是否必需
            required = param.default == inspect.Parameter.empty

            # 获取默认值
            default = None if required else param.default

            # 获取参数描述
            description = param_descriptions.get(param_name, f"Parameter {param_name}")

            # 创建参数对象
            param_obj = Parameter(
                name=param_name,
                description=description,
                type_annotation=param_type,
                required=required,
                default=default,
                example=None,  # 示例值需要单独设置
            )

            parameters.append(param_obj)

        return parameters

    def _parse_docstring_params(self, docstring: str) -> Dict[str, str]:
        """
        解析docstring中的参数描述

        Args:
            docstring: 函数的文档字符串

        Returns:
            参数名到描述的映射
        """
        param_descriptions = {}

        # 查找Args部分
        args_pattern = re.compile(
            r"(?:Args|Parameters):(.*?)(?:\n\n|\n[A-Z]|\Z)", re.DOTALL
        )
        args_match = args_pattern.search(docstring)

        if args_match:
            args_section = args_match.group(1).strip()
            # 匹配参数名和描述
            param_pattern = re.compile(
                r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:(.+?)(?=\n\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:|$)",
                re.MULTILINE | re.DOTALL,
            )
            for match in param_pattern.finditer(args_section):
                param_name, description = match.groups()
                param_descriptions[param_name.strip()] = description.strip()

        return param_descriptions

    async def run(self, *args, **kwargs):
        """
        运行工具。所有工具实现必须为异步函数。
        """
        if self.func is not None:
            result = await self.func(*args, **kwargs)

            # 如果启用了 too_long_to_file 功能，检查结果是否需要截断
            if self.too_long_to_file:
                result = _truncate_tool_result_if_needed(self.name, result)
                result = _truncate_multimodal_text_if_needed(self.name, result)

            return result

        raise NotImplementedError(
            "Subclasses must implement an async run method or provide an async function."
        )

    def _is_optional_type(self, type_annotation: Type) -> bool:
        """
        判断类型是否为Optional[X]或Union[X, None]
        """
        origin = get_origin(type_annotation)
        if origin is Union:
            args = get_args(type_annotation)
            return type(None) in args
        return False

    def _get_inner_type(self, type_annotation: Type) -> Type:
        """
        提取Optional[X]或Union[X, None]中的X
        """
        if self._is_optional_type(type_annotation):
            args = get_args(type_annotation)
            return next(arg for arg in args if arg is not type(None))
        return type_annotation

    def _get_type_schema(self, type_annotation: Type) -> Dict[str, Any]:
        """
        从Python类型生成JSON Schema类型定义

        Args:
            type_annotation: Python类型标注

        Returns:
            对应的JSON Schema类型定义
        """
        #
        # 处理Optional类型
        if self._is_optional_type(type_annotation):
            inner_type = self._get_inner_type(type_annotation)
            schema = self._get_type_schema(inner_type)
            # 允许为null
            if "type" in schema:
                if isinstance(schema["type"], list):
                    if "null" not in schema["type"]:
                        schema["type"].append("null")
                else:
                    schema["type"] = [schema["type"], "null"]
            return schema

        # 基本类型映射
        if type_annotation is str:
            return {"type": "string"}
        elif type_annotation is int:
            return {"type": "integer"}
        elif type_annotation is float:
            return {"type": "number"}
        elif type_annotation is bool:
            return {"type": "boolean"}

        # 处理多模态类型
        if type_annotation is ImgPath:
            return {"type": "string", "description": "本地图片路径"}
        elif type_annotation is ImgUrl:
            return {"type": "string", "description": "网络图片URL"}
        elif type_annotation is Text:
            return {"type": "string", "description": "文本内容"}

        # 处理列表类型
        origin = get_origin(type_annotation)
        args = get_args(type_annotation)

        if origin is list:
            if args:
                return {"type": "array", "items": self._get_type_schema(args[0])}
            return {"type": "array"}

        # 处理字典类型
        if origin is dict:
            if len(args) >= 2:
                return {
                    "type": "object",
                    "additionalProperties": self._get_type_schema(args[1]),
                }
            return {"type": "object"}

        # 处理Pydantic模型
        if isinstance(type_annotation, type) and issubclass(type_annotation, BaseModel):
            # 使用Pydantic的model_json_schema方法获取模型的JSON Schema
            schema = type_annotation.model_json_schema()
            return {"type": "object", "properties": schema.get("properties", {})}

        # 默认为字符串类型
        return {"type": "string"}

    def to_openai_tool(self) -> Dict[str, Any]:
        """
        序列化工具为OpenAI工具格式

        Returns:
            符合OpenAI Function Calling API格式的工具描述字典
        """
        properties = {}
        required_params = []

        for param in self.parameters:
            # 获取类型的JSON Schema表示
            type_schema = self._get_type_schema(param.type_annotation)

            param_schema = {**type_schema, "description": param.description}

            # 添加示例值
            if param.example is not None:
                param_schema["example"] = param.example

            # 处理默认值
            if param.default is not None:
                param_schema["default"] = param.default

            properties[param.name] = param_schema

            # 如果参数是必需的，添加到required列表
            if param.required:
                required_params.append(param.name)

        # 构建符合OpenAI格式的工具描述
        tool_spec = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties},
            },
        }

        # 只有在有必需参数时才添加required字段
        if required_params:
            tool_spec["function"]["parameters"]["required"] = required_params  # type: ignore

        return tool_spec

    @staticmethod
    def serialize_tools(
        tools: List[Tool | Callable[..., Awaitable[Any]]],
    ) -> List[Dict[str, Any]]:
        """
        将多个工具序列化为OpenAI工具列表

        Args:
            tools: 要序列化的工具列表

        Returns:
            符合OpenAI Function Calling API格式的工具描述列表
        """
        try:
            result = [
                (
                    tool.to_openai_tool()
                    if isinstance(tool, Tool)
                    else getattr(tool, "_tool").to_openai_tool()
                )
                for tool in tools
            ]
        except AttributeError as e:
            push_error(
                f"Tool serialization failed: the provided list may contain non-Tool objects or functions not decorated with @tool: {e}"
            )
            raise AttributeError(e)
        except Exception as e:
            push_error(f"Unknown error during tool serialization: {e}")
            raise Exception(e)

        return result


# 工具装饰器函数
T = TypeVar("T")


def tool(
    name: str,
    description: str,
    best_practices: Optional[Sequence[str]] = None,
    prompt_injection_builder: Optional[
        Callable[[Dict[str, Any]], Optional[str]]
    ] = None,
    too_long_to_file: bool = False,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    工具装饰器，用于将函数转换为Tool对象。

    请务必好好写tool函数的DocString，因为这会被作为工具描述信息的一部分。

    ⚠️ 被 @tool 装饰的函数必须定义为 `async def`，同步函数将会被拒绝。

    工具的描述信息是: `description + "\\n" + docstring`

    ## 一个工具函数支持的传入参数类型：
    - **基本类型**: str, int, float, bool
    - **复合类型**: List[T], Dict[K, V]
    - **可选类型**: Optional[T], Union[T, None]
    - **Pydantic模型**: 继承自BaseModel的类，会自动解析为JSON Schema
    - **多模态类型**: ImgPath（本地图片）, ImgUrl（网络图片）, Text（文本）
    - **多模态列表**: List[ImgPath], List[ImgUrl], List[Text]（LLM传递字符串数组，自动转换为多模态对象数组）

    ## 一个工具函数支持的返回值类型：
    - **基本类型**: str, int, float, bool, dict, list等可序列化类型
    - **多模态返回**:
      - ImgPath: 返回本地图片路径，用于生成图表、处理后的图片等
      - ImgUrl: 返回网络图片URL，用于搜索到的图片、在线资源等
      - List[ImgPath | ImgUrl]: 返回多张图片
      - Tuple[str, ImgPath]: 返回说明文本和图片的组合
      - Tuple[str, ImgUrl]: 返回说明文本和网络图片的组合
      - Tuple[str, List[ImgPath | ImgUrl]]: 返回说明文本和多张图片的组合
    - **注意**: 返回值必须是可序列化的，不支持复杂对象

    ## 参数描述解析：
    装饰器会自动解析函数docstring中的Args/Parameters部分，格式为：
    ```
    Args:
        param_name: 参数描述信息
        another_param: 另一个参数的描述
    ```

    Args:
        name: 工具名称，在LLM工具调用中使用
        description: 工具简短描述，更详细的内容可以在被装饰函数的docstring中给出
        best_practices: 可选最佳实践提示，会被 llm_chat / llm_function
            自动注入到 system prompt。
        prompt_injection_builder: 可选注入器，接收上下文字典并返回
            一段要拼接进 system prompt 的工具专属引导文本。
        too_long_to_file: 可选参数，当工具返回的字符串内容超过 20000 tokens 时，
            将完整结果保存到临时文件，并截断返回内容到前 20000 tokens。
            默认为 False（不启用此功能）。
            注意：启用此功能要求 agent 必须配备了文件读写能力。

    Returns:
        装饰器函数，保持原函数功能的同时添加_tool属性

    Example:
        ```python
        from SimpleLLMFunc.tool import tool
        from SimpleLLMFunc.type import ImgPath, ImgUrl
        from pydantic import BaseModel, Field
        from typing import List

        class Location(BaseModel):
            lat: float = Field(..., description="纬度")
            lng: float = Field(..., description="经度")

    @tool(name="generate_map", description="生成位置地图")
    async def generate_map(location: Location, zoom: int = 10) -> ImgPath:
            '''
            根据位置信息生成地图图片

            Args:
                location: 位置坐标信息
                zoom: 地图缩放级别，默认为10

            Returns:
                生成的地图图片路径
            '''
            # 实现地图生成逻辑
            map_path = "./generated_map.png"
            return ImgPath(map_path)

        @tool(name="process_images", description="批量处理图片")
        async def process_images(
            local_images: List[ImgPath],
            remote_images: List[ImgUrl]
        ) -> str:
            '''
            处理多个图片文件

            Args:
                local_images: 本地图片路径列表（LLM传递字符串数组，自动转换为ImgPath对象列表）
                remote_images: 网络图片URL列表（LLM传递字符串数组，自动转换为ImgUrl对象列表）

            Returns:
                处理结果描述
            '''
            # LLM传递的字符串数组会自动转换为多模态对象数组
            for img_path in local_images:
                print(f"处理本地图片: {img_path.path}")
            for img_url in remote_images:
                print(f"处理网络图片: {img_url.url}")
            return "处理完成"
        ```
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"被 @tool 装饰的函数 '{func.__name__}' 必须是 async 函数")
        # 创建工具对象
        tool_obj = Tool(
            name=name,
            description=description,
            func=func,
            best_practices=best_practices,
            prompt_injection_builder=prompt_injection_builder,
            too_long_to_file=too_long_to_file,
        )

        # 保留原始函数的功能，同时附加工具对象
        setattr(func, "_tool", tool_obj)

        return func

    return decorator
