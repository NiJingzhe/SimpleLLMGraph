from __future__ import annotations
import json
import os
import asyncio
from typing import Optional, Dict, Literal, Iterable, Any, AsyncGenerator
from typing_extensions import override
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from SimpleLLMFunc.interface.llm_interface import DEFAULT_CONTEXT_WINDOW, LLM_Interface
from SimpleLLMFunc.interface.key_pool import APIKeyPool
from SimpleLLMFunc.interface.token_bucket import rate_limit_manager
from SimpleLLMFunc.logger import (
    app_log,
    push_warning,
    push_error,
    get_location,
    get_current_trace_id,
    push_debug,
)
from SimpleLLMFunc.logger.logger import (
    push_critical,
    get_current_context_attribute,
    set_current_context_attribute,
)


class OpenAICompatible(LLM_Interface):
    """与OpenAI API兼容的LLM接口实现，支持任何符合OpenAI格式的API接口。

    这个类提供了一个通用的接口，可以连接任何兼容OpenAI API格式的大语言模型服务，
    而不需要为每个供应商创建特定的实现。只需要提供正确的base_url和模型名称即可。
    """

    def _count_tokens(self, response: Any) -> tuple[int, int]:
        """计算响应中的token数量

        Args:
            response: OpenAI API的响应对象

        Returns:
            (输入token数, 输出token数)的元组
        """
        try:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            return prompt_tokens, completion_tokens
        except (AttributeError, TypeError):
            # 如果无法获取token计数,返回0
            return 0, 0

    async def _close_stream_response(self, response: Any) -> None:
        """Best-effort close for streaming responses.

        Some providers keep stream connections open after a terminal chunk.
        If we stop consuming early (e.g. finish grace timeout), explicitly
        closing the stream prevents leaked HTTP connections/background tasks.
        """

        close_method = getattr(response, "close", None)
        if callable(close_method):
            try:
                close_result = close_method()
                if hasattr(close_result, "__await__"):
                    await close_result
            except Exception as close_exc:
                push_warning(
                    f"{self.model_name} failed to close stream response: {close_exc}",
                    location=get_location(),
                )
            return

        aclose_method = getattr(response, "aclose", None)
        if callable(aclose_method):
            try:
                aclose_result = aclose_method()
                if hasattr(aclose_result, "__await__"):
                    await aclose_result
            except Exception as close_exc:
                push_warning(
                    f"{self.model_name} failed to close stream response: {close_exc}",
                    location=get_location(),
                )

    @classmethod
    def load_from_json_file(
        cls, json_path: str
    ) -> Dict[str, Dict[str, OpenAICompatible]]:
        """从JSON字符串加载OpenAICompatible实例

        Args:
            json_str: JSON字符串，包含API密钥和模型名称

            例如:
            ```
            {
                "openai": [
                    {
                        "model_name": "gpt-3.5-turbo",
                        "api_keys": [key1, key2, key3],
                        "base_url": "https://api.openai.com/v1",
                        "max_retries": 5,
                        "retry_delay": 1.0,
                        "rate_limit_capacity": 10,
                        "rate_limit_refill_rate": 1.0
                    },
                    {
                        "model_name": "gpt-4",
                        "api_keys": [key1, key2, key3],
                        "base_url": "https://api.openai.com/v1",
                        "max_retries": 5,
                        "retry_delay": 1.0,
                        "rate_limit_capacity": 5,
                        "rate_limit_refill_rate": 0.5
                    }
                ],
                "zhipu": [
                    {
                        "model_name": "gpt-3.5-turbo",
                        "api_keys": [key1, key2, key3],
                        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
                        "max_retries": 5,
                        "retry_delay": 1.0,
                        "rate_limit_capacity": 15,
                        "rate_limit_refill_rate": 2.0
                    },
                    {
                        "model_name": "gpt-4",
                        "api_keys": [key1, key2, key3],
                        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
                        "max_retries": 5,
                        "retry_delay": 1.0,
                        "rate_limit_capacity": 8,
                        "rate_limit_refill_rate": 1.5
                    }
                ]
            }
            ```

        Returns:
            OpenAICompatible实例的字典, 可以这样访问：
            ```python
            from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible

            all_models = OpenAICompatible.load_from_json(json_str)
            gpt_3_5 = all_models["openai"]["gpt-3.5-turbo"]
            gpt_4 = all_models["openai"]["gpt-4"]
            ```
        """

        if not os.path.exists(json_path):
            push_critical(
                f"JSON 文件 {json_path} 不存在。请检查您的配置。",
                location=get_location(),
            )
            raise FileNotFoundError(f"JSON 文件 {json_path} 不存在。")

        with open(json_path, "r", encoding="utf-8") as f:
            json_str = f.read()
        # 解析JSON字符串
        try:
            all_providers_info = json.loads(json_str)
        except json.JSONDecodeError as e:
            push_critical(
                f"Failed to parse JSON string: {e}", location=get_location()
            )
            raise ValueError(f"Failed to parse JSON string: {e}")
        # 检查JSON格式

        try:
            all_providers_dict: Dict[str, Dict[str, OpenAICompatible]] = {}
            for provider_id, models in all_providers_info.items():
                all_providers_dict[provider_id] = {}
                app_log(
                    f"Loading OpenAICompatible instances for provider: {provider_id}",
                    location=get_location(),
                )

                if not isinstance(models, list):
                    push_critical(
                        f"Invalid model format under provider {provider_id}. Expected a list.",
                        location=get_location(),
                    )
                    raise TypeError(
                        f"Invalid model format under provider {provider_id}. Expected a list."
                    )

                for model_info in models:
                    model_name = model_info["model_name"]
                    api_keys = model_info["api_keys"]
                    base_url = model_info["base_url"]
                    context_window = model_info.get(
                        "context_window", DEFAULT_CONTEXT_WINDOW
                    )
                    max_retries = model_info.get("max_retries", 5)
                    retry_delay = model_info.get("retry_delay", 1.0)
                    rate_limit_capacity = model_info.get("rate_limit_capacity", 10)
                    rate_limit_refill_rate = model_info.get(
                        "rate_limit_refill_rate", 1.0
                    )
                    api_params = model_info.get("api_params", None)

                    # 创建APIKeyPool实例
                    key_pool = APIKeyPool(api_keys, f"{provider_id}-{model_name}")

                    # 创建OpenAICompatible实例
                    instance = OpenAICompatible(
                        api_key_pool=key_pool,
                        model_name=model_name,
                        base_url=base_url,
                        context_window=context_window,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        rate_limit_capacity=rate_limit_capacity,
                        rate_limit_refill_rate=rate_limit_refill_rate,
                        api_params=api_params,
                    )

                    all_providers_dict[provider_id][model_name] = instance

                    app_log(
                        f"Loaded OpenAICompatible instance for provider {provider_id} model {model_name}",
                        location=get_location(),
                    )
        except ValueError as e:
            push_critical(
                f"Error while loading OpenAICompatible instances: {e}",
                location=get_location(),
            )
            raise ValueError(f"Error while loading OpenAICompatible instances: {e}")
        except TypeError as e:
            push_critical(f"Invalid type in JSON: {e}", location=get_location())
            raise ValueError(f"Invalid type in JSON: {e}")
        except KeyError as e:
            push_critical(f"Missing required key in JSON: {e}", location=get_location())
            raise ValueError(f"Missing required key in JSON: {e}")
        except Exception as e:
            push_critical(
                f"Unknown error while loading OpenAICompatible instances: {e}",
                location=get_location(),
            )
            raise ValueError(
                f"Unknown error while loading OpenAICompatible instances: {e}"
            )

        return all_providers_dict

    def __repr__(self) -> str:
        """返回OpenAICompatible实例的字符串表示"""
        return (
            f"OpenAICompatible(model_name={self.model_name}, base_url={self.base_url})"
        )

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """获取当前实例的令牌桶状态

        Returns:
            包含令牌桶状态信息的字典
        """
        return self.token_bucket.get_info()

    def reset_rate_limit(self) -> None:
        """重置令牌桶（填满令牌）"""
        self.token_bucket.reset()

    def __init__(
        self,
        api_key_pool: APIKeyPool,
        model_name: str,
        base_url: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        rate_limit_capacity: int = 10,
        rate_limit_refill_rate: float = 1.0,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        api_params: Optional[Dict[str, Any]] = None,
    ):
        """初始化OpenAI兼容的LLM接口

        Args:
            api_key_pool: API密钥池，用于管理和分配API密钥
            model_name: 要使用的模型名称
            base_url: API基础URL，例如"https://api.openai.com/v1"或"https://open.bigmodel.cn/api/paas/v4/"
            max_retries: 最大重试次数
            retry_delay: 重试间隔时间（秒）
            rate_limit_capacity: 令牌桶容量（最大令牌数）
            rate_limit_refill_rate: 令牌补充速率（令牌数/秒）
            context_window: 模型上下文窗口大小；未指定时默认使用 200000 占位
            api_params: 额外透传到 API 调用的参数，如 {"reasoning_effort": "high"}
        """
        super().__init__(
            api_key_pool,
            model_name,
            base_url=base_url,
            context_window=context_window,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self.model_name = model_name
        self.key_pool = api_key_pool

        # 创建令牌桶，使用provider和model作为唯一标识
        bucket_id = f"{base_url}_{model_name}"
        self.token_bucket = rate_limit_manager.get_or_create_bucket(
            bucket_id=bucket_id,
            capacity=rate_limit_capacity,
            refill_rate=rate_limit_refill_rate,
        )

        self._api_params: Dict[str, Any] = dict(api_params) if api_params else {}

        self.client = AsyncOpenAI(
            api_key=api_key_pool.get_least_loaded_key(), base_url=self.base_url
        )
        # Grace window to receive post-finish usage chunk from providers.
        self._stream_finish_grace_timeout = 1.0

    async def _get_or_create_client(self, key: str) -> AsyncOpenAI:
        """获取或创建客户端，确保使用正确的API密钥"""
        # 如果当前客户端的API密钥不匹配，或者客户端为None，创建新的客户端
        if (
            not hasattr(self, "_current_key")
            or self._current_key != key  # type: ignore
            or not hasattr(self, "client")
            or self.client is None
        ):
            # 关闭旧客户端
            if hasattr(self, "client") and self.client is not None:
                try:
                    await self.client.close()  # type: ignore
                except Exception:
                    # 忽略关闭异常
                    pass

            # 创建新客户端
            self.client = AsyncOpenAI(api_key=key, base_url=self.base_url)
            self._current_key = key

        return self.client

    async def aclose(self):
        """关闭客户端连接"""
        if hasattr(self, "client") and self.client is not None:
            try:
                await self.client.close()  # type: ignore
            except Exception:
                pass
            finally:
                self.client = None

    @override
    async def chat(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[False] = False,
        messages: Iterable[Dict[str, str]] = [
            {
                "role": "system",
                "content": "你是一位乐于助人的助手，可以帮助用户解决各种问题。",
            }
        ],
        timeout: Optional[int] = 30,
        *args,
        **kwargs,
    ) -> ChatCompletion:
        """执行非流式LLM对话请求

        Args:
            trace_id: 跟踪ID，用于日志记录
            stream: 是否使用流式响应，这里必须为False
            messages: 消息历史，包含角色和内容的字典列表
            timeout: 请求超时时间（秒）
            *args, **kwargs: 传递给OpenAI API的其他参数

        Returns:
            LLM的响应内容
        """
        key = self.key_pool.get_least_loaded_key()
        client = await self._get_or_create_client(key)

        attempt = 0
        while attempt < self.max_retries:
            try:
                # 获取令牌桶令牌，设置30秒超时
                token_acquired = await self.token_bucket.acquire(
                    tokens_needed=1, timeout=30.0
                )
                if not token_acquired:
                    push_warning(
                        f"{self.model_name} token bucket acquire timed out; skipping request",
                        location=get_location(),
                    )
                    raise Exception("Rate limit: token bucket acquire timed out")

                self.key_pool.increment_task_count(key)
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                push_debug(
                    f"OpenAICompatible::chat: {self.model_name} request with API key: {key}, and message: {data}",
                    location=get_location(),
                )

                # Merge instance-level api_params with call-level kwargs (call-level wins)
                call_params = {**self._api_params, **kwargs}
                response: ChatCompletion = await client.chat.completions.create(  # type: ignore
                    messages=messages,  # type: ignore
                    model=self.model_name,
                    stream=stream,
                    timeout=timeout,
                    *args,
                    **call_params,
                )

                # 统计token
                if not (
                    response.choices
                    and response.choices[0].message
                    and response.choices[0].message.tool_calls
                ):  # type: ignore
                    prompt_tokens, completion_tokens = self._count_tokens(response)

                    # 更新上下文中的token计数
                    input_tokens = get_current_context_attribute("input_tokens") or 0
                    output_tokens = get_current_context_attribute("output_tokens") or 0

                    set_current_context_attribute(
                        "input_tokens", input_tokens + prompt_tokens
                    )
                    set_current_context_attribute(
                        "output_tokens", output_tokens + completion_tokens
                    )

                self.key_pool.decrement_task_count(key)
                return response  # 请求成功，返回结果

            except Exception as e:
                self.key_pool.decrement_task_count(key)
                attempt += 1
                location = get_location()
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                push_warning(
                    f"{self.model_name} Interface attempt {attempt} failed: With message : {data} send, \n but exception : {str(e)} was caught",
                    location=get_location(),
                )

                key = self.key_pool.get_least_loaded_key()
                client = await self._get_or_create_client(key)

                if attempt >= self.max_retries:
                    push_error(
                        f"Max retries reached. {self.model_name} Failed to get a response for {data}",
                        location=location,
                    )
                    raise e  # 达到最大重试次数后抛出异常
                await asyncio.sleep(self.retry_delay)  # 重试前等待一段时间
        return ChatCompletion(
            id="", choices=[], created=0, model="", object="chat.completion", usage=None
        )  # 添加默认返回以满足类型检查，实际上这行代码永远不会执行

    @override
    async def chat_stream(
        self,
        trace_id: str = get_current_trace_id(),
        stream: Literal[True] = True,
        messages: Iterable[Dict[str, str]] = [
            {
                "role": "system",
                "content": "你是一位乐于助人的助手，可以帮助用户解决各种问题。",
            }
        ],
        timeout: Optional[int] = 30,
        *args,
        **kwargs,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """执行流式LLM对话请求

        Args:
            trace_id: 跟踪ID，用于日志记录
            stream: 是否使用流式响应，这里必须为True
            messages: 消息历史，包含角色和内容的字典列表
            timeout: 请求超时时间（秒）
            *args, **kwargs: 传递给OpenAI API的其他参数

        Yields:
            LLM的响应块
        """
        key = self.key_pool.get_least_loaded_key()
        client = await self._get_or_create_client(key)

        attempt = 0
        while attempt < self.max_retries:
            try:
                # 获取令牌桶令牌，设置30秒超时
                token_acquired = await self.token_bucket.acquire(
                    tokens_needed=1, timeout=30.0
                )
                if not token_acquired:
                    push_warning(
                        f"{self.model_name} stream token bucket acquire timed out; skipping request",
                        location=get_location(),
                    )
                    raise Exception("Rate limit: token bucket acquire timed out")

                self.key_pool.increment_task_count(key)
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                push_debug(
                    f"OpenAICompatible::chat_stream: {self.model_name} request with API key: {key}, and message: {data}",
                    location=get_location(),
                )

                # Merge instance-level api_params with call-level kwargs (call-level wins)
                request_kwargs = {**self._api_params, **kwargs}
                auto_stream_options_added = False
                if "stream_options" not in request_kwargs:
                    request_kwargs["stream_options"] = {"include_usage": True}
                    auto_stream_options_added = True
                else:
                    stream_options = request_kwargs.get("stream_options")
                    if isinstance(stream_options, dict):
                        merged_stream_options = dict(stream_options)
                        merged_stream_options.setdefault("include_usage", True)
                        request_kwargs["stream_options"] = merged_stream_options

                response = None

                try:
                    response = await client.chat.completions.create(  # type: ignore
                        messages=messages,  # type: ignore
                        model=self.model_name,
                        stream=stream,
                        timeout=timeout,
                        *args,
                        **request_kwargs,
                    )
                except Exception as create_exc:
                    # 部分兼容提供方不支持 stream_options，回退到无该参数。
                    if (
                        auto_stream_options_added
                        and "stream_options" in str(create_exc).lower()
                    ):
                        request_kwargs.pop("stream_options", None)
                        response = await client.chat.completions.create(  # type: ignore
                            messages=messages,  # type: ignore
                            model=self.model_name,
                            stream=stream,
                            timeout=timeout,
                            *args,
                            **request_kwargs,
                        )
                    else:
                        raise

                try:
                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    saw_finish_reason = False
                    saw_usage_chunk = False
                    finish_deadline: Optional[float] = None

                    loop = asyncio.get_running_loop()

                    response_iter = response.__aiter__()

                    while True:
                        try:
                            if saw_finish_reason:
                                if finish_deadline is None:
                                    finish_deadline = (
                                        loop.time() + self._stream_finish_grace_timeout
                                    )

                                remaining = finish_deadline - loop.time()
                                if remaining <= 0:
                                    break

                                chunk = await asyncio.wait_for(
                                    response_iter.__anext__(),
                                    timeout=remaining,
                                )
                            else:
                                chunk = await response_iter.__anext__()
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            break

                        yield chunk  # 按块返回生成器中的数据
                        first_choice = chunk.choices[0] if chunk.choices else None
                        usage = getattr(chunk, "usage", None)

                        if usage is not None:
                            saw_usage_chunk = True
                            prompt_tokens, completion_tokens = self._count_tokens(chunk)
                            # stream usage 是累计值（整次请求总量），应覆盖为最新值，
                            # 而不是逐块相加，否则在多次 usage chunk 场景下会重复计数。
                            total_prompt_tokens = prompt_tokens
                            total_completion_tokens = completion_tokens

                        if saw_finish_reason and saw_usage_chunk:
                            break

                        if first_choice and first_choice.finish_reason is not None:
                            if usage is not None:
                                break
                            # 某些供应商会在 finish_reason 之后再发送 usage chunk。
                            # 进入短暂 grace 模式等待收尾 chunk，避免 UI 卡住且保留 token 统计。
                            saw_finish_reason = True

                    # 在整个流结束后统计token
                    input_tokens = get_current_context_attribute("input_tokens") or 0
                    output_tokens = get_current_context_attribute("output_tokens") or 0

                    set_current_context_attribute(
                        "input_tokens", input_tokens + total_prompt_tokens
                    )
                    set_current_context_attribute(
                        "output_tokens", output_tokens + total_completion_tokens
                    )
                finally:
                    if response is not None:
                        await self._close_stream_response(response)

                self.key_pool.decrement_task_count(key)
                break  # 如果成功，跳出重试循环
            except Exception as e:
                self.key_pool.decrement_task_count(key)
                attempt += 1
                data = json.dumps(messages, ensure_ascii=False, indent=4)
                push_warning(
                    f"{self.model_name} Interface attempt {attempt} failed: With message : {data} send, \n but exception : {str(e)} was caught",
                    location=get_location(),
                )

                key = self.key_pool.get_least_loaded_key()
                client = await self._get_or_create_client(key)

                if attempt >= self.max_retries:
                    push_error(
                        f"Max retries reached. {self.model_name} Failed to get a response for {data}",
                        location=get_location(),
                    )
                    raise e
                await asyncio.sleep(self.retry_delay)

        # 下面是一个空生成器，用于满足类型检查，实际上永远不会执行到这里
        if False:
            yield {}
