# llm_chat 装饰器

本文档介绍 SimpleLLMFunc 库中的聊天装饰器 `llm_chat`。该装饰器专门用于实现与大语言模型的对话功能，支持多轮对话、历史记录管理和工具调用。

## llm_chat 装饰器概述

### 装饰器作用

`llm_chat` 装饰器用于构建对话式应用，特别适合以下场景：

- **多轮对话**: 自动管理对话历史，支持上下文连续性
- **流式响应**: 支持实时流式返回响应内容
- **智能助手**: 集成工具调用能力，让 LLM 可以执行外部操作
- **聊天机器人**: 适合构建实时交互的聊天应用

### 主要功能特性

- **多轮对话支持**: 自动管理对话历史记录，保持上下文
- **流式响应**: 返回异步生成器，支持实时流式输出
- **工具集成**: 支持在对话中调用工具，扩展 LLM 的能力范围
- **灵活参数处理**: 智能处理历史记录参数和用户消息
- **完整的日志记录**: 与框架日志系统集成，自动追踪对话

## 装饰器用法

> ⚠️ **重要说明**：`llm_chat` 只能装饰 `async def` 定义的异步函数。装饰后得到的是 `LLMChat` callable instance；调用它会返回 **异步生成器**，请使用 `async for` 消费 `ReactOutput`。

### 基本语法

```python
from typing import List, Dict
from SimpleLLMFunc import llm_chat

@llm_chat(
    llm_interface=llm_interface,           # LLM interface instance (required)
    toolkit=None,                          # Tool list (optional)
    max_tool_calls=None,                   # Max tool calls (optional)
    stream=True,                           # Stream mode (optional)
    self_reference=None,                   # Shared SelfReference object (optional)
    self_reference_key=None,               # SelfReference memory key (optional)
    **llm_kwargs                           # Other LLM kwargs
)
async def your_chat_function(
    message: str,
    history: List[Dict[str, str]] | None = None,
):
    """
    Describe assistant role and behavior here.
    This docstring is used as the system prompt.
    """
    pass
```

> 提示：函数体不会被执行，DocString 才是 Prompt；建议直接使用 `pass`。

### 参数说明

- **llm_interface** (必需): LLM 接口实例，用于与大语言模型通信
- **toolkit** (可选): 工具列表，可以是 Tool 对象或被 @tool 装饰的函数
- **max_tool_calls** (可选): 最大工具调用次数；默认为 `None`，表示框架不主动施加工具调用上限。如需更严格保护，请显式传入较小整数。
- **stream** (可选): 是否启用流式模式，默认为 False
- **strict_signature** (可选): 当为 True 时强制 `agent(history, message: str, _template_params=None)` 规范签名
- **self_reference** (可选): 共享的 `SelfReference` 实例；若未显式传入，`llm_chat` 会从 `PyRepl` runtime backend 中自动探测
- **self_reference_key** (可选): 本聊天函数的记忆键，默认为函数名
- ****llm_kwargs**: 额外的关键字参数，将直接传递给 LLM 接口（如 temperature、top_p 等）

### 运行时中断（AbortSignal）

你可以在调用时传入 `_abort_signal` 来中断正在执行的回合（停止流式输出并取消正在执行的工具调用）。

推荐使用常量 `ABORT_SIGNAL_PARAM`，避免硬编码参数名：

```python
from SimpleLLMFunc.hooks import AbortSignal, ABORT_SIGNAL_PARAM

abort_signal = AbortSignal()

async for output in your_chat_function(
    "你好",
    history=[],
    **{ABORT_SIGNAL_PARAM: abort_signal},
):
    ...

# 在其他协程中触发中断
abort_signal.abort("user_interrupt")
```

详见 [中断与取消](abort.md)。

### 返回值

`llm_chat` 装饰后得到 `LLMChat` callable instance。调用该对象返回异步生成器，每次迭代产生 `ReactOutput`：

- `ResponseYield`: 包含响应和消息列表
- `EventYield`: 包含 ReAct 循环中的事件（如工具调用开始/结束、LLM 调用等）

请用 `is_response_yield(output)` / `is_event_yield(output)` 区分输出类型，不要按旧版 `(chunk, history)` 元组消费。

> 注意：历史参数名应为 `history` 或 `chat_history`。若未提供符合格式的历史，框架会忽略历史并发出警告。若历史中包含 `system` 消息，最新的 `system` 会覆盖 DocString 作为系统提示，其余 `system` 会被过滤。

## 使用示例

### 示例 1: 基础聊天助手

最简单的对话助手实现：

```python
import asyncio
from typing import AsyncGenerator, Dict, List, Tuple
from SimpleLLMFunc import llm_chat, OpenAICompatible
from SimpleLLMFunc.hooks.stream import is_response_yield

# 初始化 LLM 接口
llm = OpenAICompatible.load_from_json_file("provider.json")["openai"]["gpt-3.5-turbo"]

# 创建聊天函数
@llm_chat(llm_interface=llm, stream=True)
async def simple_chat(
    message: str,
    history: List[Dict[str, str]] | None = None,
):
    """你是一个友好的聊天助手，善于回答各种问题。"""
    pass

# 使用示例
async def main():
    history = []
    user_message = "你好，请介绍一下你自己"

    print(f"用户: {user_message}")
    print("助手: ", end="", flush=True)

    # 流式获取响应
    async for output in simple_chat(user_message, history):
        if is_response_yield(output):
            print(output.response, end="", flush=True)
            history = output.messages

    print()  # 换行

asyncio.run(main())
```

### 示例 2: 带工具调用的聊天助手

展示如何在对话中使用工具：

```python
import asyncio
from typing import AsyncGenerator, Dict, List, Tuple
from SimpleLLMFunc import llm_chat, tool, OpenAICompatible
from SimpleLLMFunc.hooks.stream import is_response_yield

# 定义工具
@tool(name="get_weather", description="获取指定城市的天气信息")
async def get_weather(city: str) -> Dict[str, str]:
    """
    获取指定城市的天气信息

    Args:
        city: 城市名称

    Returns:
        包含温度、湿度和天气状况的字典
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temperature": "25°C", "humidity": "60%", "condition": "晴朗"},
        "上海": {"temperature": "28°C", "humidity": "75%", "condition": "多云"},
        "广州": {"temperature": "30°C", "humidity": "80%", "condition": "小雨"}
    }
    return weather_data.get(city, {"temperature": "20°C", "humidity": "50%", "condition": "未知"})

# 初始化 LLM
llm = OpenAICompatible.load_from_json_file("provider.json")["openai"]["gpt-3.5-turbo"]

# 创建带工具的聊天函数
@llm_chat(llm_interface=llm, toolkit=[get_weather], stream=True)
async def weather_chat(
    message: str,
    history: List[Dict[str, str]] | None = None,
):
    """
    你是一个天气助手，可以查询城市天气信息。
    当用户询问天气时，使用 get_weather 工具来获取实时信息。
    """
    pass

# 使用示例
async def main():
    history = []
    query = "北京今天天气怎么样？"

    print(f"用户: {query}")
    print("助手: ", end="", flush=True)

    async for output in weather_chat(query, history):
        if is_response_yield(output):
            print(output.response, end="", flush=True)
            history = output.messages

    print()

asyncio.run(main())
```

### 示例 3: 交互式多轮对话

展示如何维护完整的对话会话：

```python
import asyncio
from typing import AsyncGenerator, Dict, List, Tuple
from SimpleLLMFunc import llm_chat, OpenAICompatible
from SimpleLLMFunc.hooks.stream import is_response_yield

llm = OpenAICompatible.load_from_json_file("provider.json")["openai"]["gpt-3.5-turbo"]

@llm_chat(llm_interface=llm, stream=True)
async def multi_turn_chat(
    message: str,
    history: List[Dict[str, str]] | None = None,
):
    """你是一个专业的编程助手，精通 Python 和 JavaScript。"""
    pass

async def interactive_chat_session():
    """运行交互式聊天会话"""
    history = []

    print("=== 编程助手（输入 'quit' 退出）===\n")

    # 这里使用 input() 只是为了演示，实际应用中应使用异步输入
    while True:
        # 在实际应用中，应该使用更好的异步输入方法
        user_input = input("你: ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        print("助手: ", end="", flush=True)

        response_text = ""
        async for output in multi_turn_chat(user_input, history):
            if is_response_yield(output):
                text = str(output.response or "")
                print(text, end="", flush=True)
                response_text += text
                history = output.messages

        print("\n")

# 非交互式演示（避免阻塞 input()）
async def demo():
    """演示版本，不使用交互式输入"""
    history = []

    messages = [
        "Python 中什么是列表推导式？",
        "如何使用异步编程？",
    ]

    for user_message in messages:
        print(f"\n用户: {user_message}")
        print("助手: ", end="", flush=True)

        async for output in multi_turn_chat(user_message, history):
            if is_response_yield(output):
                print(output.response, end="", flush=True)
                history = output.messages

        print()

asyncio.run(demo())
```

## 高级特性

### SelfReference + runtime primitives

当挂载了带 runtime 的工具（例如 `PyRepl`）时，框架会在 system prompt 顶部注入一段去重后的工具最佳实践块；runtime primitive 的指引会包含在这些工具自己的 best practices 中。

这段 guidance 会在每个回合按当前 runtime 状态重新构建，因此不会把临时运行时说明写进持久化 context；需要持久修改上下文时，优先使用 `runtime.selfref.context.remember(...)` 或在里程碑后调用 `runtime.selfref.context.compact(...)`。

这段 guidance 会告诉 agent：

- 如何发现当前挂载的 runtime 能力（`runtime.list_primitives()`）
- 如何查看单个契约（`runtime.get_primitive_spec(name)`）或按条件筛选契约（`runtime.list_primitive_specs(names=[...], contains="...")`）
- 如何查看 selfref 命名空间 guidance（`runtime.selfref.guide()`）
- `reset_repl` 会清理 REPL 变量，并继续保留当前 runtime backend 状态
- 当绑定了 `SelfReference` 上下文时，本 chat 函数对应的 memory key 作用域

`PyRepl()` 默认会安装 builtin `selfref` pack；`llm_chat` 会直接从 toolkit 的 runtime backend 解析并复用这份默认 backend。

当 chat 绑定了 `SelfReference` 后，框架会在 ReAct 生命周期里自动同步当前 turn 状态：工具批次前绑定最新 history，工具批次后优先提交 pending compaction / context message 更新，并在 finalize 阶段兜底提交最终 context。

示例：

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import PyRepl, SelfReference

repl = PyRepl()
self_reference = repl.get_runtime_backend("selfref")
assert isinstance(self_reference, SelfReference)

@llm_chat(
    llm_interface=llm,
    toolkit=repl.toolset,
    self_reference_key="agent_main",
)
async def agent(message: str, history=None):
    """You are a practical coding assistant."""
```

单次调用的高级覆盖（可选）：

```python
await agent(
    "task",
    _template_params={
        "__self_reference_key_override": "agent_alt",
        "__self_reference_toolkit_override": repl.toolset,
    },
)
```

在 `execute_code` 中，优先通过 runtime primitives 处理上下文：

```python
runtime.selfref.context.remember(
    "User preference: answer in concise bullet points.",
)
```

Runtime self-reference 原语参考：

- `runtime.selfref.guide()`: 返回命名空间概览与 fork / context 最佳实践清单。
- `runtime.selfref.context.inspect(key=None)`: 返回当前上下文快照，包括 `experiences`、结构化 `summary`、以及完整只读 `messages`。
- `runtime.selfref.context.remember(text, key=None)`: 向 system 内的经验块追加一条 durable experience。
- `runtime.selfref.context.forget(experience_id, key=None)`: 通过 id 删除一条错误或过时的 durable experience。
- `runtime.selfref.context.compact(..., key=None)`: 排队一次 milestone compaction。当前工具批次结束后会优先提交，让下一次同 turn 的 LLM 调用看到结构化 assistant summary；如果当前 turn 不再继续调用 LLM，finalize 阶段会兜底提交。
- `runtime.selfref.fork.spawn(message, ...)`: 异步创建子 self-fork（chat 形态）。
- 子 fork 继承的是 fork 发生前的上下文快照，不会把父 agent 当前尚未闭合的 fork tool-call 场景当成自己的当前动作。
- `runtime.selfref.fork.gather_all(fork_id_or_list=None, include_history=False)`: 聚合 fork 结果，返回 `dict[fork_id -> ForkResult]`（用 `.items()` / `.values()` 遍历）。

默认情况下 fork 结果是紧凑模式（`history_included=False`，并提供 `status`、`response`、`result`、`memory_key`、`history_count` 等元数据），这样主上下文可以保持简洁。
读取结果时先检查 `status`，成功后读取 `response` 或 `result`，失败时检查 `error_type` / `error_message`。
确实需要完整子历史时，再显式使用 `include_history=True`。


```python
from SimpleLLMFunc.hooks import is_event_yield

async for output in agent("analyze and split"):
    if not is_event_yield(output):
        continue
    if output.origin.fork_id:
        print(f"fork={output.origin.fork_id} type={output.event.event_type}")
    else:
        print(f"main type={output.event.event_type}")
```

清理上下文：

- 使用 `reset_repl` 清理 Python runtime 变量。
- 使用 `runtime.selfref.context.inspect()` 查看当前完整上下文。
- 使用 `runtime.selfref.context.forget(...)` 删除错误的 durable experience。
- 在 milestone 完成后使用 `runtime.selfref.context.compact(...)` 保留结构化 assistant summary 并清空 stale working transcript；如果当前 turn 还有下一次模型调用，它会直接看到 compact 后的上下文。

### 返回模式



```python
# 返回文本（默认）
async def text_mode_chat(message: str, history=None):
    """聊天函数"""
    pass

# 返回原始响应对象（用于获取 token 使用量等详细信息）
async def raw_mode_chat(message: str, history=None):
    """聊天函数"""
    pass
```

### 并发聊天会话

使用 `asyncio.gather` 处理多个并发的聊天会话：

```python
async def concurrent_chats():
    """并发处理多个聊天会话"""

    @llm_chat(llm_interface=llm, stream=True)
    async def chat(message: str, history=None):
        """通用聊天函数"""
        pass

    # 定义多个会话
    sessions = [
        {"user_id": "user_1", "message": "你好"},
        {"user_id": "user_2", "message": "如何学习Python？"},
        {"user_id": "user_3", "message": "告诉我一个笑话"},
    ]

    async def handle_session(session):
        """处理单个会话"""
        history = []
        results = []

        async for output in chat(session["message"], history):
            if is_response_yield(output):
                results.append(str(output.response or ""))
                history = output.messages

        return session["user_id"], "".join(results)

    # 并发执行所有会话
    results = await asyncio.gather(
        *[handle_session(session) for session in sessions]
    )

    for user_id, response in results:
        print(f"{user_id}: {response}\n")

asyncio.run(concurrent_chats())
```

## 最佳实践

### 1. 错误处理

```python
async def robust_chat():
    history = []
    try:
        async for output in multi_turn_chat("测试", history):
            if is_response_yield(output):
                print(output.response, end="", flush=True)
                history = output.messages
    except Exception as e:
        print(f"聊天出错: {e}")
```

### 2. 超时控制

```python
async def chat_with_timeout():
    history = []
    try:
        async with asyncio.timeout(30):  # Python 3.11+
            async for output in multi_turn_chat("测试", history):
                if is_response_yield(output):
                    print(output.response, end="", flush=True)
                    history = output.messages
    except asyncio.TimeoutError:
        print("聊天超时")
```

### 3. 历史记录限制

为避免上下文过长，限制历史记录长度：

```python
MAX_HISTORY_LENGTH = 10

def trim_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """保留最近的 N 条消息"""
    if len(history) > MAX_HISTORY_LENGTH:
        return history[-MAX_HISTORY_LENGTH:]
    return history

async def chat_with_limited_history():
    history = []

    messages = ["第一条消息", "第二条消息", "第三条消息"]

    for msg in messages:
        # 限制历史记录
        history = trim_history(history)

        async for output in multi_turn_chat(msg, history):
            if is_response_yield(output):
                print(output.response, end="", flush=True)
                history = output.messages
        print()

asyncio.run(chat_with_limited_history())
```

### 4. 日志与调试

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# SimpleLLMFunc 日志
logger = logging.getLogger("SimpleLLMFunc")
logger.setLevel(logging.DEBUG)
```

### 5. 事件流（Event Stream）

事件流是 SimpleLLMFunc v0.5.0+ 引入的高级特性，允许你实时观察 ReAct 循环的完整执行过程。

通过消费 `ReactOutput` 事件流，你可以：

- **实时监控**：观察 LLM 调用、工具调用的实时状态
- **性能分析**：获取详细的执行统计和性能指标
- **自定义 UI**：基于事件构建丰富的用户界面
- **调试支持**：深入了解 ReAct 循环的执行细节

**基本用法**：

```python
@llm_chat(llm_interface=llm)
async def chat(message: str, history=None):
    """智能助手"""
    pass

# 处理事件和响应
from SimpleLLMFunc.hooks import ResponseYield, EventYield

async for output in chat("查询天气"):
    if isinstance(output, ResponseYield):
        # 原始响应对象（流式时为 chunk）。
        # 文本渲染建议使用 LLMChunkArriveEvent 的 accumulated_content。
        pass
    elif isinstance(output, EventYield):
        print(f"事件: {output.event.event_type}")
```

**详细文档**：请参考 [事件流文档](event_stream.md) 了解完整的事件类型、使用示例和最佳实践。

## 常见问题

### Q: 如何保存和恢复对话历史？

```python
import json

def save_history(history: List[Dict[str, str]], filename: str):
    """保存对话历史到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(filename: str) -> List[Dict[str, str]]:
    """从文件加载对话历史"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# 使用
history = load_history("chat_history.json")
# ... 继续对话 ...
save_history(history, "chat_history.json")
```

### Q: 如何处理 LLM 拒绝或无效响应？

```python
async def robust_chat_with_retry():
    history = []
    max_retries = 3

    for attempt in range(max_retries):
        try:
            collected = ""
            async for output in multi_turn_chat("测试", history):
                if is_response_yield(output):
                    collected += str(output.response or "")
                    history = output.messages

            if collected.strip():
                print(f"成功: {collected}")
                break
            else:
                print(f"尝试 {attempt + 1}: 收到空响应，重试...")
        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
            if attempt == max_retries - 1:
                raise
```

---

通过这些示例和最佳实践，你可以构建功能强大的对话应用。`llm_chat` 装饰器提供了简洁而强大的方式来实现复杂的对话逻辑。
