# 事件流系统（Event Stream）

事件流是 SimpleLLMFunc v0.5.0+ 引入的高级特性，允许你实时观察 ReAct 循环的完整执行过程。通过启用事件流，你可以监控 LLM 调用、工具调用、流式响应等所有关键环节，实现更精细的控制和更好的用户体验。

在当前版本中，事件流还提供了统一来源元数据（`EventYield.origin`），用于稳定识别主会话与 fork 子会话。

**适用范围**：事件流系统同时支持 `@llm_chat` 和 `@llm_function` 装饰器，提供统一的事件观测体验。

## 概述

### 什么是事件流？

事件流（Event Stream）是一种观察者模式的应用，它会在 ReAct 循环执行过程中产生一系列事件，包括：

- **LLM 调用事件**：LLM 调用开始、流式 chunk 到达、调用结束
- **工具调用事件**：工具调用开始、执行完成、批次处理
- **ReAct 循环事件**：循环开始、迭代开始/结束、循环结束
- **执行统计**：Token 使用量、执行耗时、调用次数等

### 为什么需要事件流？

在传统使用中，你只能获得最终的响应结果。而事件流提供了：

1. **实时监控**：观察 LLM 和工具调用的实时状态
2. **性能分析**：获取详细的执行统计和性能指标（Token 用量、耗时等）
3. **自定义 UI**：基于事件构建丰富的用户界面（进度条、流式显示等）
4. **调试支持**：深入了解 ReAct 循环的执行细节
5. **状态管理**：通过 Wrapper 模式实现 Context Compression、状态持久化等

### 设计哲学：Agent 作为无状态函数

SimpleLLMFunc 遵循一个核心设计哲学：**Agent 本身是一个函数，不管理自己的状态**。

这意味着：

- ✅ **无状态设计**：`@llm_chat` 装饰的函数是纯函数，每次调用都是独立的
- ✅ **状态外置**：所有状态（包括 `history`）都通过参数传入和返回值传出
- ✅ **可组合性**：通过 wrapper 函数实现状态管理和高级功能

**为什么这样设计？**

1. **函数式编程**：符合函数式编程理念，易于测试和推理
2. **灵活性**：状态管理完全由用户控制，可以实现各种自定义逻辑
3. **可扩展性**：通过 wrapper 函数可以轻松添加 Context Compression、状态持久化等功能

**如何实现状态管理和修改？**

通过事件流，你可以：

1. **监控状态**：通过事件获取 Agent 的内部状态信息（消息历史、工具调用等）
2. **修改状态**：在 wrapper 函数中拦截和修改 `history`，实现 Context Compression 等功能
3. **状态持久化**：在 wrapper 中实现状态的保存和恢复

下面我们将详细展示如何通过 wrapper 函数实现这些功能。

## 启用事件流

### 基本用法

`@llm_chat` 默认产生 `ReactOutput` 事件流；`@llm_function` 使用 `.stream(...)` 消费事件流：

#### llm_chat 的事件流

`@llm_chat` 装饰器在启用事件流时，返回一个生成器，yield `ReactOutput`：

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.hooks import ReactOutput, ResponseYield, EventYield

@llm_chat(
    llm_interface=llm,
    toolkit=[calculate, get_weather],
    stream=True,
)
async def chat(message: str, history=None):
    """智能助手"""
    pass

# 使用事件流
async for output in chat("Hello"):
    if isinstance(output, ResponseYield):
        # 处理响应数据
        print(output.response)
    elif isinstance(output, EventYield):
        # 处理事件
        event = output.event
        print(f"事件类型: {event.event_type}")
```

#### llm_function 的事件流

`@llm_function` 装饰器在启用事件流时，也返回一个生成器，yield `ReactOutput`：

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.hooks import ReactOutput, ResponseYield, EventYield

@llm_function(
    llm_interface=llm,
    toolkit=[calculate, get_weather],
)
async def analyze_text(text: str) -> str:
    """分析文本内容"""
    pass

# 使用事件流
async for output in analyze_text("Hello world"):
    if isinstance(output, ResponseYield):
        # 处理响应数据（最终结果）
        result = output.response
        print(f"分析结果: {result}")
    elif isinstance(output, EventYield):
        # 处理事件（LLM 调用、工具调用等）
        event = output.event
        print(f"事件: {event.event_type}")
```

### 返回值变化

启用事件流后，函数的返回值类型会发生变化：

**llm_chat 输出**:

```python
async for output in chat("Hello"):
    if is_response_yield(output):
        print(output.response)
```

**llm_chat 事件输出**:

```python
async for output in chat("Hello"):
    if isinstance(output, ResponseYield):
        print(output.response)
    elif isinstance(output, EventYield):
        print(f"事件: {output.event.event_type}")
```

**llm_function 普通调用**:

```python
result = await analyze_text("Hello")
print(result)
```

**llm_function 事件流调用**:

```python
async for output in analyze_text("Hello"):
    if isinstance(output, ResponseYield):
        result = output.response
        print(f"结果: {result}")
    elif isinstance(output, EventYield):
        print(f"事件: {output.event.event_type}")
```

## 核心类型

### ReactOutput

`ReactOutput` 是一个 Tagged Union 类型，包含两种可能的值：

```python
from SimpleLLMFunc.hooks import ReactOutput, ResponseYield, EventYield

# ReactOutput = ResponseYield | EventYield
```

### ResponseYield

响应数据，包含 LLM 的响应内容和消息历史：

```python
@dataclass
class ResponseYield:
    response: Union[LLMResponse, LLMStreamChunk, str]
    messages: MessageList
    type: Literal["response"] = "response"
```

说明：

- `llm_chat` 事件流模式中，`response` 为原始响应对象或流式 chunk
- `llm_function` 事件流模式中，`response` 为**解析后的最终结果**（如 `str` / Pydantic 对象）

### EventYield

事件数据，包含 ReAct 循环中的各种事件：

```python
@dataclass
class EventYield:
    event: ReActEvent  # 事件对象
    origin: EventOrigin  # 事件来源（会话/分叉/工具关联元数据）
    type: Literal["event"] = "event"
```

### EventOrigin（来源元数据）

`EventOrigin` 用于描述事件在调用树中的来源。常用字段如下：

- `session_id`: 当前会话 ID（同一会话内稳定）
- `agent_call_id`: 当前 agent 调用 ID
- `parent_agent_call_id`: 父 agent 调用 ID（如果存在）
- `event_seq`: 会话内递增事件序号
- `fork_id`: fork 场景下的分叉 ID（主会话通常为 `None`）
- `fork_depth`: 分叉深度（主会话通常为 `0`）
- `fork_seq`: fork 内事件序号（若存在）
- `selfref_instance_id`: SelfReference 实例标识（若存在）
- `memory_key` / `source_memory_key`: selfref 记忆键上下文
- `tool_name` / `tool_call_id`: 工具事件关联信息

快速示例：

```python
from SimpleLLMFunc.hooks import is_event_yield

async for output in chat("请并行处理多个子任务"):
    if not is_event_yield(output):
        continue

    if output.origin.fork_id:
        print(
            f"[fork:{output.origin.fork_id} depth={output.origin.fork_depth}] "
            f"{output.event.event_type}"
        )
    else:
        print(f"[main] {output.event.event_type}")
```

## 事件类型

事件流包含以下类型的事件，按执行顺序排列：

### 1. ReactStartEvent

ReAct 循环开始事件，在循环开始时触发。

```python
@dataclass
class ReactStartEvent(ReActEvent):
    user_task_prompt: str  # 用户任务提示
    initial_messages: MessageList  # 初始消息列表
    available_tools: ToolDefinitionList  # 可用工具列表
```

**使用场景**：初始化 UI、显示开始提示、记录 trace_id

### 2. ReactIterationStartEvent

ReAct 迭代开始事件，每次迭代开始时触发。

```python
@dataclass
class ReactIterationStartEvent(ReActEvent):
    current_messages: MessageList  # 当前消息历史
```

**使用场景**：显示迭代进度、更新消息历史

### 3. LLMCallStartEvent

LLM 调用开始事件，在 LLM 调用前触发。

```python
@dataclass
class LLMCallStartEvent(ReActEvent):
    messages: MessageList  # 消息列表
    tools: ToolDefinitionList  # 工具定义列表
    llm_kwargs: Dict[str, Any]  # LLM 调用参数
    stream: bool  # 是否流式调用
```

**使用场景**：显示"正在思考..."提示、记录调用参数

### 4. LLMChunkArriveEvent

LLM 流式 chunk 到达事件（仅流式模式）。

```python
@dataclass
class LLMChunkArriveEvent(ReActEvent):
    chunk: LLMStreamChunk  # LLM 返回的 chunk 对象
    accumulated_content: str  # 累积的内容
    chunk_index: int  # Chunk 序号
```

**使用场景**：实时渲染流式响应、显示打字效果

### 5. LLMCallEndEvent

LLM 调用结束事件，在调用完成后触发。

```python
@dataclass
class LLMCallEndEvent(ReActEvent):
    response: LLMResponse  # LLM 响应对象
    messages: MessageList  # 更新后的消息列表
    tool_calls: List[ToolCall]  # 提取的工具调用列表
    execution_time: float  # 执行耗时
    usage: Optional[LLMUsage]  # Token 使用统计
```

**使用场景**：显示 Token 使用量、记录性能指标、检查工具调用

### 6. ToolCallsBatchStartEvent

工具调用批次开始事件，当 LLM 返回多个工具调用时触发。

```python
@dataclass
class ToolCallsBatchStartEvent(ReActEvent):
    tool_calls: List[ToolCall]  # 工具调用列表
    batch_size: int  # 批次大小
```

**使用场景**：显示工具调用批次信息、准备并行执行

### 7. ToolCallStartEvent

单个工具调用开始事件。

```python
@dataclass
class ToolCallStartEvent(ReActEvent):
    tool_name: str  # 工具名称
    tool_call_id: str  # 工具调用 ID
    arguments: ToolCallArguments  # 工具调用参数
    tool_call: ToolCall  # 完整的工具调用对象
```

**使用场景**：显示工具调用信息、记录调用参数

### 8. ToolCallArgumentsDeltaEvent

工具调用参数增量事件（仅流式模式）。

```python
@dataclass
class ToolCallArgumentsDeltaEvent(ReActEvent):
    tool_name: str
    tool_call_id: str
    argname: str
    argcontent_delta: str
```

**使用场景**：在工具调用尚未开始前，增量渲染参数拼装过程（适合 TUI 实时面板）。

### 9. ToolCallEndEvent

单个工具调用结束事件（**关键**：立即获取结果）。

```python
@dataclass
class ToolCallEndEvent(ReActEvent):
    tool_name: str  # 工具名称
    tool_call_id: str  # 工具调用 ID
    arguments: ToolCallArguments  # 工具调用参数
    result: ToolResult  # 工具执行结果
    execution_time: float  # 执行耗时
    success: bool  # 是否成功执行
```

**使用场景**：显示工具执行结果、更新 UI、记录执行时间

### 10. CustomEvent

工具内部主动发射的自定义事件，用于进度、日志、阶段状态等实时反馈。

```python
@dataclass
class CustomEvent(ReActEvent):
    event_name: str  # 自定义事件名
    data: Any = None  # 自定义数据
    tool_name: Optional[str] = None  # 所属工具（如果可用）
    tool_call_id: Optional[str] = None  # 所属工具调用 ID（如果可用）
```

**使用场景**：在 TUI 或自定义 UI 中增量渲染工具执行过程（例如 PyRepl stdout/stderr、批处理进度）。

### 11. ToolCallsBatchEndEvent

工具调用批次结束事件，批次全部完成后触发。

```python
@dataclass
class ToolCallsBatchEndEvent(ReActEvent):
    tool_results: List[ToolCallResult]  # 所有工具调用结果
    batch_size: int  # 批次大小
    total_execution_time: float  # 总执行时间
    success_count: int  # 成功数量
    error_count: int  # 失败数量
```

**使用场景**：显示批次统计、性能分析

### 12. ReactIterationEndEvent

ReAct 迭代结束事件，每次迭代完成时触发。

```python
@dataclass
class ReactIterationEndEvent(ReActEvent):
    messages: MessageList  # 更新后的消息历史
    iteration_time: float  # 迭代耗时
    tool_calls_count: int  # 本次迭代的工具调用数量
```

**使用场景**：显示迭代统计、更新进度

### 13. ReactEndEvent

ReAct 循环结束事件，循环结束时触发。

```python
@dataclass
class ReactEndEvent(ReActEvent):
    final_response: str  # 最终响应内容
    final_messages: MessageList  # 完整的消息历史
    total_iterations: int  # 总迭代次数
    total_execution_time: float  # 总执行时间
    total_tool_calls: int  # 总工具调用次数
    total_llm_calls: int  # 总 LLM 调用次数
    total_token_usage: Optional[LLMUsage]  # 总 token 使用统计
```

当通过 `AbortSignal` 中断回合时，`ReactEndEvent.extra` 会附加：

- `aborted: true`
- `abort_reason: <reason>`（如果调用 `abort(reason)` 时提供了 reason）

**使用场景**：显示完整统计、性能分析、保存执行记录

## Fork 场景事件路由

在 fork agent 场景中，事件大致分为两类：

1. **标准 ReAct 事件**（如 `LLMCallStartEvent`、`ToolCallStartEvent`）
   - 通过 `EventYield.origin.fork_id` / `fork_depth` 标识归属 fork。
   - 适合在 UI 层按 fork 维度拆分消息流与工具卡片。

2. **selfref 生命周期与流式自定义事件**（`CustomEvent`）
   - 生命周期：`selfref_fork_start` / `selfref_fork_spawned` / `selfref_fork_end` / `selfref_fork_error`
   - 流式文本：`selfref_fork_stream_open` / `selfref_fork_stream_delta` / `selfref_fork_stream_close`
   - 适合展示 fork 启停状态与子任务增量输出。

如果你的处理链只保留了 `ReActEvent` 对象，也可从 `event.extra["origin"]` 读取同源元数据。

## 使用示例

### 示例 1: 基本事件处理

```python
import asyncio
from SimpleLLMFunc import llm_chat, tool
from SimpleLLMFunc.hooks import (
    ReactOutput,
    ResponseYield,
    EventYield,
    ReactStartEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ReactEndEvent,
)

@tool(name="calculate", description="执行数学计算")
async def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

@llm_chat(
    llm_interface=llm,
    toolkit=[calculate],
    stream=True,
)
async def chat(message: str, history=None):
    """智能助手"""
    pass

async def main():
    async for output in chat("帮我计算 25 * 4 + 18"):
        if isinstance(output, EventYield):
            event = output.event
            
            if isinstance(event, ReactStartEvent):
                print(f"🚀 开始处理 (trace_id: {event.trace_id[:8]}...)")
            
            elif isinstance(event, LLMCallStartEvent):
                print(f"🤖 LLM 调用开始 ({'流式' if event.stream else '非流式'})")
            
            elif isinstance(event, LLMChunkArriveEvent):
                # 实时显示流式响应
                chunk_content = event.chunk.choices[0].delta.content
                if chunk_content:
                    print(chunk_content, end="", flush=True)
            
            elif isinstance(event, ToolCallStartEvent):
                print(f"\n🛠️  调用工具: {event.tool_name}")
                print(f"   参数: {event.arguments}")
            
            elif isinstance(event, ToolCallEndEvent):
                print(f"   ✅ 结果: {event.result} ({event.execution_time:.2f}s)")
            
            elif isinstance(event, ReactEndEvent):
                print(f"\n📊 统计:")
                print(f"   总耗时: {event.total_execution_time:.2f}s")
                print(f"   LLM 调用: {event.total_llm_calls} 次")
                print(f"   工具调用: {event.total_tool_calls} 次")
                if event.total_token_usage:
                    print(f"   Token: {event.total_token_usage.total_tokens}")
        
        elif isinstance(output, ResponseYield):
            # 处理响应数据（如果需要）
            pass

asyncio.run(main())
```

### 示例 2: 使用类型守卫

```python
from SimpleLLMFunc.hooks import is_response_yield, is_event_yield

async for output in chat("Hello"):
    if is_response_yield(output):
        # 类型检查器会知道 output 是 ResponseYield
        print(output.response)
    elif is_event_yield(output):
        # 类型检查器会知道 output 是 EventYield
        print(output.event.event_type)
```

### 示例 3: 使用过滤器

```python
from SimpleLLMFunc.hooks import (
    responses_only,
    events_only,
    filter_events,
    ReActEventType,
)

# 只获取响应（向后兼容）
async for response, history in responses_only(chat("Hello")):
    print(response)

# 只获取事件
async for event in events_only(chat("Hello")):
    print(f"事件: {event.event_type}")

# 过滤特定事件类型
async for event in filter_events(
    chat("查询天气"),
    event_types={ReActEventType.TOOL_CALL_END}
):
    print(f"工具调用完成: {event.tool_name}")
    print(f"结果: {event.result}")
```

### 示例 4: 事件观测器

使用 `@with_event_observer` 装饰器自动处理所有事件：

```python
from SimpleLLMFunc.hooks import with_event_observer

async def log_event(event: ReActEvent):
    """记录所有事件到日志系统"""
    print(f"[{event.timestamp}] {event.event_type}: {event.func_name}")

@with_event_observer(log_event)
@llm_chat(llm_interface=llm)
async def observed_chat(message: str, history=None):
    """带事件观测的聊天"""
    pass

# 使用
async for output in observed_chat("Hello"):
    # 所有事件都会自动被 log_event 处理
    if is_response_yield(output):
        print(output.response)
```

### 示例 5: 完整的事件流 Chatbot

参考 [examples/event_stream_chatbot.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/event_stream_chatbot.py) 获取一个功能完整的示例，包括：

- 实时流式响应渲染（使用 Rich 库）
- 工具调用可视化
- 完整的执行统计
- 美观的终端 UI

## 最佳实践

### 1. 事件处理顺序

事件按照 ReAct 循环的执行顺序产生，建议按照以下顺序处理：

```text
ReactStartEvent
  → ReactIterationStartEvent
    → LLMCallStartEvent
      → LLMChunkArriveEvent (流式模式)
      → LLMCallEndEvent
    → ToolCallsBatchStartEvent (如果有工具调用)
      → ToolCallStartEvent
      → ToolCallEndEvent
      → ...
    → ToolCallsBatchEndEvent
  → ReactIterationEndEvent
  → (重复迭代...)
→ ReactEndEvent
```

### 2. 性能考虑

- 事件处理应该是轻量级的，避免阻塞主流程
- 使用异步操作处理事件（如日志记录、UI 更新）
- 对于高频事件（如 `LLMChunkArriveEvent`），考虑批量处理

### 3. 错误处理

事件处理中的错误不应影响主流程：

```python
async for output in chat("Hello"):
    if is_event_yield(output):
        try:
            # 处理事件
            handle_event(output.event)
        except Exception as e:
            # 记录错误但不中断流程
            logger.error(f"事件处理失败: {e}")
```

### 4. UI 更新

使用事件流构建响应式 UI：

```python
class ChatUI:
    def __init__(self):
        self.response_text = ""
        self.tool_calls = []
    
    async def handle_event(self, event: ReActEvent):
        if isinstance(event, LLMChunkArriveEvent):
            # 实时更新响应文本
            self.response_text = event.accumulated_content
            self.update_ui()
        
        elif isinstance(event, ToolCallEndEvent):
            # 更新工具调用列表
            self.tool_calls.append({
                "name": event.tool_name,
                "result": event.result,
            })
            self.update_ui()
```

(state-management-wrapper-mode)=
### 5. 状态管理和修改（Wrapper 模式）

由于 Agent 是无状态的，所有状态都通过参数和返回值传递。通过 wrapper 函数，你可以：

- **监控状态**：通过事件获取内部状态
- **修改状态**：拦截和修改 `history` 参数
- **实现高级功能**：Context Compression、状态持久化等

#### 5.1 基础 Wrapper：状态监控

```python
from typing import List, Dict, Any
from SimpleLLMFunc.hooks import ReactOutput, ResponseYield, EventYield, ReactEndEvent

def with_state_monitoring(chat_func):
    """Wrapper：监控 Agent 的状态变化"""
    async def wrapper(message: str, history: List[Dict[str, str]] = None):
        # 记录初始状态
        initial_history_length = len(history) if history else 0
        print(f"初始历史长度: {initial_history_length}")
        
        # 代理调用
        async for output in chat_func(message, history):
            if isinstance(output, EventYield):
                event = output.event
                
                # 监控状态变化
                if isinstance(event, ReactEndEvent):
                    print(f"最终历史长度: {len(event.final_messages)}")
                    print(f"总迭代次数: {event.total_iterations}")
                    print(f"总工具调用: {event.total_tool_calls}")
            
            yield output
    
    return wrapper

# 使用
@llm_chat(llm_interface=llm)
async def chat(message: str, history=None):
    """智能助手"""
    pass

# 包装函数
monitored_chat = with_state_monitoring(chat)

# 使用包装后的函数
async for output in monitored_chat("Hello", history=[]):
    if isinstance(output, ResponseYield):
        print(output.response)
```

#### 5.2 高级 Wrapper：Context Compression

通过修改 `history` 参数，可以实现 Context Compression（上下文压缩）：

```python
from typing import List, Dict, Optional
from SimpleLLMFunc.hooks import ResponseYield, EventYield
from copy import deepcopy

def with_context_compression(
    chat_func,
    max_history_length: int = 10,
    compression_strategy: str = "keep_recent"
):
    """
    Wrapper：实现上下文压缩
    
    Args:
        chat_func: 原始的 chat 函数
        max_history_length: 最大历史长度
        compression_strategy: 压缩策略（"keep_recent" 或 "keep_important"）
    """
    def compress_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """压缩历史记录"""
        if not history or len(history) <= max_history_length:
            return history
        
        if compression_strategy == "keep_recent":
            # 保留最近的消息
            return history[-max_history_length:]
        elif compression_strategy == "keep_important":
            # 保留系统消息和最近的消息
            system_messages = [msg for msg in history if msg.get("role") == "system"]
            recent_messages = history[-max_history_length:]
            # 去重（如果系统消息已经在最近消息中）
            combined = system_messages + recent_messages
            seen = set()
            result = []
            for msg in combined:
                msg_id = id(msg)  # 使用对象 ID 去重
                if msg_id not in seen:
                    seen.add(msg_id)
                    result.append(msg)
            return result
        else:
            return history[-max_history_length:]
    
    async def wrapper(message: str, history: List[Dict[str, str]] = None):
        # 压缩输入的历史记录
        compressed_history = compress_history(history or [])
        
        # 代理调用
        async for output in chat_func(message, compressed_history):
            if isinstance(output, ResponseYield):
                # 压缩返回的历史记录
                compressed_messages = compress_history(output.messages)
                
                # 创建新的 ResponseYield，包含压缩后的历史
                yield ResponseYield(
                    response=output.response,
                    messages=compressed_messages
                )
            else:
                yield output
    
    return wrapper

# 使用
@llm_chat(llm_interface=llm)
async def chat(message: str, history=None):
    """智能助手"""
    pass

# 包装函数（最多保留 10 条历史）
compressed_chat = with_context_compression(chat, max_history_length=10)

# 使用
async for output in compressed_chat("Hello", history=long_history):
    if isinstance(output, ResponseYield):
        print(output.response)
```

#### 5.3 高级 Wrapper：状态持久化

```python
import json
from pathlib import Path
from typing import List, Dict, Optional

def with_state_persistence(chat_func, state_file: str = "chat_state.json"):
    """
    Wrapper：实现状态持久化
    
    自动保存和恢复对话历史
    """
    state_path = Path(state_file)
    
    async def wrapper(message: str, history: List[Dict[str, str]] = None):
        # 恢复状态
        if history is None and state_path.exists():
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                print(f"已恢复历史记录: {len(history)} 条消息")
            except Exception as e:
                print(f"恢复状态失败: {e}")
                history = []
        
        # 代理调用
        final_history = history or []
        async for output in chat_func(message, history):
            if isinstance(output, ResponseYield):
                # 更新历史
                final_history = output.messages
                
                # 保存状态
                try:
                    with open(state_path, 'w', encoding='utf-8') as f:
                        json.dump(final_history, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"保存状态失败: {e}")
            
            yield output
    
    return wrapper

# 使用
@llm_chat(llm_interface=llm)
async def chat(message: str, history=None):
    """智能助手"""
    pass

# 包装函数
persistent_chat = with_state_persistence(chat, "my_chat_state.json")

# 使用（自动保存和恢复）
async for output in persistent_chat("Hello"):
    if isinstance(output, ResponseYield):
        print(output.response)
```

#### 5.4 组合多个 Wrapper

你可以组合多个 wrapper 实现复杂的功能：

```python
# 组合多个 wrapper
@llm_chat(llm_interface=llm)
async def chat(message: str, history=None):
    """智能助手"""
    pass

# 组合：监控 + 压缩 + 持久化
enhanced_chat = with_state_persistence(
    with_context_compression(
        with_state_monitoring(chat),
        max_history_length=10
    ),
    state_file="chat_state.json"
)

# 使用
async for output in enhanced_chat("Hello"):
    if isinstance(output, ResponseYield):
        print(output.response)
```

### 6. 向后兼容

如果需要同时支持事件流和传统模式，可以使用 `responses_only`：

```python
# 启用事件流但只处理响应
@llm_chat(llm_interface=llm)
async def chat(message: str, history=None):
    pass

# 使用 responses_only 获得传统 API
async for response, history in responses_only(chat("Hello")):
    print(response)
```

### 7. 中断与取消（AbortSignal）

事件流模式下可以通过 `AbortSignal` 中断当前回合，停止流式输出并取消正在执行的工具调用。
中断后仍会收到 `ReactEndEvent`（`extra.aborted == true`）。

```python
from SimpleLLMFunc.hooks import AbortSignal, ABORT_SIGNAL_PARAM

abort_signal = AbortSignal()

async for output in chat(
    "请列出 30 个要点",
    history=[],
    **{ABORT_SIGNAL_PARAM: abort_signal},
):
    ...

abort_signal.abort("user_interrupt")
```

更完整的说明请参阅 [中断与取消](abort.md)。

## 常见问题

### Q: 事件流会影响性能吗？

A: 事件流本身是轻量级的，不会显著影响性能。事件处理是异步的，不会阻塞主流程。如需降低 UI/日志开销，可减少事件消费者侧处理。

### Q: 如何只监听特定类型的事件？

A: 使用 `filter_events` 函数：

```python
async for event in filter_events(
    chat("Hello"),
    event_types={ReActEventType.TOOL_CALL_END}
):
    # 只处理工具调用结束事件
    pass
```

### Q: 事件流和流式响应有什么区别？

A:

- **流式响应** (`stream=True`)：控制 LLM 是否以流式方式返回响应
- **事件流**：`llm_chat` 调用和 `llm_function.stream(...)` 产生 `ReactOutput`，用于观察执行过程

两者可以独立使用，也可以同时启用。

### Q: 可以在事件处理中修改执行流程吗？

A: 事件是只读的观察点，不能直接修改执行流程。但你可以通过 wrapper 函数在调用前后修改 `history` 参数，从而间接影响 Agent 的行为。例如，可以实现 Context Compression、状态过滤等功能。

### Q: 如何修改 Agent 的状态？

A: 由于 Agent 是无状态的，所有状态都通过 `history` 参数传递。你可以：

1. **在调用前修改**：通过 wrapper 函数修改传入的 `history` 参数
2. **在调用后修改**：通过 wrapper 函数修改返回的 `history`（在 `ResponseYield` 中）
3. **通过事件监控**：通过事件获取状态信息，但修改需要通过 wrapper 实现

详见 [状态管理和修改（Wrapper 模式）](#state-management-wrapper-mode) 章节。

### Q: llm_function 支持事件流吗？

A: 是的！`@llm_function` 也执行 ReAct 循环，从 v0.5.0+ 开始完全支持事件流。使用 `your_function.stream(...)` 时会返回一个生成器，yield 事件和最终响应。

**重要区别**：`@llm_function` 的 `ResponseYield.response` 包含的是**解析后的结果**（`str`、Pydantic 对象等），而不是原始的 `ChatCompletion` 对象。这符合 `llm_function` 的设计理念：将 LLM 响应转换为指定的返回类型。

示例：

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.hooks.stream import is_response_yield

@llm_function(llm_interface=llm)
async def summarize(text: str) -> str:
    """生成摘要"""
    pass

async for output in summarize(text="..."):
    if is_response_yield(output):
        # output.response 是解析后的字符串，不是 ChatCompletion
        print(output.response)  # 直接是摘要文本
```

参考示例：

- [Token 用量监控](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_token_usage.py) - 字符串返回类型
- [Pydantic 结构化输出](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_event_pydantic.py) - Pydantic 对象返回类型

### Q: 事件中的 `trace_id` 有什么用？

A: `trace_id` 用于追踪整个 ReAct 循环的执行过程，可以用于日志关联、调试和性能分析。

## 相关资源

- [llm_chat 装饰器文档](llm_chat.md) - 了解 `llm_chat` 的基础用法
- [事件流 Chatbot 示例](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/event_stream_chatbot.py) - 完整示例代码
- [工具系统文档](tool.md) - 了解工具调用的详细信息
