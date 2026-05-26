# 示例代码

本章节收集了 SimpleLLMFunc 框架的各种使用示例。这些示例展示了框架的核心功能和最佳实践。

快速入口：可先查看 `examples/README.md`，其中包含按场景整理后的可运行命令（含无需 API Key 的本地示例）。

> ⚠️ **重要提示**：本框架中的所有装饰器（`@llm_function`、`@llm_chat`、`@tool`）均要求被装饰的函数使用 `async def` 定义，并在调用时通过 `await`（或 `asyncio.run`）执行。

## 基础示例

### llm_function 结构化输出（Pydantic）

**文件**: [examples/llm_function_pydantic_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_pydantic_example.py)

这个例子展示了如何使用 `@llm_function` 返回复杂 Pydantic 结构：
- 多层嵌套模型定义
- 自动结构化解析
- 类型安全返回值处理

### 动态模板参数

**文件**: [examples/dynamic_template_demo.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/dynamic_template_demo.py)

演示如何在调用时通过 `_template_params` 动态填充 DocString 模板：
- 一个函数适配多种任务
- 按角色/风格切换 Prompt
- 降低重复函数定义

### llm_function 事件流 + Pydantic

**文件**: [examples/llm_function_event_pydantic.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_event_pydantic.py)

展示 `@llm_function(...).stream(...)` 的事件流与结构化输出协同：
- 捕获 LLM 调用事件
- 观察执行统计
- 处理最终结构化结果

## 高级示例

### 事件流观测示例

**文件**: [examples/event_stream_chatbot.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/event_stream_chatbot.py)

**⭐ 全新功能！** 展示如何使用 SimpleLLMFunc v0.5.0+ 的事件流功能构建功能完整的聊天机器人。

**核心特性**：
- ✨ **实时流式响应**：使用 Rich 库渲染 Markdown 格式的响应
- 🔧 **工具调用可视化**：实时显示工具调用的参数、执行过程和结果
- 📊 **完整执行统计**：Token 使用量、执行耗时、调用次数等详细信息
- 🎯 **事件驱动架构**：在外部函数中处理事件，实现自定义 UI 和逻辑
- 🧭 **Origin 元数据路由**：通过 `output.origin` 区分主链路与 fork 子链路

**关键代码片段**：

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.hooks import (
    ReactOutput, ResponseYield, EventYield,
    ReactStartEvent, LLMChunkArriveEvent, ToolCallStartEvent
)

# 启用事件流
@llm_chat(
    llm_interface=llm,
    toolkit=[calculate, get_weather, search_knowledge],
    stream=True,
)
async def chat(user_message: str, chat_history: List[Dict[str, str]] = None):
    """智能助手"""
    pass

# 在外部处理事件
async for output in chat(user_message="帮我计算 25 * 4 + 18"):
    if isinstance(output, EventYield):
        # 处理事件：LLM 调用、工具调用等
        event = output.event
        origin = output.origin
        if origin.fork_id:
            print(f"fork={origin.fork_id} depth={origin.fork_depth}")
        if isinstance(event, ToolCallStartEvent):
            print(f"工具调用: {event.tool_name}")
            print(f"参数: {event.arguments}")
    
    elif isinstance(output, ResponseYield):
        # 原始响应对象（流式时为 chunk）。
        # 文本渲染建议使用 LLMChunkArriveEvent 的 accumulated_content。
        messages = output.messages
```

**依赖安装**：
```bash
pip install rich
```

**运行示例**：
```bash
python examples/event_stream_chatbot.py
```

**详细文档**：了解更多事件流的使用方法，请参考 [事件流系统](detailed_guide/event_stream.md)。

### 开箱即用终端 TUI

**文件**: [examples/tui_chat_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/tui_chat_example.py)

展示 `@tui` 与 `@llm_chat` 叠加后的终端体验：

- 聊天输入循环（底部输入框）
- 模型流式输出 + Markdown 渲染
- 灰色 reasoning delta（模型支持时）
- 工具调用参数/结果可视化
- 自定义 tool event hook（`custom_event_hook`）
- 示例同时注册 `PyRepl`（`execute_code`）与 `batch_process`，演示内置 hook + 自定义 hook

核心写法：

```python
@tui(custom_event_hook=[my_hook])
@llm_chat(..., stream=True)
async def agent(message: str, history=None):
    ...

if __name__ == "__main__":
    agent()
```

Run:

```bash
poetry run python examples/tui_chat_example.py
```

### Runtime memory primitives (local, no model)

**File**: [examples/runtime_primitives_basic_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/runtime_primitives_basic_example.py)

Shows how to use runtime primitives without any LLM provider:

- Start `PyRepl` with its builtin `selfref` backend and access it through `repl.get_runtime_backend("selfref")`
- Declare one custom `PrimitivePack` (`constants.get`) as extension example
- Inspect context and queue compaction through `runtime.selfref.context.*`
- Persist durable experiences into system context via `runtime.selfref.context.remember(...)`

Run:

```bash
poetry run python examples/runtime_primitives_basic_example.py
```

### General TUI Agent (selfref + file tools)

**File**: [examples/tui_general_agent_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/tui_general_agent_example.py)

Recommended single entry for a TUI-first agent workflow:

- One agent uses both `runtime.selfref.context.*` and `runtime.selfref.fork.*`
- `FileToolset` provides read/grep/sed/echo tools scoped to the workspace
- `llm_chat` injects runtime primitive guidance through tool-owned best practices (when `PyRepl` is mounted)
- Forked contexts inherit parent memory snapshot from the same selfref key
- Built-in lifecycle stream events (`selfref_fork_start/spawned/end/error`, `selfref_fork_stream_*`) are rendered in TUI
- Workspace is scoped to `./sandbox` under the project root

Run:

```bash
poetry run python examples/tui_general_agent_example.py
```

### Agent as a tool (stacked decorators)

**文件**: [examples/agent_as_tool_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/agent_as_tool_example.py)

展示如何把一个 LLM 驱动的 specialist 暴露成另一个 agent 的工具：

- `@tool` 外层 + `@llm_function` 内层，将 child agent 暴露进 `toolkit`
- parent `@llm_chat` 作为 supervisor，负责路由和最终答复
- child agent 保持独立 prompt / 职责边界，适合 reviewer / planner / router 组合
- 当前推荐子 agent 使用普通 `await @llm_function` 调用；若要复用 `@llm_chat`，通常需要先手写一层 adapter tool 消费其 `ReactOutput`

Run:

```bash
poetry run python examples/agent_as_tool_example.py
```

### llm_function 事件流与 Token 用量监控

**文件**: [examples/llm_function_token_usage.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_token_usage.py)

展示如何在 `@llm_function` 中使用事件流来实时监控 Token 使用情况。适用于需要精确计量 API 调用成本的场景。

**核心特性**：
- 🔍 **实时 Token 监控**：捕获每次 LLM 调用的 Token 用量
- 💰 **成本追踪**：记录 Prompt、Completion 和总 Token 数
- 📊 **统计汇总**：自动累计总用量
- ⚡ **零工具调用**：简单的单次 LLM 调用示例

**关键代码片段**：

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.hooks.events import LLMCallEndEvent
from SimpleLLMFunc.hooks.stream import is_response_yield

@llm_function(
    llm_interface=llm,
)
async def summarize_text(text: str) -> str:
    """将给定的文本进行简洁的摘要"""
    return ""

# 捕获 Token 用量
async for output in summarize_text(text="..."):
    if is_response_yield(output):
        # 处理响应结果
        print(output.response)
    else:
        event = output.event
        if isinstance(event, LLMCallEndEvent):
            # 获取 Token 用量
            usage = event.usage
            if usage:
                print(f"Prompt Tokens: {usage.prompt_tokens}")
                print(f"Completion Tokens: {usage.completion_tokens}")
                print(f"Total Tokens: {usage.total_tokens}")
```

**运行示例**：
```bash
poetry run python examples/llm_function_token_usage.py
```

### llm_chat 事件流聊天应用

**文件**: [examples/event_stream_chatbot.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/event_stream_chatbot.py)

展示如何使用 `@llm_chat` + 事件流构建完整对话应用：
- 多轮对话历史管理
- 流式响应与事件并行消费
- 工具调用可视化与统计

### 自定义 Tool 事件

**文件**: [examples/custom_tool_event_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/custom_tool_event_example.py)

演示在工具执行期间发射并消费自定义事件：
- `ToolEventEmitter` 进度事件上报
- `CustomEvent` 消费与渲染
- 批处理任务过程可视化

### 并行工具调用

**文件**: [examples/parallel_toolcall_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/parallel_toolcall_example.py)

演示高级的工具调用特性：
- 多个工具的并行执行
- 工具调用的优化和性能
- 大规模工具集的管理

### 多模态内容处理

**文件**: [examples/multi_modality_toolcall.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/multi_modality_toolcall.py)

展示多模态功能的使用：
- 图片 URL (`ImgUrl`) 的处理
- 本地图片路径 (`ImgPath`) 的处理
- 文本和图片的混合输入输出

## 供应商配置示例

### Provider 配置文件

**文件**: [examples/provider.json](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/provider.json)

示范 provider.json 的完整配置结构（提供商 -> 模型配置列表）：
- `model_name` 作为索引键
- API 密钥、重试与限流参数
- 适配任意 OpenAI 兼容服务

### Provider 模板

**文件**: [examples/provider_template.json](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/provider_template.json)

提供可复用的配置模板：
- 多供应商示例与参数说明
- 多密钥负载均衡
- 限流与重试最佳实践

## 按功能分类的示例

### 文本处理
- **结构化提取**: 见 [llm_function_pydantic_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_pydantic_example.py)
- **动态模板**: 见 [dynamic_template_demo.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/dynamic_template_demo.py)
- **事件流监控**: 见 [llm_function_token_usage.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_token_usage.py)

### 工具调用
- **内置 REPL 工具调用**: 见 [pyrepl_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/pyrepl_example.py)
- **PyRepl 图片产物返回**: 见 [pyrepl_seaborn_multimodal_images.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/pyrepl_seaborn_multimodal_images.py)
- **多工具并行调用**: 见 [parallel_toolcall_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/parallel_toolcall_example.py)
- **Agent as a Tool**: 见 [agent_as_tool_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/agent_as_tool_example.py)

### 对话与 Agent
- **事件流聊天**: 见 [event_stream_chatbot.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/event_stream_chatbot.py)
- **自定义工具事件**: 见 [custom_tool_event_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/custom_tool_event_example.py)
- **终端 TUI 聊天**: 见 [tui_chat_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/tui_chat_example.py)
- **事件流观测**: 见 [event_stream_chatbot.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/event_stream_chatbot.py) ⭐ 通过 `ReactOutput` 实时观察 ReAct 循环执行过程

### 多模态处理
- **图片分析**: 见 [multi_modality_toolcall.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/multi_modality_toolcall.py)
- **多图工具返回**: 见 [pyrepl_seaborn_multimodal_images.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/pyrepl_seaborn_multimodal_images.py)
- **混合输入输出**: 见 [multi_modality_toolcall.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/multi_modality_toolcall.py)

## 快速运行示例

### 前置要求
1. 安装 SimpleLLMFunc: `pip install SimpleLLMFunc`
2. 配置 API 密钥（见 [快速开始](quickstart.md)）
3. 创建或编辑 `provider.json` 文件

### 运行方式

```bash
# 进入 examples 目录
cd examples

# 运行基础 LLM 函数示例
python llm_function_pydantic_example.py

# 运行聊天示例
python event_stream_chatbot.py

# 运行并行工具调用示例
python parallel_toolcall_example.py

# 运行多模态示例
python multi_modality_toolcall.py

# 运行事件流 Chatbot 示例（需要先安装 rich: pip install rich）
python event_stream_chatbot.py

# 运行 Textual TUI 示例
python tui_chat_example.py
```

## 完整的 Examples 目录

所有示例代码都位于仓库的 `examples/` 目录中：

**仓库链接**: https://github.com/NiJingzhe/SimpleLLMFunc/tree/master/examples

在该目录中你可以找到：
- 各种装饰器的使用示例
- 不同 LLM 供应商的配置示例
- 最佳实践的参考实现
- 环境变量配置的示例

## 学习路径建议

### 初级用户
1. 阅读 [快速开始](quickstart.md) 文档
2. 运行 `llm_function_pydantic_example.py`
3. 修改示例代码，尝试自己的 Prompt

### 中级用户
1. 学习 [llm_chat 装饰器文档](detailed_guide/llm_chat.md)
2. 运行 `event_stream_chatbot.py`
3. 尝试 `parallel_toolcall_example.py`

### 高级用户
1. 阅读 [LLM 接口层文档](detailed_guide/llm_interface.md)
2. 学习多模态处理：`multi_modality_toolcall.py`
3. 学习事件流处理：`event_stream_chatbot.py` ⭐
4. 自定义 LLM 接口和工具系统

## 常见问题

### Q: 示例代码在哪里？
A: 所有示例代码都在 GitHub 仓库的 `examples/` 目录中。你可以直接查看或下载运行。

### Q: 如何修改示例代码？
A:
1. 克隆仓库：`git clone https://github.com/NiJingzhe/SimpleLLMFunc.git`
2. 编辑 `examples/` 目录中的文件
3. 运行修改后的代码

### Q: 示例是否支持所有 LLM 供应商？
A: 示例代码使用 `provider.json` 配置，支持任何兼容 OpenAI API 的供应商。参考 `provider_template.json` 配置你的供应商。

### Q: 我遇到了问题，该怎么办？
A:
1. 检查 [快速开始](quickstart.md) 中的配置部分
2. 查看详细的 [使用指南](guide.md)
3. 在 GitHub 提交 Issue：https://github.com/NiJingzhe/SimpleLLMFunc/issues

## 贡献新示例

如果你想为项目贡献新的示例代码：

1. Fork 仓库
2. 在 `examples/` 目录中创建新文件
3. 遵循现有示例的代码风格和注释
4. 提交 Pull Request

详细信息见 [贡献指南](contributing.md)。

## 相关资源

- **官方仓库**: https://github.com/NiJingzhe/SimpleLLMFunc
- **完整文档**: Mintlify documentation site
- **发布日志**: https://github.com/NiJingzhe/SimpleLLMFunc/releases
- **问题反馈**: https://github.com/NiJingzhe/SimpleLLMFunc/issues
