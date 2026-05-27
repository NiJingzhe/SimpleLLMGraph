![SimpleLLMFunc](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/img/repocover_1_1.png?raw=true)

<center>
<h2 style="font-size:2em;">LLM as Function, Prompt as Code, Context-Centric</h2>
</center>

<div align="center">
  <a href="README.md" style="font-size: 1.2em; font-weight: bold; color: #007acc; text-decoration: none; border: 2px solid #007acc; padding: 8px 16px; border-radius: 6px; background: linear-gradient(135deg, #f0f8ff, #e6f3ff);">
    English README
  </a>
</div>

----

![Github Stars](https://img.shields.io/github/stars/NiJingzhe/SimpleLLMFunc.svg?style=social)
![Github Forks](https://img.shields.io/github/forks/NiJingzhe/SimpleLLMFunc.svg?style=social)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/SimpleLLMFunc)](https://pypi.org/project/SimpleLLMFunc/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/NiJingzhe/SimpleLLMFunc/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NiJingzhe/SimpleLLMFunc/pulls)

### 更新说明 (0.8.3)

**依赖约束清理**：runtime、development 和 build-system 包依赖现在统一使用无上限的 `>=...` 约束，不再使用 Poetry caret、固定版本或显式包版本上限。Poetry lockfile 已按新约束重新生成，未继续支持的 `uv.lock` 也已从发布流程中移除。详情见 **[更新日志](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/CHANGELOG.md)**。

### 文档

> [中文文档](https://simplellmfunc.cn/) | [English Docs](https://simplellmfunc.cn/en)

-----

## 设计哲学

| 原则 | 含义 |
|------|------|
| **LLM is Function** | LLM 调用与 Python 函数调用无区别：签名、类型标注、返回值 |
| **Prompt as Code** | DocString 即系统提示，代码与 Prompt 永不分离 |
| **Context-Centric** | 每次 LLM 请求由 invocation 配置、对话记录/history 和内部运行时补丁编译而来 |

## 快速开始

### 安装

```bash
pip install SimpleLLMFunc
```

### 30 行构建通用 Agent

只需这些代码——一个具备持久 REPL、文件工具、自反记忆、上下文压缩和并行 Fork 的编码 Agent：

```python
from SimpleLLMFunc import llm_chat, OpenAICompatible, tui
from SimpleLLMFunc.builtin import PyRepl, FileToolset

llm = OpenAICompatible.load_from_json_file("provider.json")["openrouter"]["gpt-5.4"]
repl = PyRepl()
file_tools = FileToolset("./sandbox").toolset

@tui
@llm_chat(
    llm_interface=llm,
    toolkit=[*repl.toolset, *file_tools],
    stream=True,
    self_reference_key="agent_main",
)
async def agent(message: str, history=None):
    """你是一个实用的本地编码 Agent。

    ## 规则
    - 编辑前先阅读文件，优先小范围本地修改。
    - 使用 execute_code 执行 Python，使用文件工具进行 read/grep/sed。
    - 里程碑完成后，通过以下方式压缩上下文：
      runtime.selfref.context.compact(...)
    - 并行子任务使用 fork：
      runtime.selfref.fork.spawn(...)
      然后用 runtime.selfref.fork.gather_all(...) 收集结果
    """

if __name__ == "__main__":
    agent()  # 启动交互式 TUI
```

运行：

```bash
python agent.py
```

Agent 自动获得终端 UI——流式 Markdown 渲染、工具调用面板、Token 统计、Fork 生命周期可视化，无需额外代码。

完整生产版本见 `examples/tui_general_agent_example.py`，包含环境信息注入、工作区配置和调试日志。

### 更简单的起步 — LLM 即类型安全函数

如果你只需要一个带类型安全返回值的 LLM 函数：

```python
import asyncio
from SimpleLLMFunc import llm_function, OpenAICompatible

llm = OpenAICompatible.load_from_json_file("provider.json")["your_provider"]["model"]

@llm_function(llm_interface=llm)
async def classify_sentiment(text: str) -> str:
    """
    分析文本的情感倾向。

    Args:
        text: 要分析的文本

    Returns:
        情感分类，可为 'positive', 'negative', 或 'neutral'
    """
    pass  # Prompt as Code!

async def main():
    result = await classify_sentiment("这个产品太棒了！")
    print(f"情感分类: {result}")

asyncio.run(main())
```

### 初始化配置

1. 复制配置模板：

```bash
cp env_template .env
```

2. 在 `.env` 中配置 API 密钥。可选配置 `LOG_DIR`、`LANGFUSE_BASE_URL`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_PUBLIC_KEY`。

3. 查看 `examples/provider_template.json` 了解多供应商配置。

## 架构

SimpleLLMFunc 按五层组织，每层有严格边界：

```
┌──────────────────────────────────────────────────────────────────────────┐
│  L1. 装饰器层                @llm_function / @llm_chat / @tool          │
│      Python 调用 → InvocationSpec + PromptContract + TranscriptSeed     │
├──────────────────────────────────────────────────────────────────────────┤
│  L2. 编译边界                compile_pipeline.py（唯一入口）             │
│      应用 Mutations → 组装 system prompt → 渲染 LLM messages            │
├──────────────────────────────────────────────────────────────────────────┤
│  L3. ReAct 运行时            react_loop.py（Event-Only 核心）           │
│      LLM 调用 → 工具批量执行 → Mutation 收集 → 循环                     │
├──────────────────────────────────────────────────────────────────────────┤
│  L4. 接口层                  OpenAICompatible / OpenAIResponsesAPI      │
│      供应商适配、Key 池、令牌桶限流                                      │
├──────────────────────────────────────────────────────────────────────────┤
│  L5. 基础设施                hooks / logger / observability / type      │
│      事件流、Langfuse span、结构化日志、多模态类型                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### 核心数据流：Mutation 驱动的上下文演进

核心规则：**上下文变更通过编译时 Mutation 应用来完成**，工具和 SelfRef 不能直接改写运行时上下文。

```
     ┌──────────┐   ┌──────────────────────┐
     │ LLM 调用 │──►│AssistantMessageMutation│
     └──────────┘   └──────────┬───────────┘
     ┌──────────┐   ┌──────────────────────┐│
     │ 工具执行 │──►│  ToolResultMutation   ││
     └──────────┘   └──────────┬───────────┘│
     ┌──────────┐   ┌──────────────────────┐│   ┌───────────────┐
     │SelfRef   │──►│ ExperienceRemember /  ││──►│compile_context│──► 编译后上下文
     │ Hooks    │   │ ContextSummary        ││   │ apply_mutations│   （模型可见）
     └──────────┘   └──────────────────────┘│   └───────────────┘
     ┌──────────┐   ┌──────────────────────┐│
     │ 中止信号 │──►│Truncated / Cancelled  ││
     └──────────┘   └──────────────────────┘│
                                         所有 mutations
```

9 种 Mutation 类型覆盖所有上下文变更：助手消息、工具结果、多模态输出、整体替换、上下文压缩、经验记忆/遗忘、截断和取消。

### SelfRef：元上下文编辑

SelfRef 让 Agent 在运行时**读写并编辑自身上下文**，同时遵守 Mutation 边界：

| 操作 | 作用 | 产生的 Mutation |
|------|------|----------------|
| **Remember** | 添加跨轮次持久经验 | `ExperienceRememberMutation` |
| **Forget** | 按 ID 删除经验 | `ExperienceForgetMutation` |
| **Compact** | 用结构化摘要替换工作上下文 | `ContextSummaryMutation` |
| **Fork** | 派生继承上下文快照的子 Agent | （子 Agent 独立运行） |

所有变更在下一个 Compile 边界生效——SelfRef 不能绕过 Compile 直接修改运行时上下文。

## 项目地图

```
SimpleLLMFunc/
├── llm_decorator/             # L1: 装饰器层
│   ├── llm_function_decorator.py  # @llm_function — 无状态 LLM → 类型化结果
│   ├── llm_chat_decorator.py      # @llm_chat public facade / LLMChat callable
│   ├── chat_call_context.py       # 绑定参数/template/runtime 调用上下文
│   ├── chat_selfref.py            # SelfRef 绑定/收尾辅助逻辑
│   ├── chat_toolkit.py            # Runtime toolkit 与 fork toolkit 辅助逻辑
│   ├── chat_types.py              # 共享装饰器常量/类型
│   ├── invocation_spec.py         # InvocationSpec / PromptContract / TranscriptSeed
│   ├── invocation_builder.py      # function/chat 模式的 Spec 构建
│   ├── prompt_contract.py         # Prompt 模板 + XML Schema 生成
│   ├── signature.py               # 签名绑定 + trace_id + 日志上下文
│   └── utils/tools.py             # Tool 处理 + 规格收集
│
├── base/                      # L2+L3: 编译边界 + ReAct 运行时
│   ├── compile_pipeline.py        # 唯一编译入口：reduce + convert
│   ├── context_compile.py         # Mutation 应用引擎
│   ├── llm_input_render.py        # 临时 system prompt 渲染
│   ├── react_loop.py              # ReAct 主循环（Event-Only）
│   ├── llm_call.py                # 单次 LLM 调用执行
│   ├── tool_scheduler.py          # 并发工具调度
│   ├── post_process.py            # Response → 类型化结果（XML → Pydantic）
│   ├── types/                     # 核心类型契约（全部为 dataclass）
│   │   ├── source.py              # CompileSource / DataFromAgentConfig / DataFromSelfRef
│   │   ├── context.py             # ContextState / CompiledContext
│   │   ├── compile.py             # ReducedTurnContext / CompiledTurnContext
│   │   ├── mutation.py            # ContextMutation（9 种变体联合类型）
│   │   ├── react.py               # ReactLoopState
│   │   ├── llm.py                 # SingleLLMCallResult
│   │   └── scheduler.py           # ToolSchedulerResult
│   ├── messages/                  # 消息构建 / 提取 / 验证
│   ├── tool_call/                 # 工具调用提取 / 执行 / 流式 / 验证
│   └── type_resolve/              # 类型描述 + XML 往返
│
├── tool/                      # @tool 装饰器 + Tool 基类
├── builtin/                   # 内置工具
│   ├── pyrepl.py                  # PyRepl public facade
│   ├── pyrepl_execution.py        # execute/reset 编排
│   ├── pyrepl_worker_client.py    # 子进程 + queue 生命周期
│   ├── pyrepl_worker_mixin.py     # facade 兼容 worker wrapper
│   ├── pyrepl_primitive_host.py   # runtime primitive host / backend bridge
│   ├── pyrepl_tools.py            # execute_code/reset_repl tool factory
│   ├── pyrepl_audit.py            # audit log writer
│   ├── pyrepl_input_bridge.py     # process-wide input bridge
│   ├── pyrepl_input_mixin.py      # PyRepl input submission API
│   └── file_tools.py              # workspace-scoped 文件工具
│
├── runtime/                   # Runtime Primitive 体系
│   ├── primitives.py              # PrimitiveRegistry / PrimitivePack / @primitive()
│   ├── worker_proxy.py            # WorkerRuntimeProxy（runtime.selfref.*）
│   └── selfref/
│       ├── state.py               # SelfReference public facade
│       ├── store.py               # durable history/source store
│       ├── active_turn.py         # active memory/fork/toolkit/template contextvars
│       ├── mutations.py           # pending compaction/context mutation queues
│       ├── memory_api.py          # self_reference.memory[...] proxy/handle
│       ├── context_memory.py      # context memory、experiences、compaction、direct edits
│       ├── agent_binding.py       # 绑定 recursive agent callable 状态
│       ├── fork_manager.py        # fork/spawn/gather 生命周期
│       ├── fork_utils.py          # fork helper functions/constants
│       ├── session.py             # SelfRefSession（调用级插件）
│       ├── context_ops.py         # 上下文 解析 / 构建 / 规范化
│       └── primitives.py          # selfref runtime primitives
│
├── hooks/                     # 事件流系统
│   ├── events.py                  # 14 种 ReActEvent 子类型
│   ├── stream.py                  # ReactOutput / ResponseYield / EventYield
│   ├── event_bus.py               # 事件入口 + origin 元数据
│   └── event_emitter.py           # 工具自定义事件发射器
│
├── interface/                 # L4: LLM 接口层
│   ├── llm_interface.py           # 抽象基类
│   ├── openai_compatible.py       # OpenAI Compatible 适配器
│   ├── openai_responses_compatible.py  # Responses API 适配器
│   ├── key_pool.py                # API Key 轮转池
│   └── token_bucket.py            # 令牌桶限流
│
├── logger/                    # 结构化日志 + trace_id
├── observability/             # Langfuse 集成
├── type/                      # 多模态类型（Text / ImgUrl / ImgPath）
└── utils/tui/                 # Textual TUI 集成
```

## 详细指南

### @llm_function — 无状态类型化 LLM 调用

直接返回 Pydantic 模型、基础类型、dict 或 list：

```python
from typing import List
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    rating: int = Field(..., description="产品评分，1-5分")
    pros: List[str] = Field(..., description="产品优点")
    cons: List[str] = Field(..., description="产品缺点")
    summary: str = Field(..., description="评价总结")

@llm_function(llm_interface=llm)
async def analyze_review(product_name: str, review_text: str) -> ProductReview:
    """你是一个专业的产品评测专家，客观分析产品评论并生成结构化报告。

    Args:
        product_name: 产品名称
        review_text: 用户评论内容

    Returns:
        结构化的 ProductReview 对象
    """
    pass

result = await analyze_review("XYZ无线耳机", "音质很好但连接不稳定...")
print(result.rating)   # 4
print(result.pros)     # ["音质出色", ...]
```

### @llm_chat — 对话型 Agent

多轮历史、流式响应、工具调用、SelfRef 集成：

```python
@llm_chat(llm_interface=llm, toolkit=[search_tool, calculator], stream=True)
async def agent(user_message: str, history=None):
    """智能助手，可以搜索信息和计算"""
    pass

async for response, updated_history in agent("你好", []):
    print(response)
```

### @tui — @llm_chat 的终端 UI

开箱即用的 Textual TUI——流式 Markdown 渲染、工具调用面板、Token 统计、Fork 可视化：

```python
from SimpleLLMFunc import llm_chat, tui

@tui(custom_event_hook=[...])
@llm_chat(llm_interface=llm, stream=True)
async def agent(message: str, history=None):
    """你的 Agent 提示词"""

if __name__ == "__main__":
    agent()  # 启动 TUI
```

完整示例见 `examples/tui_chat_example.py`。每个 `EventYield` 携带 `origin` 元数据，支持 Fork 感知的事件路由。

### @tool — 将函数注册为 LLM 工具

```python
from SimpleLLMFunc.tool import tool
from SimpleLLMFunc.type import ImgPath

@tool(name="generate_chart", description="根据数据生成图表")
async def generate_chart(data: str, chart_type: str = "bar") -> ImgPath:
    """根据提供的数据生成图表

    Args:
        data: CSV 格式数据
        chart_type: 图表类型，默认柱状图

    Returns:
        生成的图表文件路径
    """
    chart_path = "./generated_chart.png"
    # ... 图表生成逻辑
    return ImgPath(chart_path)
```

`@tool` 可以与 `@llm_function` 叠加在同一个函数上。

工具也可以返回多张图片并附带可选文本，例如 `tuple[str, list[ImgPath | ImgUrl]]`。当 PyRepl 的 `execute_code` 代码通过 `display(Image(...))` 或图片型最后表达式输出图片时，也会走这条多模态返回链路。

### 多模态支持

```python
from SimpleLLMFunc.type import UserChatMessage, ImgPath, ImgUrl, Text

@llm_function(llm_interface=llm)
async def analyze_image(
    description: Text,        # 文本描述
    web_image: ImgUrl,        # 网络图片 URL 或 data: URL
    local_image: ImgPath      # 本地图片路径，会编码为 data URL
) -> str:
    """根据描述分析图像"""
    pass

result = await analyze_image(
    description=Text("请描述这两张图片的区别"),
    web_image=ImgUrl("https://example.com/image.jpg"),
    local_image=ImgPath("./reference.jpg")
)

@llm_chat(llm_interface=llm)
async def vision_agent(message: UserChatMessage, history=None):
    """回答用户多模态消息中的问题。"""
    pass

async for output in vision_agent(
    UserChatMessage.multimodal(
        "这张图里有什么？",
        ImgUrl("https://example.com/cat.jpg", detail="high"),
    ),
    history=[],
):
    ...
```

`llm_function` 保持普通 Python 函数风格：通过显式声明 `ImgUrl` / `ImgPath` / 对应列表参数传入图片。`llm_chat` 则接受显式的 OpenAI-compatible `UserChatMessage`，让 Agent 可以在同一条 user message 中接收文本与图片内容。

### 工具调用上限默认值

`@llm_function` 和 `@llm_chat` 默认 `max_tool_calls=None`（无上限）。传入整数如 `max_tool_calls=8` 设置安全上限：

```python
@llm_chat(llm_interface=llm, stream=True, max_tool_calls=12)
async def cautious_agent(message: str, history=None):
    """显式设置安全上限的对话 Agent"""
    pass
```

### 装饰器参数

```python
@llm_function(
    llm_interface=llm_interface,
    toolkit=[tool1, tool2],
    retry_on_exception=True,
    timeout=60
)
async def my_function(param: str) -> str:
    """支持 {language} 的 {style} 分析"""
    pass

result = await my_function(
    "输入内容",
    _template_params={"language": "中文", "style": "专业"},
)
```

`_template_params` 在**调用时**传入，仅用于对 DocString 执行 `str.format` 替换。该参数在签名绑定前被移除，不属于 LLM 输入。

### LLM 供应商接口

**支持的供应商：**

- OpenAI (GPT-4 等)
- Deepseek
- Anthropic Claude
- 火山引擎 Ark
- 百度千帆
- 本地 LLM (Ollama, vLLM 等)
- 任何兼容 OpenAI API 的服务
- OpenAI Responses API（`OpenAIResponsesCompatible`）

```python
from SimpleLLMFunc import APIKeyPool, OpenAICompatible, OpenAIResponsesCompatible

# 从 JSON 配置文件加载
provider = OpenAICompatible.load_from_json_file("provider.json")
llm = provider["deepseek"]["v3-turbo"]

# 直接创建
llm = OpenAICompatible(
    api_key_pool=APIKeyPool(["sk-xxx"], provider_id="deepseek-chat"),
    base_url="https://api.deepseek.com/v1",
    model_name="deepseek-chat",
)

# Responses API
responses_llm = OpenAIResponsesCompatible(
    api_key_pool=APIKeyPool(["sk-xxx"], provider_id="openrouter-gpt-5.4"),
    base_url="https://openrouter.ai/api/v1",
    model_name="gpt-5.4",
)
```

Responses API 路径下仍按普通 docstring 和 history 编写。system prompt 到 `instructions` 的映射及 `reasoning={...}` 的透传由 adapter 负责。

#### provider.json

```json
{
    "deepseek": [{
        "model_name": "deepseek-v3.2",
        "api_keys": ["sk-key-1", "sk-key-2"],
        "base_url": "https://api.deepseek.com/v1",
        "max_retries": 5,
        "retry_delay": 1.0,
        "rate_limit_capacity": 10,
        "rate_limit_refill_rate": 1.0
    }]
}
```

### 日志与可观测性

| 特性 | 说明 |
|------|------|
| **Trace ID 追踪** | 每次调用自动生成唯一 trace_id，关联所有相关日志 |
| **结构化日志** | 多级别（DEBUG–CRITICAL），彩色控制台输出 |
| **上下文传播** | Async-safe contextvars，trace_id 自动关联 |
| **文件持久化** | 自动轮换和归档 |
| **Langfuse 集成** | 可视化 LLM 调用链路，工具/LLM 嵌套 span |

```python
from SimpleLLMFunc.logger import app_log, push_error, log_context

app_log("开始处理请求", trace_id="request_123")

with log_context(trace_id="task_456", function_name="analyze_text"):
    app_log("开始分析")       # 自动继承 trace_id
    push_error("分析失败")    # 自动继承 trace_id
```

### 内置工具

**PyRepl** — 持久 IPython 子进程，变量持久化、执行超时、事件流输出、Primitive Pack 支持：

```python
from SimpleLLMFunc.builtin import PyRepl
repl = PyRepl(working_directory="./workspace")
tools = repl.toolset  # [execute_code, reset_repl, list_variables]
```

**FileToolset** — 工作区级文件工具，带过时写入保护：

```python
from SimpleLLMFunc.builtin import FileToolset
file_tools = FileToolset("./workspace").toolset  # [read_file, read_image, grep, sed, echo_into]
```

### API 密钥管理与流量控制

- 多 API Key 小根堆负载均衡
- 令牌桶算法限流
- `provider.json` 中按模型配置

### 异步原生

所有装饰器返回 async 函数，使用 `await` 或 `asyncio.run()` 调用：

```python
# 并发 LLM 调用
results = await asyncio.gather(*[classify_text(t) for t in texts])
```

## 常见使用场景

### 数据处理

```python
@llm_function(llm_interface=llm)
async def extract_entities(text: str) -> Dict[str, List[str]]:
    """从文本中提取命名实体"""
    pass

entities = await extract_entities("张三在北京的Apple公司工作")
# {"person": ["张三"], "location": ["北京"], "organization": ["Apple"]}
```

### 通用 Agent

```python
@llm_chat(llm_interface=llm, toolkit=[*repl.toolset, *file_tools], stream=True, self_reference_key="main")
async def agent(message: str, history=None):
    """具备 REPL、文件工具和自反记忆的编码 Agent"""
    pass
```

### 批量处理

```python
results = await asyncio.gather(*[classify_text(t) for t in texts])
```

### 多模态

```python
@llm_function(llm_interface=llm)
async def analyze_images(local_img: ImgPath, web_img: ImgUrl) -> str:
    """对比分析两张图片"""
    pass
```

## 运行示例

```bash
pip install SimpleLLMFunc
cp env_template .env
# 编辑 .env 填入 API 密钥

python examples/tui_general_agent_example.py    # 完整编码 Agent + TUI
python examples/llm_function_pydantic_example.py # 结构化输出
python examples/event_stream_chatbot.py          # 对话 + 事件流
python examples/parallel_toolcall_example.py     # 并发工具调用
python examples/pyrepl_example.py                # 持久 REPL
python examples/pyrepl_seaborn_multimodal_images.py # PyRepl 图片输出
python examples/response_api_example.py          # Responses API
```

## 导出 Agent Skills

```bash
simplellmfunc-skill usage ~/.config/opencode/skills
simplellmfunc-skill developer ~/.config/opencode/skills
```

加 `--force` 覆盖已有文件。

## 配置

优先级（高→低）：程序配置 → 环境变量 → `.env` 文件。

```bash
# .env
LOG_DIR=./logs
LOG_LEVEL=INFO
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk_xxx
LANGFUSE_SECRET_KEY=sk_xxx
LANGFUSE_EXPORT_ALL_SPANS=true
```

## 贡献指南

- Bug 报告：[GitHub Issues](https://github.com/NiJingzhe/SimpleLLMFunc/issues)
- 功能建议欢迎
- 文档改进欢迎
- 示例代码欢迎

## 更多资源

- [中文文档](https://simplellmfunc.cn/) | [English Docs](https://simplellmfunc.cn/en)
- [更新日志](CHANGELOG.md)
- [GitHub 仓库](https://github.com/NiJingzhe/SimpleLLMFunc)

## Star History

<a href="https://www.star-history.com/#NiJingzhe/SimpleLLMFunc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date" />
 </picture>
</a>

## 引用

```bibtex
@software{ni2025simplellmfunc,
  author = {Jingzhe Ni},
  month = {February},
  title = {{SimpleLLMFunc: A New Approach to Build LLM Applications}},
  url = {https://github.com/NiJingzhe/SimpleLLMFunc},
  version = {0.8.3},
  year = {2026}
}
```

## 许可证

MIT
