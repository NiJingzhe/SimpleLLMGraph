# 项目模块地图

<!-- DOC_SUMMARY_START -->

本文档提供 SimpleLLMFunc 项目的模块路径映射和模块职责说明。帮助 Agent 快速定位模块位置，了解各模块的作用，以便在对应路径下查找详细的 spec 文档。

<!-- DOC_SUMMARY_END -->

<!-- DOC_MAP_START -->

## 文档目录 (Document Map)

- [项目模块地图](#项目模块地图)
  - [文档更新规范](#文档更新规范)
    - [更新时机](#更新时机)
    - [修改规范](#修改规范)
    - [相关文件](#相关文件)
  - [项目结构概览](#项目结构概览)
    - [目录组织原则](#目录组织原则)
    - [模块分类](#模块分类)
  - [核心模块 (SimpleLLMFunc/)](#核心模块-simplellmfunc)
    - [llm_decorator 模块](#llm_decorator-模块)
    - [base 模块](#base-模块)
    - [runtime 模块](#runtime-模块)
    - [builtin 模块](#builtin-模块)
    - [hooks 模块](#hooks-模块)
    - [interface 模块](#interface-模块)
    - [tool 模块](#tool-模块)
    - [type 模块](#type-模块)
    - [logger 模块](#logger-模块)
    - [observability 模块](#observability-模块)
    - [utils 模块](#utils-模块)
  - [根目录文件](#根目录文件)
  - [测试与示例](#测试与示例)
    - [tests 目录](#tests-目录)
    - [examples 目录](#examples-目录)
    - [mintlify_docs 目录](#mintlify_docs-目录)
    - [spec 目录](#spec-目录)

<!-- DOC_MAP_END -->

<!-- DOC_META_START -->

## 文档更新规范

### 更新时机

以下情况需要更新本文档：

- **添加新模块**: 当项目中新增功能模块时，需要在相应章节添加模块信息
- **修改模块结构**: 当模块的路径、作用或技术栈发生变化时，需要同步更新对应模块的描述
- **重构项目结构**: 当项目目录结构发生变化时，需要更新项目结构概览部分
- **模块职责变更**: 当模块的职责或功能发生变化时，需要更新模块的作用说明

### 修改规范

1. **保持格式一致性**:
   - 遵循 `spec/meta.md` 中定义的格式规范
   - 每个模块描述包含：路径、作用、技术栈（如适用）、架构特点（如适用）
   - 保持模块描述的格式统一

2. **更新目录结构**:
   - 添加新模块后，必须同步更新 DOC_MAP
   - 确保所有锚点 ID 使用英文，格式为 `{#id}`
   - 验证所有链接可正确跳转

3. **模块信息完整性**:
   - 每个模块必须包含路径和作用说明
   - 技术栈信息应准确反映模块使用的技术
   - 子模块信息应清晰说明模块的组成部分

4. **与项目结构同步**:
   - 确保文档中的路径与实际项目目录结构一致
   - 模块分类应准确
   - 文件列表应与实际文件保持一致

### 相关文件

- **本文档**: `spec/project-map.md`
- **格式规范**: `spec/meta.md`
- **通用规范**: `spec/overall-spec.md`
- **模块 Spec**: 各模块目录下的 `spec/` 文件夹
<!-- DOC_META_END -->

## 项目结构概览

<!-- SECTION_SUMMARY_START -->

SimpleLLMFunc 是一个轻量级的 LLM/Agent 应用开发框架，采用分层架构设计。所有核心代码位于 `SimpleLLMFunc/` 目录下。

<!-- SECTION_SUMMARY_END -->

<!-- SECTION_TOC_START -->

### 目录组织原则

- [目录组织原则](#目录组织原则)
- [模块分类](#模块分类)
<!-- SECTION_TOC_END -->

### 目录组织原则

项目目录结构遵循以下原则：

```text
SimpleLLMFunc/
├── SimpleLLMFunc/              # 主包
│   ├── llm_decorator/         # 装饰器入口与 InvocationSpec 构建
│   ├── base/                  # 编译边界 + event-only ReAct runtime
│   ├── runtime/               # runtime primitive 与 SelfRef 后端/会话
│   ├── builtin/               # PyRepl、FileToolset 等内置工具
│   ├── hooks/                 # 事件流系统
│   ├── interface/             # LLM Provider 适配器
│   ├── tool/                  # Tool 定义与 @tool 装饰器
│   ├── type/                  # 消息、多模态、工具调用等类型别名
│   ├── logger/                # 日志系统
│   ├── observability/         # Langfuse 可观测性
│   ├── config.py              # 配置
│   └── utils/                 # stdio / TUI 等工具
├── tests/                     # 测试目录
├── examples/                  # 示例目录
├── mintlify_docs/             # 用户文档站点内容
└── spec/                      # Spec 规范目录
```

**路径说明**：

- **SimpleLLMFunc/**: 核心框架代码
- **tests/**: 单元测试和集成测试
- **examples/**: 使用示例
- **mintlify_docs/**: 用户文档站点内容
- **spec/**: Spec 规范文档

### 模块分类

项目模块按职责分为以下几类：

1. **装饰器层**: 提供 @llm_function、@llm_chat、@tool，并把 Python 调用构建为 `InvocationSpec`
2. **编译边界层**: 通过 `compile_pipeline.py` / `context_compile.py` 将 transcript、SelfRef source 与 `ContextMutation` 编译为 LLM 可见消息
3. **ReAct Runtime 层**: `react_loop.py` 的 event-only 主循环，编排 LLM phase、工具批次、hooks 与 finalize
4. **接口层**: LLM Provider 适配器、API key pool、限流
5. **Runtime / Builtin 层**: runtime primitive、SelfRef、PyRepl、FileToolset
6. **基础设施层**: 事件流、日志、Langfuse 可观测性、TUI

## 核心模块 (SimpleLLMFunc/)

<!-- SECTION_SUMMARY_START -->

核心模块位于包目录 `SimpleLLMFunc/` 下，包含框架的所有核心功能实现。当前实现以 `InvocationSpec -> compile_pipeline -> react_loop` 为主路径，模块间通过明确的类型契约与事件流交互。

<!-- SECTION_SUMMARY_END -->

### llm_decorator 模块

**路径**: `SimpleLLMFunc/llm_decorator/`

**作用**: 提供 @llm_function 和 @llm_chat 装饰器，是框架的 Python 调用入口；负责从函数签名、DocString、调用参数、工具与 SelfRef 会话构建 `InvocationSpec`，不直接承担 provider wire format 转换。

**架构特点**:

- `@llm_function` 返回 `LLMFunction` callable instance，普通调用返回 typed result，并通过 `.stream(...)` 暴露事件流
- `@llm_chat` 返回 `LLMChat` callable instance，调用结果是 `AsyncGenerator[ReactOutput, None]`
- DocString 模板、参数绑定、PromptContract、TranscriptSeed 都在 decorator 边界构建
- 正式 ReAct 执行路径进入 `base.react_loop.ReAct_loop(...)`；不存在 `llm_decorator/steps/` 主路径

**子模块**:

- `llm_function_decorator.py`: @llm_function 装饰器与 `LLMFunction`
- `llm_chat_decorator.py`: @llm_chat 装饰器与 `LLMChat`
- `invocation_spec.py`: `InvocationSpec`、`PromptContract`、`TranscriptSeed` 等调用契约
- `invocation_builder.py`: function/chat 的 spec builder
- `prompt_contract.py`: DocString 模板、参数/返回类型描述、初始 transcript 构建
- `signature.py`: 函数签名绑定、template params 提取、trace_id/log context
- `utils/tools.py`: 工具处理、tool prompt specs 与 prompt block 清理

**Spec 位置**: `SimpleLLMFunc/llm_decorator/` 目录下

### base 模块

**路径**: `SimpleLLMFunc/base/`

**作用**: 提供统一编译边界与 event-only ReAct runtime。

**架构特点**:

- `compile_pipeline.py` 是从 `InvocationSpec` 到 LLM input messages 的正式入口
- `context_compile.py` 通过 `ContextMutation` 应用上下文变化；工具和 SelfRef 不应直接改 live ReAct context
- `react_loop.py` 是主循环，始终产出 `ReactOutput`（`ResponseYield` / `EventYield`），不再维护 event / non-event 双模式
- `llm_call.py` 和 `tool_scheduler.py` 分别负责单次 LLM phase 与并发工具批次
- 消息、工具调用、类型解析与响应后处理拆在子目录中

**子模块**:

- `compile_pipeline.py`: `compile_invocation_turn()`、`reduce_turn_context()`、`convert_to_llm_request()`
- `context_compile.py`: `apply_mutations()`、`compile_context()`、`ContextState` / `CompiledContext` 处理
- `react_loop.py`: `run_react_loop()`、`ReAct_loop()`、`execute_single_llm_call()` event-only 主路径
- `llm_call.py`: 单次 LLM 调用、stream chunk 聚合、assistant mutation 生成
- `tool_scheduler.py`: 并发工具执行、工具事件、tool-result mutation 生成
- `react_hooks.py`: ReAct hook 调用与 mutation 收集
- `types/`: CompileSource、ContextState、ContextMutation、ReactLoopState 等核心类型契约
- `messages/`: assistant/tool 消息构建、usage 提取、多模态消息、校验
- `tool_call/`: 工具调用提取、执行、streaming state、校验
- `type_resolve/`: 类型描述、XML round-trip、多模态检测
- `post_process.py`: LLM 原始响应到 typed result 的后处理

**Spec 位置**: `SimpleLLMFunc/base/` 目录下

### runtime 模块

**路径**: `SimpleLLMFunc/runtime/`

**作用**: 提供 runtime primitive 系统与 SelfRef 后端/会话。

**架构特点**:

- Runtime primitive 是 host-registered callable，经 PyRepl worker 中的 `runtime.*` 代理调用，不是原生 LLM tool call
- `SelfReference` 是 durable backend 的 public facade；`SelfRefSession` 是 invocation-scoped hook/plugin
- SelfRef 在 active ReAct turn 内通过 pending intent -> `ContextMutation` 影响上下文

**子模块**:

- `primitives.py`: `PrimitiveRegistry`、`PrimitivePack`、`PrimitiveCallContext`、`@primitive()`
- `worker_proxy.py`: worker 侧 `runtime.namespace.name(...)` 动态代理
- `selfref/state.py`: SelfReference public facade，保留兼容 API 并编排各子组件
- `selfref/store.py`: durable history/source store
- `selfref/active_turn.py`: active memory key、fork context、runtime toolkit、template params、active ReAct state contextvars
- `selfref/mutations.py`: pending compaction/context mutation/destructive mutation queues
- `selfref/memory_api.py`: `self_reference.memory[...]` handle/proxy API
- `selfref/context_memory.py`: context message compile/snapshot、experience CRUD、compaction commit、direct memory editing API
- `selfref/agent_binding.py`: recursive self-fork agent callable binding
- `selfref/fork_manager.py`: fork/spawn/gather、fork task lifecycle、fork event forwarding/result materialization
- `selfref/fork_utils.py`: fork helper pure-ish functions and compatibility constants
- `selfref/session.py`: SelfRefSession ReAct hook lifecycle 与 mutation 收集
- `selfref/context_ops.py`: SelfRef context parse/build/canonicalize 纯函数
- `selfref/primitives.py`: 内置 selfref runtime primitives

### builtin 模块

**路径**: `SimpleLLMFunc/builtin/`

**作用**: 提供内置 Agent 工具与 runtime host。

**子模块**:

- `pyrepl.py`: PyRepl public facade
- `pyrepl_execution.py`: execute/reset worker protocol orchestration
- `pyrepl_worker_mixin.py`: PyRepl facade 兼容 worker lifecycle wrapper
- `pyrepl_worker_client.py`: subprocess / multiprocessing queue lifecycle
- `pyrepl_primitive_host.py`: runtime backend、primitive registry、primitive RPC 与 fork clone 集成
- `pyrepl_tools.py`: `execute_code` / `reset_repl` tool factory 与输出格式化
- `pyrepl_audit.py`: audit log
- `pyrepl_input_bridge.py`: process-wide input() request/reply bridge
- `pyrepl_input_mixin.py`: PyRepl input submission API
- `pyrepl_worker.py`: PyRepl 子进程 worker，负责 fd/stdout/stderr 捕获、执行命名空间与 runtime 代理
- `file_tools.py`: workspace-scoped 文件工具，带 stale-write protection

### hooks 模块

**路径**: `SimpleLLMFunc/hooks/`

**作用**: 提供事件流系统，支持实时观察 LLM 调用过程

**架构特点**:

- 丰富的事件类型（LLM 调用、工具调用、ReAct 循环等）
- 支持自定义事件发射
- 事件流异步 yield，支持实时处理

**子模块**:

- `events.py`: 事件类型定义（ReActEvent, CustomEvent 等）
- `stream.py`: 事件流处理（EventYield, ResponseYield 等）
- `event_emitter.py`: 自定义事件发射器（ToolEventEmitter）

**Spec 位置**: `SimpleLLMFunc/hooks/` 目录下

### interface 模块

**路径**: `SimpleLLMFunc/interface/`

**作用**: 提供 LLM 接口抽象，支持多种 LLM 提供商

**架构特点**:

- 统一的 LLM 接口抽象
- 支持 OpenAI 兼容的所有 API
- 内置 API Key 池管理和限流

**子模块**:

- `llm_interface.py`: LLM 接口基类
- `openai_compatible.py`: OpenAI 兼容接口实现
- `key_pool.py`: API Key 池管理
- `token_bucket.py`: 令牌桶限流实现

### tool 模块

**路径**: `SimpleLLMFunc/tool/`

**作用**: 提供 @tool 装饰器，用于定义可被 LLM 调用的工具

**架构特点**:

- 使用装饰器模式定义工具
- 自动提取函数签名生成工具 schema
- 支持多模态参数

**子模块**:

- `tool.py`: Tool 类和 @tool 装饰器实现

### type 模块

**路径**: `SimpleLLMFunc/type/`

**作用**: 定义框架内部使用的类型

**子模块**:

- `message.py`: 消息类型与 OpenAI-compatible content part 类型
- `chat_input.py`: `UserChatMessage` canonical chat user input object
- `tool_call.py`: 工具调用类型
- `llm.py`: LLM 相关类型
- `multimodal.py`: 多模态基础类型（`Text` / `ImgUrl` / `ImgPath`）
- `hooks.py`: 事件相关类型

### logger 模块

**路径**: `SimpleLLMFunc/logger/`

**作用**: 提供统一的日志系统

**架构特点**:

- 支持多种日志级别
- 日志上下文管理（trace_id）
- 支持文件和控制台输出

**子模块**:

- `core.py`: 日志核心实现
- `logger.py`: 日志工具函数
- `context_manager.py`: 日志上下文管理
- `logger_config.py`: 日志配置
- `formatters.py`: 日志格式化器

### observability 模块

**路径**: `SimpleLLMFunc/observability/`

**作用**: 提供可观测性支持，如 Langfuse 集成

**子模块**:

- `langfuse_client.py`: Langfuse 客户端
- `langfuse_config.py`: Langfuse 配置

### utils 模块

**路径**: `SimpleLLMFunc/utils/`

**作用**: 提供框架外围工具，包括 stdio 装饰器与 Textual TUI 集成。

**子模块**:

- `stdio/`: 标准输入输出相关装饰器
- `tui/`: Textual TUI app、widgets、formatters、tool cards 与 `@tui` 装饰器

## 根目录文件

**路径**: `SimpleLLMFunc/`

**文件列表**:

- `config.py`: 框架配置管理
- `utils/`: 通用工具函数与 TUI 组件
- `__init__.py`: 包入口文件

## 测试与示例

### tests 目录

**路径**: `tests/`

**作用**: 单元测试和集成测试

**子目录**:

- `test_hooks/`: hooks 模块测试
- `test_base/`: base 模块测试
- `test_llm_decorator/`: 装饰器测试
- `conftest.py`: pytest 配置和 fixtures

### examples 目录

**路径**: `examples/`

**作用**: 使用示例和演示代码

**文件列表**:

- `llm_function_*.py`: llm_function 使用示例
- `llm_function_event_*.py`: 事件流使用示例
- `event_stream_*.py`: 事件流示例
- `multi_modality_*.py`: 多模态示例
- `parallel_toolcall_*.py`: 并行工具调用示例

### mintlify_docs 目录

**路径**: `mintlify_docs/`

**作用**: 用户文档站点内容（Mintlify），包含英文与中文页面。`docs/` 目录不是当前主要文档源。

### spec 目录

**路径**: `spec/`

**作用**: Spec 规范文档

**文件列表**:

- `meta.md`: 文档格式规范
- `project-map.md`: 项目模块地图
- `overall-spec.md`: 最佳实践规范
