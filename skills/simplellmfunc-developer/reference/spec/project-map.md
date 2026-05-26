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
    - [hooks 模块](#hooks-模块)
    - [interface 模块](#interface-模块)
    - [tool 模块](#tool-模块)
    - [type 模块](#type-模块)
    - [logger 模块](#logger-模块)
    - [observability 模块](#observability-模块)
  - [根目录文件](#根目录文件)
  - [测试与示例](#测试与示例)
    - [tests 目录](#tests-目录)
    - [examples 目录](#examples-目录)
    - [docs 目录](#docs-目录)
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
│   ├── llm_decorator/         # 装饰器模块
│   ├── base/                  # 核心实现
│   ├── builtin/               # 内置工具
│   ├── hooks/                 # 事件系统
│   ├── interface/             # LLM 接口
│   ├── tool/                  # 工具定义
│   ├── type/                  # 类型定义
│   ├── logger/                # 日志系统
│   ├── observability/         # 可观测性
│   ├── config.py              # 配置
│   └── utils/                 # 工具函数与 TUI
├── tests/                     # 测试目录
├── examples/                  # 示例目录
├── docs/                      # 文档目录
└── spec/                      # Spec 规范目录
```

**路径说明**：

- **SimpleLLMFunc/**: 核心框架代码
- **tests/**: 单元测试和集成测试
- **examples/**: 使用示例
- **docs/**: 项目文档
- **spec/**: Spec 规范文档

### 模块分类

项目模块按职责分为以下几类：

1. **装饰器层**: 提供 @llm_function, @llm_chat, @tool 装饰器
2. **执行层**: phase-based ReAct 循环实现，工具调用编排与统一 finalize 收口
3. **运行时状态层**: runtime primitive、selfref state、纯 context transform
4. **接口层**: LLM 接口抽象，支持多提供商
5. **基础设施层**: 日志、事件流、可观测性

## 核心模块 (SimpleLLMFunc/)

<!-- SECTION_SUMMARY_START -->

核心模块位于 `SimpleLLMFunc/SimpleLLMFunc/` 目录，包含框架的所有核心功能实现。每个模块职责清晰，模块间通过接口进行交互。

<!-- SECTION_SUMMARY_END -->

### llm_decorator 模块

**路径**: `SimpleLLMFunc/SimpleLLMFunc/llm_decorator/`

**作用**: 提供 @llm_function 和 @llm_chat 装饰器，是框架的核心入口

**架构特点**:

- 使用装饰器模式包装普通函数为 LLM 调用
- 支持单次调用（llm_function）和对话模式（llm_chat）
- 内置 ReAct 执行逻辑，支持工具调用和多轮对话
- 通过 `LLMChat` callable instance 与 `SelfRefSession` 将 `llm_chat` 生命周期和 `SelfReference` 状态同步桥接

**子模块**:

- `llm_function_decorator.py`: @llm_function 装饰器实现
- `llm_chat_decorator.py`: @llm_chat 装饰器实现
- `selfref_sync.py` (legacy compatibility shim; active SelfRef session binding is in `LLMChat` + `runtime/selfref/session.py`) (legacy compatibility shim; active SelfRef session binding is in `LLMChat` + `runtime/selfref/session.py`): `llm_chat` 与 `SelfReference` 的生命周期同步桥接
- `steps/function/react.py`: LLM Function 的 ReAct 执行逻辑
- `steps/chat/react.py`: LLM Chat 的 ReAct 执行逻辑
- `steps/common/`: 通用步骤工具（signature, prompt, types 等）
- `utils/tools.py`: 工具注册和处理逻辑

**Spec 位置**: `SimpleLLMFunc/SimpleLLMFunc/llm_decorator/` 目录下

### base 模块

**路径**: `SimpleLLMFunc/SimpleLLMFunc/base/`

**作用**: 提供 ReAct 循环的核心实现和基础工具

**架构特点**:

- ReAct 循环核心实现，支持工具调用编排
- `ReAct_loop()` 已按 phase 拆分，统一 LLM phase、tool batch phase 与 terminal finalize phase
- 工具调用执行、验证、提取逻辑
- 消息处理和类型解析

**子模块**:

- `ReAct.py`: ReAct 循环核心实现
- `tool_call/execution.py`: 工具调用执行逻辑
- `tool_call/validation.py`: 工具结果验证
- `tool_call/extraction.py`: 工具调用提取
- `post_process.py`: 响应后处理
- `messages/`: 消息处理相关
- `type_resolve/`: 类型解析相关

**Spec 位置**: `SimpleLLMFunc/SimpleLLMFunc/base/` 目录下

### hooks 模块

**路径**: `SimpleLLMFunc/SimpleLLMFunc/hooks/`

**作用**: 提供事件流系统，支持实时观察 LLM 调用过程

**架构特点**:

- 丰富的事件类型（LLM 调用、工具调用、ReAct 循环等）
- 支持自定义事件发射
- 事件流异步 yield，支持实时处理

**子模块**:

- `events.py`: 事件类型定义（ReActEvent, CustomEvent 等）
- `stream.py`: 事件流处理（EventYield, ResponseYield 等）
- `event_emitter.py`: 自定义事件发射器（ToolEventEmitter）

**Spec 位置**: `SimpleLLMFunc/SimpleLLMFunc/hooks/` 目录下

### runtime 模块

**路径**: `SimpleLLMFunc/SimpleLLMFunc/runtime/`

**作用**: 提供 runtime primitive 系统、backend 生命周期管理，以及 selfref 状态与纯 context transform 能力

**架构特点**:

- `runtime/primitives.py` 负责 primitive registry 与契约
- `runtime/selfref/state.py` 是 SelfReference public facade，保留兼容 API 并编排子组件
- `runtime/selfref/store.py` 负责 durable history/source store
- `runtime/selfref/active_turn.py` 负责 active memory/fork/toolkit/template contextvars 与 active ReAct state lookup
- `runtime/selfref/mutations.py` 负责 pending compaction/context/destructive mutation queues
- `runtime/selfref/context_memory.py` 负责 context snapshots、experience CRUD、compaction commit 与 direct memory editing
- `runtime/selfref/fork_manager.py` 负责 fork/spawn/gather 生命周期与结果物化
- `runtime/selfref/context_ops.py` 负责纯函数：context parse/render/canonicalize

**子模块**:

- `primitives.py`: runtime primitive registry 与 pack 相关能力
- `selfref/state.py`: SelfReference public facade / lifecycle / compatibility
- `selfref/store.py`: durable history/source store
- `selfref/active_turn.py`: active turn contextvars
- `selfref/mutations.py`: pending mutation queues
- `selfref/memory_api.py`: memory proxy/handle API
- `selfref/context_memory.py`: context memory、experience、compaction、direct editing
- `selfref/agent_binding.py`: recursive agent callable binding
- `selfref/fork_manager.py`: fork/spawn/gather lifecycle
- `selfref/fork_utils.py`: fork helpers/constants
- `selfref/context_ops.py`: selfref context 纯函数转换

### interface 模块

**路径**: `SimpleLLMFunc/SimpleLLMFunc/interface/`

**作用**: 提供 LLM 接口抽象，支持多种 LLM 提供商

**架构特点**:

- 统一的 LLM 接口抽象
- 支持 OpenAI 兼容的所有 API
- 内置 API Key 池管理和限流

**子模块**:

- `llm_interface.py`: LLM 接口基类
- `openai_compatible.py`: OpenAI 兼容接口实现
- `openai_responses_compatible.py`: OpenAI Responses API 适配实现
- `key_pool.py`: API Key 池管理
- `token_bucket.py`: 令牌桶限流实现

### tool 模块

**路径**: `SimpleLLMFunc/SimpleLLMFunc/tool/`

**作用**: 提供 @tool 装饰器，用于定义可被 LLM 调用的工具

**架构特点**:

- 使用装饰器模式定义工具
- 自动提取函数签名生成工具 schema
- 支持多模态参数

**子模块**:

- `tool.py`: Tool 类和 @tool 装饰器实现

### type 模块

**路径**: `SimpleLLMFunc/SimpleLLMFunc/type/`

**作用**: 定义框架内部使用的类型

**子模块**:

- `message.py`: 消息类型与 OpenAI-compatible content part 类型
- `chat_input.py`: `UserChatMessage` canonical chat user input object
- `tool_call.py`: 工具调用类型
- `llm.py`: LLM 相关类型
- `multimodal.py`: 多模态基础类型（`Text` / `ImgUrl` / `ImgPath`）
- `hooks.py`: 事件相关类型

### logger 模块

**路径**: `SimpleLLMFunc/SimpleLLMFunc/logger/`

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

**路径**: `SimpleLLMFunc/SimpleLLMFunc/observability/`

**作用**: 提供可观测性支持，如 Langfuse 集成

**子模块**:

- `langfuse_client.py`: Langfuse 客户端
- `langfuse_config.py`: Langfuse 配置

## 根目录文件

**路径**: `SimpleLLMFunc/`

**文件列表**:

- `config.py`: 框架配置管理
- `utils/`: 通用工具函数与 TUI 组件
- `__init__.py`: 包入口文件

## 测试与示例

### tests 目录

**路径**: `SimpleLLMFunc/tests/`

**作用**: 单元测试和集成测试

**子目录**:

- `test_hooks/`: hooks 模块测试
- `test_base/`: base 模块测试
- `test_llm_decorator/`: 装饰器测试
- `conftest.py`: pytest 配置和 fixtures

### examples 目录

**路径**: `SimpleLLMFunc/examples/`

**作用**: 使用示例和演示代码

**文件列表**:

- `llm_function_*.py`: llm_function 使用示例
- `llm_function_event_*.py`: 事件流使用示例
- `event_stream_*.py`: 事件流示例
- `multi_modality_*.py`: 多模态示例
- `parallel_toolcall_*.py`: 并行工具调用示例

### docs 目录

**路径**: `SimpleLLMFunc/mintlify_docs/`

**作用**: 项目文档（使用 Mintlify）

### spec 目录

**路径**: `SimpleLLMFunc/spec/`

**作用**: Spec 规范文档

**文件列表**:

- `meta.md`: 文档格式规范
- `project-map.md`: 项目模块地图
- `overall-spec.md`: 最佳实践规范
