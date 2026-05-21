# 最佳实践规范

<!-- DOC_SUMMARY_START -->

本文档定义 SimpleLLMFunc 框架的开发规范、使用模式和开发流程。帮助 Agent 理解如何正确使用和扩展框架，包括核心设计理念、开发规范、使用模式和调试指南。

<!-- DOC_SUMMARY_END -->

<!-- DOC_MAP_START -->

## 文档目录 (Document Map)

- [最佳实践规范](#最佳实践规范)
  - [文档更新规范](#文档更新规范)
    - [更新时机](#更新时机)
    - [修改规范](#修改规范)
    - [相关文件](#相关文件)
  - [框架设计原则](#框架设计原则)
    - [核心设计理念](#核心设计理念)
    - [架构分层](#架构分层)
    - [核心上下文规则](#核心上下文规则)
  - [开发规范](#开发规范)
    - [代码组织](#代码组织)
    - [命名规范](#命名规范)
    - [类型标注](#类型标注)
  - [使用模式](#使用模式)
    - [LLM Function 模式](#llm-function-模式)
    - [LLM Chat 模式](#llm-chat-模式)
    - [Tool 定义模式](#tool-定义模式)
    - [Event Stream 模式](#event-stream-模式)
    - [SelfRef / Runtime Primitive 模式](#selfref--runtime-primitive-模式)
  - [事件系统最佳实践](#事件系统最佳实践)
    - [内置事件类型](#内置事件类型)
    - [自定义事件发射](#自定义事件发射)
  - [开发流程](#开发流程)
    - [需求分析](#需求分析)
    - [方案设计](#方案设计)
    - [实现步骤](#实现步骤)
    - [测试验证](#测试验证)
  - [调试指南](#调试指南)
    - [常见问题排查](#常见问题排查)
    - [日志使用](#日志使用)

<!-- DOC_MAP_END -->

<!-- DOC_META_START -->

## 文档更新规范

### 更新时机

以下情况需要更新本文档：

- **添加新的使用模式**: 当框架支持新的使用方式时，需要添加相应章节
- **修改框架设计**: 当框架的核心设计理念或架构发生变化时，需要更新设计原则部分
- **总结新的开发经验**: 当开发过程中总结出新的最佳实践时，需要更新相应章节

### 修改规范

1. **保持格式一致性**:
   - 遵循 `spec/meta.md` 中定义的格式规范
   - 更新 DOC_MAP 时，确保所有链接的锚点 ID 正确
   - 使用示例应简洁明了

2. **内容准确性**:
   - 确保开发规范与实际开发流程一致
   - 使用模式应准确反映框架的使用方式
   - 调试指南应包含常见问题的解决方案

3. **示例质量**:
   - 示例应包含完整上下文
   - 关键点应有注释说明

### 相关文件

- **本文档**: `spec/overall-spec.md`
- **格式规范**: `spec/meta.md`
- **项目地图**: `spec/project-map.md`
<!-- DOC_META_END -->

## 框架设计原则

### 核心设计理念

SimpleLLMFunc 遵循以下核心设计理念：

1. **LLM as Function**: 将 LLM 调用视为普通 Python 函数调用，降低使用门槛
2. **Prompt as Code**: Prompt 直接写在函数 DocString 中，与代码共存
3. **Context-Centric**: 上下文是单一事实源；运行时变化通过结构化 `ContextMutation` 在编译边界生效

### 架构分层

框架采用分层架构：

- **装饰器层**: @llm_function、@llm_chat、@tool；从 Python 函数调用构建 `InvocationSpec`
- **编译边界层**: `compile_pipeline.py` / `context_compile.py`；应用 `ContextMutation` 并渲染 LLM 可见消息
- **ReAct Runtime 层**: `react_loop.py` event-only 主循环；编排 LLM phase、工具批次、hooks 与 finalize
- **Runtime / Builtin 层**: runtime primitive、SelfRef、PyRepl、FileToolset
- **接口层**: LLM Provider 适配器，支持 OpenAI-compatible 与 Responses-compatible 实现
- **基础设施层**: 事件流、日志、Langfuse 可观测性、TUI

### 核心上下文规则

所有 ReAct turn 内的上下文变化必须经过编译边界：

1. LLM 调用、工具执行、SelfRef hooks、abort 等生产 `ContextMutation`
2. `react_loop.py` 在下一次 compile boundary 前收集 pending mutations
3. `compile_context()` / `apply_mutations()` 按顺序应用 mutation
4. `compile_invocation_turn()` 生成最终 provider-facing messages

不要在 ReAct 主循环中直接修改 live transcript。SelfRef 在 active turn 内也应通过 pending intent -> `ContextMutation` 的路径影响模型可见上下文。

## 开发规范

### 代码组织

模块划分原则：

- 按功能职责划分模块
- 相关功能集中在同一目录下
- 使用 __init__.py 导出公共接口

### 命名规范

- 文件名: 使用小写下划线 (snake_case)
- 类名: 使用大驼峰 (PascalCase)
- 函数名: 使用小写下划线 (snake_case)
- 常量: 使用大写下划线 (UPPER_SNAKE_CASE)

### 类型标注

- 所有公开函数应包含类型标注
- 使用 typing 模块提供泛型支持
- 复杂类型使用 TypeAlias 定义

## 使用模式

### LLM Function 模式

适用于单次 LLM 调用场景：

- 使用 @llm_function 装饰器
- 函数体为空，Prompt 写在 DocString
- 普通调用 `await func(...)` 返回 typed result
- 如需观察事件流，使用 `async for output in func.stream(...): ...`
- 支持工具调用；最终响应会按返回类型做后处理

### LLM Chat 模式

适用于对话与 Agent 场景：

- 使用 @llm_chat 装饰器
- `@llm_chat` 调用结果是 `AsyncGenerator[ReactOutput, None]`
- 支持多轮上下文，可通过 `history` / `chat_history` 参数注入消息历史
- chat runtime surface 已统一为事件流；不要使用已废弃的 `return_mode` 或 `enable_event` 设计
- 当启用 SelfRef 时，decorated callable instance 提供稳定 agent identity，便于 fork 和 durable memory 绑定

### Tool 定义模式

定义可被 LLM 调用的工具：

- 使用 @tool 装饰器
- 明确 name 和 description
- 支持多模态参数

### Event Stream 模式

观察 LLM 调用过程：

- 核心 ReAct runtime 始终是 event-only，不存在 `enable_event=True/False` 双模式
- `@llm_chat` 直接产出 `ReactOutput`；`@llm_function` 通过 `.stream(...)` 产出 `ReactOutput`
- `ReactOutput` 是 `ResponseYield | EventYield`
- 使用 `is_response_yield()` / `is_event_yield()` 或 `responses_only()` / `events_only()` 过滤输出
- 根据 `output.event.event_type` 处理 LLM 调用、工具调用、ReAct lifecycle、自定义事件等

### SelfRef / Runtime Primitive 模式

适用于带持久上下文、自我压缩、fork 子任务的 Agent：

- `SelfReference` 是 durable backend 的 public facade；`SelfRefSession` 是 invocation-scoped ReAct hook/plugin
- PyRepl 内安装 runtime primitive 后，模型通过 `execute_code` 中的 `runtime.namespace.name(...)` 调用 primitive
- Runtime primitive 不是原生 LLM tool call；它是 worker 内的 host-registered callable
- active ReAct turn 内的 remember / forget / compact 应通过 pending intent 转为 `ContextMutation`，在下一次 compile boundary 生效

## 事件系统最佳实践

### 内置事件类型

框架提供丰富的内置事件：

- LLM 调用相关: `LLM_CALL_START`, `LLM_CALL_END`, `LLM_CHUNK_ARRIVE`
- 工具调用相关: `TOOL_CALLS_BATCH_START`, `TOOL_CALL_START`, `TOOL_CALL_ARGUMENTS_DELTA`, `TOOL_CALL_END`, `TOOL_CALL_ERROR`, `TOOL_CALLS_BATCH_END`
- ReAct 循环相关: `REACT_START`, `REACT_ITERATION_START`, `REACT_ITERATION_END`, `REACT_END`
- 自定义事件: `CUSTOM_EVENT`

### 自定义事件发射

在 Tool 中发射自定义事件：

- 声明 `event_emitter` 参数
- 使用 `await event_emitter.emit(event_name, data)` 发射事件
- 自定义事件会汇入统一 Event Stream，并携带 `EventOrigin` 元数据

## 开发流程

### 需求分析

1. 阅读文档了解框架能力
2. 使用 grep 搜索相关代码
3. 理解现有实现模式

### 方案设计

1. 确定需要修改的文件
2. 设计新增的类型和函数
3. 评估兼容性影响

### 实现步骤

1. 定义类型（如需要）
2. 实现核心逻辑
3. 更新导出接口
4. 编写单元测试

### 测试验证

1. 运行单元测试
2. 创建示例验证
3. 检查日志输出

## 调试指南

### 常见问题排查

- Tool 参数检查问题: 检查 Tool 对象的 parameters 列表
- 事件不显示问题: 确认消费的是 `ReactOutput` event stream，并检查 `event_emitter` 是否正确传递
- 类型错误: 检查类型标注是否正确

### 日志使用

- 使用 LOG_LEVEL 控制日志级别
- 在关键路径添加调试日志
- 使用 trace_id 追踪调用链
