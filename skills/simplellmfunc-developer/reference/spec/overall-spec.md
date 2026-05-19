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
  - [开发规范](#开发规范)
    - [代码组织](#代码组织)
    - [命名规范](#命名规范)
    - [类型标注](#类型标注)
  - [使用模式](#使用模式)
    - [LLM Function 模式](#llm-function-模式)
    - [LLM Chat 模式](#llm-chat-模式)
    - [Tool 定义模式](#tool-定义模式)
    - [Event Stream 模式](#event-stream-模式)
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
3. **Code as Doc**: 函数定义本身就是完整文档，享受 IDE 类型提示

### 架构分层

框架采用分层架构：

- **装饰器层**: @llm_function, @llm_chat, @tool 装饰器
- **执行层**: phase-based ReAct 循环实现，工具调用编排与统一 finalize 收口
- **运行时状态层**: runtime primitive、selfref state、纯 context transform
- **接口层**: LLM 接口抽象，支持多提供商
- **基础设施层**: 日志、事件流、可观测性

## 开发规范

### 代码组织

模块划分原则：

- 按功能职责划分模块
- 相关功能集中在同一目录下
- 使用 __init__.py 导出公共接口
- 优先明确区分纯函数转换、本地状态变更、以及外部副作用/编排逻辑

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
- 支持工具调用和多轮对话

### LLM Chat 模式

适用于对话场景：

- 使用 @llm_chat 装饰器
- 支持多轮上下文
- 可注入消息历史

### Tool 定义模式

定义可被 LLM 调用的工具：

- 使用 @tool 装饰器
- 明确 name 和 description
- 支持多模态参数

### Event Stream 模式

观察 LLM 调用过程：

- 消费 `ReactOutput` 事件流
- 遍历 yield 输出
- 根据事件类型处理

## 事件系统最佳实践

### 内置事件类型

框架提供丰富的内置事件：

- LLM 调用相关: LLM_CALL_START, LLM_CALL_END, LLM_CHUNK_ARRIVE
- 工具调用相关: TOOL_CALL_START, TOOL_CALL_END, TOOL_CALL_ERROR
- ReAct 循环相关: REACT_START, REACT_ITERATION_START, REACT_END

### 自定义事件发射

在 Tool 中发射自定义事件：

- 声明 event_emitter 参数
- 使用 emit() 方法发射事件
- 事件会自动汇入 Event Stream

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

### ReAct / selfref 维护约束

- `base/ReAct.py` 中的新终态分支应复用统一 finalize 路径，避免绕过 `before_finalize`
- selfref 相关纯 context transform 优先放在 `runtime/selfref/context_ops.py`
- `runtime/selfref/state.py` 负责有状态存储与 mutation，不要把纯 parse/render 逻辑重新塞回去
- `LLMChat` callable instance 与 `SelfRefSession` 负责 `llm_chat` 和 `SelfReference` 的生命周期桥接；`selfref_sync.py` 仅为 legacy compatibility shim
- Provider-specific API 差异优先留在 `interface/` 适配层；例如 Responses API 的 `instructions`、reasoning、stream chunk 适配，不要直接塞进 `ReAct`
- selfref fork child 的可见上下文应来自 pre-fork 快照，不要把父 agent 当前 pending `tool_calls` message 重新带回 child history

### 测试验证

1. 运行单元测试
2. 创建示例验证
3. 检查日志输出

## 调试指南

### 常见问题排查

- Tool 参数检查问题: 检查 Tool 对象的 parameters 列表
- 事件不显示问题: 检查 event_emitter 是否正确传递
- 类型错误: 检查类型标注是否正确

### 日志使用

- 使用 LOG_LEVEL 控制日志级别
- 在关键路径添加调试日志
- 使用 trace_id 追踪调用链
