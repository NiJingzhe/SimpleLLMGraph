# Stage 2 统一 Invocation / Compile Boundary 重构计划

<!-- DOC_SUMMARY_START -->
本文档定义 SimpleLLMFunc 在 ReAct Core Stage 1 之后的 Stage 2 重构目标：统一 `llm_chat` 与 `llm_function` 的调用建模、唯一化 compile / convert-to-LLM 边界、并将 `selfref` 收敛为 invocation-scoped session/plugin。文档重点固化边界、数据传递方式、selfref 复杂度控制、迁移阶段、验收标准与测试策略。
<!-- DOC_SUMMARY_END -->

<!-- DOC_MAP_START -->
## 文档目录 (Document Map)

- [Status](#status)
- [Why Stage 2](#why-stage-2)
- [Stage 2 目标](#stage-2-目标)
  - [核心目标](#核心目标)
  - [非目标](#非目标)
- [架构原则](#架构原则)
- [稳定兼容边界](#稳定兼容边界)
- [核心术语](#核心术语)
- [五层正式边界](#五层正式边界)
  - [A. Decorator Boundary](#a-decorator-boundary)
  - [B. Attachment Resolution Boundary](#b-attachment-resolution-boundary)
  - [C. Compile / Convert-to-LLM Boundary](#c-compile--convert-to-llm-boundary)
  - [D. ReAct Runtime Boundary](#d-react-runtime-boundary)
  - [E. Finalize Boundary](#e-finalize-boundary)
- [统一数据模型](#统一数据模型)
  - [InvocationSpec](#invocationspec)
  - [PromptContract](#promptcontract)
  - [TranscriptSeed](#transcriptseed)
  - [InvocationAttachments](#invocationattachments)
  - [CompiledTurnContext](#compiledturncontext)
  - [FinalTurnState](#finalturnstate)
- [统一执行流程](#统一执行流程)
  - [llm_function 路径](#llm_function-路径)
  - [llm_chat 路径](#llm_chat-路径)
  - [统一 Invocation Runner](#统一-invocation-runner)
- [Selfref Stage 2 设计](#selfref-stage-2-设计)
  - [backend 与 session 分层](#backend-与-session-分层)
  - [SelfRefSession 允许的行为](#selfrefsession-允许的行为)
  - [SelfRefSession 禁止的行为](#selfrefsession-禁止的行为)
  - [Selfref 的状态传递方式](#selfref-的状态传递方式)
  - [History Authority 规则](#history-authority-规则)
  - [Fork 规则](#fork-规则)
- [Compile Pipeline 设计](#compile-pipeline-设计)
- [Compatibility / Adapter 策略](#compatibility--adapter-策略)
- [模块与文件迁移计划](#模块与文件迁移计划)
- [分阶段实施计划](#分阶段实施计划)
- [测试策略](#测试策略)
- [验收标准](#验收标准)
- [一句话总结](#一句话总结)
<!-- DOC_MAP_END -->

<!-- DOC_META_START -->
## 文档更新规范

### 更新时机
- Stage 2 设计边界、Invocation 数据模型、Selfref session 协议发生变化时
- `llm_function` / `llm_chat` 的统一路径发生变化时
- `selfref` fork / finalize / history authority 规则发生变化时
- compile / convert-to-LLM 主入口的模块职责发生变化时

### 修改规范
- 保持“边界 / 输入 / 输出 / 禁止项”格式，避免再次回到模糊叙述
- 所有新增数据结构需说明：拥有者、传递方向、生命周期
- 所有 selfref 相关规则必须明确写出：允许什么、禁止什么、何时生效
- 文档更新时必须同时检查 `llm_function` 与 `llm_chat` 是否都被覆盖

### 相关文件
- 本文档: `design/stage-2-unified-invocation.md`
- Stage 1 规划: `design/react-core-refactor.md`
- Stage 1 已实现架构: `design/react-core-implemented-architecture.md`
- 当前 compile source: `SimpleLLMFunc/base/context_source.py`
- 当前 compile 实现: `SimpleLLMFunc/base/context_compile.py`, `SimpleLLMFunc/base/context_source_compile.py`
<!-- DOC_META_END -->

## Status {#status}

- Branch target: future `refactor/stage-2-unified-invocation`
- Status: design document before Stage 2 implementation
- Scope:
  - `SimpleLLMFunc/llm_decorator/*`
  - `SimpleLLMFunc/base/*` compile boundary related modules
  - `SimpleLLMFunc/runtime/selfref/*`
  - `SimpleLLMFunc/builtin/pyrepl.py` fork/runtime backend integration points
  - compatibility adapters between public decorator APIs and the event-only ReAct core

---

## Why Stage 2 {#why-stage-2}

Stage 1 已经完成了重要工作：

1. `ReAct.py` 被拆小，主循环进入 `base/react_loop.py`
2. Core 改成 event-only
3. 引入了 structured mutations
4. compile-time mutation application 成为明确规则
5. selfref 的 same-turn compaction/remember/forget 已经部分 mutation 化

但当前系统仍然存在 Stage 2 必须处理的结构性问题：

1. **compile 入口不唯一**
   - `context_compile.py`
   - `context_source_compile.py`
   - `steps/function/prompt.py`
   - `steps/chat/message.py`
   这些模块都在部分承担“把调用语义变成模型输入”的职责。

2. **`llm_function` 与 `llm_chat` 仍存在双轨架构**
   - chat 路径已经引入 `CompileSource`
   - function 路径仍大量直接 build provider-facing messages

3. **decorator 过胖**
   decorator 当前不只是收集 Python 调用信息，还在承担 runtime 组装、selfref 生命周期、history finalize、fork cloning 等复杂责任。

4. **selfref 仍然没有被完全关进一个明确的 invocation-scoped 边界**
   它虽然已不再完全直接改 live context，但仍对 decorator 结构造成明显侵入。

Stage 2 的任务不是“继续把 core 拆成更多文件”，而是：

> 统一 invocation 建模、唯一化 compile / convert-to-LLM 边界，并把 selfref 从 decorator 负担升级为正式 session/plugin。

---

## Stage 2 目标 {#stage-2-目标}

### 核心目标 {#核心目标}

1. **Decorator 瘦身**
   - decorator 只负责把 Python 调用变成 `InvocationSpec`
   - decorator 不再直接构建 provider-facing messages
   - decorator 不再直接承载 selfref active key / finalize / fork cloning 细节

2. **Compile Boundary 唯一化**
   - `llm_function` 与 `llm_chat` 必须共享一个正式 compile / convert-to-LLM 主入口
   - 该入口负责从 invocation 语义与 runtime state 生成最终 LLM 请求

3. **Selfref Invocation 化**
   - `SelfReference` 继续作为 durable backend 保留
   - 每次调用创建 `SelfRefSession`
   - selfref 只通过：source snapshot、pending intents -> mutations、finalize side effects 进入系统

4. **Compatibility 外移**
   - core 只认 event-only / structured data boundary
   - raw/text/tuple/typed 等 public API 兼容逻辑全部外移到 adapter 层

### 非目标 {#非目标}

Stage 2 不做以下事情：

1. 不改变 `LLM_Interface` 下边界
2. 不要求 provider adapter 重写
3. 不改变 `@llm_function` / `@llm_chat` 的公开装饰器名字
4. 不推倒现有 ReAct core loop 的 event-only 方向
5. 不把 selfref 简化成普通 KV memory；selfref 仍然支持 compaction / durable experiences / fork

---

## 架构原则 {#架构原则}

Stage 2 固化以下原则：

1. **Decorator 不拥有 runtime context**
   decorator 只收集调用语义，不拥有 provider-facing messages，也不拥有 runtime 持久状态。

2. **系统里只能有一个正式 convert-to-LLM 边界**
   不允许 function/chat 各自拥有独立的最终 prompt/message 组装路径。

3. **Runtime 只负责执行，不负责解释调用语义**
   ReAct core 只处理 llm call / tool batch / event / abort / mutations。

4. **Selfref 只能通过 snapshot + mutations + finalize 影响系统**
   selfref 不允许直接修改 live transcript。

5. **Compatibility 只能存在于最外层 adapter**
   core 内部不再混用 tuple/raw/text/event 多套协议。

6. **所有 invocation state 必须是 per-call 的**
   不允许 decorator 闭包共享调用结果状态，不允许同一个 decorated function 并发时共享结果容器。

---

## 稳定兼容边界 {#稳定兼容边界}

Stage 2 必须保留以下稳定边界：

1. **LLM Interface Layer 保持稳定**
   - `LLM_Interface.chat(...)`
   - `LLM_Interface.chat_stream(...)`
   - provider-facing message payload shape

2. **Public Decorator API 保持稳定**
   - `@llm_function(...)`
   - `@llm_chat(...)`
   - `@tool(...)`
   - `enable_event=True` 的事件流能力

3. **Core Event Model 保持稳定方向**
   - core 继续 event-only
   - `ReactOutput` 继续是 core 唯一输出模式

4. **Selfref 能力保留**
   - durable experience remember/forget
   - context compaction
   - forked sub-agent
   - same-turn compaction before next compile boundary

---

## 核心术语 {#核心术语}

- **Invocation**: 一次用户对 decorated function 的实际调用
- **InvocationSpec**: decorator 收集到的“调用语义契约”
- **PromptContract**: prompt 约束的结构化表示，不等于最终 provider message
- **Transcript**: 语义消息历史，作为 source-of-truth
- **LLM Messages**: 最终 provider-safe 模型输入
- **Compile / Convert-to-LLM**: 从 transcript + prompt contract + runtime overlays 生成 LLM 请求的唯一正式边界
- **Attachment**: invocation 运行时依赖，如 tool registry / selfref session / hooks / observability
- **SelfRefSession**: 一次 invocation 生命周期内 selfref 的 session/plugin 包装
- **Finalize**: invocation 结束后执行 side effects 与 public surface adaptation 的阶段

---

## 五层正式边界 {#五层正式边界}

## A. Decorator Boundary {#a-decorator-boundary}

### 输入
- Python `args/kwargs`
- 原始函数对象
- decorator 配置（`llm_interface`, `toolkit`, `stream`, `enable_event`, `return_mode`, `max_tool_calls` 等）

### 输出
- `InvocationSpec`
- `AttachmentRequest`

### 负责
- 签名绑定
- template params 提取
- 历史参数提取
- docstring 读取
- return type / parameter contract 建模
- trace_id 生成
- invocation mode 决定（`function` or `chat`）

### 禁止
- 构建最终 provider-facing messages
- 直接执行 `ReAct_loop`
- 直接操作 selfref backend
- 直接 finalize history / selfref store
- 直接 clone fork toolkit / runtime backend

---

## B. Attachment Resolution Boundary {#b-attachment-resolution-boundary}

### 输入
- `InvocationSpec`
- decorator 声明的 toolkit / selfref / hooks / observability config

### 输出
- `InvocationAttachments`

### 负责
- tool registry 准备
- selfref session 创建
- hooks/plugin manager 组装
- observability context 组装
- fork service 绑定
- abort signal 绑定

### 禁止
- compile prompt/messages
- llm call
- 修改 transcript
- 产生 provider-facing message

---

## C. Compile / Convert-to-LLM Boundary {#c-compile--convert-to-llm-boundary}

### 输入
- `InvocationSpec`
- `InvocationAttachments`
- `TranscriptState`
- `PendingMutations`

### 输出
- `CompiledTurnContext`
- `LLMRequest`

### 负责
- apply context mutations
- 合并 selfref source overlay
- 生成 system prompt
- 注入 tool prompt specs / must principles
- 生成 provider-safe `llm_messages`
- 生成本轮 LLM 请求

### 禁止
- 真正调用 LLM
- 真正执行 tool
- 直接 persist selfref
- 直接同步用户传入的 raw history list

---

## D. ReAct Runtime Boundary {#d-react-runtime-boundary}

### 输入
- `LLMRequest`
- tool registry
- plugins/hooks
- abort signal

### 输出
- `ReactOutput` stream
- `RuntimeMutations`
- `FinalTurnState`

### 负责
- llm call
- tool scheduler
- event bus
- abort handling
- mutation accumulation
- loop lifecycle

### 禁止
- 理解 docstring 语义
- 决定 function/chat 初始 prompt 契约
- 直接写 selfref store
- 直接 sync 用户 history 参数

---

## E. Finalize Boundary {#e-finalize-boundary}

### 输入
- `FinalTurnState`
- `InvocationSpec`
- `InvocationAttachments`

### 输出
- plugin side effects
- external history sync
- final public return surface

### 负责
- selfref session finalize
- external history ref sync
- typed result parse（`llm_function`）
- text/raw/history/event adapters
- observability closeout

### 禁止
- 再次 compile
- 再次调用 llm
- 在 finalize 阶段补造“伪 turn”

---

## 统一数据模型 {#统一数据模型}

## InvocationSpec {#invocationspec}

```python
@dataclass(frozen=True)
class InvocationSpec:
    mode: Literal["function", "chat"]
    func_name: str
    trace_id: str
    docstring: str
    bound_args: Mapping[str, Any]
    type_hints: Mapping[str, Any]
    return_type: Any
    template_params: Mapping[str, Any] | None
    llm_kwargs: Mapping[str, Any]
    stream: bool
    enable_event: bool
    return_mode: Literal["text", "raw", "typed"] | None
    prompt_contract: PromptContract
    transcript_seed: TranscriptSeed
```

说明：
- `InvocationSpec` 是 decorator 的最终产物
- 它表达“这次 Python 调用想做什么”
- 它不是 provider request，也不是最终 LLM messages

---

## PromptContract {#promptcontract}

```python
@dataclass(frozen=True)
class PromptContract:
    base_instruction: str
    parameter_contract: list[ParameterContract]
    return_contract: ReturnContract | None
    tool_prompt_specs: list[ToolPromptSpec]
    include_must_principles: bool
```

说明：
- `base_instruction` 通常来自 docstring
- `parameter_contract` 描述参数如何进入 prompt 语义
- `return_contract` 对 `llm_function` 特别重要，用于 typed / structured output 约束
- `tool_prompt_specs` 不再由 decorator 直接注入 system prompt，而是交给 compile boundary

---

## TranscriptSeed {#transcriptseed}

```python
@dataclass(frozen=True)
class TranscriptSeed:
    initial_messages: NormalizedMessageList
    external_history_ref: list[dict[str, Any]] | None = None
    history_authority: Literal["external", "selfref", "seed"] = "seed"
```

说明：
- `initial_messages` 是 source-level 初始 transcript
- `external_history_ref` 只在需要同步用户传入 history 列表时存在
- `history_authority` 明确谁是本次 invocation 的 transcript authority

---

## InvocationAttachments {#invocationattachments}

```python
@dataclass
class InvocationAttachments:
    tool_registry: ToolRegistry
    plugins: list[InvocationPlugin]
    selfref_session: SelfRefSession | None
    abort_signal: AbortSignal | None
    observability: ObservabilityContext | None
    fork_service: ForkService | None
```

说明：
- 所有 runtime 依赖都在这里
- decorator 只请求 attachments，不直接操纵 attachments 内部细节

---

## CompiledTurnContext {#compiledturncontext}

```python
@dataclass(frozen=True)
class CompiledTurnContext:
    transcript: NormalizedMessageList
    system_prompt: str | None
    llm_messages: NormalizedMessageList
    selfref_snapshot: SelfRefSourceSnapshot | None
```

说明：
- `transcript` 是 source-of-truth
- `llm_messages` 才是最终送给模型的 provider-safe messages
- 这两个必须显式区分，不能混用

---

## FinalTurnState {#finalturnstate}

```python
@dataclass(frozen=True)
class FinalTurnState:
    transcript: NormalizedMessageList
    llm_messages: NormalizedMessageList
    final_response: Any
    final_text: str
    usage: Any
    aborted: bool
    total_llm_calls: int
    total_tool_calls: int
```

说明：
- finalizer 与 public result adapter 只消费 `FinalTurnState`
- 不再依赖 decorator 内部的隐式局部变量

---

## 统一执行流程 {#统一执行流程}

## llm_function 路径 {#llm_function-路径}

Stage 2 中，`llm_function` 不再拥有独立的“直接 build initial prompts -> 调 core”路径。

它需要提供的差异只包括：

1. `mode="function"`
2. `PromptContract.return_contract` 明确输出约束
3. `TranscriptSeed.initial_messages` 通常只包含：
   - 由参数构造出的当前 user task seed
   - 可选多模态输入
4. Finalize 阶段使用 typed result adapter：
   - `str/int/float/bool`
   - `list/dict/Union`
   - Pydantic model

换言之：

> `llm_function` 的特殊性体现在 contract 和 result adapter，不体现在另起一条 compile 架构。

---

## llm_chat 路径 {#llm_chat-路径}

Stage 2 中，`llm_chat` 也进入同一 Invocation -> Compile -> Runtime -> Finalize 主线。

它需要提供的差异只包括：

1. `mode="chat"`
2. `TranscriptSeed.initial_messages` 包含过滤后的历史与当前用户输入
3. `history_authority` 可以是：
   - `external`
   - `selfref`
   - `seed`
4. Finalize 阶段使用 chat result adapter：
   - `(text, history)`
   - `(raw, history)`
   - `ReactOutput` stream

---

## 统一 Invocation Runner {#统一-invocation-runner}

统一主流程应为：

```python
spec = build_invocation_spec(...)
attachments = resolve_attachments(spec, ...)
result = await run_invocation(spec, attachments)
return adapt_public_result(result, spec, attachments)
```

其中 `run_invocation(...)` 的 ReAct 主循环概念为：

```python
while True:
    pending_mutations.extend(plugins.collect_pending_mutations())
    compiled = compile_invocation_turn(spec, attachments, state, pending_mutations)

    llm_result = await single_llm_call(compiled.llm_messages)
    pending_mutations.extend(llm_result.mutations)

    if not llm_result.tool_calls:
        break

    tool_result = await schedule_tool_batch(llm_result.tool_calls)
    pending_mutations.extend(tool_result.mutations)
```

---

## Selfref Stage 2 设计 {#selfref-stage-2-设计}

## backend 与 session 分层 {#backend-与-session-分层}

Stage 2 必须明确拆分：

1. **`SelfReference` backend**
   - durable store
   - memory key namespace
   - experience / summary / messages 数据持久化
   - fork parent/child durable relationship

2. **`SelfRefSession`**
   - invocation-scoped plugin/session
   - 负责本次调用中 selfref 的所有 runtime 交互

建议引入：

```python
class SelfRefSession(InvocationPlugin):
    ...
```

---

## SelfRefSession 允许的行为 {#selfrefsession-允许的行为}

`SelfRefSession` 允许：

1. 解析本次 invocation 使用的 memory key
2. 提供 source snapshot：
   - base_system_prompt
   - experiences
   - summary
   - summary_message
   - 必要的 transcript snapshot
3. 记录 pending intents：
   - remember
   - forget
   - compact
4. 在 compile 边界输出 structured mutations
5. 在 finalize 边界将最终 transcript 持久化回 backend
6. 为 fork child invocation 派生 child session

---

## SelfRefSession 禁止的行为 {#selfrefsession-禁止的行为}

`SelfRefSession` 禁止：

1. 直接 append / replace live transcript
2. 在 decorator 中直接修改用户传入的 history list
3. 直接 build provider-facing message
4. 在 decorator 里完成 pyrepl/toolkit clone
5. 绕过 compile boundary 直接让 remember/forget/compact 进入模型上下文

---

## Selfref 的状态传递方式 {#selfref-的状态传递方式}

Stage 2 固化 selfref 传递规则：

> 工具/primitive 只能向 `SelfRefSession` 写入“意图”；runtime 只在 compile 边界 pull 出 mutations。

即：

### Tool / Primitive 阶段允许
```python
selfref_session.queue_remember(...)
selfref_session.queue_forget(...)
selfref_session.queue_compaction(...)
```

### Compile 边界允许
```python
pending = selfref_session.collect_pending_mutations()
```

### Finalize 边界允许
```python
selfref_session.finalize(final_turn_state)
```

### 不允许
```python
# 禁止在 tool / primitive 内部直接改 live transcript
selfref_backend.messages.append(...)
```

该规则的目的：
- 保持 mutation boundary 纯净
- 避免 selfref 在任意阶段越权改上下文
- 让 same-turn selfref 变更只在下一个 compile boundary 生效

---

## History Authority 规则 {#history-authority-规则}

Stage 2 明确 transcript authority：

### 情况 1：显式传入 `history`
- `history_authority = "external"`
- transcript 以用户传入 history 为准
- selfref 只作为 system overlay + persistence sink
- finalize 时同步更新 external history ref 与 selfref backend

### 情况 2：无 external history，但 selfref 有已有上下文
- `history_authority = "selfref"`
- 初始 transcript 由 selfref snapshot 提供
- finalize 时只写回 selfref

### 情况 3：两者都没有
- `history_authority = "seed"`
- transcript 由当前 invocation 的 seed messages 构成
- 如果存在 selfref session，则 finalize 后可成为新的 durable context

该规则必须显式存在，不能靠 decorator 里的隐式分支推断。

---

## Fork 规则 {#fork-规则}

Stage 2 中 fork 不再由 decorator 直接操纵 toolkit / selfref / pyrepl。

建议引入：

```python
class ForkService:
    async def spawn_child(
        self,
        parent_spec: InvocationSpec,
        parent_attachments: InvocationAttachments,
        request: ForkRequest,
    ) -> ChildInvocationHandle:
        ...
```

### ForkService 负责
- child origin / fork metadata
- child attachments 派生
- child selfref session 派生
- child runtime backend / pyrepl clone
- child invocation 启动

### selfref 在 fork 中只负责
- child memory key / source memory key 派生
- child snapshot policy
- child finalize policy

### decorator 在 fork 中禁止负责
- 不直接 clone toolkit
- 不直接 patch child history
- 不直接写 event origin

---

## Compile Pipeline 设计 {#compile-pipeline-设计}

Stage 2 建议新增统一模块，例如：

- `SimpleLLMFunc/base/compile_pipeline.py`

它对外提供唯一正式入口：

```python
compile_invocation_turn(...)
```

其内部可拆为两步，但对外只暴露一个正式 compile boundary：

### 1. reduce_turn_context

```python
reduce_turn_context(
    transcript: NormalizedMessageList,
    pending_mutations: list[ContextMutation],
    selfref_snapshot: SelfRefSourceSnapshot | None,
) -> ReducedTurnContext
```

职责：
- apply mutations
- 更新 transcript
- 处理 summary / experience 变更
- 保持 transcript 合法

### 2. convert_to_llm_request

```python
convert_to_llm_request(
    reduced: ReducedTurnContext,
    prompt_contract: PromptContract,
) -> CompiledTurnContext
```

职责：
- 生成 system prompt
- 注入 tool prompt specs
- 注入 must principles
- 渲染 provider-safe `llm_messages`

该拆分明确区分：
- **语义状态归约**
- **语义状态 -> 模型请求转换**

这也是 Stage 2 的核心：

> compile 不能再既代表 mutation application，又含混代表 prompt rendering；必须把“语义归约”与“convert-to-LLM”在同一个正式边界下清晰拆分。

---

## Compatibility / Adapter 策略 {#compatibility--adapter-策略}

Stage 2 中，compatibility 只允许存在于 public result adapter 层。

### core 内部只认
- `ReactOutput`
- `ContextMutation`
- `CompiledTurnContext`
- `FinalTurnState`

### adapter 层负责
- `llm_function` typed result parse
- `llm_chat` `(response, history)` tuple surface
- `raw` / `text` 模式
- `enable_event=False` 的非事件表面
- 旧行为兼容的 coercion

### 结果
- `base/react_loop.py` / `base/llm_call.py` / `base/tool_scheduler.py` 不再关心 tuple/raw/text
- `steps/function/react.py` / `steps/chat/react.py` 逐步退化为 thin adapters，最终可合并

---

## 模块与文件迁移计划 {#模块与文件迁移计划}

### 新增建议模块
- `SimpleLLMFunc/llm_decorator/invocation_spec.py`
- `SimpleLLMFunc/llm_decorator/invocation_builder.py`
- `SimpleLLMFunc/base/compile_pipeline.py`
- `SimpleLLMFunc/runtime/plugins/protocol.py`
- `SimpleLLMFunc/runtime/selfref/session.py`
- `SimpleLLMFunc/runtime/fork_service.py`
- `SimpleLLMFunc/llm_decorator/result_adapters.py`

### 保留为核心 runtime 的模块
- `SimpleLLMFunc/base/react_loop.py`
- `SimpleLLMFunc/base/llm_call.py`
- `SimpleLLMFunc/base/tool_scheduler.py`
- `SimpleLLMFunc/base/mutation.py`
- `SimpleLLMFunc/hooks/*`

### 需要收口或弱化职责的模块
- `SimpleLLMFunc/base/context_compile.py`
  - 保留 mutation apply 能力
  - 不再单独代表完整 compile 入口

- `SimpleLLMFunc/base/context_source_compile.py`
  - 迁入 `compile_pipeline.py` 或退化成内部 helper
  - 不再作为另一条独立 compile 路径

- `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`
  - 保留 contract build 逻辑
  - 停止直接构建最终 provider-facing messages

- `SimpleLLMFunc/llm_decorator/steps/chat/message.py`
  - 保留 history extraction / transcript seed build
  - 停止承担最终 compile 责任

- `SimpleLLMFunc/llm_decorator/steps/function/react.py`
- `SimpleLLMFunc/llm_decorator/steps/chat/react.py`
  - 逐步退化为 thin compatibility adapters

- `SimpleLLMFunc/llm_decorator/llm_chat_decorator.py`
  - 移除 selfref 生命周期与 fork cloning 细节

- `SimpleLLMFunc/llm_decorator/llm_function_decorator.py`
  - 移除 per-decorator shared result state
  - 统一走 invocation runner

---

## 分阶段实施计划 {#分阶段实施计划}

### Stage 2A: InvocationSpec 化
目标：让 decorator 先只负责生成 `InvocationSpec`。

动作：
1. 引入 `InvocationSpec` / `PromptContract` / `TranscriptSeed`
2. function/chat 分别实现 spec builder
3. 移除 decorator 闭包共享调用状态
4. 让每次调用都具备 invocation-scoped state 容器

退出标准：
- decorator 可以独立生成完整 invocation spec
- 不再使用共享 `parsed_result` 一类 per-decorator 可变状态

### Stage 2B: Compile Boundary 唯一化
目标：function/chat 共用一个 compile / convert-to-LLM 主入口。

动作：
1. 新建 `base/compile_pipeline.py`
2. 收口 `context_compile.py` 与 `context_source_compile.py`
3. `llm_function` 停止直接 build initial prompts
4. `llm_chat` 停止持有单独的最终 compile 路径

退出标准：
- 系统中只有一个正式 compile 主入口
- `llm_function` 与 `llm_chat` 都通过该入口生成 `llm_messages`

### Stage 2C: SelfRefSession 化
目标：把 selfref 从 decorator 细节提升为正式 plugin/session。

动作：
1. 引入 `SelfRefSession`
2. remember/forget/compact 统一排队为 intent
3. compile 边界 pull mutations
4. finalize 边界统一落盘与 history sync
5. 明确 history authority 规则

退出标准：
- `llm_chat_decorator.py` 中不再拥有复杂 selfref 生命周期逻辑
- selfref 不再直接修改 live transcript

### Stage 2D: Fork Service + Adapter 外移
目标：把 fork 与兼容逻辑从 decorator 移走。

动作：
1. 引入 `ForkService`
2. pyrepl/runtime backend clone 迁入 fork service
3. result adapters 外移到 `result_adapters.py`
4. 清理 steps 层重复 coercion

退出标准：
- decorator 不再负责 fork clone
- compatibility 只在 adapter 层存在

---

## 测试策略 {#测试策略}

Stage 2 需要新增“架构测试”，而不仅是功能测试。

### 1. Invocation isolation test
验证同一 decorated function 并发调用时：
- typed result 不串
- selfref session 不串
- history ref 不串

### 2. Compile single-entry test
验证：
- `llm_function`
- `llm_chat`
都走同一 compile 主入口。

### 3. History authority test
分别验证：
- `external`
- `selfref`
- `seed`
三种 authority 的初始化与 finalize 行为。

### 4. Selfref pending mutation timing test
验证：
- remember/forget/compact 在 tool batch 中只记录 intent
- 仅在 compile boundary 生效
- finalize 时 pending-on-finalize policy 正确

### 5. Fork inheritance test
验证 child invocation：
- fork origin 正确
- child selfref snapshot 正确
- parent pending tool scene 不泄漏
- child pyrepl/runtime backend 独立 clone

### 6. No provider-message construction in decorator test
验证 decorator 层不再生成最终 provider-facing messages。

---

## 验收标准 {#验收标准}

Stage 2 完成时必须满足：

1. `llm_function` 与 `llm_chat` 共用一个正式 compile / convert-to-LLM 主入口
2. decorator 只负责 spec build + attachment resolve
3. selfref 变成 invocation-scoped session/plugin
4. external history sync 只发生在 finalize
5. compatibility 只存在于 adapter 层
6. 不再存在 decorator 闭包共享调用结果状态

最好同时满足：

1. `llm_chat_decorator.py` 明显瘦身
2. `llm_function_decorator.py` 走统一 invocation runner
3. `_clone_toolkit_for_fork` 迁出 decorator
4. `context_source_compile.py` 不再作为独立 compile 路径存在

---

## 一句话总结 {#一句话总结}

Stage 2 要解决的不是“继续拆更多 base 文件”，而是：

> 把“Python 调用语义”与“LLM 请求渲染”彻底分开，并让 `llm_function`、`llm_chat`、`selfref` 都只通过统一 invocation / compile / finalize 边界进入系统。

这是 Stage 2 的唯一北极星。
