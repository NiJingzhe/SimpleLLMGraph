# SimpleLLMFunc 架构毒舌审查报告

> 立场声明：本文不批判项目的哲学。`LLM is Function`、`Prompt as Code`、`Context-Centric` 这三件事作为产品判断可以成立；本文只审视当前代码里的架构是否配得上这些哲学。
>
> 审查日期：2026-05-20
> 审查对象：当前工作区 `SimpleLLMFunc`，版本号见 `pyproject.toml` 为 `0.8.0`。

## 一句话结论

这是一个**有明确架构野心、核心边界正在变干净、但外围还没有摆脱“作者本人脑内编译器”的 Agent Framework**。

好消息是：核心 ReAct 已经从早期框架常见的“一坨流程脚本”进化为较清晰的事件流、Mutation 和编译边界；这是品味在线的。坏消息是：`llm_chat_decorator.py`、`SelfReference`、`PyRepl`、Provider Adapter 和结构化输出链路仍然把太多策略、状态、兼容逻辑和协议适配塞在同一层。项目不是没有架构，而是架构的**正确部分还没有统治全局**。

如果按 Agent Framework 的成熟度打分：

| 维度 | 评价 |
| --- | --- |
| 核心 ReAct 抽象 | 8/10，有清晰方向 |
| 上层 Decorator 架构 | 5/10，API 漂亮，内部臃肿 |
| Context / Mutation 模型 | 8/10，最值得保留的资产 |
| SelfRef / Fork | 6/10，能力强但边界危险 |
| Provider 抽象 | 5/10，还停留在 OpenAI ChatCompletion 伪通用层 |
| Tool / Runtime Primitive | 6/10，有想法，但语义分裂 |
| 类型系统与 Python 品味 | 4/10，`Any` 和 dict 海洋太宽 |
| 可维护性 | 6/10，测试救了很多，但模块体积和责任混杂拖后腿 |

---

## 1. 当前架构概览

从源码与文档看，项目真实架构大致是：

```text
L1 Decorator
  @llm_function / @llm_chat / @tool
  Python 签名 + DocString + kwargs -> InvocationSpec / Tool schema

L2 Compile Boundary
  InvocationSpec + transcript + mutations + selfref snapshot
  -> provider-facing messages

L3 ReAct Runtime
  event-only async generator
  LLM phase -> tool batch -> mutation collection -> compile -> loop

L4 Provider Adapter
  OpenAICompatible / OpenAIResponsesCompatible
  对外统一为 ChatCompletion / ChatCompletionChunk 风格

L5 Runtime / Builtins / Infra
  PyRepl, FileToolset, SelfReference, PrimitiveRegistry, TUI, logger, Langfuse
```

项目最核心的正交性在这里：

- `InvocationSpec` 描述一次 Python 调用的语义，而不是 provider 请求。
- `compile_invocation_turn()` 是正式的转换边界。
- `ContextMutation` 是上下文变化的唯一正式货币。
- `run_react_loop()` 是 event-only 的核心循环。
- `SelfRefSession` 通过 hook 和 mutation 进入核心，而不是直接改 live transcript。

这些方向是正确的，而且明显是经过痛苦重构后才长出来的。

---

## 2. 值得肯定的架构设计

### 2.1 `InvocationSpec` 是一个有品味的边界

`llm_decorator/invocation_spec.py` 把 Python 调用语义封装成：

- `ParameterContract`
- `ReturnContract`
- `PromptContract`
- `TranscriptSeed`
- `InvocationSpec`

这件事很重要。很多轻量 LLM 框架一上来就在 decorator 里拼 provider messages，然后到处散落 prompt 字符串，最后任何改动都像在改祖传 Excel 宏。这里至少知道：

> decorator 层不应该知道 provider wire format。

这是高级的。

### 2.2 Compile Pipeline 是全项目最强的架构资产

`base/compile_pipeline.py` 和 `base/context_compile.py` 建立了一个明确模型：

```text
transcript + pending_mutations + selfref_snapshot
  -> ReducedTurnContext
  -> CompiledTurnContext
  -> llm_messages
```

它的价值在于：

1. 上下文演化可测试。
2. SelfRef、工具结果、abort、summary、experience 都通过统一 mutation 进入。
3. provider-facing prompt 注入是 ephemeral 的，不污染 durable context。
4. history 与 system prompt 终于不是随便拼字符串。

这比很多所谓“Agent Framework”只维护一个 `messages: list[dict]` 然后到处 append 要好太多。

### 2.3 ReAct Core 拆分方向正确

当前 `base/react_loop.py`、`base/llm_call.py`、`base/tool_scheduler.py`、`base/context_compile.py` 的分工基本成立：

- `llm_call.py` 做单次模型调用与流式解析。
- `tool_scheduler.py` 做并发工具调度。
- `react_loop.py` 做 orchestration。
- `context_compile.py` 做 context state transition。

这比老式 monolithic `ReAct.py` 明显强。`base/ReAct.py` 只保留兼容入口的方向也对。

### 2.4 Event-only Core 是正确选择

`ReactOutput = ResponseYield | EventYield`，核心始终产出事件流，而不是维护事件 / 非事件双模式，这个判断很好。

Agent 框架如果一开始就允许“有时 stream，有时 return，有时 callback，有时 event”，后面基本必然腐烂。当前核心统一事件面，decorator 再派生便捷 API，是正确方向。

### 2.5 SelfRef 通过 hook + mutation 进入核心，方向正确

`SelfRefSession.collect_context_mutations()` 把 pending compaction / remember / forget 转为 mutation，而不是让 primitive 在 ReAct loop 中间直接改 messages。这是全项目里最有架构纪律的一部分。

尤其是：

- compaction 延迟到 compile boundary 生效；
- active turn 与 outside-turn 的行为分开；
- fork 继承 pre-fork snapshot，避免 pending assistant tool-call 泄漏给 child；

这些不是 toy framework 会考虑的事。

### 2.6 测试在守架构，而不只是守功能

`tests/test_stage2_unified_invocation.py` 明确检查：

- function/chat 都走 compile pipeline；
- decorator 不再 import steps layer；
- event stream 是唯一 chat runtime surface；
- 不保留 ceremony-only wrapper modules。

这是非常好的信号。项目作者知道架构会退化，并且用测试卡住退化路径。

---

## 3. 主要架构问题

### 3.1 `llm_chat_decorator.py` 是目前最不优雅的核心入口

`SimpleLLMFunc/llm_decorator/llm_chat_decorator.py` 超过 1000 行。它同时负责：

- 参数签名解析后的业务解释；
- SelfReference 发现、绑定、key 解析；
- fork toolkit clone；
- prompt block 清洗；
- history authority 决策；
- Langfuse span；
- compile source 构造；
- ReAct loop 调用；
- ReactEndEvent 上 finalize selfref；
- contextvar reset 和异常兜底；
- backward-compatible tuple output 适配。

这不是 decorator，这是一个穿着 decorator 外套的应用服务器。

问题不在于它长；问题在于它占据了太多**生命周期编排权**。Decorator 层应该只做：

```text
Python call -> InvocationSpec + ExecutionPlan
```

最多再调用 runner。现在它实际上还知道 SelfRef 内部 store 细节、fork PyRepl clone 细节、prompt block marker、history merge 策略。这导致架构上的“单一 compile boundary”虽然存在，但上层还是有大量旁路知识。

建议拆成：

```text
llm_decorator/
  llm_chat_decorator.py       # 只保留 public wrapper
  chat_call_context.py        # bind args/template/runtime toolkit/selfref key
  selfref_binding.py          # resolve/bind/seed/finalize selfref
  fork_toolkit.py             # PyRepl clone 策略
  chat_runner.py              # InvocationSpec -> ReAct loop -> output stream
```

不要为了“文件少”牺牲认知边界。现在这个文件是项目里最明显的维护雷区。

### 3.2 `SelfReference` 是一个 God Object

`runtime/selfref/state.py` 约 1800 行，包含：

- memory store；
- context parsing / canonicalization 代理；
- active state contextvars；
- mutation queue；
- history CRUD；
- durable source snapshot；
- fork spawn / gather / event forwarding；
- agent instance binding；
- runtime backend lifecycle。

这已经不是 SelfReference 了，这是 SelfReference 操作系统。

它的问题是：能力都合理，但边界不合理。尤其危险的是它同时维护：

```python
_history_store
_source_store
_active_react_states_by_key
_pending_compactions
_pending_context_mutations
_fork_tasks
_fork_results
```

这些状态之间有隐含一致性协议。现在靠大量测试和作者脑内模型维持。长期看会变成“只要改一个 fork 行为就要读半个项目”。

建议拆为：

```text
runtime/selfref/
  store.py          # history/source durable store
  active_turn.py    # contextvars + active ReAct state lookup
  mutations.py      # pending compaction/remember/forget queues
  memory_api.py     # memory handle/proxy CRUD
  fork_manager.py   # fork/spawn/gather/event forwarding
  state.py          # facade only
```

Facade 可以保留 public API，但内部必须拆。否则 SelfRef 一定会成为未来所有架构债的垃圾桶。

### 3.3 `PyRepl` 同样过度集中

`builtin/pyrepl.py` 约 1670 行，负责：

- subprocess 生命周期；
- multiprocessing queue；
- IPython worker RPC；
- runtime primitive registry；
- built-in primitive registration；
- selfref pack 安装；
- tool schema 创建；
- event streaming；
- audit log；
- input() 协议；
- timeout/interruption；
- fork clone。

PyRepl 是很强的功能，但架构上显然应该拆：

```text
builtin/pyrepl/
  repl.py               # public facade
  worker_client.py      # process/queue lifecycle
  primitive_host.py     # primitive registry + RPC handling
  tools.py              # execute/reset tool creation
  audit.py              # audit log
  input_bridge.py       # input() request/reply
```

现在 `PyRepl` 和 `SelfReference` 互相知道对方太多细节：`PyRepl.DEFAULT_SELF_REFERENCE_BACKEND_NAME`、fork clone backend overrides、runtime pack 安装等。这种耦合会限制后续把 runtime primitives 接到非 PyRepl worker 的可能性。

### 3.4 Provider 抽象仍然是“OpenAI ChatCompletion 中心主义”

表面上有 `LLM_Interface`，实际上内部核心大量依赖 OpenAI SDK 对象语义：

- `ChatCompletion`
- `ChatCompletionChunk`
- `choices[0].message.content`
- `choices[0].message.tool_calls`
- `delta.tool_calls`
- `CompletionUsage`

`OpenAIResponsesCompatible` 并不是让核心理解 Responses API，而是把 Responses 强行伪装回 ChatCompletion。这是短期兼容上合理，长期架构上偏脆。

更干净的设计应该有框架自己的 provider-neutral 类型：

```python
@dataclass
class ModelMessageDelta: ...

@dataclass
class ModelResponse:
    content: str
    tool_calls: list[ToolCall]
    reasoning: list[ReasoningPart]
    usage: Usage | None
    raw: Any | None = None
```

Provider adapter 应该输出 framework-native `ModelResponse / ModelStreamEvent`，OpenAI SDK 对象只留在 adapter 内部。现在 `base/post_process.py`、`base/tool_call/extraction.py`、`llm_call.py` 都被 OpenAI object shape 牵着走。

这会导致：每支持一个非 OpenAI 风格 provider，就要写一层“伪 OpenAI 化”的兼容胶水。短期省事，长期不优雅。

### 3.5 类型系统没有撑起架构复杂度

项目宣称 typed，且有 `py.typed`，但内部真实情况是：

- 大量 `Dict[str, Any]`；
- `MessageList` 本质是 dict union；
- `ContextMutation` 是 dataclass union，但没有 discriminated tag；
- tool call、provider response、runtime primitive payload 到处 cast；
- `rg` 下 `Any / cast / type: ignore` 数量很高；
- `LLM_Interface.chat()` 默认参数还使用 mutable list literal。

`dict[str, Any]` 并不是罪；Agent 框架需要处理开放协议。但开放协议应该被困在边界，不应该在核心到处自由流动。

最该加强类型的地方：

1. Message model：内部 normalized message 应该是 dataclass 或 Pydantic model，而不是裸 dict。
2. ToolCall model：工具调用提取后应尽快转为 framework-native object。
3. Mutation model：建议加 `kind: Literal[...]` 或使用 Pydantic discriminated union，方便序列化、调试和测试。
4. Provider response model：不要让 OpenAI SDK 类型穿透核心。
5. Hook protocol：`hooks: Any` 应该变成明确 `Protocol`。

现在的类型系统像一件高级西装外套，里面穿的是睡衣。

### 3.6 `llm_function` 的结构化输出策略太旧派

当前复杂返回类型主要依赖 XML prompt + XML parse：

- `build_return_type_description()` 生成 XML schema 与 example；
- LLM 被要求输出 well-formed XML；
- `post_process.py` 再转 dict / list / Pydantic。

这个设计与“Prompt as Code”哲学不冲突，但作为现代 Agent Framework 架构，它有几个问题：

1. XML parser 是脆弱 fallback，不是强约束。
2. 已经有 tool/function schema 与 provider structured outputs，却没有形成统一 output schema 通道。
3. `llm_function` 与 provider adapter 的结构化能力割裂。
4. Pydantic schema -> XML schema -> XML -> dict -> Pydantic，这条链路过长。

更合理的是抽象出：

```python
class OutputAdapter:
    def build_prompt_contract(...)
    def build_provider_response_format(...)
    def parse(raw_response) -> T
```

然后按 provider capability 选择：

- JSON Schema / response_format；
- tool-call-as-output；
- XML fallback；
- plain text。

目前 XML 作为默认复杂输出通道，显得有点“2023 年的聪明”。

### 3.7 Tool 与 Runtime Primitive 的双系统语义有点乱

项目里有两套“模型可调用能力”：

1. `@tool` / OpenAI function calling：模型直接 tool_calls。
2. Runtime Primitive：模型在 PyRepl 里写 `runtime.selfref.context.inspect()`。

这个设计有合理性：primitive 是 host-registered callable，不是 LLM 原生 tool；它允许 REPL 内组合。但架构上现在靠 prompt block 解释二者区别：

> Runtime primitives are not standalone tool calls; call them inside execute_code.

这说明边界在用户心智里还不够清晰。工具和 primitive 都是 capability，但生命周期、调用通道、安全模型、返回约定、文档生成方式都不同。

建议引入统一的 Capability Registry：

```text
Capability
  - ToolCapability: visible as native model tool
  - RuntimePrimitiveCapability: visible inside worker runtime
  - PromptGuidanceCapability: only injects guidance
```

这样 `tool_prompt_specs`、runtime primitive prompt injection、best practices、must principles 都能从同一个 capability graph 生成，而不是现在散落在 Tool、PyRepl、PromptContract、llm_chat_decorator 的不同角落。

### 3.8 “编译边界唯一”还没有完全落实到外围

核心文档强调：所有 context changes 必须通过 mutation boundary。核心 ReAct 基本做到了。但外围仍有不少直接修改：

- `SelfReference.memory[...]` CRUD 可直接改 history。
- active turn 下 `_mutate_messages()` 仍会直接改 active state messages。
- `llm_chat_decorator.py` 在 ReactEndEvent 上手动 finalize 与替换 history。
- `SelfReference.commit_pending_compaction()` 仍可直接 set context messages。

这些可能是兼容成本，但架构上要明确标记为“escape hatch”。否则新贡献者会误以为直接改 store 是正常路径。

建议：

1. 把 direct mutation API 命名上显式标注为 admin/direct，例如 `memory_admin`。
2. active ReAct turn 内禁止直接 `_mutate_messages()`，统一 queue mutation；至少默认禁止，提供 explicit unsafe API。
3. 所有 finalize 进入 `SelfRefSession.finalize()`，不要 decorator 手写一份 `_finalize_self_reference_history()`。

现在的风险是：核心刚建立起来的边界，会被外围的便利 API 慢慢侵蚀。

### 3.9 文档和代码存在旧架构残影

`spec/project-map.md` 仍提到：

- `llm_decorator/steps/...`
- `base/ReAct.py` 是核心实现
- Event Stream 使用 `enable_event=True`

但当前代码与 Stage 2 测试已经明确：steps package 不存在，core event-only，`enable_event` 已移除。

这不是小问题。架构文档如果落后，会让 Agent 和新人按照旧世界修改代码。对于这个项目尤其危险，因为它本身就是给 Agent 用的框架，文档漂移会直接变成错误 patch。

建议立即更新 `spec/project-map.md` 和 `spec/overall-spec.md`，让它们与 README / tests / current core 对齐。

### 3.10 `__init__.py` 的 import-star 和副作用不够克制

顶层 `SimpleLLMFunc/__init__.py`：

```python
from rich import traceback
traceback.install(show_locals=True)

from SimpleLLMFunc.config import *
from SimpleLLMFunc.llm_decorator import *
...
```

一个库在 import 时安装 rich traceback，并且 `show_locals=True`，这从框架品味上讲非常粗鲁。应用可以这么干，库不该默认这么干。它可能泄漏局部变量、影响宿主程序异常展示、污染全局行为。

建议：

- 去掉 import-time traceback install；
- 提供 `SimpleLLMFunc.enable_rich_traceback()`；
- 顶层导出显式 `__all__`，不要星号 import 全家桶。

这是 Python 库的基本礼貌。

---

## 4. 分层逐项评价

### 4.1 Decorator Layer

优点：

- Callable object 替代纯 closure，是正确方向；SelfRef 需要稳定 agent identity。
- `InvocationSpec` builder 把 decorator 从 provider messages 中解放出来。
- `llm_function.stream()` 与 `llm_chat()` event-only 输出有统一趋势。

问题：

- `LLMChat.__call__` 太重。
- `llm_chat` 知道 SelfRef、PyRepl fork clone、Langfuse、history merge 太多。
- strict signature 是可选的，默认宽松导致 runtime 行为隐含。
- `parse_function_signature()` 直接 pop `_template_params`，这种 magic kwarg 缺少类型层表达。

建议：

- 引入 `CallContext` 对象，统一承载 bound args、template params、trace、runtime toolkit、selfref session。
- 将 SelfRef 绑定逻辑从 decorator 移出。
- `_template_params`、abort signal 这类 meta kwargs 应统一进入 `InvocationOptions`，而不是散落 pop。

### 4.2 Compile / Context Layer

优点：

- Mutation boundary 很好。
- `compile_invocation_turn()` 是正确中心。
- prompt injection ephemeral 化是高质量设计。
- `validate_tool_linkage()` 在 mutation apply 后执行，方向正确。

问题：

- `apply_mutations()` 中夹带 SelfRef experience 解析 / 重建逻辑，核心 compile 对 selfref marker 知道太多。
- Mutation 缺少显式版本与 kind，不利于未来持久化 / replay / trace。
- `ContextSummaryMutation` 的 summary message 是 dict，不是 typed object。

建议：

- 把 selfref-specific mutation application 抽成 handler registry 或专门 reducer。
- Mutation 加 `kind` 与 schema version。
- 让 `ContextState` 成为不可变值对象，减少 list copy / mutation 的不确定性。

### 4.3 ReAct Runtime

优点：

- event-only core 是正确选择。
- 单 LLM phase、tool scheduler、loop orchestration 的拆分基本合理。
- abort 作为 mutation-producing behavior 很好。
- 工具并发调度有明确事件。

问题：

- `react_loop.py` 仍然 600 行，承担 Langfuse、final LLM call、max tool cap、hook、compile、event bus 全部编排。
- max_tool_calls 达上限时会触发一次无工具 final LLM call，这个策略硬编码在 core；它可能应该是 policy。
- `hooks: Any` 没有 protocol。
- `compile_invocation_turn()` 在部分路径重复调用，且 invocation_spec fallback 构造逻辑仍在 react_loop 中。

建议：

- 抽出 `FinalizePolicy`、`ToolCapPolicy`、`ObservationSink`。
- `ReActHook` 定义 Protocol。
- `run_react_loop()` 接收已构造的 `ReActRunSpec`，避免参数列表无限膨胀。

### 4.4 Provider Interface

优点：

- API key pool、token bucket、OpenAI-compatible 入口实用。
- Responses adapter 能保持上层不感知 Responses API，短期工程价值高。
- per-model `api_params` 合理。

问题：

- 抽象层不是 provider-neutral，而是 OpenAI-neutral-looking。
- `LLM_Interface` 默认参数使用 list literal，这是 Python 基础坏味道。
- token accounting 通过 context attribute 间接传递，不够显式。
- adapter 负责 retry、rate limit、request build、response normalize、token update、logging，职责偏多。

建议：

- 定义 framework-native `ModelResponse` / `ModelStreamEvent`。
- Provider adapter 只负责 wire conversion，retry/rate limit 抽为 transport middleware。
- 消除 mutable default args。

### 4.5 Tool System

优点：

- `@tool` 要求 async，避免同步阻塞，是对的。
- tool schema 从签名生成，符合 Pythonic 使用体验。
- `too_long_to_file` 是非常实际的 Agent 工程设计。
- FileToolset 的 stale-write protection 很好。

问题：

- Tool schema 生成对复杂 typing 支持有限，且 fallback 到 string 太安静。
- `event_emitter` 作为隐藏参数的处理需要更明确的 schema policy。
- Tool 返回多模态时会转成 UserMessageMutation，这个语义不够直观，需要更清晰的 capability/attachment 模型。

建议：

- 使用 Pydantic TypeAdapter / JSON schema 统一参数 schema。
- hidden injected params 标准化，例如 `Injected[T]`。
- Tool result 建立 typed result envelope：text / json / multimodal / artifact。

### 4.6 Runtime Primitive / PyRepl

优点：

- primitive 与 Tool 分开有现实意义：REPL 内组合能力很强。
- primitive docstring 强制 Best Practices，说明作者理解 LLM-facing API 需要操作手册。
- Worker proxy 的 `runtime.namespace.name()` 体验很好。

问题：

- primitive contract 又是一套 parser、XML renderer、best practices 机制，和 Tool prompt specs 重复。
- `PrimitiveRegistry` 太大，既做 schema、docstring parser、registration、execution、error enhancement、XML serialization。
- PyRepl 是另一个 God Object。

建议：

- 把 primitive contract parser / renderer 独立出来。
- 统一 Tool 与 Primitive 的 capability metadata。
- PyRepl 拆 facade/client/host/tools/audit。

### 4.7 TUI / Infra

本次主要审查 core，没有完整展开 TUI。但从目录看，`utils/tui/app.py` 与 `utils/tui/core.py` 体量也偏大。TUI 作为 presentation layer，应该尽量只消费 event stream，不知道 ReAct 内部策略。建议后续单独审查。

---

## 5. 关键风险清单

### P0：架构文档漂移

`spec/` 仍残留旧 steps / enable_event 叙述。这个必须修，因为项目显然希望 Agent 按文档协助开发。

### P1：Decorator 与 SelfRef God Object 拖垮可维护性

`llm_chat_decorator.py`、`SelfReference`、`PyRepl` 是三个最需要拆的文件。它们不是“代码长”这么简单，而是责任边界已经混杂。

### P1：Provider abstraction 迟早顶不住多模型生态

只要继续以 ChatCompletion 为内部标准，支持 Responses、Anthropic、Gemini 原生特性都会越来越像变魔术。

### P2：类型边界不足导致复杂行为不可推理

Agent Framework 最怕隐藏状态 + dict Any。当前核心的哲学强调可编译上下文，但类型系统没有充分保护这个模型。

### P2：Structured output 体系需要现代化

XML fallback 可以保留，但不应该是复杂返回类型的一等默认架构中心。

---

## 6. 建议路线图

### 第一阶段：止血与对齐（1-3 天）

1. 更新 `spec/project-map.md`、`spec/overall-spec.md`：移除 steps / enable_event 旧叙述。
2. 顶层 `__init__.py` 去掉 import-time rich traceback 副作用。
3. 为 hook 定义 `Protocol`，替换核心路径的 `hooks: Any`。
4. 为 `ContextMutation` 加 `kind` 或至少文档化每个 mutation 的 reducer 行为。
5. 修掉 `LLM_Interface` / adapter 中 mutable default args。

### 第二阶段：拆最大 God Objects（1-2 周）

1. 拆 `llm_chat_decorator.py`：
   - `chat_call_context.py`
   - `selfref_binding.py`
   - `fork_toolkit.py`
   - `chat_runner.py`
2. 拆 `SelfReference`：
   - store / active_turn / mutation_queue / fork_manager / memory_api。
3. 拆 `PyRepl`：
   - worker client / primitive host / tool factory / audit / input bridge。

拆的时候不要改 public API，用 façade 保持兼容。

### 第三阶段：建立 framework-native model I/O（2-4 周）

1. 定义：
   - `ModelRequest`
   - `ModelResponse`
   - `ModelStreamEvent`
   - `Usage`
   - `ToolCallDelta`
2. Provider adapter 输出 native 类型。
3. OpenAI SDK 对象只留在 adapter 与 debug raw 字段。
4. `llm_call.py` 从 OpenAI shape extraction 中解放出来。

### 第四阶段：统一 Capability 与 Output Schema（长期）

1. Tool / Primitive / PromptGuidance 统一 metadata。
2. 结构化输出抽象为 OutputAdapter。
3. 优先使用 provider-native structured output，XML 成为 fallback。
4. Tool result 建立 typed envelope，支持 artifact/multimodal/file reference。

---

## 7. 最终评价

SimpleLLMFunc 当前不是一个“轻量但随便”的项目。它有明确的核心架构追求，而且最重要的方向——`InvocationSpec`、compile boundary、mutation-driven context、event-only ReAct——都是对的。

但它也还不是一个真正干净的 Agent Framework。它更像一个非常聪明的系统，在经历重构后，核心骨架已经长正了，但旧肉还没割干净，新器官也长得有点野：Decorator 太胖，SelfRef 太神，PyRepl 太全能，Provider 抽象太 OpenAI，类型边界太松。

毒辣一点说：

> 这个框架最好的部分像是一个有洁癖的架构师写的；最差的部分像是同一个人凌晨三点为了让 demo 跑起来写的。

不过这并不是坏消息。真正糟糕的框架是没有可挽救的中心思想；SimpleLLMFunc 有。下一步不是推倒重来，而是让已经正确的 compile/mutation/event 核心成为全项目的法律，而不是只在 `base/` 目录里当模范市民。
