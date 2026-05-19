# Stage 2 Simplify Plan：少抽象、少包装、少步骤层

<!-- DOC_SUMMARY_START -->
本文档补充 `stage-2-unified-invocation.md`：Stage 2 的目标不是继续增加架构层，而是在保留 InvocationSpec、唯一 compile boundary、SelfRefSession 三个必要边界的前提下，删除历史 step 层和 wrapper-only 模块，降低系统总体代码量与参数转发复杂度。
<!-- DOC_SUMMARY_END -->

## Status

- Scope: `SimpleLLMFunc/llm_decorator/*`, `SimpleLLMFunc/base/compile_pipeline.py`, `runtime/selfref/session.py`
- Goal: simplify Stage 2 implementation after introducing unified invocation / compile boundary
- Principle: **boundary clarity without ceremony**

---

## 北极星

Stage 2 的简化方向是：

> 不再为了“每一层都有自己的表达”创建只做 wrap / parameter remap 的函数或模块。  
> 只保留真正承担语义、状态、生命周期、规则的抽象。

最终调用路径应接近：

```python
sig, template_params = parse_function_signature(...)
spec = build_invocation_spec(...)
tools, tool_map = process_tools(...)
async for output in ReAct_loop(..., invocation_spec=spec):
    yield_or_adapt(output)
```

而不是：

```python
decorator -> builder -> attachments -> runner -> adapter -> steps/react -> ReAct_loop
```

---

## 保留哪些抽象

只保留三类必要边界。

### 1. InvocationSpec / invocation_builder

保留原因：
- 它表达 Python 调用语义
- 它统一 `llm_function` / `llm_chat` 的输入建模
- 它不是 provider-facing request

保留文件：
- `SimpleLLMFunc/llm_decorator/invocation_spec.py`
- `SimpleLLMFunc/llm_decorator/invocation_builder.py`

允许职责：
- signature 解析后的语义建模
- prompt contract 建模
- transcript seed 建模
- function/chat 差异建模

禁止职责：
- 调 LLM
- 执行 ReAct loop
- finalize selfref
- provider-facing message rendering

---

### 2. compile_pipeline

保留原因：
- 它是唯一正式 convert-to-LLM boundary
- 它承担 mutation application + semantic state -> llm messages
- 这是 Stage 2 最核心的架构收益

保留文件：
- `SimpleLLMFunc/base/compile_pipeline.py`

唯一主入口：

```python
compile_invocation_turn(...)
```

允许职责：
- apply pending mutations
- 合并 selfref snapshot
- 注入 tool prompt specs
- 注入 must principles
- 生成 provider-safe `llm_messages`

禁止职责：
- 调 LLM
- 执行 tool
- 同步 external history
- 持久化 selfref store

---

### 3. SelfRefSession

保留原因：
- selfref 的确有 invocation-scoped 生命周期
- active memory key / pending mutation / finalize / contextvar reset 不应留在 decorator 主体中

保留文件：
- `SimpleLLMFunc/runtime/selfref/session.py`

允许职责：
- 设置/恢复 active memory key
- collect pending selfref mutations
- snapshot selfref source
- finalize selfref history
- 清理 contextvar token

禁止职责：
- 变成通用 plugin framework
- 变成 attachment resolver
- 变成 fork service 总入口
- 只是包装 `SelfReference` 方法名

---

## 删除或弱化哪些抽象

### 1. 删除 wrapper-only 模块

这些模块如果只做参数转发或轻量 remap，应删除，不作为 Stage 2 正式层。

已删除或禁止重建：

- `SimpleLLMFunc/llm_decorator/attachments.py`
- `SimpleLLMFunc/llm_decorator/invocation_runner.py`
- `SimpleLLMFunc/llm_decorator/result_adapters.py`
- `SimpleLLMFunc/runtime/plugins/protocol.py`
- `SimpleLLMFunc/runtime/plugins/__init__.py`

对应 guard：

```python
tests/test_stage2_unified_invocation.py::test_stage2_does_not_keep_wrapper_only_modules
```

规则：

> 如果一个模块的主要价值只是把参数从 A 名字改成 B 名字再传下去，它不应该存在。

---

### 2. 弱化 / 删除 `llm_decorator/steps/*`

当前最重、最容易制造冗余的是 step 层。

它的问题：
- 名义上拆分 decorator，实际上形成一串 procedural wrappers
- `function/chat` 双轨路径被 step 层固化
- 很多函数只做参数 remap、tuple/event 兼容、轻量调用转发
- 和新的 InvocationSpec / compile_pipeline 重叠

目标：

```text
llm_decorator/steps/* 最终不是 Stage 2 正式架构层
```

---

## Step 层处理计划

## Phase S1：砍 react step wrapper

目标文件：
- `SimpleLLMFunc/llm_decorator/steps/function/react.py`
- `SimpleLLMFunc/llm_decorator/steps/chat/react.py`

当前职责：
- process tools
- 调 `ReAct_loop`
- tuple/event 兼容转换
- retry
- response collection

问题：
- 这些职责不构成独立边界
- 它让路径变成 decorator -> steps/react -> ReAct_loop -> core

目标形态：

decorator 直接：

```python
tools, tool_map = process_tools(toolkit, spec.func_name)
react_stream = ReAct_loop(
    llm_interface=llm_interface,
    messages=spec.transcript_seed.initial_messages,
    tools=tools,
    tool_map=tool_map,
    invocation_spec=spec,
    ...
)
```

保留策略：
- 初期可以保留 thin compatibility function 给旧 tests
- decorator 主路径不再依赖它
- 后续删除或移动到 tests-compatible shim

退出标准：
- `llm_function_decorator.py` 不 import `steps.function.execute_react_loop`
- `llm_chat_decorator.py` 不 import `steps.chat.execute_react_loop_streaming`
- function/chat 都直接调用 `ReAct_loop(..., invocation_spec=spec)`

测试：
- function/chat compile single-entry test
- existing react step tests 改成 core-level tests 或删除重复覆盖

---

## Phase S2：砍 response step wrapper

目标文件：
- `SimpleLLMFunc/llm_decorator/steps/function/response.py`
- `SimpleLLMFunc/llm_decorator/steps/chat/response.py`

当前职责：
- function: `process_response(...)` 薄包装
- chat: text/raw extraction + stream end marker

问题：
- function response wrapper 几乎没有独立价值
- chat response 可以作为 decorator 最外层 compatibility 逻辑，不需要 step 层

目标形态：
- function typed parse 直接调用 `process_response(...)`
- chat text/raw adaptation 内联在 decorator 或收敛到 `base/post_process.py`

退出标准：
- no import from `steps.function.response`
- no import from `steps.chat.response`
- tests 直接覆盖 public decorator 行为，不测试 thin wrapper 本身

---

## Phase S3：把 prompt/message step 改成 seed/contract helper

目标文件：
- `SimpleLLMFunc/llm_decorator/steps/function/prompt.py`
- `SimpleLLMFunc/llm_decorator/steps/chat/message.py`

当前问题：
- function prompt step 还叫 `build_initial_prompts`，实际会构造 messages
- chat message step 还提供 `build_compile_source_for_chat`
- 这些名字和职责与 Stage 2 compile boundary 冲突

目标形态：

将其收敛到 `invocation_builder.py` 或少量私有 helper：

```python
_build_function_prompt_contract(...)
_build_function_transcript_seed(...)
_build_chat_transcript_seed(...)
_extract_external_history_ref(...)
```

禁止继续使用：

```python
build_initial_prompts(...)
build_compile_source_for_chat(...)
```

退出标准：
- `llm_function_decorator.py` 不 import `build_initial_prompts`
- `llm_chat_decorator.py` 不 import `build_chat_messages` / `build_compile_source_for_chat`
- `build_compile_source_for_chat` 删除或只作为 deprecated test shim

---

## Phase S4：重新组织 common/signature

当前文件：
- `SimpleLLMFunc/llm_decorator/steps/common/signature.py`
- `SimpleLLMFunc/llm_decorator/steps/common/types.py`
- `SimpleLLMFunc/llm_decorator/steps/common/prompt.py`

判断：
- signature parsing 是 decorator boundary 的真实职责，可以保留
- 但不需要放在 `steps/common`

目标形态：

```text
SimpleLLMFunc/llm_decorator/signature.py
SimpleLLMFunc/llm_decorator/prompt_contract.py   # optional, only if enough logic remains
```

退出标准：
- `steps/common` 不再作为 public internal layer
- `FunctionSignature` 可迁移为 dataclass 或保留轻量 tuple，但位置不再叫 steps

---

## Phase S5：删除 step package

最终目标：

```text
SimpleLLMFunc/llm_decorator/steps/  # gone or only compatibility shims during transition
```

最终结构倾向：

```text
SimpleLLMFunc/llm_decorator/
  llm_function_decorator.py
  llm_chat_decorator.py
  invocation_spec.py
  invocation_builder.py
  signature.py
```

如果某些 helper 仍然很大，允许拆出，但必须满足：
- 不是 wrapper-only
- 有独立规则
- 不只是参数改名
- 不重复 compile boundary

---

## TDD / Guard 策略

每个 simplify phase 先加 guard，再删代码。

### 已有 guard

1. function/chat 共用 compile boundary
2. no shared decorator result state
3. no wrapper-only modules

### 需要新增 guard

#### 1. decorator does not import react steps

```python
assert "steps.function.react" not in source
assert "steps.chat.react" not in source
```

#### 2. decorator does not import response steps

```python
assert "steps.function.response" not in source
assert "steps.chat.response" not in source
```

#### 3. no compile source builder in chat step

```python
assert not hasattr(chat_message_module, "build_compile_source_for_chat")
```

#### 4. no provider-facing construction in decorator

Decorator source should not contain:
- `render_llm_input_messages`
- direct `llm_messages =`
- direct provider request construction

#### 5. public behavior remains stable

Keep existing integration tests for:
- `llm_function` typed result
- `llm_chat` text/raw/event mode
- selfref remember/forget/compact
- fork inheritance
- history sync

---

## 代码量指标

Simplify 不只看文件数量，也看路径深度。

### 指标 1：decorator call depth

目标从：

```text
decorator -> step prompt/message -> step react -> ReAct wrapper -> run_react_loop -> compile helper
```

降到：

```text
decorator -> invocation_builder -> ReAct_loop -> run_react_loop -> compile_pipeline
```

### 指标 2：wrapper-only module count

目标：0。

### 指标 3：steps package LOC

目标：逐阶段下降，最终删除或只剩迁移 shim。

### 指标 4：parameter remap count

减少此类函数：

```python
def foo(a, b, c):
    return bar(x=a, y=b, z=c)
```

除非它承担兼容或验证规则。

---

## 非目标

Simplify 阶段不做：

1. 不改公开 decorator 名称
2. 不改 LLM interface / provider adapter
3. 不改 ReAct core event-only 方向
4. 不重写 selfref durable backend
5. 不为了删除文件破坏已有 public behavior

---

## 最终验收标准

Stage 2 simplify 完成时：

1. `llm_function` / `llm_chat` 都直接以 `InvocationSpec` 进入 runtime
2. 唯一 provider-facing message 生成入口是 `compile_invocation_turn`
3. `llm_decorator/steps/*` 不再是主路径依赖
4. wrapper-only modules 不存在
5. selfref session 只承担 invocation lifecycle，不变成通用插件框架
6. 全量测试通过
7. 代码路径更短，而不是只是文件名更“架构化”

---

## 一句话总结

> Stage 2 simplify 的目标不是把系统拆得更“分层”，而是把历史上为了拆 decorator 产生的 step ceremony 删除掉；只留下 InvocationSpec、compile_pipeline、SelfRefSession 这三个真正有规则密度的抽象。
