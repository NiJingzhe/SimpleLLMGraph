# PyRepl 代码执行

SimpleLLMFunc 提供内置的 PyRepl 支持，允许 LLM 在一个连续上下文中执行 Python 代码。与传统的一次性代码执行不同，PyRepl 保持变量和状态，让 LLM 可以分步执行复杂任务。

## 功能特性

- **IPython 子进程后端**：每个 `PyRepl` 实例对应一个独立子进程，内部运行 `IPython InteractiveShell`
- **连续上下文**：变量在多次调用间持久化，LLM 可以分步执行任务
- **实时流式输出**：通过 `event_emitter` 实时获取 stdout/stderr 输出，包括 Python `print()`、直接 fd 写入和子进程 stdout/stderr
- **图片产物返回**：`display(Image(...))` 或最后表达式产生的图片会被捕获为多模态工具返回
- **异步不阻塞**：代码执行在独立线程运行，不阻塞主事件循环（适合 TUI/流式 UI）
- **Session 隔离**：不同的 PyRepl 实例相互独立，互不影响
- **完整工具集**：提供 execute_code、reset_repl 等工具
- **超时保护**：单次 `execute_code` 默认 600 秒活动执行超时；等待 `input()` 不计时，且每次收到输入后会重置超时窗口；单次 `input()` 默认 300 秒空闲超时。也可在每次调用时通过 `timeout_seconds` 单独覆盖
- **长输出自动截断**：`execute_code` 已启用 `too_long_to_file`，当输出超过 20000 tokens 时自动保存到临时文件并截断返回
- **Runtime 原语**：通过 `runtime.selfref.context.*` 与 `runtime.selfref.fork.*` 暴露受控的 self-reference API
- **Origin 感知事件流**：每个 `EventYield` 都携带 `origin` 元数据，方便在 TUI 或自定义 UI 中稳定区分主链路和 fork 链路
- **终端隔离**：PyRepl worker 默认不继承宿主终端的 fd 0/1/2，并在 POSIX 上尽量脱离控制终端，避免子进程污染 TUI/daemon 宿主的终端状态

## 快速开始

### 基本用法

```python
import asyncio
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import PyRepl

# 创建 PyRepl 实例
repl = PyRepl()

# 获取工具集
tools = repl.toolset

# 使用 repl 工具创建聊天机器人
@llm_chat(
    llm_interface=llm,
    toolkit=tools,
)
async def python_assistant(message: str, history=None):
    """
    你是一个 Python 编程助手。
    用户会给你编程任务，你需要编写代码来完成。
    记住：变量会在后续调用中保持。
    """

# 使用
async for output in python_assistant("创建一个列表并计算均值"):
    # 处理输出
    pass
```

### 多 PyRepl 隔离

```python
# 创建两个独立的 repl
repl1 = PyRepl()
repl2 = PyRepl()

# 分别使用
@llm_chat(toolkit=repl1.toolset, ...)
async def chat1(message: str, history=None):
    """使用 repl1 的助手"""

@llm_chat(toolkit=repl2.toolset, ...)
async def chat2(message: str, history=None):
    """使用 repl2 的助手"""
```

## 工具详解

### execute_code

执行 Python 代码，返回执行结果。

> 说明：`execute_code` 默认有 600 秒活动执行超时保护。等待 `input()` 期间不会计入超时；每次 `input()` 成功回填后会重置超时计时。同时，单次 `input()` 默认 300 秒空闲超时。任一超时触发时 `success=False`，并在 `error`/`stderr` 中返回超时信息。你也可以在工具调用时传入 `timeout_seconds` 为该次执行单独设置超时。

> **长输出自动截断**：`execute_code` 已启用 `too_long_to_file` 功能。当代码输出超过 20000 tokens 时，完整结果会自动保存到临时文件（路径会在 `<system-reminder>` 中告知），返回内容会被截断到前 20000 tokens。这避免了超长输出导致的上下文溢出问题。

> 工具说明（发送给模型）的指引：请直接编写当前 REPL 会话可执行的顶层代码，支持 `input()`；先用 `runtime.list_primitives()` 发现原语，再用 `runtime.list_primitives(contains="<namespace>.")` 做命名空间过滤；查看契约时优先使用 `runtime.get_primitive_spec(name)`，需要批量查看时再用 `runtime.list_primitive_specs(names=[...], contains="...")`；契约默认返回 XML，需要在代码里直接读字段时使用 `format="dict"`；使用 `reset_repl` 可以清理 REPL 变量并继续保留当前 runtime backend 状态。

**参数：**

| 参数 | 类型 | 描述 |
|------|------|------|
| code | str | 要执行的 Python 代码 |
| timeout_seconds | float | 可选，单次调用活动执行超时（秒），不传则使用实例默认值 |
| event_emitter | ToolEventEmitter | 可选，事件发射器用于实时输出 |

**工具输出（面向模型）：**

返回自然语言摘要字符串，包含执行状态、耗时、stdout/stderr、return_value 与错误信息。
如需结构化数据，请直接调用 `PyRepl.execute()`。

**Python API 返回值：**

```python
{
    "success": bool,              # 是否成功执行
    "stdout": str,                # 标准输出
    "stderr": str,                # 标准错误
    "return_value": Any,          # 最后表达式的值
    "artifacts": list[dict],      # 图片等执行产物（execute_code 工具会转成多模态返回）
    "error": str | None,          # 错误信息（可直接定位到输入代码行）
    "error_details": dict | None, # 结构化错误详情（行号/列号/代码片段/指针等）
    "execution_time_ms": float    # 执行时间（毫秒）
}
```

### 图片输出

当代码输出图片时，作为 Agent 工具使用的 `execute_code` 会返回多模态工具结果：模型会看到执行摘要，也会看到图片内容。

推荐写法：

```python
from IPython.display import Image, display

display(Image(filename="chart.png"))
```

或把图片对象作为最后表达式：

```python
from IPython.display import Image

Image(filename="chart.png")
```

生成图表时，保存文件并显式返回 `ImgPath` 最清晰：

```python
import matplotlib.pyplot as plt
from SimpleLLMFunc.type import ImgPath

plt.plot([1, 2, 3])
path = "chart.png"
plt.savefig(path)
ImgPath(path)
```

直接调用 `await repl.execute(...)` 时，图片会出现在返回 dict 的 `artifacts` 列表中；经由 `execute_code` 工具调用时，框架会自动把 artifact 转成 `ImgPath` / `ImgUrl` 多模态返回。

### 错误定位增强

`execute_code` 会尽量直接返回输入代码的定位信息，而不是仅显示框架内部 `exec` 栈。

典型字段（在 `error_details` 中）：

- `error_type`: 异常类型（如 `SyntaxError`、`ZeroDivisionError`）
- `message`: 异常原始消息
- `line` / `column`: 出错行列（若可解析）
- `snippet`: 出错行源码
- `pointer`: 列指针（例如 `^`）
- `summary`: 面向模型/用户的简洁可读报错摘要
- `user_traceback`: 聚焦用户代码栈的 traceback 文本

示例：

```python
repl = PyRepl()
result = await repl.execute(code="for i in range(2)\n    print(i)")
if not result["success"]:
    print(result["error"])
    print(result["error_details"])
```

### reset_repl

重置 repl 状态，清除所有变量。

> 面向模型的工具描述为英文，明确说明：`reset_repl` 只清理 REPL 变量，保留已注册的 runtime backend。

```python
result = await repl.reset()
# 返回: "REPL 已重置，所有变量已清除"
```

## Streaming 事件

当消费 `ReactOutput` 事件流时，`execute_code` 会实时发射以下事件：

| 事件名 | data 字段 | 描述 |
|--------|-----------|------|
| `kernel_stdout` | `{text: str}` | 标准输出 |
| `kernel_stderr` | `{text: str}` | 标准错误 |
| `kernel_input_request` | `{request_id: str, prompt: str, idle_timeout_seconds: float}` | `input()` 请求用户输入（含本次输入空闲超时） |

### 捕获 Streaming 事件

```python
from SimpleLLMFunc.hooks import is_event_yield, CustomEvent

async for output in llm_chat_function(message):
    if is_event_yield(output):
        event = output.event
        if isinstance(event, CustomEvent):
            if event.event_name == "kernel_stdout":
                print(f"[stdout] {event.data['text']}", end="")
            elif event.event_name == "kernel_stderr":
                print(f"[stderr] {event.data['text']}", end="", file=sys.stderr)

# 说明：当 event_name == "kernel_input_request" 时，
# 你可以把用户输入通过 PyRepl.submit_input(request_id, value) 回填。
```

当你消费 `@llm_chat` 的 `ReactOutput` 事件流时，可直接使用 `output.origin` 区分主链路和 fork 链路：

```python
from SimpleLLMFunc.hooks import is_event_yield

async for output in data_helper("分叉执行任务"):
    if not is_event_yield(output):
        continue
    if output.origin.fork_id:
        print(f"fork={output.origin.fork_id} depth={output.origin.fork_depth}")
```

## 使用示例

### 数据分析助手

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.hooks import is_event_yield, CustomEvent
import sys

repl = PyRepl()

@llm_chat(
    llm_interface=llm,
    toolkit=repl.toolset,
)
async def data_helper(message: str, history=None):
    """
    你是一个数据分析助手。
    使用 Python 代码完成数据分析任务。
    每次只执行一小段代码，使用 print() 输出结果。
    """

# 执行任务
async for output in data_helper(
    "创建一个包含100个随机数的列表，计算均值和标准差"
):
    if is_event_yield(output):
        event = output.event
        if isinstance(event, CustomEvent):
            if event.event_name == "kernel_stdout":
                print(event.data['text'], end="")
```

### 连续编程上下文

```python
repl = PyRepl()

# 第一次调用：定义数据
result1 = await repl.execute(code="""
import random
data = [random.randint(1, 100) for _ in range(10)]
print(f"创建了 {len(data)} 个随机数")
print(f"数据: {data}")
""")

# 第二次调用：使用之前的数据
result2 = await repl.execute(code="""
mean = sum(data) / len(data)
print(f"均值: {mean}")
""")

# 变量 data 仍然可用！
print(result2['stdout'])  # "均值: 52.3"
```

## 配置选项

### PyRepl 构造函数参数

```python
# 默认活动执行超时为 600 秒
repl = PyRepl()

# 可按需调整活动执行超时（单位：秒）
repl = PyRepl(execution_timeout_seconds=180)

# 也可在单次调用中覆盖超时（单位：秒）
result = await repl.execute("import time\ntime.sleep(2)", timeout_seconds=5)

# 也可调整 input 空闲超时（单位：秒，默认 300）
repl = PyRepl(input_idle_timeout_seconds=300)

# 两者都可配置
repl = PyRepl(execution_timeout_seconds=180, input_idle_timeout_seconds=300)

# 设置初始工作目录（子进程启动后即生效）
repl = PyRepl(working_directory="./sandbox")
```

## 使用 SelfReference 后端的 Runtime 原语

`PyRepl()` 启动时会默认安装内置 `selfref` pack。

如果你需要在宿主侧预先写入或读取同一份记忆状态，可以直接获取这份默认 backend：

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.builtin import SelfReference

repl = PyRepl()
self_reference = repl.get_runtime_backend("selfref")
assert isinstance(self_reference, SelfReference)

@llm_chat(
    llm_interface=llm,
    toolkit=repl.toolset,
    self_reference_key="agent_main",
)
async def agent(message: str, history=None):
    ...
```

### 通用 runtime 后端 / 原语注册

`PyRepl` 也支持通用 runtime 扩展点，不仅限于 `selfref` 包：

若需要更完整的概念说明，见 [Primitive 原语](detailed_guide/primitive.md)。

其中 `pack(..., guidance="...")` 适合描述这一整包 runtime 能力的心智模型；具体 primitive 的细节仍然通过 `runtime.get_primitive_spec(name)` / `runtime.list_primitive_specs(...)` 查询。

对大多数自定义 runtime primitive 场景，推荐直接采用 `pack -> @pack.primitive -> install_pack` 这条路径；它会把 namespace、共享 backend、pack guidance 和安装生命周期放在同一个抽象里。

- `pack(name, backend=..., backend_name=None, guidance="")`
- `install_pack(pack, replace=False)`
- `@repl.primitive(name, backend="...")`
- `register_runtime_backend(name, backend, replace=False)`
- `register_primitive(name, handler, description="", backend_name=None, replace=False)`
- `register_primitive_pack_installer(pack_name, installer, replace=False)`
- `install_primitive_pack(pack_name, **options)`
- `list_runtime_backends()` and `list_primitives()`
- `list_installed_packs()`
- `get_primitive_contract(name)` / `list_primitive_contracts(...)`
- `runtime.get_primitive_spec(name)`（在 REPL 内）查看单个原语契约
- `runtime.list_primitives(contains="<namespace>.")`（在 REPL 内）按命名空间发现原语
- `runtime.list_primitive_specs(names=[...], contains="...")`（在 REPL 内）按条件过滤查看：名称、描述、输入/输出、参数与最佳实践

```python
class GitHubRepoAPI:
    def list_open_issues(self, repo: str) -> list[dict[str, str]]:
        # In production, call GitHub REST/GraphQL here.
        return [{"id": "42", "title": "Bug: tool timeout", "repo": repo}]


repl = PyRepl()

github_repo = repl.pack(
    "github_repo",
    backend=GitHubRepoAPI(),
    guidance="github_repo = repository issue/query primitives backed by GitHubRepoAPI.",
)

@github_repo.primitive(
    "list_open_issues",
    description="List open issues from a GitHub repository.",
)
def list_open_issues(ctx, repo: str) -> list[dict[str, str]]:
    backend = ctx.backend
    if not isinstance(backend, GitHubRepoAPI):
        raise RuntimeError("backend must be a GitHubRepoAPI")
    return backend.list_open_issues(repo)

repl.install_pack(github_repo)

await repl.execute('print(runtime.github_repo.list_open_issues("owner/repo"))')
```

若只是补一个轻量级原语，也可以直接使用装饰器糖：

```python
repo_backend = GitHubRepoAPI()

repl.register_runtime_backend("github_repo", repo_backend, replace=True)


@repl.primitive("github_repo.list_open_issues", backend="github_repo", replace=True)
def list_open_issues(ctx, repo: str) -> list[dict[str, str]]:
    backend = ctx.get_backend("github_repo")
    if not isinstance(backend, GitHubRepoAPI):
        raise RuntimeError("backend must be a GitHubRepoAPI")
    return backend.list_open_issues(repo)
```

建议：primitive handler 优先通过 `ctx.backend` / `ctx.get_backend(...)` 访问能力，
这样框架才能在 fork/clone 时正确管理依赖。

#### RuntimePrimitiveBackend 生命周期

如果你的 backend 是一个带状态的服务对象，建议让它实现 `RuntimePrimitiveBackend`：

- `clone_for_fork(context=...)`：fork child 时如何复制/共享 backend（默认共享，返回 self）
- `on_install(repl)`：backend 安装到 PyRepl 时回调
- `on_close(repl)`：PyRepl 关闭时回调，用于释放资源或清理状态

这让 fork 时的 copy 策略和生命周期变得可控，也更容易保证子 agent 行为稳定。

当 `@llm_chat(...)` 使用带 runtime 的工具（如 `PyRepl`）时，框架会在 prompt 顶部注入去重后的 `Tool Best Practices` 块；runtime 原语指引会包含在工具自己的最佳实践条目中。

由于 `PyRepl()` 默认已经安装 builtin `selfref` pack，`llm_chat` 可以直接从 toolkit 的 runtime backend 自动解析 `SelfReference`。

首次回合安全：若 memory key 为空，会在执行工具前把当前 system prompt 写入 `self_reference`，确保 runtime 读取不为空且包含 system 消息。

当 `llm_chat` 绑定 `SelfReference` 时，框架还会在 ReAct 生命周期里自动同步 turn 状态：工具批次前绑定最新 history，工具批次后优先提交 pending compaction / context message 更新，finalize 阶段再兜底提交最终 context。

在 `execute_code` 中通过 runtime 原语访问上下文：

```python
# Run inside execute_code
snapshot = runtime.selfref.context.inspect()
print(len(snapshot['messages']))
```

### 将持久经验追加到 system context

这是最常见且推荐的模式：把用户偏好或稳定经验落到 system 内的 experience block 中。

```python
# Run inside execute_code
runtime.selfref.context.remember(
    "User preference: answer in concise bullet points.",
)
runtime.selfref.context.remember(
    "Always include one actionable next step.",
)

print(runtime.selfref.context.inspect()['experiences'])
```

### 常见上下文操作示例

```python
snapshot = runtime.selfref.context.inspect()
print(snapshot['active_key'])
print(snapshot['summary'])
print(len(snapshot['messages']))

remembered = runtime.selfref.context.remember("remember this")
print(remembered['id'])

runtime.selfref.context.forget(remembered['id'])

payload = runtime.selfref.context.compact(
    goal="Current task goal",
    instruction="Current task instruction",
    discoveries=["Important discovery"],
    completed=["Completed item"],
    current_status="Ready for the next milestone.",
    likely_next_work=["Next step"],
    relevant_files_directories=["src/app.py"],
)
print(payload['assistant_message'])
```

Runtime SelfReference 原语参考：

- `runtime.selfref.guide()`: 返回命名空间概览与 fork/context 最佳实践清单。
- `runtime.selfref.context.inspect(key=None)`: 读取完整上下文快照，包含 `experiences`、结构化 `summary` 和只读 `messages`。
- `runtime.selfref.context.remember(text, key=None)`: 向 system experience block 追加 durable experience。
- `runtime.selfref.context.forget(experience_id, key=None)`: 删除一条错误或过时的 durable experience。
- `runtime.selfref.context.compact(..., key=None)`: 排队 milestone compaction；当前工具批次结束后会优先提交，让下一次同 turn 的 LLM 调用看到结构化 assistant summary；如果当前 turn 不再继续调用 LLM，finalize 阶段会兜底提交。
- `runtime.selfref.fork.spawn(message, ...)`: 异步创建子 fork（chat 形态）。
- 子 fork 继承的是 fork 前的上下文快照，不会把父 agent 当前尚未闭合的 fork tool-call 场景当成自己的当前动作。
- `runtime.selfref.fork.gather_all(fork_id_or_list=None, include_history=False)`: 聚合 fork 结果，返回 `dict[fork_id -> ForkResult]`（用 `.items()`/`.values()` 遍历）。

默认情况下 fork 结果为紧凑模式，返回 `fork_id`、`memory_key`、`status`、`response`、`result`、`history_count`、`history_included=False` 等元数据。
读取结果时先看 `status`，成功后读 `response` 或 `result`，失败时检查 `error_type` / `error_message`。
只有确实需要子历史时才设置 `include_history=True`。

Fork 规划清单（`runtime.selfref.guide()` 会返回同样的 guidance）：

1. 每一层 agent 只做本层规划，执行下放给子 fork。
2. 无依赖的任务尽量并行 `fork.spawn(...)`。
3. fork 前先整理上下文：弱相关信息先总结或落盘。
4. fork 提示词写清完成边界和验收标准，优先文件+回传消息交接。
5. fork 结果默认是紧凑模式；需要子历史时再按需 `include_history=True`。
6. 每个里程碑后回收并整理上下文，再进入下一阶段。

清理上下文的注意事项：

- 使用 `reset_repl` 清理 REPL 命名空间中的 Python 变量。
- 需要查看原始消息时，使用 `runtime.selfref.context.inspect()` 并从 `messages` 字段读取。
- 需要删除错误经验时，使用 `runtime.selfref.context.forget(...)`。
- 需要在 milestone 结束后清空 stale transcript 时，使用 `runtime.selfref.context.compact(...)`。

所有操作都会写入 `SelfReference` 的内部存储，而不是直接暴露原始列表。单次对话中的记忆变更会在工具批次后尽早合并到同 turn 的后续上下文里，并且最迟会在回合结束时合并进返回的 `ReactEndEvent.final_messages`。

### 单个 REPL 中多 agent 共享

为不同 agent 使用不同的 `self_reference_key`：

```python
@llm_chat(..., self_reference_key="agent_1")
async def agent_1(message: str, history=None):
    ...

@llm_chat(..., self_reference_key="agent_2")
async def agent_2(message: str, history=None):
    ...
```

这样可以为每个 agent 提供隔离的上下文空间。

## 最佳实践

### 1. Session 隔离

```python
# 为不同任务创建独立的 repl
analysis_repl = PyRepl()
experiment_repl = PyRepl()

# 分析任务使用 analysis_repl
@llm_chat(toolkit=analysis_repl.toolset, ...)
async def analyze(message: str, history=None):
    pass

# 实验任务使用 experiment_repl
@llm_chat(toolkit=experiment_repl.toolset, ...)
async def experiment(message: str, history=None):
    pass
```

### 2. 错误处理

```python
repl = PyRepl()
result = await repl.execute(code="可能出错的代码")

if not result['success']:
    print(f"执行错误: {result['error']}")
    # 可选：读取结构化定位信息
    print(result.get('error_details'))
else:
    print(result['stdout'])
```

### 5. 审计日志（每实例独立）

`PyRepl` 会把代码执行审计记录落盘到独立目录：

- 根目录来自 `.env` / 环境变量中的 `LOG_DIR`
- 每个实例独立子目录：`<LOG_DIR>/pyrepl/<instance_id>/`
- 审计文件：`executions.jsonl`

每条记录包含：执行时间、代码、执行结果、结构化错误详情、超时配置等。

```python
repl = PyRepl()
print(repl.instance_id)
print(repl.audit_log_dir)
print(repl.audit_log_file)
```

### 3. 实时反馈

启用 `event_emitter` 获取实时输出，提供更好的用户体验：

```python
from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

emitter = ToolEventEmitter()

repl = PyRepl()
result = await repl.execute(
    code="for i in range(10): print(i); import time; time.sleep(0.5)",
    event_emitter=emitter
)

# 同时处理事件和最终结果
events = await emitter.get_events()
for event in events:
    print(event.event.data)
```

### 4. 重置状态

当需要重新开始时，可以使用 `reset_repl` 清除所有变量：

```python
# 重置 repl
result = await repl.reset()
print(result)  # "REPL 已重置，所有变量已清除"
```

## Related Links

- Example: `examples/pyrepl_example.py`
- Image artifact example: `examples/pyrepl_seaborn_multimodal_images.py`
- Local runtime memory demo: `examples/runtime_primitives_basic_example.py`
- General TUI agent demo: `examples/tui_general_agent_example.py` (workspace scoped to `./sandbox`)
