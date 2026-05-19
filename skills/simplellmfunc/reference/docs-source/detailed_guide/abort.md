# 中断与取消（AbortSignal）

AbortSignal 是一个轻量级的中断机制，用于在运行中的 Agent 回合里停止流式输出、取消正在执行的工具调用，并尽快收敛到安全的终止状态。

适用场景包括：

- 用户主动打断长回复
- UI/服务端超时控制
- 上下文切换（先停掉旧回合，再启动新回合）

## 核心概念

- `AbortSignal`：一个可在多个协程之间共享的中断信号。
- `_abort_signal`：调用参数名（推荐用常量 `ABORT_SIGNAL_PARAM` 避免硬编码）。
- `abort(reason)`：触发中断，并可附带原因说明。

> 提示：`_abort_signal` 只用于运行时控制，不会进入 prompt，也不会传给模型或工具。

## 基本用法

### llm_chat 示例

```python
import asyncio
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.hooks import AbortSignal, ABORT_SIGNAL_PARAM

@llm_chat(llm_interface=llm, stream=True)
async def chat(message: str, history=None):
    """你的系统提示"""
    pass

abort_signal = AbortSignal()

async def run():
    async def abort_later():
        await asyncio.sleep(1.5)
        abort_signal.abort("user_interrupt")

    asyncio.create_task(abort_later())

    async for output in chat(
        "请详细解释 transformer",
        history=[],
        **{ABORT_SIGNAL_PARAM: abort_signal},
    ):
        # 正常处理事件流
        ...

asyncio.run(run())
```

### llm_function 示例

```python
import asyncio
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.hooks import AbortSignal, ABORT_SIGNAL_PARAM

@llm_function(llm_interface=llm)
async def analyze(text: str) -> str:
    """分析文本"""
    pass

abort_signal = AbortSignal()

async def run():
    async def abort_later():
        await asyncio.sleep(1.0)
        abort_signal.abort("timeout")

    asyncio.create_task(abort_later())

    async for output in analyze(
        "一段很长的文本",
        **{ABORT_SIGNAL_PARAM: abort_signal},
    ):
        ...

asyncio.run(run())
```

## 运行行为

- **流式输出**：中断后不再接收新的 chunk。
- **工具调用**：正在执行的工具调用会被取消；在事件流中可能看到 `ToolCallErrorEvent`，其 `error_type` 为 `CancelledError`。
- **事件流收尾**：事件流收尾时，`ReactEndEvent.extra` 会包含：
  - `aborted: true`
  - `abort_reason: <reason>`（如果传入了 reason）
- **非事件流模式**：生成器会提前结束，不会额外产出 `ReactEndEvent`。

> 注意：AbortSignal 是协作式中断。如果工具本身不可取消或处于阻塞状态，实际停止时间可能滞后。

## TUI 的内置中断

在 `@tui` 场景下，当 Agent 仍在回复时再次发送消息，会自动触发中断：

- 当前回合被中止，新的用户消息进入队列
- 新消息会自动加上中断提示语（`"我要打断你的回复。"`）以引导模型及时收束

如果你需要更细的控制（比如自定义中断提示语），可以绕过内置流程，手动创建并传入 `AbortSignal`。
