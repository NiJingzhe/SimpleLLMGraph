"""
PyRepl 代码执行示例

这个示例展示如何使用 SimpleLLMFunc 的 builtin PyRepl 实现交互式 Python 代码执行。
PyRepl 的核心优势是保持连续上下文，变量可以在多次调用间持久化。

关键特性：
- 独立进程后端：每个实例运行在独立 subprocess，内部使用 IPython InteractiveShell
- 变量持久化：定义的变量可以在后续调用中使用
- 实时输出：通过 event_emitter 实时获取 stdout/stderr
- Session 隔离：不同的 PyRepl 实例相互独立

运行要求：
    安装 SimpleLLMFunc 依赖（包含 IPython）

使用示例：
    python pyrepl_example.py
"""

import asyncio
import sys

from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.builtin import PyRepl
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible
from SimpleLLMFunc.hooks import (
    is_event_yield,
    CustomEvent,
)


def load_llm():
    """加载 LLM 接口"""
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    provider_json_path = os.path.join(current_dir, "provider.json")

    try:
        return OpenAICompatible.load_from_json_file(provider_json_path)["openrouter"][
            "minimax/minimax-m2.5"
        ]
    except Exception:
        return None


async def demo_basic_execution():
    """演示基本代码执行"""
    print("=" * 70)
    print("示例 1: 基本代码执行")
    print("=" * 70)

    repl = PyRepl()
    tools = repl.toolset

    execute_tool = None
    for tool in tools:
        if tool.name == "execute_code":
            execute_tool = tool
            break

    if not execute_tool:
        print("未找到 execute_code 工具")
        return

    result = await execute_tool.func(
        code="""
x = 10
y = 20
result = x + y
print(f"{x} + {y} = {result}")
result
"""
    )

    print("\n执行结果:")
    print(f"  stdout: {result['stdout']}")
    print(f"  return_value: {result['return_value']}")
    print(f"  execution_time_ms: {result['execution_time_ms']:.2f}")


async def demo_continuous_context():
    """演示连续上下文（变量持久化）"""
    print("\n" + "=" * 70)
    print("示例 2: 连续上下文 - 变量持久化")
    print("=" * 70)

    repl = PyRepl()
    tools = repl.toolset
    execute_tool = next(t for t in tools if t.name == "execute_code")

    print("\n[第一次调用] 定义变量 x = [1, 2, 3, 4, 5]")
    result1 = await execute_tool.func(
        code="""
x = [1, 2, 3, 4, 5]
print(f"x = {x}")
x
"""
    )
    print(f"  stdout: {result1['stdout']}")

    print("\n[第二次调用] 使用变量 x，计算均值")
    result2 = await execute_tool.func(
        code="""
import statistics
mean_val = statistics.mean(x)
print(f"均值: {mean_val}")
mean_val
"""
    )
    print(f"  stdout: {result2['stdout']}")
    print(f"  return_value: {result2['return_value']}")

    print("\n[第三次调用] 在之前基础上继续计算")
    result3 = await execute_tool.func(
        code="""
std_val = statistics.stdev(x)
print(f"标准差: {std_val}")
print(f"均值 + 标准差 = {mean_val + std_val}")
mean_val + std_val
"""
    )
    print(f"  stdout: {result3['stdout']}")


async def demo_streaming_output():
    """演示实时 streaming 输出"""
    print("\n" + "=" * 70)
    print("示例 3: 实时 Streaming 输出")
    print("=" * 70)

    repl = PyRepl()
    tools = repl.toolset
    execute_tool = next(t for t in tools if t.name == "execute_code")

    from SimpleLLMFunc.hooks.event_emitter import ToolEventEmitter

    emitter = ToolEventEmitter()

    print("\n执行循环打印代码，实时输出:")
    result = await execute_tool.func(
        code="""
for i in range(5):
    print(f"计数: {i}")
    import time
    time.sleep(0.2)
print("完成!")
""",
        event_emitter=emitter,
    )

    events = await emitter.get_events()

    print("\n捕获的事件:")
    for event in events:
        if isinstance(event.event, CustomEvent):
            print(f"  [{event.event.event_name}] {event.event.data}")


async def demo_llm_integration():
    """演示与 LLM 集成"""
    print("\n" + "=" * 70)
    print("示例 4: 与 LLM 集成进行问题解决")
    print("=" * 70)

    llm = load_llm()
    if not llm:
        print("请配置 provider.json 后再试")
        return

    repl = PyRepl()
    tools = repl.toolset

    print("\n任务：使用 LLM + PyRepl 解决数据分析问题")
    print("-" * 70)

    @llm_chat(
        llm_interface=llm,
        toolkit=tools,
    )
    async def data_assistant(message: str, history=None):
        """
        你是一个 Python 数据分析助手。
        用户会给你一个数据分析任务，你需要编写 Python 代码来完成。
        每次只执行一小段代码，并展示结果。

        记住：
        - 使用 print() 输出结果
        - 变量会在后续调用中保持
        - 可以分步执行复杂任务
        """

    async for output in data_assistant(
        message="""请完成以下数据分析任务：
1. 创建一个包含 100 个随机数的列表 data
2. 计算均值和标准差
3. 找出大于均值的所有数
4. 计算这些数的和"""
    ):
        if is_event_yield(output):
            event = output.event

            if isinstance(event, CustomEvent):
                if event.event_name == "kernel_stdout":
                    print(f"[stdout] {event.data['text']}", end="")
                elif event.event_name == "kernel_stderr":
                    print(f"[stderr] {event.data['text']}", end="", file=sys.stderr)
        else:
            print(f"\n{'=' * 70}")
            print("LLM 最终响应:")
            print(f"{'=' * 70}")
            print(output.response)


async def demo_multiple_repls():
    """演示多个独立的 PyRepl"""
    print("\n" + "=" * 70)
    print("示例 5: 多个独立 PyRepl")
    print("=" * 70)

    repl1 = PyRepl()
    repl2 = PyRepl()

    print(f"\nPyRepl 1: {id(repl1)}")
    print(f"PyRepl 2: {id(repl2)}")

    exec1 = next(t for t in repl1.toolset if t.name == "execute_code")
    await exec1.func(code="repl1_var = 'hello from repl1'")

    exec2 = next(t for t in repl2.toolset if t.name == "execute_code")
    await exec2.func(code="repl2_var = 'hello from repl2'")

    result1 = await exec1.func(code="repl1_var")
    result2 = await exec2.func(code="repl2_var")

    print(f"\nPyRepl1 中的 repl1_var: {result1['return_value']}")
    print(f"PyRepl2 中的 repl2_var: {result2['return_value']}")

    result1_cross = await exec1.func(code="repl2_var")
    print(
        f"\nPyRepl1 尝试访问 repl2_var: {result1_cross['error'] or '成功 (应该是错误)'}"
    )


async def demo_reset():
    """演示重置功能"""
    print("\n" + "=" * 70)
    print("示例 6: 重置 PyRepl 状态")
    print("=" * 70)

    repl = PyRepl()
    tools = repl.toolset
    execute_tool = next(t for t in tools if t.name == "execute_code")
    reset_tool = next(t for t in tools if t.name == "reset_repl")

    await execute_tool.func(code="x = 100")
    print("设置 x = 100")

    result = await execute_tool.func(code="x")
    print(f"读取 x: {result['return_value']}")

    print("\n执行 reset_repl...")
    reset_result = await reset_tool.func()
    print(f"Reset 结果: {reset_result}")

    result = await execute_tool.func(code="x")
    print(f"重置后读取 x: {result['error'] or '成功读取 (应该报错)'}")


async def demo_list_variables():
    """演示列出变量功能"""
    print("\n" + "=" * 70)
    print("示例 7: 列出变量")
    print("=" * 70)

    repl = PyRepl()
    tools = repl.toolset
    execute_tool = next(t for t in tools if t.name == "execute_code")
    list_tool = next(t for t in tools if t.name == "list_variables")

    await execute_tool.func(code="x = 10")
    await execute_tool.func(code="name = 'test'")
    await execute_tool.func(code="import os")
    print("创建了变量: x, name, os")

    result = await list_tool.func()
    print(f"\n当前变量: {result}")


async def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("PyRepl 示例")
    print("=" * 70)

    await demo_basic_execution()
    await demo_continuous_context()
    await demo_streaming_output()
    await demo_list_variables()
    await demo_reset()
    await demo_multiple_repls()

    print("\n" + "=" * 70)
    print("所有示例完成!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
