"""
自定义 Tool 事件发射示例

这个示例展示如何在 Tool 函数内部发射自定义事件，并在 Event Stream 中捕获这些事件。
适用于需要实时报告工具执行进度、批量处理状态等场景。
"""

import asyncio
import os
from SimpleLLMFunc import tool, llm_function, OpenAICompatible
from SimpleLLMFunc.hooks import (
    EventYield,
    ResponseYield,
    is_event_yield,
    is_response_yield,
    ToolEventEmitter,
)
from SimpleLLMFunc.hooks.events import CustomEvent, ToolCallStartEvent, ToolCallEndEvent

# 初始化 LLM 接口
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")

llm = OpenAICompatible.load_from_json_file(provider_json_path)["openrouter"][
    "minimax/minimax-m2.5"
]


@tool(name="batch_process", description="批量处理多个项目")
async def batch_process(
    items: list,
    event_emitter: ToolEventEmitter = None,
) -> str:
    """
    批量处理项目列表

    Args:
        items: 要处理的项目列表
        event_emitter: 事件发射器，用于报告进度
    """
    total = len(items)
    results = []

    for i, item in enumerate(items):
        # 发射自定义进度事件
        if event_emitter:
            await event_emitter.emit(
                event_name="batch_progress",
                data={
                    "current": i + 1,
                    "total": total,
                    "item": item,
                    "percent": int((i + 1) / total * 100),
                },
            )

        # 模拟处理逻辑
        await asyncio.sleep(0.1)
        results.append(f"processed_{item}")

        # 发射完成事件
        if event_emitter:
            await event_emitter.emit(
                event_name="item_completed",
                data={"item": item, "result": results[-1]},
            )

    # 发射最终完成事件
    if event_emitter:
        await event_emitter.emit(
            event_name="batch_completed",
            data={"total_processed": total, "results": results},
        )

    return f"完成处理 {total} 个项目"


@llm_function(
    llm_interface=llm,
    toolkit=[batch_process],
)
async def process_task(task: str) -> str:
    """
    处理批量任务

    Args:
        task: 任务描述

    Returns:
        处理结果
    """
    pass


async def main():
    print("=" * 70)
    print("示例：自定义 Tool 事件发射")
    print("=" * 70)
    print("\n任务：批量处理 ['item1', 'item2', 'item3', 'item4', 'item5']\n")
    print("-" * 70)

    # 调用 LLM 函数
    async for output in process_task(
        task="请处理以下项目：item1, item2, item3, item4, item5"
    ):
        # 处理响应结果
        if is_response_yield(output):
            print(f"\n最终结果：\n{output.response}\n")
            print("-" * 70)

        # 处理事件
        elif is_event_yield(output):
            event = output.event

            # 捕获 Tool 调用事件
            if isinstance(event, ToolCallStartEvent):
                print(f"[Tool] 开始调用工具: {event.tool_name}")
                print(f"         参数: {event.arguments}")

            elif isinstance(event, ToolCallEndEvent):
                print(f"[Tool] 工具调用完成: {event.tool_name}")
                print(f"         执行时间: {event.execution_time:.2f}s")
                print(f"         结果: {str(event.result)[:50]}...")

            # 捕获自定义事件
            elif isinstance(event, CustomEvent):
                print(f"[Custom] 事件: {event.event_name}")
                print(f"         数据: {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
