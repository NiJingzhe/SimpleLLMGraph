"""
使用 Event Stream 捕获 llm_function 的 Token 用量信息

这个示例展示如何使用 @llm_function 装饰器结合 Event Stream 来实时监控 Token 使用情况。
适用于需要精确计量 API 调用成本的场景。
"""

import asyncio
import os
from SimpleLLMFunc import llm_function, OpenAICompatible
from SimpleLLMFunc.hooks.events import LLMCallEndEvent, ReactEndEvent
from SimpleLLMFunc.hooks.stream import is_response_yield

# 初始化 LLM 接口
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")

llm = OpenAICompatible.load_from_json_file(provider_json_path)["openrouter"][
    "minimax/minimax-m2.5"
]


# 定义一个简单的 LLM 函数，用于文本摘要
@llm_function(
    llm_interface=llm,
)
async def summarize_text(text: str) -> str:
    """
    将给定的文本进行简洁的摘要，提取核心要点。

    Args:
        text: 需要摘要的文本内容

    Returns:
        文本的简洁摘要
    """
    return ""  # 函数体不会被执行，由 LLM 处理


async def main():
    # 测试文本
    test_text = """
    人工智能（AI）正在改变我们的世界。从自动驾驶汽车到智能助手，
    AI 技术已经渗透到日常生活的方方面面。机器学习算法使计算机能够
    从数据中学习并做出决策，而深度学习则通过模拟人脑神经网络的工作方式，
    实现了图像识别、自然语言处理等复杂任务。随着计算能力的提升和
    数据量的增长，AI 的应用前景将更加广阔。
    """

    print("=" * 60)
    print("示例：使用 Event Stream 捕获 Token 用量")
    print("=" * 60)
    print(f"\n原始文本：\n{test_text.strip()}\n")
    print("-" * 60)

    # Token 统计变量
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # 调用 LLM 函数，处理事件流
    async for output in summarize_text(text=test_text):
        # 处理响应结果
        if is_response_yield(output):
            # output.response 已经是解析后的字符串，不是原始的 ChatCompletion
            print(f"\n摘要结果：\n{output.response}\n")
            print("-" * 60)

        # 处理事件
        else:
            event = output.event

            # 捕获 LLM 调用结束事件，获取 Token 用量
            if isinstance(event, LLMCallEndEvent):
                usage = event.usage
                if usage:
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    tokens = usage.total_tokens

                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_tokens += tokens

                    print(f"\n[Token 用量] 本次调用:")
                    print(f"  - Prompt Tokens: {prompt_tokens}")
                    print(f"  - Completion Tokens: {completion_tokens}")
                    print(f"  - Total Tokens: {tokens}")

            # 捕获 ReAct 循环结束事件
            elif isinstance(event, ReactEndEvent):
                print(f"\n[ReAct 结束] 总计:")
                print(f"  - 总 Prompt Tokens: {total_prompt_tokens}")
                print(f"  - 总 Completion Tokens: {total_completion_tokens}")
                print(f"  - 总 Tokens: {total_tokens}")
                print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
