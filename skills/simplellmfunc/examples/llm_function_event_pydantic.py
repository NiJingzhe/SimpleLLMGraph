"""
使用 Event Stream 与 Pydantic 结构化输出

展示 llm_function 在事件模式下返回 Pydantic 对象的情况。
"""

import asyncio
import os
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function, OpenAICompatible
from SimpleLLMFunc.hooks.events import LLMCallEndEvent, ReactEndEvent
from SimpleLLMFunc.hooks.stream import is_response_yield

# 初始化 LLM 接口
current_dir = os.path.dirname(os.path.abspath(__file__))
provider_json_path = os.path.join(current_dir, "provider.json")

llm = OpenAICompatible.load_from_json_file(provider_json_path)["openrouter"][
    "minimax/minimax-m2.5"
]


# 定义结构化输出模型
class MovieReview(BaseModel):
    """电影评论分析结果"""

    title: str = Field(description="电影标题")
    rating: float = Field(description="评分（1-5分）", ge=1, le=5)
    sentiment: str = Field(description="情感倾向：positive, neutral, negative")
    summary: str = Field(description="评论摘要")
    tags: list[str] = Field(description="关键标签列表")


@llm_function(
    llm_interface=llm,
)
async def analyze_movie_review(review_text: str) -> MovieReview:
    """
    分析电影评论，提取关键信息并返回结构化数据。

    Args:
        review_text: 用户的电影评论文本

    Returns:
        MovieReview: 结构化的评论分析结果
    """
    return MovieReview(title="", rating=0, sentiment="", summary="", tags=[])  # 占位符


async def main():
    review = """
    《流浪地球2》真是太震撼了！特效场面宏大，故事情节紧凑，
    演员演技在线。虽然有些科学细节可以更严谨，但整体来说是一部
    非常优秀的中国科幻电影。强烈推荐！4.5分！
    """

    print("=" * 60)
    print("示例：Event Stream + Pydantic 结构化输出")
    print("=" * 60)
    print(f"\n电影评论：\n{review.strip()}\n")
    print("-" * 60)

    total_tokens = 0

    async for output in analyze_movie_review(review_text=review):
        if is_response_yield(output):
            # output.response 现在是解析后的 Pydantic 对象
            result: MovieReview = output.response  # type: ignore
            print(f"\n分析结果（Pydantic 对象）：")
            print(f"  电影标题: {result.title}")
            print(f"  评分: {result.rating}")
            print(f"  情感: {result.sentiment}")
            print(f"  摘要: {result.summary}")
            print(f"  标签: {', '.join(result.tags)}")
            print(f"\n对象类型: {type(result)}")
            print("-" * 60)
        else:
            event = output.event
            if isinstance(event, LLMCallEndEvent):
                if event.usage:
                    total_tokens += event.usage.total_tokens
                    print(f"\n[Token 用量] {event.usage.total_tokens} tokens")

            elif isinstance(event, ReactEndEvent):
                print(f"\n[总计] {total_tokens} tokens")
                print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
