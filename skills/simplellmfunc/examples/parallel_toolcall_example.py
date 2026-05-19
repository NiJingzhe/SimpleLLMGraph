"""
并行工具调用的示例

这个示例展示了 SimpleLLMFunc 框架如何支持在单个 LLM 调用中同时执行多个工具调用。
框架会自动并发执行这些工具，充分利用异步特性来提高效率。

关键特点：
1. LLM 可以在单个回复中请求多个工具调用
2. 所有工具调用通过 asyncio.gather() 并发执行
3. 工具执行顺序保持原始顺序（即使并发执行）
4. 支持工具之间的依赖关系（通过消息历史）
5. 完整的错误处理和工具结果验证

运行前准备：请在 examples/provider.json 中配置兼容的提供方与模型。
建议选择支持函数调用/工具调用的 OpenAI 兼容模型。
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from SimpleLLMFunc import tool
from SimpleLLMFunc.llm_decorator import llm_chat
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible


# ============ Provider & Interface ============
current_dir: str = os.path.dirname(os.path.abspath(__file__))
provider_json_path: str = os.path.join(current_dir, "provider.json")

# 选用与现有示例一致的 key（可按需修改到你可用的 provider/model）
llm_interface = OpenAICompatible.load_from_json_file(provider_json_path)["openrouter"][
    "minimax/minimax-m2.5"
]


# ============ 定义多个工具 ============
@tool(name="get_weather", description="获取指定城市的天气信息")
async def get_weather(city: str) -> Dict[str, str]:
    """
    获取城市天气。

    Args:
        city: 城市名

    Returns:
        包含温度/湿度/天气状况的字典
    """
    # 模拟 API 调用延迟
    await asyncio.sleep(0.5)

    weather_data = {
        "北京": {"temperature": "12°C", "humidity": "45%", "condition": "晴天"},
        "上海": {"temperature": "18°C", "humidity": "65%", "condition": "多云"},
        "广州": {"temperature": "25°C", "humidity": "75%", "condition": "小雨"},
        "深圳": {"temperature": "26°C", "humidity": "70%", "condition": "晴天"},
    }
    return weather_data.get(
        city, {"temperature": "未知", "humidity": "未知", "condition": "未知"}
    )


@tool(name="get_stock_price", description="获取指定股票的当前价格")
async def get_stock_price(symbol: str) -> Dict[str, Any]:
    """
    获取股票价格。

    Args:
        symbol: 股票代码（如 AAPL, MSFT）

    Returns:
        包含价格、涨跌幅等信息的字典
    """
    # 模拟 API 调用延迟
    await asyncio.sleep(0.7)

    stock_data = {
        "AAPL": {"price": 150.25, "change": "+2.5%", "currency": "USD"},
        "MSFT": {"price": 320.15, "change": "-1.2%", "currency": "USD"},
        "GOOGL": {"price": 140.85, "change": "+3.1%", "currency": "USD"},
        "TSLA": {"price": 240.50, "change": "+5.8%", "currency": "USD"},
    }
    return stock_data.get(
        symbol, {"price": "未知", "change": "未知", "currency": "USD"}
    )


@tool(name="search_product", description="在电商平台搜索产品")
async def search_product(keyword: str, category: str = "all") -> Dict[str, Any]:
    """
    搜索产品。

    Args:
        keyword: 搜索关键词
        category: 产品分类（可选）

    Returns:
        包含搜索结果的字典
    """
    # 模拟 API 调用延迟
    await asyncio.sleep(0.6)

    results = {
        "laptop": [
            {"name": "MacBook Pro 16", "price": "$2499", "rating": 4.8},
            {"name": "Dell XPS 15", "price": "$1999", "rating": 4.6},
        ],
        "phone": [
            {"name": "iPhone 15 Pro", "price": "$999", "rating": 4.7},
            {"name": "Samsung Galaxy S24", "price": "$899", "rating": 4.5},
        ],
        "headphone": [
            {"name": "AirPods Pro", "price": "$249", "rating": 4.6},
            {"name": "Sony WH-1000XM5", "price": "$399", "rating": 4.8},
        ],
    }

    category_results = results.get(keyword.lower(), results.get(category, []))
    return {
        "keyword": keyword,
        "category": category,
        "found": len(category_results),
        "products": category_results,
    }


@tool(name="get_exchange_rate", description="获取汇率信息")
async def get_exchange_rate(from_currency: str, to_currency: str) -> Dict[str, Any]:
    """
    获取汇率。

    Args:
        from_currency: 源货币代码（如 USD, EUR, CNY）
        to_currency: 目标货币代码

    Returns:
        包含汇率和转换示例的字典
    """
    # 模拟 API 调用延迟
    await asyncio.sleep(0.4)

    rates = {
        ("USD", "CNY"): 7.25,
        ("EUR", "USD"): 1.10,
        ("GBP", "USD"): 1.28,
        ("JPY", "USD"): 0.0095,
    }

    rate = rates.get((from_currency, to_currency), 1.0)
    return {
        "from": from_currency,
        "to": to_currency,
        "rate": rate,
        "timestamp": datetime.now().isoformat(),
    }


# ============ 并行工具调用的 LLM 函数 ============
@llm_chat(
    llm_interface=llm_interface,
    toolkit=[get_weather, get_stock_price, search_product, get_exchange_rate],
    stream=False,
)
async def parallel_info_query(
    history: Optional[List[Dict[str, Any]]] = None,
    query: str = "",
):
    """
    并行查询多个信息源。

    此函数可以在单个 LLM 调用中请求多个工具调用，
    这些工具将被自动并发执行。

    示例场景：
    - 查询多个城市的天气
    - 获取多只股票的价格
    - 搜索多个产品
    - 获取多个汇率
    """
    pass


@llm_chat(
    llm_interface=llm_interface,
    toolkit=[get_weather, get_stock_price, search_product],
    stream=True,
)
async def streaming_parallel_query(
    history: Optional[List[Dict[str, Any]]] = None,
    query: str = "",
):
    """
    流式版本的并行查询。
    支持实时查看 LLM 的响应和工具调用情况。
    """
    pass


# ============ 辅助函数 ============
def print_separator(title: str = "") -> None:
    """打印分隔符"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n{'-' * 60}\n")


def print_tool_call_info(raw_response: Any) -> None:
    """打印工具调用信息"""
    if hasattr(raw_response, "choices") and raw_response.choices:
        msg = raw_response.choices[0].message

        # 打印文本回复
        if hasattr(msg, "content") and msg.content:
            print("[LLM 回复]")
            print(msg.content)

        # 打印工具调用
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("\n[并行工具调用]")
            for i, tc in enumerate(msg.tool_calls, 1):
                print(f"  {i}. 工具: {tc.function.name}")
                try:
                    args = json.loads(tc.function.arguments)
                    print(f"     参数: {json.dumps(args, ensure_ascii=False)}")
                except:
                    print(f"     参数: {tc.function.arguments}")


async def demo_parallel_weather() -> None:
    """演示：并行查询多个城市的天气"""
    print_separator("演示 1: 并行查询多个城市的天气")

    query = "请同时查询北京、上海、广州和深圳四个城市的天气情况，然后帮我分析天气特点"
    print(f"查询: {query}\n")

    history: List[Dict[str, Any]] = []
    async for raw, messages in parallel_info_query(history=history, query=query):
        print_tool_call_info(raw)
        # 消息历史中包含所有的工具调用和结果
        print("\n[工具执行结果已整合到消息历史]")


async def demo_parallel_stocks() -> None:
    """演示：并行查询多只股票的价格"""
    print_separator("演示 2: 并行查询多只股票的价格")

    query = (
        "请同时查询 AAPL、MSFT、GOOGL 和 TSLA 四只股票的价格，然后比较它们的涨跌情况"
    )
    print(f"查询: {query}\n")

    history: List[Dict[str, Any]] = []
    async for raw, messages in parallel_info_query(history=history, query=query):
        print_tool_call_info(raw)
        print("\n[工具执行结果已整合到消息历史]")


async def demo_parallel_shopping() -> None:
    """演示：并行搜索多个产品"""
    print_separator("演示 3: 并行搜索多个产品")

    query = "请帮我同时搜索笔记本电脑、手机和耳机产品，并列出各类别的热门产品"
    print(f"查询: {query}\n")

    history: List[Dict[str, Any]] = []
    async for raw, messages in parallel_info_query(history=history, query=query):
        print_tool_call_info(raw)
        print("\n[工具执行结果已整合到消息历史]")


async def demo_parallel_mixed() -> None:
    """演示：并行查询混合类型的信息"""
    print_separator("演示 4: 并行查询混合信息（天气 + 股票 + 汇率）")

    query = "请同时帮我：1) 查询北京的天气，2) 查询苹果股票价格，3) 查询美元对人民币的汇率，然后总结这些信息"
    print(f"查询: {query}\n")

    history: List[Dict[str, Any]] = []
    async for raw, messages in parallel_info_query(history=history, query=query):
        print_tool_call_info(raw)
        print("\n[工具执行结果已整合到消息历史]")


async def demo_streaming_parallel() -> None:
    """演示：流式并行查询"""
    print_separator("演示 5: 流式并行查询（实时查看 LLM 响应）")

    query = "请同时查询北京和上海的天气，然后比较它们的差异"
    print(f"查询: {query}\n")
    print("[流式响应]")

    history: List[Dict[str, Any]] = []
    async for raw, messages in streaming_parallel_query(history=history, query=query):
        # 流式响应处理
        if hasattr(raw, "choices") and raw.choices:
            choice = raw.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                delta = choice.delta
                # 打印文本增量
                if getattr(delta, "content", None):
                    print(delta.content, end="", flush=True)
                # 检查工具调用增量
                if getattr(delta, "tool_calls", None):
                    print("\n[检测到工具调用]", flush=True)


async def demo_sequential_vs_parallel_performance() -> None:
    """演示：顺序执行与并行执行的性能对比"""
    print_separator("演示 6: 性能对比 - 顺序 vs 并行执行")

    print("场景: 同时查询 3 个股票和 2 个城市的天气信息")
    print("\n[预期性能分析]")
    print(
        "- 顺序执行: 需要 5 次单独调用，总时间 ≈ 0.7 + 0.7 + 0.7 + 0.5 + 0.5 = 3.1 秒"
    )
    print("- 并行执行: 所有调用并发，总时间 ≈ max(0.7, 0.5) = 0.7 秒")
    print("- 性能提升: 3.1 / 0.7 ≈ 4.4 倍\n")

    query = "请查询 AAPL、MSFT、GOOGL 的股票价格，并同时查询北京和上海的天气"
    print(f"执行查询: {query}\n")

    start_time = asyncio.get_event_loop().time()

    history: List[Dict[str, Any]] = []
    async for raw, messages in parallel_info_query(history=history, query=query):
        print_tool_call_info(raw)

    end_time = asyncio.get_event_loop().time()
    elapsed = end_time - start_time

    print(f"\n[实际执行时间] {elapsed:.2f} 秒")


async def main() -> None:
    """主函数"""
    print("\n" + "=" * 60)
    print("  SimpleLLMFunc 并行工具调用示例")
    print("=" * 60)
    print("\n本示例展示如何利用 SimpleLLMFunc 框架的并行工具调用能力")
    print("框架会自动并发执行 LLM 请求的多个工具，提高效率。\n")

    try:
        # 运行各个演示
        await demo_parallel_weather()
        await demo_parallel_stocks()
        await demo_parallel_shopping()
        await demo_parallel_mixed()
        await demo_streaming_parallel()
        await demo_sequential_vs_parallel_performance()

        print_separator("所有演示完成")
        print("[✓] 并行工具调用示例执行成功！\n")

    except KeyboardInterrupt:
        print("\n[⚠] 用户中断执行")
    except Exception as e:
        print(f"\n[✗] 发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
