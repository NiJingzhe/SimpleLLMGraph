# Langfuse 集成指南

SimpleLLMFunc 框架已集成 Langfuse 可观测性平台，支持对 LLM 生成和工具调用进行全面追踪。

## 功能特性

- **LLM 生成追踪**: 自动记录所有 LLM 调用的输入、输出、模型参数和使用统计
- **工具调用观测**: 追踪工具调用的参数、执行结果和性能指标
- **嵌套跨度支持**: 支持复杂的多层调用链追踪
- **流式响应支持**: 兼容流式和非流式 LLM 响应

## 安装和配置

### 1. 安装 Langfuse

Langfuse 已包含在框架依赖中：

```bash
# 如果使用 poetry
poetry install

# 如果使用 pip
pip install langfuse
```

### 2. 获取 Langfuse 凭据

1. 访问 [Langfuse](https://langfuse.com) 并注册账户
2. 创建新项目
3. 获取项目的 Public Key 和 Secret Key

### 3. 配置环境变量

```bash
export LANGFUSE_PUBLIC_KEY="your_public_key"
export LANGFUSE_SECRET_KEY="your_secret_key"
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"  # 可选
export LANGFUSE_EXPORT_ALL_SPANS="true"  # 可选（默认保持 v3 的 export-all 行为）
export LANGFUSE_ENABLED="True"  # 可选（当前实现不会自动禁用调用）
```

### 4. 初始化观测系统

SimpleLLMFunc 会自动从环境变量读取 Langfuse 配置，无需额外初始化。框架内部会在需要时自动连接 Langfuse：

> 提示：当前实现不会根据 `LANGFUSE_ENABLED` 自动跳过观测调用；请确保 key 可用，或在应用层自行控制是否启用。

```python
# 只需设置环境变量，框架会自动处理 Langfuse 连接
# 不需要手动初始化观测器

from SimpleLLMFunc import llm_function

llm = ...  # 你的 LLM 接口实例

@llm_function(llm_interface=llm)
async def my_function(text: str) -> str:
    """功能描述"""
    pass

# 所有调用都会自动追踪到 Langfuse（如果已配置）
result = await my_function("test")
```

如需手动获取 Langfuse 客户端或配置信息，可使用以下方式：

```python
from SimpleLLMFunc.observability import langfuse_config, get_langfuse_client

# 获取配置对象
config = langfuse_config
print(f"Langfuse 已启用: {config.LANGFUSE_ENABLED}")

# 获取 Langfuse 客户端（用于高级场景）
client = get_langfuse_client()
if client:
    # 手动追踪其他操作
    pass
```

## 使用示例

### 基本 LLM 函数追踪

```python
import asyncio
from SimpleLLMFunc import llm_function, OpenAICompatible

# 配置 LLM 接口
llm = OpenAICompatible.load_from_json_file("provider.json")["openai"]["gpt-3.5-turbo"]

@llm_function(llm_interface=llm)
async def analyze_text(text: str) -> str:
    """分析文本内容并提供摘要"""
    pass

# 使用函数 - 自动追踪到 Langfuse（如果已配置环境变量）
async def main():
    result = await analyze_text("这是一段需要分析的文本...")
    print(result)

asyncio.run(main())
```

### 带工具调用的追踪

```python
from SimpleLLMFunc import llm_function, tool

@tool(name="calculate", description="执行数学计算")
async def calculate(expression: str) -> dict:
    """执行数学表达式计算"""
    result = eval(expression)  # 实际使用中应使用更安全的方法
    return {"expression": expression, "result": result}

@llm_function(
    llm_interface=llm,
    toolkit=[calculate],
    max_tool_calls=3
)
async def math_assistant(question: str) -> str:
    """数学助手，可以回答数学问题并进行计算"""
    pass

# 使用 - 工具调用也会被自动追踪
result = await math_assistant("计算 15 * 8 + 32 的结果")
```

### 聊天对话追踪

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.hooks.stream import is_response_yield

@llm_chat(
    llm_interface=llm,
    toolkit=[calculate],
    max_tool_calls=2
)
async def chat_bot(message: str, history: list = None):
    """智能聊天机器人"""
    pass

# 使用 - 每轮对话都会被追踪
history = []
async for output in chat_bot("你好，请帮我计算一些数学题", history):
    if is_response_yield(output):
        if str(output.response).strip():
            print(output.response)
        history = output.messages
```

## 追踪数据结构

### Generation 追踪

每个 LLM 调用会创建一个 Generation 观测，包含：

- **输入**: 发送给 LLM 的消息列表
- **输出**: LLM 的响应内容和工具调用
- **模型信息**: 模型名称和参数
- **使用统计**: Token 使用量和成本信息
- **元数据**: 流式模式、可用工具数量等

### Tool 追踪

每个工具调用会创建一个 Tool 观测，包含：

- **输入**: 工具调用的参数
- **输出**: 工具执行的结果
- **元数据**: 工具调用 ID、执行时间等

### 层级结构

```
Function Call (Span)
├── Initial Generation
│   ├── Input: Messages
│   ├── Output: Response + Tool Calls
│   └── Usage: Token counts
├── Tool Call 1 (Tool)
│   ├── Input: Parameters
│   └── Output: Result
├── Tool Call 2 (Tool)
│   ├── Input: Parameters
│   └── Output: Result
└── Follow-up Generation
    ├── Input: Updated Messages
    ├── Output: Final Response
    └── Usage: Token counts
```

## 配置选项

### 环境变量

| 变量名 | 描述 | 默认值 | 必需 |
|--------|------|--------|------|
| `LANGFUSE_PUBLIC_KEY` | Langfuse 公钥 | - | 是 |
| `LANGFUSE_SECRET_KEY` | Langfuse 私钥 | - | 是 |
| `LANGFUSE_BASE_URL` | Langfuse 服务器地址 | `https://cloud.langfuse.com` | 否 |
| `LANGFUSE_EXPORT_ALL_SPANS` | 导出所有 OpenTelemetry spans（保持 v3 行为） | `true` | 否 |
| `LANGFUSE_ENABLED` | 是否启用观测（当前实现不强制） | `true` | 否 |

### Langfuse v4 注意事项

- v4 默认会过滤非 LLM 的 OpenTelemetry spans；框架默认保持 v3 的“导出全部 spans”行为，若希望使用 v4 的智能过滤，请设置 `LANGFUSE_EXPORT_ALL_SPANS=false`。
- v4 会将 `metadata` 值规范化为字符串并限制长度，框架侧已自动做字符串化处理。

### 高级配置

如需自定义 Langfuse 配置（例如在生产环境动态设置密钥），可在启动应用前修改环境变量：

```python
import os

# 在导入 SimpleLLMFunc 之前设置环境变量
os.environ["LANGFUSE_PUBLIC_KEY"] = "your_public_key"
os.environ["LANGFUSE_SECRET_KEY"] = "your_secret_key"
os.environ["LANGFUSE_BASE_URL"] = "https://cloud.langfuse.com"
os.environ["LANGFUSE_ENABLED"] = "true"

# 然后导入并使用 SimpleLLMFunc
from SimpleLLMFunc import llm_function

# ... 后续代码 ...
```

## 最佳实践

### 1. 环境分离

```bash
# 开发环境
export LANGFUSE_ENABLED="false"

# 生产环境
export LANGFUSE_ENABLED="true"
export LANGFUSE_PUBLIC_KEY="prod_public_key"
export LANGFUSE_SECRET_KEY="prod_secret_key"
```

> 说明：当前版本不会自动读取 `LANGFUSE_ENABLED` 来跳过观测调用。如需按环境开关，请在应用层自行判断。

### 2. 错误处理

框架会自动处理 Langfuse 相关错误，但建议监控日志：

```python
import logging

# 启用 SimpleLLMFunc 日志
logging.getLogger("SimpleLLMFunc").setLevel(logging.INFO)
```

### 3. 性能考虑

- Langfuse 调用是异步的，不会阻塞主要业务逻辑
- 在高并发场景下，考虑设置合适的 Langfuse 客户端配置
- 可以通过 `LANGFUSE_ENABLED=false` 临时禁用观测

### 4. 数据隐私

- 确保敏感数据在发送到 Langfuse 前已脱敏
- 考虑使用自托管的 Langfuse 实例处理敏感数据

## 故障排除

### 常见问题

1. **观测器未启用**
   ```
   警告: Langfuse 观测功能未启用
   ```
   - 检查环境变量是否正确设置
   - 确认 Langfuse 包已安装

2. **连接失败**
   ```
   Langfuse 初始化失败: Connection error
   ```
   - 检查网络连接
   - 验证 API 密钥是否正确
   - 确认 LANGFUSE_BASE_URL 设置正确

3. **数据未显示**
   - 检查 Langfuse 仪表板的项目设置
   - 确认使用的是正确的项目密钥
   - 等待数据同步（通常几秒钟）

### 调试模式

```python
import logging

# 启用详细日志
logging.getLogger("SimpleLLMFunc").setLevel(logging.DEBUG)

# 查看 Langfuse 相关日志
logging.getLogger("langfuse").setLevel(logging.DEBUG)
```

## 示例项目

可参考以下示例了解完整链路：

- `examples/llm_function_token_usage.py`（事件流 + 用量统计）
- `examples/event_stream_chatbot.py`（对话 + 工具调用 + 事件流）

## 相关链接

- [Langfuse 官方文档](https://langfuse.com/docs)
