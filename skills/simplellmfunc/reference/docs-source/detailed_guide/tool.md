# Tool 工具系统

## 实现功能

SimpleLLMFunc 的工具系统为大语言模型提供了调用外部函数和 API 的能力，让 LLM 能够执行计算、查询数据、调用服务等操作。工具系统支持两种创建方式，并能自动将 Python 函数转换为 LLM 可理解的工具描述格式。

### 核心功能特性
- **智能类型推断**: 自动从函数签名中提取参数类型和描述信息
- **文档字符串解析**: 支持从 docstring 中解析参数描述
- **JSON Schema 生成**: 自动生成符合 OpenAI Function Calling API 的工具描述
- **类型安全**: 支持基本类型、容器类型、Pydantic 模型等多种类型
- **灵活创建**: 支持装饰器和继承两种创建方式
- **批量序列化**: 支持将多个工具一次性序列化为 API 格式
- **长文本截断**: 可选开启，自动将超长返回结果（>20000 tokens）保存到临时文件并截断

### 支持的数据类型（参数类型）
- **基本类型**: `str`, `int`, `float`, `bool`
- **容器类型**: `List[T]`, `Dict[K, V]`
- **多模态类型**: `Text`, `ImgPath`, `ImgUrl`
- **多模态列表**: `List[Text]`, `List[ImgPath]`, `List[ImgUrl]`（由字符串列表自动转换）
- **Pydantic 模型**: 自动解析模型字段和验证规则
- **可选参数**: 支持带默认值的可选参数
- **复杂嵌套**: 支持嵌套的容器类型和复杂对象

**⚠️ 注意： 由于LLM提供给工具的参数只能是json支持的类型，所以实际上`Tuple`, `Set`等容器是无法作为工具参数的**

### 支持的返回类型
工具函数可以返回多种数据格式，系统会自动处理这些返回值并将其传递给 LLM：

#### 基本返回类型
- **字符串 (`str`)**: 直接作为文本内容返回给 LLM
- **JSON 可序列化对象**: 包括 `dict`, `list`, `int`, `float`, `bool`, `None`
- **Pydantic 模型**: 自动序列化为 JSON 格式

#### 多模态返回类型
- **图片URL (`ImgUrl`)**: 返回网络图片链接，LLM 可以"看到"图片内容
- **本地图片 (`ImgPath`)**: 返回本地图片文件路径，自动转换为 base64 格式
- **文本+图片组合 (`Tuple[str, ImgUrl]` 或 `Tuple[str, ImgPath]`)**: 同时返回文本说明和图片

#### 返回类型示例

```python
from SimpleLLMFunc import tool
from SimpleLLMFunc.type import ImgUrl, ImgPath
from typing import Dict, List, Tuple, Any

# 1. 基本类型返回
@tool(name="calculate", description="执行数学计算")
async def calculate(expression: str) -> float:
    """返回计算结果（浮点数）"""
    return eval(expression)

@tool(name="get_status", description="获取系统状态")
async def get_status() -> str:
    """返回状态信息（字符串）"""
    return "系统运行正常"

# 2. JSON 对象返回
@tool(name="get_user_info", description="获取用户信息")
async def get_user_info(user_id: int) -> Dict[str, Any]:
    """返回用户信息（字典）"""
    return {
        "id": user_id,
        "name": "张三",
        "age": 25,
        "skills": ["Python", "AI", "数据分析"]
    }

@tool(name="search_results", description="搜索并返回结果列表")
async def search_results(query: str) -> List[Dict[str, str]]:
    """返回搜索结果（字典列表）"""
    return [
        {"title": "结果1", "url": "https://example1.com"},
        {"title": "结果2", "url": "https://example2.com"}
    ]

# 3. 多模态返回 - 单独图片
@tool(name="get_chart", description="生成数据图表")
async def get_chart(data: List[float]) -> ImgPath:
    """返回图表图片（本地文件）"""
    # 在chart path下有一个对应的图片文件
    chart_path = "/path/to/generated/chart.png"
    return ImgPath(chart_path)

@tool(name="fetch_image", description="获取网络图片")
async def fetch_image(image_url: str) -> ImgUrl:
    """返回网络图片URL"""
    return ImgUrl(image_url)

# 4. 多模态返回 - 文本+图片组合
@tool(name="analyze_image", description="分析图片并生成报告")
async def analyze_image(image_path: str) -> Tuple[str, ImgPath]:
    """返回分析报告和标注后的图片"""
    analysis_text = "检测到3个对象：2个人、1辆汽车"
    annotated_image = ImgPath("/path/to/annotated_image.png")
    return (analysis_text, annotated_image)
```

#### 返回类型处理机制

1. **基本类型**: 直接序列化为 JSON 字符串传递给 LLM
2. **图片类型**: 
   - `ImgUrl`: 直接使用网络 URL
   - `ImgPath`: 自动转换为 base64 编码的 data URL
3. **组合类型**: 将文本和图片组合成多模态消息，LLM 可以同时看到文本说明和图片内容
4. **错误处理**: 不支持的返回类型会自动转换为字符串格式

**多模态返回的特殊处理**：

- 当工具返回图片或文本+图片组合时，框架会用 **assistant + user** 的多模态消息对替代标准 `tool` 消息
- 这是因为 OpenAI tool_call 消息无法直接携带多模态内容
- 因此在事件流与历史中，工具调用会被折叠为一条说明性 assistant 消息与一条包含多模态内容的 user 消息

**⚠️ 返回类型注意事项**:
- 确保本地图片文件路径存在且可读
- 网络图片 URL 应该是公开可访问的
- 组合类型的元组必须是 `(str, ImgPath)` 或 `(str, ImgUrl)` 格式, 不能交换`str`和`ImgPath`/`ImgUrl`的顺序
- 避免返回过大的数据结构，以免影响 LLM 处理效率

### 长文本截断功能

当工具可能返回非常长的文本结果时（例如代码执行输出、大量搜索结果等），可以启用 `too_long_to_file` 功能来自动处理：

**工作原理：**

1. 当工具返回 `str` 类型结果时，框架会估算其 token 数量
2. 如果超过 **20000 tokens**，框架会：
   - 将完整结果保存到系统临时文件（路径格式：`/tmp/tool_result_{tool_name}_{random}.txt`）
   - 截断返回内容到前 20000 tokens
   - 在末尾附加 `<system-reminder>` 提示，告知完整结果的文件路径

**启用方式：**

```python
@tool(name="execute_code", description="执行代码", too_long_to_file=True)
async def execute_code(code: str) -> str:
    """执行 Python 代码并返回输出"""
    # 当输出超过 20000 tokens 时，会自动截断
    result = run_python(code)
    return result
```

**截断后的返回示例：**

```
[前 20000 tokens 的内容...]

<system-reminder> 
tool return was too long. This is truncated result. Full result could be found in /tmp/tool_result_execute_code_a1b2c3d4.txt, you can use file operation toolkits or run code through pyrepl to progressivly reading necessary part of the result file. If you have no way to read file, you should warn the user and let they help you.
</system-reminder>
```

**使用建议：**

- 仅在工具可能返回大量文本时启用（如 REPL 执行输出、文件搜索结果等）
- 启用此功能要求 agent 具备文件读取能力（如 `FileToolset` 或 `PyRepl`）
- 对于返回结构化数据（dict/list）的工具，此功能不会生效

**⚠️ 注意：`@tool` 装饰器要求被装饰的函数本身定义为 `async def`，以便在异步执行链路中无缝协作。**

## 内置工具集

### FileToolset（文件工具）

提供安全的文件读取、搜索与编辑能力，带哈希校验以避免陈旧写入。仅允许访问工作区内的非隐藏文件。

**工具列表：**

- `read_file`：读取文件内容，可指定起止行并附带行号
- `grep`：在工作区内进行正则搜索（必须提供 `path_pattern`）
- `sed`：按行范围进行正则替换（需先 read_file）
- `echo_into`：覆盖写入整个文件（需先 read_file）

**使用示例：**

```python
from SimpleLLMFunc.builtin import FileToolset
from SimpleLLMFunc import llm_chat

file_tools = FileToolset("/path/to/workspace").toolset

@llm_chat(toolkit=file_tools)
async def agent(message: str, history=None):
    """文件操作助手"""
```

**注意事项：**

- 修改前务必先 `read_file`，否则会返回 “file has been changed” 提示
- `grep` 必须提供 `path_pattern` 以限定搜索范围
- 隐藏文件/目录不可访问

## 使用方法

### 基本语法

#### 方式一：装饰器方式（推荐）

```python
from SimpleLLMFunc import tool

@tool(name="工具名称", description="工具简短描述")
async def your_function(param1: Type1, param2: Type2 = default_value) -> ReturnType:
    """
    详细的函数说明，这部分会被包含在工具描述中
    
    Args:
        param1: 参数1的详细描述
        param2: 参数2的详细描述
        
    Returns:
        返回值描述
    """
    # 函数实现
    pass
```

#### 方式二：继承方式（兼容旧版本）

```python
from SimpleLLMFunc import Tool

class YourTool(Tool):
    def __init__(self):
        super().__init__(
            name="工具名称",
            description="工具描述"
        )
    
    async def run(self, *args, **kwargs):
        # 工具执行逻辑
        pass
```

### 参数说明

#### @tool 装饰器参数
- **name** (必需): 工具名称，应该简洁明了，符合函数命名规范
- **description** (必需): 工具的简短描述，说明工具的主要功能
- **best_practices** (可选): 工具最佳实践提示，`llm_chat` / `llm_function` 会自动注入 system prompt
- **prompt_injection_builder** (可选): 动态注入器，接收上下文并返回一段工具专属的 system prompt 引导
- **too_long_to_file** (可选): 是否启用长文本截断功能，默认 `False`。启用后，当工具返回的字符串超过 20000 tokens 时，会自动将完整结果保存到临时文件，并截断返回内容。**注意：启用此功能要求 agent 配备文件读写能力。**

#### 函数要求
- **类型标注**: 建议为所有参数添加类型标注，以便自动生成准确的 JSON Schema
- **文档字符串**: 建议编写详细的 docstring，特别是 Args 部分的参数描述
- **返回类型**: 建议添加返回类型标注

### 工具调用流程

1. **工具注册**: 使用 `@tool` 装饰器或继承 `Tool` 类创建工具
2. **参数解析**: 系统自动从函数签名中提取参数信息
3. **Schema 生成**: 自动生成符合 OpenAI API 的工具描述
4. **LLM 调用**: LLM 根据工具描述决定是否调用工具
5. **参数验证**: 系统验证 LLM 提供的参数是否符合要求
6. **函数执行**: 调用原始 Python 函数并返回结果

## 实现方法

### 核心类结构

#### Parameter 类
```python
class Parameter:
    """工具参数的包装类"""
    def __init__(self, name, description, type_annotation, required, default=None, example=None):
        self.name = name                    # 参数名
        self.description = description      # 参数描述
        self.type_annotation = type_annotation  # Python 类型标注
        self.required = required            # 是否必需
        self.default = default             # 默认值
        self.example = example             # 示例值
```

#### Tool 类
```python
class Tool(ABC):
    """抽象工具基类"""
    def __init__(self, name, description, func=None, too_long_to_file=False):
        self.name = name
        self.description = description
        self.func = func                   # 关联的函数
        self.too_long_to_file = too_long_to_file  # 是否启用长文本截断
        self.parameters = self._extract_parameters()  # 参数列表
    
    def run(self, *args, **kwargs):
        """执行工具"""
        
    def to_openai_tool(self):
        """转换为 OpenAI 工具格式"""
        
    @staticmethod
    def serialize_tools(tools):
        """批量序列化工具"""
```

### 关键实现细节

#### 参数提取机制
系统通过以下步骤提取函数参数信息：

1. **签名分析**: 使用 `inspect.signature()` 获取函数签名
2. **类型提示**: 使用 `get_type_hints()` 获取类型标注
3. **文档解析**: 使用正则表达式解析 docstring 中的参数描述
4. **默认值处理**: 识别可选参数和默认值

#### 类型转换规则
```python
# 基本类型映射
str  -> {"type": "string"}
int  -> {"type": "integer"}
float -> {"type": "number"}
bool -> {"type": "boolean"}

# 容器类型
List[T] -> {"type": "array", "items": schema_of_T}
Dict[K, V] -> {"type": "object", "additionalProperties": schema_of_V}

# Pydantic 模型
BaseModel -> {"type": "object", "properties": model_json_schema}
```

#### 文档字符串解析
系统支持标准的 docstring 格式：

```python
def example_function(param1: str, param2: int = 10):
    """
    函数的主要描述
    
    Args:
        param1: 第一个参数的描述
        param2: 第二个参数的描述，可选
        
    Returns:
        返回值描述
    """
```

## 兼容写法

### 装饰器方式示例

```python
from SimpleLLMFunc import tool
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 示例1：基本数据类型
@tool(name="calculate", description="执行数学计算")
async def calculate(expression: str) -> float:
    """
    计算数学表达式的值
    
    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4"
        
    Returns:
        计算结果
    """
    return eval(expression)

# 示例2：带可选参数
@tool(name="search_web", description="搜索网络信息")
async def search_web(query: str, max_results: int = 10, language: str = "zh") -> List[Dict[str, str]]:
    """
    在网络上搜索信息
    
    Args:
        query: 搜索关键词
        max_results: 最大返回结果数量，默认10条
        language: 搜索语言，默认中文
        
    Returns:
        搜索结果列表，每个结果包含标题和链接
    """
    # 模拟搜索实现
    return [
        {"title": f"搜索结果 {i}", "url": f"http://example.com/{i}"} 
        for i in range(max_results)
    ]

# 示例3：使用 Pydantic 模型
class Location(BaseModel):
    latitude: float = Field(..., description="纬度")
    longitude: float = Field(..., description="经度")
    name: Optional[str] = Field(None, description="位置名称")

@tool(name="get_weather", description="获取天气信息")
async def get_weather(location: Location, days: int = 1) -> Dict[str, Any]:
    """
    获取指定位置的天气预报
    
    Args:
        location: 位置信息，包含经纬度坐标
        days: 预报天数，1-7天，默认1天
        
    Returns:
        天气预报数据，包含温度、湿度、天气状况等
    """
    return {
        "location": location.name or f"{location.latitude},{location.longitude}",
        "forecast": [
            {
                "day": i + 1,
                "temperature": 25,
                "humidity": 60,
                "condition": "晴朗"
            } for i in range(days)
        ]
    }

# 示例4：复杂数据处理
@tool(name="analyze_data", description="分析数据集")
async def analyze_data(
    data: List[Dict[str, Any]], 
    analysis_type: str,
    include_charts: bool = False
) -> Dict[str, Any]:
    """
    对数据集进行统计分析
    
    Args:
        data: 要分析的数据集，每个元素是一个包含字段的字典
        analysis_type: 分析类型，可选值：summary, trend, correlation
        include_charts: 是否包含图表数据，默认否
        
    Returns:
        分析结果，包含统计信息和可选的图表数据
    """
    result = {
        "type": analysis_type,
        "record_count": len(data),
        "summary": "数据分析完成"
    }
    
    if include_charts:
        result["charts"] = {"type": "bar", "data": "chart_data"}
    
    return result

# 示例5：多模态返回类型
from SimpleLLMFunc.type import ImgUrl, ImgPath
from typing import Tuple

@tool(name="generate_chart", description="生成数据图表")
async def generate_chart(data: List[float], chart_type: str = "bar") -> ImgPath:
    """
    根据数据生成图表并保存为本地文件
    
    Args:
        data: 数据列表
        chart_type: 图表类型，如 bar、line、pie
        
    Returns:
        生成的图表文件路径
    """
    # 模拟图表生成
    import matplotlib.pyplot as plt
    import tempfile
    import os
    
    plt.figure(figsize=(10, 6))
    if chart_type == "bar":
        plt.bar(range(len(data)), data)
    elif chart_type == "line":
        plt.plot(data)
    
    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name)
    plt.close()
    
    return ImgPath(temp_file.name)

@tool(name="fetch_web_image", description="获取网络图片")
async def fetch_web_image(image_url: str) -> ImgUrl:
    """
    验证并返回网络图片URL
    
    Args:
        image_url: 图片的网络地址
        
    Returns:
        验证后的图片URL对象
    """
    # 可以添加URL验证逻辑
    return ImgUrl(image_url, detail="high")

@tool(name="analyze_image_with_report", description="分析图片并生成详细报告")
async def analyze_image_with_report(image_path: str) -> Tuple[str, ImgPath]:
    """
    分析图片内容并生成带标注的图片
    
    Args:
        image_path: 要分析的图片路径
        
    Returns:
        分析报告文本和标注后的图片路径
    """
    # 模拟图像分析
    analysis_report = """
    图像分析报告：
    - 检测到 3 个对象
    - 主要颜色：蓝色、绿色
    - 场景类型：户外风景
    - 置信度：95%
    """
    
    # 模拟生成标注图片
    annotated_image_path = "/path/to/annotated_image.png"
    
    return (analysis_report.strip(), ImgPath(annotated_image_path))

@tool(name="create_data_visualization", description="创建在线数据可视化")
async def create_data_visualization(dataset: Dict[str, Any]) -> Tuple[str, ImgUrl]:
    """
    创建数据可视化并上传到云端
    
    Args:
        dataset: 包含数据和配置的字典
        
    Returns:
        可视化说明和在线图片URL
    """
    # 模拟数据可视化处理
    description = f"""
    数据可视化已创建：
    - 数据点数量：{len(dataset.get('data', []))}
    - 图表类型：{dataset.get('chart_type', '未指定')}
    - 创建时间：刚刚
    """
    
    # 模拟上传到云端并获取URL
    visualization_url = ImgUrl("https://example.com/visualizations/chart_12345.png")
    
    return (description.strip(), visualization_url)
```

### 继承方式示例

```python
from SimpleLLMFunc import Tool
import requests
from typing import Dict, Any, List

class WebSearchTool(Tool):
    """网络搜索工具的继承实现方式"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="在网络上搜索信息并返回相关结果"
        )
    
    async def run(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        执行网络搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数量
            
        Returns:
            搜索结果列表
        """
        # 实际的搜索逻辑
        results = []
        for i in range(max_results):
            results.append({
                "title": f"搜索结果 {i+1}: {query}",
                "url": f"https://example.com/result/{i+1}",
                "snippet": f"关于{query}的相关信息..."
            })
        return results

class APICallTool(Tool):
    """API 调用工具"""
    
    def __init__(self, api_base_url: str):
        super().__init__(
            name="api_call",
            description="调用外部 API 获取数据"
        )
        self.api_base_url = api_base_url
    
    async def run(self, endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        调用 API 端点
        
        Args:
            endpoint: API 端点路径
            method: HTTP 方法
            data: 请求数据
            
        Returns:
            API 响应数据
        """
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        
        if method.upper() == "GET":
            response = requests.get(url, params=data)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"不支持的 HTTP 方法: {method}")
        
        return response.json()
```

### 工具使用示例

```python
import asyncio
from SimpleLLMFunc import llm_function, llm_chat, OpenAICompatible

# 初始化 LLM（从配置文件加载）
models = OpenAICompatible.load_from_json_file("provider.json")
llm = models["openai"]["gpt-3.5-turbo"]

# 在 llm_function 中使用工具
@llm_function(
    llm_interface=llm,
    toolkit=[calculate, search_web, get_weather]
)
async def intelligent_assistant(query: str) -> str:
    """
    智能助手，可以进行计算、搜索和查询天气。
    根据用户查询的内容，选择合适的工具来提供准确的答案。
    """
    pass

# 在 llm_chat 中使用工具
@llm_chat(
    llm_interface=llm,
    toolkit=[calculate, search_web, get_weather, analyze_data]
)
async def chat_with_tools(message: str, history: List[Dict[str, str]] | None = None):
    """
    支持工具调用的聊天助手。
    可以执行计算、搜索网络、查询天气和分析数据。
    """
    pass

# 使用示例
async def main():
    # 使用 llm_function
    result = await intelligent_assistant("帮我计算 25 * 4 + 18 的结果")
    print(f"计算结果: {result}")

    # 使用 llm_chat
    history = []
    async for output in chat_with_tools("北京今天天气怎么样？", history):
        if is_response_yield(output):
            print(output.response, end="")
            history = output.messages

if __name__ == "__main__":
    asyncio.run(main())
```

#### 将 `@llm_function` 暴露为工具（Agent as a Tool）

当你希望一个上层 agent 把下层 specialist 当成工具调用时，可以把
`@tool` 叠加在 `@llm_function` 外层：

```python
from typing import List, Dict
from SimpleLLMFunc import llm_function, llm_chat, tool


@tool(
    name="requirements_specialist",
    description="Delegate focused requirement analysis",
)
@llm_function(llm_interface=llm)
async def requirements_specialist(task: str) -> str:
    """Analyze the task and return a compact requirement breakdown.

    Args:
        task: The task to analyze.
    """
    pass


@llm_chat(
    llm_interface=llm,
    toolkit=[requirements_specialist],
    stream=True,
)
async def supervisor(
    message: str,
    history: List[Dict[str, str]] | None = None,
):
    """Use requirements_specialist for focused sub-analysis before answering."""
    pass
```

要点：

- 推荐顺序写成 `@tool` 外层、`@llm_function` 内层。
- 子 agent 推荐使用普通 `await @llm_function` 调用，这样它表现为普通 `async` tool。
- `@llm_chat` 返回的是异步生成器；如果一定要把它当工具复用，通常需要先写一个手动 wrapper。
- 可运行完整示例见 `examples/agent_as_tool_example.py`。

### 高级用法

#### 工具序列化和检查

```python
from SimpleLLMFunc import Tool
import json

# 创建工具列表
tools = [calculate, search_web, get_weather]

# 序列化为 OpenAI 格式
openai_tools = Tool.serialize_tools(tools)

# 查看生成的工具描述
for tool_spec in openai_tools:
    print(json.dumps(tool_spec, indent=2, ensure_ascii=False))
```

#### 动态工具创建

```python
def create_tool_from_config(config: Dict[str, Any]):
    """根据配置动态创建工具"""

    @tool(name=config["name"], description=config["description"])
    async def dynamic_tool(**kwargs):
        # 根据配置执行相应逻辑
        return config["handler"](kwargs)

    return dynamic_tool

# 配置示例
tool_config = {
    "name": "custom_processor",
    "description": "自定义数据处理器",
    "handler": lambda data: {"processed": True, "data": data}
}

custom_tool = create_tool_from_config(tool_config)
```

#### 工具链组合

```python
@tool(name="multi_step_analysis", description="多步骤数据分析")
async def multi_step_analysis(data: List[Dict[str, Any]], steps: List[str]) -> Dict[str, Any]:
    """
    执行多步骤数据分析流程
    
    Args:
        data: 原始数据
        steps: 分析步骤列表，如 ["clean", "analyze", "visualize"]
        
    Returns:
        分析结果
    """
    results = {"steps_completed": []}
    
    for step in steps:
        if step == "clean":
            # 数据清洗
            results["cleaned_records"] = len(data)
        elif step == "analyze":
            # 数据分析
            results["analysis"] = {"mean": 0, "std": 0}
        elif step == "visualize":
            # 数据可视化
            results["charts"] = ["bar", "line", "pie"]
        
        results["steps_completed"].append(step)
    
    return results
```

### 返回类型最佳实践

#### 选择合适的返回类型

1. **文本输出优先**: 如果工具主要产生文本结果，使用 `str` 或结构化的 `Dict`
2. **结构化数据**: 复杂数据使用 `Dict` 或 `List`，便于 LLM 理解和处理
3. **多模态内容**: 需要展示图片时使用 `ImgPath`、`ImgUrl` 或组合类型
4. **组合输出**: 需要同时提供说明和图片时使用 `Tuple[str, ImgPath/ImgUrl]`

#### 性能优化建议

```python
# ✅ 推荐：结构化返回，便于LLM理解
@tool(name="search_products", description="搜索商品")
async def search_products(query: str) -> Dict[str, Any]:
    return {
        "total": 10,
        "products": [
            {"name": "商品1", "price": 99.9, "in_stock": True},
            {"name": "商品2", "price": 149.9, "in_stock": False}
        ],
        "query_time": "2024-01-01 12:00:00"
    }

# ✅ 推荐：多模态组合返回
@tool(name="generate_report", description="生成分析报告")
async def generate_report(data: List[Dict]) -> Tuple[str, ImgPath]:
    summary = f"分析了 {len(data)} 条记录，发现 3 个关键趋势"
    chart_path = ImgPath("/tmp/analysis_chart.png")
    return (summary, chart_path)

# ❌ 避免：返回过大的数据结构
def bad_example() -> Dict:
    return {
        "huge_data": list(range(10000)),  # 过大的数据
        "binary_content": b"..."  # 二进制数据无法JSON序列化
    }
```

#### 错误处理模式

```python
@tool(name="safe_division", description="安全除法运算")
async def safe_division(a: float, b: float) -> Dict[str, Any]:
    """
    安全的除法运算，包含错误处理
    
    Returns:
        包含结果或错误信息的字典
    """
    if b == 0:
        return {
            "success": False,
            "error": "除数不能为零",
            "result": None
        }
    
    return {
        "success": True,
        "error": None,
        "result": a / b
    }

@tool(name="robust_image_tool", description="鲁棒的图像处理工具")
async def robust_image_tool(image_path: str) -> Tuple[str, ImgPath]:
    """
    带错误处理的图像工具
    """
    try:
        # 图像处理逻辑
        processed_image = process_image(image_path)
        return ("图像处理成功", ImgPath(processed_image))
    except Exception as e:
        # 返回错误信息和默认图片
        error_msg = f"图像处理失败: {str(e)}"
        default_img = ImgPath("/path/to/error_placeholder.png")
        return (error_msg, default_img)
```

#### 多模态类型详细说明

```python
from SimpleLLMFunc.type import ImgUrl, ImgPath

# ImgPath 使用示例
img_local = ImgPath(
    path="/path/to/image.jpg",
    detail="high"  # 可选：图片细节级别 low/high
)

# ImgUrl 使用示例  
img_url = ImgUrl(
    url="https://example.com/image.jpg",
    detail="low"  # 网络图片建议使用low以节省token
)

# 组合类型使用
def complex_analysis() -> Tuple[str, ImgPath]:
    analysis = """
    检测结果：
    - 人数：3人
    - 车辆：1辆
    - 置信度：92%
    """
    annotated_img = ImgPath("/tmp/detection_result.jpg")
    return (analysis.strip(), annotated_img)
```

---

工具系统提供了强大而灵活的扩展机制，让 LLM 能够调用各种外部功能。通过装饰器方式，开发者可以轻松地将现有函数转换为 LLM 可用的工具，而继承方式则提供了更多的自定义控制。系统自动处理类型转换和参数验证，确保工具调用的安全性和准确性。

**关键要点总结**：
- 支持多种返回类型：基本类型、JSON对象、多模态内容
- 多模态支持：单独图片或文本+图片组合
- 自动类型转换：本地图片转base64，网络图片直接使用URL
- 错误处理：推荐返回结构化的错误信息
- 性能考虑：避免返回过大数据，网络图片使用低细节级别
