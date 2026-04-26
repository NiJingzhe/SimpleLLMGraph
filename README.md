![SimpleLLMFunc](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/img/repocover_1_1.png?raw=true)

<center>
<h2 style="font-size:2em;">LLM as Function, Prompt as Code</h2>
</center>

<div align="center">
  <a href="README_ZH.md" style="font-size: 1.2em; font-weight: bold; color: #007acc; text-decoration: none; border: 2px solid #007acc; padding: 8px 16px; border-radius: 6px; background: linear-gradient(135deg, #f0f8ff, #e6f3ff);">
    📖 中文版 README 可用
  </a>
</div>

----

![Github Stars](https://img.shields.io/github/stars/NiJingzhe/SimpleLLMFunc.svg?style=social)
![Github Forks](https://img.shields.io/github/forks/NiJingzhe/SimpleLLMFunc.svg?style=social)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/SimpleLLMFunc)](https://pypi.org/project/SimpleLLMFunc/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/NiJingzhe/SimpleLLMFunc/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NiJingzhe/SimpleLLMFunc/pulls)

### Update Notes (0.7.8)

🧠 **Responses API Support**: added `OpenAIResponsesCompatible` as a first-class adapter for OpenAI Responses API endpoints, including `provider.json` loading, direct construction, reasoning passthrough, and system-prompt to `instructions` mapping.

🧩 **Selfref Fork Context Fixes**: child forks now inherit the pre-fork context snapshot instead of the parent's pending tool-call scene, and `gather_all()` results now expose both `response` and `result` for easier model-generated parsing.

🧪 **Regression Coverage + Examples**: added focused tests for the Responses adapter and selfref fork behavior, plus a new `response_api_example.py` TUI demo using runtime selfref and file tools.

📘 **Docs + Skills Sync**: README, Mintlify docs, packaged skills, and examples have been updated to reflect Responses support, refined selfref fork semantics, and current runtime-primitive guidance. See **[CHANGELOG](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/CHANGELOG.md)** for details.

### 📚 Complete Documentation

> Read detailed documentation: [Chinese Docs](https://simplellmfunc.cn/) | [English Docs](https://simplellmfunc.cn/en)
-----

## 💡 Project Introduction

**SimpleLLMFunc** is a lightweight yet comprehensive LLM/Agent application development framework. Its core philosophy is:

### 🎯 Core Design Philosophy

- **"LLM as Function"** - Treat LLM calls as ordinary Python function calls
- **"Prompt as Code"** - Prompts are directly written in function DocStrings, clear at a glance
- **"Code as Doc"** - Function definitions serve as complete documentation

Through simple decorators, you can integrate LLM capabilities into Python applications with minimal code and the most intuitive approach.

### 🤔 Problems Solved

If you've encountered these dilemmas in LLM development:

1. **Over-abstraction** - Low-code frameworks introduce too much abstraction for custom functionality, making code difficult to understand and maintain
2. **Lack of type safety** - Workflow frameworks lack type hints, leading to errors in complex flows and uncertainty about return formats
3. **Steep learning curve** - Frameworks like LangChain have cumbersome documentation, requiring extensive reading just to implement simple requirements
4. **Flow limitations** - Many frameworks only support DAG (Directed Acyclic Graph), unable to build complex logic with loops or branches
5. **Code duplication** - Without frameworks, you have to manually write API call code, repeating the same work every time, with prompts scattered throughout the code
6. **Insufficient observability** - Lack of complete log tracking and performance monitoring capabilities

**SimpleLLMFunc** is designed specifically to solve these pain points.

### ✨ Core Advantages

- ✅ **Code as Documentation** - Prompts in function DocStrings, clear at a glance
- ✅ **Type Safety** - Python type annotations + Pydantic models, enjoy IDE code completion and type checking
- ✅ **Extremely Simple** - Only one decorator needed, automatically handles API calls, message building, response parsing
- ✅ **Complete Freedom** - Function-based design, supports arbitrary flow control logic (loops, branches, recursion, etc.)
- ✅ **Async Native** - Full async support, naturally adapts to high-concurrency scenarios, no additional configuration needed
- ✅ **Complete Features** - Built-in tool system, multimodal support, API key management, traffic control, structured logging, observability integration
- ✅ **Provider Agnostic** - OpenAI-compatible adaptation, easily switch between multiple model vendors
- ✅ **Responses API Ready** - Includes `OpenAIResponsesCompatible` so the same decorator model can target OpenAI Responses API endpoints
- ✅ **Easy to Extend** - Modular design, supports custom LLM interfaces and tools

> ⚠️ **Important** - All LLM interaction decorators (`@llm_function`, `@llm_chat`, `@tool`, etc.) support decorating both sync and async functions, but all returned results are async functions. Please call them using `await` or `asyncio.run()`.

-----

## 🚀 Quick Start

### Installation

**Method 1: PyPI (Recommended)**

```bash
pip install SimpleLLMFunc
```

**Method 2: Source Installation**

```bash
git clone https://github.com/NiJingzhe/SimpleLLMFunc.git
cd SimpleLLMFunc
poetry install
```

### Initial Configuration

1. Copy configuration template:

```bash
cp env_template .env
```

2. Configure API keys and other parameters in `.env`. It's recommended to configure `LOG_DIR` and `LANGFUSE_BASE_URL`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` for logging and Langfuse tracking.

3. Check `examples/provider_template.json` to understand how to configure multiple LLM providers

### Export Packaged Agent Skills

If you want to install the bundled Agent Skills into tools such as OpenCode, export them with:

```bash
simplellmfunc-skill usage ~/.config/opencode/skills
simplellmfunc-skill developer ~/.config/opencode/skills
```

This creates:

- `~/.config/opencode/skills/simplellmfunc`
- `~/.config/opencode/skills/simplellmfunc-developer`

Use `--force` if you want to overwrite an existing exported skill folder.

### A Simple Example

```python
import asyncio
from SimpleLLMFunc import llm_function, OpenAICompatible

# Load LLM interface from configuration file
llm = OpenAICompatible.load_from_json_file("provider.json")["your_provider"]["model"]

@llm_function(llm_interface=llm)
async def classify_sentiment(text: str) -> str:
    """
    Analyze the sentiment tendency of text.

    Args:
        text: Text to analyze

    Returns:
        Sentiment classification, can be 'positive', 'negative', or 'neutral'
    """
    pass  # Prompt as Code!

async def main():
    result = await classify_sentiment("This product is amazing!")
    print(f"Sentiment classification: {result}")

asyncio.run(main())
```

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| **@llm_function decorator** | Transform any async function into an LLM-driven function, automatically handles Prompt building, API calls, and response parsing |
| **@llm_chat decorator** | Build conversational Agents, supports streaming responses and tool calls |
| **@tool decorator** | Register async functions as LLM-available tools, supports multimodal returns (images, text, etc.) |
| **Type Safety** | Python type annotations + Pydantic models ensure type correctness, enjoy IDE code completion |
| **Async Native** | Fully async design, native asyncio support, naturally adapts to high-concurrency scenarios |
| **Multimodal Support** | Supports `Text`, `ImgUrl`, `ImgPath` multimodal input/output |
| **OpenAI Compatible** | Supports any OpenAI API-compatible model service (OpenAI, Deepseek, Claude, LocalLLM, etc.) |
| **API Key Management** | Automatic load balancing of multiple API keys, optimize resource utilization |
| **Traffic Control** | Token bucket algorithm implements intelligent traffic smoothing, prevents rate limiting |
| **Structured Logging** | Complete trace_id tracking, automatically records requests/responses/tool calls |
| **Observability Integration** | Integrated Langfuse, complete LLM observability support |
| **Flexible Configuration** | JSON format provider configuration, easily manage multiple models and vendors |

## 📖 Detailed Guide

### 1. LLM Function Decorator - "Prompt As Code"

The core philosophy of SimpleLLMFunc is **"Prompt as Code, Code as Doc"**. By writing Prompts directly in function DocStrings, it achieves:

| Advantage | Description |
|-----------|-------------|
| **Code Readability** | Prompts are tightly integrated with functions, no need to search for Prompt variables everywhere |
| **Type Safety** | Type annotations + Pydantic models ensure input/output correctness |
| **IDE Support** | Complete code completion and type checking |
| **Self-documenting** | DocString serves as both function documentation and LLM Prompt |

#### @llm_function - Stateless Functions

```python
"""
Example using LLM function decorator
"""
import asyncio
from typing import List
from pydantic import BaseModel, Field
from SimpleLLMFunc import llm_function, OpenAICompatible, app_log

# Define a Pydantic model as return type
class ProductReview(BaseModel):
    rating: int = Field(..., description="Product rating, 1-5 points")
    pros: List[str] = Field(..., description="List of product advantages")
    cons: List[str] = Field(..., description="List of product disadvantages")
    summary: str = Field(..., description="Review summary")

# Use decorator to create an LLM function
@llm_function(
    llm_interface=OpenAICompatible.load_from_json_file("provider.json")["volc_engine"]["deepseek-v3-250324"]
)
async def analyze_product_review(product_name: str, review_text: str) -> ProductReview:
    """You are a professional product review expert who needs to objectively analyze the following product review and generate a structured review report.
    
    The report should include:
    1. Overall product rating (1-5 points)
    2. List of main product advantages
    3. List of main product disadvantages
    4. Summary evaluation
    
    Rating rules:
    - 5 points: Perfect, almost no disadvantages
    - 4 points: Excellent, advantages clearly outweigh disadvantages
    - 3 points: Average, advantages and disadvantages are basically equal
    - 2 points: Poor, disadvantages clearly outweigh advantages
    - 1 point: Very poor, almost no advantages
    
    Args:
        product_name: Name of the product to review
        review_text: User's review content of the product
        
    Returns:
        A structured ProductReview object containing rating, advantages list, disadvantages list, and summary
    """
    pass  # Prompt as Code, Code as Doc

async def main():
    
    app_log("Starting example code")
    # Test product review analysis
    product_name = "XYZ Wireless Headphones"
    review_text = """
    I've been using these XYZ wireless headphones for a month. The sound quality is very good, especially the bass performance is excellent,
    and they're comfortable to wear, can be used for long periods without fatigue. The battery life is also strong, can last about 8 hours after full charge.
    However, the connection is occasionally unstable, sometimes suddenly disconnects. Also, the touch controls are not sensitive enough, often need to click multiple times to respond.
    Overall, these headphones have great value for money, suitable for daily use, but if you need them for professional audio work, they might not be enough.
    """
    
    try:
        print("\n===== Product Review Analysis =====")
        result = await analyze_product_review(product_name, review_text)
        # result is directly a Pydantic model instance
        # no need to deserialize
        print(f"Rating: {result.rating}/5")
        print("Advantages:")
        for pro in result.pros:
            print(f"- {pro}")
        print("Disadvantages:")
        for con in result.cons:
            print(f"- {con}")
        print(f"Summary: {result.summary}")
    except Exception as e:
        print(f"Product review analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

Output:

```text
===== Product Review Analysis =====
Rating: 4/5
Advantages:
- Very good sound quality, especially excellent bass performance
- Comfortable to wear, can be used for long periods without fatigue
- Strong battery life, can last about 8 hours after full charge
- Great value for money, suitable for daily use
Disadvantages:
- Connection occasionally unstable, sometimes suddenly disconnects
- Touch controls not sensitive enough, often need to click multiple times to respond
- Might not be enough for professional audio work
Summary: Excellent sound quality and battery life, comfortable to wear, but insufficient connection stability and touch control sensitivity, suitable for daily use but not for professional audio work.
```

**Key Points:**

- ✅ Only need to declare function, types, and DocString, decorator handles the rest automatically
- ✅ Directly returns Pydantic object, no manual deserialization needed
- ✅ Supports complex nested Pydantic models
- ✅ Small models may not output correct JSON, framework will automatically retry

#### @llm_chat - Conversations and Agents

Also supports creating **conversational functions** and **Agent systems**. llm_chat supports:

- Multi-turn conversation history management
- Real-time streaming responses
- LLM tool calls and automatic execution
- Flexible return modes (text or raw response)

If you want to build a complete Agent framework, you can refer to our sister project [SimpleManus](https://github.com/NiJingzhe/SimpleManus).

#### @tui - Out-of-the-box terminal UI for llm_chat

SimpleLLMFunc provides a ready-to-use Textual TUI powered by event streaming:

- Alternating user/assistant chat timeline
- Streaming markdown rendering
- Tool call argument/result panels
- Model and tool stats (latency, token usage)
- Custom tool-event hooks for live tool output updates
- Origin-aware event routing for parent and forked agent calls
- Built-in selfref fork lifecycle and stream visualization
- Built-in quit controls: `/exit` `/quit` `/q`, `Ctrl+Q`, `Ctrl+C`

```python
from SimpleLLMFunc import llm_chat, tui


@tui(custom_event_hook=[...])
@llm_chat(llm_interface=my_llm_interface, stream=True)
async def agent(message: str, history=None):
    """Your agent prompt"""


if __name__ == "__main__":
    agent()
```

See `examples/tui_chat_example.py` for a full example.

Each `EventYield` includes `origin` metadata. This is especially useful for forked agent trees:

```python
from SimpleLLMFunc.hooks import is_event_yield

async for output in agent("split this into parallel subtasks"):
    if not is_event_yield(output):
        continue

    if output.origin.fork_id:
        print(
            f"[fork:{output.origin.fork_id} depth={output.origin.fork_depth}] "
            f"{output.event.event_type}"
        )
    else:
        print(f"[main] {output.event.event_type}")
```

#### Async Native Design

Both `llm_function` and `llm_chat` are natively async designed, no additional configuration needed:

```python
from SimpleLLMFunc import llm_function, llm_chat


@llm_function(llm_interface=my_llm_interface)
async def async_analyze_text(text: str) -> str:
    """Async text content analysis"""
    pass


@llm_chat(llm_interface=my_llm_interface, stream=True)
async def async_chat(message: str, history: List[Dict[str, str]]):
    """Async chat functionality, supports streaming responses"""
    pass


async def main():
    result = await async_analyze_text("Text to analyze")

    async for response, updated_history in async_chat("Hello", []):
        print(response)
```

#### Tool-Call Limit Default

`llm_function` and `llm_chat` now default to `max_tool_calls=None`.

- `None` means SimpleLLMFunc does not impose a framework-level tool-call iteration cap by default
- This is better for long-horizon agents and deep tool-using workflows
- If you want stricter protection against looping or runaway tool plans, pass an explicit integer such as `max_tool_calls=8`

```python
@llm_function(llm_interface=my_llm_interface, max_tool_calls=None)
async def analyze(text: str) -> str:
    """Analyze the text."""
    pass


@llm_chat(llm_interface=my_llm_interface, stream=True, max_tool_calls=12)
async def cautious_agent(message: str, history=None):
    """Chat agent with an explicit safety cap."""
    pass
```

#### Multimodal Support

SimpleLLMFunc supports multiple modalities of input and output, allowing LLMs to process text, images, and other content:

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.type import ImgPath, ImgUrl, Text

@llm_function(llm_interface=my_llm_interface)
async def analyze_image(
    description: Text,           # Text description
    web_image: ImgUrl,          # Web image URL
    local_image: ImgPath        # Local image path
) -> str:
    """Analyze images and provide detailed explanations based on descriptions
    
    Args:
        description: Specific requirements for image analysis
        web_image: Web image URL to analyze
        local_image: Local reference image path for comparison
        
    Returns:
        Detailed image analysis results
    """
    pass

import asyncio


async def main():
    result = await analyze_image(
        description=Text("Please describe the differences between these two images in detail"),
        web_image=ImgUrl("https://example.com/image.jpg"),
        local_image=ImgPath("./reference.jpg")
    )
    print(result)


asyncio.run(main())
```

#### Decorator Parameters and Advanced Features

@llm_function and @llm_chat support rich configuration parameters:

```python
@llm_function(
    llm_interface=llm_interface,          # LLM interface instance
    toolkit=[tool1, tool2],                # Tool list
    retry_on_exception=True,               # Auto retry on exception
    timeout=60                              # Timeout setting
)
async def my_function(param: str) -> str:
    """Supports {language} {style} analysis"""
    pass

result = await my_function(
    "some input",
    _template_params={
        "language": "English",
        "style": "Professional",
    },
)
```

`_template_params` is passed **at call time** and only used to format the function DocString via `str.format`. It is removed before signature binding and is not part of the LLM input. If a placeholder is missing, the original DocString is used (with a warning).

### 2. LLM Provider Interface

SimpleLLMFunc provides flexible LLM interface support:

**Supported Providers and Adapter Paths:**

- ✅ OpenAI (GPT-4, GPT-3.5, etc.)
- ✅ Deepseek
- ✅ Anthropic Claude
- ✅ Volc Engine Ark
- ✅ Baidu Qianfan
- ✅ Local LLM (Ollama, vLLM, etc.)
- ✅ Any OpenAI API-compatible service
- ✅ OpenAI Responses API endpoints via `OpenAIResponsesCompatible`

#### Quick Integration Example

```python
from SimpleLLMFunc import APIKeyPool, OpenAICompatible, OpenAIResponsesCompatible

# Method 1: Load from JSON configuration file
provider_config = OpenAICompatible.load_from_json_file("provider.json")
llm = provider_config["deepseek"]["v3-turbo"]
responses_config = OpenAIResponsesCompatible.load_from_json_file("provider.json")
responses_llm = responses_config["openrouter"]["gpt-5.4"]

# Method 2: Direct creation
llm = OpenAICompatible(
    api_key_pool=APIKeyPool(["sk-xxx"], provider_id="deepseek-chat"),
    base_url="https://api.deepseek.com/v1",
    model_name="deepseek-chat",
)

responses_llm = OpenAIResponsesCompatible(
    api_key_pool=APIKeyPool(["sk-xxx"], provider_id="openrouter-gpt-5.4-responses"),
    base_url="https://openrouter.ai/api/v1",
    model_name="gpt-5.4",
)

@llm_function(llm_interface=llm)
async def my_function(text: str) -> str:
    """Process text"""
    pass
```

For Responses API usage, keep writing normal docstrings and chat history. The adapter is responsible for mapping the selected system prompt to Responses `instructions` and forwarding `reasoning={...}`.

#### provider.json Configuration File

```json
{
    "deepseek": [
        {
            "model_name": "deepseek-v3.2",
            "api_keys": ["sk-your-api-key-1", "sk-your-api-key-2"],
            "base_url": "https://api.deepseek.com/v1",
            "max_retries": 5,
            "retry_delay": 1.0,
            "rate_limit_capacity": 10,
            "rate_limit_refill_rate": 1.0
        }
    ],
    "openai": [
        {
            "model_name": "gpt-4",
            "api_keys": ["sk-your-api-key"],
            "base_url": "https://api.openai.com/v1",
            "max_retries": 5,
            "retry_delay": 1.0,
            "rate_limit_capacity": 10,
            "rate_limit_refill_rate": 1.0
        }
    ]
}
```

#### Custom LLM Interface

You can implement completely custom LLM interfaces by inheriting from the `LLM_Interface` base class:

```python
from SimpleLLMFunc.interface import LLM_Interface

class CustomLLMInterface(LLM_Interface):
    async def call_llm(self, messages, **kwargs):
        # Implement your own LLM calling logic
        pass
```

### 3. Logging and Observability System

SimpleLLMFunc includes complete log tracking and observability capabilities to help you gain deep insights into LLM application performance.

#### Core Features

| Feature | Description |
|---------|-------------|
| **Trace ID Auto Tracking** | Each call automatically generates a unique trace_id, associating all related logs |
| **Structured Logging** | Supports multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| **Context Propagation** | Automatically preserves context in async environments, trace_id automatically associated |
| **Colored Output** | Beautified console output, improves readability |
| **File Persistence** | Automatically writes to local log files, supports rotation and archiving |
| **Langfuse Integration** | Out-of-the-box observability integration, visualizes LLM call chains |

#### Trace Example

```
GLaDos_c790a5cc-e629-4cbd-b454-ab102c42d125  <- Auto-generated trace_id
├── Function call input parameters
├── LLM request content
├── Token usage statistics
├── Tool calls (if any)
├── LLM response content
└── Execution time and performance metrics
```

#### Logging Usage Example

```python
from SimpleLLMFunc.logger import app_log, push_error, log_context

# 1. Basic logging
app_log("Starting request processing", trace_id="request_123")
push_error("Error occurred", trace_id="request_123", exc_info=True)

# 2. Use context manager to automatically associate logs
with log_context(trace_id="task_456", function_name="analyze_text"):
    app_log("Starting text analysis")  # Automatically inherits context trace_id
    try:
        # Execute operations...
        app_log("Analysis completed")
    except Exception:
        push_error("Analysis failed", exc_info=True)  # Also automatically inherits trace_id
```

### 4. Tool System - Let LLMs Interact with Environment

SimpleLLMFunc implements a complete tool system, allowing LLMs to call external functions and APIs. Tools support two definition methods.

#### @tool Decorator Method (Recommended)

The most concise way: use the `@tool` decorator to register async functions as LLM-available tools.

> ⚠️ The `@tool` decorator only supports decorating functions defined with `async def`

```python
from pydantic import BaseModel, Field
from SimpleLLMFunc.tool import tool

# Define Pydantic model for complex parameters
class Location(BaseModel):
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")

# Use decorator to create tool
@tool(name="get_weather", description="Get weather information for specified location")
async def get_weather(location: Location, days: int = 1) -> dict:
    """
    Get weather forecast for specified location
    
    Args:
        location: Location information, including latitude and longitude
        days: Forecast days, default is 1 day
        
    Returns:
        Weather forecast information
    """
    # Actual implementation would call weather API
    return {
        "location": f"{location.latitude},{location.longitude}",
        "forecast": [{"day": i, "temp": 25, "condition": "Sunny"} for i in range(days)]
    }
```

**Advantages:**

- ✅ Concise and intuitive, automatically extracts parameter information from function signature
- ✅ Supports Python native types and Pydantic models
- ✅ Can still be called directly after decoration, convenient for unit testing
- ✅ Supports multimodal returns (text, images, etc.)
- ✅ Can be stacked: one function can be decorated with both `@llm_function` and `@tool`

#### Multimodal Tool Example

```python
from SimpleLLMFunc.tool import tool
from SimpleLLMFunc.type import ImgPath, ImgUrl

@tool(name="generate_chart", description="Generate charts based on data")
async def generate_chart(data: str, chart_type: str = "bar") -> ImgPath:
    """
    Generate charts based on provided data
    
    Args:
        data: CSV format data
        chart_type: Chart type, default is bar chart
        
    Returns:
        Generated chart file path
    """
    # Actual implementation would generate chart and save locally
    chart_path = "./generated_chart.png"
    # ... Chart generation logic
    return ImgPath(chart_path)

@tool(name="search_web_image", description="Search web images")
async def search_web_image(query: str) -> ImgUrl:
    """
    Search web images
    
    Args:
        query: Search keywords
        
    Returns:
        Found image URL
    """
    # Actual implementation would call image search API
    image_url = "https://example.com/search_result.jpg"
    return ImgUrl(image_url)
```

#### Class Inheritance Method (Compatible)

You can also define tools by inheriting from the `Tool` base class (for complex logic or special requirements):

```python
from SimpleLLMFunc.tool import Tool

class WebSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search information on the internet"
        )

    async def run(self, query: str, max_results: int = 5) -> dict:
        """Execute web search"""
        # Implement search logic
        return {"results": [...]}
```

#### Tool Integration into LLM Functions

All tools can be passed to `@llm_function` or `@llm_chat`:

```python
@llm_function(
    llm_interface=llm,
    toolkit=[get_weather, search_web, WebSearchTool()],
)
async def answer_question(question: str) -> str:
    """
    Answer user questions, use tools when necessary.

    Args:
        question: User's question

    Returns:
        Answer
    """
    pass
```

### 5. API Key Management and Traffic Control

SimpleLLMFunc provides production-level key and traffic management capabilities.

#### API Key Load Balancing

- Supports multiple API key configurations
- Automatically selects the key with lowest load
- Uses min-heap algorithm for efficient optimal key selection
- Automatically tracks usage for each key

#### Traffic Control

- Token bucket algorithm implements traffic smoothing
- Prevents API rate limiting
- Supports burst traffic buffering
- Can configure rate limiting parameters for each model in `provider.json`

For example, configure in provider.json:

```json
{
    "model_config": {
        "rate_limit": 100,      // Maximum 100 requests per minute
        "burst": 10              // Maximum 10 burst requests
    }
}
```

### 7. Project Structure and Module Organization

SimpleLLMFunc adopts modular design with clear structure, easy to maintain:

#### Core Modules

```
SimpleLLMFunc/
├── SimpleLLMFunc/
│   ├── llm_decorator/         # LLM decorator module
│   │   ├── llm_function_decorator.py    # @llm_function implementation
│   │   ├── llm_chat_decorator.py        # @llm_chat implementation
│   │   ├── invocation_spec.py           # Unified invocation contracts
│   │   ├── invocation_builder.py        # InvocationSpec builders
│   │   ├── prompt_contract.py           # Prompt/seed helpers
│   │   ├── signature.py                 # Signature binding + log context
│   │   └── utils/                       # Decorator utilities
│   ├── tool/                  # Tool system
│   │   └── tool.py            # @tool decorator and Tool base class
│   ├── builtin/               # Builtin tools
│   │   ├── pyrepl.py          # Python REPL toolset
│   │   └── self_reference.py  # SelfReference memory/fork backend
│   ├── hooks/                 # Event stream system
│   │   ├── events.py          # ReAct event definitions
│   │   ├── stream.py          # Event/response stream wrappers
│   │   ├── event_emitter.py   # Tool custom event emitter
│   │   └── event_bus.py       # Unified event ingress + origin metadata
│   ├── interface/             # LLM interface layer
│   │   ├── llm_interface.py   # Abstract base class
│   │   ├── openai_compatible.py    # OpenAI compatible implementation
│   │   ├── openai_responses_compatible.py # OpenAI Responses API adapter
│   │   ├── key_pool.py        # API key management
│   │   └── token_bucket.py    # Traffic control
│   ├── base/                  # Core execution engine
│   │   ├── ReAct.py           # ReAct engine and tool calls
│   │   ├── messages/          # Message building
│   │   ├── post_process.py    # Response parsing and type conversion
│   │   ├── tool_call/         # Tool call extraction/execution/validation
│   │   └── type_resolve/      # Type resolution
│   ├── logger/                # Logging and observability
│   │   ├── logger.py          # Logging API
│   │   ├── logger_config.py   # Logging configuration
│   │   └── context_manager.py # Context management
│   ├── observability/         # Observability integration
│   │   └── langfuse_client.py # Langfuse integration
│   ├── type/                  # Multimodal types
│   │   └── __init__.py        # Text, ImgUrl, ImgPath, etc.
│   ├── utils/
│   │   ├── __init__.py        # Shared utility exports
│   │   └── tui/               # Textual chat UI integration
│   ├── config.py              # Global configuration
│   └── __init__.py            # Package initialization and API exports
├── examples/                  # Usage examples
│   ├── llm_function_pydantic_example.py  # Structured output examples
│   ├── event_stream_chatbot.py      # Chat + event stream examples
│   ├── parallel_toolcall_example.py # Concurrency examples
│   ├── multi_modality_toolcall.py   # Multimodal examples
│   ├── pyrepl_example.py            # Builtin PyRepl usage
│   ├── runtime_primitives_basic_example.py # Local runtime memory primitives
│   ├── tui_general_agent_example.py  # General TUI agent demo (selfref + file tools)
│   ├── response_api_example.py      # Responses API TUI agent demo
│   ├── custom_tool_event_example.py # Custom tool event examples
│   ├── tui_chat_example.py          # Textual TUI example
│   ├── provider.json          # Provider configuration examples
│   └── provider_template.json # Configuration template
├── pyproject.toml             # Poetry configuration
├── README.md                  # Project documentation (you are here)
├── CHANGELOG.md               # Changelog
└── env_template               # Environment variable template
```

#### Module Responsibility Description

| Module | Responsibility |
|--------|----------------|
| **llm_decorator** | Provides @llm_function and @llm_chat decorators |
| **tool** | Tool system, @tool decorator and Tool base class |
| **builtin** | Builtin tools (e.g. persistent Python REPL) |
| **hooks** | Event stream definitions, emitters, and stream wrappers |
| **interface** | LLM interface abstraction plus `OpenAICompatible` and `OpenAIResponsesCompatible` adapters |
| **base** | ReAct engine, message processing, type conversion |
| **logger** | Structured logging, trace_id tracking |
| **observability** | Langfuse integration, complete LLM observability |
| **type** | Multimodal type definitions (Text, ImgUrl, ImgPath) |
| **utils** | Shared utilities and Textual TUI integration |
| **config** | Global configuration and environment variable management |

### Configuration and Environment Variables

SimpleLLMFunc supports flexible configuration:

**Priority (from high to low):**

1. Direct configuration in program
2. Environment variables
3. `.env` file

**Common Configuration:**

```bash
# .env file example
LOG_DIR=./logs                          # Log directory (optional)
LOG_LEVEL=INFO                          # Log level, only controls console log output, doesn't affect file log output
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # Langfuse base URL (optional)
LANGFUSE_PUBLIC_KEY=pk_xxx             # Langfuse public key (optional)
LANGFUSE_SECRET_KEY=sk_xxx             # Langfuse secret key (optional)
LANGFUSE_EXPORT_ALL_SPANS=true         # Export all OpenTelemetry spans (optional)
```

## 🎯 Common Use Cases

SimpleLLMFunc is suitable for various LLM application development scenarios:

### Data Processing and Analysis

```python
@llm_function(llm_interface=llm)
async def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities (people, places, organizations, etc.) from text"""
    pass

# Usage
entities = await extract_entities("John works at Apple in Beijing")
# Returns: {"person": ["John"], "location": ["Beijing"], "organization": ["Apple"]}
```

### Intelligent Agents and Conversations

```python
@llm_chat(llm_interface=llm, toolkit=[search_tool, calculator_tool])
async def agent(user_message: str, history: List[Dict]) -> str:
    """Intelligent assistant that can search information and do math calculations"""
    pass

# Usage
response = await agent("What's the weather like in Beijing tomorrow? And calculate what temperature it would be if it drops 5 degrees", [])
```

### Batch Data Processing

```python
import asyncio

@llm_function(llm_interface=llm)
async def classify_text(text: str) -> str:
    """Classify text"""
    pass

# Batch processing, fully utilize async
texts = ["Text 1", "Text 2", "Text 3", ...]
results = await asyncio.gather(*[classify_text(t) for t in texts])
```

### Multimodal Content Processing

```python
from SimpleLLMFunc.type import ImgPath, ImgUrl

@llm_function(llm_interface=llm)
async def analyze_images(local_img: ImgPath, web_img: ImgUrl) -> str:
    """Compare and analyze two images"""
    pass
```

## 📚 Running Example Code

The project includes rich examples for quick start:

```bash
# Install dependencies
pip install SimpleLLMFunc

# Set up API keys
cp env_template .env
# Edit .env file, enter your API keys

# Run examples
python examples/llm_function_pydantic_example.py
python examples/event_stream_chatbot.py
python examples/parallel_toolcall_example.py
python examples/runtime_primitives_basic_example.py
python examples/tui_general_agent_example.py
```

## 🤝 Contributing Guide

Welcome to submit Issues and Pull Requests!

- 🐛 **Bug Report** - Report issues in [GitHub Issues](https://github.com/NiJingzhe/SimpleLLMFunc/issues)
- ✨ **Feature Suggestions** - Welcome to discuss new features
- 📝 **Documentation Improvement** - Help improve documentation
- 💡 **Example Code** - Share your use cases

## 📖 More Resources

- 📚 [Chinese Docs](https://simplellmfunc.cn/) | [English Docs](https://simplellmfunc.cn/en)
- 🔄 [Changelog](CHANGELOG.md)
- 🔗 [GitHub Repository](https://github.com/NiJingzhe/SimpleLLMFunc)
- 🤖 [SimpleManus (Agent Framework)](https://github.com/NiJingzhe/SimpleManus)

## Star History

<a href="https://www.star-history.com/#NiJingzhe/SimpleLLMFunc&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=NiJingzhe/SimpleLLMFunc&type=Date" />
 </picture>
</a>

## Citation

If you have used SimpleLLMFunc in your research or projects, please cite the following information:

```bibtex
@software{ni2025simplellmfunc,
  author = {Jingzhe Ni},
  month = {February},
  title = {{SimpleLLMFunc: A New Approach to Build LLM Applications}},
  url = {https://github.com/NiJingzhe/SimpleLLMFunc},
  version = {0.7.8},
  year = {2026}
}
```

## License

MIT
