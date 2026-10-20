"""连接计算器和天气 MCP 服务，将两边工具交给同一个异步 Agent。"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def build_client(weather_url: str | None = None) -> MultiServerMCPClient:
    port = os.getenv("MCP_WEATHER_PORT", "8000")
    url = weather_url or os.getenv("MCP_WEATHER_URL", f"http://127.0.0.1:{port}/mcp")
    return MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(Path(__file__).resolve().with_name("math_server.py"))],
        },
        "weather": {
            "transport": "streamable-http",
            "url": url,
        },
    })


def build_model() -> BaseChatModel:
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("请在环境变量或 .env 中配置 OPENROUTER_API_KEY。")
    return init_chat_model(
        model=os.getenv("MCP_AGENT_MODEL", "openai/gpt-5-mini"),
        model_provider="openrouter",
    )


async def run_demo(
    model: BaseChatModel | None = None,
    *,
    weather_url: str | None = None,
) -> dict:
    """两次调用都在两个有效会话内部完成；测试可注入离线模型。"""
    if model is None:
        model = build_model()
    client = build_client(weather_url)

    async with (
        client.session("math") as math_session,
        client.session("weather") as weather_session,
    ):
        math_tools = await load_mcp_tools(math_session)
        weather_tools = await load_mcp_tools(weather_session)
        tools = math_tools + weather_tools

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt="你是一个乐于助人的助手，回答问题必须使用工具完成！",
        )
        math_result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "(3 + 4) * 5 等于多少"}],
        })
        weather_result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "西安的天气怎么样？"}],
        })
        return {
            "tool_names": [t.name for t in tools],
            "math_result": math_result,
            "weather_result": weather_result,
        }


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    result = await run_demo()
    print("已加载工具：", ", ".join(result["tool_names"]))
    print("计算结果：", result["math_result"]["messages"][-1].content)
    print("天气结果：", result["weather_result"]["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
