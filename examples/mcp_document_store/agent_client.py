"""先经 MCP 读取 FAQ，再用本地工具把缓存文档交给 Agent。"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.documents.base import Blob
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient

FAQ_URI = "file:///help/faq.json"


def build_client() -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "DocumentStore": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(Path(__file__).resolve().with_name("docs_server.py"))],
        },
    })


def build_model() -> BaseChatModel:
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("请在环境变量或 .env 中配置 OPENROUTER_API_KEY。")
    return init_chat_model(
        model=os.getenv("MCP_DOCS_MODEL", "openai/gpt-5-mini"),
        model_provider="openrouter",
    )


def make_doc_tool(blobs: list[Blob]):
    """闭包保存已读到的 Blob；调用工具时不再向 MCP 发请求。"""
    @tool
    def search_aq_docs(query: str) -> str:
        """查阅已加载的重置密码 FAQ；当前不按 query 筛选。"""
        results = []
        for blob in blobs:
            if blob.mimetype == "text/plain":
                results.append(blob.as_string())
        return "\n\n".join(results) if results else "未找到相关文档"

    return search_aq_docs


async def run_demo(model: BaseChatModel | None = None) -> dict:
    client = build_client()
    # 此调用完成资源读取及 Blob 转换，并退出内部 MCP 会话。
    blobs = await client.get_resources("DocumentStore", uris=[FAQ_URI])
    search_docs = make_doc_tool(blobs)
    if model is None:
        model = build_model()
    agent = create_agent(
        model=model,
        tools=[search_docs],
        system_prompt="务必调用工具，根据文档回答；文档未提供的信息请明确说明。",
    )
    return await agent.ainvoke({
        "messages": [{"role": "user", "content": "如何重置密码？请查阅文档"}],
    })


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    response = await run_demo()
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
