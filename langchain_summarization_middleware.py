"""使用内置摘要中间件管理天气 Agent 的线程消息。

主模型与摘要模型分别显式选择 OpenRouter。天气工具只返回演示数据。
导入本模块不会读取密钥或调用模型。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from openrouter.errors import OpenRouterError


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的演示天气；返回值固定为阳光明媚。"""
    return f"在{city}总是阳光明媚！"


def build_openrouter_model(model_name: str) -> BaseChatModel:
    """创建独立模型客户端，自动读取 OPENROUTER_API_KEY。"""
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("请在环境变量或 .env 中配置 OPENROUTER_API_KEY。")
    return init_chat_model(
        model=model_name,
        model_provider="openrouter",
    )


def build_agent(
    chat_model: BaseChatModel | None = None,
    summary_model: BaseChatModel | None = None,
    *,
    trigger=("tokens", 30),
    keep=("messages", 3),
    token_counter=count_tokens_approximately,
):
    """组合两个模型和中间件；测试可注入模型及确定性的 Token 计数器。"""
    if chat_model is None:
        chat_model = build_openrouter_model(
            os.getenv("SUMMARY_CHAT_MODEL", "openai/gpt-5-mini")
        )
    if summary_model is None:
        summary_model = build_openrouter_model(
            os.getenv("SUMMARY_MODEL", "openai/gpt-5-mini")
        )

    return create_agent(
        model=chat_model,
        tools=[get_weather_for_location],
        system_prompt="你是一位乐于助人的助手。",
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model=summary_model,
                trigger=trigger,
                keep=keep,
                token_counter=token_counter,
            ),
        ],
    )


def print_messages(messages: list[BaseMessage]) -> None:
    """显示消息角色、摘要标记和内容，不展开供应商元数据。"""
    for index, message in enumerate(messages, start=1):
        is_summary = message.additional_kwargs.get("lc_source") == "summarization"
        marker = " [摘要]" if is_summary else ""
        print(f"{index}. {type(message).__name__}{marker}：{message.content}")
        for call in getattr(message, "tool_calls", []):
            print(f"   工具调用：{call['name']}({call['args']})")


def run_demo() -> None:
    """在同一个线程中从北京天气继续追问太原，观察消息变化。"""
    load_dotenv(Path(__file__).with_name(".env"))
    agent = build_agent()
    config = {
        "configurable": {"thread_id": os.getenv("SUMMARY_THREAD_ID", "11")},
        "recursion_limit": 30,
    }
    prompts = ["北京的天气怎么样？", "太原呢？"]

    try:
        for turn, prompt in enumerate(prompts, start=1):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
            )
            print(f"第 {turn} 轮用户：{prompt}")
            print(f"第 {turn} 轮回复：{response['messages'][-1].content}")
            print_messages(response["messages"])
    except OpenRouterError as error:
        print("状态码：", error.status_code)
        print("详细错误：", error.body)
        raise


if __name__ == "__main__":
    run_demo()
