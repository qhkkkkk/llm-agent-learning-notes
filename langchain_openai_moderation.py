"""OpenAI 官方审核中间件的同步调用示例。

主模型仍通过 OpenRouter；审核接口需要单独的 OpenAI 官方 API Key。
只有 OpenRouter Key 时，使用 langchain_openrouter_input_moderation.py。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai.middleware import OpenAIModerationError, OpenAIModerationMiddleware
from openai import OpenAI, OpenAIError

from langchain_openrouter_input_moderation import build_openrouter_model, get_weather_for_location


def build_moderation_client():
    """独立配置官方审核客户端，避免复用 OpenRouter 的密钥和兼容地址。"""
    api_key = os.getenv("OPENAI_MODERATION_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("官方审核示例需要单独配置 OPENAI_MODERATION_API_KEY。")
    return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")


def build_agent(
    chat_model=None,
    *,
    moderation_client=None,
    check_input=True,
    check_output=False,
    check_tool_results=False,
    exit_behavior="replace",
):
    """显式声明审核范围；默认行为对应只检查输入。"""
    if chat_model is None:
        chat_model = build_openrouter_model(
            os.getenv("MODERATION_CHAT_MODEL", "openai/gpt-5-mini")
        )
    if moderation_client is None:
        moderation_client = build_moderation_client()
    return create_agent(
        model=chat_model,
        tools=[get_weather_for_location],
        system_prompt="你是一位乐于助人的助手。",
        middleware=[
            OpenAIModerationMiddleware(
                client=moderation_client,
                model="omni-moderation-latest",
                check_input=check_input,
                check_output=check_output,
                check_tool_results=check_tool_results,
                exit_behavior=exit_behavior,
                violation_message="[审核未通过，类别：{categories}]",
            ),
        ],
    )


def run_demo():
    load_dotenv(Path(__file__).with_name(".env"))
    agent = build_agent(exit_behavior=os.getenv("MODERATION_EXIT_BEHAVIOR", "replace"))
    try:
        response = agent.invoke({
            "messages": [{
                "role": "user",
                "content": os.getenv("MODERATION_QUERY", "北京的天气怎么样？"),
            }],
        })
        print(response["messages"][-1].content)
    except OpenAIModerationError as error:
        # 审核已完成且判定未通过，与 HTTP 401 等请求失败不同。
        print("捕获到审核未通过异常：", error)
    except OpenAIError as error:
        print("OpenAI 审核服务请求失败：", error)
        raise


if __name__ == "__main__":
    run_demo()
