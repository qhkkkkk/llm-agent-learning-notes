"""按 Runtime Context 中的用户角色动态生成本轮系统提示词。

默认脚本角色为专家；上下文缺失时使用初学者规则。
导入模块不会读取密钥或调用模型，测试可注入离线模型。
"""

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


class Context(TypedDict):
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    context = request.runtime.context or {}  # 没传 context 时也可使用默认角色。
    user_role = context.get("user_role", "初学者")
    base_prompt = "你是一位乐于助人的专家。"

    if user_role == "初学者":
        return f"{base_prompt}用简单易懂的语言描述概念，避免使用专业术语。"
    elif user_role == "专家":
        return f"{base_prompt}尽量提供详细的技术解答，需要使用专业术语。"

    return base_prompt


def build_agent(model: BaseChatModel | None = None):
    """创建同一个可按每次 context 调整提示词的 Agent。"""
    if model is None:
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("请在环境变量或 .env 中配置 OPENROUTER_API_KEY。")
        model = init_chat_model(
            model=os.getenv("DYNAMIC_PROMPT_MODEL", "openai/gpt-5-mini"),
            model_provider="openrouter",
        )
    return create_agent(
        model=model,
        context_schema=Context,
        middleware=[user_role_prompt],
    )


def main() -> None:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    agent = build_agent()
    response = agent.invoke(
        {"messages": [{
            "role": "user",
            "content": os.getenv("DYNAMIC_PROMPT_QUERY", "解释一下机器学习"),
        }]},
        context={"user_role": os.getenv("DYNAMIC_PROMPT_ROLE", "专家")},
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
