"""按 Agent 状态筛选本轮模型可见工具；不是运行时新增工具的实现。

未认证仅展示 public_，已认证仅展示 private_，保留练习的互斥规则。
三个搜索工具只返回演示字符串；auth 是演示标记，不是真实登录认证。
导入模块不会读取密钥或调用模型。
"""

import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool


@tool
def public_search(query: str) -> str:
    """公开搜索演示：返回无需认证的固定示例文本。"""
    result = f"[公开搜索结果] 关于 '{query}' 的基础信息：这是公开可获取的内容。"
    print(result)
    return result


@tool
def private_search(query: str) -> str:
    """私有搜索演示：返回固定示例文本，不访问真实私密数据。"""
    result = f"[私有搜索结果] 关于 '{query}' 的私密数据：仅限认证用户查看。"
    print(result)
    return result


@tool
def advanced_search(query: str) -> str:
    """高级搜索演示：返回固定的分析示例文本。"""
    result = f"[高级搜索结果] 关于 '{query}' 的深度分析报告：包含详细统计和趋势。"
    print(result)
    return result


class State(AgentState):
    auth: bool  # 由调用方或上游逻辑提供；这里没有自动验证身份。


@wrap_model_call(state_schema=State)
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """筛选已注册工具，然后继续模型调用链。"""
    state = request.state
    is_auth = state.get("auth", False)

    if not is_auth:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    else:
        tools = [t for t in request.tools if t.name.startswith("private_")]
        request = request.override(tools=tools)

    return handler(request)  # 框架将本轮工具清单绑定给模型，再调用模型。


def build_agent(
    model: BaseChatModel | None = None,
    *,
    tools: list[BaseTool] | None = None,
):
    """默认预注册三个工具；测试可以注入离线模型和工具列表。"""
    if model is None:
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("请在环境变量或 .env 中配置 OPENROUTER_API_KEY。")
        model = init_chat_model(
            model=os.getenv("DYNAMIC_TOOLS_MODEL", "openai/gpt-5-mini"),
            model_provider="openrouter",
        )
    return create_agent(
        model=model,
        tools=[public_search, private_search, advanced_search] if tools is None else tools,
        system_prompt="你是一位乐于助人的客服助手。根据用户的问题，选择合适的工具来提供答案。",
        middleware=[state_based_tools],
    )


def main() -> None:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    auth_text = os.getenv("DYNAMIC_TOOLS_AUTH", "true").strip().lower()
    if auth_text not in {"true", "false"}:
        raise ValueError("DYNAMIC_TOOLS_AUTH 必须是 true 或 false。")
    agent = build_agent()
    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": os.getenv("DYNAMIC_TOOLS_QUERY", "北京的天气如何"),
        }],
        "auth": auth_text == "true",
    })
    print(response)


if __name__ == "__main__":
    main()
