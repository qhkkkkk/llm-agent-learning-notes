"""LangChain 天气 Agent：工具、运行上下文、结构化输出与两轮记忆。

天气数据和用户位置均为演示规则，不会访问真实天气服务。
运行前配置 OPENROUTER_API_KEY；导入本模块不会调用模型。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.memory import InMemoryStore
from openrouter.errors import OpenRouterError


@dataclass
class Context:
    """由调用方提供的本次运行上下文。"""

    user_id: str


@dataclass
class ResponseFormat:
    """最终回答的结构；字段格式不等于天气事实已被验证。"""

    punny_response: str
    weather_conditions: str | None = None


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的演示天气；返回值固定为阳光明媚。"""
    return f"在{city}总是阳光明媚！"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据运行上下文中的用户 ID 获取演示位置。"""
    user_id = runtime.context.user_id
    memory_store = runtime.store
    if memory_store is None:
        raise RuntimeError("该工具需要在 create_agent 中配置 store。")

    # namespace + key 定位一条记录；同一个 key 再次 put 会覆盖旧记录。
    memory_store.put(("users",), user_id, {"name": f"name_{user_id}"})
    user_info = memory_store.get(("users",), user_id)
    if user_info is None:
        raise RuntimeError("未能读取刚写入的演示用户记录。")
    print(f"user_name:{user_info.value.get('name')}")

    # 位置由这条规则决定，不是从用户记录中查出的城市。
    return "北京" if user_id == "1" else "上海"


SYSTEM_PROMPT = """你是一位擅长用双关语表达的天气预报专家。
你拥有以下两种工具的使用权：
- get_weather_for_location：使用此功能可获取特定地点的天气情况
- get_user_location：使用此功能可获取用户的当前位置
如果用户向您询问天气情况，请务必先确认其所在位置。如果从问题中不能推断出他们指的是其所在的具体地点，那么请使用“get_user_location”工具来获取他们的位置信息。"""


def build_model() -> BaseChatModel:
    """使用 OpenRouter 集成，自动读取 OPENROUTER_API_KEY。"""
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("请在环境变量或 .env 中配置 OPENROUTER_API_KEY。")
    return init_chat_model(
        model=os.getenv("WEATHER_MODEL", "openai/gpt-5-mini"),
        model_provider="openrouter",
    )


def build_agent(
    model: BaseChatModel | None = None,
    *,
    store: InMemoryStore | None = None,
):
    """创建 Agent；可以注入离线模型和共享 Store 进行验证。"""
    # structured_response 会进入检查点。只登记本例要恢复的自定义类，
    # 同时兼容直接运行时的 __main__ 和被测试模块导入时的真实模块名。
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            (ResponseFormat.__module__, ResponseFormat.__name__),
        ],
    )
    return create_agent(
        model=build_model() if model is None else model,
        name="weather_agent",
        system_prompt=SYSTEM_PROMPT,
        tools=[get_user_location, get_weather_for_location],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=InMemorySaver(serde=serde),
        store=InMemoryStore() if store is None else store,
    )


def run_demo() -> None:
    """复用同一个 Agent 和线程执行两轮对话。"""
    load_dotenv(Path(__file__).with_name(".env"))
    agent = build_agent()
    config = {
        "configurable": {"thread_id": os.getenv("WEATHER_THREAD_ID", "11")},
        "recursion_limit": 30,
    }
    context = Context(user_id=os.getenv("WEATHER_USER_ID", "2"))

    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "我这里外面的天气怎么样？"}]},
            config=config,
            context=context,
        )
        print(response["structured_response"])

        response2 = agent.invoke(
            {"messages": [{"role": "user", "content": "谢谢！"}]},
            config=config,
            context=context,
        )
        print(response2["structured_response"])
    except OpenRouterError as error:
        print("状态码：", error.status_code)
        print("详细错误：", error.body)
        raise


if __name__ == "__main__":
    run_demo()
