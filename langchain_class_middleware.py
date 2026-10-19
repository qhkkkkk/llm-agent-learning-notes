"""类式中间件：观察 [Logging2Middleware(), LoggingMiddleware()] 的嵌套顺序。

两个模型包装器各最多尝试三次，叠加后每个模型节点最多尝试九次。
这是同步教学示例，沿用宽泛异常捕获；天气工具只返回固定演示值。
导入模块不会读取密钥或调用模型。
"""

import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的演示天气，固定返回阳光明媚。"""
    return f"在{city}总是阳光明媚！"


class LoggingMiddleware(AgentMiddleware):
    """打印【1】的中间件；位置由注册列表决定。"""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("【1】即将执行Agent")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("【1】Agent执行完成")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(3):
            latest = request.messages[-1].content if request.messages else "（无消息）"
            print(f"【1】【wrap model】最新消息：{latest}")
            try:
                result = handler(request)
                print("【1】【wrap model】模型调用完成")
                return result
            except Exception as error:
                if attempt == 2:
                    raise
                print(
                    f"【1】【wrap model】第{attempt + 1}次尝试失败，"
                    f"准备第{attempt + 2}/3次尝试，错误信息为：{error}"
                )
        raise AssertionError("【1】模型重试循环未返回或抛错")


class Logging2Middleware(AgentMiddleware):
    """打印【2】的中间件；默认注册在最外层。"""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("【2】即将执行Agent")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("【2】Agent执行完成")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 默认列表中，这个 handler 会先进入 LoggingMiddleware 的包装器。
        for attempt in range(3):
            latest = request.messages[-1].content if request.messages else "（无消息）"
            print(f"【2】【wrap model】最新消息：{latest}")
            try:
                result = handler(request)
                print("【2】【wrap model】模型调用完成")
                return result
            except Exception as error:
                if attempt == 2:
                    raise
                print(
                    f"【2】【wrap model】第{attempt + 1}次尝试失败，"
                    f"准备第{attempt + 2}/3次尝试，错误信息为：{error}"
                )
        raise AssertionError("【2】模型重试循环未返回或抛错")


def build_agent(
    model: BaseChatModel | None = None,
    *,
    middleware: list[AgentMiddleware] | None = None,
    tools: list[BaseTool] | None = None,
):
    """默认保留练习中的 [2, 1] 顺序；测试可注入模型、工具和中间件列表。"""
    if model is None:
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("请在环境变量或 .env 中配置 OPENROUTER_API_KEY。")
        model = init_chat_model(
            model=os.getenv("CLASS_MIDDLEWARE_MODEL", "openai/gpt-5-mini"),
            model_provider="openrouter",
        )
    return create_agent(
        model=model,
        tools=[get_weather_for_location] if tools is None else tools,
        system_prompt="你是一位乐于助人的助手。",
        middleware=(
            [Logging2Middleware(), LoggingMiddleware()]
            if middleware is None else middleware
        ),
    )


def main() -> None:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    agent = build_agent()
    query = os.getenv("CLASS_MIDDLEWARE_QUERY", "北京的天气如何")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    print(response["messages"])


if __name__ == "__main__":
    main()
