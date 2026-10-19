"""六种自定义中间件：生命周期日志、模型重试与工具调用包装。

天气是固定演示值。模型包装器最多尝试三次；工具包装器只记录和重新抛错。
导入本模块不会读取密钥或调用外部服务，测试可注入离线模型和工具。
"""

import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    after_agent,
    after_model,
    before_agent,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的演示天气，固定返回阳光明媚。"""
    return f"在{city}总是阳光明媚！"


@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("即将执行Agent")
    return None  # 不更新状态；不是停止 Agent。


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("Agent执行完成")
    return None


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("即将执行LLM")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("LLM执行完成")
    return None


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # 教学示例沿用宽泛捕获；接真实服务时应只重试确认可恢复的异常。
    for attempt in range(3):
        latest = request.messages[-1].content if request.messages else "（无消息）"
        print(f"【wrap model】最新消息：{latest}")
        try:
            result = handler(request)
            print("【wrap model】模型调用完成")
            return result  # 成功立即结束，不会固定调用三次。
        except Exception as error:
            if attempt == 2:
                raise  # 保留原始异常；after_model/after_agent 不是 finally。
            print(
                f"【wrap model】第{attempt + 1}次尝试失败，"
                f"准备第{attempt + 2}/3次尝试，错误信息为：{error}"
            )
    raise AssertionError("模型重试循环未返回或抛错")


@wrap_tool_call
def retry_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    # 保留练习中的函数名；这里只有一次 handler 调用，并没有工具重试。
    print(f"【wrap tool】执行工具：{request.tool_call['name']}")
    print(f"【wrap tool】参数：{request.tool_call['args']}")
    try:
        result = handler(request)
        print("【wrap tool】工具调用完成")
        return result  # 透传 ToolMessage 或 Command，包括已封装的错误消息。
    except Exception as error:
        print(f"【wrap tool】工具调用出现错误：{error}")
        raise  # 不能只打印错误后隐式返回 None。


def build_agent(
    model: BaseChatModel | None = None,
    *,
    tools: list[BaseTool] | None = None,
):
    """组合真实中间件；传入离线模型时不读取 OpenRouter 密钥。"""
    if model is None:
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("请在环境变量或 .env 中配置 OPENROUTER_API_KEY。")
        model = init_chat_model(
            model=os.getenv("CUSTOM_MIDDLEWARE_MODEL", "openai/gpt-5-mini"),
            model_provider="openrouter",
        )
    return create_agent(
        model=model,
        tools=[get_weather_for_location] if tools is None else tools,
        system_prompt="你是一位乐于助人的助手。",
        middleware=[
            log_before_agent,
            log_after_agent,
            log_before_model,
            log_after_model,
            retry_model,
            retry_tool,
        ],
    )


def main() -> None:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    agent = build_agent()
    query = os.getenv("CUSTOM_MIDDLEWARE_QUERY", "北京的天气如何")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    print(response["messages"])


if __name__ == "__main__":
    main()
