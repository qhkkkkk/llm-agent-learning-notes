"""使用 LangGraph 手写一个带 Tavily 搜索工具的最小 Agent。"""

import json
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_tavily import TavilySearch
from langgraph.constants import END, START
from langgraph.graph import StateGraph


# 从项目根目录的 .env 加载 API Key 和模型名称。
load_dotenv()


# 1. 准备工具和模型
search = TavilySearch(max_results=4)
tools = [search]

# 可以在 .env 中通过 MODEL_NAME 修改模型。
# 当前模型必须支持 Tool Calling，并且本地已配置相应提供商的 API Key。
model_name = os.getenv("MODEL_NAME", "deepseek-v4-pro")
model = init_chat_model(model_name, temperature=0)
model_with_tools = model.bind_tools(tools)


# 2. 定义共享状态
class MessagesState(TypedDict):
    # operator.add 让每个节点返回的新消息追加到原消息列表。
    messages: Annotated[list[AnyMessage], operator.add]
    # 没有 Reducer，因此新整数会覆盖旧整数。
    llm_calls: int


# 3. 定义模型节点
def llm_call(state: MessagesState):
    """调用模型，让模型决定直接回答还是申请调用工具。"""
    messages = state["messages"]

    result = model_with_tools.invoke(
        [SystemMessage(content="你是一个乐于助人的助手，支持调用工具进行搜索")]
        + messages
    )

    return {
        "messages": [result],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# 通过工具名找到真正的工具对象。
tools_by_name = {tool.name: tool for tool in tools}


# 4. 定义工具节点
def tool_node(state: MessagesState):
    """执行 AIMessage 中申请的全部工具调用。"""
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        raise TypeError("进入 tool_node 时，最后一条消息必须是 AIMessage")

    result = []

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        # ToolMessage.content 使用字符串最稳妥；Tavily 可能返回字典或列表。
        if isinstance(observation, str):
            content = observation
        else:
            content = json.dumps(observation, ensure_ascii=False, default=str)

        result.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )

    return {"messages": result}


# 5. 定义条件路由
def should_continue(state: MessagesState):
    """有工具调用就进入工具节点，否则结束。"""
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"

    return END


# 6. 构建并编译图
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        END: END,
    },
)
agent_builder.add_edge("tool_node", "llm_call")

agent_search = agent_builder.compile()


# 7. 执行图
if __name__ == "__main__":
    result = agent_search.invoke(
        {
            "messages": [HumanMessage(content="今天离石的天气怎么样？")],
            "llm_calls": 0,
        },
        config={"recursion_limit": 10},
    )

    print(f"一共调用了 {result['llm_calls']} 次 LLM")
    for message in result["messages"]:
        message.pretty_print()

