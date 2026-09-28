"""LangGraph PostgreSQL 持久化、状态历史、重放与分叉示例。"""

import os
from typing import Annotated, NotRequired, TypedDict
from uuid import uuid4

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
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
THREAD_ID = os.getenv("THREAD_ID", "langgraph-postgres-demo")
QUESTION = os.getenv("PERSISTENCE_QUESTION", "今天西安的天气如何？")
FORK_QUESTION = os.getenv(
    "FORK_QUESTION",
    "我们之前聊过宠物相关的话题吗？",
)
TIME_TRAVEL_MODE = os.getenv("TIME_TRAVEL_MODE", "history").lower()


search = TavilySearch(max_results=4)
tools = [search]
tools_by_name = {tool.name: tool for tool in tools}

model = init_chat_model(
    model=MODEL_NAME,
    model_provider="openai",
    base_url=OPENAI_BASE_URL,
    temperature=0,
)
model_with_tools = model.bind_tools(tools)


# 1. 状态定义：消息使用 add_messages 合并，计数器使用覆盖更新
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: NotRequired[int]


# 2. 模型节点：根据当前消息决定直接回答还是调用搜索工具
def llm_call(state: MessagesState):
    """让模型读取完整对话，并决定是否发起工具调用。"""
    result = model_with_tools.invoke(
        [
            SystemMessage(
                content="你是一个乐于助人的助手，需要最新信息时可以调用搜索工具。"
            )
        ]
        + state["messages"]
    )

    return {
        "messages": [result],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# 3. 工具节点：执行模型提出的全部工具调用
def tool_node(state: MessagesState):
    """执行最新 AIMessage 中的工具调用，并返回 ToolMessage。"""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError("tool_node 只能处理带有工具调用的 AIMessage。")

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(
                content=str(observation),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_messages}


# 4. 条件路由：有工具调用就执行工具，否则结束本轮
def should_continue(state: MessagesState):
    """根据最新 AIMessage 是否包含 tool_calls 选择下一步。"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END


# 5. 构建图；checkpointer 在 main() 中连接 PostgreSQL 后传入
def build_agent(checkpointer: PostgresSaver):
    """编译带 PostgreSQL 检查点存储的搜索 Agent。"""
    builder = StateGraph(MessagesState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END],
    )
    builder.add_edge("tool_node", "llm_call")

    return builder.compile(checkpointer=checkpointer)


def print_history(history):
    """按“最新在前”的顺序打印关键检查点信息。"""
    print("\n检查点历史（最新在前）：")
    for snapshot in history:
        configurable = snapshot.config["configurable"]
        checkpoint_id = configurable["checkpoint_id"]
        step = snapshot.metadata.get("step")
        source = snapshot.metadata.get("source")
        next_nodes = snapshot.next or ("END",)
        print(
            f"step={step!s:>3} | source={source:<6} | "
            f"next={next_nodes} | checkpoint_id={checkpoint_id}"
        )


def find_input_checkpoint(history, message_id: str):
    """找到本次用户输入已保存、但 llm_call 尚未执行的检查点。"""
    for snapshot in history:
        messages = snapshot.values.get("messages", [])
        if (
            snapshot.next == ("llm_call",)
            and messages
            and isinstance(messages[-1], HumanMessage)
            and messages[-1].id == message_id
        ):
            return snapshot

    raise RuntimeError("未找到本次输入对应的检查点，无法演示时间旅行。")


def main():
    """运行 Agent，并按配置查看历史、重放或创建状态分叉。"""
    postgres_uri = os.getenv("POSTGRES_URI")
    if not postgres_uri:
        raise RuntimeError(
            "缺少 POSTGRES_URI。请复制 .env.example 为 .env，"
            "并填写自己的 PostgreSQL 连接信息。"
        )

    valid_modes = {"history", "replay", "fork", "all"}
    if TIME_TRAVEL_MODE not in valid_modes:
        raise ValueError(
            f"TIME_TRAVEL_MODE 必须是 {sorted(valid_modes)} 之一，"
            f"当前值为 {TIME_TRAVEL_MODE!r}。"
        )

    config = {"configurable": {"thread_id": THREAD_ID}}

    with PostgresSaver.from_conn_string(postgres_uri) as checkpointer:
        # 首次运行必须 setup；重复调用会检查并应用尚未执行的迁移。
        checkpointer.setup()
        agent = build_agent(checkpointer)

        input_message = HumanMessage(content=QUESTION, id=str(uuid4()))
        result = agent.invoke(
            {"messages": [input_message]},
            config=config,
        )

        print("\n本轮最终回答：")
        result["messages"][-1].pretty_print()
        print(f"累计模型调用次数：{result.get('llm_calls', 0)}")

        latest_snapshot = agent.get_state(config)
        print(f"\n最新检查点下一节点：{latest_snapshot.next or ('END',)}")

        history = list(agent.get_state_history(config))
        print_history(history)
        input_checkpoint = find_input_checkpoint(history, input_message.id)

        if TIME_TRAVEL_MODE in {"replay", "all"}:
            print("\n开始 Replay：从模型调用前的检查点重新执行...")
            replay_result = agent.invoke(None, config=input_checkpoint.config)
            replay_result["messages"][-1].pretty_print()

        if TIME_TRAVEL_MODE in {"fork", "all"}:
            print("\n开始 Fork：替换历史问题并从该检查点继续执行...")
            replacement_message = HumanMessage(
                content=FORK_QUESTION,
                id=input_message.id,
            )
            fork_config = agent.update_state(
                input_checkpoint.config,
                {"messages": [replacement_message]},
            )
            fork_result = agent.invoke(None, config=fork_config)
            fork_result["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
