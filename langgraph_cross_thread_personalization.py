"""LangGraph 结构化记忆提取与跨线程个性化 Agent 示例。"""

import json
import os
from dataclasses import dataclass
from typing import Annotated, NotRequired, TypedDict

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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
USER_ID = os.getenv("LONG_TERM_USER_ID", "user_123")
FIRST_THREAD_ID = os.getenv("PERSONALIZATION_THREAD_ONE", "profile-session")
SECOND_THREAD_ID = os.getenv("PERSONALIZATION_THREAD_TWO", "recommend-session")
FIRST_MESSAGE = os.getenv(
    "PERSONALIZATION_FIRST_MESSAGE",
    "我叫小明，身高1.75米，我喜欢川菜里的回锅肉。",
)
SECOND_MESSAGE = os.getenv(
    "PERSONALIZATION_SECOND_MESSAGE",
    "根据我的口味推荐几道菜；如果需要最新餐厅信息，可以搜索。",
)

MEMORY_KEY = "profile"


class PersonMemory(BaseModel):
    """从用户本人陈述中提取的长期记忆。"""

    name: str | None = Field(default=None, description="用户本人的姓名")
    height_meters: float | None = Field(
        default=None,
        description="用户本人以米为单位的身高",
    )
    favorite_foods: list[str] | None = Field(
        default=None,
        description="用户本人喜欢的食物列表",
    )


@dataclass(frozen=True)
class UserContext:
    """不会写入图状态、但每次运行都可访问的用户上下文。"""

    user_id: str


class MessagesState(TypedDict):
    """图状态：线程消息与累计模型调用次数。"""

    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: NotRequired[int]


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
memory_extractor = model.with_structured_output(PersonMemory)


def memory_namespace(user_id: str) -> tuple[str, ...]:
    """为一名用户生成独立的长期记忆命名空间。"""
    return (user_id, "person_memory")


def require_store(runtime: Runtime[UserContext]) -> BaseStore:
    """取得编译图时注入的 Store，并在缺失时给出明确错误。"""
    if runtime.store is None:
        raise RuntimeError("图没有配置 Store，无法读写长期记忆。")
    return runtime.store


def latest_human_message(state: MessagesState) -> HumanMessage:
    """只取本轮最近的用户消息，避免从 AI 或工具输出中提取个人信息。"""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message
    raise ValueError("状态中没有 HumanMessage，无法提取用户记忆。")


def merge_person_memory(
    store: BaseStore,
    user_id: str,
    extracted: PersonMemory,
) -> dict[str, object]:
    """把新信息与现有记忆合并，空值不会覆盖旧值。"""
    namespace = memory_namespace(user_id)
    current_item = store.get(namespace, MEMORY_KEY)
    current = dict(current_item.value) if current_item else {}

    existing_foods = current.get("favorite_foods", [])
    if not isinstance(existing_foods, list):
        existing_foods = []

    new_foods = extracted.favorite_foods or []
    merged_foods = list(dict.fromkeys([*existing_foods, *new_foods]))

    merged: dict[str, object] = {}
    name = (
        extracted.name
        if extracted.name is not None
        else current.get("name")
    )
    height = (
        extracted.height_meters
        if extracted.height_meters is not None
        else current.get("height_meters")
    )

    if name is not None:
        merged["name"] = name
    if height is not None:
        merged["height_meters"] = height
    if merged_foods:
        merged["favorite_foods"] = merged_foods

    # 如果本轮和历史都没有可保存的信息，就不创建空记忆。
    if merged:
        store.put(namespace, MEMORY_KEY, merged)

    return merged


def extract_person_memory(
    state: MessagesState,
    runtime: Runtime[UserContext],
):
    """从最新用户消息提取资料，并更新该用户的长期记忆。"""
    user_message = latest_human_message(state)
    extracted = memory_extractor.invoke(
        [
            SystemMessage(
                content=(
                    "你是个人信息提取器。只提取用户明确描述的、关于用户本人"
                    "的信息；不要提取他人的信息，不要推测。未知字段返回 null。"
                )
            ),
            user_message,
        ]
    )

    store = require_store(runtime)
    merged = merge_person_memory(store, runtime.context.user_id, extracted)
    print(f"长期记忆更新结果：{merged or '本轮没有新增信息'}")

    return {"llm_calls": state.get("llm_calls", 0) + 1}


def load_person_memory(store: BaseStore, user_id: str) -> dict[str, object]:
    """精确读取一名用户的个人记忆。"""
    item = store.get(memory_namespace(user_id), MEMORY_KEY)
    return dict(item.value) if item else {}


def llm_call(
    state: MessagesState,
    runtime: Runtime[UserContext],
):
    """读取长期记忆，让模型据此回答或决定是否调用搜索工具。"""
    store = require_store(runtime)
    person_memory = load_person_memory(store, runtime.context.user_id)
    memory_text = (
        json.dumps(person_memory, ensure_ascii=False)
        if person_memory
        else "暂无已保存的用户信息"
    )

    result = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "你是一个乐于助人的个性化助手。仅在与当前问题有关时参考"
                    f"以下长期记忆：{memory_text}。需要最新信息时可以调用搜索工具。"
                )
            )
        ]
        + state["messages"]
    )

    return {
        "messages": [result],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState):
    """执行最新 AIMessage 中提出的全部工具调用。"""
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


def should_continue(state: MessagesState):
    """有工具调用时进入工具节点，否则结束本轮。"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END


def build_agent(store: BaseStore):
    """编译带线程 Checkpointer 和跨线程 Store 的个性化 Agent。"""
    builder = StateGraph(MessagesState, context_schema=UserContext)
    builder.add_node("extract_person_memory", extract_person_memory)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "extract_person_memory")
    builder.add_edge("extract_person_memory", "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END],
    )
    builder.add_edge("tool_node", "llm_call")

    return builder.compile(
        checkpointer=InMemorySaver(),
        store=store,
    )


def run_demo() -> None:
    """用两个不同线程演示同一用户的长期记忆可以跨线程复用。"""
    store = InMemoryStore()
    agent = build_agent(store)
    context = UserContext(user_id=USER_ID)

    first_config = {"configurable": {"thread_id": FIRST_THREAD_ID}}
    first_result = agent.invoke(
        {"messages": [HumanMessage(content=FIRST_MESSAGE)]},
        config=first_config,
        context=context,
    )
    print("\n第一次对话回答：")
    first_result["messages"][-1].pretty_print()
    print(f"保存后的长期记忆：{load_person_memory(store, USER_ID)}")

    # thread_id 不同，所以第二个线程没有第一段聊天记录；
    # user_id 相同，所以仍能读取同一命名空间中的长期记忆。
    second_config = {"configurable": {"thread_id": SECOND_THREAD_ID}}
    second_result = agent.invoke(
        {"messages": [HumanMessage(content=SECOND_MESSAGE)]},
        config=second_config,
        context=context,
    )
    print("\n第二次对话回答：")
    second_result["messages"][-1].pretty_print()
    print(f"第二个线程读取到的长期记忆：{load_person_memory(store, USER_ID)}")


if __name__ == "__main__":
    run_demo()
