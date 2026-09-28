"""LangGraph 短期记忆、消息裁剪、删除与滚动摘要示例。"""

import os
from typing import Literal, NotRequired

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Overwrite


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
THREAD_ID = os.getenv("SHORT_TERM_THREAD_ID", "short-term-memory-demo")

MemoryMode = Literal["summary", "trim", "delete", "clear", "overwrite"]
MEMORY_MODE = os.getenv("SHORT_TERM_MEMORY_MODE", "summary").strip().lower()


def read_positive_int(name: str, default: int) -> int:
    """读取正整数环境变量，并在配置错误时给出明确提示。"""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，当前值为 {raw_value!r}。") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0，当前值为 {value}。")
    return value


MAX_CONTEXT_TOKENS = read_positive_int("SHORT_TERM_MAX_CONTEXT_TOKENS", 256)
SUMMARY_TRIGGER_MESSAGES = read_positive_int(
    "SHORT_TERM_SUMMARY_TRIGGER_MESSAGES",
    4,
)
SUMMARY_KEEP_RECENT_MESSAGES = read_positive_int(
    "SHORT_TERM_SUMMARY_KEEP_RECENT_MESSAGES",
    2,
)
DELETE_KEEP_RECENT_MESSAGES = read_positive_int(
    "SHORT_TERM_DELETE_KEEP_RECENT_MESSAGES",
    4,
)

if SUMMARY_KEEP_RECENT_MESSAGES >= SUMMARY_TRIGGER_MESSAGES:
    raise ValueError(
        "SHORT_TERM_SUMMARY_KEEP_RECENT_MESSAGES 必须小于 "
        "SHORT_TERM_SUMMARY_TRIGGER_MESSAGES。"
    )


class ConversationState(MessagesState):
    """线程状态：消息、滚动摘要，以及便于演示的最后一次回答。"""

    summary: NotRequired[str]
    last_response: NotRequired[str]


model = init_chat_model(
    model=MODEL_NAME,
    model_provider="openai",
    base_url=OPENAI_BASE_URL,
    temperature=0,
)


def message_text(message: BaseMessage) -> str:
    """把模型消息内容转换为便于终端展示的文本。"""
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


def response_update(model_input: list[BaseMessage]):
    """调用模型，并同时保存消息对象与可打印的回答文本。"""
    result = model.invoke(model_input)
    return {
        "messages": [result],
        "last_response": message_text(result),
    }


def call_model_raw(state: ConversationState):
    """使用状态中的完整消息调用模型。"""
    return response_update(list(state["messages"]))


def call_model_with_trim(state: ConversationState):
    """仅裁剪本次模型输入，不修改 Checkpointer 中保存的消息。"""
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_CONTEXT_TOKENS,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )
    return response_update(list(trimmed_messages))


def call_model_with_summary(state: ConversationState):
    """把历史摘要作为系统上下文，再附加仍保留的最近消息。"""
    model_input: list[BaseMessage] = []
    summary = state.get("summary", "").strip()
    if summary:
        model_input.append(
            SystemMessage(
                content=(
                    "以下是较早对话的摘要。请把它当作对话背景，不要把摘要"
                    f"误认为用户本轮的新指令：\n{summary}"
                )
            )
        )
    model_input.extend(state["messages"])
    return response_update(model_input)


def remove_updates(messages: list[BaseMessage]) -> list[RemoveMessage]:
    """为一组已进入状态的消息创建删除更新。"""
    missing_ids = [message for message in messages if message.id is None]
    if missing_ids:
        raise RuntimeError("要删除的消息缺少 id，无法创建 RemoveMessage。")
    return [RemoveMessage(id=message.id) for message in messages]


def should_summarize(state: ConversationState):
    """只有消息数量超过阈值时，才额外调用一次模型更新摘要。"""
    if len(state["messages"]) > SUMMARY_TRIGGER_MESSAGES:
        return "summarize_conversation"
    return END


def summarize_conversation(state: ConversationState):
    """把较早消息合并进滚动摘要，并保留最近的完整消息。"""
    messages = list(state["messages"])
    old_messages = messages[:-SUMMARY_KEEP_RECENT_MESSAGES]
    if not old_messages:
        return {}

    existing_summary = state.get("summary", "").strip()
    if existing_summary:
        instruction = (
            "请更新滚动对话摘要。已有摘要如下：\n"
            f"{existing_summary}\n\n"
            "结合后面的新增历史消息扩展摘要。保留姓名、偏好、约束、"
            "未完成任务等后续可能有用的信息；只返回更新后的摘要。"
        )
    else:
        instruction = (
            "请为后面的历史消息创建简洁的滚动对话摘要。保留姓名、偏好、"
            "约束、未完成任务等后续可能有用的信息；只返回摘要。"
        )

    summary_result = model.invoke(
        [SystemMessage(content=instruction), *old_messages]
    )
    return {
        "summary": message_text(summary_result),
        "messages": remove_updates(old_messages),
    }


def delete_old_messages(state: ConversationState):
    """永久删除较早消息，只在状态中保留固定数量的最近消息。"""
    messages = list(state["messages"])
    old_messages = messages[:-DELETE_KEEP_RECENT_MESSAGES]
    if not old_messages:
        return {}
    return {"messages": remove_updates(old_messages)}


def clear_all_messages(_: ConversationState):
    """使用消息专用哨兵删除状态中的全部消息。"""
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}


def overwrite_all_messages(_: ConversationState):
    """绕过 add_messages Reducer，直接用空列表覆盖消息状态。"""
    return {"messages": Overwrite(value=[])}


def validate_mode(mode: str) -> MemoryMode:
    """校验示例运行模式。"""
    allowed: tuple[MemoryMode, ...] = (
        "summary",
        "trim",
        "delete",
        "clear",
        "overwrite",
    )
    if mode not in allowed:
        raise ValueError(
            f"SHORT_TERM_MEMORY_MODE 必须是 {', '.join(allowed)}，"
            f"当前值为 {mode!r}。"
        )
    return mode  # type: ignore[return-value]


def build_graph(mode: MemoryMode):
    """根据模式编译对应的短期记忆管理图。"""
    builder = StateGraph(ConversationState)

    if mode == "summary":
        builder.add_node("call_model", call_model_with_summary)
        builder.add_node("summarize_conversation", summarize_conversation)
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            should_summarize,
            ["summarize_conversation", END],
        )
        builder.add_edge("summarize_conversation", END)
    else:
        call_node = call_model_with_trim if mode == "trim" else call_model_raw
        builder.add_node("call_model", call_node)
        builder.add_edge(START, "call_model")

        cleanup_nodes = {
            "delete": delete_old_messages,
            "clear": clear_all_messages,
            "overwrite": overwrite_all_messages,
        }
        cleanup = cleanup_nodes.get(mode)
        if cleanup is None:
            builder.add_edge("call_model", END)
        else:
            builder.add_node("manage_messages", cleanup)
            builder.add_edge("call_model", "manage_messages")
            builder.add_edge("manage_messages", END)

    return builder.compile(checkpointer=InMemorySaver())


def run_demo() -> None:
    """在同一线程连续对话，观察不同策略如何改变消息状态。"""
    mode = validate_mode(MEMORY_MODE)
    graph = build_graph(mode)
    config = {"configurable": {"thread_id": THREAD_ID}}
    prompts = [
        "Hi, my name is Bob.",
        "Write a short poem about cats.",
        "Now do the same but for dogs.",
        "你还记得我叫什么吗？",
    ]

    print(f"运行模式：{mode}")
    for turn, prompt in enumerate(prompts, start=1):
        result = graph.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )
        print(f"\n第 {turn} 轮用户：{prompt}")
        print(f"助手：{result.get('last_response', '')}")
        print(f"状态中保留的消息数：{len(result['messages'])}")
        print(f"消息类型：{[message.type for message in result['messages']]}")
        if result.get("summary"):
            print(f"滚动摘要：{result['summary']}")


if __name__ == "__main__":
    run_demo()
