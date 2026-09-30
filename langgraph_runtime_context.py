"""LangGraph Runtime Context 与可变 State 的边界示例。"""

import os
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


load_dotenv()

Language = Literal["en", "zh"]


@dataclass(frozen=True)
class ContextSchema:
    """一次图运行期间可读、但不应由节点修改的上下文。"""

    user_id: str
    language: Language = "en"


class State(TypedDict, total=False):
    """节点可以读取并返回更新的工作流状态。"""

    messages: list[str]
    user_name: str


def greeting_node(
    state: State,
    runtime: Runtime[ContextSchema],
) -> State:
    """根据运行时语言和状态中的用户名生成问候语。"""
    greetings = {
        "en": "Hello",
        "zh": "你好",
    }
    greeting = greetings[runtime.context.language]

    # user_name 属于可变业务数据，因此从 State 读取。
    user_name = state.get("user_name", "Guest").strip() or "Guest"

    # messages 没有配置 Reducer，这次返回会覆盖旧列表。
    return {
        "messages": [f"{greeting}，{user_name}！"],
    }


def build_graph():
    """构建并编译最小 Runtime Context 工作流。"""
    builder = StateGraph(State, context_schema=ContextSchema)
    builder.add_node("greeting", greeting_node)
    builder.add_edge(START, "greeting")
    builder.add_edge("greeting", END)
    return builder.compile()


def read_language(raw_value: str) -> Language:
    """把环境变量转换为受支持的语言值。"""
    value = raw_value.strip().lower()
    if value not in {"en", "zh"}:
        raise ValueError(
            "RUNTIME_LANGUAGE 只支持 'en' 或 'zh'，"
            f"当前值为 {raw_value!r}。"
        )
    return cast(Language, value)


def read_context() -> ContextSchema:
    """从环境变量创建类型明确的运行时上下文实例。"""
    user_id = os.getenv("RUNTIME_USER_ID", "user_123").strip()
    if not user_id:
        raise ValueError("RUNTIME_USER_ID 不能为空。")

    language = read_language(os.getenv("RUNTIME_LANGUAGE", "zh"))
    return ContextSchema(user_id=user_id, language=language)


def read_initial_state() -> State:
    """从环境变量创建本次运行的初始状态。"""
    user_name = os.getenv("RUNTIME_USER_NAME", "小明").strip()
    if not user_name:
        return {}
    return {"user_name": user_name}


def run_demo() -> None:
    """运行示例并打印上下文与最终状态。"""
    graph = build_graph()
    context = read_context()
    initial_state = read_initial_state()

    result = graph.invoke(initial_state, context=context)

    print(f"本次调用的 user_id：{context.user_id}")
    print(f"本次调用的 language：{context.language}")
    print(f"最终 State：{result}")


if __name__ == "__main__":
    run_demo()
