"""LangGraph interrupt 与人工审阅、编辑内容示例。"""

import json
import os
from typing import Literal, NotRequired, TypedDict, cast

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt


load_dotenv()

ReviewAction = Literal["通过", "编辑"]
ReviewStatus = Literal["等待", "通过", "已编辑"]

THREAD_ID = os.getenv("REVIEW_THREAD_ID", "human-review-demo")
INITIAL_TEXT = os.getenv("REVIEW_INITIAL_TEXT", "初始文章……")


class ReviewState(TypedDict):
    """待审阅文本以及审阅状态。"""

    text: str
    review_status: NotRequired[ReviewStatus]


class ReviewResponse(TypedDict):
    """调用方恢复中断时提交的结构化审阅结果。"""

    action: ReviewAction
    content: NotRequired[str]


def validate_review_response(value: object) -> ReviewResponse:
    """校验来自人工审阅界面或终端的恢复值。"""
    if not isinstance(value, dict):
        raise ValueError("审阅结果必须是包含 action 的字典。")

    action = value.get("action")
    if action not in ("通过", "编辑"):
        raise ValueError(
            f"action 必须是“通过”或“编辑”，当前值为 {action!r}。"
        )

    if action == "通过":
        return {"action": "通过"}

    content = value.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("选择“编辑”时，content 必须是非空字符串。")

    return {
        "action": cast(ReviewAction, action),
        "content": content.strip(),
    }


def review_node(state: ReviewState):
    """暂停图，让审阅者通过原文或提交编辑后的内容。"""
    response = validate_review_response(
        interrupt(
            {
                "instruction": "请查看内容，选择直接通过或提交编辑版本。",
                "content": state["text"],
                "options": ["通过", "编辑"],
                "response_examples": [
                    {"action": "通过"},
                    {"action": "编辑", "content": "编辑后的内容"},
                ],
            }
        )
    )

    if response["action"] == "通过":
        return {"review_status": "通过"}

    edited_text = response["content"]
    return {
        "text": edited_text,
        "review_status": "已编辑",
    }


def build_graph():
    """构建带内存 Checkpointer 的人工审阅图。"""
    builder = StateGraph(ReviewState)
    builder.add_node("review_node", review_node)
    builder.add_edge(START, "review_node")
    builder.add_edge("review_node", END)
    return builder.compile(checkpointer=InMemorySaver())


def read_demo_response() -> ReviewResponse:
    """读取环境变量或终端输入，构造结构化恢复值。"""
    configured_action = os.getenv("REVIEW_ACTION", "").strip()
    action = configured_action or input("请选择审阅动作【通过/编辑】：").strip()

    if action == "通过":
        return validate_review_response({"action": action})

    if action == "编辑":
        configured_text = os.getenv("REVIEW_EDITED_TEXT", "").strip()
        edited_text = configured_text or input("请输入编辑后的内容：")
        return validate_review_response(
            {"action": action, "content": edited_text}
        )

    return validate_review_response({"action": action})


def run_demo() -> None:
    """先运行到审阅中断，再使用相同线程提交人工结果。"""
    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    pending_result = graph.invoke(
        {
            "text": INITIAL_TEXT,
            "review_status": "等待",
        },
        config=config,
    )

    pending_interrupts = pending_result.get("__interrupt__", ())
    if not pending_interrupts:
        raise RuntimeError("图没有在预期位置中断。")

    request = pending_interrupts[0]
    print("\n收到人工审阅请求：")
    print(json.dumps(request.value, ensure_ascii=False, indent=2))
    print(f"中断 ID：{request.id}")

    response = read_demo_response()
    print("\n使用同一 thread_id 恢复：")
    print(json.dumps(response, ensure_ascii=False, indent=2))

    final_result = graph.invoke(
        Command(resume=response),
        config=config,
    )

    print("\n最终状态：")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_demo()
