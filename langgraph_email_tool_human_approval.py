"""LangGraph 邮件工具调用、人工审批与参数修改示例。

本示例只在终端打印邮件信息，不连接真实邮件服务。
"""

import json
import os
from typing import Literal, NotRequired, TypedDict, cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt


load_dotenv()

ReviewAction = Literal["同意", "不同意"]
Route = Literal["tools", "__end__"]

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://openrouter.ai/api/v1",
)
THREAD_ID = os.getenv("EMAIL_APPROVAL_THREAD_ID", "email-approval-demo")
DEMO_REQUEST = os.getenv(
    "EMAIL_DEMO_REQUEST",
    "发送电子邮件至 alice@example.com，主题是：请假，内容是：回老家。",
)


class EmailReviewResponse(TypedDict):
    """人工恢复工具中断时提交的结构化结果。"""

    action: ReviewAction
    to: NotRequired[str]
    subject: NotRequired[str]
    body: NotRequired[str]
    reason: NotRequired[str]


def _validate_header(value: object, field_name: str, max_length: int) -> str:
    """校验收件人和主题，阻止空值与换行符进入邮件头。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串。")

    normalized = value.strip()
    if "\r" in normalized or "\n" in normalized:
        raise ValueError(f"{field_name} 不能包含换行符。")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} 不能超过 {max_length} 个字符。")
    return normalized


def validate_email_fields(to: object, subject: object, body: object) -> tuple[str, str, str]:
    """对模型参数和人工修改后的邮件字段执行最小运行时校验。"""
    final_to = _validate_header(to, "to", 320)
    final_subject = _validate_header(subject, "subject", 200)

    if "@" not in final_to:
        raise ValueError("to 必须包含一个可识别的电子邮箱地址。")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("body 必须是非空字符串。")

    final_body = body.strip()
    if len(final_body) > 10_000:
        raise ValueError("body 不能超过 10000 个字符。")
    return final_to, final_subject, final_body


def validate_review_response(value: object) -> EmailReviewResponse:
    """把来自审批界面的恢复值限制为明确、可预测的协议。"""
    if not isinstance(value, dict):
        raise ValueError("审批结果必须是包含 action 的字典。")

    action = value.get("action")
    if action not in ("同意", "不同意"):
        raise ValueError(
            f"action 必须是“同意”或“不同意”，当前值为 {action!r}。"
        )

    response: EmailReviewResponse = {
        "action": cast(ReviewAction, action),
    }
    for field in ("to", "subject", "body", "reason"):
        field_value = value.get(field)
        if field_value is not None:
            if not isinstance(field_value, str):
                raise ValueError(f"{field} 必须是字符串。")
            if field == "to":
                response["to"] = field_value
            elif field == "subject":
                response["subject"] = field_value
            elif field == "body":
                response["body"] = field_value
            else:
                response["reason"] = field_value
    return response


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """在人工批准后模拟发送电子邮件；不会连接真实邮件服务。"""
    original_to, original_subject, original_body = validate_email_fields(
        to,
        subject,
        body,
    )

    review = validate_review_response(
        interrupt(
            {
                "action": "发送邮件",
                "message": "请检查邮件草稿；可以拒绝，或批准并修改字段。",
                "draft": {
                    "to": original_to,
                    "subject": original_subject,
                    "body": original_body,
                },
                "options": ["同意", "不同意"],
                "response_examples": [
                    {"action": "同意"},
                    {
                        "action": "同意",
                        "subject": "修改后的主题",
                    },
                    {
                        "action": "不同意",
                        "reason": "收件人不正确",
                    },
                ],
                "warning": "教学示例只会模拟发送，不会调用真实邮件服务。",
            }
        )
    )

    if review["action"] == "不同意":
        reason = review.get("reason", "未提供原因").strip() or "未提供原因"
        return json.dumps(
            {
                "status": "cancelled",
                "message": "用户取消发送邮件",
                "reason": reason,
            },
            ensure_ascii=False,
        )

    final_to, final_subject, final_body = validate_email_fields(
        review.get("to", original_to),
        review.get("subject", original_subject),
        review.get("body", original_body),
    )
    email_info = {
        "status": "simulated",
        "to": final_to,
        "subject": final_subject,
        "body": final_body,
    }
    print(f"【模拟发送邮件】{json.dumps(email_info, ensure_ascii=False)}")
    return json.dumps(email_info, ensure_ascii=False)


def create_tool_bound_model() -> Runnable[LanguageModelInput, AIMessage]:
    """创建模型，并把邮件工具 Schema 提供给模型。"""
    model = init_chat_model(
        model=MODEL_NAME,
        model_provider="openai",
        base_url=OPENAI_BASE_URL,
    )
    return model.bind_tools([send_email])


def build_graph(model_with_tools=None):
    """构建“模型 → 工具审批 → 模型总结”的循环图。"""
    bound_model = model_with_tools or create_tool_bound_model()

    def llm_call(state: MessagesState):
        result = bound_model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是邮件助手。用户明确要求发送邮件时，最多调用一次 "
                        "send_email 工具；工具结果返回后，向用户简要说明最终状态。"
                    )
                ),
                *state["messages"],
            ]
        )
        return {"messages": [result]}

    def route_after_model(state: MessagesState) -> Route:
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    builder = StateGraph(MessagesState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tools", ToolNode([send_email]))

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        route_after_model,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "llm_call")

    return builder.compile(checkpointer=InMemorySaver())


def read_review_response() -> EmailReviewResponse:
    """从环境变量或终端读取审批决定与可选字段修改。"""
    configured_action = os.getenv("EMAIL_REVIEW_ACTION", "").strip()
    action = configured_action or input("请选择【同意/不同意】：").strip()

    raw_response: dict[str, str] = {"action": action}
    if action == "不同意":
        reason = os.getenv("EMAIL_REVIEW_REASON", "").strip()
        if reason:
            raw_response["reason"] = reason
        return validate_review_response(raw_response)

    for env_name, field in (
        ("EMAIL_REVIEW_TO", "to"),
        ("EMAIL_REVIEW_SUBJECT", "subject"),
        ("EMAIL_REVIEW_BODY", "body"),
    ):
        configured_value = os.getenv(env_name, "").strip()
        if configured_value:
            raw_response[field] = configured_value

    if not configured_action:
        should_edit = input("是否修改邮件字段？【y/N】：").strip().lower()
        if should_edit == "y":
            for prompt, field in (
                ("新收件人（留空表示不修改）：", "to"),
                ("新主题（留空表示不修改）：", "subject"),
                ("新正文（留空表示不修改）：", "body"),
            ):
                new_value = input(prompt)
                if new_value.strip():
                    raw_response[field] = new_value

    return validate_review_response(raw_response)


def run_demo() -> None:
    """运行到邮件工具中断，再用相同 thread_id 恢复图。"""
    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    pending_result = graph.invoke(
        {"messages": [HumanMessage(content=DEMO_REQUEST)]},
        config=config,
    )
    pending_interrupts = pending_result.get("__interrupt__", ())
    if not pending_interrupts:
        raise RuntimeError("模型没有调用邮件工具，图未在预期位置中断。")

    request = pending_interrupts[0]
    print("\n收到邮件审批请求：")
    print(json.dumps(request.value, ensure_ascii=False, indent=2))
    print(f"中断 ID：{request.id}")

    review = read_review_response()
    print("\n使用同一 thread_id 恢复：")
    print(json.dumps(review, ensure_ascii=False, indent=2))

    final_result = graph.invoke(
        Command(resume=review),
        config=config,
    )
    print("\n助手最终答复：")
    final_result["messages"][-1].pretty_print()


if __name__ == "__main__":
    run_demo()
