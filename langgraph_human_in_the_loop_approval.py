"""LangGraph interrupt、Command 与人工审批恢复示例。"""

import json
import os
from typing import Literal, NotRequired, TypedDict, cast

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt


load_dotenv()

ApprovalDecision = Literal["批准", "拒绝"]
ApprovalStatus = Literal["等待", "批准", "拒绝"]
ApprovalDestination = Literal["proceed_node", "cancel_node"]

THREAD_ID = os.getenv("APPROVAL_THREAD_ID", "approval-demo")
ACTION_DETAILS = os.getenv(
    "APPROVAL_ACTION_DETAILS",
    "模拟转账 30000 元（教学示例，不会连接真实支付系统）",
)


class ApprovalState(TypedDict):
    """审批流程共享状态。"""

    action_details: str
    status: ApprovalStatus
    outcome: NotRequired[str]


def validate_decision(value: object) -> ApprovalDecision:
    """把恢复值限制为两个明确选项，避免未知文本默认进入拒绝分支。"""
    if not isinstance(value, str):
        raise ValueError("审批结果必须是字符串：批准或拒绝。")

    normalized = value.strip()
    if normalized not in ("批准", "拒绝"):
        raise ValueError(
            f"审批结果必须是“批准”或“拒绝”，当前值为 {value!r}。"
        )
    return cast(ApprovalDecision, normalized)


def approval_node(
    state: ApprovalState,
) -> Command[ApprovalDestination]:
    """暂停图，等待人工决定；恢复后更新状态并跳转到对应节点。"""
    decision = validate_decision(
        interrupt(
            {
                "question": "是否批准此操作？",
                "details": state["action_details"],
                "options": ["批准", "拒绝"],
                "warning": "这是教学审批请求；真实高风险操作仍需独立鉴权与审计。",
            }
        )
    )

    next_node: ApprovalDestination = (
        "proceed_node" if decision == "批准" else "cancel_node"
    )
    return Command(
        update={"status": decision},
        goto=next_node,
    )


def proceed_node(state: ApprovalState):
    """演示批准后的业务分支，不执行任何真实外部操作。"""
    outcome = f"已批准：{state['action_details']}（仅模拟，未真实执行）"
    print(outcome)
    return {"status": "批准", "outcome": outcome}


def cancel_node(state: ApprovalState):
    """演示拒绝后的业务分支。"""
    outcome = f"已拒绝：{state['action_details']}"
    print(outcome)
    return {"status": "拒绝", "outcome": outcome}


def build_graph():
    """构建带内存 Checkpointer 的可中断审批图。"""
    builder = StateGraph(ApprovalState)
    builder.add_node("approval_node", approval_node)
    builder.add_node("proceed_node", proceed_node)
    builder.add_node("cancel_node", cancel_node)

    builder.add_edge(START, "approval_node")
    builder.add_edge("proceed_node", END)
    builder.add_edge("cancel_node", END)

    return builder.compile(checkpointer=InMemorySaver())


def read_demo_decision() -> ApprovalDecision:
    """优先读取环境变量；未配置时由终端用户现场输入审批结果。"""
    configured = os.getenv("APPROVAL_DECISION", "").strip()
    if configured:
        return validate_decision(configured)
    return validate_decision(input("请输入审批结果【批准/拒绝】："))


def run_demo() -> None:
    """先运行到中断点，再使用相同线程恢复执行。"""
    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    pending_result = graph.invoke(
        {
            "action_details": ACTION_DETAILS,
            "status": "等待",
        },
        config=config,
    )

    pending_interrupts = pending_result.get("__interrupt__", ())
    if not pending_interrupts:
        raise RuntimeError("图没有在预期位置中断。")

    request = pending_interrupts[0]
    print("\n收到审批请求：")
    print(json.dumps(request.value, ensure_ascii=False, indent=2))
    print(f"中断 ID：{request.id}")

    decision = read_demo_decision()
    print(f"\n使用同一 thread_id 恢复，审批结果：{decision}")
    final_result = graph.invoke(
        Command(resume=decision),
        config=config,
    )

    print("\n最终状态：")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_demo()
