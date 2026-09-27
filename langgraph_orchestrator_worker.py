"""使用 LangGraph 构建 Orchestrator-Worker 动态并行报告工作流。"""

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

model = init_chat_model(
    model=MODEL_NAME,
    model_provider="openai",
    base_url=OPENAI_BASE_URL,
    temperature=0,
)


# 1. 定义协调者的结构化输出
class Section(BaseModel):
    """协调者规划出的一个报告章节。"""

    name: str = Field(description="章节标题")
    description: str = Field(description="这一章需要覆盖的核心内容")


class Sections(BaseModel):
    """协调者规划出的完整章节列表。"""

    sections: list[Section]


planner = model.with_structured_output(Sections)


# 2. 定义主图状态和工作者接收的局部状态
class CompletedSection(TypedDict):
    """一个工作者完成的章节。"""

    index: int
    name: str
    content: str


class InputState(TypedDict):
    topic: str


class OutputState(TypedDict):
    final_report: str


class State(InputState, OutputState):
    sections: list[Section]
    # 多个工作者会并行写入该字段，operator.add 会把各自的列表合并。
    completed_sections: Annotated[list[CompletedSection], operator.add]


class WorkerState(TypedDict):
    """通过 Send 单独传给某个工作者的数据。"""

    section: Section
    section_index: int


# 3. 协调者：根据主题动态规划章节
def orchestrator(state: InputState):
    """分析报告主题，并生成三到五个章节。"""
    print("协调者正在拆分任务...")

    result = planner.invoke(
        [
            HumanMessage(
                content=(
                    f"请为主题“{state['topic']}”制定报告大纲。"
                    "大纲必须包含三到五个章节。"
                    "每个章节都要提供简洁的标题和明确的内容要求。"
                )
            )
        ]
    )

    print(f"协调者已生成 {len(result.sections)} 个章节。")
    return {"sections": result.sections}


# 4. 条件边：在运行时为每个章节创建一个工作者任务
def assign_workers(state: State) -> list[Send]:
    """把每个章节及其序号发送给一个 worker 节点实例。"""
    return [
        Send(
            "worker",
            {
                "section": section,
                "section_index": index,
            },
        )
        for index, section in enumerate(state["sections"])
    ]


# 5. 工作者：只编写分配给自己的章节
def worker(state: WorkerState):
    """根据局部任务状态编写一个报告章节。"""
    section = state["section"]
    print(f"工作者正在编写：{section.name}")

    result = model.invoke(
        [
            HumanMessage(
                content=(
                    f"请编写报告章节《{section.name}》。\n"
                    f"内容要求：{section.description}\n"
                    "使用 Markdown，内容准确、结构清楚、语言简洁。"
                    "直接返回本章节正文，不要解释写作过程。"
                )
            )
        ]
    )

    return {
        "completed_sections": [
            {
                "index": state["section_index"],
                "name": section.name,
                "content": result.content,
            }
        ]
    }


# 6. 汇总器：等待所有工作者完成后合并报告
def synthesizer(state: State):
    """按原大纲顺序合并所有工作者生成的章节。"""
    print("汇总器正在合并全部章节...")

    ordered_sections = sorted(
        state["completed_sections"],
        key=lambda item: item["index"],
    )
    final_report = "\n\n---\n\n".join(
        f"## {item['name']}\n\n{item['content']}" for item in ordered_sections
    )

    return {"final_report": final_report}


# 7. 组装并编译“规划 → 动态并行 → 汇总”工作流
builder = StateGraph(
    State,
    input_schema=InputState,
    output_schema=OutputState,
)
builder.add_node("orchestrator", orchestrator)
builder.add_node("worker", worker)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["worker"],
)
builder.add_edge("worker", "synthesizer")
builder.add_edge("synthesizer", END)

report_workflow = builder.compile()


if __name__ == "__main__":
    topic = os.getenv("REPORT_TOPIC", "雅思与托福的区别和备考建议")
    result = report_workflow.invoke({"topic": topic})

    print("\n" + "*" * 50)
    print("最终报告：\n")
    print(result["final_report"])
