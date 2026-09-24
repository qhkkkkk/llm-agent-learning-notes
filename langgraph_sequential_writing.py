"""使用 LangGraph 构建“大纲 → 初稿 → 润色 → 终稿”的顺序写作工作流。"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

model = init_chat_model(
    model=MODEL_NAME,
    model_provider="openai",
    base_url=OPENAI_BASE_URL,
    temperature=0.7,
)


# 1. 定义外部输入、外部输出和工作流内部状态
class InputState(TypedDict):
    topic: str


class OutputState(TypedDict):
    final_content: str


class State(InputState, OutputState):
    outline: str
    draft: str
    polished_draft: str


# 2. 节点一：生成大纲
OUTLINE_PROMPT = """根据主题生成文章大纲。

主题：{topic}

要求：
1. 只保留两个最核心的标题。
2. 不要解释，只返回最终大纲。"""


def generate_outline(state: InputState):
    """根据主题生成内容大纲。"""
    print("*" * 50)
    print("内容大纲生成中...\n")

    prompt = OUTLINE_PROMPT.format(topic=state["topic"])
    result = model.invoke([HumanMessage(content=prompt)])

    print(f"大纲已生成：\n{result.content}\n")
    return {"outline": result.content}


# 3. 节点二：生成初稿
DRAFT_PROMPT = """根据下面的主题和大纲生成文章初稿。

主题：{topic}
大纲：
{outline}

要求：
1. 每个标题下最多写三句话。
2. 不要解释，只返回文章初稿。"""


def write_draft(state: State):
    """根据内容大纲生成初稿。"""
    print("*" * 50)
    print("初稿生成中...\n")

    prompt = DRAFT_PROMPT.format(
        topic=state["topic"],
        outline=state["outline"],
    )
    result = model.invoke([HumanMessage(content=prompt)])

    print(f"初稿已生成：\n{result.content}\n")
    return {"draft": result.content}


# 4. 节点三：润色初稿
POLISH_PROMPT = """请润色下面的文章初稿。

主题：{topic}
初稿：
{draft}

要求：
1. 提升表达的连贯性和可读性。
2. 不要扩写得太长。
3. 不要解释，只返回润色后的文章。"""


def polish_draft(state: State):
    """润色文章初稿。"""
    print("*" * 50)
    print("初稿润色中...\n")

    prompt = POLISH_PROMPT.format(
        topic=state["topic"],
        draft=state["draft"],
    )
    result = model.invoke([HumanMessage(content=prompt)])

    print(f"初稿润色完成：\n{result.content}\n")
    return {"polished_draft": result.content}


# 5. 节点四：生成终稿
FINAL_PROMPT = """根据文章大纲和润色稿生成最终文章。

主题：{topic}
大纲：
{outline}
润色稿：
{polished_draft}

要求：
1. 检查文章是否紧扣主题和大纲。
2. 保持结构完整、表达简洁。
3. 不要解释，只返回最终文章。"""


def finalize_article(state: State):
    """检查大纲与润色稿，生成最终文章。"""
    print("*" * 50)
    print("终稿生成中...\n")

    prompt = FINAL_PROMPT.format(
        topic=state["topic"],
        outline=state["outline"],
        polished_draft=state["polished_draft"],
    )
    result = model.invoke([HumanMessage(content=prompt)])

    print(f"终稿已完成：\n{result.content}\n")
    return {"final_content": result.content}


# 6. 组装并编译顺序工作流
builder = StateGraph(
    State,
    input_schema=InputState,
    output_schema=OutputState,
)
builder.add_sequence(
    [
        generate_outline,
        write_draft,
        polish_draft,
        finalize_article,
    ]
)
builder.add_edge(START, "generate_outline")
builder.add_edge("finalize_article", END)

chain = builder.compile()


if __name__ == "__main__":
    topic = os.getenv("ARTICLE_TOPIC", "人工智能的未来发展")
    result = chain.invoke({"topic": topic})

    print("*" * 50)
    print("工作流最终输出：")
    print(result)
