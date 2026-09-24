"""使用 LangGraph 构建带检索、相关性评分和问题重写的 Agentic RAG。"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.messages import HumanMessage, filter_messages
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.tools.retriever import create_retriever_tool
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_NAMES = [
    "企业介绍.md",
    "C++开发方向.md",
    "Java开发方向.md",
    "测试开发方向.md",
    "脚手架级微服务租房平台Q&A.md",
]
DOCUMENT_PATHS = [BASE_DIR / "Docs" / "markdown" / name for name in DOCUMENT_NAMES]

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def load_knowledge_base():
    """加载并切分本地 Markdown 知识库。"""
    missing_paths = [str(path) for path in DOCUMENT_PATHS if not path.exists()]
    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"以下知识库文件不存在：\n{missing}")

    documents = []
    for path in DOCUMENT_PATHS:
        documents.extend(UnstructuredMarkdownLoader(str(path)).load())

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1000,
        chunk_overlap=50,
    )
    return text_splitter.split_documents(documents)


# 1. 准备模型、向量库和检索工具
model = init_chat_model(
    model=MODEL_NAME,
    model_provider="openai",
    base_url=OPENAI_BASE_URL,
    temperature=0,
)
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OPENAI_BASE_URL,
    check_embedding_ctx_length=False,
)

vectorstore = InMemoryVectorStore.from_documents(
    documents=load_knowledge_base(),
    embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_knowledge_base",
    "搜索并返回有关比特就业课程或租房平台项目的知识库内容。",
)


# 2. 定义节点
def generate_query_or_respond(state: MessagesState):
    """让模型决定直接回答，还是调用知识库检索工具。"""
    response = model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


retriever_node = ToolNode([retriever_tool])

REWRITE_PROMPT = """请分析下面的问题，并把它改写成更适合知识库检索的查询。

原问题：
{question}

只返回改写后的问题。"""


def rewrite_question(state: MessagesState):
    """当检索结果不相关时，改写最近一次用户查询。"""
    latest_question = filter_messages(
        state["messages"], include_types="human"
    )[-1].content
    prompt = REWRITE_PROMPT.format(question=latest_question)
    response = model.invoke([HumanMessage(content=prompt)])
    return {"messages": [HumanMessage(content=response.content)]}


GENERATE_PROMPT = """你是负责回答问题的助手。
请仅根据检索到的上下文回答；如果上下文没有答案，就明确说不知道。
最多使用三句话，回答要简明扼要。

用户问题：{question}

检索上下文：
{context}"""


def generate_answer(state: MessagesState):
    """使用原始问题和通过评分的检索结果生成最终答案。"""
    original_question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(
        question=original_question,
        context=context,
    )
    return {"messages": [model.invoke([HumanMessage(content=prompt)])]}


GRADE_PROMPT = """你是文档相关性评分员。

检索文档：
{context}

用户问题：{question}

如果文档包含回答问题所需的关键词或语义信息，返回 yes；否则返回 no。"""


class GradeDocuments(BaseModel):
    """检索结果相关性评分。"""

    score: Literal["yes", "no"] = Field(description="相关返回 yes，否则返回 no")


def grade_documents(
    state: MessagesState,
) -> Literal["rewrite_question", "generate_answer"]:
    """判断检索结果是否足以回答最近一次查询。"""
    latest_question = filter_messages(
        state["messages"], include_types="human"
    )[-1].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(
        context=context,
        question=latest_question,
    )
    result = model.with_structured_output(GradeDocuments).invoke(
        [HumanMessage(content=prompt)]
    )
    return "generate_answer" if result.score == "yes" else "rewrite_question"


# 3. 组装并编译状态图
workflow = StateGraph(MessagesState)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        "__end__": END,
    },
)
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
    },
)
workflow.add_edge("rewrite_question", "generate_query_or_respond")
workflow.add_edge("generate_answer", END)

graph = workflow.compile()


# 4. 流式执行，观察每个节点如何更新消息状态
if __name__ == "__main__":
    question = os.getenv("QUESTION", "测试开发方向的主线课程有哪些？")

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 12},
    ):
        for node, update in chunk.items():
            print(f"由节点 {node} 更新消息")
            update["messages"][-1].pretty_print()
            print()
