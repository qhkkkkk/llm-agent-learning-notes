"""LangGraph 内存检查点、状态历史与 update_state 分叉示例。"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://openrouter.ai/api/v1",
)
THREAD_ID = os.getenv("JOKE_TIME_TRAVEL_THREAD_ID", "joke-time-travel-demo")
FORK_TOPIC = os.getenv("JOKE_FORK_TOPIC", "程序员趣事")

TOPIC_PROMPT = "生成一个搞笑的笑话主题，只返回五个字以内的主题。"
JOKE_PROMPT = "写一个关于{topic}的简短笑话，只返回笑话正文。"


class State(TypedDict, total=False):
    """工作流状态；首次调用允许不提供任何字段。"""

    topic: str
    joke: str


def create_chat_model():
    """根据环境变量创建 OpenAI 兼容聊天模型。"""
    return init_chat_model(
        model=MODEL_NAME,
        model_provider="openai",
        base_url=OPENAI_BASE_URL,
        temperature=0.7,
    )


def require_text(message: object, source: str) -> str:
    """确保模型返回可写入状态的非空文本。"""
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{source}没有返回非空文本。")
    return content.strip()


def build_graph(chat_model=None, checkpointer=None):
    """构建“生成主题 → 生成笑话”的顺序工作流。"""
    active_model = chat_model or create_chat_model()
    active_checkpointer = checkpointer or InMemorySaver()

    def generate_topic(_: State):
        """生成笑话主题。"""
        result = active_model.invoke(TOPIC_PROMPT)
        return {"topic": require_text(result, "主题模型")}

    def generate_joke(state: State):
        """根据状态中的主题生成笑话。"""
        topic = state.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("生成笑话前，state['topic'] 必须是非空字符串。")

        result = active_model.invoke(JOKE_PROMPT.format(topic=topic.strip()))
        return {"joke": require_text(result, "笑话模型")}

    builder = StateGraph(State)
    builder.add_sequence([generate_topic, generate_joke])
    builder.add_edge(START, "generate_topic")
    builder.add_edge("generate_joke", END)
    return builder.compile(checkpointer=active_checkpointer)


def find_snapshot_before_joke(history):
    """按待执行节点查找“主题已生成、笑话尚未生成”的快照。"""
    for snapshot in history:
        topic = snapshot.values.get("topic")
        if snapshot.next == ("generate_joke",) and isinstance(topic, str):
            return snapshot

    raise RuntimeError(
        "未找到 next=('generate_joke',) 的状态快照，无法创建分叉。"
    )


def print_history(history) -> None:
    """按 get_state_history 返回的“最新在前”顺序打印历史。"""
    print("\n检查点历史（最新在前）：")
    for snapshot in history:
        configurable = snapshot.config.get("configurable", {})
        checkpoint_id = configurable.get("checkpoint_id", "<unknown>")
        step = (snapshot.metadata or {}).get("step")
        source = (snapshot.metadata or {}).get("source")
        next_nodes = snapshot.next or ("END",)
        print(
            f"step={step!s:>3} | source={source!s:<6} | "
            f"next={next_nodes} | checkpoint_id={checkpoint_id}"
        )


def run_demo() -> None:
    """先完成原流程，再修改中间状态并从新分支继续执行。"""
    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    print("执行原始工作流……")
    original_result = graph.invoke({}, config=config)
    print(f"原主题：{original_result['topic']}")
    print(f"原笑话：{original_result['joke']}")

    original_latest = graph.get_state(config)
    history = list(graph.get_state_history(config))
    print_history(history)

    fork_base = find_snapshot_before_joke(history)
    print("\n选中的分叉起点：")
    print(f"topic={fork_base.values['topic']!r}")
    print(f"next={fork_base.next}")
    print(f"config={fork_base.config}")

    # as_node 表示这次状态更新等价于 generate_topic 刚刚完成，
    # 因而新检查点的下一节点仍然是 generate_joke。
    fork_config = graph.update_state(
        fork_base.config,
        values={"topic": FORK_TOPIC},
        as_node="generate_topic",
    )

    fork_snapshot = graph.get_state(fork_config)
    if fork_snapshot.next != ("generate_joke",):
        raise RuntimeError(
            "更新后的检查点没有指向 generate_joke，请检查 as_node。"
        )

    print("\n从修改后的检查点继续执行……")
    fork_result = graph.invoke(None, config=fork_config)
    print(f"新主题：{fork_result['topic']}")
    print(f"新笑话：{fork_result['joke']}")

    # update_state 会创建分支，不会就地改写旧检查点。
    preserved_original = graph.get_state(original_latest.config)
    print("\n原分支仍然可以读取：")
    print(f"原主题：{preserved_original.values['topic']}")
    print(f"原笑话：{preserved_original.values['joke']}")


if __name__ == "__main__":
    run_demo()
