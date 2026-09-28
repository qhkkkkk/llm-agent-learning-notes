"""LangGraph Store 长期记忆、命名空间与语义搜索示例。"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore


load_dotenv()

STORE_BACKEND = os.getenv("STORE_BACKEND", "memory").lower()
USER_ID = os.getenv("LONG_TERM_USER_ID", "user_123")
MEMORY_QUERY = os.getenv("MEMORY_QUERY", "用户喜欢的中国音乐")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
STORE_EMBEDDING_MODEL = os.getenv(
    "STORE_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
STORE_EMBEDDING_DIMS = int(os.getenv("STORE_EMBEDDING_DIMS", "1536"))


def build_index_config() -> dict[str, Any]:
    """创建 Store 的向量索引配置。"""
    embeddings = init_embeddings(
        model=f"openai:{STORE_EMBEDDING_MODEL}",
        base_url=OPENAI_BASE_URL,
        # OpenAI 兼容服务通常需要直接接收文本，而不是本地切分后的 token。
        check_embedding_ctx_length=False,
    )
    return {
        "embed": embeddings,
        "dims": STORE_EMBEDDING_DIMS,
        "fields": ["text"],
    }


@contextmanager
def open_store() -> Iterator[BaseStore]:
    """根据环境变量打开内存或 PostgreSQL Store。"""
    index_config = build_index_config()

    if STORE_BACKEND == "memory":
        yield InMemoryStore(index=index_config)
        return

    if STORE_BACKEND == "postgres":
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise RuntimeError(
                "STORE_BACKEND=postgres 时必须配置 POSTGRES_URI。"
            )

        with PostgresStore.from_conn_string(
            postgres_uri,
            index=index_config,
        ) as store:
            # 首次使用必须执行；重复调用会检查并应用尚未执行的迁移。
            store.setup()
            yield store
        return

    raise ValueError(
        "STORE_BACKEND 只能是 'memory' 或 'postgres'，"
        f"当前值为 {STORE_BACKEND!r}。"
    )


def seed_memories(store: BaseStore) -> dict[str, tuple[str, ...]]:
    """写入一组可重复运行的示例长期记忆。"""
    namespaces = {
        "food": (USER_ID, "preferences", "food"),
        "music": (USER_ID, "preferences", "music"),
        "profile": (USER_ID, "profile"),
    }

    # namespace + key 共同组成唯一地址；相同地址再次 put 会更新旧值。
    store.put(
        namespace=namespaces["food"],
        key="favorite-food",
        value={
            "text": "用户喜欢吃披萨。",
            "category": "food",
            "source": "demo",
        },
    )
    store.put(
        namespace=namespaces["music"],
        key="favorite-song",
        value={
            "text": "用户喜欢周杰伦的中文歌曲《夜曲》。",
            "category": "music",
            "source": "demo",
        },
    )
    store.put(
        namespace=namespaces["profile"],
        key="preferred-language",
        value={
            "text": "用户希望助手优先使用中文回答。",
            "category": "profile",
            "source": "demo",
        },
    )

    return namespaces


def print_item(label: str, item: Any) -> None:
    """以适合初学者阅读的格式输出 Store 条目。"""
    if item is None:
        print(f"{label}：未找到")
        return

    print(
        f"{label}：namespace={item.namespace}, "
        f"key={item.key}, value={item.value}"
    )


def main() -> None:
    """演示写入、精确读取、前缀检索和语义搜索。"""
    with open_store() as store:
        namespaces = seed_memories(store)
        print(f"当前 Store：{STORE_BACKEND}")

        # 1. 精确读取：必须同时知道完整 namespace 和 key。
        exact_item = store.get(
            namespace=namespaces["food"],
            key="favorite-food",
        )
        print_item("\n精确读取", exact_item)

        # 2. 前缀检索：取回当前用户 preferences 下的所有分类记忆。
        preference_items = store.search((USER_ID, "preferences"))
        print("\n用户偏好（命名空间前缀检索）：")
        for item in preference_items:
            print_item("- 记忆", item)

        # 3. 语义搜索：查询文字不必与记忆原文完全相同。
        semantic_items = store.search(
            (USER_ID, "preferences"),
            query=MEMORY_QUERY,
            limit=2,
        )
        print(f"\n语义搜索：{MEMORY_QUERY!r}")
        for item in semantic_items:
            score = getattr(item, "score", None)
            score_text = "无" if score is None else f"{score:.4f}"
            print_item(f"- 相似度={score_text}", item)


if __name__ == "__main__":
    main()
