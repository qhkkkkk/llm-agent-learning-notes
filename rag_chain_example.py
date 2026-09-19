from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_redis import RedisConfig, RedisVectorStore


embedding = OpenAIEmbeddings(model="text-embedding-3-large")
model = ChatOpenAI(model="gpt-4o-mini")


config = RedisConfig(
    index_name="qa",
    redis_url="redis://192.168.100.238:6379",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"},
    ],
)


vector_store = RedisVectorStore(
    embeddings=embedding,
    config=config,
)

retriever = vector_store.as_retriever()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """你是负责回答问题的助手。使用以下检索到的上下文片段来回答问题。
如果你不知道答案，就说不知道。最多回复三句话的结果，回答要简明扼要。

Question: {question}
Context: {context}
Answer:
""",
        )
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


for chunk in chain.stream("介绍一下这个项目"):
    print(chunk, end="|", flush=True)
