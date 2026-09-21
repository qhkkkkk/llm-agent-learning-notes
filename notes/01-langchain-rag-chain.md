# 01｜LangChain RAG 问答链

这是我学习大模型应用开发时整理的阶段性笔记。本仓库目前记录了一个基于 LangChain 的 RAG 问答链路示例，重点是理解如何把检索器、提示词、大模型和输出解析器串成一条可运行的链。

## 学习主题

本次学习主要围绕 LangChain 的基础链式调用展开，并完成了一个最小可用的 RAG 流程。

RAG 的基本思想是：

```text
不是直接问大模型，而是先从知识库中查资料，再让大模型基于资料回答。
```

这样可以让回答更贴近已有资料，也能降低模型凭空编造的概率。

## 整体流程

这个示例实现的流程如下：

```text
用户问题
  ↓
Retriever 检索相关文档
  ↓
format_docs 整理检索结果
  ↓
PromptTemplate 组装提示词
  ↓
ChatOpenAI 调用大模型
  ↓
StrOutputParser 输出字符串结果
```

## 核心组件

### OpenAIEmbeddings

```python
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
```

`OpenAIEmbeddings` 用来把文本转换成向量。文本被向量化之后，才能存入向量数据库，并用于后续的相似度检索。

### ChatOpenAI

```python
model = ChatOpenAI(model="gpt-4o-mini")
```

`ChatOpenAI` 是最终负责生成回答的大语言模型。它会根据用户问题和检索到的上下文生成简洁回答。

### RedisVectorStore

```python
vector_store = RedisVectorStore(
    embeddings=embedding,
    config=config,
)
```

这里使用 Redis 作为向量数据库。Redis 中保存文档向量，并支持根据用户问题检索相关内容。

### Retriever

```python
retriever = vector_store.as_retriever()
```

`retriever` 是检索器。它会根据用户输入的问题，从 Redis 向量库中找出最相关的文档。

### ChatPromptTemplate

```python
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
```

提示词模板规定了模型的回答方式。这里要求模型基于检索到的上下文回答，且不知道时直接说明不知道。

### RunnablePassthrough

```python
"question": RunnablePassthrough()
```

`RunnablePassthrough` 会把用户输入原样传递下去。在这个链里，用户问题一方面用于检索，另一方面也要被放进 Prompt。

### StrOutputParser

```python
StrOutputParser()
```

`StrOutputParser` 用来把模型输出解析成普通字符串，方便最终打印或返回给用户。

## 完整代码

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_redis import RedisConfig, RedisVectorStore


# 1. 定义嵌入模型
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# 2. 定义聊天模型
model = ChatOpenAI(model="gpt-4o-mini")


# 3. Redis 配置
config = RedisConfig(
    index_name="qa",
    redis_url="redis://localhost:6379",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"},
    ],
)


# 4. 初始化 Redis 向量存储
vector_store = RedisVectorStore(
    embeddings=embedding,
    config=config,
)

# 5. 创建检索器
retriever = vector_store.as_retriever()


# 6. 构建提示词模板
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


# 7. 将检索出来的文档转换为文本
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 8. 构建 RAG 链
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


# 9. 流式输出结果
for chunk in chain.stream("介绍一下这个项目"):
    print(chunk, end="|", flush=True)
```

## 链式结构拆解

最关键的代码是这一段：

```python
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

它可以拆成两条并行分支理解。

第一条分支负责生成上下文：

```python
"context": retriever | format_docs
```

用户输入问题后，`retriever` 会去 Redis 向量库中检索相关文档。检索出来的文档再经过 `format_docs` 转换成字符串，作为 Prompt 中的 `context`。

第二条分支负责保留原始问题：

```python
"question": RunnablePassthrough()
```

这一支会把用户输入的问题原样传递给 Prompt，作为 `question`。

最终会形成类似这样的数据：

```python
{
    "context": "检索出来的相关文档内容",
    "question": "介绍一下这个项目"
}
```

然后这些数据会被填入提示词模板，再交给大模型生成回答。

## 本次学习收获

通过这个示例，我走通了 LangChain 中 RAG 链式调用的核心流程：

- 使用 Embedding 模型将文本向量化
- 使用 Redis 作为向量数据库
- 创建 Retriever 检索器
- 使用 PromptTemplate 组织输入
- 使用 RunnablePassthrough 透传原始问题
- 使用管道符 `|` 把多个组件串成一条链
- 使用 StrOutputParser 得到最终字符串结果
- 使用 `chain.stream()` 实现流式输出

## 我的理解

这个例子本质上是一个最小可用的 RAG 问答系统。

大模型本身不直接知道项目资料，而是先通过检索器从知识库中查找相关内容，再基于这些内容回答问题。这个流程非常适合用于项目文档问答、知识库问答、企业内部资料问答等场景。

后续可以继续扩展的方向包括：

- 增加文档入库流程
- 增加元数据过滤检索
- 优化 Prompt 模板
- 加入多轮对话记忆
- 封装成 API 服务
- 做成一个可交互的 Web 页面
