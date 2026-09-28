# LLM Agent Learning Notes

这是我的大模型应用与 Agent 开发学习笔记。每个主题都尽量包含：通俗解释、流程图、可运行代码、预期结果、常见错误和复习问题，方便零基础读者理解，也方便自己回顾。

## 学习目录

| 编号 | 主题 | 学到什么 | 资料 |
| --- | --- | --- | --- |
| 01 | LangChain RAG 问答链 | 检索器、Prompt、模型与输出解析器的链式组合 | [学习笔记](notes/01-langchain-rag-chain.md) · [示例代码](rag_chain_example.py) |
| 02 | LangGraph 包裹配送状态图 | 共享状态、Reducer、节点、固定边与条件路由 | [学习笔记](notes/02-langgraph-package-delivery.md) · [示例代码](langgraph_package_delivery.py) |
| 03 | LangGraph 工具调用搜索 Agent | 消息状态、Tool Calling、条件循环、Tavily 搜索与终止判断 | [学习笔记](notes/03-langgraph-tool-calling-search-agent.md) · [示例代码](langgraph_tavily_search_agent.py) |
| 04 | LangGraph Agentic RAG | 知识库工具、相关性评分、问题改写、循环检索与流式执行 | [学习笔记](notes/04-langgraph-agentic-rag.md) · [示例代码](langgraph_agentic_rag.py) |
| 05 | LangGraph 顺序写作工作流 | 输入与输出 Schema、内部状态、Prompt Chaining 与顺序节点 | [学习笔记](notes/05-langgraph-sequential-writing-workflow.md) · [示例代码](langgraph_sequential_writing.py) |
| 06 | LangGraph Orchestrator-Worker | 结构化任务拆分、Send 动态并行、Reducer 汇总与扇出/扇入 | [学习笔记](notes/06-langgraph-orchestrator-worker.md) · [示例代码](langgraph_orchestrator_worker.py) |
| 07 | LangGraph PostgreSQL 持久化与时间旅行 | PostgresSaver、线程记忆、状态快照、Replay 与 Fork | [学习笔记](notes/07-langgraph-postgres-persistence-time-travel.md) · [示例代码](langgraph_postgres_time_travel.py) |
| 08 | LangGraph 长期记忆 Store | 命名空间、精确读取、前缀检索、PostgresStore 与语义搜索 | [学习笔记](notes/08-langgraph-long-term-memory-store.md) · [示例代码](langgraph_long_term_memory_store.py) |
| 09 | LangGraph 跨线程个性化 Agent | 应用与框架能力边界、结构化记忆提取、Runtime Context、记忆合并与跨线程复用 | [学习笔记](notes/09-langgraph-cross-thread-personalization.md) · [示例代码](langgraph_cross_thread_personalization.py) |

## 建议学习顺序

1. 先运行包裹配送示例。它不调用大模型，也不需要 API Key，适合用来理解 LangGraph 的基本结构。
2. 再阅读 RAG 示例，理解如何把外部知识检索结果交给大模型生成回答。
3. 接着学习工具调用搜索 Agent，理解模型如何申请工具调用、程序如何执行工具，以及图如何循环并结束。
4. 学习顺序写作工作流，理解输入、内部过程状态、输出过滤和多阶段 Prompt Chaining。
5. 学习 Orchestrator-Worker，理解协调者如何用 `Send` 动态分配并行任务，再用 Reducer 汇总结果。
6. 学习 PostgreSQL 持久化与时间旅行，理解线程记忆、状态历史、重放和状态分叉。
7. 学习长期记忆 Store，理解跨线程数据、命名空间、键值读写和语义搜索。
8. 把 Store 接入个性化 Agent，先区分应用能力与 LangGraph 原语，再理解结构化提取、Runtime、记忆合并和跨线程复用。
9. 最后学习 Agentic RAG，观察系统如何检查检索质量，并在结果不理想时改写问题、重新检索。
10. 尝试完成每篇笔记末尾的扩展练习，把“看懂”变成“会写”。

## 快速开始

```bash
git clone https://github.com/qhkkkkk/llm-agent-learning-notes.git
cd llm-agent-learning-notes

python -m venv .venv
```

激活虚拟环境：

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

安装依赖并运行无需外部服务的 LangGraph 示例：

```bash
pip install -r requirements.txt
python langgraph_package_delivery.py
```

需要模型或搜索服务的示例，请先复制环境变量模板并填写自己的密钥：

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

运行工具调用搜索 Agent：

```bash
python langgraph_tavily_search_agent.py
```

运行带相关性评分与问题改写的 Agentic RAG：

```bash
python langgraph_agentic_rag.py
```

运行“大纲 → 初稿 → 润色 → 终稿”的顺序写作工作流：

```bash
python langgraph_sequential_writing.py
```

运行“协调者拆分 → 工作者并行生成 → 汇总器合并”的报告工作流：

```bash
python langgraph_orchestrator_worker.py
```

运行带 PostgreSQL 记忆、检查点历史和时间旅行的搜索 Agent：

```bash
python langgraph_postgres_time_travel.py
```

运行长期记忆 Store 示例（默认使用内存后端，也可切换 PostgreSQL）：

```bash
python langgraph_long_term_memory_store.py
```

运行能够跨两个对话线程复用用户偏好的个性化 Agent：

```bash
python langgraph_cross_thread_personalization.py
```

> 最小 RAG 示例还需要可连接的 Redis 服务以及已经写入向量库的文档。搜索 Agent 需要 Tavily API Key；PostgreSQL 持久化示例还需要可连接的 PostgreSQL 数据库。长期记忆 Store 示例需要 Embedding API，切换 PostgreSQL 后端时还需要数据库和 `pgvector`。Agentic RAG 会在启动时读取 `Docs/markdown/` 中的本地资料并创建内存向量索引。具体配置和限制见对应学习笔记。

## 仓库结构

```text
.
├── Docs/markdown/                  # Agentic RAG 使用的原始知识库资料
├── notes/
│   ├── 01-langchain-rag-chain.md
│   ├── 02-langgraph-package-delivery.md
│   ├── 03-langgraph-tool-calling-search-agent.md
│   ├── 04-langgraph-agentic-rag.md
│   ├── 05-langgraph-sequential-writing-workflow.md
│   ├── 06-langgraph-orchestrator-worker.md
│   ├── 07-langgraph-postgres-persistence-time-travel.md
│   ├── 08-langgraph-long-term-memory-store.md
│   └── 09-langgraph-cross-thread-personalization.md
├── .env.example
├── langgraph_agentic_rag.py
├── langgraph_cross_thread_personalization.py
├── langgraph_long_term_memory_store.py
├── langgraph_orchestrator_worker.py
├── langgraph_package_delivery.py
├── langgraph_postgres_time_travel.py
├── langgraph_sequential_writing.py
├── langgraph_tavily_search_agent.py
├── rag_chain_example.py
└── requirements.txt
```

## 当前进度

- [x] LangChain：最小 RAG 问答链
- [x] LangGraph：状态、Reducer 与条件路由
- [x] LangGraph：循环与终止条件
- [x] LangGraph：记忆与持久化
- [x] Agent：工具调用与多步骤任务
- [x] Agentic RAG：检索、评分、改写与重新检索
- [x] LangGraph：输入输出 Schema 与顺序写作工作流
- [x] LangGraph：Orchestrator-Worker 与 Send 动态并行
- [x] LangGraph：状态历史、Replay 与 Fork 时间旅行
- [x] LangGraph：Store 长期记忆、命名空间与语义搜索
- [x] LangGraph：结构化长期记忆与跨线程个性化
