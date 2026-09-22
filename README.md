# LLM Agent Learning Notes

这是我的大模型应用与 Agent 开发学习笔记。每个主题都尽量包含：通俗解释、流程图、可运行代码、预期结果、常见错误和复习问题，方便零基础读者理解，也方便自己回顾。

## 学习目录

| 编号 | 主题 | 学到什么 | 资料 |
| --- | --- | --- | --- |
| 01 | LangChain RAG 问答链 | 检索器、Prompt、模型与输出解析器的链式组合 | [学习笔记](notes/01-langchain-rag-chain.md) · [示例代码](rag_chain_example.py) |
| 02 | LangGraph 包裹配送状态图 | 共享状态、Reducer、节点、固定边与条件路由 | [学习笔记](notes/02-langgraph-package-delivery.md) · [示例代码](langgraph_package_delivery.py) |
| 03 | LangGraph 工具调用搜索 Agent | 消息状态、Tool Calling、条件循环、Tavily 搜索与终止判断 | [学习笔记](notes/03-langgraph-tool-calling-search-agent.md) · [示例代码](langgraph_tavily_search_agent.py) |

## 建议学习顺序

1. 先运行包裹配送示例。它不调用大模型，也不需要 API Key，适合用来理解 LangGraph 的基本结构。
2. 再阅读 RAG 示例，理解如何把外部知识检索结果交给大模型生成回答。
3. 最后学习工具调用搜索 Agent，理解模型如何申请工具调用、程序如何执行工具，以及图如何循环并结束。
4. 尝试完成每篇笔记末尾的扩展练习，把“看懂”变成“会写”。

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

运行工具调用搜索 Agent：

```bash
python langgraph_tavily_search_agent.py
```

> RAG 示例还需要 OpenAI API Key、可连接的 Redis 服务以及已经写入向量库的文档。搜索 Agent 需要 Tavily API Key、模型 API Key，并要求所选模型支持 Tool Calling；具体说明见对应学习笔记。

## 仓库结构

```text
.
├── README.md
├── notes/
│   ├── 01-langchain-rag-chain.md
│   ├── 02-langgraph-package-delivery.md
│   └── 03-langgraph-tool-calling-search-agent.md
├── rag_chain_example.py
├── langgraph_package_delivery.py
├── langgraph_tavily_search_agent.py
└── requirements.txt
```

## 当前进度

- [x] LangChain：最小 RAG 问答链
- [x] LangGraph：状态、Reducer 与条件路由
- [x] LangGraph：循环与终止条件
- [ ] LangGraph：记忆与持久化
- [x] Agent：工具调用与多步骤任务

