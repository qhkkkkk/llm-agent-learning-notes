# MCP 文档资源与查阅文档 Agent

[完整笔记](../../notes/24-langchain-mcp-document-store.md) · [返回首页](../../README.md)

本例由两份 Python 文件组成：

| 文件 | 作用 |
| --- | --- |
| [docs_server.py](docs_server.py) | stdio 服务端，提供用户指南与密码 FAQ 两个资源 |
| [agent_client.py](agent_client.py) | 客户端启动服务、预先读取 FAQ，再创建本地工具供 Agent 调用 |

在仓库根目录使用 Python 3.11，先创建虚拟环境并安装根目录依赖，再配置 `.env`：

```ini
OPENROUTER_API_KEY=填写自己的密钥
MCP_DOCS_MODEL=openai/gpt-5-mini
```

只需运行客户端：

```powershell
.\.venv\Scripts\python.exe examples\mcp_document_store\agent_client.py
```

客户端通过 `sys.executable` 自动启动同目录的服务端。无需另外运行 `docs_server.py`，无需端口或浏览器。真实运行会向 OpenRouter 请求模型服务。

预期回答应包含“点击忘记密码链接”，措辞由模型决定，系统提示词不能代替工具调用的执行证据。资源是固定演示数据，不会实际重置密码。

验证不需要模型 Key：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_mcp_document_store.py -v
```

版本沿用前章：FastMCP 2.14.5、MCP 1.x、langchain-mcp-adapters 0.3.2。测试使用真实 stdio 文档服务与离线模型；Windows 运行环境需允许创建异步子进程命名管道。

注意：`query` 在这个最小示例中没有参与检索；工具只返回提前读到的 `text/plain` 文档。`file:///help/faq.json` 是服务注册的资源标识，不要求本机真的存在这个文件。
