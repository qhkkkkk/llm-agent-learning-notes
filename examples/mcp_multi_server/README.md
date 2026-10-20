# MCP 双服务 Agent

三份 Python 文件是一套示例，详细原理见[第 23 篇学习笔记](../../notes/23-langchain-mcp-multi-server.md)。

| 文件 | 职责 | 谁启动 |
| --- | --- | --- |
| [math_server.py](math_server.py) | add、multiply，通过 stdio 提供工具 | 客户端自动启动子进程 |
| [weather_server.py](weather_server.py) | 固定演示天气，通过 Streamable HTTP 提供工具 | 先在单独终端启动 |
| [agent_client.py](agent_client.py) | 打开两个 MCP 会话，合并工具，运行异步 Agent | 在另一个终端启动 |

从仓库根目录执行，两个终端都使用同一个虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

本例固定 `fastmcp==2.14.5`、`langchain-mcp-adapters==0.3.2`，使用 `mcp>=1.24,<2`。不要直接将这套适配器与要求 MCP 2.x 的 FastMCP 4.x 混装。

尚未建立 `.env` 时，从根目录模板复制；已有文件直接补充：

```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
MCP_AGENT_MODEL=openai/gpt-5-mini
MCP_WEATHER_PORT=8000
```

终端 A，启动天气服务并保持运行：

```powershell
python examples/mcp_multi_server/weather_server.py
```

终端 B，启动客户端：

```powershell
.venv\Scripts\Activate.ps1
python examples/mcp_multi_server/agent_client.py
```

计算器服务由客户端使用当前 Python 解释器启动，无需额外开一个终端运行它。默认天气地址是 `http://127.0.0.1:8000/mcp`。换端口时修改根目录 `.env` 的 `MCP_WEATHER_PORT`，重启天气服务和客户端。

如果配置了代理，在客户端终端确保本机地址绕过代理：

```powershell
$env:NO_PROXY="127.0.0.1,localhost"
```

客户端在两个会话内部完成所有工具调用；`get_tools()` 的按次建会话方式、持久会话区别和 401 排错见完整笔记。天气仅为固定演示值，系统提示词的“必须使用工具”不是协议级强制保证。

不调用真实模型的本机验证：

```powershell
python -m unittest discover -s tests -p test_mcp_multi_server.py -v
```

测试会启动临时本机 HTTP 天气服务及 stdio 计算器子进程；使用空闲端口并清理自己的进程。外部 HTTP 请求被阻止，模型使用离线替身。
