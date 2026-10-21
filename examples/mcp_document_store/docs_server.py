"""通过 stdio 暴露两份固定文档资源；客户端负责启动这个子进程。"""

import json

from fastmcp import FastMCP

mcp = FastMCP("DocumentStore")


@mcp.resource("file:///help/guide.txt")
def get_guide() -> str:
    return """# 用户指南
1. 首先登录系统
2. 点击“新建项目”
3. 输入项目名称
"""


@mcp.resource("file:///help/faq.json")
def get_faq() -> str:
    # 返回 JSON 文本，未显式声明 MIME，因此本例仍是 text/plain。
    return json.dumps({
        "q1": "如何重置密码？",
        "a1": "请点击“忘记密码”链接。",
    }, ensure_ascii=False)


if __name__ == "__main__":
    # stdio 的 stdout 用于 MCP 协议，不在这里 print 业务日志。
    mcp.run(transport="stdio", show_banner=False)
