"""天气 MCP 服务；单独启动，通过 Streamable HTTP 接收请求。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
def get_weather_for_location(city: str) -> str:
    """返回指定城市的演示天气，固定为阳光明媚。"""
    return f"在{city}总是阳光明媚！"


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    mcp.run(
        transport="streamable-http",
        show_banner=False,
        host="127.0.0.1",
        port=int(os.getenv("MCP_WEATHER_PORT", "8000")),
    )
