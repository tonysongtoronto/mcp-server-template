import sys
import asyncio
import logging
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)

mcp = FastMCP("MCP Server Template")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """两个数字相加"""
    return a + b


@mcp.tool()
def multiply_numbers(a: int, b: int) -> int:
    """两个数字相乘"""
    return a * b


@mcp.tool()
def get_server_info() -> str:
    """返回服务器信息"""
    return "MCP Server Template 运行中，平台: {}, Python: {}".format(
        sys.platform, sys.version.split()[0]
    )


@mcp.resource("welcome://message")
def welcome_message() -> str:
    """欢迎资源"""
    return "欢迎使用企业级 MCP Server 模板"


@mcp.resource("info://server")
def server_info() -> str:
    """服务器信息资源"""
    return f"运行在 {sys.platform} 平台，Python {sys.version}"


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # 单独运行时用 --dev 启动 SSE 模式，可以在浏览器测试
    if "--dev" in sys.argv:
        print("🚀 开发模式启动，访问 http://127.0.0.1:6274", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        # 默认 stdio 模式，给 MCP 客户端（Claude Desktop 等）用
        mcp.run(transport="stdio")