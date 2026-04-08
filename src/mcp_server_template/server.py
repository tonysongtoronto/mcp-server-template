from fastmcp import FastMCP
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("mcp-server-template")

@mcp.tool()
async def add_numbers(a: int, b: int) -> int:
    """两个数字相加"""
    return a + b

@mcp.tool()
async def get_server_info() -> dict:
    """返回服务器基本信息"""
    return {
        "name": "MCP Server Template",
        "version": "0.1.0",
        "python_version": os.getenv("PYTHON_VERSION", "3.12"),
    }

@mcp.resource("welcome://message")
async def get_welcome_message() -> str:
    """欢迎资源"""
    return "欢迎使用企业级 MCP Server 模板！（uv + pyproject.toml + Docker）"

@mcp.resource("echo://{text}")
async def echo(text: str) -> str:
    """简单回显资源（演示带参数的 Resource）"""
    return f"Echo: {text}"

if __name__ == "__main__":
    print("🚀 MCP Server 启动中... (按 Ctrl+C 停止)")
    mcp.run(transport="stdio")