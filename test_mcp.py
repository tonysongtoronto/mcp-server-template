import asyncio
import sys
from pathlib import Path

from fastmcp import Client
from src.mcp_server_template.server import mcp   # 现在可以正常导入了


async def test_mcp_server():
  
    print("🚀 开始程序内测试 MCP Server...")

    async with Client(mcp) as client:        # in-memory 方式，最适合测试
        print("✅ 已连接到 MCP Server (in-memory)")

        # 1. 测试列出所有 Tools
        tools = await client.list_tools()
        print(f"可用 Tools: {[t.name for t in tools]}")

        # 2. 测试调用 add_numbers
        result = await client.call_tool("add_numbers", {"a": 10, "b": 25})
        print(f"add_numbers(10, 25) 结果: {result}")

        # 3. 测试 get_server_info
        info = await client.call_tool("get_server_info", {})
        print(f"服务器信息: {info}")

        # 4. 测试 Resource
        resource_content = await client.read_resource("welcome://message")
        print(f"欢迎资源内容: {resource_content}")

    print("🎉 所有测试完成！")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())