import asyncio
import sys
import os
import traceback
from pathlib import Path
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# server.py 的正确路径
SERVER_PATH = Path(__file__).parent / "src" / "mcp_server_template" / "server.py"


async def test_mcp_server():
    print("=" * 60)
    print("🚀 开始测试 MCP Server（stdio）")
    print(f"平台: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Server 路径: {SERVER_PATH}")

    if not SERVER_PATH.exists():
        print(f"❌ 找不到 server.py：{SERVER_PATH}")
        return
    print("=" * 60)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", str(SERVER_PATH)],
        env={
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            **os.environ,
        },
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:

                # 1. 初始化
                print("\n步骤 1: 初始化 MCP 连接...")
                await client.initialize()
                print("✅ 初始化成功")

                # 2. 列出工具
                print("\n步骤 2: 列出可用工具...")
                tools = await client.list_tools()
                print(f"✅ Tools: {[t.name for t in tools.tools]}")

                # 3. 调用工具
                print("\n步骤 3: 调用 add_numbers(10, 25)...")
                result = await client.call_tool("add_numbers", {"a": 10, "b": 25})
                print(f"✅ 结果: {result.content[0].text}")

                print("\n步骤 4: 调用 multiply_numbers(6, 7)...")
                result = await client.call_tool("multiply_numbers", {"a": 6, "b": 7})
                print(f"✅ 结果: {result.content[0].text}")

                # 4. 列出资源
                print("\n步骤 5: 列出可用资源...")
                resources = await client.list_resources()
                print(f"✅ Resources: {[str(r.uri) for r in resources.resources]}")

                # 5. 读取资源
                print("\n步骤 6: 读取 welcome://message...")
                res = await client.read_resource("welcome://message")
                print(f"✅ 内容: {res.contents[0].text}")

                print("\n" + "=" * 60)
                print("🎉 所有测试通过！")
                print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(test_mcp_server())