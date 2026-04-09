import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://127.0.0.1:8000/sse") as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()

            # 列出所有工具
            tools = await client.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            # 测试 add_numbers
            r = await client.call_tool("add_numbers", {"a": 3, "b": 7})
            print("add_numbers:", r.content[0].text)

            # 测试 fetch_url
            r = await client.call_tool(
                "fetch_url", {"url": "https://api.github.com/zen"}
            )
            print("fetch_url:", r.content[0].text)

            # 测试 dataframe_summary
            import json
            data = json.dumps([
                {"name": "Alice", "score": 90},
                {"name": "Bob", "score": 75},
            ])
            r = await client.call_tool("dataframe_summary", {"records_json": data})
            print("dataframe_summary:\n", r.content[0].text)


asyncio.run(main())