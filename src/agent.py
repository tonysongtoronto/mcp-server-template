import asyncio
import json
import os
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = "http://127.0.0.1:8000/sse"

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


async def run_agent(session: ClientSession, openai_tools: list, user_question: str):
    """复用已有 session 和工具列表，执行单次对话"""
    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"\n🤖 模型决策: finish_reason = {finish_reason}")

        if finish_reason == "tool_calls" and msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"🔧 调用工具: {func_name}，参数: {func_args}")

                result = await session.call_tool(func_name, func_args)
                result_text = str(result.content[0].text if result.content else "无结果")

                print(f"📦 工具结果: {result_text}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        elif finish_reason == "stop":
            print(f"\n✨ 最终答案:\n{msg.content}")
            break

        else:
            print(f"⚠️ 未知状态: {finish_reason}")
            break


async def main():
    test_cases = [
        "请用 add_numbers 工具计算 123 + 456",
        "请用 multiply_numbers 工具计算 579 × 2",
        "请把 123 和 456 相加，然后再把结果乘以 2",
        "请调用 get_server_info 工具，告诉我服务器信息",
        "请用 fetch_url 工具获取 https://api.github.com/zen 的内容",
        '请用 post_json 工具向 https://httpbin.org/post 发送 POST 请求，payload 为 {"name": "Alice", "score": 95}',
        """请用 dataframe_summary 工具分析以下数据：
[{"name":"Alice","department":"Engineering","salary":9000,"age":30},
 {"name":"Bob","department":"Marketing","salary":7500,"age":25},
 {"name":"Charlie","department":"Engineering","salary":11000,"age":35},
 {"name":"Diana","department":"Marketing","salary":8000,"age":28},
 {"name":"Eve","department":"HR","salary":6500,"age":32}]""",
        """请用 group_and_aggregate 工具对以下数据按 department 分组，统计 salary 的总和：
[{"name":"Alice","department":"Engineering","salary":9000},
 {"name":"Bob","department":"Marketing","salary":7500},
 {"name":"Charlie","department":"Engineering","salary":11000},
 {"name":"Diana","department":"Marketing","salary":8000},
 {"name":"Eve","department":"HR","salary":6500}]""",
        """请用 group_and_aggregate 工具对以下数据按 department 分组，计算 salary 的平均值：
[{"name":"Alice","department":"Engineering","salary":9000},
 {"name":"Bob","department":"Marketing","salary":7500},
 {"name":"Charlie","department":"Engineering","salary":11000},
 {"name":"Diana","department":"Marketing","salary":8000},
 {"name":"Eve","department":"HR","salary":6500}]""",
        """请用 group_and_aggregate 工具对以下数据按 department 分组，统计每个部门的人数（agg_col 用 salary，agg_func 用 count）：
[{"name":"Alice","department":"Engineering","salary":9000},
 {"name":"Bob","department":"Marketing","salary":7500},
 {"name":"Charlie","department":"Engineering","salary":11000},
 {"name":"Diana","department":"Marketing","salary":8000},
 {"name":"Eve","department":"HR","salary":6500}]""",
    ]

    # ✅ 只建立一次连接，只加载一次工具
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools_result.tools
            ]
            print(f"✅ 工具加载完成（共 {len(openai_tools)} 个）: {[t['function']['name'] for t in openai_tools]}")

            # 依次跑所有测试
            total = len(test_cases)
            passed = 0

            for i, question in enumerate(test_cases, 1):
                print(f"\n{'='*60}")
                print(f"📋 测试 {i}/{total}")
                print(f"❓ 问题: {question[:80].strip()}...")
                print('='*60)
                try:
                    await run_agent(session, openai_tools, question)
                    passed += 1
                    print(f"✅ 测试 {i} 完成")
                except Exception as e:
                    print(f"❌ 测试 {i} 失败: {e}")

            print(f"\n{'='*60}")
            print(f"🏁 全部测试完成：{passed}/{total} 通过")
            print('='*60)


asyncio.run(main())