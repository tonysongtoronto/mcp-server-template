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


def pretty_args(args: dict) -> str:
    """把参数字典格式化为多行缩进，records_json 太长则截断显示"""
    display = {}
    for k, v in args.items():
        if k == "records_json" and len(str(v)) > 80:
            display[k] = str(v)[:80] + "... (已截断)"
        else:
            display[k] = v
    return json.dumps(display, ensure_ascii=False, indent=4)


def pretty_result(text: str, max_len: int = 300) -> str:
    """结果超长时截断，并保持缩进对齐"""
    lines = text.strip().splitlines()
    formatted = "\n    ".join(lines)  # 每行缩进 4 空格
    if len(formatted) > max_len:
        return formatted[:max_len] + "\n    ... (已截断)"
    return formatted


async def run_agent(session: ClientSession, openai_tools: list, user_question: str):
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

        print(f"\n  🤖 模型决策: {finish_reason}")

        if finish_reason == "tool_calls" and msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"\n  🔧 调用工具: {func_name}")
                print(f"  📥 参数:\n    {pretty_args(func_args)}")

                result = await session.call_tool(func_name, func_args)
                result_text = str(
                    result.content[0].text if result.content else "无结果"
                )

                print(f"\n  📦 工具结果:\n    {pretty_result(result_text)}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )

        elif finish_reason == "stop":
            print(f"\n  ✨ 最终答案:\n    {msg.content}")
            break

        else:
            print(f"\n  ⚠️ 未知状态: {finish_reason}")
            break


async def main():
    test_cases = [
        # 1. 加法
        "123 加 456 等于多少？",
        # 2. 乘法
        "579 乘以 2 等于多少？",
        # 3. 加法 + 乘法组合
        "请把 123 和 456 相加，然后再把结果乘以 2",
        # 4. get_server_info
        "你现在运行在什么平台上？Python 版本是多少？",
        # 5. fetch_url
        "帮我访问 https://api.github.com/zen 看看返回什么内容",
        # 6. post_json
        "帮我向 https://httpbin.org/post 提交一份数据，内容是 name 为 Alice，score 为 95",
        # 7. dataframe_summary
        """帮我统计分析一下这批员工数据：
        [{"name":"Alice","department":"Engineering","salary":9000,"age":30},
        {"name":"Bob","department":"Marketing","salary":7500,"age":25},
        {"name":"Charlie","department":"Engineering","salary":11000,"age":35},
        {"name":"Diana","department":"Marketing","salary":8000,"age":28},
        {"name":"Eve","department":"HR","salary":6500,"age":32}]""",
        # 8. group_and_aggregate — sum
        """每个部门的薪资总支出是多少？数据如下：
        [{"name":"Alice","department":"Engineering","salary":9000},
        {"name":"Bob","department":"Marketing","salary":7500},
        {"name":"Charlie","department":"Engineering","salary":11000},
        {"name":"Diana","department":"Marketing","salary":8000},
        {"name":"Eve","department":"HR","salary":6500}]""",
        # 9. group_and_aggregate — mean
        """各部门的平均薪资是多少？数据如下：
        [{"name":"Alice","department":"Engineering","salary":9000},
        {"name":"Bob","department":"Marketing","salary":7500},
        {"name":"Charlie","department":"Engineering","salary":11000},
        {"name":"Diana","department":"Marketing","salary":8000},
        {"name":"Eve","department":"HR","salary":6500}]""",
        # 10. group_and_aggregate — count
        """每个部门各有几个人？数据如下：
        [{"name":"Alice","department":"Engineering","salary":9000},
        {"name":"Bob","department":"Marketing","salary":7500},
        {"name":"Charlie","department":"Engineering","salary":11000},
        {"name":"Diana","department":"Marketing","salary":8000},
        {"name":"Eve","department":"HR","salary":6500}]""",
    ]

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

            tool_names = [t["function"]["name"] for t in openai_tools]
            print(f"\n✅ 工具加载完成（共 {len(openai_tools)} 个）")
            print(f"   {tool_names}\n")

            total = len(test_cases)
            passed = 0

            for i, question in enumerate(test_cases, 1):
                print(f"\n{'━'*60}")
                print(f"  📋 测试 {i:02d}/{total}")
                print(
                    f"  ❓ {question.strip()[:60]}{'...' if len(question.strip()) > 60 else ''}"
                )
                print(f"{'━'*60}")

                try:
                    await run_agent(session, openai_tools, question)
                    passed += 1
                    print(f"\n  ✅ 测试 {i:02d} 通过")
                except Exception as e:
                    print(f"\n  ❌ 测试 {i:02d} 失败: {e}")

            print(f"\n{'━'*60}")
            print(f"  🏁 全部完成：{passed}/{total} 通过")
            print(f"{'━'*60}\n")


asyncio.run(main())
