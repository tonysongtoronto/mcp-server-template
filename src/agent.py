import asyncio
import json
import os
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = "http://127.0.0.1:8000/sse"

# 初始化 OpenRouter 客户端
client = OpenAI(
    base_url="https://api.deepseek.com", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

async def run_agent(user_question: str):
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1️⃣ 获取 MCP 工具列表，转成 OpenAI 格式
            tools_result = await session.list_tools()
            openai_tools = []
            for tool in tools_result.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                })

            print(f"✅ 已加载 {len(openai_tools)} 个工具: {[t['function']['name'] for t in openai_tools]}")

            # 2️⃣ 开始对话循环
            messages = [{"role": "user", "content": user_question}]

            while True:
                response = client.chat.completions.create(
                    model="deepseek-chat",  # DeepSeek 模型名称：deepseek-chat 或 deepseek-coder  
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )

                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                print(f"\n🤖 模型决策: finish_reason = {finish_reason}")

                # 3️⃣ 模型决定调用工具
                if finish_reason == "tool_calls" and msg.tool_calls:
                    # 把模型回复加入历史
                    messages.append(msg)

                    # 逐个执行工具
                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)

                        print(f"🔧 调用工具: {func_name}，参数: {func_args}")

                        # 在 MCP Server 上真正执行
                        result = await session.call_tool(func_name, func_args)
                        result_text = str(result.content[0].text if result.content else "无结果")

                        print(f"📦 工具结果: {result_text}")

                        # 把结果反馈给模型
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        })

                # 4️⃣ 模型给出最终答案
                elif finish_reason == "stop":
                    print(f"\n✨ 最终答案:\n{msg.content}")
                    break

                else:
                    print(f"⚠️ 未知状态: {finish_reason}")
                    break


asyncio.run(run_agent("请把 123 和 456 相加，然后再把结果乘以 2"))