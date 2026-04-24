import asyncio
import os
import sys
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()  # ✅ 关键！允许嵌套事件循环

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from langgraph_stdio_agent import build_graph

SERVER_PATH = Path(__file__).parent.parent / "src" / "mcp_server_template" / "server.py"

params = StdioServerParameters(
    command=sys.executable,
    args=["-u", str(SERVER_PATH)],
    env={
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        **os.environ,
    },
)

async def run_evaluation():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ 工具初始化完成，开始评估...")

            graph = build_graph()

            def target(inputs):
                messages = inputs["messages"]
                # ✅ 用 asyncio.run() 替代 run_until_complete()
                result = asyncio.run(
                    graph.ainvoke({
                        "messages": messages,
                        "task_plan": [],
                        "current_task_index": 0,
                        "next_agent": "",
                    })
                )
                return {"messages": result["messages"]}

            results = evaluate(
                target,
                data="tonyset",        # ✅ 修改为你的 dataset 名称
                evaluators=[],
                experiment_prefix="test-run"
            )
            print(results)

asyncio.run(run_evaluation())

# uv run tests/eval_agent.py