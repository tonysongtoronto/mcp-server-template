from dotenv import load_dotenv
load_dotenv()

import os
import sys
import asyncio

# ── 路径修复：把项目根目录加入 sys.path ──────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ── 环境变量映射（旧版 LANGCHAIN_* → 新版 LANGSMITH_*）──
os.environ.setdefault("LANGSMITH_API_KEY",  os.environ.get("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGSMITH_ENDPOINT", os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))
os.environ.setdefault("LANGSMITH_TRACING",  "true")

from langchain_core.messages import HumanMessage
from langsmith import evaluate, Client
from mcp import ClientSession
from mcp.client.stdio import stdio_client

# ── 导入你的 agent ────────────────────────────────────
from src.langgraph_stdio_agent import (
    graph,
    load_tools,
    _tools,
    _init_registry,
    _rebuild_graph,
    mcp_params,
    _get_message_content,
)

# ── Windows 异步兼容 ──────────────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ── LangSmith 客户端 & 数据集 ─────────────────────────
client = Client()
dataset_name = "tonyset"


# ── 评估函数 ──────────────────────────────────────────
def check_ai_response(outputs: dict, reference_outputs: dict) -> bool:
    """比较实际输出与参考输出中 AI 最后一条消息的内容是否一致。"""
    def get_last_ai_content(msg_list):
        for msg in reversed(msg_list):
            # 兼容 dict 格式
            if isinstance(msg, dict) and msg.get("type") == "ai":
                return msg.get("content", "")
            # 兼容 AIMessage 对象
            if hasattr(msg, "type") and msg.type == "ai":
                return msg.content or ""
        return ""

    actual   = get_last_ai_content(outputs.get("messages", []))
    expected = get_last_ai_content(reference_outputs.get("messages", []))
    return actual.strip() == expected.strip()


# ── 真实 Agent 调用（同步包装异步）────────────────────
def my_agent(inputs: dict) -> dict:
    """
    接收 dataset 的 inputs，调用真实 MCP agent，返回结果。
    inputs 结构：{"messages": [{"role": "human", "content": "..."}]}
    """
    messages = inputs["messages"]
    first_msg = messages[0]
    user_content = (
        first_msg["content"]
        if isinstance(first_msg, dict)
        else first_msg.content
    )

    async def _run():
        async with stdio_client(mcp_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 每次调用前重新加载工具，确保状态干净
                loaded = await load_tools(session)
                _tools.clear()
                _tools.extend(loaded)
                _init_registry(loaded)
                _rebuild_graph()

                result = await graph.ainvoke({
                    "messages":   [HumanMessage(content=user_content)],
                    "task_plan":  [],
                    "next_agent": "",
                })
                return result

    result = asyncio.run(_run())
    return {"messages": result.get("messages", [])}


# ── 运行评估 ──────────────────────────────────────────
evaluate(
    my_agent,
    data=dataset_name,
    evaluators=[check_ai_response],
    experiment_prefix="tonyset real agent",
)

# uv run tests/run_dataset.py