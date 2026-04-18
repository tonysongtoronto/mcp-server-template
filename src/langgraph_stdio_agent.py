"""
src/langgraph_stdio_agent.py

两种运行方式：
  1. uv run langgraph dev   → langgraph 调用 lifespan 钩子初始化工具
  2. python -m src.langgraph_stdio_agent  → __main__ 手动初始化工具
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from pathlib import Path
from pydantic import create_model

load_dotenv()

# ══════════════════════════════════════════════════════
# 1. LLM
# ══════════════════════════════════════════════════════
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# ══════════════════════════════════════════════════════
# 2. State
# ══════════════════════════════════════════════════════
class AgentState(TypedDict):
    messages: list

# ══════════════════════════════════════════════════════
# 3. MCP server 路径 & 启动参数
# ══════════════════════════════════════════════════════
SERVER_PATH = Path(__file__).parent / "mcp_server_template" / "server.py"

def mcp_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-u", str(SERVER_PATH)],
        env={"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **os.environ},
    )

# ══════════════════════════════════════════════════════
# 4. 工具加载
# ══════════════════════════════════════════════════════
async def load_tools(session: ClientSession) -> list[StructuredTool]:
    lc_tools: list[StructuredTool] = []
    for t in (await session.list_tools()).tools:
        schema    = t.inputSchema or {}
        required  = set(schema.get("required", []))
        fields    = {
            name: (Any, ...) if name in required else (Optional[Any], None)
            for name in schema.get("properties", {})
        }
        DynSchema = create_model(f"{t.name}_schema", **fields) if fields else None
        tool_name = t.name

        async def _call(_name=tool_name, **kwargs) -> str:
            print(f"    🔧 [MCP] {_name}({kwargs})")
            res  = await session.call_tool(_name, kwargs)
            text = res.content[0].text if res.content else "（无结果）"
            print(f"    ✅ {text[:200]}")
            return text

        lc_tools.append(StructuredTool.from_function(
            coroutine=_call, name=t.name,
            description=t.description or "", args_schema=DynSchema,
        ))

    print(f"✅ 已加载 {len(lc_tools)} 个工具：{[t.name for t in lc_tools]}")
    return lc_tools

# ══════════════════════════════════════════════════════
# 5. 共享工具容器 + graph 构建
#    _tools 是一个列表引用，build_graph 的闭包持有它。
#    无论谁（lifespan 或 __main__）向它 extend，
#    graph 的节点函数下次执行时自动读到新工具。
# ══════════════════════════════════════════════════════
_tools: list[StructuredTool] = []

def build_graph(tools_ref: list[StructuredTool]):
    MAX_STEPS = 20
    SYSTEM = SystemMessage(content=(
        "你是一个能够使用工具完成任务的助手。"
        "每次只调用一个工具，拿到结果后再决定下一步。"
        "所有子任务完成后，用中文给出最终答案。"
    ))

    async def reason(state: AgentState) -> AgentState:
        agent_llm = llm.bind_tools(tools_ref) if tools_ref else llm
        response  = await agent_llm.ainvoke([SYSTEM] + state["messages"])
        return {"messages": state["messages"] + [response]}

    async def act(state: AgentState) -> AgentState:
        by_name  = {t.name: t for t in tools_ref}
        last     = state["messages"][-1]
        new_msgs = []
        for tc in last.tool_calls:
            tool = by_name.get(tc["name"])
            if not tool:
                result = f"❌ 未知工具：{tc['name']}"
            else:
                # 过滤 None 后直接调底层协程，跳过 StructuredTool 的 schema 重新验证
                args   = {k: v for k, v in tc["args"].items() if v is not None}
                result = await tool.coroutine(**args)
            new_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        return {"messages": state["messages"] + new_msgs}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if sum(1 for m in state["messages"] if isinstance(m, ToolMessage)) >= MAX_STEPS:
            print(f"  ⚠️ 已达最大步数 {MAX_STEPS}，强制终止")
            return END
        return "act"

    g = StateGraph(AgentState)
    g.add_node("reason", reason)
    g.add_node("act",    act)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", route, {"act": "act", END: END})
    g.add_edge("act", "reason")
    return g.compile()

# langgraph dev 引用的就是这个对象
graph = build_graph(_tools)

# ══════════════════════════════════════════════════════
# 6. lifespan —— 仅 langgraph dev 调用
# ══════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app):
    if not SERVER_PATH.exists():
        raise FileNotFoundError(f"找不到 MCP server：{SERVER_PATH}")
    async with stdio_client(mcp_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _tools.extend(await load_tools(session))
            print("🚀 [lifespan] MCP 就绪")
            yield
    _tools.clear()
    print("🛑 [lifespan] MCP 已关闭")

# ══════════════════════════════════════════════════════
# 7. __main__ —— 单独运行测试
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    if not SERVER_PATH.exists():
        print(f"❌ 找不到 server.py：{SERVER_PATH}")
        sys.exit(1)

    QUESTIONS = [
        "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",
        "把 88 和 12 相加，再把结果乘以 5",
        """分析这批数据：[{"name":"Alice","dept":"Eng","salary":9000},
         {"name":"Bob","dept":"Mkt","salary":7500},
         {"name":"Charlie","dept":"Eng","salary":11000}]
         按 dept 分组，对 salary 求平均""",
        "访问 https://api.github.com/zen 返回了什么？",
    ]

    async def main():
        async with stdio_client(mcp_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                _tools.extend(await load_tools(session))  # 填充同一个 _tools
                for q in QUESTIONS:
                    print(f"\n{'━'*60}\n❓ {q}\n{'━'*60}")
                    result = await graph.ainvoke(
                        {"messages": [HumanMessage(content=q)]}
                    )
                    print(f"\n✨ {result['messages'][-1].content}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())   # stdio_client 在这个循环里开和关，不会跨循环 ✅