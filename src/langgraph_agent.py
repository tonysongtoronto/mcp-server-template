"""
langgraph_agent.py — 优化版 v3

相对 v2 的核心改动：
1. StructuredTool 的执行函数改为真正调用 session.call_tool() 的闭包
2. tool_node 改用 LangChain ToolNode，通过工具名查找并执行，不再手动解析 tool_calls
3. AgentState 移除 mcp_session 字段（session 已闭包进每个工具的执行函数）
4. 新增全局 tools_by_name 字典，供 tool_node 按名查找工具
"""

import asyncio
import json
import os
import sys
from typing import TypedDict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode          # ← 新增：使用预置 ToolNode
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import create_model
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


# ── 1. 定义 State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: list   # 对话历史（HumanMessage / AIMessage / ToolMessage）
    # ↑ 移除了 mcp_session：session 已闭包进各工具执行函数，无需经 state 传递


# ── 2. 创建 LLM + 全局缓存 ────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0,
)

llm_with_tools: Any = None          # 启动时由 init_tools() 赋值
tools_by_name: dict = {}            # 工具名 → StructuredTool，供 tool_node 查找


# ── 3. 初始化全局工具（启动时调用一次）────────────────────────────────────────
async def init_tools(session: ClientSession) -> list[StructuredTool]:
    """
    从 MCP Server 拉取工具列表，为每个工具构建一个真正可执行的 StructuredTool：
    - args_schema  : 由 inputSchema 动态生成的 Pydantic 模型
    - coroutine    : 闭包，内部调用 session.call_tool()，LangChain 可直接 invoke

    同时完成 llm.bind_tools() 全局绑定，整个生命周期只执行一次。
    """
    global llm_with_tools, tools_by_name

    tools_result = await session.list_tools()
    lc_tools: list[StructuredTool] = []

    for t in tools_result.tools:
        schema = t.inputSchema or {}
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        # 动态 Pydantic Schema，bind_tools 需要它导出参数结构给大模型
        field_definitions = {
            name: (Any, ...) if name in required_fields else (Optional[Any], None)
            for name in properties
        }
        DynamicSchema = (
            create_model(f"{t.name}_schema", **field_definitions)
            if field_definitions
            else None
        )

        # ── 关键改动：用闭包捕获 tool_name，真正调用 session.call_tool() ──────
        tool_name = t.name  # 闭包变量，避免循环变量陷阱

        async def _call_tool(_tool_name=tool_name, **kwargs) -> str:
            """LangChain 调用此函数 → 转发给 MCP session 执行真实工具"""
            print(f"\n🔧 [LangChain] 调用工具: {_tool_name}")
            print(f"   参数: {kwargs}")
            result = await session.call_tool(_tool_name, kwargs)
            result_text = result.content[0].text if result.content else "（无结果）"
            print(f"   ✅ 返回: {result_text[:300]}{'...' if len(result_text) > 300 else ''}")
            print("-" * 60)
            return result_text

        tool = StructuredTool.from_function(
            coroutine=_call_tool,
            name=t.name,
            description=t.description or "",
            args_schema=DynamicSchema,
        )
        lc_tools.append(tool)

    # 全局绑定：LLM 知道有哪些工具及其 Schema
    llm_with_tools = llm.bind_tools(lc_tools)

    # 按名称索引，tool_node 按名查找工具对象
    tools_by_name = {t.name: t for t in lc_tools}

    return lc_tools


# ── 4. 定义节点 ────────────────────────────────────────────────────────────────

async def agent_node(state: AgentState) -> dict:
    """
    LLM 推理节点。直接使用全局 llm_with_tools，无需重新 bind_tools()。
    """
    last_human_msg = state["messages"][-1]
    print(f"\n🤖 Agent 正在思考: {last_human_msg.content!r}")

    response = await llm_with_tools.ainvoke(state["messages"])

    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"   → 决定调用工具: {tool_names}")
    else:
        print("   → 直接回答，无需调用工具")

    return {"messages": state["messages"] + [response]}


async def tool_node(state: AgentState) -> dict:
    """
    工具执行节点（LangChain 原生方式）。

    通过 tools_by_name 按名查找 StructuredTool，
    调用 tool.ainvoke()，LangChain 内部会执行闭包中的 session.call_tool()。
    不再需要手动从 state 取 session，也不需要解析 result.content。
    """
    last_msg = state["messages"][-1]
    tool_messages = []

    for tool_call in last_msg.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        tool = tools_by_name.get(name)
        if tool is None:
            result_text = f"❌ 未找到工具：{name}"
        else:
            # ← 核心：通过 LangChain 工具接口调用，不再直接操作 session
            result_text = await tool.ainvoke(args)

        tool_messages.append(
            ToolMessage(content=result_text, tool_call_id=tool_call["id"])
        )

    return {"messages": state["messages"] + tool_messages}


def should_continue(state: AgentState) -> str:
    """路由函数：LLM 有 tool_calls → tool_node；否则结束。"""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── 5. 构建 LangGraph ──────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── 6. 主入口 ──────────────────────────────────────────────────────────────────
async def main():
    SERVER_PATH = Path(__file__).parent / "mcp_server_template" / "server.py"

    print(f"🔍 查找 server.py: {SERVER_PATH}")
    if not SERVER_PATH.exists():
        print(f"❌ 找不到 server.py：{SERVER_PATH}")
        return
    print(f"✅ 找到 server.py")
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

    print("🚀 正在启动 MCP Server 并连接（stdio）...")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP Server 初始化成功！")

            # 工具初始化：Schema 拉取 + 闭包绑定 + llm_with_tools 全局绑定
            tools = await init_tools(session)
            print(f"✅ 已加载 {len(tools)} 个工具，LangChain 闭包绑定完成：")
            for t in tools:
                print(f"   - {t.name}: {t.description[:60]}...")

            print("\n" + "=" * 60)
            print("🚀 LangGraph Agent 启动，开始处理问题...\n")

            agent = build_graph()

            questions = [
                "把 42 和 58 相加，然后把结果乘以 3",
                "帮我分析这批数据的统计摘要：[{\"name\":\"Alice\",\"score\":90},{\"name\":\"Bob\",\"score\":75},{\"name\":\"Charlie\",\"score\":85}]",
                "用 fetch_url 工具测试一下，获取 https://www.toutiao.com/ 的内容",
            ]

            for q in questions:
                print(f"\n{'=' * 70}")
                print(f"📝 问题：{q}")
                print("=" * 70)

                # ↓ 移除了 mcp_session，state 更简洁
                init_state: AgentState = {
                    "messages": [HumanMessage(content=q)],
                }

                result = await agent.ainvoke(init_state)
                final = result["messages"][-1]
                print(f"\n🎯 最终答案：{final.content}")
                print("=" * 70)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())