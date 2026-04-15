"""
langgraph_agent.py — 优化版 v2

主要改动（相对于 v1）：
1. llm_with_tools 提升为全局变量，整个生命周期只 bind_tools() 一次
2. init_tools() 负责拉取 Schema 并完成全局绑定，替换原 get_tool_schemas()
3. agent_node 直接使用全局 llm_with_tools，无需每轮重复 bind_tools()
4. AgentState 移除 tools 字段（已无需通过 state 传递）
5. 其余逻辑保持不变
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
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import create_model
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


# ── 1. 定义 State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: list       # 对话历史（HumanMessage / AIMessage / ToolMessage）
    mcp_session: object  # MCP ClientSession，tool_node 用它直接调用工具
    # ↑ 移除了 tools 字段：Schema 已缓存在全局 llm_with_tools，无需再经 state 传递


# ── 2. 创建 LLM + 全局工具绑定占位 ────────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0,
)

# 全局缓存：启动时由 init_tools() 赋值，之后所有节点直接引用
llm_with_tools = None


# ── 3. 初始化全局工具（启动时调用一次）────────────────────────────────────────
async def init_tools(session: ClientSession) -> list[StructuredTool]:
    """
    从 MCP Server 拉取工具列表，构建 StructuredTool Schema，
    并将 llm.bind_tools() 结果写入全局 llm_with_tools。

    整个程序生命周期只执行一次，后续所有 agent_node 调用均复用。

    注意：StructuredTool 此处不含真正执行函数——
    实际工具调用由 tool_node 通过 session.call_tool() 完成。
    """
    global llm_with_tools

    tools_result = await session.list_tools()
    lc_tools = []

    for t in tools_result.tools:
        schema = t.inputSchema or {}
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        # 根据 inputSchema 动态生成 Pydantic 模型，供 bind_tools 导出 Schema
        field_definitions = {
            name: (Any, ...) if name in required_fields else (Optional[Any], None)
            for name in properties
        }
        DynamicSchema = (
            create_model(f"{t.name}_schema", **field_definitions)
            if field_definitions
            else None
        )

        # placeholder coroutine：bind_tools 只需要 Schema，此函数体不会被 graph 调用
        async def _placeholder(**kwargs):
            pass

        lc_tools.append(
            StructuredTool.from_function(
                coroutine=_placeholder,
                name=t.name,
                description=t.description or "",
                args_schema=DynamicSchema,
            )
        )

    # 核心：bind 一次，全局复用，避免每轮 agent_node 重复绑定
    llm_with_tools = llm.bind_tools(lc_tools)
    return lc_tools


# ── 4. 定义节点 ────────────────────────────────────────────────────────────────

async def agent_node(state: AgentState) -> dict:
    """
    LLM 推理节点。
    - 直接使用全局 llm_with_tools，无需从 state 读取或重新 bind_tools()
    - 返回 LLM 响应（可能包含 tool_calls，也可能是最终文本）
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
    工具执行节点。
    - 直接通过 MCP session 调用工具，不经过 StructuredTool 的函数体
    - 将每个工具结果包装成 ToolMessage，追加到消息历史
    """
    session = state["mcp_session"]
    last_msg = state["messages"][-1]
    tool_messages = []

    for tool_call in last_msg.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        print(f"\n🔧 调用工具: {name}")
        print(f"   参数: {args}")

        result = await session.call_tool(name, args)
        result_text = result.content[0].text if result.content else "（无结果）"

        print(f"   ✅ 返回: {result_text[:300]}{'...' if len(result_text) > 300 else ''}")
        print("-" * 60)

        tool_messages.append(
            ToolMessage(content=result_text, tool_call_id=tool_call["id"])
        )

    return {"messages": state["messages"] + tool_messages}


def should_continue(state: AgentState) -> str:
    """路由函数：LLM 有 tool_calls → 去 tool_node；否则结束。"""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── 5. 构建 LangGraph ──────────────────────────────────────────────────────────
def build_graph() -> any:
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

            # 工具 Schema 获取 + llm_with_tools 全局绑定，只执行一次
            tools = await init_tools(session)
            print(f"✅ 已加载 {len(tools)} 个工具，llm_with_tools 全局绑定完成：")
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

                # ↓ 移除了 tools 字段，state 更简洁
                init_state: AgentState = {
                    "messages": [HumanMessage(content=q)],
                    "mcp_session": session,
                }

                result = await agent.ainvoke(init_state)
                final = result["messages"][-1]
                print(f"\n🎯 最终答案：{final.content}")
                print("=" * 70)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())