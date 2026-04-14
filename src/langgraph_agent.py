# src/langgraph_agent.py
import asyncio
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

# ── 1. 定义 State ──────────────────────────────────────────
class AgentState(TypedDict):
    messages: list
    mcp_session: object

# ── 2. 创建 LLM ────────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0,
)

# ── 3. 把 MCP 工具转成 LangChain 格式 ───────────────────────
async def get_langchain_tools(session: ClientSession):
    tools_result = await session.list_tools()
    lc_tools = []

    for t in tools_result.tools:
        schema = t.inputSchema or {}
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        type_map = {
            "string": str, "integer": int, "number": float,
            "boolean": bool, "object": dict, "array": list,
        }

        field_definitions = {}
        for field_name, field_info in properties.items():
            json_type = field_info.get("type", "string")
            py_type = type_map.get(json_type, Any)
            if field_name not in required_fields:
                py_type = Optional[py_type]
            field_definitions[field_name] = (py_type, ...)

        DynamicSchema = create_model(f"{t.name}_schema", **field_definitions) if field_definitions else None

        def make_tool_fn(tool_name: str, tool_desc: str, s: ClientSession):
            async def tool_fn(**kwargs):
                print(f"\n🔧 【工具调用】 {tool_name}")
                print(f"   参数: {kwargs}")

                result = await s.call_tool(tool_name, kwargs)

                result_text = result.content[0].text if result.content else "无结果"
                print(f"   ✅ 返回: {result_text[:300]}{'...' if len(result_text) > 300 else ''}")
                print("-" * 60)

                return result_text

            tool_fn.__name__ = tool_name
            tool_fn.__doc__ = tool_desc
            return tool_fn

        fn = make_tool_fn(t.name, t.description or t.name, session)

        lc_tool = StructuredTool.from_function(
            coroutine=fn,
            name=t.name,
            description=t.description or "",
            args_schema=DynamicSchema,
        )
        lc_tools.append(lc_tool)

    return lc_tools


# ── 4. 定义节点 ──────────────────────────────────────────────
async def agent_node(state: AgentState):
    session = state["mcp_session"]
    tools = await get_langchain_tools(session)
    llm_with_tools = llm.bind_tools(tools)

    print(f"\n🤖 Agent 正在思考问题: {state['messages'][-1].content}")
    response = await llm_with_tools.ainvoke(state["messages"])
    
    if response.tool_calls:
        print(f"   → 决定调用 {len(response.tool_calls)} 个工具")
    else:
        print("   → 决定直接回答，无需调用工具")
    
    return {"messages": state["messages"] + [response]}


async def tool_node(state: AgentState):
    session = state["mcp_session"]
    last_msg = state["messages"][-1]
    tool_messages = []

    for tool_call in last_msg.tool_calls:
        result = await session.call_tool(tool_call["name"], tool_call["args"])
        result_text = result.content[0].text if result.content else "无结果"
        tool_messages.append(
            ToolMessage(content=result_text, tool_call_id=tool_call["id"])
        )

    return {"messages": state["messages"] + tool_messages}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── 5. 构建 LangGraph ────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── 6. 运行 ──────────────────────────────────────────────────
async def main():
    SERVER_PATH = Path(__file__).parent / "mcp_server_template" / "server.py"
    
    print(f"🔍 正在查找 server.py: {SERVER_PATH}")
    print(f"当前文件位置: {Path(__file__).resolve()}")
    
    if not SERVER_PATH.exists():
        print(f"❌ 找不到 server.py：{SERVER_PATH}")
        return
    
    print(f"✅ 找到 server.py: {SERVER_PATH}")
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

    print("🚀 正在通过 stdio 启动 MCP Server 并连接...")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP Server 初始化成功！")

            tools = await get_langchain_tools(session)
            print(f"✅ 已成功加载 {len(tools)} 个工具：")
            for t in tools:
                print(f"   - {t.name}")

            print("\n" + "=" * 60)
            print("🚀 LangGraph Agent 已启动，开始处理问题...\n")

            agent = build_graph()

            questions = [
                "把 42 和 58 相加，然后把结果乘以 3",
                "帮我分析这批数据的统计摘要：[{\"name\":\"Alice\",\"score\":90},{\"name\":\"Bob\",\"score\":75},{\"name\":\"Charlie\",\"score\":85}]",
                "用 fetch_url 工具测试一下，获取 https://www.toutiao.com/ 的内容"
            ]

            for q in questions:
                print(f"\n{'='*70}")
                print(f"📝 问题：{q}")
                print("="*70)

                init_state = {
                    "messages": [HumanMessage(content=q)],
                    "mcp_session": session,
                }

                result = await agent.ainvoke(init_state)
                final = result["messages"][-1]
                print(f"\n🎯 最终答案：{final.content}")
                print("="*70)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())