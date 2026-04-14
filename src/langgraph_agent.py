# src/langgraph_agent.py
import asyncio
import os
import sys
from typing import TypedDict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import create_model
from dotenv import load_dotenv

load_dotenv()

# ── 1. 定义 State ──────────────────────────────────────────
class AgentState(TypedDict):
    messages: list      # 完整对话历史
    mcp_session: object # MCP 连接（传递给节点用）

# ── 2. 创建 LLM ────────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0,
)

# ── 3. 把 MCP 工具转成 LangChain 格式 ───────────────────────
async def get_langchain_tools(session: ClientSession):
    """从 MCP Session 获取所有工具，转成 LangChain 可用格式"""
    tools_result = await session.list_tools()
    lc_tools = []

    for t in tools_result.tools:
        # ✅ 修复1：从 MCP inputSchema 动态构建 Pydantic 模型
        schema = t.inputSchema or {}
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        # JSON Schema type → Python type 映射
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }

        field_definitions = {}
        for field_name, field_info in properties.items():
            json_type = field_info.get("type", "string")
            py_type = type_map.get(json_type, Any)
            if field_name not in required_fields:
                py_type = Optional[py_type]
            field_definitions[field_name] = (py_type, ...)

        DynamicSchema = (
            create_model(f"{t.name}_schema", **field_definitions)
            if field_definitions
            else None
        )

        # ✅ 修复2：用普通 def 做外层闭包，避免循环变量被覆盖
        def make_tool_fn(tool_name: str, tool_desc: str, s: ClientSession):
            async def tool_fn(**kwargs):
                result = await s.call_tool(tool_name, kwargs)
                return result.content[0].text if result.content else "无结果"
            tool_fn.__name__ = tool_name
            tool_fn.__doc__ = tool_desc
            return tool_fn

        fn = make_tool_fn(t.name, t.description or t.name, session)

        lc_tool = StructuredTool.from_function(
            coroutine=fn,
            name=t.name,
            description=t.description or "",
            args_schema=DynamicSchema,  # ✅ 传入完整参数 schema，LLM 才会触发 tool_calls
        )
        lc_tools.append(lc_tool)

    return lc_tools

# ── 4. 定义节点 ──────────────────────────────────────────────
async def agent_node(state: AgentState):
    """Agent 思考节点：让 LLM 决定调用哪个工具"""
    session = state["mcp_session"]
    tools = await get_langchain_tools(session)
    llm_with_tools = llm.bind_tools(tools)

    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": state["messages"] + [response]}

async def tool_node(state: AgentState):
    """工具执行节点：调用 MCP 工具，获取结果"""
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
    """路由函数：有 tool_calls 就继续，否则结束"""
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
    graph.add_edge("tools", "agent")  # 工具完成 → 回到 agent 继续思考

    return graph.compile()

# ── 6. 运行 ──────────────────────────────────────────────────
async def main():
    MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 调试：打印已注册工具及其 schema，确认工具正确加载
            tools = await get_langchain_tools(session)
            print("已加载工具：")
            for t in tools:
                print(f"  - {t.name}: args_schema={t.args_schema}")

            agent = build_graph()

            questions = [
                "把 42 和 58 相加，然后把结果乘以 3",
                "帮我分析这批数据的统计摘要：[{\"name\":\"Alice\",\"score\":90},{\"name\":\"Bob\",\"score\":75},{\"name\":\"Charlie\",\"score\":85}]",
                "用 fetch_url 工具测试一下，获取 https://www.toutiao.com/ 的内容"
            ]

            for q in questions:
                print(f"\n{'='*50}")
                print(f"问题：{q}")
                print("="*50)

                init_state = {
                    "messages": [HumanMessage(content=q)],
                    "mcp_session": session,
                }

                result = await agent.ainvoke(init_state)
                final = result["messages"][-1]
                print(f"答案：{final.content}")

if __name__ == "__main__":
    asyncio.run(main())