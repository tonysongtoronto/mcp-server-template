# src/langgraph_agent.py
import asyncio
import os
import sys
from typing import TypedDict, Annotated
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── 1. 定义 State ──────────────────────────────────────────
class AgentState(TypedDict):
    messages: list          # 完整对话历史
    mcp_session: object     # MCP 连接（传递给节点用）

# ── 2. 创建 LLM（Claude 作为大脑）────────────────────────────
llm = ChatOpenAI(
    model="deepseek-chat",                    # DeepSeek 的主流模型
    api_key=os.getenv("DEEPSEEK_API_KEY"),    # ← 必须使用 DEEPSEEK_API_KEY
    base_url="https://api.deepseek.com",      # 推荐不加 /v1（官方文档支持两种）
    temperature=0.0,
)

# ── 3. 把 MCP 工具转成 LangChain 格式 ───────────────────────
async def get_langchain_tools(session: ClientSession):
    """从 MCP Session 获取所有工具，转成 LangChain 可用格式"""
    tools_result = await session.list_tools()
    lc_tools = []
    for t in tools_result.tools:
        # 动态创建一个异步函数作为工具
        async def make_tool_fn(tool_name, s):
            async def tool_fn(**kwargs):
                result = await s.call_tool(tool_name, kwargs)
                return result.content[0].text if result.content else "无结果"
            tool_fn.__name__ = tool_name
            tool_fn.__doc__ = t.description
            return tool_fn
        
        fn = await make_tool_fn(t.name, session)
        # 用 LangChain 的 StructuredTool 包装
        from langchain_core.tools import StructuredTool
        lc_tool = StructuredTool.from_function(
            coroutine=fn,
            name=t.name,
            description=t.description or "",
            args_schema=None,  # 先简单处理
        )
        lc_tools.append(lc_tool)
    return lc_tools

# ── 4. 定义节点 ──────────────────────────────────────────────
async def agent_node(state: AgentState):
    """Agent 思考节点：让 Claude 决定调用哪个工具"""
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
    graph.add_edge("tools", "agent")   # 工具完成 → 回到 agent 继续思考
    
    return graph.compile()

# ── 6. 运行 ─────────────────────────────────────────────────
async def main():
    MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")
    
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            agent = build_graph()
            
            questions = [
                "把 42 和 58 相加，然后把结果乘以 3",
                "帮我分析这批数据的统计摘要：[{\"name\":\"Alice\",\"score\":90},{\"name\":\"Bob\",\"score\":75},{\"name\":\"Charlie\",\"score\":85}]",
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