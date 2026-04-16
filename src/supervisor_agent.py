# src/supervisor_agent.py
"""
supervisor_agent.py — 修复版

参考 langgraph_agent.py 的核心改动：
1. 工具全局初始化（init_tools），闭包捕获 session，不再经 state 传递
2. AgentState 移除 mcp_session 字段
3. run_agent 复用全局 tools_by_name，不再每次重新 list_tools()
4. EventLoop 策略与 langgraph_agent.py 保持一致（ProactorEventLoop）
5. supervisor 的结构化输出改为 JSON 手动解析，避免 DeepSeek 兼容问题
"""

import asyncio
import json
import os
import sys
from typing import TypedDict, Literal, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel, create_model

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
# 2. 全局工具缓存（启动时由 init_tools 填充）
# ══════════════════════════════════════════════════════
tools_by_name: dict[str, StructuredTool] = {}   # 工具名 → StructuredTool


# ══════════════════════════════════════════════════════
# 3. State —— 移除 mcp_session，session 已闭包进工具
# ══════════════════════════════════════════════════════
class AgentState(TypedDict):
    messages: list        # 完整对话历史
    next_agent: str       # supervisor 决定的下一站
    task_results: list    # 各 agent 完成后的结果汇总


# ══════════════════════════════════════════════════════
# 4. 全局工具初始化（参考 langgraph_agent.py init_tools）
# ══════════════════════════════════════════════════════
async def init_tools(session: ClientSession) -> list[StructuredTool]:
    """
    从 MCP Server 拉取工具列表，为每个工具构建闭包式 StructuredTool。
    整个生命周期只调用一次，结果缓存在全局 tools_by_name。
    """
    global tools_by_name

    tools_result = await session.list_tools()
    lc_tools: list[StructuredTool] = []

    for t in tools_result.tools:
        schema = t.inputSchema or {}
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        field_definitions = {
            name: (Any, ...) if name in required_fields else (Optional[Any], None)
            for name in properties
        }
        DynamicSchema = (
            create_model(f"{t.name}_schema", **field_definitions)
            if field_definitions
            else None
        )

        # ── 闭包捕获 tool_name 和 session，避免循环变量陷阱 ──────────────
        tool_name = t.name

        async def _call_tool(_tool_name=tool_name, **kwargs) -> str:
            print(f"\n🔧 [MCP] 调用工具: {_tool_name}")
            print(f"   参数: {kwargs}")
            result = await session.call_tool(_tool_name, kwargs)
            result_text = result.content[0].text if result.content else "（无结果）"
            print(f"   ✅ 返回: {result_text[:200]}{'...' if len(result_text) > 200 else ''}")
            return result_text

        tool = StructuredTool.from_function(
            coroutine=_call_tool,
            name=t.name,
            description=t.description or "",
            args_schema=DynamicSchema,
        )
        lc_tools.append(tool)

    tools_by_name = {t.name: t for t in lc_tools}
    print(f"✅ 已加载 {len(lc_tools)} 个工具：{list(tools_by_name.keys())}")
    return lc_tools


# ══════════════════════════════════════════════════════
# 5. Supervisor 路由决策
#    改用 JSON 手动解析，避免 DeepSeek with_structured_output 兼容问题
# ══════════════════════════════════════════════════════
SUPERVISOR_SYSTEM = """你是任务调度器，把用户问题分配给最合适的专业 Agent。

可用 Agent：
- math_agent  ：数学计算，加法、乘法等数字运算
- data_agent  ：数据分析，统计摘要、分组聚合
- http_agent  ：网络请求，GET/POST URL
- FINISH      ：所有子任务已完成，可以汇总输出最终答案

已完成的任务结果：
{results}

根据用户问题和已有结果，决定下一步路由。
如果问题已被完整回答，选择 FINISH。

请严格只输出如下 JSON，不要有任何其他内容：
{{"next": "math_agent", "reason": "需要做数学计算"}}"""


async def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor：分析任务，输出 JSON 路由决策（手动解析，兼容 DeepSeek）"""
    results_text = "\n".join(state.get("task_results") or ["（暂无）"])

    response = await llm.ainvoke([
        SystemMessage(content=SUPERVISOR_SYSTEM.format(results=results_text)),
        *state["messages"],
    ])

    # 手动解析 JSON，比 with_structured_output 对 DeepSeek 更兼容
    raw = response.content.strip()
    # 去掉可能的 ```json ``` 包裹
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw.strip())
        next_agent = data.get("next", "FINISH")
        reason = data.get("reason", "")
    except json.JSONDecodeError:
        print(f"  ⚠️  JSON 解析失败，原始输出：{raw[:100]}")
        next_agent = "FINISH"
        reason = "解析失败，直接结束"

    # 校验合法值
    valid = {"math_agent", "data_agent", "http_agent", "FINISH"}
    if next_agent not in valid:
        next_agent = "FINISH"

    print(f"\n  🧭 Supervisor → {next_agent}（{reason}）")
    return {**state, "next_agent": next_agent}


# ══════════════════════════════════════════════════════
# 6. 通用 Agent 执行器
#    复用全局 tools_by_name，不再每次重新拉取工具列表
# ══════════════════════════════════════════════════════
async def run_agent(
    state: AgentState,
    name: str,
    system_prompt: str,
    allowed_tools: list[str],
) -> AgentState:
    # 从全局缓存中筛选本 Agent 允许使用的工具
    lc_tools = [
        tools_by_name[t]
        for t in allowed_tools
        if t in tools_by_name
    ]

    if not lc_tools:
        summary = f"[{name}] ⚠️ 没有可用工具：{allowed_tools}"
        return {**state, "task_results": (state.get("task_results") or []) + [summary]}

    agent_llm = llm.bind_tools(lc_tools)
    msgs = [SystemMessage(content=system_prompt)] + state["messages"]

    # 第一次推理：让 LLM 决定调用哪个工具
    response = await agent_llm.ainvoke(msgs)

    tool_msgs = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            print(f"  🔧 {name} 调用 {tc['name']}({tc['args']})")
            tool = tools_by_name.get(tc["name"])
            if tool:
                # ← 通过 LangChain 工具接口调用闭包（与 langgraph_agent.py 一致）
                result_text = await tool.ainvoke(tc["args"])
            else:
                result_text = f"❌ 未找到工具：{tc['name']}"
            tool_msgs.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))

        # 第二次推理：把工具结果告诉 LLM，生成自然语言总结
        final = await agent_llm.ainvoke(msgs + [response] + tool_msgs)
        summary = f"[{name}] {final.content}"
    else:
        summary = f"[{name}] {response.content}"

    new_results = (state.get("task_results") or []) + [summary]
    return {**state, "task_results": new_results}


# ══════════════════════════════════════════════════════
# 7. 三个专业 Agent 节点
# ══════════════════════════════════════════════════════
async def math_agent(state: AgentState) -> AgentState:
    return await run_agent(
        state, "Math Agent",
        system_prompt="你是数学专家，使用工具完成计算，给出步骤和结果。",
        allowed_tools=["add_numbers", "multiply_numbers"],
    )

async def data_agent(state: AgentState) -> AgentState:
    return await run_agent(
        state, "Data Agent",
        system_prompt="你是数据分析专家，使用工具对数据做统计分析，给出清晰结论。",
        allowed_tools=["dataframe_summary", "group_and_aggregate"],
    )

async def http_agent(state: AgentState) -> AgentState:
    return await run_agent(
        state, "HTTP Agent",
        system_prompt="你是网络请求专家，使用工具发送请求，整理并返回关键信息。",
        allowed_tools=["fetch_url", "post_json"],
    )


# ══════════════════════════════════════════════════════
# 8. Final Answer 节点
# ══════════════════════════════════════════════════════
async def final_answer_node(state: AgentState) -> AgentState:
    results_text = "\n".join(state.get("task_results") or ["（无结果）"])
    response = await llm.ainvoke([
        SystemMessage(content=f"根据以下执行结果，用中文给用户一个清晰完整的最终答案：\n\n{results_text}"),
        state["messages"][0],
    ])
    return {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}


# ══════════════════════════════════════════════════════
# 9. 路由函数 + 构建图
# ══════════════════════════════════════════════════════
def route(state: AgentState) -> str:
    next_node = state.get("next_agent", "FINISH")
    return "final_answer" if next_node == "FINISH" else next_node


def build_supervisor_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("supervisor",   supervisor_node)
    g.add_node("math_agent",   math_agent)
    g.add_node("data_agent",   data_agent)
    g.add_node("http_agent",   http_agent)
    g.add_node("final_answer", final_answer_node)

    g.set_entry_point("supervisor")

    g.add_conditional_edges("supervisor", route, {
        "math_agent":   "math_agent",
        "data_agent":   "data_agent",
        "http_agent":   "http_agent",
        "final_answer": "final_answer",
    })

    for agent in ["math_agent", "data_agent", "http_agent"]:
        g.add_edge(agent, "supervisor")

    g.add_edge("final_answer", END)

    return g.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════
# 10. 主入口
# ══════════════════════════════════════════════════════
async def main():
    MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

    questions = [
        "把 88 和 12 相加，再把结果乘以 5",
        '分析这批数据：[{"name":"Alice","dept":"Eng","salary":9000},{"name":"Bob","dept":"Mkt","salary":7500},{"name":"Charlie","dept":"Eng","salary":11000}]',
        "访问 https://api.github.com/zen 返回了什么？",
    ]

    print(f"🚀 连接 MCP Server：{MCP_URL}")

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP Session 初始化成功！")

            # ── 关键：全局初始化工具（闭包绑定 session），只执行一次 ──────
            await init_tools(session)

            print("\n" + "=" * 60)
            print("🚀 Supervisor Agent 启动，开始处理问题...\n")

            agent = build_supervisor_graph()

            for q in questions:
                print(f"\n{'━' * 60}")
                print(f"❓ {q}")
                print("━" * 60)

                result = await agent.ainvoke({
                    "messages":     [HumanMessage(content=q)],
                    "next_agent":   "",
                    "task_results": [],
                })

                print(f"\n✨ 最终答案：{result['messages'][-1].content}")

    print("\n🎉 所有问题处理完毕！")


if __name__ == "__main__":
    # 与 langgraph_agent.py 保持一致，Windows 使用 ProactorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())