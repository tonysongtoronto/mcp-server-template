# src/supervisor_agent.py
import asyncio
import json
import os
import sys
from typing import TypedDict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from mcp import ClientSession
from mcp.client.sse import sse_client
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
# 2. 全局工具缓存
# ══════════════════════════════════════════════════════
tools_by_name: dict[str, StructuredTool] = {}


# ══════════════════════════════════════════════════════
# 3. State（已移除 mcp_session）
# ══════════════════════════════════════════════════════
class AgentState(TypedDict):
    messages: list
    next_agent: str
    task_results: list


# ══════════════════════════════════════════════════════
# 4. 全局工具初始化
# ══════════════════════════════════════════════════════
async def init_tools(session: ClientSession) -> list[StructuredTool]:
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

        tool_name = t.name

        async def _call_tool(_tool_name=tool_name, **kwargs) -> str:
            print(f"    🔧 [MCP] {_tool_name}({kwargs})")
            result = await session.call_tool(_tool_name, kwargs)
            result_text = result.content[0].text if result.content else "（无结果）"
            print(f"    ✅ 返回: {result_text[:200]}")
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
# 5. Supervisor（JSON 手动解析，兼容 DeepSeek）
# ══════════════════════════════════════════════════════
SUPERVISOR_SYSTEM = """你是任务调度器，把用户问题分配给最合适的专业 Agent。

可用 Agent：
- math_agent  ：数学计算，加法、乘法等数字运算
- data_agent  ：数据分析，统计摘要、分组聚合
- http_agent  ：网络请求，GET/POST URL
- FINISH      ：所有子任务已完成，可以汇总输出最终答案

已完成的任务结果：
{results}

重要规则：
- 每个 Agent 会在单次调用中完成所有相关工具调用（包括多步计算）
- 已出现在"已完成的任务结果"中的 Agent，不要再重复调用
- 如果结果已经包含问题所需的所有答案，选择 FINISH

请严格只输出如下 JSON，不要有任何其他内容：
{{"next": "math_agent", "reason": "原因"}}"""


async def supervisor_node(state: AgentState) -> AgentState:
    results_text = "\n".join(state.get("task_results") or ["（暂无）"])

    response = await llm.ainvoke([
        SystemMessage(content=SUPERVISOR_SYSTEM.format(results=results_text)),
        *state["messages"],
    ])

    raw = response.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw.strip())
        next_agent = data.get("next", "FINISH")
        reason = data.get("reason", "")
    except json.JSONDecodeError:
        print(f"  ⚠️  JSON 解析失败：{raw[:100]}")
        next_agent = "FINISH"
        reason = "解析失败"

    valid = {"math_agent", "data_agent", "http_agent", "FINISH"}
    if next_agent not in valid:
        next_agent = "FINISH"

    print(f"\n  🧭 Supervisor → {next_agent}（{reason}）")
    return {**state, "next_agent": next_agent}


# ══════════════════════════════════════════════════════
# 6. 通用 Agent 执行器
#    ★ 核心修复：ReAct 循环，持续调用工具直到 LLM 不再发出 tool_calls
# ══════════════════════════════════════════════════════
async def run_agent(
    state: AgentState,
    name: str,
    system_prompt: str,
    allowed_tools: list[str],
) -> AgentState:

    lc_tools = [tools_by_name[t] for t in allowed_tools if t in tools_by_name]

    if not lc_tools:
        summary = f"[{name}] ⚠️ 没有可用工具：{allowed_tools}"
        return {**state, "task_results": (state.get("task_results") or []) + [summary]}

    agent_llm = llm.bind_tools(lc_tools)
    msgs = [SystemMessage(content=system_prompt)] + state["messages"]

    print(f"\n  ▶ {name} 开始执行（工具：{allowed_tools}）")

    # ── ★ ReAct 循环：持续推理 + 工具调用，直到 LLM 返回纯文本答案 ──────
    max_steps = 10
    last_response = None

    for step in range(max_steps):
        response = await agent_llm.ainvoke(msgs)
        last_response = response
        msgs.append(response)

        # 无 tool_calls → LLM 给出最终答案，跳出循环
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            print(f"  ✅ {name} 完成（{step + 1} 轮推理）")
            break

        # 有 tool_calls → 逐一执行，结果追加进消息历史
        print(f"  📋 {name} 第 {step + 1} 轮，调用工具：")
        for tc in response.tool_calls:
            tool = tools_by_name.get(tc["name"])
            if tool:
                result_text = await tool.ainvoke(tc["args"])
            else:
                result_text = f"❌ 未找到工具：{tc['name']}"
            msgs.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
    else:
        print(f"  ⚠️ {name} 达到最大步数 {max_steps}，强制终止")

    summary = f"[{name}] {last_response.content}"
    new_results = (state.get("task_results") or []) + [summary]
    return {**state, "task_results": new_results}


# ══════════════════════════════════════════════════════
# 7. 三个专业 Agent 节点
# ══════════════════════════════════════════════════════
async def math_agent(state: AgentState) -> AgentState:
    return await run_agent(
        state, "Math Agent",
        system_prompt="你是数学专家，使用工具完成所有计算步骤，每步调用对应工具，最后给出完整结果。",
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
        
         """分析这批数据：[{"name":"Alice","dept":"Eng","salary":9000},
         {"name":"Bob","dept":"Mkt","salary":7500},
         {"name":"Charlie","dept":"Eng","salary":11000}] ，
         分组列名， "department"  agg_col  - 聚合列名 "salary" """,
         
        "访问 https://api.github.com/zen 返回了什么？",
    ]

    print(f"🚀 连接 MCP Server：{MCP_URL}")

    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP Session 初始化成功！")

            await init_tools(session)

            print("\n" + "=" * 60)
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
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())