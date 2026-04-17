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
# 3. State
#    ★ 核心改动：新增 task_plan（任务清单）和 current_task_index（当前任务指针）
#       task_results 现在按任务记录，而非按 Agent 记录
# ══════════════════════════════════════════════════════
class Task(TypedDict):
    task_id: int          # 任务编号，从 0 开始
    description: str      # 任务描述，例如 "计算 3+5"
    agent: str            # 负责的 Agent，例如 "math_agent"
    status: str           # "pending" | "done"
    result: str           # 执行结果


class AgentState(TypedDict):
    messages: list
    task_plan: list[Task]         # 任务清单（由 planner 一次性生成）
    current_task_index: int       # 当前执行到哪个任务
    next_agent: str               # 路由用


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
# 5. Planner —— ★ 新增节点
#    作用：把用户问题一次性拆解为有序任务清单
#    只在图的最开始执行一次，之后 Supervisor 只负责派发
# ══════════════════════════════════════════════════════
PLANNER_SYSTEM = """你是任务规划器。把用户的问题拆解为若干有序的子任务，每个子任务分配给最合适的 Agent。

可用 Agent：
- math_agent  ：数学计算，加法、乘法等数字运算
- data_agent  ：数据分析，统计摘要、分组聚合
- http_agent  ：网络请求，GET/POST URL

任务描述规则（非常重要）：
- 描述必须完整、具体，Agent 不需要做任何额外推断
- math_agent：写明具体数字和运算符，例如"计算 88 + 12"，不要写"计算加法"
- data_agent：必须明确写出 records_json（完整 JSON 字符串）、group_by 列名、agg_col 列名、agg_func（只写用户要求的，若用户未指定则只写 sum）
  ✅ 正确示例："对以下数据按 dept 分组，对 salary 列做 sum 聚合。数据：[{...}]"
  ❌ 错误示例："对数据做统计分析（如总和、平均值等）" ← 绝对禁止写"如…等"模糊描述
- http_agent：写明完整 URL 和请求方法
- 同一个 Agent 可以出现多次（例如有两个独立的计算任务）
- 任务按执行顺序排列

严格只输出如下 JSON 数组，不要有任何其他内容：
[
  {"task_id": 0, "description": "计算 3+5", "agent": "math_agent", "status": "pending", "result": ""},
  {"task_id": 1, "description": "用 GET 方法访问 https://api.github.com/zen，返回响应文本", "agent": "http_agent", "status": "pending", "result": ""},
  {"task_id": 2, "description": "计算 10×20", "agent": "math_agent", "status": "pending", "result": ""}
]"""


async def planner_node(state: AgentState) -> AgentState:
    """一次性把用户问题拆解为任务清单，只执行一次。"""

    # 如果已经有任务清单，跳过（避免重入）
    if state.get("task_plan"):
        return state

    print("\n  📋 Planner 开始拆解任务...")

    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_SYSTEM),
        state["messages"][0],
    ])

    raw = response.content.strip()
    # 清理 markdown 代码块
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        task_plan: list[Task] = json.loads(raw.strip())
        print(f"  ✅ 拆解出 {len(task_plan)} 个任务：")
        for t in task_plan:
            print(f"     [{t['task_id']}] {t['agent']} ← {t['description']}")
    except json.JSONDecodeError:
        print(f"  ⚠️ JSON 解析失败：{raw[:200]}")
        # 兜底：把整个问题作为一个任务
        task_plan = [{"task_id": 0, "description": state["messages"][0].content,
                      "agent": "math_agent", "status": "pending", "result": ""}]

    return {**state, "task_plan": task_plan, "current_task_index": 0}


# ══════════════════════════════════════════════════════
# 6. Supervisor —— ★ 大幅简化
#    不再做 AI 决策，只做简单的任务指针推进
#    找到下一个 pending 任务 → 派发给对应 Agent
# ══════════════════════════════════════════════════════
async def supervisor_node(state: AgentState) -> AgentState:
    task_plan: list[Task] = state.get("task_plan", [])

    # 找下一个 pending 任务
    for task in task_plan:
        if task["status"] == "pending":
            print(f"\n  🧭 Supervisor → {task['agent']}（任务[{task['task_id']}]: {task['description']}）")
            return {**state, "next_agent": task["agent"], "current_task_index": task["task_id"]}

    # 所有任务都完成了
    print("\n  🧭 Supervisor → FINISH（所有任务已完成）")
    return {**state, "next_agent": "FINISH"}


# ══════════════════════════════════════════════════════
# 7. 通用 Agent 执行器 —— ★ 改为执行当前任务，而非整个用户问题
# ══════════════════════════════════════════════════════
async def run_agent(
    state: AgentState,
    name: str,
    system_prompt: str,
    allowed_tools: list[str],
) -> AgentState:

    lc_tools = [tools_by_name[t] for t in allowed_tools if t in tools_by_name]
    task_plan: list[Task] = state.get("task_plan", [])
    task_index: int = state.get("current_task_index", 0)

    # 取出当前任务
    current_task = next((t for t in task_plan if t["task_id"] == task_index), None)
    if not current_task:
        print(f"  ⚠️ [{name}] 找不到任务 {task_index}")
        return state

    task_description = current_task["description"]

    if not lc_tools:
        result_text = f"⚠️ 没有可用工具：{allowed_tools}"
        current_task["status"] = "done"
        current_task["result"] = result_text
        return {**state, "task_plan": task_plan}

    agent_llm = llm.bind_tools(lc_tools)

    # ★ 关键：只把当前任务描述发给 Agent，而不是整个用户消息历史
    #   这样 math_agent 第二次调用时，收到的是"计算 10×20"而不是全部问题
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_description),
    ]

    print(f"\n  ▶ {name} 执行任务[{task_index}]：{task_description}（工具：{allowed_tools}）")

    # ReAct 循环
    max_steps = 10
    last_response = None

    for step in range(max_steps):
        response = await agent_llm.ainvoke(msgs)
        last_response = response
        msgs.append(response)

        if not (hasattr(response, "tool_calls") and response.tool_calls):
            print(f"  ✅ {name} 任务[{task_index}] 完成（{step + 1} 轮推理）")
            break

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

    # ★ 把结果写回任务清单，标记为 done
    result_summary = last_response.content if last_response else "（无结果）"
    current_task["status"] = "done"
    current_task["result"] = result_summary

    return {**state, "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 8. 三个专业 Agent 节点（不变）
# ══════════════════════════════════════════════════════
async def math_agent(state: AgentState) -> AgentState:
    return await run_agent(
        state, "Math Agent",
        system_prompt=(
            "你是数学专家。任务描述中已给出所有必要参数，直接调用对应工具完成计算，"
            "得到结果后立即返回，不要做多余的步骤。"
        ),
        allowed_tools=["add_numbers", "multiply_numbers"],
    )

async def data_agent(state: AgentState) -> AgentState:
    return await run_agent(
        state, "Data Agent",
        system_prompt=(
            "你是数据分析专家。严格按照任务描述执行，任务要求几个工具调用就调用几次，"
            "不要自行扩展（禁止额外调用任务未要求的聚合函数）。"
            "任务描述中已给出 records_json、group_by、agg_col、agg_func，直接使用，不要猜测或补充。"
            "工具调用完成后，给出简洁结论即可。"
        ),
        allowed_tools=["dataframe_summary", "group_and_aggregate"],
    )

async def http_agent(state: AgentState) -> AgentState:
    return await run_agent(
        state, "HTTP Agent",
        system_prompt=(
            "你是网络请求专家。按任务描述发送请求，成功拿到响应后立即返回结果，"
            "不要重试或发送额外请求。若工具参数有默认值问题，timeout 请使用 10。"
        ),
        allowed_tools=["fetch_url", "post_json"],
    )


# ══════════════════════════════════════════════════════
# 9. Final Answer 节点 —— ★ 从 task_plan 里汇总所有结果
# ══════════════════════════════════════════════════════
async def final_answer_node(state: AgentState) -> AgentState:
    task_plan: list[Task] = state.get("task_plan", [])

    # 按任务顺序整理结果
    results_text = "\n".join(
        f"任务[{t['task_id']}] {t['description']}：{t['result']}"
        for t in task_plan
    )

    print(f"\n  📝 汇总结果：\n{results_text}")

    response = await llm.ainvoke([
        SystemMessage(content=f"根据以下各子任务的执行结果，用中文给用户一个清晰完整的最终答案：\n\n{results_text}"),
        state["messages"][0],
    ])

    return {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}


# ══════════════════════════════════════════════════════
# 10. 路由函数 + 构建图 —— ★ 新增 planner 节点
# ══════════════════════════════════════════════════════
def route(state: AgentState) -> str:
    next_node = state.get("next_agent", "FINISH")
    return "final_answer" if next_node == "FINISH" else next_node


def build_supervisor_graph(checkpointer=None):
    g = StateGraph(AgentState)

    # ★ 新增 planner 节点
    g.add_node("planner",      planner_node)
    g.add_node("supervisor",   supervisor_node)
    g.add_node("math_agent",   math_agent)
    g.add_node("data_agent",   data_agent)
    g.add_node("http_agent",   http_agent)
    g.add_node("final_answer", final_answer_node)

    # ★ 入口改为 planner
    g.set_entry_point("planner")

    # planner → supervisor（只走一次）
    g.add_edge("planner", "supervisor")

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
# 11. 主入口
# ══════════════════════════════════════════════════════
async def main():
    MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

    questions = [
        # ★ 测试核心场景：同一个 Agent（math_agent）需要被调用两次
        "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",

        # 原来的三个测试用例保留
        "把 88 和 12 相加，再把结果乘以 5",

        """分析这批数据：[{"name":"Alice","dept":"Eng","salary":9000},
         {"name":"Bob","dept":"Mkt","salary":7500},
         {"name":"Charlie","dept":"Eng","salary":11000}] ，
         分组列名 "department"，agg_col 聚合列名 "salary" """,

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
                    "messages":           [HumanMessage(content=q)],
                    "task_plan":          [],   # 由 planner 填充
                    "current_task_index": 0,
                    "next_agent":         "",
                })

                print(f"\n✨ 最终答案：{result['messages'][-1].content}")

    print("\n🎉 所有问题处理完毕！")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())