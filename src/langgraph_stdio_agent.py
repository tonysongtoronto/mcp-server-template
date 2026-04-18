"""
src/langgraph_stdio_agent.py

两种运行方式：
  1. uv run langgraph dev   → langgraph 调用 lifespan 钩子初始化工具
  2. python -m src.langgraph_stdio_agent  → __main__ 手动初始化工具
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
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
# 2. MCP server 路径 & 启动参数
# ══════════════════════════════════════════════════════
SERVER_PATH = Path(__file__).parent / "mcp_server_template" / "server.py"

def mcp_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-u", str(SERVER_PATH)],
        env={"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **os.environ},
    )

# ══════════════════════════════════════════════════════
# 3. State
#    task_plan：任务清单（由 planner 一次性生成）
#    current_task_index：当前执行到哪个任务
#    next_agent：路由用
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
# 4. 共享工具容器
#    _tools 是一个列表引用，build_graph 的闭包持有它。
#    无论谁（lifespan 或 __main__）向它 extend，
#    graph 的节点函数下次执行时自动读到新工具。
# ══════════════════════════════════════════════════════
_tools: list[StructuredTool] = []

# ══════════════════════════════════════════════════════
# 5. 工具加载
# ══════════════════════════════════════════════════════
async def load_tools(session: ClientSession) -> list[StructuredTool]:
    lc_tools: list[StructuredTool] = []
    for t in (await session.list_tools()).tools:
        schema   = t.inputSchema or {}
        required = set(schema.get("required", []))
        fields   = {
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
# 6. Planner
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
# 7. Supervisor
#    不做 AI 决策，只做简单的任务指针推进
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
# 8. 通用 Agent 执行器
#    执行当前任务（而非整个用户问题），结果写回 task_plan
# ══════════════════════════════════════════════════════
async def run_agent(
    state: AgentState,
    name: str,
    system_prompt: str,
    allowed_tools: list[str],
    tools_ref: list[StructuredTool],
) -> AgentState:

    by_name   = {t.name: t for t in tools_ref}
    lc_tools  = [by_name[t] for t in allowed_tools if t in by_name]
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

    # 只把当前任务描述发给 Agent，而不是整个用户消息历史
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_description),
    ]

    print(f"\n  ▶ {name} 执行任务[{task_index}]：{task_description}（工具：{allowed_tools}）")

    # ReAct 循环
    max_steps    = 10
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
            tool = by_name.get(tc["name"])
            if tool:
                # 过滤 None 后直接调底层协程，跳过 StructuredTool 的 schema 重新验证
                args   = {k: v for k, v in tc["args"].items() if v is not None}
                result_text = await tool.coroutine(**args)
            else:
                result_text = f"❌ 未找到工具：{tc['name']}"
            msgs.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
    else:
        print(f"  ⚠️ {name} 达到最大步数 {max_steps}，强制终止")

    # 把结果写回任务清单，标记为 done
    result_summary = last_response.content if last_response else "（无结果）"
    current_task["status"] = "done"
    current_task["result"] = result_summary

    return {**state, "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 9. 图构建
#    闭包捕获 tools_ref，三个专业 Agent 节点通过它获取工具
# ══════════════════════════════════════════════════════
def build_graph(tools_ref: list[StructuredTool]):

    # ── 三个专业 Agent 节点 ──────────────────────────
    async def math_agent(state: AgentState) -> AgentState:
        return await run_agent(
            state, "Math Agent",
            system_prompt=(
                "你是数学专家。任务描述中已给出所有必要参数，直接调用对应工具完成计算，"
                "得到结果后立即返回，不要做多余的步骤。"
            ),
            allowed_tools=["add_numbers", "multiply_numbers"],
            tools_ref=tools_ref,
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
            tools_ref=tools_ref,
        )

    async def http_agent(state: AgentState) -> AgentState:
        return await run_agent(
            state, "HTTP Agent",
            system_prompt=(
                "你是网络请求专家。按任务描述发送请求，成功拿到响应后立即返回结果，"
                "不要重试或发送额外请求。若工具参数有默认值问题，timeout 请使用 10。"
            ),
            allowed_tools=["fetch_url", "post_json"],
            tools_ref=tools_ref,
        )

    # ── Final Answer 节点 ────────────────────────────
    async def final_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])

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

    # ── 路由函数 ─────────────────────────────────────
    def route(state: AgentState) -> str:
        next_node = state.get("next_agent", "FINISH")
        return "final_answer" if next_node == "FINISH" else next_node

    # ── 构建图 ───────────────────────────────────────
    g = StateGraph(AgentState)

    g.add_node("planner",      planner_node)
    g.add_node("supervisor",   supervisor_node)
    g.add_node("math_agent",   math_agent)
    g.add_node("data_agent",   data_agent)
    g.add_node("http_agent",   http_agent)
    g.add_node("final_answer", final_answer_node)

    g.set_entry_point("planner")
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

    return g.compile()


# langgraph dev 引用的就是这个对象
graph = build_graph(_tools)


# ══════════════════════════════════════════════════════
# 10. lifespan —— 仅 langgraph dev 调用
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
# 11. __main__ —— 单独运行测试
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    if not SERVER_PATH.exists():
        print(f"❌ 找不到 server.py：{SERVER_PATH}")
        sys.exit(1)

    QUESTIONS = [
        # 测试核心场景：同一个 Agent（math_agent）需要被调用两次
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
                    result = await graph.ainvoke({
                        "messages":           [HumanMessage(content=q)],
                        "task_plan":          [],   # 由 planner 填充
                        "current_task_index": 0,
                        "next_agent":         "",
                    })
                    print(f"\n✨ 最终答案：{result['messages'][-1].content}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())   # stdio_client 在这个循环里开和关，不会跨循环 ✅