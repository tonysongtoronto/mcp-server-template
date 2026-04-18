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
#
# Task.inputs 声明运行时入参来源，格式：
#   {
#     "参数语义名": {"from_task": <task_id>, "field": "result"}
#   }
# Supervisor 在派发前把依赖任务的结果按参数名注入到 _resolved_description，
# Agent 收到"意图 + 【运行时参数】"后，自己推理如何调工具。
# ══════════════════════════════════════════════════════
class TaskInput(TypedDict):
    from_task: int   # 依赖哪个任务的结果
    field: str       # 取哪个字段（目前固定为 "result"）

class Task(TypedDict):
    task_id: int
    description: str                  # 意图描述，不含具体数值
    agent: str                        # math_agent / data_agent / http_agent
    inputs: dict[str, TaskInput]      # 参数名 → 来源，无依赖时为 {}
    depends_on: list[int]             # 依赖的 task_id 列表
    status: str                       # "pending" | "done"
    result: str                       # 执行结果
    _resolved_description: str        # Supervisor 注入参数后的描述（运行时填充）

class AgentState(TypedDict):
    messages: list
    task_plan: list[Task]
    current_task_index: int
    next_agent: str


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
#    把用户问题拆解为 DAG 任务列表。
#    每个任务只描述意图，通过 inputs 声明依赖参数来源，
#    不提前计算任何数值，也不感知具体工具名称。
# ══════════════════════════════════════════════════════
PLANNER_SYSTEM = """你是任务规划器。把用户问题拆解为有序子任务，像函数调用一样声明每个任务的入参来源。

可用 Agent：
- math_agent  ：数学计算
- data_agent  ：数据分析、统计聚合
- http_agent  ：网络请求

规则（严格遵守）：
1. description 只写意图，绝对不要提前计算任何数值或结果
2. inputs 声明运行时需要从哪些前置任务获取参数：
   - key   = 参数的语义名称（如"加数A"、"被乘数"、"数据集"），供 Agent 理解用途
   - value = {"from_task": <被依赖的task_id>, "field": "result"}
3. depends_on 从 inputs 的 from_task 自动推导，列出所有依赖的 task_id
4. 没有依赖的任务：inputs 为 {}，depends_on 为 []
5. 同一个 Agent 可出现多次
6. 任务按拓扑顺序排列（被依赖的任务排在前面）

严格只输出 JSON 数组，不要有任何其他内容，示例：
[
  {
    "task_id": 0,
    "description": "计算 88 加 12",
    "agent": "math_agent",
    "inputs": {},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  },
  {
    "task_id": 1,
    "description": "把前一步的结果乘以 5",
    "agent": "math_agent",
    "inputs": {
      "被乘数": {"from_task": 0, "field": "result"}
    },
    "depends_on": [0],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }
]"""


async def planner_node(state: AgentState) -> AgentState:
    """一次性把用户问题拆解为 DAG 任务清单，只执行一次。"""

    if state.get("task_plan"):
        return state

    print("\n  📋 Planner 开始拆解任务...")

    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_SYSTEM),
        state["messages"][0],
    ])

    raw = response.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        task_plan: list[Task] = json.loads(raw.strip())
        print(f"  ✅ 拆解出 {len(task_plan)} 个任务：")
        for t in task_plan:
            deps    = t.get("depends_on", [])
            inputs  = t.get("inputs", {})
            dep_str = f"  依赖→{deps} 参数→{list(inputs.keys())}" if deps else ""
            print(f"     [{t['task_id']}] {t['agent']} ← {t['description']}{dep_str}")
    except json.JSONDecodeError:
        print(f"  ⚠️ JSON 解析失败：{raw[:200]}")
        task_plan = [{
            "task_id": 0,
            "description": state["messages"][0].content,
            "agent": "math_agent",
            "inputs": {},
            "depends_on": [],
            "status": "pending",
            "result": "",
            "_resolved_description": "",
        }]

    return {**state, "task_plan": task_plan, "current_task_index": 0}


# ══════════════════════════════════════════════════════
# 7. Supervisor
#    检查依赖是否就绪 → 解析 inputs → 注入参数到 _resolved_description → 派发
#    像函数调用传参：参数名=来源任务结果，注入后 Agent 自己推理工具调用
# ══════════════════════════════════════════════════════
async def supervisor_node(state: AgentState) -> AgentState:
    task_plan: list[Task] = state.get("task_plan", [])
    done_map = {t["task_id"]: t for t in task_plan if t["status"] == "done"}

    for task in task_plan:
        if task["status"] != "pending":
            continue

        # 检查依赖是否全部完成
        unmet = [dep for dep in task.get("depends_on", []) if dep not in done_map]
        if unmet:
            print(f"\n  ⏳ 任务[{task['task_id']}] 等待依赖 {unmet}，跳过寻找其他可执行任务")
            continue

        # ★ 像函数传参：把 inputs 解析为具名参数，注入到任务描述
        resolved_inputs: dict[str, str] = {}
        for param_name, source in task.get("inputs", {}).items():
            from_id  = source["from_task"]
            field    = source.get("field", "result")
            src_task = done_map.get(from_id, {})
            resolved_inputs[param_name] = src_task.get(field, "")

        if resolved_inputs:
            params_text = "\n".join(f"  {k} = {v}" for k, v in resolved_inputs.items())
            task["_resolved_description"] = (
                f"{task['description']}\n\n"
                f"【运行时参数】\n{params_text}"
            )
        else:
            task["_resolved_description"] = task["description"]

        print(f"\n  🧭 Supervisor → {task['agent']}（任务[{task['task_id']}]）")
        print(f"     {task['_resolved_description'].replace(chr(10), ' | ')}")
        return {**state, "next_agent": task["agent"], "current_task_index": task["task_id"]}

    # 还有 pending 但所有剩余任务依赖都无法满足 → 死锁，强制结束
    pending = [t for t in task_plan if t["status"] == "pending"]
    if pending:
        print(f"\n  ⚠️ 任务 {[t['task_id'] for t in pending]} 依赖无法满足，强制跳过")
        for t in pending:
            t["status"] = "done"
            t["result"] = "⚠️ 依赖未满足，跳过"

    print("\n  🧭 Supervisor → FINISH（所有任务已完成）")
    return {**state, "next_agent": "FINISH"}


# ══════════════════════════════════════════════════════
# 8. 通用 Agent 执行器
#    读取 _resolved_description（已注入参数的描述）执行任务
#    Agent 通过 LLM 推理决定调哪个工具、传什么参数
# ══════════════════════════════════════════════════════
async def run_agent(
    state: AgentState,
    name: str,
    system_prompt: str,
    allowed_tools: list[str],
    tools_ref: list[StructuredTool],
) -> AgentState:

    by_name  = {t.name: t for t in tools_ref}
    lc_tools = [by_name[t] for t in allowed_tools if t in by_name]
    task_plan: list[Task] = state.get("task_plan", [])
    task_index: int = state.get("current_task_index", 0)

    current_task = next((t for t in task_plan if t["task_id"] == task_index), None)
    if not current_task:
        print(f"  ⚠️ [{name}] 找不到任务 {task_index}")
        return state

    # ★ 优先用 Supervisor 注入参数后的描述，无则回退到原始描述
    task_description = current_task.get("_resolved_description") or current_task["description"]

    if not lc_tools:
        current_task["status"] = "done"
        current_task["result"] = f"⚠️ 没有可用工具：{allowed_tools}"
        return {**state, "task_plan": task_plan}

    agent_llm = llm.bind_tools(lc_tools)

    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_description),
    ]

    print(f"\n  ▶ {name} 执行任务[{task_index}]（工具：{allowed_tools}）")

    max_steps     = 10
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
                args        = {k: v for k, v in tc["args"].items() if v is not None}
                result_text = await tool.coroutine(**args)
            else:
                result_text = f"❌ 未找到工具：{tc['name']}"
            msgs.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
    else:
        print(f"  ⚠️ {name} 达到最大步数 {max_steps}，强制终止")

    current_task["status"] = "done"
    current_task["result"] = last_response.content if last_response else "（无结果）"

    return {**state, "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 9. 图构建
#    Agent 的 system_prompt 只描述角色和行为准则，
#    不感知具体工具名，工具由 allowed_tools 白名单控制。
#    将来新增工具只需：① 在 load_tools 里加载，② 在对应 Agent 的 allowed_tools 里注册。
# ══════════════════════════════════════════════════════
def build_graph(tools_ref: list[StructuredTool]):

    async def math_agent(state: AgentState) -> AgentState:
        return await run_agent(
            state, "Math Agent",
            system_prompt=(
                "你是数学计算专家。根据任务描述和【运行时参数】（如果有），"
                "推断需要做什么运算，调用合适的工具完成计算，得到结果后立即返回。"
                "只输出最终数值结果，不要解释过程。"
            ),
            allowed_tools=["add_numbers", "multiply_numbers"],
            tools_ref=tools_ref,
        )

    async def data_agent(state: AgentState) -> AgentState:
        return await run_agent(
            state, "Data Agent",
            system_prompt=(
                "你是数据分析专家。根据任务描述和【运行时参数】（如果有），"
                "提取数据集、分组列、聚合列、聚合函数等信息，调用合适的工具完成分析。"
                "任务要求几种聚合就做几种，不要自行扩展。完成后给出简洁结论。"
            ),
            allowed_tools=["dataframe_summary", "group_and_aggregate"],
            tools_ref=tools_ref,
        )

    async def http_agent(state: AgentState) -> AgentState:
        return await run_agent(
            state, "HTTP Agent",
            system_prompt=(
                "你是网络请求专家。根据任务描述和【运行时参数】（如果有），"
                "发送对应的 HTTP 请求，成功拿到响应后立即返回结果，不要重试。"
                "timeout 默认使用 10。"
            ),
            allowed_tools=["fetch_url", "post_json"],
            tools_ref=tools_ref,
        )

    async def final_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])

        results_text = "\n".join(
            f"任务[{t['task_id']}]（{t['description']}）：{t['result']}"
            for t in task_plan
        )
        print(f"\n  📝 汇总结果：\n{results_text}")

        response = await llm.ainvoke([
            SystemMessage(content=(
                "根据以下各子任务的执行结果，用中文给用户一个清晰完整的最终答案。\n\n"
                f"{results_text}"
            )),
            state["messages"][0],
        ])

        return {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}

    def route(state: AgentState) -> str:
        next_node = state.get("next_agent", "FINISH")
        return "final_answer" if next_node == "FINISH" else next_node

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
        # 依赖链测试：task1 依赖 task0，math_agent 连续被调用两次
        "把 88 和 12 相加，再把结果乘以 5",

        # 多 Agent + 无依赖并列任务
        "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",

        # 纯数据分析
        """分析这批数据：[{"name":"Alice","dept":"Eng","salary":9000},
         {"name":"Bob","dept":"Mkt","salary":7500},
         {"name":"Charlie","dept":"Eng","salary":11000}]
         按 dept 分组，对 salary 求平均""",

        # 无依赖单任务
        "访问 https://api.github.com/zen 返回了什么？",
    ]

    async def main():
        async with stdio_client(mcp_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                _tools.extend(await load_tools(session))

                for q in QUESTIONS:
                    print(f"\n{'━'*60}\n❓ {q}\n{'━'*60}")
                    result = await graph.ainvoke({
                        "messages":           [HumanMessage(content=q)],
                        "task_plan":          [],
                        "current_task_index": 0,
                        "next_agent":         "",
                    })
                    print(f"\n✨ 最终答案：{result['messages'][-1].content}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())