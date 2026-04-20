"""
src/langgraph_stdio_agent.py

两种运行方式：
  1. uv run langgraph dev   → langgraph 调用 lifespan 钩子初始化工具
  2. python -m src.langgraph_stdio_agent  → __main__ 手动初始化工具

★ 重构说明（相比上一版本）：
   - 移除 router 节点，消除"混合任务被短路到 direct_answer"的问题
   - Planner 承担意图判断：纯问答任务输出 agent="direct" 的特殊任务
   - Supervisor 路由扩展：检测到 agent="direct" 时直接走 direct_answer，
     不经过任何工具 agent 和 replanner
   - LLM 调用次数减少一次（原 router + planner 各一次，现只有 planner）
   - 混合任务（部分闲聊 + 部分工具）现在可以正确执行所有子任务

★ 动态工具注册表（ToolRegistry）
   - 启动时从 MCP 加载工具，自动构建 agent → tools 映射
   - Planner / Supervisor 的 system prompt 动态注入工具描述
   - 每个 Agent 的 allowed_tools 从注册表查找，代码零硬编码
   - 新增/删除工具只需修改 MCP server，LangGraph 代码无需改动

★ Re-Planner 节点
   每个工具 Agent 执行完任务后，Re-Planner 检查结果，
   决定是否需要调整剩余计划，再交还给 Supervisor 继续执行。
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
# 3. 动态工具注册表
#
#   设计思路：
#   - MCP server 上的每个工具，通过 tags / prefix 约定归属到某个 agent
#   - 归属规则写在 AGENT_TOOL_PATTERNS 中，支持前缀匹配和精确匹配
#   - 未匹配到任何 agent 的工具会被放入 "default_agent"
#   - 注册表构建完成后暴露：
#       registry.agents          → list[str]  所有工具 agent 名（不含 "direct"）
#       registry.tools_for(name) → list[StructuredTool]
#       registry.tool_desc_block → str  供 LLM prompt 使用的工具描述块
#       registry.agent_desc_block→ str  供 LLM prompt 使用的 agent 描述块
# ══════════════════════════════════════════════════════

# Agent 工具归属规则：每个 agent 匹配哪些工具名（支持前缀 * 通配）
AGENT_TOOL_PATTERNS: dict[str, list[str]] = {
    "math_agent": ["add_numbers", "multiply_numbers", "subtract_numbers",
                   "divide_numbers", "power*", "sqrt*", "math_*"],
    "data_agent": ["dataframe_summary", "group_and_aggregate", "filter_rows",
                   "sort_dataframe", "pivot_table", "data_*", "df_*"],
    "http_agent": ["fetch_url", "post_json", "http_get", "http_post",
                   "http_*", "fetch_*", "request_*"],
}

# Agent 的语义描述（供 Planner prompt 使用）
AGENT_DESCRIPTIONS: dict[str, str] = {
    "math_agent": "数学计算（加减乘除、幂、开方等数值运算）",
    "data_agent": "数据分析（统计、聚合、分组、过滤等结构化数据处理）",
    "http_agent": "网络请求（GET/POST、访问 URL、调用外部 API）",
    # ★ "direct" 是特殊虚拟 agent，不在此注册，由 planner prompt 单独说明
}

def _match_agent(tool_name: str) -> str:
    """根据 AGENT_TOOL_PATTERNS 把工具名映射到 agent 名，未匹配返回 'default_agent'"""
    for agent, patterns in AGENT_TOOL_PATTERNS.items():
        for pat in patterns:
            if pat.endswith("*"):
                if tool_name.startswith(pat[:-1]):
                    return agent
            else:
                if tool_name == pat:
                    return agent
    return "default_agent"


@dataclass
class ToolRegistry:
    """
    启动时从 MCP 工具列表构建，之后只读。
    提供 prompt 片段和工具查找接口，供各节点直接使用。
    注意："direct" 是虚拟 agent，不存储在注册表里。
    """
    _agent_tools: dict[str, list[StructuredTool]] = field(default_factory=dict)
    _all_tools: dict[str, StructuredTool] = field(default_factory=dict)
    _tool_desc_block: str = ""
    _agent_desc_block: str = ""

    @classmethod
    def build(cls, lc_tools: list[StructuredTool]) -> "ToolRegistry":
        reg = cls()
        for tool in lc_tools:
            reg._all_tools[tool.name] = tool
            agent = _match_agent(tool.name)
            reg._agent_tools.setdefault(agent, []).append(tool)

        lines_tools: list[str] = ["【可用工具列表】"]
        for agent, tools in reg._agent_tools.items():
            desc = AGENT_DESCRIPTIONS.get(agent, "通用处理")
            lines_tools.append(f"\n▸ {agent}（{desc}）：")
            for t in tools:
                lines_tools.append(f"    - {t.name}：{t.description or '（无描述）'}")

        lines_agents: list[str] = ["【可用 Agent 列表】"]
        for agent in reg.agents:
            desc = AGENT_DESCRIPTIONS.get(agent, "通用处理")
            tool_names = [t.name for t in reg._agent_tools.get(agent, [])]
            lines_agents.append(f"  - {agent}：{desc}（工具：{', '.join(tool_names)}）")
        # 在 agent 列表里补充 direct 的说明
        lines_agents.append("  - direct：直接用语言模型回答，不调用任何工具（闲聊/知识问答）")

        reg._tool_desc_block  = "\n".join(lines_tools)
        reg._agent_desc_block = "\n".join(lines_agents)

        print(f"✅ ToolRegistry 构建完成：")
        for agent, tools in reg._agent_tools.items():
            print(f"   {agent}: {[t.name for t in tools]}")
        return reg

    @property
    def agents(self) -> list[str]:
        """返回真实工具 agent 列表，不含虚拟的 'direct'"""
        return list(self._agent_tools.keys())

    @property
    def tool_desc_block(self) -> str:
        return self._tool_desc_block

    @property
    def agent_desc_block(self) -> str:
        return self._agent_desc_block

    def tools_for(self, agent: str) -> list[StructuredTool]:
        return self._agent_tools.get(agent, [])

    def tool_names_for(self, agent: str) -> list[str]:
        return [t.name for t in self.tools_for(agent)]

    def get_tool(self, name: str) -> StructuredTool | None:
        return self._all_tools.get(name)


# ══════════════════════════════════════════════════════
# 4. State
# ══════════════════════════════════════════════════════
class TaskInput(TypedDict):
    from_task: int
    field: str

class Task(TypedDict):
    task_id: int
    description: str
    agent: str          # 工具 agent 名，或特殊值 "direct"
    inputs: dict[str, TaskInput]
    depends_on: list[int]
    status: str
    result: str
    _resolved_description: str

class AgentState(TypedDict):
    messages: list
    task_plan: list[Task]
    current_task_id: int
    next_agent: str     # supervisor 写入，供图的条件路由使用；"FINISH" 表示全部完成


# ══════════════════════════════════════════════════════
# 5. 共享容器（lifespan / __main__ 初始化后填充）
# ══════════════════════════════════════════════════════
_tools: list[StructuredTool] = []
_registry: ToolRegistry = ToolRegistry()


# ══════════════════════════════════════════════════════
# 6. 工具加载
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


def _init_registry(tools: list[StructuredTool]) -> None:
    global _registry
    _registry = ToolRegistry.build(tools)


# ══════════════════════════════════════════════════════
# 7. Planner
#
#   ★ 重构核心：Planner 现在同时承担意图判断职责
#   - 需要工具的任务 → 正常输出工具 agent 任务
#   - 纯问答/闲聊   → 输出 agent="direct" 的特殊任务，
#                      description 字段直接写好回答内容
#   - 混合任务       → direct 任务和工具任务可在同一个 task_plan 中共存
# ══════════════════════════════════════════════════════
def _planner_system() -> str:
    valid_agents = ", ".join(_registry.agents) if _registry.agents else "（无可用工具 Agent）"
    return f"""你是任务规划器。把用户问题拆解为有序子任务列表。

{_registry.agent_desc_block}

━━ agent 选择规则 ━━
1. 需要调用工具才能完成的任务 → 选择对应的工具 agent（{valid_agents}）
2. 不需要工具、仅凭语言模型即可回答的任务 → agent 填 "direct"，
   并在 description 里直接写好完整回答内容（这将作为该任务的最终输出）
   适用场景：闲聊、问候、概念解释、知识性问答等

━━ 其他规则（严格遵守）━━
3. description 只写意图（工具任务）或完整回答（direct 任务），绝不提前计算数值
4. inputs 声明运行时需要从哪些前置任务获取参数：
   - key   = 参数的语义名称（如"加数A"、"被乘数"），供 Agent 理解用途
   - value = {{"from_task": <被依赖的task_id>, "field": "result"}}
5. depends_on 从 inputs 的 from_task 自动推导
6. 没有依赖的任务：inputs 为 {{}}，depends_on 为 []
7. 同一个 agent 可出现多次
8. 任务按拓扑顺序排列（被依赖的任务排在前面）
9. 如果用户消息中已直接包含数据（JSON 数组、数字、文本等），
   不要单独拆"获取数据"任务，直接在处理任务的 description 里完整引用

严格只输出 JSON 数组，不要有任何其他内容，示例：
[
  {{
    "task_id": 0,
    "description": "你好！我是 AI 助手，有什么可以帮你的吗？",
    "agent": "direct",
    "inputs": {{}},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }},
  {{
    "task_id": 1,
    "description": "计算 88 加 12",
    "agent": "math_agent",
    "inputs": {{}},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }},
  {{
    "task_id": 2,
    "description": "把前一步的结果乘以 5",
    "agent": "math_agent",
    "inputs": {{
      "被乘数": {{"from_task": 1, "field": "result"}}
    }},
    "depends_on": [1],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }}
]"""


async def planner_node(state: AgentState) -> AgentState:
    if state.get("task_plan"):
        return state

    print("\n  📋 Planner 开始拆解任务...")
    response = await llm.ainvoke([
        SystemMessage(content=_planner_system()),
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
            # direct 任务截断描述显示
            desc_preview = t['description'][:60] + "…" if len(t['description']) > 60 else t['description']
            print(f"     [{t['task_id']}] {t['agent']} ← {desc_preview}{dep_str}")
    except json.JSONDecodeError:
        print(f"  ⚠️ JSON 解析失败：{raw[:200]}")
        # 兜底：当作一个 direct 任务，让 LLM 直接回答
        task_plan = [{
            "task_id": 0,
            "description": state["messages"][0].content,
            "agent": "direct",
            "inputs": {},
            "depends_on": [],
            "status": "pending",
            "result": "",
            "_resolved_description": "",
        }]

    return {**state, "task_plan": task_plan, "current_task_id": 0}


# ══════════════════════════════════════════════════════
# 8. Supervisor
#
#   ★ 重构：新增对 agent="direct" 的识别
#   - 找到第一个 pending 任务
#   - 若 agent == "direct" → 直接标记为 done，result 取 description，
#     next_agent 设为 "direct_answer"，跳过 replanner
#   - 若 agent 为工具 agent   → 正常派发，走 replanner
#   - 若无 pending 任务       → next_agent = "FINISH"
# ══════════════════════════════════════════════════════
async def supervisor_node(state: AgentState) -> AgentState:
    task_plan: list[Task] = state.get("task_plan", [])
    done_map = {t["task_id"]: t for t in task_plan if t["status"] == "done"}

    for task in task_plan:
        if task["status"] != "pending":
            continue

        unmet = [dep for dep in task.get("depends_on", []) if dep not in done_map]
        if unmet:
            print(f"\n  ⏳ 任务[{task['task_id']}] 等待依赖 {unmet}，跳过")
            continue

        # ★ direct 任务：无需工具，直接完成
        if task.get("agent") == "direct":
            print(f"\n  🧭 Supervisor → direct_answer（任务[{task['task_id']}]）")
            task["status"] = "done"
            task["result"] = task["description"]   # description 就是回答内容
            task["_resolved_description"] = task["description"]
            return {
                **state,
                "next_agent": "direct_answer",
                "current_task_id": task["task_id"],
                "task_plan": task_plan,
            }

        # 正常工具任务：解析运行时参数
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
        return {
            **state,
            "next_agent": task["agent"],
            "current_task_id": task["task_id"],
            "task_plan": task_plan,
        }

    # 所有任务完成（或依赖无法满足）
    pending = [t for t in task_plan if t["status"] == "pending"]
    if pending:
        print(f"\n  ⚠️ 任务 {[t['task_id'] for t in pending]} 依赖无法满足，强制跳过")
        for t in pending:
            t["status"] = "done"
            t["result"] = "⚠️ 依赖未满足，跳过"

    print("\n  🧭 Supervisor → FINISH")
    return {**state, "next_agent": "FINISH", "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 9. Re-Planner（仅在工具 agent 执行后触发，direct 任务不经过此节点）
# ══════════════════════════════════════════════════════
def _replanner_system() -> str:
    valid_agents = ", ".join(_registry.agents) if _registry.agents else "（无可用 Agent）"
    return f"""你是任务再规划器。在每个工具子任务完成后，你需要判断是否需要调整剩余计划。

{_registry.agent_desc_block}

你会收到：
1. 用户的原始问题
2. 已完成任务的结果摘要
3. 还未执行的 pending 任务列表

你的决策：

【情况A】计划不需要调整（绝大多数情况）
直接输出：{{"action": "continue"}}

【情况B】需要调整剩余计划（仅当出现以下情况时）
- 某个任务的结果表明后续任务的方向需要改变
- 发现原计划遗漏了必要步骤
- 某个任务失败，需要用替代方案
输出修改后的完整 pending 任务列表：
{{"action": "replan", "new_pending_tasks": [...]}}

注意事项：
- new_pending_tasks 是完整的新 pending 列表，会替换掉所有旧的 pending 任务
- task_id 从当前已完成任务的最大 id + 1 开始重新编号
- inputs 里的 from_task 如果引用已完成任务，task_id 保持原来的值不变
- status 统一写 "pending"，result 和 _resolved_description 写空字符串
- agent 只能是：{valid_agents}（replan 时不应产生 direct 任务）
- 不要新增任何"等待用户输入"、"获取数据"、"读取数据"类的占位任务
- 如果不确定要不要 replan，选择 continue

严格只输出 JSON，不要有任何其他内容。"""


async def replanner_node(state: AgentState) -> AgentState:
    task_plan: list[Task]  = state.get("task_plan", [])
    current_task_id: int   = state.get("current_task_id", -1)

    done_tasks    = [t for t in task_plan if t["status"] == "done"]
    pending_tasks = [t for t in task_plan if t["status"] == "pending"]

    if not pending_tasks:
        print("\n  🔄 Re-Planner：无剩余任务，跳过")
        return state

    done_summary    = "\n".join(
        f"  任务[{t['task_id']}]（{t['description']}）→ 结果：{t['result'][:100]}"
        for t in done_tasks
    )
    pending_summary = json.dumps(pending_tasks, ensure_ascii=False, indent=2)

    print(f"\n  🔄 Re-Planner 检查任务[{current_task_id}]完成后是否需要调整计划...")

    response = await llm.ainvoke([
        SystemMessage(content=_replanner_system()),
        HumanMessage(content=(
            f"用户原始问题：{state['messages'][0].content}\n\n"
            f"已完成任务：\n{done_summary}\n\n"
            f"待执行任务（pending）：\n{pending_summary}"
        )),
    ])

    raw = response.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        decision = json.loads(raw.strip())
    except json.JSONDecodeError:
        print(f"  ⚠️ Re-Planner JSON 解析失败，保持原计划：{raw[:100]}")
        return state

    if decision.get("action") == "continue":
        print("  ✅ Re-Planner：计划无需调整，继续执行")
        return state

    if decision.get("action") == "replan":
        new_pending: list[Task] = decision.get("new_pending_tasks", [])
        if not new_pending:
            print("  ✅ Re-Planner：replan 但 new_pending_tasks 为空，继续执行")
            return state

        valid_agents = set(_registry.agents)
        new_pending = [t for t in new_pending if t.get("agent") in valid_agents]
        if not new_pending:
            print("  ⚠️ Re-Planner：所有新任务 agent 非法，保持原计划")
            return state

        updated_plan = done_tasks + new_pending
        print(f"  🔁 Re-Planner：计划已调整，新增/修改 {len(new_pending)} 个 pending 任务：")
        for t in new_pending:
            print(f"     [{t['task_id']}] {t['agent']} ← {t['description']}")
        return {**state, "task_plan": updated_plan}

    print("  ✅ Re-Planner：未识别的 action，保持原计划")
    return state


# ══════════════════════════════════════════════════════
# 10. 通用工具 Agent 执行器
# ══════════════════════════════════════════════════════
async def run_agent(
    state: AgentState,
    agent_name: str,
    system_prompt: str,
) -> AgentState:
    lc_tools   = _registry.tools_for(agent_name)
    tool_names = _registry.tool_names_for(agent_name)
    by_name    = {t.name: t for t in lc_tools}

    task_plan: list[Task] = state.get("task_plan", [])
    task_id: int          = state.get("current_task_id", 0)

    current_task = next((t for t in task_plan if t["task_id"] == task_id), None)
    if not current_task:
        print(f"  ⚠️ [{agent_name}] 找不到任务 task_id={task_id}")
        return state

    task_description = current_task.get("_resolved_description") or current_task["description"]

    if not lc_tools:
        current_task["status"] = "done"
        current_task["result"] = f"⚠️ 没有可用工具（{agent_name} 未注册任何工具）"
        return {**state, "task_plan": task_plan}

    tool_hint = "可用工具：" + "、".join(
        f"{t.name}（{t.description or '无描述'}）" for t in lc_tools
    )
    full_system = f"{system_prompt}\n\n{tool_hint}"

    agent_llm = llm.bind_tools(lc_tools)
    msgs = [
        SystemMessage(content=full_system),
        HumanMessage(content=task_description),
    ]

    print(f"\n  ▶ {agent_name} 执行任务[{task_id}]（工具：{tool_names}）")

    max_steps     = 10
    last_response = None

    for step in range(max_steps):
        response = await agent_llm.ainvoke(msgs)
        last_response = response
        msgs.append(response)

        if not (hasattr(response, "tool_calls") and response.tool_calls):
            print(f"  ✅ {agent_name} 任务[{task_id}] 完成（{step + 1} 轮推理）")
            break

        print(f"  📋 {agent_name} 第 {step + 1} 轮，调用工具：")
        for tc in response.tool_calls:
            tool = by_name.get(tc["name"])
            if tool:
                args        = {k: v for k, v in tc["args"].items() if v is not None}
                result_text = await tool.coroutine(**args)
            else:
                result_text = f"❌ 未找到工具：{tc['name']}"
            msgs.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
    else:
        print(f"  ⚠️ {agent_name} 达到最大步数 {max_steps}，强制终止")

    current_task["status"] = "done"
    current_task["result"] = last_response.content if last_response else "（无结果）"

    return {**state, "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 11. 图构建
#
#   ★ 重构：移除 router 节点，入口直接为 planner
#   ★ supervisor 路由扩展：新增 "direct_answer" 分支
#   ★ direct_answer 节点不经过 replanner，直接回到 supervisor
#     （处理可能存在的后续工具任务）
#
#   图结构（文字版）：
#
#   planner → supervisor ─┬─→ direct_answer ──────────────────┐
#                         ├─→ math_agent → replanner → supervisor
#                         ├─→ data_agent → replanner → supervisor
#                         ├─→ http_agent → replanner → supervisor
#                         └─→ final_answer → END
# ══════════════════════════════════════════════════════

AGENT_SYSTEM_PROMPTS: dict[str, str] = {
    "math_agent": (
        "你是数学计算专家。根据任务描述和【运行时参数】（如果有），"
        "推断需要做什么运算，调用合适的工具完成计算，得到结果后立即返回。"
        "只输出最终数值结果，不要解释过程。"
    ),
    "data_agent": (
        "你是数据分析专家。根据任务描述和【运行时参数】（如果有），"
        "提取数据集、分组列、聚合列、聚合函数等信息，调用合适的工具完成分析。"
        "任务要求几种聚合就做几种，不要自行扩展。完成后给出简洁结论。"
    ),
    "http_agent": (
        "你是网络请求专家。根据任务描述和【运行时参数】（如果有），"
        "发送对应的 HTTP 请求，成功拿到响应后立即返回结果，不要重试。"
        "timeout 默认使用 10。"
    ),
}

DEFAULT_AGENT_SYSTEM_PROMPT = (
    "你是通用任务执行专家。根据任务描述，调用合适的工具完成任务，给出简洁结果。"
)


def build_graph() -> Any:

    # ── direct_answer 节点 ────────────────────────────
    # ★ 重构：direct_answer 现在从 task_plan 里取当前任务的 result（即 description）
    #          输出后回到 supervisor，继续处理后续可能存在的工具任务
    async def direct_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])
        task_id: int          = state.get("current_task_id", 0)
        current_task = next((t for t in task_plan if t["task_id"] == task_id), None)
        answer = current_task["result"] if current_task else "（无回答）"
        print(f"\n  💬 direct_answer：{answer[:80]}")
        # 把回答追加到消息流，然后回到 supervisor 处理后续任务
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=answer)],
        }

    # ── 最终汇总节点 ──────────────────────────────────
    async def final_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])

        # 工具任务的结果需要汇总；direct 任务的结果已经在消息流里，无需再次包含
        tool_results = [t for t in task_plan if t.get("agent") != "direct"]

        if not tool_results:
            # 全部都是 direct 任务，消息流里已有完整回答，无需再调 LLM
            print("\n  📝 所有任务均为 direct，最终回答已在消息流中")
            return state

        results_text = "\n".join(
            f"任务[{t['task_id']}]（{t['description']}）：{t['result']}"
            for t in tool_results
        )
        print(f"\n  📝 汇总工具任务结果：\n{results_text}")

        response = await llm.ainvoke([
            SystemMessage(content=(
                "根据以下各子任务的执行结果，用中文给用户一个清晰完整的最终答案。\n\n"
                f"{results_text}"
            )),
            state["messages"][0],
        ])
        return {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}

    # ── 路由函数 ──────────────────────────────────────
    def supervisor_route(state: AgentState) -> str:
        next_node = state.get("next_agent", "FINISH")
        if next_node == "FINISH":
            return "final_answer"
        return next_node   # "direct_answer" 或工具 agent 名

    # ── 动态创建工具 agent 节点 ───────────────────────
    def make_agent_node(name: str):
        system_prompt = AGENT_SYSTEM_PROMPTS.get(name, DEFAULT_AGENT_SYSTEM_PROMPT)
        async def _node(state: AgentState) -> AgentState:
            return await run_agent(state, name, system_prompt)
        _node.__name__ = name
        return _node

    # ── 图构建 ────────────────────────────────────────
    g = StateGraph(AgentState)

    # ★ 不再有 router 节点，入口直接是 planner
    g.add_node("planner",       planner_node)
    g.add_node("supervisor",    supervisor_node)
    g.add_node("replanner",     replanner_node)
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("final_answer",  final_answer_node)

    known_agents = _registry.agents or list(AGENT_SYSTEM_PROMPTS.keys())
    for agent_name in known_agents:
        g.add_node(agent_name, make_agent_node(agent_name))

    # ── 边 ────────────────────────────────────────────
    g.set_entry_point("planner")
    g.add_edge("planner", "supervisor")

    # supervisor → direct_answer / 工具 agent / final_answer
    agent_routes = {name: name for name in known_agents}
    agent_routes["direct_answer"] = "direct_answer"
    agent_routes["final_answer"]  = "final_answer"
    g.add_conditional_edges("supervisor", supervisor_route, agent_routes)

    # ★ direct_answer 完成后回到 supervisor，继续处理后续工具任务
    g.add_edge("direct_answer", "supervisor")

    # 工具 agent 完成后进 replanner，再回 supervisor
    for agent_name in known_agents:
        g.add_edge(agent_name, "replanner")
    g.add_edge("replanner", "supervisor")

    g.add_edge("final_answer", END)

    return g.compile()


# ══════════════════════════════════════════════════════
# 12. 图实例（延迟初始化）
# ══════════════════════════════════════════════════════
graph = build_graph()


def _rebuild_graph() -> None:
    global graph
    graph = build_graph()
    print("🔄 Graph 已用真实 ToolRegistry 重建")


# ══════════════════════════════════════════════════════
# 13. lifespan —— 仅 langgraph dev 调用
# ══════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app):
    if not SERVER_PATH.exists():
        raise FileNotFoundError(f"找不到 MCP server：{SERVER_PATH}")
    async with stdio_client(mcp_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await load_tools(session)
            _tools.extend(loaded)
            _init_registry(loaded)
            _rebuild_graph()
            print("🚀 [lifespan] MCP + ToolRegistry 就绪")
            yield
    _tools.clear()
    print("🛑 [lifespan] MCP 已关闭")


# ══════════════════════════════════════════════════════
# 14. __main__ —— 单独运行测试
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    if not SERVER_PATH.exists():
        print(f"❌ 找不到 server.py：{SERVER_PATH}")
        sys.exit(1)

    QUESTIONS = [
        # 纯闲聊 → direct 任务，不走工具
        "你好，我叫 tony",
        "什么是机器学习？",
        # 纯工具任务
        "把 88 和 12 相加，再把结果乘以 5",
        "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",
        # 混合任务：闲聊 + 工具（原来会被短路，现在能正确执行）
        "先介绍一下你自己，然后帮我计算 99 乘以 9",
        # 数据分析
        """分析这批数据：[{"name":"Alice","dept":"Eng","salary":9000},
         {"name":"Bob","dept":"Mkt","salary":7500},
         {"name":"Charlie","dept":"Eng","salary":11000}]
         按 dept 分组，对 salary 求平均""",
        # 网络请求
        "访问 https://api.github.com/zen 返回了什么？",
    ]

    async def main():
        async with stdio_client(mcp_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                loaded = await load_tools(session)
                _tools.extend(loaded)
                _init_registry(loaded)
                _rebuild_graph()

                for q in QUESTIONS:
                    print(f"\n{'━'*60}\n❓ {q}\n{'━'*60}")
                    result = await graph.ainvoke({
                        "messages":        [HumanMessage(content=q)],
                        "task_plan":       [],
                        "current_task_id": 0,
                        "next_agent":      "",
                    })
                    print(f"\n✨ 最终答案：{result['messages'][-1].content}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())