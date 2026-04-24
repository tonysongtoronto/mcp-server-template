"""
src/langgraph_stdio_agent.py

两种运行方式：
  1. uv run langgraph dev   → langgraph 调用 lifespan 钩子初始化工具
  2. python -m src.langgraph_stdio_agent  → __main__ 手动初始化工具

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
★ 本版本重构说明（在原版基础上的新增改动）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  【原版保留】
    重构1  移除 router 节点，Planner 承担意图判断
    重构2  direct_answer_node 只回答当前子任务
    重构3  planner_node JSON 解析失败最多重试 3 次
    修复4  supervisor in_progress 容错
    修复5  _extract_json 统一抽取
    修复6  final_answer_node 混合任务 direct 结果纳入汇总
    修复8  run_agent 工具调用强制兜底
    修复9  兼容 llm.ainvoke() 返回 dict
    修复10 兼容 LangGraph Studio messages 反序列化为 dict

  【新版改动 A】并行任务调度（Send API + Map-Reduce）
    - 问题：所有任务串行执行，互相独立的任务也要一个等一个
    - 方案：
        · AgentState.task_plan 加 Annotated Reducer，安全合并并发写入
        · 新增 WorkerState，每个并发 worker 持有私有执行上下文
        · supervisor_node 改为 supervisor_dispatch：
            找出本轮所有"依赖已满足"的 pending 任务，
            用 Send API 一次性全部并发分发，不再串行一个个路由
        · 新增 collect_node 作为并发汇聚点：
            所有 worker 完成后统一汇聚，再触发下一轮调度
        · 图结构从 supervisor→单agent→replanner→supervisor
            改为   supervisor_dispatch→[并发agents]→collect→supervisor_dispatch

  【新版改动 B】删除 _validate_and_fix_task_plan
    - 问题：关键词硬匹配不考虑语境，会误改 Planner 的合理判断
    - 方案：直接删除该函数及 AGENT_TRIGGER_KEYWORDS、_task_needs_tool_agent，
            依赖 Planner 系统提示的强约束保证 agent 选择正确性

  【新版改动 C】replanner_node 改为纯逻辑失败检测器
    - 问题：每次都调 LLM 询问"要不要 replan"，但因 prompt 措辞
            几乎永远返回 continue，既浪费 token 又毫无作用
    - 方案：移除 LLM 调用，改为纯逻辑检测：
            检查刚完成的任务结果是否包含失败标记，
            若失败则级联跳过所有直接/间接依赖它的后续任务（BFS），
            正常完成时零开销直接透传

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  新图结构：

  planner ──(FINISH)──→ END
          ──(ok)──────→ supervisor_dispatch
                              │
                    ┌─────────┼─────────┐   ← Send 并发分发
                    ▼         ▼         ▼
               math_agent  http_agent  direct_answer  ...
                    │         │         │
                    └────┬────┘─────────┘
                         ▼
                    collect_node        ← 汇聚点（等所有并发完成）
                         │
                         ▼
                  supervisor_dispatch  ← 下一轮调度（有依赖被解锁的任务）
                         │
                         ▼ (无更多就绪任务)
                    final_answer ──→ END
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import json
import operator
import os
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.types import Send
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
# ══════════════════════════════════════════════════════

AGENT_TOOL_PATTERNS: dict[str, list[str]] = {
    "math_agent": ["add_numbers", "multiply_numbers", "subtract_numbers",
                   "divide_numbers", "power*", "sqrt*", "math_*"],
    "data_agent": ["dataframe_summary", "group_and_aggregate", "filter_rows",
                   "sort_dataframe", "pivot_table", "data_*", "df_*"],
    "http_agent": ["fetch_url", "post_json", "http_get", "http_post",
                   "http_*", "fetch_*", "request_*"],
}

AGENT_DESCRIPTIONS: dict[str, str] = {
    "math_agent": "数学计算（加减乘除、幂、开方等数值运算）",
    "data_agent": "数据分析（统计、聚合、分组、过滤等结构化数据处理）",
    "http_agent": "网络请求（GET/POST、访问 URL、调用外部 API）",
}

def _match_agent(tool_name: str) -> str:
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
    _agent_tools: dict[str, list[StructuredTool]] = field(default_factory=dict)
    _all_tools: dict[str, StructuredTool] = field(default_factory=dict)
    _tool_desc_block: str = ""
    _agent_desc_block: str = ""
    _agent_desc_brief: str = ""

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
        brief_lines: list[str]  = []
        for agent in reg.agents:
            desc = AGENT_DESCRIPTIONS.get(agent, "通用处理")
            tool_names = [t.name for t in reg._agent_tools.get(agent, [])]
            lines_agents.append(f"  - {agent}：{desc}（工具：{', '.join(tool_names)}）")
            brief_lines.append(f"  · {agent}：{desc}")
        lines_agents.append(
            "  - direct：直接用语言模型回答，不调用任何工具"
            "（仅限：闲聊/问候/概念解释/知识性问答）"
        )

        reg._tool_desc_block  = "\n".join(lines_tools)
        reg._agent_desc_block = "\n".join(lines_agents)
        reg._agent_desc_brief = (
            "\n".join(brief_lines) if brief_lines else "  （暂无已注册的工具 Agent）"
        )

        print("✅ ToolRegistry 构建完成：")
        for agent, tools in reg._agent_tools.items():
            print(f"   {agent}: {[t.name for t in tools]}")
        return reg

    @property
    def agents(self) -> list[str]:
        """真实工具 agent 列表，不含虚拟的 'direct'"""
        return list(self._agent_tools.keys())

    @property
    def tool_desc_block(self) -> str:
        return self._tool_desc_block

    @property
    def agent_desc_block(self) -> str:
        return self._agent_desc_block

    @property
    def agent_desc_brief(self) -> str:
        return self._agent_desc_brief

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
    agent: str
    inputs: dict[str, TaskInput]
    depends_on: list[int]
    status: str
    result: str
    _resolved_description: str


def _merge_task_plan(old: list[Task], new: list[Task]) -> list[Task]:
    """
    ★ 新增改动A：task_plan 的自定义 Reducer。
    并发 worker 各自只修改自己负责的那个 task，
    合并时按 task_id 做字典合并，防止并发写入互相覆盖。
    """
    merged = {t["task_id"]: t for t in old}
    for t in new:
        merged[t["task_id"]] = t
    # 按 task_id 升序排列，保持顺序一致
    return sorted(merged.values(), key=lambda t: t["task_id"])


class AgentState(TypedDict):
    """主图 State"""
    messages:   list
    task_plan:  Annotated[list[Task], _merge_task_plan]   # ★ Reducer 并发安全
    next_agent: str   # 仅 planner→END 短路路由使用，并行后不再用于 agent 路由


class WorkerState(TypedDict):
    """
    ★ 新增改动A：每个并发 worker 的私有执行上下文。
    Send API 把不同的 WorkerState 发给不同的 agent 节点，
    各 worker 互不干扰，完成后结果通过 Reducer 安全归并到主 AgentState。
    """
    task_plan:       list[Task]   # 完整计划（只读，用于查依赖上下文）
    current_task_id: int          # 本 worker 负责的任务 ID
    messages:        list         # 原始用户消息（direct_answer 需要）


# ══════════════════════════════════════════════════════
# 5. 共享容器
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
# 公共工具函数
# ══════════════════════════════════════════════════════

def _extract_json(raw: str) -> str:
    """
    修复5：统一的 JSON 代码块清理函数。
    去除 ```json ... ``` 或 ``` ... ``` 包裹，并 strip 空白。
    """
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        inner = parts[1] if len(parts) > 1 else parts[0]
        inner = re.sub(r"^[a-zA-Z]+\n", "", inner)
        return inner.strip()
    return raw


def _extract_llm_content(response: Any) -> str:
    """
    修复9：兼容 llm.ainvoke() 返回 AIMessage 对象或原始 dict 两种情况。
    DeepSeek + langchain_openai 在部分版本下会返回 dict 而非 AIMessage。
    """
    if hasattr(response, "content"):
        return response.content or ""
    if isinstance(response, dict):
        return (
            response.get("content")
            or response.get("text")
            or response.get("output")
            or str(response)
        )
    return str(response)


def _get_message_content(msg: Any) -> str:
    """
    修复10：兼容 LangGraph Studio 下 messages 反序列化为 dict 的情况。
    """
    if hasattr(msg, "content"):
        return msg.content or ""
    if isinstance(msg, dict):
        return msg.get("content") or msg.get("text") or str(msg)
    return str(msg)


def _get_first_user_message(state: AgentState) -> Any:
    """
    从 state["messages"] 中取第一条用户消息，
    兼容 HumanMessage 对象和 dict 两种格式。
    """
    msgs = state.get("messages", [])
    if not msgs:
        return HumanMessage(content="")
    msg = msgs[0]
    if isinstance(msg, dict):
        return HumanMessage(content=msg.get("content") or msg.get("text") or "")
    return msg


# ══════════════════════════════════════════════════════
# 7. Planner
#
#   重构1：承担意图判断，支持 agent="direct" 特殊任务
#   重构3：JSON 解析失败最多重试 3 次，全部失败则内部闭环处理
#   ★ 新版改动B：移除 _validate_and_fix_task_plan 调用，
#               不再做关键词硬匹配后处理
# ══════════════════════════════════════════════════════

def _planner_system() -> str:
    valid_agents = ", ".join(_registry.agents) if _registry.agents else "（无可用工具 Agent）"
    return f"""你是任务规划器。把用户问题拆解为有序子任务列表。

{_registry.agent_desc_block}

{_registry.tool_desc_block}

━━ agent 选择规则（严格遵守，违反将导致系统错误）━━

【重要】agent 的选择直接决定是否调用工具，请仔细判断：

✅ 必须使用工具 agent 的情况：
  - 任何数值计算（加、减、乘、除、幂、开方等）→ math_agent
  - 任何网络请求（访问 URL、调用 HTTP API、fetch 等）→ http_agent
  - 任何数据分析（统计、分组、聚合、过滤等）→ data_agent

❌ 严禁使用 direct 的情况（以下场景必须用工具 agent）：
  - "计算 3+5" → 必须用 math_agent（不能自己算，必须调工具）
  - "访问 https://..." → 必须用 http_agent（不能模拟，必须真实请求）
  - "分析这批数据" → 必须用 data_agent（不能自行统计，必须调工具）

✅ 可以使用 direct 的情况（仅限以下场景）：
  - 闲聊、问候（如"你好"、"介绍一下你自己"）
  - 纯知识性问答（如"什么是加权平均数"）
  - 不涉及任何计算、网络请求、数据处理的场景

判断口诀：只要涉及"算数字"、"访问网络"、"处理数据"，一律用工具 agent，绝不用 direct。

━━ 其他规则（严格遵守）━━
1. description 只写任务意图，绝不提前计算数值或给出最终答案
2. inputs 声明运行时需要从哪些前置任务获取参数：
   - key   = 参数的语义名称（如"加数A"、"被乘数"），供 Agent 理解用途
   - value = {{"from_task": [被依赖的task_id], "field": "result"}}
3. depends_on 从 inputs 的 from_task 自动推导
4. 没有依赖的任务：inputs 为 {{}}，depends_on 为 []
5. 同一个 agent 可出现多次
6. 任务按拓扑顺序排列（被依赖的任务排在前面）
7. 如果用户消息中已直接包含数据（JSON 数组、数字、文本等），
   不要单独拆"获取数据"任务，直接在处理任务的 description 里完整引用

严格只输出 JSON 数组，不要有任何其他内容、代码块标记或说明文字。

<example>
[
  {{
    "task_id": 0,
    "description": "用户打招呼，介绍自己叫 tony",
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
  }},
  {{
    "task_id": 3,
    "description": "访问 https://api.github.com/zen 获取随机箴言",
    "agent": "http_agent",
    "inputs": {{}},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }}
]
</example>

注意：上面示例中"计算 88 加 12"用了 math_agent，"访问 URL"用了 http_agent，这是正确的。
绝对不能把这类任务写成 agent="direct"。"""


def _planner_retry_system(attempt: int, last_raw: str) -> str:
    """重试时使用更严格的提示，附上上次的错误输出片段"""
    if attempt == 1:
        return (
            f"{_planner_system()}\n\n"
            f"⚠️ 注意：你上一次的输出 JSON 解析失败，无法被程序解析。\n"
            f"上次输出片段（前200字符）：\n{last_raw[:200]}\n\n"
            f"请检查并修正，确保只输出合法的 JSON 数组，"
            f"不要包含任何代码块标记（如 ```json）、说明文字或多余字符。"
        )
    else:
        valid_agents = ", ".join(_registry.agents) if _registry.agents else "direct"
        return (
            "你是任务规划器。严格按照以下要求输出，不得有任何偏差：\n\n"
            "1. 只输出一个 JSON 数组，数组里是任务对象\n"
            "2. 不要输出任何其他文字、代码块标记、注释\n"
            f"3. agent 只能是以下值之一：{valid_agents}, direct\n"
            "4. 每个任务对象必须包含这些字段：\n"
            '   task_id(int), description(str), agent(str), inputs(dict),\n'
            '   depends_on(list), status("pending"), result(""), _resolved_description("")\n\n'
            f"⚠️ 上次输出仍然解析失败，片段：{last_raw[:200]}\n\n"
            f"用户问题：请重新规划。"
        )


async def planner_node(state: AgentState) -> AgentState:
    if state.get("task_plan"):
        return state

    print("\n  📋 Planner 开始拆解任务...")

    user_message = _get_first_user_message(state)
    last_raw     = ""
    task_plan: list[Task] | None = None
    max_attempts = 3

    for attempt in range(max_attempts):
        if attempt == 0:
            sys_msg = SystemMessage(content=_planner_system())
        else:
            print(f"  🔁 Planner 第 {attempt + 1} 次重试（JSON 解析失败）...")
            sys_msg = SystemMessage(content=_planner_retry_system(attempt, last_raw))

        response = await llm.ainvoke([sys_msg, user_message])
        raw = _extract_llm_content(response).strip()
        raw = _extract_json(raw)
        last_raw = raw

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise json.JSONDecodeError("期望 JSON 数组", raw, 0)
            task_plan = parsed
            print(f"  ✅ 拆解出 {len(task_plan)} 个任务（第 {attempt + 1} 次尝试成功）：")
            for t in task_plan:
                deps         = t.get("depends_on", [])
                inputs       = t.get("inputs", {})
                dep_str      = f"  依赖→{deps} 参数→{list(inputs.keys())}" if deps else ""
                desc_preview = (
                    t["description"][:60] + "…"
                    if len(t["description"]) > 60
                    else t["description"]
                )
                print(f"     [{t['task_id']}] {t['agent']} ← {desc_preview}{dep_str}")
            break

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠️ 第 {attempt + 1} 次解析失败：{e}  输出片段：{raw[:100]}")

    # 三次全部失败：在 planner 内部闭环，直接告知用户
    if task_plan is None:
        print("  ❌ Planner 连续 3 次解析失败，终止流程")
        agent_scope = _registry.agent_desc_brief
        failure_msg = (
            "⚠️ 任务规划失败，无法处理您的请求。\n\n"
            "━━ 失败详情 ━━\n"
            f"原因：Planner 连续 {max_attempts} 次均无法生成合法的任务计划（JSON 格式错误）\n"
            f"最后一次输出片段：\n  {last_raw[:200]}{'…' if len(last_raw) > 200 else ''}\n\n"
            "━━ 可能的原因 ━━\n"
            "  · 问题描述过于复杂或存在歧义，导致模型输出不稳定\n"
            "  · 问题中包含特殊字符或格式干扰了 JSON 输出\n"
            "  · 请求超出了当前可用工具的处理范围\n\n"
            "━━ 当前可用能力 ━━\n"
            f"{agent_scope}\n\n"
            "建议：请尝试换一种更简洁清晰的方式重新描述您的问题。"
        )
        return {
            **state,
            "messages":   state["messages"] + [AIMessage(content=failure_msg)],
            "task_plan":  [],
            "next_agent": "FINISH",
        }

    # ★ 新版改动B：直接返回，不再调用 _validate_and_fix_task_plan
    #   Planner 系统提示已有足够强的约束，无需关键词硬匹配后处理
    return {
        **state,
        "task_plan":  task_plan,
        "next_agent": "",
    }


# ══════════════════════════════════════════════════════
# 8. Supervisor（并行调度器）
#
#   ★ 新版改动A：从"一次路由一个 agent"改为"批量 Send 并发分发"
#   原 supervisor_node 每次只 return 一个 next_agent，整图串行。
#   新版找出本轮所有依赖已满足的 pending 任务，
#   用 Send API 一次性全部并发发射，互相独立的任务真正同时执行。
# ══════════════════════════════════════════════════════

def _resolve_task_inputs(task: Task, done_map: dict[int, Task]) -> None:
    """把任务的 inputs 依赖解析为运行时参数，写入 _resolved_description。"""
    resolved_inputs: dict[str, str] = {}
    for param_name, source in task.get("inputs", {}).items():
        # source 不是 dict 或没有 from_task 字段，跳过
        if not isinstance(source, dict) or "from_task" not in source:
            continue
        from_id   = source["from_task"]
        src_field = source.get("field", "result")
        src_task  = done_map.get(from_id, {})
        resolved_inputs[param_name] = src_task.get(src_field, "")

    if resolved_inputs:
        params_text = "\n".join(f"  {k} = {v}" for k, v in resolved_inputs.items())
        task["_resolved_description"] = (
            f"{task['description']}\n\n【运行时参数】\n{params_text}"
        )
    else:
        task["_resolved_description"] = task["description"]


def supervisor_dispatch(state: AgentState) -> list[Send]:
    """
    ★ 新版改动A：核心调度函数，返回 list[Send]。
    LangGraph 检测到返回值是 list[Send] 时自动并发执行所有 Send。

    逻辑：
      1. 找出所有 done 任务，构建 done_map
      2. 找出本轮"依赖已全部满足"的 pending 任务（就绪任务）
      3. 每个就绪任务构造独立的 WorkerState，用 Send 并发发射
      4. 若无就绪任务，检查是否有卡死的 pending，强制跳过后走 final_answer
    """
    task_plan = state.get("task_plan", [])
    done_map  = {t["task_id"]: t for t in task_plan if t["status"] == "done"}

    # 找出本轮所有可以立即执行的任务
    ready_tasks = [
        t for t in task_plan
        if t["status"] == "pending"
        and all(dep in done_map for dep in t.get("depends_on", []))
    ]

    if not ready_tasks:
        # 检查是否还有永远无法就绪的任务（依赖链断裂）
        stuck = [t for t in task_plan if t["status"] == "pending"]
        if stuck:
            print(f"\n  ⚠️ 任务 {[t['task_id'] for t in stuck]} 依赖无法满足，强制跳过")
            for t in stuck:
                t["status"] = "done"
                t["result"] = "⚠️ 依赖未满足，跳过"
        print("\n  🧭 Supervisor → final_answer（无更多就绪任务）")
        return [Send("final_answer", state)]

    # 并发分发所有就绪任务
    print(f"\n  🚀 本轮并行分发 {len(ready_tasks)} 个任务：")
    sends: list[Send] = []
    for task in ready_tasks:
        task["status"] = "in_progress"
        _resolve_task_inputs(task, done_map)

        target = task["agent"] if task["agent"] != "direct" else "direct_answer"
        print(f"     [{task['task_id']}] → {target}: {task['description'][:55]}")

        # 每个 worker 持有独立的私有 state，互不干扰
        worker_state: WorkerState = {
            "task_plan":       task_plan,   # 只读参考（查依赖上下文）
            "current_task_id": task["task_id"],
            "messages":        state.get("messages", []),
        }
        sends.append(Send(target, worker_state))

    return sends


# ══════════════════════════════════════════════════════
# 9. Replanner（纯逻辑失败检测器）
#
#   ★ 新版改动C：移除 LLM 调用，改为零开销纯逻辑检测。
#   原版每次都调 LLM 询问"要不要 replan"，因 prompt 措辞
#   几乎永远返回 continue，既浪费 token 又毫无作用。
#
#   新版逻辑：
#     · 检查刚完成的任务结果是否包含失败标记
#     · 失败：BFS 找出所有直接/间接依赖它的后续任务，全部标记跳过
#     · 正常：直接透传，零开销
#
#   注意：并行架构下 replanner 在 collect_node 内部被调用，
#         不再作为独立图节点存在（见图构建部分）。
# ══════════════════════════════════════════════════════

_FAILURE_MARKERS = ("❌", "Error", "error", "失败", "timeout", "⚠️ 依赖", "exception", "Exception")

def _is_task_failed(result: str) -> bool:
    """判断任务结果是否包含失败标记。"""
    return any(marker in result for marker in _FAILURE_MARKERS)


def _cascade_skip(task_plan: list[Task], failed_id: int) -> None:
    """
    从 failed_id 出发，BFS 找出所有直接或间接依赖它的 pending 任务，
    全部标记为跳过。防止脏数据在依赖链上传播。
    """
    to_skip: set[int] = set()
    queue = [failed_id]
    while queue:
        current = queue.pop()
        for task in task_plan:
            if (
                task["status"] == "pending"
                and current in task.get("depends_on", [])
                and task["task_id"] not in to_skip
            ):
                to_skip.add(task["task_id"])
                queue.append(task["task_id"])

    for task in task_plan:
        if task["task_id"] in to_skip:
            task["status"] = "done"
            task["result"] = f"⚠️ 跳过：依赖任务[{failed_id}]执行失败"
            print(f"     → 任务[{task['task_id']}]({task['description'][:40]}) 已跳过")


def _check_and_cascade_failures(task_plan: list[Task]) -> None:
    """
    遍历所有刚完成（done）且结果含失败标记的任务，触发级联跳过。
    在 collect_node 里调用，每轮并发结束后统一检查一次。
    """
    for task in task_plan:
        if task["status"] == "done" and _is_task_failed(task.get("result", "")):
            # 只对"真正失败"（非已级联跳过标记）的任务触发
            if not task["result"].startswith("⚠️ 跳过："):
                print(f"  ⚠️ Re-Planner：任务[{task['task_id']}] 失败，触发级联跳过")
                _cascade_skip(task_plan, task["task_id"])


# ══════════════════════════════════════════════════════
# 10. Collect 汇聚节点
#
#   ★ 新版改动A：并发架构新增节点。
#   所有并发 worker 完成后，结果通过 Reducer 归并到主 AgentState，
#   然后流入 collect_node。
#   collect_node 职责：
#     1. 调用失败检测器（替代原 replanner）
#     2. 触发下一轮 supervisor_dispatch（条件边返回 list[Send]）
# ══════════════════════════════════════════════════════

def collect_node(state: AgentState) -> AgentState:
    """
    并发汇聚点。所有本轮并发 worker 完成后在此汇合。
    只做失败检测，不调 LLM，然后把控制权交回 supervisor_dispatch。
    """
    task_plan = state.get("task_plan", [])

    done_count    = sum(1 for t in task_plan if t["status"] == "done")
    pending_count = sum(1 for t in task_plan if t["status"] == "pending")
    print(f"\n  📥 collect_node：已完成 {done_count} 个，待执行 {pending_count} 个")

    # ★ 新版改动C：失败级联检测（替代原 replanner 的 LLM 调用）
    _check_and_cascade_failures(task_plan)

    return {**state, "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 11. 通用工具 Agent 执行器
#
#   接收 WorkerState，完成任务后返回含更新后 task_plan 的 dict，
#   由 Reducer 安全合并回主 AgentState。
# ══════════════════════════════════════════════════════

async def run_agent(
    state: WorkerState,
    agent_name: str,
    system_prompt: str,
) -> dict:
    """
    ★ 新版改动A：入参改为 WorkerState，出参只返回 task_plan（含单任务更新），
    Reducer 负责把这个局部更新合并回主 AgentState，不影响其他并发 worker。
    """
    lc_tools   = _registry.tools_for(agent_name)
    tool_names = _registry.tool_names_for(agent_name)
    by_name    = {t.name: t for t in lc_tools}

    task_plan: list[Task] = state.get("task_plan", [])
    task_id: int          = state.get("current_task_id", 0)

    current_task = next((t for t in task_plan if t["task_id"] == task_id), None)
    if not current_task:
        print(f"  ⚠️ [{agent_name}] 找不到任务 task_id={task_id}")
        return {"task_plan": task_plan}

    task_description = current_task.get("_resolved_description") or current_task["description"]

    if not lc_tools:
        current_task["status"] = "done"
        current_task["result"] = f"⚠️ 没有可用工具（{agent_name} 未注册任何工具）"
        return {"task_plan": task_plan}

    tool_hint  = "可用工具：" + "、".join(
        f"{t.name}（{t.description or '无描述'}）" for t in lc_tools
    )
    # 修复8：在 system prompt 中明确禁止不调工具直接回答
    full_system = (
        f"{system_prompt}\n\n"
        f"{tool_hint}\n\n"
        "【重要约束】你必须通过调用工具来完成任务，严禁凭记忆、推理或编造直接给出答案。"
        "即使你认为自己知道答案，也必须先调用工具获取真实结果，再基于工具返回值回答。"
    )
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

        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

        # 修复8：第一轮没有 tool_calls，追加强制提示再重试一次
        if step == 0 and not has_tool_calls:
            tool_list_str = "、".join(tool_names)
            force_msg = (
                f"你刚才没有调用任何工具就直接回答了，这是不允许的。\n"
                f"你拥有以下工具：{tool_list_str}\n"
                f"请立即调用对应工具来完成任务，不要直接给出文字答案。"
            )
            print(f"  ⚠️ {agent_name} 第1轮未调用工具，追加强制提示重试...")
            msgs.append(HumanMessage(content=force_msg))
            continue

        if not has_tool_calls:
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
    current_task["result"] = _extract_llm_content(last_response) if last_response else "（无结果）"

    # ★ 只返回 task_plan，Reducer 会把这个局部更新安全合并回主 state
    return {"task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 12. 图构建
#
#   新图结构：
#
#   planner ──(FINISH)──────────────────────────→ END
#           ──(ok)──→ supervisor_dispatch
#                           │ list[Send]
#               ┌───────────┼───────────┐
#               ▼           ▼           ▼
#          math_agent   http_agent  direct_answer  ...（并发）
#               │           │           │
#               └─────┬─────┘───────────┘
#                     ▼
#               collect_node              ← 汇聚 + 失败检测
#                     │ list[Send]
#               ┌─────┴──────┐
#               ▼            ▼
#     supervisor_dispatch  final_answer  ← 下一轮或收尾
#                                 │
#                                 ▼
#                                END
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

    # ── direct_answer 节点 ──────────────────────────
    # 修复2：只传当前子任务的 intent 给 LLM，不传完整原始消息
    # ★ 新版改动A：入参改为 WorkerState
    async def direct_answer_node(state: WorkerState) -> dict:
        task_plan: list[Task] = state.get("task_plan", [])
        task_id: int          = state.get("current_task_id", 0)
        current_task = next((t for t in task_plan if t["task_id"] == task_id), None)

        if not current_task:
            print("  ⚠️ direct_answer：找不到当前任务")
            return {"task_plan": task_plan}

        intent = current_task.get("description", "")
        print(f"\n  💬 direct_answer 调用 LLM（意图：{intent[:60]}）")

        response = await llm.ainvoke([
            SystemMessage(content=(
                "你是一个友善的 AI 助手。请只回答当前分配给你的这一个子任务，"
                "不要回答用户原始消息中的其他问题。"
            )),
            HumanMessage(content=intent),
        ])

        answer = _extract_llm_content(response)
        print(f"  ✅ direct_answer 完成：{answer[:80]}")

        current_task["status"] = "done"
        current_task["result"] = answer

        # ★ 新版改动A：只返回 task_plan，messages 追加由 final_answer 统一处理
        return {"task_plan": task_plan}

    # ── 最终汇总节点 ────────────────────────────────
    async def final_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])

        tool_tasks   = [t for t in task_plan if t.get("agent") != "direct"]
        direct_tasks = [t for t in task_plan if t.get("agent") == "direct"]

        if not tool_tasks:
            # 全是 direct 任务，直接从 task_plan 里拿结果拼成回答
            print("\n  📝 所有任务均为 direct，汇总 direct 结果")
            combined = "\n\n".join(
                t["result"] for t in direct_tasks if t.get("result")
            )
            if combined:
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content=combined)],
                }
            return state

        # 修复6：混合任务时把 direct 结果也纳入汇总上下文
        all_results_lines: list[str] = []
        if direct_tasks:
            all_results_lines.append("【直接回答任务】")
            for t in direct_tasks:
                all_results_lines.append(
                    f"  任务[{t['task_id']}]（{t['description']}）：{t['result']}"
                )
        all_results_lines.append("【工具执行任务】")
        for t in tool_tasks:
            all_results_lines.append(
                f"  任务[{t['task_id']}]（{t['description']}）：{t['result']}"
            )

        results_text = "\n".join(all_results_lines)
        print(f"\n  📝 汇总所有任务结果：\n{results_text}")

        response = await llm.ainvoke([
            SystemMessage(content=(
                "根据以下各子任务的执行结果，用中文给用户一个清晰完整的最终答案。\n\n"
                f"{results_text}"
            )),
            _get_first_user_message(state),
        ])
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=_extract_llm_content(response))
            ],
        }

    # ── 路由函数 ────────────────────────────────────

    def planner_route(state: AgentState) -> str:
        """planner 后的条件路由：规划失败直接到 END，否则进入并行调度。"""
        if state.get("next_agent") == "FINISH":
            print("  ⛔ Planner 规划失败，直接终止")
            return "END"
        return "supervisor_dispatch"

    # collect_node 完成后再次调用 supervisor_dispatch（条件边）
    # supervisor_dispatch 返回 list[Send]，LangGraph 自动并发
    # 若返回 Send("final_answer", ...) 则走向收尾
    def collect_route(state: AgentState) -> list[Send]:
        """collect 后触发下一轮并行调度。"""
        return supervisor_dispatch(state)

    # ── 动态创建工具 agent 节点 ──────────────────────
    def make_agent_node(name: str):
        system_prompt = AGENT_SYSTEM_PROMPTS.get(name, DEFAULT_AGENT_SYSTEM_PROMPT)
        async def _node(state: WorkerState) -> dict:
            return await run_agent(state, name, system_prompt)
        _node.__name__ = name
        return _node

    # ── 图构建 ──────────────────────────────────────
    g = StateGraph(AgentState)

    g.add_node("planner",       planner_node)
    g.add_node("collect",       collect_node)
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("final_answer",  final_answer_node)

    known_agents = _registry.agents or list(AGENT_SYSTEM_PROMPTS.keys())
    for agent_name in known_agents:
        g.add_node(agent_name, make_agent_node(agent_name))

    # ── 边 ──────────────────────────────────────────
    g.set_entry_point("planner")

    # planner → END（规划失败）或 supervisor_dispatch（正常）
    # supervisor_dispatch 是条件边函数，返回 list[Send] 触发并发
    g.add_conditional_edges("planner", planner_route, {
        "END":               END,
        "supervisor_dispatch": "supervisor_dispatch_node",
    })

    # ★ supervisor_dispatch 作为条件边函数挂在虚节点上
    #   通过一个透传节点 + 条件边实现"节点 → Send 并发"
    async def _supervisor_dispatch_node(state: AgentState) -> AgentState:
        """透传节点，实际分发逻辑在条件边 supervisor_dispatch 函数里。"""
        return state

    g.add_node("supervisor_dispatch_node", _supervisor_dispatch_node)
    g.add_conditional_edges(
        "supervisor_dispatch_node",
        supervisor_dispatch,           # 返回 list[Send]，LangGraph 自动并发
        # Send 的目标节点集合（包含所有可能的 target）
        {name: name for name in [*known_agents, "direct_answer", "final_answer"]},
    )

    # 所有 worker（工具 agent + direct_answer）完成后汇入 collect
    for agent_name in known_agents:
        g.add_edge(agent_name, "collect")
    g.add_edge("direct_answer", "collect")

    # collect → 下一轮 supervisor_dispatch（条件边，返回 list[Send]）
    g.add_conditional_edges(
        "collect",
        collect_route,
        {name: name for name in [*known_agents, "direct_answer", "final_answer"]},
    )

    g.add_edge("final_answer", END)

    return g.compile()


# ══════════════════════════════════════════════════════
# 13. 图实例（延迟初始化）
# ══════════════════════════════════════════════════════
graph = build_graph()


def _rebuild_graph() -> None:
    global graph
    graph = build_graph()
    print("🔄 Graph 已用真实 ToolRegistry 重建")


# ══════════════════════════════════════════════════════
# 14. lifespan —— 仅 langgraph dev 调用
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
# 15. __main__ —— 单独运行测试
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    if not SERVER_PATH.exists():
        print(f"❌ 找不到 server.py：{SERVER_PATH}")
        sys.exit(1)

    QUESTIONS = [
        # ── 纯工具任务（3个独立任务并行，耗时≈最慢的那个）
        "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",

        # ── 混合任务：闲聊 + 工具
        # "先介绍一下你自己，然后帮我计算 99 乘以 9",

        # ── 纯闲聊
        # "你好，我叫 tony",

        # ── 数据分析
        # """分析这批数据：[{"name":"Alice","dept":"Eng","salary":9000},
        #  {"name":"Bob","dept":"Mkt","salary":7500},
        #  {"name":"Charlie","dept":"Eng","salary":11000}]
        #  按 dept 分组，对 salary 求平均""",

        # ── 综合测试（直接任务 + 并行工具 + 串行依赖）
        # "先介绍一下什么是加权平均数，然后计算 (85×3 + 90×2 + 78×5) 除以 (3+2+5) 得到加权平均分，"
        # "同时访问 https://api.github.com/zen 获取一句话，"
        # """最后分析这批学生数据：[{"name":"Alice","score":85,"weight":3},
        # {"name":"Bob","score":90,"weight":2},{"name":"Charlie","score":78,"weight":5}]
        # 按 weight 分组对 score 求平均，把网络请求的结果也附在最终答案里""",
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
                        "messages":   [HumanMessage(content=q)],
                        "task_plan":  [],
                        "next_agent": "",
                    })
                    print(f"\n✨ 最终答案：{_get_message_content(result['messages'][-1])}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())

    # uv run python src/langgraph_stdio_agent.py