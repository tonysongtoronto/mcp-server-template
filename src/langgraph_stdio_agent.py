"""
src/langgraph_stdio_agent.py

两种运行方式：
  1. uv run langgraph dev   → langgraph 调用 lifespan 钩子初始化工具
  2. python -m src.langgraph_stdio_agent  → __main__ 手动初始化工具

★ 本版本重构说明：

  【重构1】移除 router 节点
   - 原问题：router 对整个用户问题做一次性"需不需要工具"判断，
             混合任务（部分闲聊+部分工具）会被短路到 direct_answer 直接结束
   - 方案：Planner 承担意图判断，纯问答输出 agent="direct" 特殊任务，
           Supervisor 检测到 direct 时路由到 direct_answer，
           direct_answer 完成后回到 supervisor 继续处理后续工具任务

  【重构2】direct_answer_node 只回答当前子任务
   - 原问题：传入完整原始消息，LLM 看到全部问题后把所有子任务都答了一遍，
             导致重复 AI 回复
   - 方案：只传 intent（当前子任务描述）给 LLM，不传原始用户消息

  【重构3】planner_node JSON 解析失败重试策略
   - 原问题：解析失败直接兜底为一个 direct 任务，掩盖错误
   - 方案：最多重试 3 次（每次附上上次错误输出和更强的格式提示）
           3 次全部失败 → 在 planner 内部闭环处理：
             · 把带有详细失败信息的 AIMessage 写入 messages
             · 设置 next_agent="FINISH"，通过条件路由直接走向 END
             · 不进入 supervisor，不走任何 agent

  【修复4】supervisor_node in_progress 容错
   - 原问题：direct 任务标为 in_progress 后若 direct_answer_node 异常，
             任务永远卡住，流程死锁
   - 方案：supervisor 遍历时同时处理 pending 和 in_progress 状态

  【修复5】_extract_json 统一抽取，replanner 缺 .strip() 问题
   - 原问题：replanner 的代码块清理逻辑不完整，缺少 strip()，
             容易导致 JSON 解析失败
   - 方案：抽取公共函数 _extract_json，planner/replanner 统一调用

  【修复6】final_answer_node 混合任务时 direct 结果纳入汇总
   - 原问题：只汇总工具任务结果，direct 任务的回答被丢弃，
             最终答案缺失 direct 部分内容
   - 方案：把 direct 任务结果也拼入 results_text 一起发给 LLM

  【修复7】强化 Planner 工具感知 —— 杜绝"有工具却用 direct"
   - 原问题：Planner 把"计算""访问URL"等明确需要工具的任务误判为 direct，
             导致 LLM 编造结果
   - 方案：
       · Planner 系统提示新增"禁止使用 direct"的负面示例和强约束
       · 在 Planner 提示中列出每个 agent 对应的具体工具名和场景
       · Planner 规划后增加一次"合规校验"：
           检查每个任务的 agent 选择是否合理，
           发现 direct 任务包含数值计算/URL访问等关键词时自动修正
       · 工具 Agent 系统提示强化"必须调用工具"的指令，
           明确禁止凭记忆或推理直接回答，必须通过工具获取结果

  【修复8】run_agent 工具调用强制兜底
   - 原问题：LLM 有时第一轮直接给出文字答案而不调用工具
   - 方案：若第一轮没有 tool_calls，追加一条强制提示再 invoke 一次
  【修복10】兼容 LangGraph Studio 下 messages 反序列化为 dict 的问题
   - 原问题：LangGraph Studio 通过 API 传入 state 时，messages 被反序列化为
             原始 dict（{"role":"human","content":"..."}），而非 HumanMessage 对象，
             访问 state["messages"][0].content 抛出 AttributeError
   - 方案：新增 _get_message_content(msg) 和 _get_first_user_message(state) 两个函数，
           统一兼容 HumanMessage 对象和 dict 两种格式，
           所有访问 messages[N].content 的地方都通过这两个函数处理

  【修复9】全局兼容 llm.ainvoke() 返回 dict 的问题
   - 原问题：DeepSeek + langchain_openai 在部分版本下 ainvoke() 返回原始 dict
             而非 AIMessage 对象，直接访问 .content 抛出
             AttributeError: 'dict' object has no attribute 'content'
             错误发生在 replanner 节点，导致流程中断
   - 方案：抽取 _extract_llm_content(response) 公共函数，
           兼容 AIMessage / dict / 其他类型三种情况，
           所有 llm.ainvoke() 调用后统一通过此函数取内容
"""

import asyncio
import json
import os
import re
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

# ★ 修复7：每个 agent 的"触发关键词"，用于 Planner 校验兜底
AGENT_TRIGGER_KEYWORDS: dict[str, list[str]] = {
    "math_agent": ["计算", "加", "减", "乘", "除", "求和", "平均", "幂", "开方",
                   "+", "-", "×", "÷", "*", "/", "²", "√"],
    "http_agent": ["访问", "请求", "获取", "http", "https", "url", "api",
                   "fetch", "get ", "post "],
    "data_agent": ["分析", "统计", "分组", "聚合", "过滤", "排序", "数据集",
                   "dataframe", "pivot"],
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
    _agent_desc_brief: str = ""   # 给失败提示用的简短版本

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
        lines_agents.append("  - direct：直接用语言模型回答，不调用任何工具（仅限：闲聊/问候/概念解释/知识性问答）")

        reg._tool_desc_block  = "\n".join(lines_tools)
        reg._agent_desc_block = "\n".join(lines_agents)
        reg._agent_desc_brief = "\n".join(brief_lines) if brief_lines else "  （暂无已注册的工具 Agent）"

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

class AgentState(TypedDict):
    messages: list
    task_plan: list[Task]
    current_task_id: int
    next_agent: str


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
    ★ 修复5：统一的 JSON 代码块清理函数。
    去除 ```json ... ``` 或 ``` ... ``` 包裹，并 strip 空白。
    planner / replanner 统一调用此函数，避免各处逻辑不一致。
    """
    raw = raw.strip()
    if "```" in raw:
        # 取第一对 ``` 之间的内容
        parts = raw.split("```")
        # parts[0]=前缀, parts[1]=代码块内容, parts[2]=后缀（如果有）
        inner = parts[1] if len(parts) > 1 else parts[0]
        # 去掉语言标识符（如 "json\n"）
        inner = re.sub(r"^[a-zA-Z]+\n", "", inner)
        return inner.strip()
    return raw


def _extract_llm_content(response: Any) -> str:
    """
    ★ 修复9：兼容 llm.ainvoke() 返回 AIMessage 对象或原始 dict 两种情况。
    DeepSeek + langchain_openai 在部分版本/流式配置下会返回 dict 而非 AIMessage，
    直接访问 .content 会抛出 AttributeError: 'dict' object has no attribute 'content'。
    所有 llm.ainvoke() 调用后都应通过此函数取内容，不要直接用 response.content。
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
    ★ 修复10：兼容 LangGraph Studio 环境下 messages 反序列化为 dict 的情况。
    终端直接运行时 messages 是 HumanMessage/AIMessage 对象（有 .content 属性），
    但 LangGraph Studio 通过 API 传入时，messages 会被反序列化为原始 dict：
      {"role": "human", "content": "..."}
    直接访问 .content 会抛出 AttributeError。
    所有访问 state["messages"][N] 内容的地方都应使用此函数。
    """
    if hasattr(msg, "content"):
        return msg.content or ""
    if isinstance(msg, dict):
        return msg.get("content") or msg.get("text") or str(msg)
    return str(msg)


def _get_first_user_message(state: "AgentState") -> Any:
    """
    从 state["messages"] 中取第一条用户消息，
    返回原始对象（可能是 HumanMessage 也可能是 dict），
    供直接传入 llm.ainvoke() 使用（LangChain 两种格式都能处理）。
    """
    msgs = state.get("messages", [])
    if not msgs:
        return HumanMessage(content="")
    msg = msgs[0]
    # 如果是 dict，转成 HumanMessage，保证 LangChain 可以正确处理
    if isinstance(msg, dict):
        return HumanMessage(content=msg.get("content") or msg.get("text") or "")
    return msg


def _task_needs_tool_agent(description: str) -> str | None:
    """
    ★ 修复7：校验任务描述是否应当使用工具 agent 而非 direct。
    返回推荐的 agent 名，或 None 表示无需修正。
    仅在已注册该 agent 的情况下才修正。
    """
    desc_lower = description.lower()
    for agent, keywords in AGENT_TRIGGER_KEYWORDS.items():
        if agent not in _registry.agents:
            continue
        if any(kw.lower() in desc_lower for kw in keywords):
            return agent
    return None


# ══════════════════════════════════════════════════════
# 7. Planner
#
#   ★ 重构1：承担意图判断，支持 agent="direct" 特殊任务
#   ★ 重构3：JSON 解析失败最多重试 3 次，全部失败则在内部闭环处理
#   ★ 修复7：强化工具选择约束，增加负面示例和合规校验
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
   - value = {{"from_task": <被依赖的task_id>, "field": "result"}}
3. depends_on 从 inputs 的 from_task 自动推导
4. 没有依赖的任务：inputs 为 {{}}，depends_on 为 []
5. 同一个 agent 可出现多次
6. 任务按拓扑顺序排列（被依赖的任务排在前面）
7. 如果用户消息中已直接包含数据（JSON 数组、数字、文本等），
   不要单独拆"获取数据"任务，直接在处理任务的 description 里完整引用

严格只输出 JSON 数组，不要有任何其他内容、代码块标记或说明文字。示例：
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
        # 第二次重试：极简提示，降低 LLM 发挥空间
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


def _validate_and_fix_task_plan(task_plan: list[Task]) -> list[Task]:
    """
    ★ 修复7：Planner 合规校验 + 自动修正。
    检查每个 direct 任务的描述，如果描述中包含工具关键词，
    自动把 agent 修正为对应的工具 agent。
    """
    fixed_count = 0
    for task in task_plan:
        if task.get("agent") != "direct":
            continue
        description = task.get("description", "")
        recommended = _task_needs_tool_agent(description)
        if recommended:
            print(f"  🔧 [合规校验] 任务[{task['task_id']}] agent: direct → {recommended}")
            print(f"     原因：描述含工具关键词，描述={description[:60]}")
            task["agent"] = recommended
            fixed_count += 1
    if fixed_count:
        print(f"  ⚠️ 合规校验共修正 {fixed_count} 个任务的 agent 分配")
    return task_plan


async def planner_node(state: AgentState) -> AgentState:
    if state.get("task_plan"):
        return state

    print("\n  📋 Planner 开始拆解任务...")

    # ★ 修复10：兼容 Studio dict 格式和终端 HumanMessage 对象
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
        # ★ 修复9：兼容 dict / AIMessage 两种返回格式
        raw = _extract_llm_content(response).strip()

        # ★ 修复5：使用统一的 _extract_json 清理代码块标记
        raw = _extract_json(raw)
        last_raw = raw

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise json.JSONDecodeError("期望 JSON 数组", raw, 0)
            task_plan = parsed
            print(f"  ✅ 拆解出 {len(task_plan)} 个任务（第 {attempt + 1} 次尝试成功）：")
            for t in task_plan:
                deps        = t.get("depends_on", [])
                inputs      = t.get("inputs", {})
                dep_str     = f"  依赖→{deps} 参数→{list(inputs.keys())}" if deps else ""
                desc_preview = (
                    t["description"][:60] + "…"
                    if len(t["description"]) > 60
                    else t["description"]
                )
                print(f"     [{t['task_id']}] {t['agent']} ← {desc_preview}{dep_str}")
            break  # 解析成功，跳出重试循环

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠️ 第 {attempt + 1} 次解析失败：{e}  输出片段：{raw[:100]}")
            # 继续下一次重试

    # ★ 三次全部失败：在 planner 内部闭环，直接告知用户
    if task_plan is None:
        print("  ❌ Planner 连续 3 次解析失败，终止流程")

        # 构造详细的失败信息
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
            "messages":        state["messages"] + [AIMessage(content=failure_msg)],
            "task_plan":       [],
            "current_task_id": 0,
            "next_agent":      "FINISH",   # ★ 触发 planner → END 的条件路由
        }

    # ★ 修复7：对规划结果做合规校验，自动修正误判为 direct 的工具任务
    task_plan = _validate_and_fix_task_plan(task_plan)

    return {
        **state,
        "task_plan":       task_plan,
        "current_task_id": 0,
        "next_agent":      "",
    }


# ══════════════════════════════════════════════════════
# 8. Supervisor
# ══════════════════════════════════════════════════════
async def supervisor_node(state: AgentState) -> AgentState:
    task_plan: list[Task] = state.get("task_plan", [])
    done_map = {t["task_id"]: t for t in task_plan if t["status"] == "done"}

    for task in task_plan:
        # ★ 修复4：同时处理 pending 和 in_progress，防止异常后流程卡死
        if task["status"] not in ("pending", "in_progress"):
            continue

        unmet = [dep for dep in task.get("depends_on", []) if dep not in done_map]
        if unmet:
            print(f"\n  ⏳ 任务[{task['task_id']}] 等待依赖 {unmet}，跳过")
            continue

        # ★ direct 任务：supervisor 直接标记完成，next_agent → direct_answer
        #   result 此处留空，由 direct_answer_node 调用 LLM 生成真实回答后填入
        if task.get("agent") == "direct":
            print(f"\n  🧭 Supervisor → direct_answer（任务[{task['task_id']}]）")
            print(f"     意图：{task['description'][:80]}")
            task["status"] = "in_progress"   # 标为进行中，由 direct_answer_node 完成
            return {
                **state,
                "next_agent":      "direct_answer",
                "current_task_id": task["task_id"],
                "task_plan":       task_plan,
            }

        # 工具任务：解析运行时参数
        resolved_inputs: dict[str, str] = {}
        for param_name, source in task.get("inputs", {}).items():
            from_id  = source["from_task"]
            src_field = source.get("field", "result")
            src_task = done_map.get(from_id, {})
            resolved_inputs[param_name] = src_task.get(src_field, "")

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
            "next_agent":      task["agent"],
            "current_task_id": task["task_id"],
            "task_plan":       task_plan,
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
# 9. Re-Planner（仅工具 agent 执行后触发）
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
    task_plan: list[Task] = state.get("task_plan", [])
    current_task_id: int  = state.get("current_task_id", -1)

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
            # ★ 修复10：兼容 Studio dict 格式
            f"用户原始问题：{_get_message_content(state['messages'][0])}\n\n"
            f"已完成任务：\n{done_summary}\n\n"
            f"待执行任务（pending）：\n{pending_summary}"
        )),
    ])

    # ★ 修复5+9：使用统一的 _extract_json，并兼容 dict/AIMessage 返回格式
    raw = _extract_json(_extract_llm_content(response))

    try:
        decision = json.loads(raw)
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
        new_pending  = [t for t in new_pending if t.get("agent") in valid_agents]
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
    # ★ 修复8：在 system prompt 中明确禁止不调工具直接回答
    full_system = (
        f"{system_prompt}\n\n"
        f"{tool_hint}\n\n"
        "【重要约束】你必须通过调用工具来完成任务，严禁凭记忆、推理或编造直接给出答案。"
        "即使你认为自己知道答案，也必须先调用工具获取真实结果，再基于工具返回值回答。"
    )
    agent_llm   = llm.bind_tools(lc_tools)
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

        # ★ 修复8：第一轮没有 tool_calls，追加强制提示再重试一次
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
    # ★ 修复9：兼容 dict / AIMessage 两种返回格式
    current_task["result"] = _extract_llm_content(last_response) if last_response else "（无结果）"

    return {**state, "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 11. 图构建
#
#   图结构：
#
#   planner ──(next_agent=FINISH)──→ END          ← ★ 规划失败短路
#           ──(otherwise)──────────→ supervisor
#
#   supervisor ──→ direct_answer ──→ supervisor   ← ★ direct 任务回环
#              ──→ math_agent   ──→ replanner ──→ supervisor
#              ──→ data_agent   ──→ replanner ──→ supervisor
#              ──→ http_agent   ──→ replanner ──→ supervisor
#              ──→ final_answer ──→ END
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

    # ── ★ direct_answer 节点 ─────────────────────────
    # ★ 修复2：只传当前子任务的 intent 给 LLM，不传完整原始消息
    async def direct_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])
        task_id: int          = state.get("current_task_id", 0)
        current_task = next((t for t in task_plan if t["task_id"] == task_id), None)

        if not current_task:
            print("  ⚠️ direct_answer：找不到当前任务")
            return state

        intent = current_task.get("description", "")
        print(f"\n  💬 direct_answer 调用 LLM（意图：{intent[:60]}）")

        # ★ 修复2：只传 intent 作为用户消息，不传原始完整问题
        #   原来传 state["messages"][0] 导致 LLM 看到全部问题，把所有子任务都答了
        response = await llm.ainvoke([
            SystemMessage(content=(
                "你是一个友善的 AI 助手。请只回答当前分配给你的这一个子任务，"
                "不要回答用户原始消息中的其他问题。"
            )),
            HumanMessage(content=intent),
        ])

        # ★ 修复9：兼容 dict / AIMessage 两种返回格式
        answer = _extract_llm_content(response)
        print(f"  ✅ direct_answer 完成：{answer[:80]}")

        # 标记任务完成，result 存储 LLM 真实回答
        current_task["status"] = "done"
        current_task["result"] = answer

        # 把回答追加到消息流，然后回到 supervisor 处理后续任务
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=answer)],
            "task_plan": task_plan,
        }

    # ── 最终汇总节点 ──────────────────────────────────
    async def final_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])

        tool_tasks   = [t for t in task_plan if t.get("agent") != "direct"]
        direct_tasks = [t for t in task_plan if t.get("agent") == "direct"]

        if not tool_tasks:
            # 全是 direct 任务，消息流里已有完整回答，无需再调 LLM
            print("\n  📝 所有任务均为 direct，最终回答已在消息流中")
            return state

        # ★ 修复6：混合任务时把 direct 结果也纳入汇总上下文
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
            # ★ 修复10：兼容 Studio dict 格式，转为 HumanMessage 再传给 LLM
            _get_first_user_message(state),
        ])
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=_extract_llm_content(response))],
        }

    # ── 路由函数 ──────────────────────────────────────

    # ★ planner 后的条件路由：规划失败（next_agent=FINISH）直接到 END
    def planner_route(state: AgentState) -> str:
        if state.get("next_agent") == "FINISH":
            print("  ⛔ Planner 规划失败，直接终止，不进入 Supervisor")
            return "END"
        return "supervisor"

    def supervisor_route(state: AgentState) -> str:
        next_node = state.get("next_agent", "FINISH")
        if next_node == "FINISH":
            return "final_answer"
        return next_node  # "direct_answer" 或工具 agent 名

    # ── 动态创建工具 agent 节点 ───────────────────────
    def make_agent_node(name: str):
        system_prompt = AGENT_SYSTEM_PROMPTS.get(name, DEFAULT_AGENT_SYSTEM_PROMPT)
        async def _node(state: AgentState) -> AgentState:
            return await run_agent(state, name, system_prompt)
        _node.__name__ = name
        return _node

    # ── 图构建 ─────────────────────────────────────────
    g = StateGraph(AgentState)

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

    # ★ planner → END（规划失败）或 supervisor（正常）
    g.add_conditional_edges("planner", planner_route, {
        "END":        END,
        "supervisor": "supervisor",
    })

    # supervisor → direct_answer / 工具 agent / final_answer
    agent_routes = {name: name for name in known_agents}
    agent_routes["direct_answer"] = "direct_answer"
    agent_routes["final_answer"]  = "final_answer"
    g.add_conditional_edges("supervisor", supervisor_route, agent_routes)

    # ★ direct_answer 完成后回到 supervisor，继续处理后续工具任务
    g.add_edge("direct_answer", "supervisor")

    # 工具 agent → replanner → supervisor
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
        # 纯闲聊 → direct 任务（LLM 真实回答）
        # "你好，我叫 tony",
        # 纯工具任务
        "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",
        # 混合任务：闲聊 + 工具（原来会被短路，现在能正确执行全部子任务）
        # "先介绍一下你自己，然后帮我计算 99 乘以 9",
        # 数据分析
        # """分析这批数据：[{"name":"Alice","dept":"Eng","salary":9000},
        #  {"name":"Bob","dept":"Mkt","salary":7500},
        #  {"name":"Charlie","dept":"Eng","salary":11000}]
        #  按 dept 分组，对 salary 求平均""",
        # # 网络请求
        # "访问 https://api.github.com/zen 返回了什么？",
#         "先介绍一下什么是加权平均数，然后计算 (85×3 + 90×2 + 78×5) 除以 (3+2+5) 得到加权平均分，"
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
                        "messages":        [HumanMessage(content=q)],
                        "task_plan":       [],
                        "current_task_id": 0,
                        "next_agent":      "",
                    })
                    print(f"\n✨ 最终答案：{_get_message_content(result['messages'][-1])}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())


    # uv run python src/langgraph_stdio_agent.py