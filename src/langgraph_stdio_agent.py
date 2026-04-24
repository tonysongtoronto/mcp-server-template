"""
src/langgraph_stdio_agent.py

两种运行方式：
  1. uv run langgraph dev   → langgraph 调用 lifespan 钩子初始化工具
  2. python -m src.langgraph_stdio_agent  → __main__ 手动初始化工具

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
★ 本版本重构说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  【改动 A】并行任务调度（Send API + Map-Reduce）
    · AgentState.task_plan 加 Annotated Reducer，安全合并并发写入
    · 新增 WorkerState，每个并发 worker 持有私有执行上下文
    · supervisor_dispatch 找出本轮所有依赖已满足的 pending 任务，
      用 Send API 一次性全部并发分发
    · collect_node 作为并发汇聚点

  【改动 B】删除 _validate_and_fix_task_plan
    · 依赖 Planner 系统提示的强约束保证 agent 选择正确性

  【改动 C】replanner 改为纯逻辑失败检测器
    · 移除 LLM 调用，改为纯逻辑 BFS 级联跳过失败依赖任务

  【修复 D】解决 LangGraph Studio 前台工具失效（最终版）
    根因：
      ① stdio_client 的 asyncio stream 与创建它的 task 绑定，
         Studio 每次 HTTP 请求在新 task 里调用 graph，跨 task 使用
         session 导致工具失效
      ② filesystem MCP 启动异常被静默吞掉，工具缺失无报错
    方案：
      · MCPSessionManager：所有 MCP 操作在专属后台 task 里执行，
        外部通过 asyncio.Queue + Future 桥接，彻底绕开跨 task 限制
      · _init_registry() set _registry_ready event；
        run_agent 若发现 registry 为空则等待（最多60s），解决竞态
      · lifespan 不再调用 _rebuild_graph()，graph 节点运行时
        动态查全局 _registry
      · filesystem 启动失败时打印完整错误，而非静默跳过

  ★★★ langgraph.json 必须配置 lifespan ★★★
  {
    "graphs": {
      "supervisorpalner": "./src/langgraph_stdio_agent.py:graph"
    },
    "lifespan": "src.langgraph_stdio_agent:lifespan"
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  图结构：

  planner ──(FINISH)──→ END
          ──(ok)──────→ supervisor_dispatch_node
                              │ list[Send]
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
               math_agent  file_agent  direct_answer  ...
                    │         │         │
                    └────┬────┘─────────┘
                         ▼
                    collect_node
                         │ list[Send]
                    ┌────┴────┐
                    ▼         ▼
             (下一轮)   final_answer ──→ END
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import json
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

_FS_BASE_DIR = Path(os.getenv("MCP_FS_BASE_DIR", "./File Agent")).resolve()

def filesystem_mcp_params() -> StdioServerParameters:
    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    print(f"📂 Filesystem MCP BASE_DIR: {_FS_BASE_DIR}", file=sys.stderr)
    return StdioServerParameters(
        command=npx_cmd,
        args=["-y", "@modelcontextprotocol/server-filesystem", str(_FS_BASE_DIR)],
        env={**os.environ},
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
    "file_agent": ["read_file", "write_file", "edit_file",
                   "read_multiple_files", "list_directory", "create_directory",
                   "move_file", "search_files", "get_file_info",
                   "list_allowed_directories", "file_*"],
}

AGENT_DESCRIPTIONS: dict[str, str] = {
    "math_agent": "数学计算（加减乘除、幂、开方等数值运算）",
    "data_agent": "数据分析（统计、聚合、分组、过滤等结构化数据处理）",
    "http_agent": "网络请求（GET/POST、访问 URL、调用外部 API）",
    "file_agent": "文件操作（读写文件、列出目录、创建目录、移动/搜索文件）",
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
        brief_lines: list[str] = []
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
    merged = {t["task_id"]: t for t in old}
    for t in new:
        merged[t["task_id"]] = t
    return sorted(merged.values(), key=lambda t: t["task_id"])


class AgentState(TypedDict):
    messages:   list
    task_plan:  Annotated[list[Task], _merge_task_plan]
    next_agent: str


class WorkerState(TypedDict):
    task_plan:       list[Task]
    current_task_id: int
    messages:        list


# ══════════════════════════════════════════════════════
# 5. 共享容器 & registry ready event
# ══════════════════════════════════════════════════════
_tools: list[StructuredTool] = []
_registry: ToolRegistry = ToolRegistry()

# 延迟创建，避免模块导入时无 event loop
_registry_ready: asyncio.Event | None = None

def _get_registry_ready_event() -> asyncio.Event:
    global _registry_ready
    if _registry_ready is None:
        _registry_ready = asyncio.Event()
    return _registry_ready


# ══════════════════════════════════════════════════════
# 6. MCPSessionManager（修复D核心）
#
#   所有 MCP 操作在专属后台 task 里执行，外部通过
#   asyncio.Queue + Future 桥接，彻底避免跨 task 使用
#   asyncio stream 导致的工具失效问题。
#
#   队列消息格式：
#     ("list", None,           future)  → session.list_tools()
#     ("call", (name, kwargs), future)  → session.call_tool(name, kwargs)
#     None                              → 终止信号
# ══════════════════════════════════════════════════════

class MCPSessionManager:

    def __init__(self):
        self._req_queues:   dict[str, asyncio.Queue] = {}
        self._bg_tasks:     list[asyncio.Task]       = []
        self._ready_events: dict[str, asyncio.Event] = {}

    async def start(self) -> None:
        """启动所有后台 MCP 连接，等待全部就绪后返回。"""
        configs: dict[str, StdioServerParameters] = {
            "server":     mcp_params(),
            "filesystem": filesystem_mcp_params(),
        }
        for name, params in configs.items():
            self._req_queues[name]   = asyncio.Queue()
            self._ready_events[name] = asyncio.Event()
            task = asyncio.create_task(
                self._run_session(name, params),
                name=f"mcp-bg-{name}",
            )
            self._bg_tasks.append(task)

        for name, event in self._ready_events.items():
            try:
                await asyncio.wait_for(event.wait(), timeout=60)
                print(f"✅ [MCPSessionManager] '{name}' 后台 session 就绪")
            except asyncio.TimeoutError:
                raise RuntimeError(f"MCPSessionManager: '{name}' 启动超时（60s）")

    async def stop(self) -> None:
        """向所有后台 task 发送终止信号并等待退出。"""
        for q in self._req_queues.values():
            await q.put(None)
        for task in self._bg_tasks:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except Exception:
                task.cancel()
        print("🛑 [MCPSessionManager] 所有后台 session 已关闭")

    async def list_tools_for(self, session_name: str) -> Any:
        """跨 task 安全地获取工具列表。"""
        return await self._dispatch(session_name, "list", None)

    async def call_tool(self, session_name: str, tool_name: str, kwargs: dict) -> Any:
        """跨 task 安全的工具调用入口。"""
        return await self._dispatch(session_name, "call", (tool_name, kwargs))

    async def _dispatch(self, session_name: str, op: str, payload: Any) -> Any:
        q = self._req_queues.get(session_name)
        if q is None:
            raise RuntimeError(f"MCPSessionManager: 未知 session '{session_name}'")
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        q.put_nowait((op, payload, fut))
        result = await fut
        if isinstance(result, BaseException):
            raise result
        return result

    async def _run_session(self, name: str, params: StdioServerParameters) -> None:
        """
        后台专属 task：维持 stdio_client 生命周期，
        串行处理所有 list / call 请求。
        """
        req_q = self._req_queues[name]
        ready = self._ready_events[name]
        try:
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    ready.set()   # ★ session 完全就绪后才 set

                    while True:
                        item = await req_q.get()
                        if item is None:
                            break
                        op, payload, fut = item
                        if fut.cancelled():
                            continue
                        try:
                            if op == "list":
                                res = await session.list_tools()
                            elif op == "call":
                                tool_name, kwargs = payload
                                res = await session.call_tool(tool_name, kwargs)
                            else:
                                res = RuntimeError(f"未知操作：{op}")
                            fut.set_result(res)
                        except Exception as exc:
                            if not fut.done():
                                fut.set_exception(exc)

        except Exception as exc:
            print(f"❌ [MCPSessionManager] '{name}' session 异常：{exc}", file=sys.stderr)
            if not ready.is_set():
                ready.set()   # 避免 start() 永久阻塞
            # 排空队列，全部返回异常
            while not req_q.empty():
                item = req_q.get_nowait()
                if item is None:
                    break
                _, _, fut = item
                if not fut.done():
                    fut.set_exception(exc)


_mcp_manager = MCPSessionManager()


# ══════════════════════════════════════════════════════
# 7. 工具加载
# ══════════════════════════════════════════════════════
async def load_tools(
    session: "ClientSession | None" = None,
    session_name: str = "",
    use_manager: bool = False,
) -> list[StructuredTool]:
    """
    构建 StructuredTool 列表。

    use_manager=True  （langgraph dev 前台）：
      list_tools 和 call_tool 全部通过 MCPSessionManager 队列执行，
      工具闭包不持有任何 session 对象，跨 task 完全安全。

    use_manager=False （__main__ 直接运行）：
      直接使用传入的 session 对象，保持原有行为。
    """
    if use_manager:
        raw = (await _mcp_manager.list_tools_for(session_name)).tools
    else:
        assert session is not None
        raw = (await session.list_tools()).tools

    lc_tools: list[StructuredTool] = []
    for t in raw:
        schema   = t.inputSchema or {}
        required = set(schema.get("required", []))
        fields   = {
            name: (Any, ...) if name in required else (Optional[Any], None)
            for name in schema.get("properties", {})
        }
        DynSchema = create_model(f"{t.name}_schema", **fields) if fields else None
        tool_name = t.name

        if use_manager:
            _sname = session_name

            async def _call_via_manager(
                _name: str = tool_name, _sn: str = _sname, **kwargs
            ) -> str:
                print(f"    🔧 [MCP/{_sn}] {_name}({kwargs})")
                res  = await _mcp_manager.call_tool(_sn, _name, kwargs)
                text = res.content[0].text if res.content else "（无结果）"
                print(f"    ✅ {text[:200]}")
                return text

            lc_tools.append(StructuredTool.from_function(
                coroutine=_call_via_manager, name=t.name,
                description=t.description or "", args_schema=DynSchema,
            ))
        else:
            async def _call(_name: str = tool_name, **kwargs) -> str:
                print(f"    🔧 [MCP] {_name}({kwargs})")
                res  = await session.call_tool(_name, kwargs)  # type: ignore[union-attr]
                text = res.content[0].text if res.content else "（无结果）"
                print(f"    ✅ {text[:200]}")
                return text

            lc_tools.append(StructuredTool.from_function(
                coroutine=_call, name=t.name,
                description=t.description or "", args_schema=DynSchema,
            ))

    print(f"✅ [{session_name or 'direct'}] 已加载 {len(lc_tools)} 个工具："
          f"{[t.name for t in lc_tools]}")
    return lc_tools


def _init_registry(tools: list[StructuredTool]) -> None:
    global _registry
    _registry = ToolRegistry.build(tools)
    try:
        _get_registry_ready_event().set()
        print("✅ [registry] _registry_ready event 已触发")
    except RuntimeError:
        pass


# ══════════════════════════════════════════════════════
# 公共工具函数
# ══════════════════════════════════════════════════════
def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        inner = parts[1] if len(parts) > 1 else parts[0]
        inner = re.sub(r"^[a-zA-Z]+\n", "", inner)
        return inner.strip()
    return raw


def _extract_llm_content(response: Any) -> str:
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
    if hasattr(msg, "content"):
        return msg.content or ""
    if isinstance(msg, dict):
        return msg.get("content") or msg.get("text") or str(msg)
    return str(msg)


def _get_first_user_message(state: AgentState) -> Any:
    msgs = state.get("messages", [])
    if not msgs:
        return HumanMessage(content="")
    msg = msgs[0]
    if isinstance(msg, dict):
        return HumanMessage(content=msg.get("content") or msg.get("text") or "")
    return msg


# ══════════════════════════════════════════════════════
# 8. Planner
# ══════════════════════════════════════════════════════
def _planner_system() -> str:
    valid_agents = ", ".join(_registry.agents) if _registry.agents else "（无可用工具 Agent）"
    return f"""你是任务规划器。把用户问题拆解为有序子任务列表。

{_registry.agent_desc_block}

{_registry.tool_desc_block}

━━ agent 选择规则（严格遵守，违反将导致系统错误）━━

✅ 必须使用工具 agent 的情况：
  - 任何数值计算（加、减、乘、除、幂、开方等）→ math_agent
  - 任何网络请求（访问 URL、调用 HTTP API、fetch 等）→ http_agent
  - 任何数据分析（统计、分组、聚合、过滤等）→ data_agent
  - 任何文件操作（读文件、写文件、列目录、创建目录、移动文件、搜索文件）→ file_agent

❌ 严禁使用 direct 的情况：
  - "计算 3+5" → 必须用 math_agent
  - "访问 https://..." → 必须用 http_agent
  - "分析这批数据" → 必须用 data_agent
  - "读取/写入/列出文件或目录" → 必须用 file_agent

✅ 可以使用 direct 的情况（仅限以下场景）：
  - 闲聊、问候（如"你好"、"介绍一下你自己"）
  - 纯知识性问答（如"什么是加权平均数"）
  - 不涉及任何计算、网络请求、数据处理、文件操作的场景

━━ 其他规则 ━━
1. description 只写任务意图，绝不提前计算数值或给出最终答案
2. inputs 声明运行时需要从哪些前置任务获取参数：
   - key   = 参数的语义名称
   - value = {{"from_task": [被依赖的task_id], "field": "result"}}
3. depends_on 从 inputs 的 from_task 自动推导
4. 没有依赖的任务：inputs 为 {{}}，depends_on 为 []
5. 同一个 agent 可出现多次
6. 任务按拓扑顺序排列（被依赖的任务排在前面）
7. 如果用户消息中已直接包含数据，不要单独拆"获取数据"任务

严格只输出 JSON 数组，不要有任何其他内容、代码块标记或说明文字。

<example>
[
  {{
    "task_id": 0,
    "description": "计算 88 加 12",
    "agent": "math_agent",
    "inputs": {{}},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }},
  {{
    "task_id": 1,
    "description": "把前一步的结果乘以 5",
    "agent": "math_agent",
    "inputs": {{
      "被乘数": {{"from_task": 0, "field": "result"}}
    }},
    "depends_on": [0],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }}
]
</example>"""


def _planner_retry_system(attempt: int, last_raw: str) -> str:
    if attempt == 1:
        return (
            f"{_planner_system()}\n\n"
            f"⚠️ 你上一次的输出 JSON 解析失败。\n"
            f"上次输出片段：\n{last_raw[:200]}\n\n"
            "请只输出合法的 JSON 数组，不要包含任何代码块标记或说明文字。"
        )
    else:
        valid_agents = ", ".join(_registry.agents) if _registry.agents else "direct"
        return (
            "你是任务规划器。严格按以下要求输出：\n\n"
            "1. 只输出一个 JSON 数组\n"
            "2. 不要任何多余文字、代码块标记、注释\n"
            f"3. agent 只能是：{valid_agents}, direct\n"
            "4. 每个任务对象必须包含：\n"
            '   task_id(int), description(str), agent(str), inputs(dict),\n'
            '   depends_on(list), status("pending"), result(""), _resolved_description("")\n\n'
            f"⚠️ 上次输出仍解析失败，片段：{last_raw[:200]}\n\n"
            "用户问题：请重新规划。"
        )


async def planner_node(state: AgentState) -> AgentState:
    if state.get("task_plan"):
        return state

    print("\n  📋 Planner 开始拆解任务...")

    user_message = _get_first_user_message(state)
    last_raw     = ""
    task_plan: list[Task] | None = None

    for attempt in range(3):
        sys_msg = SystemMessage(
            content=_planner_system() if attempt == 0
            else _planner_retry_system(attempt, last_raw)
        )
        if attempt > 0:
            print(f"  🔁 Planner 第 {attempt + 1} 次重试...")

        response = await llm.ainvoke([sys_msg, user_message])
        raw = _extract_json(_extract_llm_content(response).strip())
        last_raw = raw

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise json.JSONDecodeError("期望 JSON 数组", raw, 0)
            task_plan = parsed
            print(f"  ✅ 拆解出 {len(task_plan)} 个任务（第 {attempt + 1} 次成功）：")
            for t in task_plan:
                deps    = t.get("depends_on", [])
                inputs  = t.get("inputs", {})
                dep_str = f"  依赖→{deps} 参数→{list(inputs.keys())}" if deps else ""
                desc    = t["description"]
                desc    = (desc[:60] + "…") if len(desc) > 60 else desc
                print(f"     [{t['task_id']}] {t['agent']} ← {desc}{dep_str}")
            break
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠️ 第 {attempt + 1} 次解析失败：{e}")

    if task_plan is None:
        print("  ❌ Planner 连续 3 次解析失败，终止流程")
        return {
            **state,
            "messages":   state["messages"] + [AIMessage(content=(
                "⚠️ 任务规划失败，Planner 连续 3 次无法生成合法计划。\n"
                "建议换一种更简洁清晰的方式描述您的问题。"
            ))],
            "task_plan":  [],
            "next_agent": "FINISH",
        }

    return {**state, "task_plan": task_plan, "next_agent": ""}


# ══════════════════════════════════════════════════════
# 9. Supervisor（并行调度器）
# ══════════════════════════════════════════════════════
def _resolve_task_inputs(task: Task, done_map: dict[int, Task]) -> None:
    resolved: dict[str, str] = {}
    for param_name, source in task.get("inputs", {}).items():
        if not isinstance(source, dict) or "from_task" not in source:
            continue
        src_task = done_map.get(source["from_task"], {})
        resolved[param_name] = src_task.get(source.get("field", "result"), "")

    if resolved:
        params_text = "\n".join(f"  {k} = {v}" for k, v in resolved.items())
        task["_resolved_description"] = (
            f"{task['description']}\n\n【运行时参数】\n{params_text}"
        )
    else:
        task["_resolved_description"] = task["description"]


def supervisor_dispatch(state: AgentState) -> list[Send]:
    task_plan = state.get("task_plan", [])
    done_map  = {t["task_id"]: t for t in task_plan if t["status"] == "done"}

    ready_tasks = [
        t for t in task_plan
        if t["status"] == "pending"
        and all(dep in done_map for dep in t.get("depends_on", []))
    ]

    if not ready_tasks:
        stuck = [t for t in task_plan if t["status"] == "pending"]
        if stuck:
            print(f"\n  ⚠️ 任务 {[t['task_id'] for t in stuck]} 依赖无法满足，强制跳过")
            for t in stuck:
                t["status"] = "done"
                t["result"] = "⚠️ 依赖未满足，跳过"
        print("\n  🧭 Supervisor → final_answer（无更多就绪任务）")
        return [Send("final_answer", state)]

    print(f"\n  🚀 本轮并行分发 {len(ready_tasks)} 个任务：")
    sends: list[Send] = []
    for task in ready_tasks:
        task["status"] = "in_progress"
        _resolve_task_inputs(task, done_map)
        target = task["agent"] if task["agent"] != "direct" else "direct_answer"
        print(f"     [{task['task_id']}] → {target}: {task['description'][:55]}")
        sends.append(Send(target, {
            "task_plan":       task_plan,
            "current_task_id": task["task_id"],
            "messages":        state.get("messages", []),
        }))

    return sends


# ══════════════════════════════════════════════════════
# 10. Replanner（纯逻辑失败检测）
# ══════════════════════════════════════════════════════
_FAILURE_MARKERS = ("❌", "Error", "error", "失败", "timeout",
                    "⚠️ 依赖", "exception", "Exception")

def _is_task_failed(result: str) -> bool:
    return any(marker in result for marker in _FAILURE_MARKERS)


def _cascade_skip(task_plan: list[Task], failed_id: int) -> None:
    to_skip: set[int] = set()
    queue = [failed_id]
    while queue:
        current = queue.pop()
        for task in task_plan:
            if (task["status"] == "pending"
                    and current in task.get("depends_on", [])
                    and task["task_id"] not in to_skip):
                to_skip.add(task["task_id"])
                queue.append(task["task_id"])
    for task in task_plan:
        if task["task_id"] in to_skip:
            task["status"] = "done"
            task["result"] = f"⚠️ 跳过：依赖任务[{failed_id}]执行失败"
            print(f"     → 任务[{task['task_id']}]({task['description'][:40]}) 已跳过")


def _check_and_cascade_failures(task_plan: list[Task]) -> None:
    for task in task_plan:
        if (task["status"] == "done"
                and _is_task_failed(task.get("result", ""))
                and not task["result"].startswith("⚠️ 跳过：")):
            print(f"  ⚠️ Re-Planner：任务[{task['task_id']}] 失败，触发级联跳过")
            _cascade_skip(task_plan, task["task_id"])


# ══════════════════════════════════════════════════════
# 11. Collect 汇聚节点
# ══════════════════════════════════════════════════════
def collect_node(state: AgentState) -> AgentState:
    task_plan     = state.get("task_plan", [])
    done_count    = sum(1 for t in task_plan if t["status"] == "done")
    pending_count = sum(1 for t in task_plan if t["status"] == "pending")
    print(f"\n  📥 collect_node：已完成 {done_count} 个，待执行 {pending_count} 个")
    _check_and_cascade_failures(task_plan)
    return {**state, "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 12. 通用工具 Agent 执行器
# ══════════════════════════════════════════════════════
async def run_agent(state: WorkerState, agent_name: str, system_prompt: str) -> dict:
    # ★ 若 registry 为空（Studio 竞态），等待 lifespan 初始化完成
    if not _registry.agents:
        print(f"  ⏳ [{agent_name}] _registry 尚未就绪，等待 lifespan 初始化...")
        try:
            await asyncio.wait_for(_get_registry_ready_event().wait(), timeout=60)
            print(f"  ✅ [{agent_name}] _registry 已就绪，继续执行")
        except asyncio.TimeoutError:
            print(f"  ❌ [{agent_name}] 等待 _registry 超时（60s）")

    lc_tools   = _registry.tools_for(agent_name)
    tool_names = _registry.tool_names_for(agent_name)
    by_name    = {t.name: t for t in lc_tools}

    task_plan    = state.get("task_plan", [])
    task_id: int = state.get("current_task_id", 0)
    current_task = next((t for t in task_plan if t["task_id"] == task_id), None)

    if not current_task:
        print(f"  ⚠️ [{agent_name}] 找不到任务 task_id={task_id}")
        return {"task_plan": task_plan}

    task_description = current_task.get("_resolved_description") or current_task["description"]

    if not lc_tools:
        current_task["status"] = "done"
        current_task["result"] = f"⚠️ 没有可用工具（{agent_name} 未注册任何工具）"
        return {"task_plan": task_plan}

    tool_hint = "可用工具：" + "、".join(
        f"{t.name}（{t.description or '无描述'}）" for t in lc_tools
    )
    full_system = (
        f"{system_prompt}\n\n{tool_hint}\n\n"
        "【重要约束】你必须通过调用工具来完成任务，严禁凭记忆、推理或编造直接给出答案。"
        "即使你认为自己知道答案，也必须先调用工具获取真实结果。"
    )
    agent_llm     = llm.bind_tools(lc_tools)
    msgs          = [SystemMessage(content=full_system), HumanMessage(content=task_description)]
    last_response = None

    print(f"\n  ▶ {agent_name} 执行任务[{task_id}]（工具：{tool_names}）")

    for step in range(10):
        response = await agent_llm.ainvoke(msgs)
        last_response = response
        msgs.append(response)

        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

        if step == 0 and not has_tool_calls:
            print(f"  ⚠️ {agent_name} 第1轮未调用工具，追加强制提示重试...")
            msgs.append(HumanMessage(content=(
                f"你刚才没有调用任何工具就直接回答了，这是不允许的。\n"
                f"你拥有以下工具：{'、'.join(tool_names)}\n"
                "请立即调用对应工具来完成任务，不要直接给出文字答案。"
            )))
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
        print(f"  ⚠️ {agent_name} 达到最大步数，强制终止")

    current_task["status"] = "done"
    current_task["result"] = _extract_llm_content(last_response) if last_response else "（无结果）"
    return {"task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 13. 图构建
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
    "file_agent": (
        "你是文件系统操作专家。所有文件操作都被限制在授权目录内，"
        "禁止访问授权目录以外的路径。\n"
        "根据任务描述和【运行时参数】（如果有），调用合适的工具完成文件操作：\n"
        "  - 读取文件内容 → read_file\n"
        "  - 写入/覆盖文件 → write_file\n"
        "  - 列出目录内容 → list_directory\n"
        "  - 创建目录 → create_directory\n"
        "  - 移动/重命名文件 → move_file\n"
        "  - 搜索文件 → search_files\n"
        "  - 获取文件信息 → get_file_info\n"
        "操作成功后返回简洁的结果说明，失败时返回具体错误信息。"
    ),
}

DEFAULT_AGENT_SYSTEM_PROMPT = (
    "你是通用任务执行专家。根据任务描述，调用合适的工具完成任务，给出简洁结果。"
)


def build_graph() -> Any:

    async def direct_answer_node(state: WorkerState) -> dict:
        task_plan    = state.get("task_plan", [])
        task_id: int = state.get("current_task_id", 0)
        current_task = next((t for t in task_plan if t["task_id"] == task_id), None)
        if not current_task:
            return {"task_plan": task_plan}

        intent = current_task.get("description", "")
        print(f"\n  💬 direct_answer（意图：{intent[:60]}）")
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
        return {"task_plan": task_plan}

    async def final_answer_node(state: AgentState) -> AgentState:
        task_plan    = state.get("task_plan", [])
        tool_tasks   = [t for t in task_plan if t.get("agent") != "direct"]
        direct_tasks = [t for t in task_plan if t.get("agent") == "direct"]

        if not tool_tasks:
            combined = "\n\n".join(t["result"] for t in direct_tasks if t.get("result"))
            if combined:
                return {**state, "messages": state["messages"] + [AIMessage(content=combined)]}
            return state

        lines: list[str] = []
        if direct_tasks:
            lines.append("【直接回答任务】")
            for t in direct_tasks:
                lines.append(f"  任务[{t['task_id']}]（{t['description']}）：{t['result']}")
        lines.append("【工具执行任务】")
        for t in tool_tasks:
            lines.append(f"  任务[{t['task_id']}]（{t['description']}）：{t['result']}")

        results_text = "\n".join(lines)
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
            "messages": state["messages"] + [AIMessage(content=_extract_llm_content(response))],
        }

    def planner_route(state: AgentState) -> str:
        if state.get("next_agent") == "FINISH":
            print("  ⛔ Planner 规划失败，直接终止")
            return "END"
        return "supervisor_dispatch"

    def collect_route(state: AgentState) -> list[Send]:
        return supervisor_dispatch(state)

    def make_agent_node(name: str):
        system_prompt = AGENT_SYSTEM_PROMPTS.get(name, DEFAULT_AGENT_SYSTEM_PROMPT)
        async def _node(state: WorkerState) -> dict:
            return await run_agent(state, name, system_prompt)
        _node.__name__ = name
        return _node

    g = StateGraph(AgentState)
    g.add_node("planner",       planner_node)
    g.add_node("collect",       collect_node)
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("final_answer",  final_answer_node)

    # 用 AGENT_SYSTEM_PROMPTS 的 key 作为固定节点集，不依赖 _registry
    known_agents = list(AGENT_SYSTEM_PROMPTS.keys())
    for agent_name in known_agents:
        g.add_node(agent_name, make_agent_node(agent_name))

    g.set_entry_point("planner")
    g.add_conditional_edges("planner", planner_route, {
        "END": END,
        "supervisor_dispatch": "supervisor_dispatch_node",
    })

    async def _supervisor_dispatch_node(state: AgentState) -> AgentState:
        return state

    g.add_node("supervisor_dispatch_node", _supervisor_dispatch_node)
    g.add_conditional_edges(
        "supervisor_dispatch_node",
        supervisor_dispatch,
        {name: name for name in [*known_agents, "direct_answer", "final_answer"]},
    )

    for agent_name in known_agents:
        g.add_edge(agent_name, "collect")
    g.add_edge("direct_answer", "collect")

    g.add_conditional_edges(
        "collect",
        collect_route,
        {name: name for name in [*known_agents, "direct_answer", "final_answer"]},
    )
    g.add_edge("final_answer", END)

    return g.compile()


# ══════════════════════════════════════════════════════
# 14. 图实例（模块导入时构建一次，不再重建）
# ══════════════════════════════════════════════════════
graph = build_graph()


# ══════════════════════════════════════════════════════
# 15. lifespan —— 仅 langgraph dev 调用
#
# ★★★ langgraph.json 必须配置 ★★★
# {
#   "graphs": {"supervisorpalner": "./src/langgraph_stdio_agent.py:graph"},
#   "lifespan": "src.langgraph_stdio_agent:lifespan"
# }
# ══════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app):
    print("🔔 [lifespan] 启动中...")
    if not SERVER_PATH.exists():
        raise FileNotFoundError(f"找不到 MCP server：{SERVER_PATH}")

    # 启动后台持久连接（内部等待全部就绪）
    await _mcp_manager.start()

    # 全走 manager 队列，不持有任何 session 对象
    server_tools = await load_tools(session_name="server",     use_manager=True)
    print(f"✅ [lifespan] server.py 工具：{[t.name for t in server_tools]}")

    fs_tools = await load_tools(session_name="filesystem", use_manager=True)
    print(f"✅ [lifespan] filesystem 工具：{[t.name for t in fs_tools]}")

    all_tools = server_tools + fs_tools
    _tools.extend(all_tools)
    _init_registry(all_tools)   # ← 内部 set _registry_ready event
    # ★ 不调用 _rebuild_graph()，run_agent 运行时动态查全局 _registry
    print(f"🚀 [lifespan] 就绪，共 {len(all_tools)} 个工具，agents: {_registry.agents}")

    yield

    await _mcp_manager.stop()
    _tools.clear()
    print("🛑 [lifespan] 已关闭")


# ══════════════════════════════════════════════════════
# 16. __main__ —— 单独运行测试
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    if not SERVER_PATH.exists():
        print(f"❌ 找不到 server.py：{SERVER_PATH}")
        sys.exit(1)

    QUESTIONS = [
        "列出 File Agent 目录下的所有文件，然后在其中创建一个名为 hello.txt 的文件，内容为：Hello from file_agent！",
        # "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",
        # "计算 99 乘以 9，然后把结果写入 File Agent/result.txt 文件",
        # "先介绍一下你自己，然后帮我计算 99 乘以 9",
        # "你好，我叫 tony",
    ]

    async def main():
        async with stdio_client(mcp_params()) as (r1, w1):
            async with ClientSession(r1, w1) as s1:
                await s1.initialize()
                server_tools = await load_tools(s1)
                print(f"✅ server.py 工具：{[t.name for t in server_tools]}")

                async with stdio_client(filesystem_mcp_params()) as (r2, w2):
                    async with ClientSession(r2, w2) as s2:
                        await s2.initialize()
                        fs_tools = await load_tools(s2)
                        print(f"✅ filesystem 工具：{[t.name for t in fs_tools]}")

                        _tools.extend(server_tools + fs_tools)
                        _init_registry(server_tools + fs_tools)

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