"""
src/langgraph_parallel_agent.py

★ 并行化改造 + Bug 修复（基于 langgraph_stdio_agent.py）

【并行架构改造】
  - supervisor_node（逐任务串行循环）→ parallel_executor_node（拓扑分层并行）
  - 同一层无互相依赖的任务通过 asyncio.gather() 真正并发执行
  - 每个任务独立 spawn/close 自己的 MCP session，彻底无并发安全问题
  - 图结构简化：
      改造前：planner → supervisor ⇄ agentX（循环）→ final_answer
      改造后：planner → parallel_executor → final_answer

【✅ 阶段一新增：MemorySaver Checkpoint 支持】
  改动点共 5 处，搜索 "★ CHECKPOINT" 可快速定位全部改动：
  1. imports 区：新增 MemorySaver + add_messages 导入
  2. AgentState：messages 字段加 Annotated[list, add_messages] reducer
  3. 全局变量区：新增 _checkpointer = MemorySaver() 单例
  4. build_graph()：接收 checkpointer 参数并传给 compile()
  5. _init_registry()：复用 _checkpointer，不再每次新建
  6. __main__ _run_question()：传入 thread_id + config

【新增函数】
  - _topo_layers()             拓扑 BFS 分层，返回按批次排列的任务列表
  - _spawn_session_for()       按 agent 类型决定 spawn 哪个 MCP server
  - run_agent_isolated()       带独立 session 生命周期的单任务执行单元
  - parallel_executor_node()   替代原 supervisor_node，驱动整个并行调度

【Bug 修复】
  - deps_done 只认 "done"，不再把 "in_progress" 视为已满足

【保留不变】
  - planner_node / direct_answer_node / final_answer_node
  - ToolRegistry / load_tools / _extract_* 等全部工具函数
  - MCP server 路径、SSE URL、AGENT_* 配置表
  - _start_mcp_sessions / _stop_mcp_sessions（供 webapp.py lifespan 调用）

原有所有修复（修复1–15）均保留，不再重复列出。
"""

import asyncio
import json
import os
import re
import sys
import time
import traceback
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

# ★ CHECKPOINT 改动1/6：新增两个导入
#
# Annotated 是 Python 类型系统的工具，用来给类型附加"元信息"。
# 这里的用途：告诉 LangGraph "messages 字段用 add_messages 函数来合并"。
#
# add_messages 是 LangGraph 内置的 reducer（合并函数）。
# 什么是 reducer？
#   - 普通字段：新值直接覆盖旧值。  旧=[A,B]  新=[C]  → 结果=[C]   ← 对话历史丢失！
#   - add_messages：新消息追加到末尾。旧=[A,B]  新=[C]  → 结果=[A,B,C] ← 对话历史保留 ✅
#
# 为什么 checkpoint 必须用 add_messages？
#   checkpoint 保存的是整个 state。下次 invoke 恢复 state 后，
#   新的 HumanMessage 需要"追加"到历史里，而不是"替换"历史。
#   如果不加 add_messages，每次对话都会把之前的消息全部清空。
from typing import Any, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages   # ← 新增

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END

# ★ CHECKPOINT 改动2/6：新增 MemorySaver 导入
#
# MemorySaver 是 LangGraph 内置的"内存版"存储后端。
# 它把每个 thread_id 的 checkpoint（state 快照）存在 Python 字典里。
# 优点：零配置，开发测试直接用。
# 缺点：进程重启后数据全部清空（阶段二换 SqliteSaver 解决）。
from langgraph.checkpoint.memory import MemorySaver   # ← 新增

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp import StdioServerParameters
from pathlib import Path
from pydantic import create_model

_dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(str(_dotenv_path), override=False)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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
SERVER_PATH    = Path(__file__).parent / "mcp_server_template" / "server.py"
DB_SERVER_PATH = Path(__file__).parent / "mcp_db_server" / "server.py"
MATH_MCP_JS    = Path(__file__).parent / "math-mcp" / "build" / "index.js"

_MCP_FS_ENV  = os.getenv("MCP_FS_BASE_DIR", "")
_FS_BASE_DIR = Path(_MCP_FS_ENV) if _MCP_FS_ENV else Path(__file__).parent.parent / "File_Agent"

_SERVER_PORT     = int(os.getenv("MCP_SERVER_PORT",     "8001"))
_FS_PROXY_PORT   = int(os.getenv("MCP_FS_PROXY_PORT",   "8002"))
_DB_SERVER_PORT  = int(os.getenv("MCP_DB_SERVER_PORT",  "8003"))
_MATH_PROXY_PORT = int(os.getenv("MCP_MATH_PROXY_PORT", "8004"))

_SERVER_SSE_URL     = f"http://127.0.0.1:{_SERVER_PORT}/sse"
_FS_PROXY_SSE_URL   = f"http://127.0.0.1:{_FS_PROXY_PORT}/sse"
_DB_SERVER_SSE_URL  = f"http://127.0.0.1:{_DB_SERVER_PORT}/sse"
_MATH_PROXY_SSE_URL = f"http://127.0.0.1:{_MATH_PROXY_PORT}/sse"


def mcp_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-u", str(SERVER_PATH)],
        env={"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **os.environ},
    )

def filesystem_mcp_params() -> StdioServerParameters:
    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    print(f"📂 Filesystem MCP BASE_DIR: {_FS_BASE_DIR}", file=sys.stderr)
    return StdioServerParameters(
        command=npx_cmd,
        args=["-y", "@modelcontextprotocol/server-filesystem", str(_FS_BASE_DIR)],
        env={**os.environ},
    )

def db_mcp_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-u", str(DB_SERVER_PATH)],
        env={"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **os.environ},
    )

def math_mcp_params() -> StdioServerParameters:
    node_cmd = "node.exe" if sys.platform == "win32" else "node"
    return StdioServerParameters(
        command=node_cmd,
        args=[str(MATH_MCP_JS)],
        env={**os.environ},
    )


# ══════════════════════════════════════════════════════
# 3. 动态工具注册表
# ══════════════════════════════════════════════════════

AGENT_TOOL_PATTERNS: dict[str, list[str]] = {
    "math_agent": ["add", "subtract", "multiply", "division"],
    "data_agent": ["dataframe_summary", "group_and_aggregate", "filter_rows",
                   "sort_dataframe", "pivot_table", "data_*", "df_*"],
    "http_agent": ["fetch_url", "post_json", "http_get", "http_post",
                   "http_*", "fetch_*", "request_*"],
    "file_agent": ["read_file", "write_file", "edit_file",
                   "read_multiple_files", "list_directory", "create_directory",
                   "move_file", "search_files", "get_file_info",
                   "list_allowed_directories", "file_*"],
    "db_agent":   ["ask_db", "query_db", "execute_db", "get_schema",
                   "db_*", "sql_*"],
}

AGENT_DESCRIPTIONS: dict[str, str] = {
    "math_agent": "数学计算（加减乘除、幂、开方等数值运算）",
    "data_agent": "数据分析（统计、聚合、分组、过滤等结构化数据处理）",
    "http_agent": "网络请求（GET/POST、访问 URL、调用外部 API）",
    "file_agent": "文件操作（读写文件、列出目录、创建目录、移动/搜索文件）",
    "db_agent":   "数据库查询（电商数据库：用户/商品/订单/评价/库存，支持自然语言和直接 SQL）",
}

AGENT_TRIGGER_KEYWORDS: dict[str, list[str]] = {
    "math_agent": ["计算", "加", "减", "乘", "除", "求和", "平均", "幂", "开方",
                   "+", "-", "×", "÷", "*", "/", "²", "√"],
    "http_agent": ["访问", "请求", "获取", "http", "https", "url", "api",
                   "fetch", "get ", "post "],
    "data_agent": ["分析", "统计", "分组", "聚合", "过滤", "排序", "数据集",
                   "dataframe", "pivot"],
    "file_agent": ["文件", "目录", "列出", "读取", "写入", "创建", "移动",
                   "搜索文件", "read_file", "write_file", "list_directory"],
    "db_agent":   ["查询", "数据库", "订单", "用户", "商品", "库存", "评价",
                   "销售额", "销售", "购买", "category", "product", "order",
                   "review", "inventory", "sql", "select", "多少", "哪些",
                   "列出所有", "找出", "统计订单", "统计用户", "统计商品",
                   "最高", "最低", "排名", "top", "最畅销"],
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


# ★ CHECKPOINT 改动3/6：AgentState 的 messages 字段加 Annotated + add_messages
#
# 改动前：messages: list
# 改动后：messages: Annotated[list, add_messages]
#
# 详细解释：
#   LangGraph 的 checkpoint 机制在每次节点执行完之后，
#   会把节点返回的 state 与已保存的 state "合并"（merge）。
#
#   合并规则由字段类型决定：
#   ┌─────────────────┬────────────────────────────────────────────┐
#   │ 字段类型         │ 合并行为                                    │
#   ├─────────────────┼────────────────────────────────────────────┤
#   │ 普通 list       │ 新值直接覆盖旧值（旧消息全部丢失！）           │
#   │ Annotated+      │ 新消息追加到旧消息末尾（对话历史完整保留）✅   │
#   │ add_messages    │                                            │
#   └─────────────────┴────────────────────────────────────────────┘
#
#   多轮对话场景：
#   第1轮：用户问"计算3+5"  → messages = [Human("计算3+5"), AI("答案是8")]
#   第2轮：用户问"刚才的结果乘以2是多少"
#          → 有 add_messages：messages = [Human("计算3+5"), AI("8"), Human("乘以2"), ...]
#          → 没有 add_messages：messages = [Human("乘以2")]  ← AI 不知道上文，无法回答！
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # ← 改动：加了 Annotated[list, add_messages]
    task_plan: list[Task]
    current_task_id: int
    next_agent: str


# ══════════════════════════════════════════════════════
# 5. 共享容器（供 webapp.py lifespan 调用的全局 session）
# ══════════════════════════════════════════════════════
_tools: list[StructuredTool] = []
_registry: ToolRegistry = ToolRegistry()
_lazy_init_lock: asyncio.Lock | None = None
_mcp_exit_stack: AsyncExitStack | None = None


# ★ CHECKPOINT 改动4/6：_checkpointer 模块级单例
#
# 为什么必须是"模块级单例"？
#
# 问题背景：
#   你的代码里 _init_registry() 会调用 build_graph()，
#   每次工具列表更新时都会重建 graph 对象。
#
# 如果在 build_graph() 内部写 checkpointer=MemorySaver()：
#   第1次 _init_registry() → build_graph() → 创建 MemorySaver_A → graph_A
#   第2次 _init_registry() → build_graph() → 创建 MemorySaver_B → graph_B
#   MemorySaver_A 里保存的所有用户对话历史全部消失！
#
# 正确做法：
#   在模块加载时创建一个 _checkpointer，它的生命周期 = 进程生命周期。
#   每次 build_graph() 都把这同一个 _checkpointer 传进去。
#   无论 graph 重建多少次，存储的数据始终在这一个 _checkpointer 里。
#
# 类比：
#   MemorySaver 就像一个"笔记本"，里面按 thread_id 分页记录每个用户的对话。
#   如果每次重建 graph 都换一本新笔记本，之前写的内容全丢了。
#   正确做法是始终用同一本笔记本，graph 重建只是换了"读笔记本的方式"，笔记本本身不变。
_checkpointer = MemorySaver()   # ← 新增：进程级单例，永不重建


async def _ensure_registry() -> None:
    global _lazy_init_lock
    if _registry.agents:
        return
    if _lazy_init_lock is None:
        _lazy_init_lock = asyncio.Lock()
    async with _lazy_init_lock:
        if _registry.agents:
            return
        print("⚡ [lazy-init] registry 为空，触发 MCP 初始化（SSE 模式）...")
        await _start_mcp_sessions()


# ══════════════════════════════════════════════════════
# MCP Session 管理（全局 session，供 registry 初始化用）
# ══════════════════════════════════════════════════════

async def _start_mcp_sessions() -> None:
    """前端路径（langgraph dev）：SSE 连接已由 webapp.py 拉起的子进程。"""
    global _mcp_exit_stack
    if _mcp_exit_stack is not None:
        print("⚠️ [MCP] _start_mcp_sessions 重复调用，跳过")
        return

    print(f"🔍 [MCP] platform={sys.platform}  python={sys.executable}")

    stack = AsyncExitStack()
    all_tools: list[StructuredTool] = []

    for tag, url in [
        ("server.py",     _SERVER_SSE_URL),
        ("filesystem",    _FS_PROXY_SSE_URL),
        ("db_server",     _DB_SERVER_SSE_URL),
        ("math-mcp",      _MATH_PROXY_SSE_URL),
    ]:
        try:
            r, w = await stack.enter_async_context(sse_client(url))
            s    = await stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            tools = await load_tools(s)
            print(f"✅ [MCP] {tag} 工具：{[t.name for t in tools]}")
            all_tools.extend(tools)
        except Exception as exc:
            print(f"❌ [MCP] {tag} SSE 连接失败（{url}）：{exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not all_tools:
        print("❌ [MCP] 所有 MCP 连接失败，registry 未就绪", file=sys.stderr)
        await stack.aclose()
        return

    _mcp_exit_stack = stack
    _tools.clear()
    _tools.extend(all_tools)
    _init_registry(all_tools)
    print(f"🚀 [MCP] 就绪，共 {len(all_tools)} 个工具，agents: {_registry.agents}")


async def _start_mcp_sessions_stdio() -> None:
    """后端路径（__main__ 直接运行）：stdio_client spawn 子进程。"""
    global _mcp_exit_stack
    if _mcp_exit_stack is not None:
        return

    print(f"🔍 [MCP-stdio] SERVER_PATH    = {SERVER_PATH}  (exists={SERVER_PATH.exists()})")
    print(f"🔍 [MCP-stdio] DB_SERVER_PATH = {DB_SERVER_PATH}  (exists={DB_SERVER_PATH.exists()})")
    print(f"🔍 [MCP-stdio] MATH_MCP_JS    = {MATH_MCP_JS}  (exists={MATH_MCP_JS.exists()})")
    print(f"🔍 [MCP-stdio] FS_BASE_DIR    = {_FS_BASE_DIR}  (exists={_FS_BASE_DIR.exists()})")

    stack = AsyncExitStack()
    all_tools: list[StructuredTool] = []

    # 1. server.py
    if not SERVER_PATH.exists():
        print(f"❌ [MCP-stdio] 找不到 MCP server：{SERVER_PATH}", file=sys.stderr)
    else:
        try:
            r, w = await stack.enter_async_context(stdio_client(mcp_params()))
            s    = await stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            tools = await load_tools(s)
            print(f"✅ [MCP-stdio] server.py 工具：{[t.name for t in tools]}")
            all_tools.extend(tools)
        except Exception as exc:
            print(f"❌ [MCP-stdio] server.py 启动失败：{exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # 2. mcp-server-filesystem
    try:
        r, w = await stack.enter_async_context(stdio_client(filesystem_mcp_params()))
        s    = await stack.enter_async_context(ClientSession(r, w))
        await s.initialize()
        tools = await load_tools(s)
        print(f"✅ [MCP-stdio] filesystem 工具：{[t.name for t in tools]}")
        all_tools.extend(tools)
    except Exception as exc:
        print(f"❌ [MCP-stdio] mcp-server-filesystem 启动失败：{exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # 3. db_server.py
    if not DB_SERVER_PATH.exists():
        print(f"❌ [MCP-stdio] 找不到 DB MCP server：{DB_SERVER_PATH}", file=sys.stderr)
    else:
        try:
            r, w = await stack.enter_async_context(stdio_client(db_mcp_params()))
            s    = await stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            tools = await load_tools(s)
            print(f"✅ [MCP-stdio] db_server 工具：{[t.name for t in tools]}")
            all_tools.extend(tools)
        except Exception as exc:
            print(f"❌ [MCP-stdio] db_server 启动失败：{exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # 4. math-mcp
    if not MATH_MCP_JS.exists():
        print(f"❌ [MCP-stdio] 找不到 math-mcp：{MATH_MCP_JS}", file=sys.stderr)
        print(f"   请先执行：cd src/math-mcp && npm install && npm run build", file=sys.stderr)
    else:
        try:
            r, w = await stack.enter_async_context(stdio_client(math_mcp_params()))
            s    = await stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            tools = await load_tools(s)
            print(f"✅ [MCP-stdio] math-mcp 工具：{[t.name for t in tools]}")
            all_tools.extend(tools)
        except Exception as exc:
            print(f"❌ [MCP-stdio] math-mcp 启动失败：{exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not all_tools:
        print("❌ [MCP-stdio] 所有 MCP 连接失败", file=sys.stderr)
        await stack.aclose()
        return

    _mcp_exit_stack = stack
    _tools.clear()
    _tools.extend(all_tools)
    _init_registry(all_tools)
    print(f"🚀 [MCP-stdio] 就绪，共 {len(all_tools)} 个工具，agents: {_registry.agents}")


async def _stop_mcp_sessions() -> None:
    global _mcp_exit_stack, _lazy_init_lock
    if _mcp_exit_stack is not None:
        await _mcp_exit_stack.aclose()
        _mcp_exit_stack = None
    _tools.clear()
    _init_registry([])
    _lazy_init_lock = None
    print("🛑 [MCP] 所有 session 已关闭")


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

        async def _call(_name=tool_name, _sess=session, **kwargs) -> str:
            print(f"    🔧 [MCP] {_name}({kwargs})")
            res  = await _sess.call_tool(_name, kwargs)
            text = res.content[0].text if res.content else "（无结果）"
            print(f"    ✅ {text[:200]}")
            return text

        lc_tools.append(StructuredTool.from_function(
            coroutine=_call, name=t.name,
            description=t.description or "", args_schema=DynSchema,
        ))

    print(f"✅ 已加载 {len(lc_tools)} 个工具：{[t.name for t in lc_tools]}")
    return lc_tools


# ★ CHECKPOINT 改动5/6：_init_registry 复用 _checkpointer
#
# 改动前：
#   def _init_registry(tools):
#       global _registry, graph
#       _registry = ToolRegistry.build(tools) if tools else ToolRegistry()
#       graph = build_graph()   ← build_graph 内部创建新的 MemorySaver，每次重建都丢数据
#
# 改动后：
#   graph = build_graph(checkpointer=_checkpointer)  ← 传入模块级单例，数据不丢
#
# 为什么 _init_registry 会被多次调用？
#   场景1：__main__ 启动 → _start_mcp_sessions_stdio() → _init_registry()
#   场景2：_stop_mcp_sessions() 也会调用 _init_registry([]) 清空 registry
#   场景3：如果未来支持热更新工具，也会触发 _init_registry()
#   每次都传同一个 _checkpointer，数据安全。
def _init_registry(tools: list[StructuredTool]) -> None:
    global _registry, graph
    _registry = ToolRegistry.build(tools) if tools else ToolRegistry()
    graph = build_graph(checkpointer=_checkpointer)   # ← 改动：传入单例 checkpointer


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


def _get_first_user_message(state: "AgentState") -> Any:
    msgs = state.get("messages", [])
    if not msgs:
        return HumanMessage(content="")
    msg = msgs[0]
    if isinstance(msg, dict):
        return HumanMessage(content=msg.get("content") or msg.get("text") or "")
    return msg


# ══════════════════════════════════════════════════════
# 7. Planner
# ══════════════════════════════════════════════════════

def _planner_system() -> str:
    return f"""你是任务规划器。把用户问题拆解为有序子任务列表。

{_registry.agent_desc_block}

{_registry.tool_desc_block}

━━ agent 选择规则（严格遵守，违反将导致系统错误）━━

✅ 必须使用工具 agent 的情况：
  - 任何数值计算（加、减、乘、除、幂、开方等）→ math_agent
  - 任何网络请求（访问 URL、调用 HTTP API、fetch 等）→ http_agent
  - 任何数据分析（统计、分组、聚合、过滤等）→ data_agent
  - 任何数据库查询（订单/用户/商品/库存/评价等）→ db_agent
  - 任何文件操作（读写文件、列目录等）→ file_agent

❌ 严禁使用 direct 的情况：
  - "计算 3+5" → 必须用 math_agent
  - "访问 https://..." → 必须用 http_agent
  - "查询订单" / "有多少用户" / "库存不足的商品" → 必须用 db_agent
  - "列出文件" / "读取文件" → 必须用 file_agent

✅ 可以使用 direct 的情况（仅限）：
  - 闲聊、问候、纯知识性问答
  - 不涉及任何计算、网络、数据库、文件操作的场景

━━ db_agent 使用说明 ━━
db_agent 连接电商数据库，包含：
  users / products / categories / orders / order_items / reviews / inventory_log
db_agent 会自己写 SQL 用 query_db 执行，禁止在 planner 层面指定使用 ask_db。

━━ inputs 格式规则（最重要，违反将导致运行时传参失败）━━

inputs 用于声明"本任务运行时需要从哪个前置任务获取什么值"。

【格式要求】inputs 的每个值必须是包含 from_task 和 field 的对象：
  "inputs": {{
    "<任意描述性key>": {{"from_task": <被依赖的task_id(整数)>, "field": "result"}}
  }}

【正确示例】
  任务2 依赖任务0的结果：
  "inputs": {{"db_result": {{"from_task": 0, "field": "result"}}}}

  任务4 同时依赖任务1和任务2的结果：
  "inputs": {{
    "quote":     {{"from_task": 1, "field": "result"}},
    "file_list": {{"from_task": 2, "field": "result"}}
  }}

【❌ 严禁以下错误格式】
  "inputs": {{"sqrt_3": 0}}              ← 值是整数，错误
  "inputs": {{"from_task_0": 0}}         ← 值是整数，错误
  "inputs": {{"data": "task0_result"}}   ← 值是字符串，错误
  "inputs": {{"value": {{"task": 0}}}}   ← 缺少 field 字段，错误

【没有依赖的任务】inputs 必须为空对象：
  "inputs": {{}}，"depends_on": []

━━ depends_on 规则 ━━
depends_on 必须与 inputs 中所有 from_task 的值完全一致。
  inputs 有 from_task:1 和 from_task:2 → depends_on 必须是 [1, 2]
  inputs 为空 → depends_on 必须是 []

━━ 其他规则 ━━
1. description 只写任务意图，不提前计算数值或给出答案
2. 同一个 agent 可出现多次
3. 任务按拓扑顺序排列（被依赖的任务排在前面）

严格只输出 JSON 数组，不要有任何其他内容或代码块标记。

【单任务无依赖示例】
[
  {{
    "task_id": 0,
    "description": "查询所有来自 Toronto 的活跃用户",
    "agent": "db_agent",
    "inputs": {{}},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }}
]

【多任务含依赖示例】查询最低评分商品，然后把商品名写入文件：
[
  {{
    "task_id": 0,
    "description": "查询评分最低的商品名称",
    "agent": "db_agent",
    "inputs": {{}},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }},
  {{
    "task_id": 1,
    "description": "将评分最低的商品名称写入 low_rating.txt",
    "agent": "file_agent",
    "inputs": {{"product_name": {{"from_task": 0, "field": "result"}}}},
    "depends_on": [0],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }}
]

【多任务多依赖示例】并行查询DB和访问URL，然后把两个结果合并写文件：
[
  {{
    "task_id": 0,
    "description": "查询订单总金额最高的用户",
    "agent": "db_agent",
    "inputs": {{}},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }},
  {{
    "task_id": 1,
    "description": "访问 https://api.github.com/zen 获取格言",
    "agent": "http_agent",
    "inputs": {{}},
    "depends_on": [],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }},
  {{
    "task_id": 2,
    "description": "把用户信息和格言合并写入 report.txt",
    "agent": "file_agent",
    "inputs": {{
      "user_info": {{"from_task": 0, "field": "result"}},
      "quote":     {{"from_task": 1, "field": "result"}}
    }},
    "depends_on": [0, 1],
    "status": "pending",
    "result": "",
    "_resolved_description": ""
  }}
]
"""


async def planner_node(state: AgentState) -> AgentState:
    await _ensure_registry()

    msgs = state.get("messages", [])
    if not msgs:
        return {**state, "next_agent": "FINISH", "task_plan": []}

    # user_msg = _get_message_content(msgs[0])
    # print(f"\n📋 [Planner] 规划任务：{user_msg[:80]}")
    
    last_human_msg = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage)),
        None
    )
    if not last_human_msg:
        return {**state, "next_agent": "FINISH", "task_plan": []}

    user_msg = _get_message_content(last_human_msg)   # ← 用这个替换原来的 user_msg
    print(f"\n📋 [Planner] 规划任务：{user_msg[:80]}")

    max_retries = 3
    task_plan: list[Task] = []

    for attempt in range(max_retries):
        try:
             # ★ 带上对话历史，让 planner 知道上下文
            # 取历史里除最后一条之外的所有消息（最后一条是当前问题，已经单独处理）
            history_msgs = msgs[:-1] if len(msgs) > 1 else []

            response = await llm.ainvoke([
                SystemMessage(content=_planner_system()),
                *history_msgs,           # ← 插入历史（Human/AI 交替的对话记录）
                HumanMessage(content=user_msg),
            ])
            raw = _extract_json(_extract_llm_content(response))
            task_plan = json.loads(raw)

            if not isinstance(task_plan, list) or len(task_plan) == 0:
                raise ValueError("Empty or invalid task plan")

            for t in task_plan:
                t.setdefault("status", "pending")
                t.setdefault("result", "")
                t.setdefault("_resolved_description", "")
                t.setdefault("inputs", {})
                t.setdefault("depends_on", [])

            # ── inputs 格式校验：每个值必须是含 from_task+field 的 dict ──
            fmt_errors: list[str] = []
            for t in task_plan:
                for key, val in t.get("inputs", {}).items():
                    if not isinstance(val, dict):
                        fmt_errors.append(
                            f"task[{t['task_id']}].inputs[{key}] 值类型错误："
                            f"期望 dict，实际 {type(val).__name__}({val!r})"
                        )
                    elif "from_task" not in val:
                        fmt_errors.append(
                            f"task[{t['task_id']}].inputs[{key}] 缺少 from_task 字段：{val!r}"
                        )
                task_deps_set = set(t.get("depends_on", []))
                input_deps = set(
                    v["from_task"] for v in t.get("inputs", {}).values()
                    if isinstance(v, dict) and "from_task" in v
                )
                orphan_inputs = input_deps - task_deps_set
                if orphan_inputs:
                    fmt_errors.append(
                        f"task[{t['task_id']}] inputs 引用了未在 depends_on 中声明的任务：{sorted(orphan_inputs)}"
                    )

            if fmt_errors:
                err_msg = "inputs 格式错误（将重试）：\n" + "\n".join(fmt_errors)
                print(f"  ⚠️ Planner 第 {attempt+1} 次：{err_msg}")
                raise ValueError(err_msg)

            print(f"  ✅ 规划完成（{len(task_plan)} 个任务）：")
            for t in task_plan:
                dep_str = f" depends_on={t['depends_on']}" if t['depends_on'] else ""
                print(f"     [{t['task_id']}] {t['agent']:12s} → {t['description'][:45]}{dep_str}")
            break

        except Exception as e:
            print(f"  ⚠️ Planner 第 {attempt+1} 次失败：{e}")
            if attempt == max_retries - 1:
                print("  ❌ Planner 全部失败，终止")
                return {**state, "next_agent": "FINISH", "task_plan": []}

    return {
        **state,
        "task_plan":       task_plan,
        "current_task_id": task_plan[0]["task_id"] if task_plan else 0,
        "next_agent":      "",
    }


# ══════════════════════════════════════════════════════
# 8. 并行调度核心
# ══════════════════════════════════════════════════════

def _topo_layers(tasks: list[Task]) -> list[list[Task]]:
    """
    拓扑 BFS 分层。返回 [[layer0], [layer1], ...]：
    - 同层内任务互无依赖，可 asyncio.gather() 并行执行
    - 层与层之间严格串行（后层依赖前层全部完成）
    """
    done_ids: set[int] = set()
    layers: list[list[Task]] = []
    remaining = list(tasks)

    while remaining:
        layer = [
            t for t in remaining
            if all(dep in done_ids for dep in t.get("depends_on", []))
        ]
        if not layer:
            print(f"  ⚠️ [topo] 依赖无法满足，剩余任务强制入队：{[t['task_id'] for t in remaining]}")
            layer = remaining
        layers.append(layer)
        done_ids |= {t["task_id"] for t in layer}
        remaining = [t for t in remaining if t not in layer]

    return layers


async def _spawn_session_for(
    agent_name: str,
    stack: AsyncExitStack,
    use_sse: bool = False,
) -> ClientSession:
    """为单个任务独立 spawn 一个 MCP session。"""
    if agent_name in ("math_agent",):
        stdio_params = math_mcp_params
        sse_url      = _MATH_PROXY_SSE_URL
    elif agent_name in ("file_agent",):
        stdio_params = filesystem_mcp_params
        sse_url      = _FS_PROXY_SSE_URL
    elif agent_name in ("db_agent",):
        stdio_params = db_mcp_params
        sse_url      = _DB_SERVER_SSE_URL
    else:
        stdio_params = mcp_params
        sse_url      = _SERVER_SSE_URL

    if use_sse:
        r, w = await stack.enter_async_context(sse_client(sse_url))
    else:
        r, w = await stack.enter_async_context(stdio_client(stdio_params()))

    session = await stack.enter_async_context(ClientSession(r, w))
    await session.initialize()
    return session


async def run_agent_isolated(
    task: Task,
    system_prompt: str,
    use_sse: bool = False,
) -> str:
    """
    单任务执行单元：独立 spawn session → load_tools → run → close session。
    返回任务结果字符串，不修改全局状态。
    """
    agent_name = task.get("agent", "default_agent")
    intent     = task.get("_resolved_description") or task.get("description", "")

    print(f"\n🤖 [{agent_name}] 任务[{task['task_id']}] 开始（独立 session）：{intent[:60]}")
    t0 = time.perf_counter()

    async with AsyncExitStack() as stack:
        try:
            session = await _spawn_session_for(agent_name, stack, use_sse=use_sse)
            tools   = await load_tools(session)
        except Exception as exc:
            print(f"  ❌ [{agent_name}] session 启动失败：{exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return f"（session 启动失败：{exc}）"

        if not tools:
            print(f"  ⚠️ [{agent_name}] 没有可用工具，直接 LLM 回答")
            resp = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=intent),
            ])
            return _extract_llm_content(resp)

        llm_with_tools = llm.bind_tools(tools)
        tool_map       = {t.name: t for t in tools}
        msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=intent),
        ]

        max_steps     = 6
        last_response = None

        for step in range(max_steps):
            response      = await llm_with_tools.ainvoke(msgs)
            last_response = response
            msgs.append(response)

            if isinstance(response, dict):
                tool_calls   = response.get("tool_calls", [])
            else:
                tool_calls   = getattr(response, "tool_calls", []) or []

            if not tool_calls:
                print(f"  ✅ [{agent_name}] task[{task['task_id']}] step={step} 完成")
                break

            print(f"  🔧 [{agent_name}] task[{task['task_id']}] step={step} 工具调用：{[tc['name'] for tc in tool_calls]}")

            for tc in tool_calls:
                tool = tool_map.get(tc["name"])
                if tool:
                    args        = {k: v for k, v in tc["args"].items() if v is not None}
                    result_text = await tool.coroutine(**args)
                else:
                    result_text = f"❌ 未找到工具：{tc['name']}"
                msgs.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
        else:
            print(f"  ⚠️ [{agent_name}] task[{task['task_id']}] 达到最大步数 {max_steps}")

    elapsed = time.perf_counter() - t0
    print(f"  ⏱️ [{agent_name}] task[{task['task_id']}] 耗时 {elapsed:.2f}s")
    return _extract_llm_content(last_response) if last_response else "（无结果）"


# ══════════════════════════════════════════════════════
# 9. direct_answer_node（不变）
# ══════════════════════════════════════════════════════


async def _run_direct_task(task: Task, state: AgentState) -> str:
    intent = task.get("description", "")
    print(f"\n  💬 direct_answer 任务[{task['task_id']}]：{intent[:60]}")

    # ★ 带上对话历史
    msgs = state.get("messages", [])
    history_msgs = msgs[:-1] if len(msgs) > 1 else []

    response = await llm.ainvoke([
        SystemMessage(content=(
            "你是一个友善的 AI 助手。请只回答当前分配给你的这一个子任务，"
            "不要回答用户原始消息中的其他问题。"
        )),
        *history_msgs,           # ← 插入历史
        HumanMessage(content=intent),
    ])
    answer = _extract_llm_content(response)
    print(f"  ✅ direct_answer 任务[{task['task_id']}] 完成：{answer[:60]}")
    return answer


# ══════════════════════════════════════════════════════
# 10. parallel_executor_node（替代原 supervisor_node）
# ══════════════════════════════════════════════════════

def _use_sse() -> bool:
    return os.environ.get("MCP_USE_SSE", "0") == "1"


async def parallel_executor_node(state: AgentState) -> AgentState:
    """
    核心并行调度节点。

    checkpoint 失败恢复机制：
      task["status"] == "done" 的任务会被跳过，不重复执行。
      如果上次执行中途失败（比如某个 MCP server 超时），
      重新 invoke 同一个 thread_id 时，已完成的任务不会重跑。
    """
    task_plan: list[Task] = state.get("task_plan", [])

    if not task_plan:
        print("\n🏁 [ParallelExecutor] task_plan 为空 → 跳过")
        return {**state, "next_agent": "FINISH"}

    # ── checkpoint 失败恢复：过滤掉已完成的任务 ──────────────────────
    # 什么时候会有 status=="done" 的任务？
    #   场景：上次执行完成了任务0和任务1，但任务2失败了，整个 graph 报错退出。
    #   下次用同一个 thread_id 重新 invoke，checkpoint 恢复了上次的 state，
    #   task_plan 里任务0和任务1的 status 已经是 "done"。
    #   这里过滤掉它们，只执行 pending/failed 的任务。
    pending_tasks = [t for t in task_plan if t.get("status") != "done"]

    if not pending_tasks:
        print("\n🏁 [ParallelExecutor] 所有任务已完成（从 checkpoint 恢复）→ 直接汇总")
        return {**state, "next_agent": "FINISH"}

    skipped = len(task_plan) - len(pending_tasks)
    if skipped > 0:
        print(f"\n⏭️  [ParallelExecutor] 跳过 {skipped} 个已完成任务（checkpoint 恢复）")

    # 分拓扑层（只对 pending 任务分层）
    layers = _topo_layers(pending_tasks)
    total  = len(pending_tasks)
    print(f"\n🚀 [ParallelExecutor] 共 {total} 个待执行任务，分 {len(layers)} 层")
    for i, layer in enumerate(layers):
        print(f"   层 {i}: {[t['task_id'] for t in layer]}")

    done_count = 0

    for layer_idx, layer in enumerate(layers):
        # ── 解析运行时 inputs（依赖上一层的结果）──────────────────────
        for task in layer:
            inputs         = task.get("inputs", {})
            resolved_parts = []

            declared_src_ids: set = set()
            for param_name, task_input in inputs.items():
                if isinstance(task_input, dict):
                    src_id = task_input.get("from_task")
                    field  = task_input.get("field", "result")
                elif isinstance(task_input, int):
                    src_id = task_input
                    field  = "result"
                else:
                    print(f"  ⚠️ inputs[{param_name}] 格式异常，跳过")
                    continue
                if src_id is not None:
                    declared_src_ids.add(src_id)
                src = next((t for t in task_plan if t["task_id"] == src_id), None)
                val = src.get(field, "") if src else ""
                resolved_parts.append(f"【{param_name}】= {val}")

            for dep_id in task.get("depends_on", []):
                if dep_id not in declared_src_ids:
                    src = next((t for t in task_plan if t["task_id"] == dep_id), None)
                    if src:
                        val = src.get("result", "")
                        resolved_parts.append(f"【任务{dep_id}的结果】= {val}")

            resolved_desc = task["description"]
            if resolved_parts:
                resolved_desc += "\n\n【运行时参数】\n" + "\n".join(resolved_parts)
            task["_resolved_description"] = resolved_desc
            task["status"] = "in_progress"

        # ── 并行执行当前层所有任务 ─────────────────────────────────────
        print(f"\n▶ [层 {layer_idx}] 并行执行 {len(layer)} 个任务："
              f"{[t['task_id'] for t in layer]}")
        t_layer_start = time.perf_counter()

        async def _exec_one(task: Task) -> tuple[int, str]:
            agent = task.get("agent", "default_agent")
            if agent == "direct":
                result = await _run_direct_task(task, state)
            else:
                system_prompt = AGENT_SYSTEM_PROMPTS.get(agent, DEFAULT_AGENT_SYSTEM_PROMPT)
                result = await run_agent_isolated(task, system_prompt, use_sse=_use_sse())
            return task["task_id"], result

        results: list[tuple[int, str]] = await asyncio.gather(
            *[_exec_one(t) for t in layer],
            return_exceptions=False,
        )

        # ── 将结果写回 task_plan ───────────────────────────────────────
        result_map = dict(results)
        for task in layer:
            task["status"] = "done"
            task["result"] = result_map.get(task["task_id"], "（无结果）")
            done_count += 1
            print(f"  ✔ 任务[{task['task_id']}] 完成：{task['result'][:60]}")

        layer_elapsed = time.perf_counter() - t_layer_start
        print(f"◀ [层 {layer_idx}] 全部完成，耗时 {layer_elapsed:.2f}s，"
              f"进度 {done_count}/{total}")

    print(f"\n🏁 [ParallelExecutor] 全部 {total} 个任务执行完毕")
    return {
        **state,
        "task_plan":  task_plan,
        "next_agent": "FINISH",
        "messages":   state["messages"],
    }


# ══════════════════════════════════════════════════════
# 11. final_answer_node（不变）
# ══════════════════════════════════════════════════════

async def final_answer_node(state: AgentState) -> AgentState:
    task_plan: list[Task] = state.get("task_plan", [])

    tool_tasks   = [t for t in task_plan if t.get("agent") != "direct"]
    direct_tasks = [t for t in task_plan if t.get("agent") == "direct"]

    all_results_lines: list[str] = []
    if direct_tasks:
        all_results_lines.append("【直接回答任务】")
        for t in direct_tasks:
            all_results_lines.append(
                f"  任务[{t['task_id']}]（{t['description']}）：{t['result']}"
            )

    if tool_tasks:
        all_results_lines.append("【工具执行任务】")
        for t in tool_tasks:
            all_results_lines.append(
                f"  任务[{t['task_id']}]（{t['description']}）：{t['result']}"
            )

    results_text = "\n".join(all_results_lines)
    print(f"\n  📝 汇总所有任务结果：\n{results_text}")
    
    msgs = state.get("messages", [])
    last_human = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage)),
        HumanMessage(content="")
    )

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


# ══════════════════════════════════════════════════════
# 12. Agent System Prompts
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
    "db_agent": (
        "你是电商数据库查询专家。数据库包含以下表：\n"
        "  users         （用户：id / name / email / age / city / status / created_at）\n"
        "  products      （商品：id / name / category_id / price / stock / status）\n"
        "  categories    （分类：id / name / parent_id）\n"
        "  orders        （订单：id / user_id / status / total / shipping_address / created_at）\n"
        "  order_items   （订单明细：id / order_id / product_id / qty / unit_price）\n"
        "  reviews       （评价：id / user_id / product_id / rating / comment）\n"
        "  inventory_log （库存日志：id / product_id / delta / reason / created_at）\n\n"
        "【工具使用规则 — 严格遵守】\n"
        "  1. 禁止使用 ask_db。ask_db 内部会再次调用 LLM，速度极慢且容易超时。\n"
        "  2. 直接根据上面的表结构自己写 SQL，使用 query_db 执行 SELECT 查询。\n"
        "  3. INSERT / UPDATE 操作使用 execute_db。\n"
        "  4. 只有在极不确定表结构时才调用 get_schema 一次，之后立即用 query_db。\n"
        "  5. 拿到查询结果后，用中文简洁总结，不要把原始 JSON 全部输出给用户。\n\n"
        "【常用 SQL 模式参考】\n"
        "  - 按城市筛选用户：SELECT * FROM users WHERE city = 'Toronto' AND status = 'active'\n"
        "  - 统计分组：SELECT city, COUNT(*) as cnt FROM users GROUP BY city ORDER BY cnt DESC\n"
        "  - 联表查询：SELECT u.name, o.total FROM orders o JOIN users u ON o.user_id = u.id\n"
        "  - TOP N：SELECT * FROM products ORDER BY price DESC LIMIT 5\n"
    ),
    "default_agent": (
        "你是通用任务执行专家。根据任务描述，调用合适的工具完成任务，给出简洁结果。"
    ),
}

DEFAULT_AGENT_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPTS["default_agent"]


# ══════════════════════════════════════════════════════
# 13. 图构建
# ══════════════════════════════════════════════════════

# ★ CHECKPOINT 改动6a/6：build_graph 接收 checkpointer 参数
#
# 改动前：def build_graph() -> Any:  ...  return g.compile()
# 改动后：def build_graph(checkpointer=None) -> Any:  ...  return g.compile(checkpointer=checkpointer)
#
# 为什么用参数而不是直接写死 _checkpointer？
#   1. webapp.py 阶段二会注入 SqliteSaver，参数形式方便替换
#   2. 测试时可以传 None（不用 checkpoint），方便单元测试
#   3. 遵循"依赖注入"原则：函数不依赖全局状态，更容易维护
#
# compile(checkpointer=...) 做了什么？
#   - 告诉 LangGraph：每次节点执行完，把当前 state 快照存到 checkpointer
#   - 每次 invoke 时如果传了 thread_id，先从 checkpointer 恢复上次的 state
#   - 这就是"记忆"的来源
def build_graph(checkpointer=None) -> Any:
    """
    图结构（极简）：
        planner → parallel_executor → final_answer → END
    """

    def planner_route(state: AgentState) -> str:
        if state.get("next_agent") == "FINISH":
            return "END"
        return "parallel_executor"

    g = StateGraph(AgentState)
    g.add_node("planner",           planner_node)
    g.add_node("parallel_executor", parallel_executor_node)
    g.add_node("final_answer",      final_answer_node)

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", planner_route, {
        "END":               END,
        "parallel_executor": "parallel_executor",
    })

    g.add_edge("parallel_executor", "final_answer")
    g.add_edge("final_answer",      END)

    # checkpointer=None 时行为与原来完全一致（不启用 checkpoint）
    return g.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════
# 14. 图实例
# ══════════════════════════════════════════════════════

# ★ CHECKPOINT 改动6b/6：初始化时就传入 checkpointer
#
# 注意：这里的 graph 是初始的"空壳"，_init_registry() 会在 MCP 连接成功后重建它。
# 重要的是两次 build_graph 都传的是同一个 _checkpointer 实例。
graph = build_graph(checkpointer=_checkpointer)


# ══════════════════════════════════════════════════════
# 15. __main__ —— 交互式 CLI / 批量测试
# ══════════════════════════════════════════════════════
if __name__ == "__main__":

    BATCH_MODE = False  # True → 自动跑完 QUESTIONS；False → 交互式 CLI
    QUESTIONS = [
        "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",
        "列出 File_Agent 目录下的所有文件，然后在其中创建一个名为 hello.txt 的文件，内容为：Hello from file_agent！",
        "查询所有来自 Toronto 的活跃用户",
    ]

    # ★ CHECKPOINT 改动7/6（额外改动）：_run_question 加入 thread_id 和 config
    #
    # 这是让 checkpoint 真正生效的"最后一步"。
    #
    # thread_id 是什么？
    #   就是一个字符串，用来区分不同的对话会话。
    #   同一个 thread_id = 同一个用户的同一个对话，
    #   checkpoint 会把这个对话的 state 存起来，下次接着用。
    #
    # config 是什么？
    #   LangGraph 的运行配置字典。
    #   {"configurable": {"thread_id": "xxx"}} 是固定格式，
    #   "configurable" 是 LangGraph 规定的命名空间，
    #   "thread_id" 是 MemorySaver/SqliteSaver 用来查找存档的 key。
    #
    # 不传 config 会怎样？
    #   graph.ainvoke(...) 没有 config → 没有 thread_id → checkpoint 不生效
    #   每次都是全新会话，和没加 MemorySaver 一样。
    #
    # thread_id 的命名建议：
    #   CLI 开发测试：  "cli_user_1"（固定，重启进程前历史一直在）
    #   多用户区分：    f"user_{user_id}"
    #   每次问题隔离：  f"session_{int(time.time())}"（每次新建）
    async def _run_question(q: str, thread_id: str = "cli_user_1") -> None:
        print(f"\n{'━' * 60}\n❓ {q}\n{'━' * 60}")
        print(f"📌 thread_id: {thread_id}")   # 打印出来，方便确认 checkpoint 在哪个会话下

        # config 是传给 graph.ainvoke 的运行时配置
        # 有了这个，LangGraph 才知道要存/读哪个 thread_id 的 checkpoint
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = await graph.ainvoke(
                {
                    "messages":        [HumanMessage(content=q)],
                    "task_plan":       [],
                    "current_task_id": 0,
                    "next_agent":      "",
                },
                config=config,   # ← 关键：传入 config
            )
            answer = _get_message_content(result["messages"][-1])
            print(f"\n{'═' * 60}")
            print(f"✨ 最终答案：\n{answer}")
            print(f"{'═' * 60}")

            # ── 验证 checkpoint 是否生效：打印当前 thread 的消息数 ──────
            # 如果 checkpoint 正常工作，第2次问题后 messages 数量应该 > 第1次
            # （因为 add_messages 会追加，不会覆盖）
            try:
                saved_state = _checkpointer.get(config)
                if saved_state:
                    channel_values = saved_state.get("channel_values", {})
                    msgs_in_cp = channel_values.get("messages", [])
                    print(f"💾 [Checkpoint] thread '{thread_id}' 已存 {len(msgs_in_cp)} 条消息")
                else:
                    print(f"💾 [Checkpoint] thread '{thread_id}' 暂无存档")
            except Exception as cp_err:
                print(f"💾 [Checkpoint] 读取存档时出错：{cp_err}")

        except Exception as e:
            print(f"\n❌ 执行出错：{e}")
            traceback.print_exc()

    async def _interactive() -> None:
        print("\n" + "═" * 60)
        print("🤖  MCP Multi-Agent 并行 CLI 就绪（已启用 MemorySaver Checkpoint）")
        print("    输入问题后回车执行，输入 'quit' / 'exit' / 'q' 退出")
        print("    输入 'batch' 快速跑完 QUESTIONS 列表")
        print("    输入 'new' 开始新会话（新 thread_id）")
        print("═" * 60)

        # 交互式模式下，同一次运行里所有问题共享一个 thread_id
        # 这样可以验证"同一个对话里的多轮记忆"
        session_thread_id = f"interactive_{int(time.time())}"
        print(f"📌 当前会话 thread_id: {session_thread_id}")

        while True:
            try:
                q = input("\n❓ > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            if q.lower() == "batch":
                for bq in QUESTIONS:
                    await _run_question(bq, thread_id="batch_session")
                continue
            if q.lower() == "new":
                session_thread_id = f"interactive_{int(time.time())}"
                print(f"📌 新会话已开始，thread_id: {session_thread_id}")
                continue
            await _run_question(q, thread_id=session_thread_id)

    async def _batch() -> None:
        print(f"\n🚀 批量测试模式，共 {len(QUESTIONS)} 个问题")
        # 批量模式：所有问题用同一个 thread_id
        # 可以验证：第2个问题执行时，checkpoint 里已有第1个问题的记录
        for q in QUESTIONS:
            await _run_question(q, thread_id="batch_session")

    async def main():
        await _start_mcp_sessions_stdio()
        try:
            if BATCH_MODE:
                await _batch()
            else:
                await _interactive()
        finally:
            await _stop_mcp_sessions()

    asyncio.run(main())