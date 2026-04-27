"""
src/langgraph_stdio_agent.py

★ db_agent 集成说明（在原有架构基础上新增）：

  【新增 db_agent】
   - 连接 mcp_db_server/server.py，工具：ask_db / query_db / execute_db / get_schema
   - SSE 模式：连接 http://127.0.0.1:8003/sse（由 webapp.py lifespan 拉起）
   - STDIO 模式：直接 spawn mcp_db_server/server.py 子进程（__main__ 测试用）
   - AGENT_TOOL_PATTERNS 新增 db_agent 工具名匹配
   - AGENT_DESCRIPTIONS / AGENT_TRIGGER_KEYWORDS 新增 db_agent 条目

以下为原有修复（保持不变）：
  【重构1】移除 router 节点
  【重构2】direct_answer_node 只回答当前子任务
  【重构3】planner_node JSON 解析失败重试策略
  【修复4】supervisor_node in_progress 容错
  【修复5】_extract_json 统一抽取
  【修复6】final_answer_node 混合任务时 direct 结果纳入汇总
  【修复7】强化 Planner 工具感知
  【修复8】run_agent 工具调用强制兜底
  【修复9】全局兼容 llm.ainvoke() 返回 dict
  【修复10】兼容 LangGraph Studio 下 messages 反序列化为 dict
  【修复11】彻底移除 asyncio.Event，改用 _registry.agents 判断就绪
  【修复12】_ensure_registry() 调用时机提前到 task_plan 判断之前
  【SSE改造】前端改用 SSE 传输，绕开 Windows ProactorLoop 限制
  【Lock修复】_lazy_init_lock 改为惰性创建
"""

import asyncio
import json
import os
import re
import sys
import traceback
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
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
DB_SERVER_PATH = Path(__file__).parent / "mcp_db_server" / "server.py"   # ★ 新增

_MCP_FS_ENV = os.getenv("MCP_FS_BASE_DIR", "")
if _MCP_FS_ENV:
    _FS_BASE_DIR = Path(_MCP_FS_ENV)
else:
    _FS_BASE_DIR = Path(__file__).parent.parent / "File_Agent"

_SERVER_PORT    = int(os.getenv("MCP_SERVER_PORT",    "8001"))
_FS_PROXY_PORT  = int(os.getenv("MCP_FS_PROXY_PORT",  "8002"))
_DB_SERVER_PORT = int(os.getenv("MCP_DB_SERVER_PORT", "8003"))   # ★ 新增

_SERVER_SSE_URL    = f"http://127.0.0.1:{_SERVER_PORT}/sse"
_FS_PROXY_SSE_URL  = f"http://127.0.0.1:{_FS_PROXY_PORT}/sse"
_DB_SERVER_SSE_URL = f"http://127.0.0.1:{_DB_SERVER_PORT}/sse"   # ★ 新增


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

def db_mcp_params() -> StdioServerParameters:                     # ★ 新增
    """后端测试：以 stdio 模式启动 db_server.py"""
    return StdioServerParameters(
        command=sys.executable,
        args=["-u", str(DB_SERVER_PATH)],
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
    "file_agent": ["read_file", "write_file", "edit_file",
                   "read_multiple_files", "list_directory", "create_directory",
                   "move_file", "search_files", "get_file_info",
                   "list_allowed_directories", "file_*"],
    # ★ 新增 db_agent
    "db_agent":   ["ask_db", "query_db", "execute_db", "get_schema",
                   "db_*", "sql_*"],
}

AGENT_DESCRIPTIONS: dict[str, str] = {
    "math_agent": "数学计算（加减乘除、幂、开方等数值运算）",
    "data_agent": "数据分析（统计、聚合、分组、过滤等结构化数据处理）",
    "http_agent": "网络请求（GET/POST、访问 URL、调用外部 API）",
    "file_agent": "文件操作（读写文件、列出目录、创建目录、移动/搜索文件）",
    # ★ 新增
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
    # ★ 新增
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
_lazy_init_lock: asyncio.Lock | None = None
_mcp_exit_stack: AsyncExitStack | None = None


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
# MCP Session 管理
# ══════════════════════════════════════════════════════

async def _start_mcp_sessions() -> None:
    """
    前端路径（langgraph dev）：SSE 连接三个独立进程：
      1. server.py    @ 8001  → math / data / http 工具
      2. mcp-proxy    @ 8002  → 文件系统工具
      3. db_server.py @ 8003  → 数据库工具 ★ 新增
    """
    global _mcp_exit_stack
    if _mcp_exit_stack is not None:
        print("⚠️ [MCP] _start_mcp_sessions 重复调用，跳过")
        return

    print(f"🔍 [MCP] platform={sys.platform}  python={sys.executable}")
    print(f"🔍 [MCP] server SSE URL:     {_SERVER_SSE_URL}")
    print(f"🔍 [MCP] filesystem SSE URL: {_FS_PROXY_SSE_URL}")
    print(f"🔍 [MCP] db server SSE URL:  {_DB_SERVER_SSE_URL}")

    stack = AsyncExitStack()
    all_tools: list[StructuredTool] = []

    # ── 1. server.py MCP（SSE @ 8001）────────────────────────────────
    try:
        r1, w1 = await stack.enter_async_context(sse_client(_SERVER_SSE_URL))
        s1     = await stack.enter_async_context(ClientSession(r1, w1))
        await s1.initialize()
        server_tools = await load_tools(s1)
        print(f"✅ [MCP] server.py 工具：{[t.name for t in server_tools]}")
        all_tools.extend(server_tools)
    except Exception as exc:
        print(f"❌ [MCP] server.py SSE 连接失败：{exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # ── 2. mcp-server-filesystem（SSE @ 8002）────────────────────────
    try:
        r2, w2 = await stack.enter_async_context(sse_client(_FS_PROXY_SSE_URL))
        s2     = await stack.enter_async_context(ClientSession(r2, w2))
        await s2.initialize()
        fs_tools = await load_tools(s2)
        print(f"✅ [MCP] filesystem 工具：{[t.name for t in fs_tools]}")
        all_tools.extend(fs_tools)
    except Exception as exc:
        print(f"❌ [MCP] filesystem SSE 连接失败（8002）：{exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # ── 3. ★ 新增 db_server.py MCP（SSE @ 8003）──────────────────────
    try:
        r3, w3 = await stack.enter_async_context(sse_client(_DB_SERVER_SSE_URL))
        s3     = await stack.enter_async_context(ClientSession(r3, w3))
        await s3.initialize()
        db_tools = await load_tools(s3)
        print(f"✅ [MCP] db_server 工具：{[t.name for t in db_tools]}")
        all_tools.extend(db_tools)
    except Exception as exc:
        print(f"❌ [MCP] db_server SSE 连接失败（8003）：{exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    if not all_tools:
        print("❌ [MCP] 所有 MCP 连接失败，registry 未就绪", file=sys.stderr)
        try:
            await stack.aclose()
        except Exception:
            pass
        return

    _mcp_exit_stack = stack
    _tools.clear()
    _tools.extend(all_tools)
    _init_registry(all_tools)
    print(f"🚀 [MCP] 就绪，共 {len(all_tools)} 个工具，agents: {_registry.agents}")


async def _start_mcp_sessions_stdio() -> None:
    """
    后端路径（__main__ 直接运行）：stdio_client spawn 三个子进程：
      1. server.py          → math / data / http 工具
      2. mcp-server-filesystem → 文件系统工具
      3. db_server.py       → 数据库工具 ★ 新增
    """
    global _mcp_exit_stack
    if _mcp_exit_stack is not None:
        return

    print(f"🔍 [MCP-stdio] SERVER_PATH    = {SERVER_PATH}  (exists={SERVER_PATH.exists()})")
    print(f"🔍 [MCP-stdio] DB_SERVER_PATH = {DB_SERVER_PATH}  (exists={DB_SERVER_PATH.exists()})")
    print(f"🔍 [MCP-stdio] FS_BASE_DIR    = {_FS_BASE_DIR}  (exists={_FS_BASE_DIR.exists()})")

    stack = AsyncExitStack()
    all_tools: list[StructuredTool] = []

    # ── 1. server.py（stdio）──────────────────────────────────────────
    if not SERVER_PATH.exists():
        print(f"❌ [MCP-stdio] 找不到 MCP server：{SERVER_PATH}", file=sys.stderr)
    else:
        try:
            r1, w1 = await stack.enter_async_context(stdio_client(mcp_params()))
            s1     = await stack.enter_async_context(ClientSession(r1, w1))
            await s1.initialize()
            server_tools = await load_tools(s1)
            print(f"✅ [MCP-stdio] server.py 工具：{[t.name for t in server_tools]}")
            all_tools.extend(server_tools)
        except Exception as exc:
            print(f"❌ [MCP-stdio] server.py 启动失败：{exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # ── 2. mcp-server-filesystem（stdio）─────────────────────────────
    try:
        r2, w2 = await stack.enter_async_context(stdio_client(filesystem_mcp_params()))
        s2     = await stack.enter_async_context(ClientSession(r2, w2))
        await s2.initialize()
        fs_tools = await load_tools(s2)
        print(f"✅ [MCP-stdio] filesystem 工具：{[t.name for t in fs_tools]}")
        all_tools.extend(fs_tools)
    except Exception as exc:
        print(f"❌ [MCP-stdio] mcp-server-filesystem 启动失败：{exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # ── 3. ★ 新增 db_server.py（stdio）───────────────────────────────
    if not DB_SERVER_PATH.exists():
        print(f"❌ [MCP-stdio] 找不到 DB MCP server：{DB_SERVER_PATH}", file=sys.stderr)
    else:
        try:
            r3, w3 = await stack.enter_async_context(stdio_client(db_mcp_params()))
            s3     = await stack.enter_async_context(ClientSession(r3, w3))
            await s3.initialize()
            db_tools = await load_tools(s3)
            print(f"✅ [MCP-stdio] db_server 工具：{[t.name for t in db_tools]}")
            all_tools.extend(db_tools)
        except Exception as exc:
            print(f"❌ [MCP-stdio] db_server 启动失败：{exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not all_tools:
        print("❌ [MCP-stdio] 所有 MCP 连接失败", file=sys.stderr)
        try:
            await stack.aclose()
        except Exception:
            pass
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
    _registry = ToolRegistry.build(tools) if tools else ToolRegistry()


# ══════════════════════════════════════════════════════
# 公共工具函数（与原版完全一致）
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


def _task_needs_tool_agent(description: str) -> str | None:
    desc_lower = description.lower()
    for agent, keywords in AGENT_TRIGGER_KEYWORDS.items():
        if agent not in _registry.agents:
            continue
        if any(kw.lower() in desc_lower for kw in keywords):
            return agent
    return None


# ══════════════════════════════════════════════════════
# 7. Planner
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
  - 任何数据库查询（订单/用户/商品/库存/评价等）→ db_agent
  - 任何文件操作（读写文件、列目录等）→ file_agent

❌ 严禁使用 direct 的情况（以下场景必须用工具 agent）：
  - "计算 3+5" → 必须用 math_agent
  - "访问 https://..." → 必须用 http_agent
  - "查询订单" / "有多少用户" / "库存不足的商品" → 必须用 db_agent
  - "列出文件" / "读取文件" → 必须用 file_agent

✅ 可以使用 direct 的情况（仅限以下场景）：
  - 闲聊、问候（如"你好"、"介绍一下你自己"）
  - 纯知识性问答（如"什么是加权平均数"）
  - 不涉及任何计算、网络请求、数据处理、数据库操作的场景

━━ db_agent 使用说明 ━━
db_agent 连接的是电商数据库，包含以下表：
  users（用户）、products（商品）、categories（分类）、
  orders（订单）、order_items（订单明细）、reviews（评价）、inventory_log（库存日志）
优先使用 ask_db 工具（自然语言查询），只有在明确知道 SQL 时才用 query_db。

━━ 其他规则（严格遵守）━━
1. description 只写任务意图，绝不提前计算数值或给出最终答案
2. inputs 声明运行时需要从哪些前置任务获取参数
3. depends_on 从 inputs 的 from_task 自动推导
4. 没有依赖的任务：inputs 为 {{}}，depends_on 为 []
5. 同一个 agent 可出现多次
6. 任务按拓扑顺序排列（被依赖的任务排在前面）

严格只输出 JSON 数组，不要有任何其他内容、代码块标记或说明文字。示例：
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
"""


async def planner_node(state: AgentState) -> AgentState:
    await _ensure_registry()

    msgs = state.get("messages", [])
    if not msgs:
        return {**state, "next_agent": "FINISH", "task_plan": []}

    user_msg = _get_message_content(msgs[0])
    print(f"\n📋 [Planner] 规划任务：{user_msg[:80]}")

    max_retries = 3
    task_plan: list[Task] = []

    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([
                SystemMessage(content=_planner_system()),
                HumanMessage(content=user_msg),
            ])
            raw = _extract_json(_extract_llm_content(response))
            task_plan = json.loads(raw)

            if not isinstance(task_plan, list) or len(task_plan) == 0:
                raise ValueError("Empty or invalid task plan")

            # 强制补齐必要字段
            for t in task_plan:
                t.setdefault("status", "pending")
                t.setdefault("result", "")
                t.setdefault("_resolved_description", "")
                t.setdefault("inputs", {})
                t.setdefault("depends_on", [])

            print(f"  ✅ 规划完成（{len(task_plan)} 个任务）：")
            for t in task_plan:
                print(f"     [{t['task_id']}] {t['agent']:12s} → {t['description'][:50]}")
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
# 8. Supervisor
# ══════════════════════════════════════════════════════

async def supervisor_node(state: AgentState) -> AgentState:
    task_plan: list[Task] = state.get("task_plan", [])

    # 找下一个 pending 任务
    next_task = next((t for t in task_plan if t.get("status") == "pending"), None)

    if next_task is None:
        print("\n🏁 [Supervisor] 所有任务完成 → FINISH")
        return {**state, "next_agent": "FINISH"}

    # 检查依赖是否已完成
    deps_done = all(
        any(t["task_id"] == dep_id and t.get("status") in ("done", "in_progress")
            for t in task_plan)
        for dep_id in next_task.get("depends_on", [])
    )
    if not deps_done:
        # 依赖未完成，跳过等待（理论上拓扑排序已保证顺序，这里只是保险）
        print(f"  ⏳ [Supervisor] 任务 [{next_task['task_id']}] 依赖未就绪，跳过")
        return {**state, "next_agent": "FINISH"}

    # 解析运行时参数
    inputs = next_task.get("inputs", {})
    resolved_parts = []
    for param_name, task_input in inputs.items():
        src_id  = task_input.get("from_task")
        field   = task_input.get("field", "result")
        src     = next((t for t in task_plan if t["task_id"] == src_id), None)
        val     = src.get(field, "") if src else ""
        resolved_parts.append(f"【{param_name}】= {val}")

    resolved_desc = next_task["description"]
    if resolved_parts:
        resolved_desc += "\n\n【运行时参数】\n" + "\n".join(resolved_parts)
    next_task["_resolved_description"] = resolved_desc

    next_task["status"] = "in_progress"
    agent = next_task.get("agent", "direct")
    print(f"\n🎯 [Supervisor] 分配任务 [{next_task['task_id']}] → {agent}")
    print(f"   描述：{next_task['description'][:60]}")

    return {
        **state,
        "current_task_id": next_task["task_id"],
        "next_agent":      agent,
        "task_plan":       task_plan,
    }


# ══════════════════════════════════════════════════════
# 9. Replanner
# ══════════════════════════════════════════════════════

async def replanner_node(state: AgentState) -> AgentState:
    """当前任务完成后，检查是否需要重新规划（目前直接返回，预留扩展点）"""
    task_plan = state.get("task_plan", [])
    done = sum(1 for t in task_plan if t.get("status") == "done")
    total = len(task_plan)
    print(f"\n🔄 [Replanner] 进度 {done}/{total}")
    return state


# ══════════════════════════════════════════════════════
# 10. run_agent（通用工具执行节点）
# ══════════════════════════════════════════════════════

async def run_agent(state: AgentState, agent_name: str, system_prompt: str) -> AgentState:
    task_plan:    list[Task] = state.get("task_plan", [])
    task_id:      int        = state.get("current_task_id", 0)
    current_task             = next((t for t in task_plan if t["task_id"] == task_id), None)

    if not current_task:
        print(f"  ⚠️ [{agent_name}] 找不到任务 {task_id}")
        return state

    intent    = current_task.get("_resolved_description") or current_task.get("description", "")
    tools     = _registry.tools_for(agent_name)
    tool_names = _registry.tool_names_for(agent_name)

    print(f"\n🤖 [{agent_name}] 执行任务：{intent[:80]}")
    print(f"   可用工具：{tool_names}")

    if not tools:
        print(f"  ⚠️ [{agent_name}] 没有可用工具，直接 LLM 回答")
        resp = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=intent),
        ])
        current_task["status"] = "done"
        current_task["result"] = _extract_llm_content(resp)
        return {**state, "task_plan": task_plan}

    # 绑定工具
    llm_with_tools = llm.bind_tools(tools)
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=intent),
    ]

    max_steps    = 6
    last_response = None

    for step in range(max_steps):
        response = await llm_with_tools.ainvoke(msgs)
        last_response = response
        msgs.append(response)

        # 兼容 dict 响应
        if isinstance(response, dict):
            tool_calls = response.get("tool_calls", [])
            text_content = response.get("content", "")
        else:
            tool_calls   = getattr(response, "tool_calls", []) or []
            text_content = getattr(response, "content", "") or ""

        if not tool_calls:
            print(f"  ✅ [{agent_name}] step={step} 无工具调用，任务完成")
            break

        print(f"  🔧 [{agent_name}] step={step} 工具调用：{[tc['name'] for tc in tool_calls]}")

        # 执行工具调用
        for tc in tool_calls:
            tool = _registry.get_tool(tc["name"])
            if tool:
                args        = {k: v for k, v in tc["args"].items() if v is not None}
                result_text = await tool.coroutine(**args)
            else:
                result_text = f"❌ 未找到工具：{tc['name']}"
            msgs.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
    else:
        print(f"  ⚠️ [{agent_name}] 达到最大步数 {max_steps}，强制终止")

    current_task["status"] = "done"
    current_task["result"] = _extract_llm_content(last_response) if last_response else "（无结果）"

    return {**state, "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 11. 图构建
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
    # ★ 新增 db_agent
    "db_agent": (
        "你是电商数据库查询专家。数据库包含以下表：\n"
        "  users（用户：id/name/email/age/city/status）\n"
        "  products（商品：id/name/category_id/price/stock/status）\n"
        "  categories（分类：id/name/parent_id）\n"
        "  orders（订单：id/user_id/status/total/shipping_address/created_at）\n"
        "  order_items（订单明细：id/order_id/product_id/qty/unit_price）\n"
        "  reviews（评价：id/user_id/product_id/rating/comment）\n"
        "  inventory_log（库存日志：id/product_id/delta/reason/created_at）\n\n"
        "工具使用原则：\n"
        "  - 优先使用 ask_db（自然语言输入，AI 自动生成 SQL）\n"
        "  - 需要查看表结构时使用 get_schema\n"
        "  - 明确知道 SQL 时可直接用 query_db（SELECT）或 execute_db（INSERT/UPDATE）\n"
        "  - 得到结果后用中文简洁总结，不要重复罗列原始数据"
    ),
}

DEFAULT_AGENT_SYSTEM_PROMPT = (
    "你是通用任务执行专家。根据任务描述，调用合适的工具完成任务，给出简洁结果。"
)


def build_graph() -> Any:

    async def direct_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])
        task_id: int          = state.get("current_task_id", 0)
        current_task = next((t for t in task_plan if t["task_id"] == task_id), None)

        if not current_task:
            print("  ⚠️ direct_answer：找不到当前任务")
            return state

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

        return {
            **state,
            "messages":  state["messages"] + [AIMessage(content=answer)],
            "task_plan": task_plan,
        }

    async def final_answer_node(state: AgentState) -> AgentState:
        task_plan: list[Task] = state.get("task_plan", [])

        tool_tasks   = [t for t in task_plan if t.get("agent") != "direct"]
        direct_tasks = [t for t in task_plan if t.get("agent") == "direct"]

        if not tool_tasks:
            print("\n  📝 所有任务均为 direct，最终回答已在消息流中")
            return state

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
            "messages": state["messages"] + [AIMessage(content=_extract_llm_content(response))],
        }

    def planner_route(state: AgentState) -> str:
        if state.get("next_agent") == "FINISH":
            return "END"
        return "supervisor"

    def supervisor_route(state: AgentState) -> str:
        next_node = state.get("next_agent", "FINISH")
        if next_node == "FINISH":
            return "final_answer"
        return next_node

    def make_agent_node(name: str):
        system_prompt = AGENT_SYSTEM_PROMPTS.get(name, DEFAULT_AGENT_SYSTEM_PROMPT)
        async def _node(state: AgentState) -> AgentState:
            return await run_agent(state, name, system_prompt)
        _node.__name__ = name
        return _node

    g = StateGraph(AgentState)

    g.add_node("planner",       planner_node)
    g.add_node("supervisor",    supervisor_node)
    g.add_node("replanner",     replanner_node)
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("final_answer",  final_answer_node)

    known_agents = _registry.agents or list(AGENT_SYSTEM_PROMPTS.keys())
    for agent_name in known_agents:
        g.add_node(agent_name, make_agent_node(agent_name))

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", planner_route, {
        "END":        END,
        "supervisor": "supervisor",
    })

    agent_routes = {name: name for name in known_agents}
    agent_routes["direct_answer"] = "direct_answer"
    agent_routes["final_answer"]  = "final_answer"
    g.add_conditional_edges("supervisor", supervisor_route, agent_routes)

    g.add_edge("direct_answer", "supervisor")

    for agent_name in known_agents:
        g.add_edge(agent_name, "replanner")
    g.add_edge("replanner", "supervisor")

    g.add_edge("final_answer", END)

    return g.compile()


# ══════════════════════════════════════════════════════
# 12. 图实例
# ══════════════════════════════════════════════════════
graph = build_graph()


# ══════════════════════════════════════════════════════
# 13. __main__ —— 后端测试（stdio 模式，直接运行）
#     命令：uv run python src/langgraph_stdio_agent.py
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    QUESTIONS = [
        # ── 纯 DB 查询测试 ──
        "查询所有来自 Toronto 的活跃用户",
        # "统计每个城市的用户数量，按数量降序排列",
        # "找出销售额最高的前 5 个商品",
        # "查询所有状态为 completed 的订单，并显示对应的用户名称",
        # "哪些商品的库存为 0？",
        # "查询平均评分低于 3 分的商品",

        # ── 混合任务测试 ──
        # "查询 Toronto 用户数量，然后计算这个数字的平方",
    ]

    async def main():
        await _start_mcp_sessions_stdio()
        try:
            for q in QUESTIONS:
                print(f"\n{'━'*60}\n❓ {q}\n{'━'*60}")
                result = await graph.ainvoke({
                    "messages":        [HumanMessage(content=q)],
                    "task_plan":       [],
                    "current_task_id": 0,
                    "next_agent":      "",
                })
                print(f"\n✨ 最终答案：{_get_message_content(result['messages'][-1])}")
        finally:
            await _stop_mcp_sessions()

    asyncio.run(main())