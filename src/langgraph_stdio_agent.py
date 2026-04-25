"""
src/langgraph_stdio_agent.py

两种运行方式：
  1. uv run langgraph dev   → langgraph 调用 webapp.py lifespan 拉起子进程 + 初始化 SSE sessions
  2. python -m src.langgraph_stdio_agent  → __main__ 直接用 stdio_client 初始化工具

★ SSE 版本改造说明（在原有修复基础上新增）：

  【改造SSE】前端（langgraph dev）改用 SSE 传输，彻底绕开 Windows ProactorLoop 限制
   - 原问题：Windows + langgraph dev 使用 SelectorEventLoop，stdio_client 无法 spawn 子进程
   - 方案：_start_mcp_sessions_sse() 用 sse_client 连接 localhost:8001/8002
           子进程由 webapp.py lifespan 负责拉起，agent 只做 HTTP 连接，不 spawn 进程
   - 后端测试路径完全保留：__main__ 仍用 stdio_client，一条命令照跑

  【修复Lock】_lazy_init_lock 改为惰性创建，彻底解决跨 event loop 问题
   - 原问题：asyncio.Lock() 在模块加载时创建，绑定到主线程 loop
             langgraph dev 的 uvicorn 启动新 loop 后，Lock.acquire() 永远挂起
             导致 _ensure_registry() 看似"就绪"实则 registry 永远为空
   - 方案：_lazy_init_lock 初始为 None，第一次 _ensure_registry() 调用时
           在当前 loop 里惰性创建，确保 Lock 始终和调用方在同一个 loop

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
from mcp.client.stdio import stdio_client       # 后端测试（__main__）用
from mcp.client.sse import sse_client           # 前端测试（langgraph dev）用  ← ★ 新增
from mcp import StdioServerParameters
from pathlib import Path
from pydantic import create_model

# ★ load_dotenv() 和 find_dotenv() 内部都会调用同步的 os.getcwd()，
#   在 async 上下文（langgraph dev）里会被 blockbuster 拦截报 BlockingError。
#   改用 __file__ 推导 .env 绝对路径，完全不调用 os.getcwd()。
_dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(str(_dotenv_path), override=False)

# ★ Windows 下后端测试（__main__）需要 ProactorEventLoop 才能 spawn 子进程。
#   前端（langgraph dev）走 SSE，不再 spawn 子进程，此设置对前端无害。
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
# 2. MCP server 路径 & 启动参数（后端测试用）
# ══════════════════════════════════════════════════════
SERVER_PATH = Path(__file__).parent / "mcp_server_template" / "server.py"

# ★ _FS_BASE_DIR：优先读环境变量（绝对路径），否则基于 __file__ 推导，
#   避免用相对路径 + .resolve()（会调用 os.getcwd() 被 blockbuster 拦截）
_MCP_FS_ENV = os.getenv("MCP_FS_BASE_DIR", "")
if _MCP_FS_ENV:
    _FS_BASE_DIR = Path(_MCP_FS_ENV)          # 环境变量里已是绝对路径，直接用
else:
    _FS_BASE_DIR = Path(__file__).parent.parent / "File_Agent"  # 绝对路径，不调 getcwd

# ★ SSE 端点配置（前端测试用，由 webapp.py lifespan 拉起对应进程）
_SERVER_PORT   = int(os.getenv("MCP_SERVER_PORT",   "8001"))
_FS_PROXY_PORT = int(os.getenv("MCP_FS_PROXY_PORT", "8002"))
_SERVER_SSE_URL   = f"http://127.0.0.1:{_SERVER_PORT}/sse"
_FS_PROXY_SSE_URL = f"http://127.0.0.1:{_FS_PROXY_PORT}/sse"


def mcp_params() -> StdioServerParameters:
    """后端测试：以 stdio 模式启动 server.py"""
    return StdioServerParameters(
        command=sys.executable,
        args=["-u", str(SERVER_PATH)],
        env={"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **os.environ},
    )

def filesystem_mcp_params() -> StdioServerParameters:
    """后端测试：以 stdio 模式启动 filesystem MCP"""
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

AGENT_TRIGGER_KEYWORDS: dict[str, list[str]] = {
    "math_agent": ["计算", "加", "减", "乘", "除", "求和", "平均", "幂", "开方",
                   "+", "-", "×", "÷", "*", "/", "²", "√"],
    "http_agent": ["访问", "请求", "获取", "http", "https", "url", "api",
                   "fetch", "get ", "post "],
    "data_agent": ["分析", "统计", "分组", "聚合", "过滤", "排序", "数据集",
                   "dataframe", "pivot"],
    "file_agent": ["文件", "目录", "列出", "读取", "写入", "创建", "移动",
                   "搜索文件", "read_file", "write_file", "list_directory"],
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

# ★ 【修复Lock】_lazy_init_lock 改为惰性创建（None → 首次调用时在当前 loop 创建）。
#   原来模块加载时直接 asyncio.Lock() 会绑定到主线程 loop，
#   langgraph dev 用 uvicorn 启动新 loop 后 Lock.acquire() 永远挂起。
_lazy_init_lock: asyncio.Lock | None = None
_mcp_exit_stack: AsyncExitStack | None = None


async def _ensure_registry() -> None:
    """
    ★ 【修复11+12+Lock】三重保险初始化。

    1. 快速路径：_registry.agents 非空则直接返回（纯 Python 属性，跨 loop 安全）
    2. 惰性创建 Lock：在当前 loop 里创建，确保不跨 loop
    3. double-check：防并发重复初始化
    """
    global _lazy_init_lock

    # 快速路径
    if _registry.agents:
        return

    # ★ 惰性创建 Lock，绑定到调用方当前 loop
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
    ★ 前端路径（langgraph dev）：通过 SSE 连接 server.py。
    server.py 已内置所有工具（http/data/math/file），一个连接搞定。
    子进程由 webapp.py lifespan 负责拉起。
    """
    global _mcp_exit_stack
    if _mcp_exit_stack is not None:
        print("⚠️ [MCP] _start_mcp_sessions 重复调用，跳过")
        return

    print(f"🔍 [MCP] platform={sys.platform}  python={sys.executable}")
    print(f"🔍 [MCP] server SSE URL: {_SERVER_SSE_URL}")

    stack = AsyncExitStack()
    all_tools: list[StructuredTool] = []

    # ── server.py MCP（含 http/data/math/file 全部工具）────
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

    # ── 提交结果 ─────────────────────────────────────────
    if not all_tools:
        print("❌ [MCP] MCP 连接失败，registry 未就绪", file=sys.stderr)
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
    ★ 后端路径（__main__ 直接运行）：用 stdio_client spawn server.py 子进程。
    server.py 已内置所有工具（http/data/math/file），不再单独连 filesystem MCP。
    仅在 __main__ 里调用，前端路径不使用此函数。
    """
    global _mcp_exit_stack
    if _mcp_exit_stack is not None:
        return

    print(f"🔍 [MCP-stdio] SERVER_PATH = {SERVER_PATH}  (exists={SERVER_PATH.exists()})")
    print(f"🔍 [MCP-stdio] FS_BASE_DIR = {_FS_BASE_DIR}  (exists={_FS_BASE_DIR.exists()})")
    print(f"🔍 [MCP-stdio] platform={sys.platform}  python={sys.executable}")

    stack = AsyncExitStack()
    all_tools: list[StructuredTool] = []

    # ── server.py MCP（stdio，含全部工具）────────────────
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

    if not all_tools:
        print("❌ [MCP-stdio] MCP 连接失败", file=sys.stderr)
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
    # ★ 重置 Lock，下次启动时在新 loop 中重新创建
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


def _validate_and_fix_task_plan(task_plan: list[Task]) -> list[Task]:
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
    # ★ 【修复12】_ensure_registry() 必须在 task_plan 判断之前调用
    await _ensure_registry()

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
        raw = _extract_json(_extract_llm_content(response).strip())
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
            "messages":        state["messages"] + [AIMessage(content=failure_msg)],
            "task_plan":       [],
            "current_task_id": 0,
            "next_agent":      "FINISH",
        }

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
        if task["status"] not in ("pending", "in_progress"):
            continue

        unmet = [dep for dep in task.get("depends_on", []) if dep not in done_map]
        if unmet:
            print(f"\n  ⏳ 任务[{task['task_id']}] 等待依赖 {unmet}，跳过")
            continue

        if task.get("agent") == "direct":
            print(f"\n  🧭 Supervisor → direct_answer（任务[{task['task_id']}]）")
            print(f"     意图：{task['description'][:80]}")
            task["status"] = "in_progress"
            return {
                **state,
                "next_agent":      "direct_answer",
                "current_task_id": task["task_id"],
                "task_plan":       task_plan,
            }

        print(f"\n  🧭 Supervisor → {task['agent']}（任务[{task['task_id']}]）")
        print(f"     {task.get('_resolved_description', task['description']).replace(chr(10), ' | ')}")

        resolved_inputs: dict[str, str] = {}
        for param_name, source in task.get("inputs", {}).items():
            from_id   = source["from_task"]
            src_field = source.get("field", "result")
            src_task  = done_map.get(from_id, {})
            resolved_inputs[param_name] = src_task.get(src_field, "")

        if resolved_inputs:
            params_text = "\n".join(f"  {k} = {v}" for k, v in resolved_inputs.items())
            task["_resolved_description"] = (
                f"{task['description']}\n\n"
                f"【运行时参数】\n{params_text}"
            )
        else:
            task["_resolved_description"] = task["description"]

        if task["agent"] not in _registry.agents:
            print(f"  ❌ [{task['agent']}] 未在 registry 中注册，跳过任务[{task['task_id']}]")
            print(f"     当前可用 agents: {_registry.agents}")
            task["status"] = "done"
            task["result"] = (
                f"⚠️ {task['agent']} 未注册（可能 MCP 未启动）。"
                f"可用 agents: {_registry.agents}"
            )
            continue

        return {
            **state,
            "next_agent":      task["agent"],
            "current_task_id": task["task_id"],
            "task_plan":       task_plan,
        }

    pending = [t for t in task_plan if t["status"] == "pending"]
    if pending:
        print(f"\n  ⚠️ 任务 {[t['task_id'] for t in pending]} 依赖无法满足，强制跳过")
        for t in pending:
            t["status"] = "done"
            t["result"] = "⚠️ 依赖未满足，跳过"

    print("\n  🧭 Supervisor → FINISH")
    return {**state, "next_agent": "FINISH", "task_plan": task_plan}


# ══════════════════════════════════════════════════════
# 9. Re-Planner
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
            f"用户原始问题：{_get_message_content(state['messages'][0])}\n\n"
            f"已完成任务：\n{done_summary}\n\n"
            f"待执行任务（pending）：\n{pending_summary}"
        )),
    ])

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
            "messages": state["messages"] + [AIMessage(content=answer)],
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
            print("  ⛔ Planner 规划失败，直接终止，不进入 Supervisor")
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
# 13. lifespan —— 仅 langgraph dev 通过 webapp.py 调用
#     ★ SSE 版本：子进程已由 webapp.py lifespan 拉起，
#       这里只负责建立 SSE 连接，不再自己 spawn 进程。
# ══════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app):
    print(f"🟢 [lifespan] 启动，连接 MCP SSE sessions...")
    await _start_mcp_sessions()
    try:
        yield
    finally:
        await _stop_mcp_sessions()


# ══════════════════════════════════════════════════════
# 14. __main__ —— 后端测试，直接用 stdio_client
#     命令：uv run python src/langgraph_stdio_agent.py
#     完全独立，不依赖 SSE 进程，不影响前端测试
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    QUESTIONS = [
        "列出 File_Agent 目录下的所有文件，然后在其中创建一个名为 hello.txt 的文件，内容为：Hello from file_agent！",
        # "计算 3+5，然后访问 https://api.github.com/zen，再计算 10×20",
    ]

    async def main():
        # ★ 后端测试专用：使用 stdio_client 直接 spawn 子进程
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

    if sys.platform == "win32":
        pass  # ProactorEventLoop 已在模块顶层设置
    asyncio.run(main())