"""
src/langgraph_parallel_agent.py

★ 并行化改造 + Bug 修复（基于 langgraph_stdio_agent.py）

【并行架构】planner → parallel_executor → final_answer
  - 同层任务 asyncio.gather() 真并发，每任务独立 spawn/close MCP session
  - 拓扑 BFS 分层：无依赖任务同层并行，有依赖任务跨层串行

【MemorySaver Checkpoint】
  - add_messages reducer：消息追加而非覆盖，跨轮对话历史完整保留
  - 模块级 _checkpointer 单例：graph 重建不丢历史
  - thread_id + config：区分不同用户会话

【历史修复 B/B2/E】
  - Planner 不传任何历史 HumanMessage，彻底消除幻觉任务
  - final_answer 用 Human+AI 交替历史解析跨轮引用

【本版本修复】

  ★ 修复G（_run_direct_task）
    原方案：只传 HumanMessage 纯文本作为 context_note，模型看不到 AI 的历史确认。
    新方案：传摘要 + 最近 20 条 Human+AI 交替消息，direct_task 阶段就能答对，
            不再靠 final_answer 纠偏（"双重推理"消除）。

  ★ 修复H
    删除 _get_first_user_message 死代码（从未被调用）。

  ★ 修复X（planner_node —— 跨轮数值引用核心修复）
    原问题：用户问"刚才的结果加4"，Planner 看不到"刚才=56"，
            description 写成"计算刚才的结果加4"，math_agent 不知道56，
            只能放弃工具调用靠 LLM 猜——最坏情况猜错。
    新方案：Planner 从 checkpoint 历史中取最近一条 AIMessage，
            注入 system prompt 作为"上一轮结果"上下文。
            Planner 看到"上一轮回复：56"，可在 description 里写"计算56+4"，
            math_agent 拿到具体数值，真正调用 add(56,4) 工具得出 60。
    注意：只取最近一条 AIMessage（不取 HumanMessage），
          避免历史 HumanMessage 触发幻觉任务（修复B成果保留）。

  ★ 修复Z（_run_question —— Checkpoint 核心修复，最重要）
    原问题：每次 invoke 传入完整 state：
              {"messages": [...], "task_plan": [], "current_task_id": 0, "next_agent": ""}
            task_plan / current_task_id / next_agent 是普通字段，LangGraph merge 规则
            是"新值覆盖旧值"，每次都传 task_plan=[] 等于每次清空任务计划。
            即使 MemorySaver 存了历史，也因为字段被强制覆盖而失效。
    新方案：invoke 时只传当前轮的 HumanMessage：
              {"messages": [HumanMessage(content=q)]}
            - messages 字段：add_messages reducer 追加，历史保留 ✅
            - 其他字段：从 checkpoint 自然恢复，再由各 node 正常更新 ✅
            这样 MemorySaver 才真正发挥作用：第N+1轮能看到前N轮完整对话历史。

  ★ 修复Y（历史窗口扩大）
    _run_direct_task 和 final_answer_node 的 recent_history 从 10 条扩大到 20 条，
    覆盖约 10 轮对话，配合摘要机制适配更长的多轮记忆场景。

  ★ 修复S（对话摘要 Summary Memory —— 解决长对话记忆丢失，两个问题的根治方案）

    问题根因：
      checkpoint 存了完整的 32 条消息，但 Planner / direct_task / final_answer
      都只取最近 N 条。对话越长，早期信息越容易滑出窗口，导致：
        - 问题1：Planner 看不到"用户叫Tony"→把"我叫什么名字"路由到 db_agent（严重）
        - 问题2：组6长对话压力测试完全失败，AI 回答"没有提供任何个人信息"（严重）

    解决方案：AgentState 新增两个字段：
      conversation_summary：滚动摘要，把用户画像信息浓缩成自然语言。
        "用户叫Tony，住多伦多，26岁，后端工程师，用Python/Go，喜欢篮球和吉他。
         后更名为Alice，职业改为数据科学家。"
      summary_turn_count：已被摘要覆盖的轮次数，用于控制摘要更新频率。

    摘要更新时机：在 final_answer_node 末尾，当新增了5轮（10条消息）时触发。
      - 异步 LLM 调用生成摘要，不阻塞主流程
      - 有 existing_summary 时做增量更新，无新信息时保留现有摘要
      - 只提取用户画像，不记录工具执行结果（避免摘要膨胀）

    摘要注入位置：
      1. planner_node：摘要 → Planner 知道用户画像 → 正确路由（问题1自然消失）
      2. _run_direct_task：摘要 + 最近20条 → 长期记忆 + 近期细节（问题2修复）
      3. final_answer_node：摘要 + 最近20条 → 同上

    效果：
      - 无论对话多长，早期信息都不会丢失（存在摘要里）
      - 近期更新的信息优先于摘要（近期消息自然覆盖摘要中的旧值）
      - 不需要任何硬规则，Planner 看到摘要就能自主正确路由

【QUESTIONS 测试题库 v4】
  6 组共 17 题，重点验证：
  - 组1：MemorySaver 跨轮记忆（消息数递增、5轮后仍能回忆第1轮信息）
  - 组2：身份信息更新覆盖（最新信息优先）
  - 组3：跨轮数值引用（Planner 注入上一轮 AI 回复 → math_agent 实际调工具）
  - 组4：复杂多子任务分拆（3层串行依赖、并行+串行混合）
  - 组5：复杂多子任务分拆（3任务同层并行、扇出型并行）
  - 组6：长对话记忆压力测试（15+轮后仍能回忆早期信息）【本版本重点修复】
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

# ★ STORE 改动1/5：InMemoryStore 导入（CLI 模式用；webapp 模式由 lifespan 注入 AsyncSqliteStore）
#
# 两种 Store 对比：
#   InMemoryStore   → Python 字典，进程重启清空，CLI 实验专用
#   AsyncSqliteStore → 写入磁盘，进程重启后数据仍在，webapp 生产使用
#
# Store 和 Checkpointer 的区别：
#   Checkpointer  → 存每个 thread_id 的对话历史（per-user、per-session）
#   Store         → 存跨 thread_id 共享的全局记忆（系统配置、管理员预置知识库等）
from langgraph.store.memory import InMemoryStore   # ← 新增

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
    # ★ 摘要记忆字段（解决长对话记忆丢失问题）
    #
    # conversation_summary：滚动摘要，把早期对话浓缩成自然语言。
    #   格式：用户叫Tony，住多伦多，26岁，后端工程师，用Python/Go，喜欢篮球和吉他。后更名为Alice，职业改为数据科学家。
    #   每5轮（10条消息）在 final_answer_node 末尾异步更新一次。
    #   注入到 planner_node / _run_direct_task / final_answer_node，
    #   让这三个节点都能"记得"早期信息，不受窗口限制。
    #
    # summary_turn_count：已被摘要覆盖的轮次数。
    #   用于判断"当前消息数 - summary_turn_count*2 >= 10"时触发更新。
    #   防止每轮都重新生成摘要（摘要生成也消耗 token）。
    conversation_summary: str   # 对话摘要，初始为空字符串
    summary_turn_count: int     # 已摘要轮次，初始为 0


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

# ★ STORE 改动2/5：_store 模块级单例（CLI 模式用 InMemoryStore）
#
# 设计说明：
#   - CLI 模式（__main__）：直接用这个 InMemoryStore，进程内持久，重启清空
#   - webapp 模式（lifespan）：lifespan 会创建 AsyncSqliteStore 并覆盖 agent_module.graph，
#     但 _store 这个变量本身不需要被覆盖，因为 graph 是 lifespan 重建的，
#     store 作为参数传进去了，和 _store 变量解耦
#
# 命名空间设计建议：
#   ("system",)      → 管理员预置的全局知识（所有用户、所有会话都能读到）
#   ("user", uid)    → 跨 thread_id 的用户级持久信息（如 VIP 标签、偏好设置）
_store = InMemoryStore()   # ← 新增：CLI 模式的 store 单例

# ══════════════════════════════════════════════════════
# 流式输出队列（阶段二：SSE 端点专用）
# ══════════════════════════════════════════════════════
#
# 设计：webapp.py 的 /chat/stream 端点在调用 graph.ainvoke() 之前，
#       先把一个 asyncio.Queue 注入到这个模块级变量。
#       final_answer_node 在流式生成 token 时，把每个 token 放进队列。
#       SSE 端点从队列里读取 token，立即推送给前端（浏览器）。
#
# 为什么用模块级变量？
#   LangGraph 节点函数只接收 state 参数，无法直接传入额外参数。
#   用模块级变量是最简单、零改动图结构的方案。
#
# 线程安全：asyncio.Queue 是协程安全的（单线程事件循环内）。
#           每次请求开始前设置，请求结束后清除，不会有并发冲突。
#           （langgraph dev 默认单进程，一次处理一个请求）
_stream_queue: asyncio.Queue | None = None


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

    # ★ 修复（原 Bug 根因）：webapp 和 CLI 两种模式统一使用 _checkpointer（模块级单例）。
    #
    # 旧代码在 webapp 模式（MCP_USE_SSE=1）下调用 build_graph()（不传 checkpointer），
    # 目的是让"平台"管持久化。但实际上：
    #   1. lifespan 会在 _start_mcp_sessions 之后再调一次 build_graph(checkpointer=saver)
    #      来覆盖——这一步是对的。
    #   2. 但 _ensure_registry() 在每次请求时都可能触发懒初始化，
    #      懒初始化会再次调用 _init_registry → 旧代码的 build_graph()（无 checkpointer），
    #      把 lifespan 辛苦注入的 checkpointer 悄悄覆盖掉。
    #   3. 结果：第二轮请求时 graph 已没有 checkpointer，checkpoint 失效，
    #      msgs 只有 1 条（当前轮），多轮记忆全部丢失。
    #
    # ★ STORE 改动3/5：_init_registry 把 _store 一并传给 build_graph
    #
    # 为什么需要传 store？
    #   build_graph 的 g.compile(store=store) 会让 LangGraph 在调用每个节点时，
    #   自动把 store 以关键字参数方式注入给声明了 `store=None` 参数的节点函数。
    #   只有 compile 时传了 store，planner_node(state, *, store=None) 才能收到它。
    #
    # webapp 模式下：
    #   lifespan 会在这一步之后用 AsyncSqliteStore 重建 graph，
    #   所以这里传 _store（InMemoryStore）只是过渡状态，不影响最终运行。
    graph = build_graph(checkpointer=_checkpointer, store=_store)


# ══════════════════════════════════════════════════════
# 公共工具函数
# ══════════════════════════════════════════════════════

def _extract_json(raw: str) -> str:
    """
    从 LLM 输出中提取 JSON 字符串。
    处理以下几种常见格式：
      1. 纯 JSON（无代码块）
      2. ```json ... ``` 或 ``` ... ```（有语言标识符或无）
      3. 代码块格式异常（空语言标识符、多余空行等）
      4. JSON 前面有废话文字（fallback：找第一个 [ 或 {）
    """
    raw = raw.strip()

    # ── 优先尝试从代码块里提取 ─────────────────────────────────────────
    if "```" in raw:
        parts = raw.split("```")
        # 奇数下标（1, 3, 5...）是代码块内容
        for i in range(1, len(parts), 2):
            candidate = parts[i]
            # 去掉可能的语言标识符行（如 "json\n"、"python\n" 等）
            candidate = re.sub(r"^[a-zA-Z]+\n", "", candidate).strip()
            if candidate:
                return candidate
        # 所有代码块都是空的，fallthrough 到下面的 fallback

    # ── fallback：找第一个 [ 或 {，去掉前面的废话文字 ──────────────────
    # 场景：模型输出 "好的，以下是任务规划：\n[{...}]"
    m = re.search(r"[\[{]", raw)
    if m:
        return raw[m.start():]

    # 实在没有，原样返回（让上层 json.loads 报错，触发重试）
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


# ══════════════════════════════════════════════════════
# 6b. Memory Store 辅助函数（实验 & 管理用）
# ══════════════════════════════════════════════════════
#
# 这三个函数包装了 _store 的 put / get / list 操作，
# 供 CLI 交互模式（!memory 命令）和 webapp 端点调用。
# 对 InMemoryStore 用同步 API，对 AsyncSqliteStore 在外层 async 里用 await。

def store_put(key: str, value: Any, namespace: tuple = ("system",)) -> None:
    """写入一条全局记忆。value 可以是任意 JSON-serializable 对象。"""
    _store.put(namespace, key, value if isinstance(value, dict) else {"value": value})
    print(f"  💾 [Store] 写入 {namespace}/{key} = {str(value)[:60]}")


def store_get(key: str, namespace: tuple = ("system",)) -> Any | None:
    """读取一条全局记忆。不存在时返回 None。"""
    try:
        item = _store.get(namespace, key)
        return item.value if item else None
    except Exception:
        return None


def store_list(namespace: tuple = ("system",)) -> dict:
    """列出命名空间下所有记忆，返回 {key: value} 字典。"""
    try:
        results = _store.search(namespace)
        return {r.key: r.value for r in results}
    except Exception:
        return {}


def store_delete(key: str, namespace: tuple = ("system",)) -> bool:
    """删除一条全局记忆。成功返回 True。"""
    try:
        _store.delete(namespace, key)
        print(f"  🗑️  [Store] 删除 {namespace}/{key}")
        return True
    except Exception as e:
        print(f"  ⚠️  [Store] 删除失败：{e}")
        return False


# ══════════════════════════════════════════════════════
# 7. 对话摘要生成（解决长对话记忆丢失）
# ══════════════════════════════════════════════════════

async def _update_summary(
    messages: list,
    existing_summary: str,
) -> str:
    """
    生成/更新对话摘要（滚动摘要策略）。

    设计原则：
      - 摘要只关注"用户画像信息"：姓名、年龄、城市、职业、爱好、编程语言等。
      - 若用户多次更新了同一信息，摘要只保留最新版本（例如 Tony→Alice）。
      - 工具任务结果（数学计算、DB 查询等）不进入摘要，避免摘要膨胀。
      - 现有摘要作为基础，只对新增信息做增量更新。

    参数：
      messages        - 完整消息列表（Human + AI 交替）
      existing_summary - 上一次的摘要（可为空字符串）

    返回：
      更新后的摘要字符串
    """
    # 把消息列表格式化成对话文本，供 LLM 读取
    # 只取 Human + AI 交替消息（过滤掉 System / Tool 等）
    convo_lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            convo_lines.append(f"用户：{_get_message_content(m)[:200]}")
        elif isinstance(m, AIMessage):
            convo_lines.append(f"AI：{_get_message_content(m)[:200]}")
    convo_text = "\n".join(convo_lines)

    existing_part = ""
    if existing_summary:
        existing_part = (
            f"\n\n【现有摘要（请在此基础上更新，不要删除已有信息，如有冲突以最新为准）】\n"
            f"{existing_summary}"
        )

    prompt = (
        "你是对话摘要助手。请从下面的对话历史中提取并更新用户的个人信息摘要。\n\n"
        "【摘要规则】\n"
        "1. 只提取用户相关的画像信息：姓名、年龄、城市/居住地、职业、编程语言、爱好等。\n"
        "2. 如果用户多次提供了同一类信息，以最新一次为准（例如名字从Tony改成Alice，只记Alice）。\n"
        "3. 不要包含工具任务结果（数学计算结果、数据库查询结果、文件操作结果等）。\n"
        "4. 用简洁的一到三句话概括，不要用列表格式，直接写成自然语言。\n"
        "5. 如果对话里没有任何用户画像信息，输出空字符串。\n"
        "6. 只输出摘要文本，不要有任何前缀（如'摘要：'）或解释。\n"
        f"{existing_part}\n\n"
        "【对话历史】\n"
        f"{convo_text}"
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = _extract_llm_content(response).strip()
        # 防止模型输出"空字符串"字面量或其他无效内容
        if summary in ("空字符串", "无", "无信息", "（无）", ""):
            return existing_summary  # 无新信息，保留现有摘要
        print(f"  📝 [Summary] 摘要已更新：{summary[:100]}")
        return summary
    except Exception as e:
        print(f"  ⚠️ [Summary] 摘要生成失败：{e}，保留现有摘要")
        return existing_summary


# ══════════════════════════════════════════════════════
# 8. Planner
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

━━ 最重要的规则：只规划当前问题 ━━
1. 你只负责拆解【当前这条用户消息】，绝对不能规划历史对话中出现过的任务。
2. 如果用户当前消息是闲聊/问候/自我介绍（如我是程序员、你好、对）→ 只规划 1 个 direct 任务。
3. 如果用户当前消息只问一件事 → 只规划完成那一件事所需的最少任务。
4. description 只写任务意图，不提前计算数值或给出答案。
   ★ 唯一例外（Memory Store / 用户画像摘要中的已知事实）：
     如果答案已经在【Memory Store 全局记忆】或【用户画像摘要】中明确存在，
     则 description 必须把相关内容直接写入，供执行器使用，而不能只写意图。
     示例（正确）：store 里有 discount_policy="所有用户享受九折优惠，VIP用户享受八折"
       用户问"折扣政策是什么" → description 必须写：
       "回答折扣政策：所有用户享受九折优惠，VIP用户享受八折"
     示例（错误）：description 只写 "回答公司的折扣政策"
       → 执行器看不到实际内容，会反问用户，这是严重错误。
5. 同一个 agent 可出现多次。
6. 任务按拓扑顺序排列（被依赖的任务排在前面）。

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


async def planner_node(state: AgentState, *, store=None) -> AgentState:
    # ★ STORE 改动5/5：planner_node 新增 store=None 参数
    #
    # LangGraph 的 store 注入机制：
    #   compile(store=store) 后，LangGraph 在调用节点前会检查函数签名。
    #   如果有 `*, store=None` 关键字参数，自动把 store 对象注入进来。
    #   这是 LangGraph 原生支持的特性，不需要修改图结构。
    await _ensure_registry()

    msgs = state.get("messages", [])
    if not msgs:
        return {**state, "next_agent": "FINISH", "task_plan": []}

    # 取最后一条 HumanMessage 作为当前问题
    last_human_msg = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage)),
        None
    )
    if not last_human_msg:
        return {**state, "next_agent": "FINISH", "task_plan": []}

    user_msg = _get_message_content(last_human_msg)
    print(f"\n📋 [Planner] 规划任务：{user_msg[:80]}")

    # ── 调试：打印 msgs 全貌，确认跨轮历史是否被平台恢复 ────────────────
    print(f"  🔍 [Planner-debug] msgs 共 {len(msgs)} 条：")
    for i, m in enumerate(msgs):
        mtype = type(m).__name__
        content = _get_message_content(m)[:60].replace("\n", " ")
        print(f"    [{i}] {mtype}: {content}")

    # ★ 修复X（升级版）：Planner 注入摘要 + 上一轮 AI 回复
    #
    # 原方案只注入"最近一条 AIMessage"，仅能解决跨轮数值引用问题。
    # 但对于"我叫什么名字？"这类问题，若早期信息已滑出窗口，
    # Planner 看不到用户曾说"我叫Tony"，就会把它路由到 db_agent 查数据库。
    #
    # 新方案：同时注入两层上下文：
    #   层1 - conversation_summary（用户画像摘要）：
    #     "用户叫Tony，住多伦多，后端工程师，喜欢篮球..."
    #     Planner 看到这条，就知道"我叫什么名字"是闲聊，答案已在摘要里，→ direct
    #     不会再误路由到 db_agent。这自然解决了问题1，无需硬规则。
    #   层2 - 上一轮 AI 回复（跨轮数值引用）：
    #     "上一轮回复：56"
    #     Planner 知道"刚才的结果=56"，description 可以写"56+4"，
    #     math_agent 拿到具体数值正确调工具。
    last_ai_context = ""

    # 层1：摘要注入
    conv_summary = state.get("conversation_summary", "")
    if conv_summary:
        last_ai_context += (
            f"\n\n【用户画像摘要（从历史对话中提取，供参考）】\n"
            f"{conv_summary}\n"
            "⚠️ 以上是对用户已知信息的摘要。若用户当前问题涉及自身信息（如'我叫什么名字'、"
            "'我住在哪里'、'我的职业是什么'），直接用 direct 回答，不要查数据库。"
        )

    # 层2：上一轮 AI 回复（跨轮数值引用）
    last_ai_msg = next(
        (m for m in reversed(msgs[:-1]) if isinstance(m, AIMessage)),
        None
    )
    if last_ai_msg:
        ai_content = _get_message_content(last_ai_msg)
        last_ai_context += (
            f"\n\n【上一轮对话结果（仅供参考，不要重新执行）】\n"
            f"{ai_content[:300]}\n"
            "⚠️ 以上是上一轮的AI回复，不是新任务。只在当前问题引用'刚才'/'上一步'/'上面'时才使用此信息。"
        )

    max_retries = 3
    task_plan: list[Task] = []

    retry_feedback:  str = ""
    last_raw_output: str = ""

    # ★ STORE 改动5/5（续）：从 Memory Store 读取全局记忆
    #
    # 读取逻辑：
    #   - 命名空间 ("system",)：管理员预置的全局知识（公司信息、业务规则等）
    #   - 命名空间 ("user", thread_id)：跨会话的用户级持久信息（VIP标签、偏好等）
    #
    # 与 conversation_summary 的区别：
    #   conversation_summary → 从本会话对话历史提取，thread 隔离
    #   store_context        → 来自 Memory Store，跨所有 thread 共享
    #
    # 异步兼容：InMemoryStore 用 .search()，AsyncSqliteStore 用 await .asearch()
    store_context = ""
    if store:
        try:
            # 读取系统全局记忆（("system",) 命名空间）
            if hasattr(store, "asearch"):
                system_results = await store.asearch(("system",))
            else:
                system_results = store.search(("system",))

            # 读取当前用户的跨会话记忆（("user", thread_id) 命名空间）
            thread_id = ""
            # thread_id 在 config 里，这里从 state 无法直接取到，用空串兜底
            user_results = []

            all_items: dict = {}
            for r in (system_results or []):
                all_items[r.key] = r.value
            for r in (user_results or []):
                all_items[r.key] = r.value

            if all_items:
                store_context = (
                    f"\n\n【Memory Store 全局记忆（跨会话持久，管理员预置）】\n"
                    f"{json.dumps(all_items, ensure_ascii=False, indent=2)}\n"
                    "以上是系统预置的全局记忆。如果用户问题涉及其中的信息：\n"
                    "  1. 使用 direct agent 直接回答，不要查数据库或调用其他工具。\n"
                    "  2. ★ 关键：必须在 description 字段中把相关记忆的实际内容写进去。\n"
                    "     例如 store 里有 discount_policy 的值，用户问折扣政策，\n"
                    "     description 必须写：'回答折扣政策：<discount_policy的实际内容>'\n"
                    "     而不能只写：'回答公司的折扣政策'（执行器没有 store 访问权，会反问用户）。"
                )
                print(f"  🗄️  [Store] 读取到 {len(all_items)} 条全局记忆：{list(all_items.keys())}")
        except Exception as e:
            print(f"  ⚠️  [Store] 读取失败（忽略）：{e}")

    for attempt in range(max_retries):
        try:
            # Planner system prompt = 基础 + 摘要/上轮上下文 + Store 全局记忆
            planner_system_with_context = _planner_system() + last_ai_context + store_context

            invoke_msgs: list = [
                SystemMessage(content=planner_system_with_context),
                HumanMessage(content=user_msg),
            ]

            if retry_feedback:
                invoke_msgs.append(AIMessage(content=last_raw_output))
                invoke_msgs.append(HumanMessage(content=(
                    f"你的上一次输出有以下问题，请仔细阅读后重新只输出 JSON 数组，"
                    f"不要有任何其他文字：\n{retry_feedback}"
                )))

            response = await llm.ainvoke(invoke_msgs)
            last_raw_output = _extract_llm_content(response)
            raw = _extract_json(last_raw_output)
            task_plan = json.loads(raw)

            if not isinstance(task_plan, list) or len(task_plan) == 0:
                raise ValueError("task_plan 必须是非空 JSON 数组")

            for t in task_plan:
                t.setdefault("status", "pending")
                t.setdefault("result", "")
                t.setdefault("_resolved_description", "")
                t.setdefault("inputs", {})
                t.setdefault("depends_on", [])

            # inputs 格式校验
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
                        f"task[{t['task_id']}] inputs 引用了未在 depends_on 中声明的任务："
                        f"{sorted(orphan_inputs)}"
                    )

            if fmt_errors:
                retry_feedback = "inputs 格式错误：\n" + "\n".join(fmt_errors)
                print(f"  ⚠️ Planner 第 {attempt+1} 次校验失败：\n    " +
                      "\n    ".join(fmt_errors))
                raise ValueError(retry_feedback)

            print(f"  ✅ 规划完成（{len(task_plan)} 个任务）：")
            for t in task_plan:
                dep_str = f" depends_on={t['depends_on']}" if t['depends_on'] else ""
                print(f"     [{t['task_id']}] {t['agent']:12s} → "
                      f"{t['description'][:45]}{dep_str}")
            break

        except Exception as e:
            if not retry_feedback:
                retry_feedback = (
                    f"JSON 解析失败：{e}\n"
                    f"你的原始输出（前300字）：{last_raw_output[:300]}\n"
                    f"请只输出合法的 JSON 数组，不要有任何额外文字或代码块标记。"
                )
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

    ★ 修复4 — 两层 fallback：
      第一层：session spawn 失败 → 改用 default_agent 的全局已注册工具重试
      第二层：spawn 成功但工具列表为空 → 同上 fallback
      两层都失败时 → 纯 LLM 语言推理兜底（不调工具，直接回答）
    """
    agent_name = task.get("agent", "default_agent")
    intent     = task.get("_resolved_description") or task.get("description", "")

    print(f"\n🤖 [{agent_name}] 任务[{task['task_id']}] 开始（独立 session）：{intent[:60]}")
    t0 = time.perf_counter()

    # ── 内部执行函数：给定工具列表，跑 LLM + 工具循环 ──────────────────
    async def _run_with_tools(tools: list, prompt: str) -> str:
        if not tools:
            # 没有工具：纯语言推理兜底
            print(f"  ⚠️ [{agent_name}] 无可用工具，降级为纯 LLM 推理")
            resp = await llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=intent),
            ])
            return _extract_llm_content(resp)

        llm_with_tools = llm.bind_tools(tools)
        tool_map       = {t.name: t for t in tools}
        msgs = [
            SystemMessage(content=prompt),
            HumanMessage(content=intent),
        ]

        max_steps     = 6
        last_response = None

        for step in range(max_steps):
            response      = await llm_with_tools.ainvoke(msgs)
            last_response = response
            msgs.append(response)

            if isinstance(response, dict):
                tool_calls = response.get("tool_calls", [])
            else:
                tool_calls = getattr(response, "tool_calls", []) or []

            if not tool_calls:
                print(f"  ✅ [{agent_name}] task[{task['task_id']}] step={step} 完成")
                break

            print(f"  🔧 [{agent_name}] task[{task['task_id']}] step={step} "
                  f"工具调用：{[tc['name'] for tc in tool_calls]}")

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

        return _extract_llm_content(last_response) if last_response else "（无结果）"

    # ── 第一层：尝试独立 spawn session ──────────────────────────────────
    async with AsyncExitStack() as stack:
        try:
            session = await _spawn_session_for(agent_name, stack, use_sse=use_sse)
            tools   = await load_tools(session)

            # 第二层（spawn 成功但工具为空）：fallback 到 default_agent 的全局工具
            if not tools:
                print(f"  ⚠️ [{agent_name}] session 已连接但工具列表为空，"
                      f"fallback → 使用 _registry 中的全局工具")
                fallback_tools   = _registry.tools_for("default_agent") or []
                fallback_prompt  = AGENT_SYSTEM_PROMPTS.get(
                    "default_agent", DEFAULT_AGENT_SYSTEM_PROMPT
                )
                result = await _run_with_tools(fallback_tools, fallback_prompt)
            else:
                result = await _run_with_tools(tools, system_prompt)

        except Exception as exc:
            # 第一层：spawn 失败 → 用 _registry 里已注册的全局工具兜底
            print(f"  ⚠️ [{agent_name}] session 启动失败：{exc}，"
                  f"fallback → 使用 _registry 中的全局工具", file=sys.stderr)
            fallback_tools  = _registry.tools_for(agent_name)
            fallback_prompt = system_prompt  # 保留原 agent 的 prompt，语义不变

            if not fallback_tools:
                # 该 agent 在 registry 里也没有工具，再退一步用 default_agent 的工具
                print(f"  ⚠️ [{agent_name}] registry 中也无工具，"
                      f"最终 fallback → default_agent 全局工具", file=sys.stderr)
                fallback_tools  = _registry.tools_for("default_agent") or []
                fallback_prompt = AGENT_SYSTEM_PROMPTS.get(
                    "default_agent", DEFAULT_AGENT_SYSTEM_PROMPT
                )

            result = await _run_with_tools(fallback_tools, fallback_prompt)

    elapsed = time.perf_counter() - t0
    print(f"  ⏱️ [{agent_name}] task[{task['task_id']}] 耗时 {elapsed:.2f}s")
    return result


# ══════════════════════════════════════════════════════
# 9. direct_answer_node（不变）
# ══════════════════════════════════════════════════════


async def _run_direct_task(task: Task, state: AgentState) -> str:
    intent = task.get("description", "")
    print(f"\n  💬 direct_answer 任务[{task['task_id']}]：{intent[:60]}")

    # ★ 修复G（升级版）：摘要 + 最近20条 Human+AI 交替消息
    #
    # 三层记忆策略：
    #   层1 - conversation_summary（长期记忆）：
    #     覆盖任意早期信息，不受窗口限制。例如第1轮说的"我叫Tony"，
    #     即使对话已进行30轮，摘要里仍有这条信息。
    #   层2 - recent_history 最近20条（近期细节）：
    #     覆盖最近10轮的完整上下文，保留近期的所有细节和数值引用。
    #     窗口从10扩大到20，覆盖约10轮对话。
    #   优先级：recent_history 里的新信息优先于 summary（自然语言理解）。
    msgs = state.get("messages", [])

    # 构建"摘要前缀"：如果有摘要，作为系统上下文的补充
    conv_summary = state.get("conversation_summary", "")
    summary_prefix = ""
    if conv_summary:
        summary_prefix = (
            f"【对话摘要（用户画像，供参考）】\n{conv_summary}\n\n"
        )

    recent_history: list = [
        m for m in msgs[:-1]
        if isinstance(m, (HumanMessage, AIMessage))
    ][-20:]  # 窗口从10扩大到20（约10轮对话）

    response = await llm.ainvoke([
        SystemMessage(content=(
            "你是一个友善的 AI 助手。请只回答当前分配给你的这一个子任务，"
            "不要回答其他历史问题，不要重复执行历史任务。\n"
            f"{summary_prefix}"
            "【重要规则】\n"
            "1. 如果对话历史中用户多次提供了姓名/职业/地点等信息，以最新一次为准。\n"
            "2. 如果对话历史中 AI 已经确认过某个信息（如职业、姓名），直接采用该确认结论。\n"
            "3. 对话摘要提供了用户的基础画像，近期对话历史中有更新时以最新为准。\n"
            "4. 只回答子任务要求的内容，不要主动补充与子任务无关的信息。"
        )),
        *recent_history,
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

            # ★ 修复4 — 第一层（执行前）：检查 registry 里该 agent 是否有工具
            #   如果一个非 direct 的 agent 在 registry 里根本没有注册工具，
            #   提前改为 default_agent，避免 spawn session 后再失败。
            #   注意：math_agent 等会独立 spawn session，registry 里可能没有对应工具，
            #   这种情况不应该 fallback，所以只对"registry 确实有工具但 agent 写错了"
            #   的情况做前置检查。
            #   判断依据：registry 里总工具数 > 0，但该 agent 的工具数 == 0
            if (agent not in ("direct",)
                    and _registry.agents          # registry 已就绪
                    and agent not in _registry.agents  # 该 agent 根本没注册
            ):
                print(f"  ⚠️ [{agent}] 不在 registry 中，"
                      f"前置 fallback → default_agent")
                task["agent"] = "default_agent"
                agent = "default_agent"

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

    # ── 取当前轮问题（最后一条 HumanMessage）────────────────────────
    # msgs[-1] 就是当前轮 HumanMessage（invoke 时传入的），
    # 但用 reversed 查找更健壮，防止极端情况。
    last_human = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage)),
        HumanMessage(content="")
    )

    # ── 构建"摘要前缀"────────────────────────────────────────────────
    conv_summary = state.get("conversation_summary", "")
    summary_prefix = ""
    if conv_summary:
        summary_prefix = (
            f"【对话摘要（用户画像，早期对话的精华提取）】\n{conv_summary}\n\n"
        )

    # ── 构建对话历史上下文（交替 HumanMessage + AIMessage）────────────
    # ★ 修复E + 升级：摘要 + 最近20条 Human+AI 交替消息。
    #
    # 三层记忆：
    #   摘要（长期记忆）：捕获任意早期信息，不受窗口限制
    #   recent_history 20条（近期细节）：覆盖约10轮完整上下文
    #   两者结合：近期对话中的新信息自然覆盖摘要里的旧信息
    #
    # 窗口从10扩大到20：
    recent_history: list = [
        m for m in msgs[:-1]
        if isinstance(m, (HumanMessage, AIMessage))
    ][-20:]  # 窗口从10扩大到20

    # ── 构建 system prompt ───────────────────────────────────────────
    system_content = (
        "你是一个简洁、专注的多轮对话 AI 助手。\n"
        "以下是本轮子任务的执行结果，请根据这些结果回答用户当前的问题：\n\n"
        f"{results_text}\n\n"
        f"{summary_prefix}"
        "【重要规则】\n"
        "1. 只回答用户当前这一轮的问题，不要主动复述历史对话内容。\n"
        "2. 如果结果已经很清晰（如纯数字），直接告知结果即可，不要过度解释。\n"
        "3. 不要在回答里加入与当前问题无关的信息（如用户的城市、年龄等）。\n"
        "4. 如果用户引用了'刚才的结果'/'上一步'等，请从对话历史中查找对应数值。\n"
        "5. 对话摘要提供了用户的基础画像，近期对话历史中有更新时以最新为准。"
    )

    # ── 流式 vs 非流式：根据 _stream_queue 是否已注入来决定 ────────────
    #
    # 场景A（CLI / __main__）：_stream_queue 为 None
    #   → 直接 ainvoke，一次性拿到完整回复，行为与改造前完全一致。
    #
    # 场景B（webapp SSE 端点）：webapp 在 invoke 前注入了 _stream_queue
    #   → 用 astream() 逐 token 生成。
    #   → 每个 token 立即放进队列，SSE 端点实时推送给浏览器。
    #   → 生成结束后向队列发送哨兵值 None，通知 SSE 端点关闭流。
    #
    msgs_for_llm = [
        SystemMessage(content=system_content),
        *recent_history,
        last_human,
    ]

    if _stream_queue is not None:
        # ── 流式路径（SSE 模式）────────────────────────────────────────
        full_content = ""
        try:
            async for chunk in llm.astream(msgs_for_llm):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_content += token
                    await _stream_queue.put(token)   # 推送 token 给 SSE 端点
        finally:
            await _stream_queue.put(None)            # 哨兵：通知流结束
        new_ai_msg = AIMessage(content=full_content)
    else:
        # ── 非流式路径（CLI 模式，行为不变）────────────────────────────
        response = await llm.ainvoke(msgs_for_llm)
        new_ai_msg = AIMessage(content=_extract_llm_content(response))

    # ── 触发摘要更新（每5轮一次）────────────────────────────────────
    # 判断条件：当前消息数（包含本轮新的 Human）- 已摘要轮次×2 >= 10
    #   即：新增了 5 轮（10条消息）以上时，更新一次摘要。
    # 为什么在 final_answer_node 末尾触发？
    #   此时本轮的 AI 回复已生成，消息列表最完整，摘要质量最高。
    #   用 asyncio.create_task 异步触发，不阻塞当前节点返回。
    #   但因为 LangGraph graph 是串行节点，这里实际同步 await，
    #   代价只是每5轮多一次 LLM 调用（约1-2秒），可以接受。
    current_msg_count = len(msgs) + 1  # +1 因为新的 AI 消息还未追加
    summary_turn_count = state.get("summary_turn_count", 0)
    new_summary = conv_summary
    new_summary_turn_count = summary_turn_count

    # 每5轮（10条消息）更新一次摘要
    if current_msg_count - summary_turn_count * 2 >= 10:
        print(f"  🔄 [Summary] 触发摘要更新（当前{current_msg_count}条消息，已摘要{summary_turn_count}轮）")
        # 把本轮的消息也加进去（包含最新的 Human 问题，注意 AI 回复用 new_ai_msg）
        all_msgs_for_summary = list(msgs) + [new_ai_msg]
        new_summary = await _update_summary(all_msgs_for_summary, conv_summary)
        new_summary_turn_count = current_msg_count // 2  # 更新已摘要轮次

    # ★ 返回新增字段。add_messages reducer 只追加 new_ai_msg，不覆盖历史。
    return {
        **state,
        "messages": [new_ai_msg],
        "conversation_summary": new_summary,
        "summary_turn_count": new_summary_turn_count,
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

# ★ STORE 改动4/5：build_graph 新增 store 参数
#
# 改动前：def build_graph(checkpointer=None) -> Any:
#           return g.compile(checkpointer=checkpointer)
#
# 改动后：def build_graph(checkpointer=None, store=None) -> Any:
#           return g.compile(checkpointer=checkpointer, store=store)
#
# compile(store=...) 做了什么？
#   - LangGraph 在调用每个 node 时，会检查函数签名
#   - 如果节点函数有 `store=None` 关键字参数，自动把 store 注入进去
#   - 这样 planner_node(state, *, store=None) 就能收到 store 对象
#   - 不需要修改任何图结构，完全透明
def build_graph(checkpointer=None, store=None) -> Any:
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

    # CLI 模式：checkpointer=MemorySaver, store=InMemoryStore
    # webapp 模式：checkpointer=AsyncSqliteSaver, store=AsyncSqliteStore
    return g.compile(checkpointer=checkpointer, store=store)


# ══════════════════════════════════════════════════════
# 14. 图实例
# ══════════════════════════════════════════════════════

# ★ 模块加载时（langgraph dev 扫描期）：不带 checkpointer
# CLI 模式下，_init_registry 会重建成带 MemorySaver 的版本；
# webapp 模式下，lifespan 会在 _start_mcp_sessions() 之后重建成带 AsyncSqliteSaver 的版本。
graph = build_graph()


# ══════════════════════════════════════════════════════
# 15. __main__ —— 交互式 CLI / 批量测试
# ══════════════════════════════════════════════════════
if __name__ == "__main__":

    BATCH_MODE = False  # True → 自动跑完 QUESTIONS；False → 交互式 CLI

    # ══════════════════════════════════════════════════════════════════
    # 批量测试题库（共 6 组，验证全部修复点）
    # 运行方式：BATCH_MODE = True，或在交互模式输入 'batch'
    #
    # 【组1】多轮闲聊 + 身份信息（验证 Planner 不产生幻觉任务）
    #   预期：每轮只有 1 个 direct 任务，任务数不随轮次增长
    # 【组2】单轮纯数学（验证 math_agent 路由正确）
    #   预期：每轮 1 个 math_agent 任务
    # 【组3】多任务并行（验证并行调度）
    #   预期：同一层并行执行 2 个任务，总耗时接近单任务耗时
    # 【组4】跨轮引用（验证 final_answer_node 能解析"刚才的结果"）
    #   预期：第2条能正确用上第1条的结果 56，输出 60
    # 【组5】数据库查询（验证 db_agent 路由 + SQL 生成）
    #   预期：1 个 db_agent 任务，结果包含真实数据
    # 【组6】综合压力测试（多轮身份更新 + 混合任务）
    #   预期：身份更新后回答"Alice"而非"Tony"；最后并行任务恰好 2 个
    # ══════════════════════════════════════════════════════════════════
  # ══════════════════════════════════════════════════════════════════
    # 批量测试题库 v5（共 8 组，24 题）
    #
    # 设计原则：
    #   - 每组只测一个维度，出错时能精确定位问题
    #   - 覆盖所有 agent 类型 + 所有拓扑结构（单任务/并行/串行/扇出/扇入）
    #   - 有意在晚期轮次混入身份问题，压测摘要记忆
    #   - HTTP 测试使用真实公网 API（api.github.com/zen 稳定返回）
    #
    # 组1  闲聊 + 身份建立               → 验证 direct 路由、摘要触发
    # 组2  纯数学（单/多步/跨轮引用）     → 验证 math_agent + Planner 数值注入
    # 组3  纯 DB 查询（单表/多表/聚合）   → 验证 db_agent SQL 生成质量
    # 组4  纯文件操作（读/写/目录）       → 验证 file_agent 路径处理
    # 组5  HTTP 请求                      → 验证 http_agent
    # 组6  串行依赖（DB→Math→File）       → 验证 3 层拓扑串行
    # 组7  并行 + 扇出 + 扇入            → 验证复杂拓扑
    # 组8  长对话记忆压力测试             → 验证摘要在 24 轮后仍正确
    # ══════════════════════════════════════════════════════════════════
    QUESTIONS = [

        # ══════════════════════════════════════════════════════════════════
        # 【组1】闲聊 + 身份建立（5 题）
        #
        # 预期行为：
        #   - 每题只有 1 个 direct 任务，绝不路由到 db_agent / math_agent
        #   - 第5轮提问时，Planner description 应出现"用户画像摘要"字样
        #   - 摘要在第5轮结束时（10条消息）首次触发
        #
        # 关键验证点：
        #   轮2 能复述轮1的信息（纯窗口记忆）
        #   轮5 能回答轮3的职业（跨2轮，还在窗口内）
        # ══════════════════════════════════════════════════════════════════
        "你好！我叫 Lily，今年 30 岁，住在温哥华",          # 轮1  → direct × 1
        "请复述一下我刚才告诉你的信息",                      # 轮2  → direct × 1，应答 Lily/30/温哥华
        "我是一名 UI 设计师，主要用 Figma 和 Sketch",       # 轮3  → direct × 1
        "我最近在学 Python，目标是转行做数据分析",           # 轮4  → direct × 1
        "我的职业是什么？我在学什么？",                      # 轮5  → direct × 1，应答 UI设计师/Python/数据分析
        #   ↑ 第5轮结束时（10条消息）触发首次摘要

        # ══════════════════════════════════════════════════════════════════
        # 【组2】纯数学（4 题）
        #
        # 预期行为：
        #   - 每题路由到 math_agent，工具调用可见（不是 LLM 猜）
        #   - 轮7"刚才的结果"：Planner 从上一轮 AI 回复读到 1764，description 写 "1764-100"
        #   - 轮8"再除以"：Planner 读到 1664，description 写 "1664÷4"
        #   - 工具调用序列：subtract(1764,100) → 1664，division(1664,4) → 416
        #
        # 失败信号：description 出现"刚才的结果"而不是具体数值 → 修复X失效
        # ══════════════════════════════════════════════════════════════════
        "计算 42 × 42",                                      # 轮6  → math_agent，结果 1764
        "刚才的结果减去 100",                                 # 轮7  → math_agent，subtract(1764,100)=1664
        "再把上面的结果除以 4",                               # 轮8  → math_agent，division(1664,4)=416
        "计算 sin(30°) 和 cos(60°)，各是多少？",             # 轮9  → math_agent × 2 或 1（取决于拆分）

        # ══════════════════════════════════════════════════════════════════
        # 【组3】纯 DB 查询（5 题）
        #
        # 覆盖：单表筛选 / 多表 JOIN / 聚合排序 / 子查询 / NULL 处理
        #
        # 预期：每题 1 个 db_agent，Planner 不拆成多个 DB 子任务
        # 失败信号：db_agent 调用 ask_db（被明确禁止）而不是 query_db
        # ══════════════════════════════════════════════════════════════════
        "查询所有来自 Vancouver 的活跃用户，显示姓名和邮箱",  # 轮10 → db_agent，WHERE city+status
        "统计每个城市的用户数量，按数量从高到低排列",          # 轮11 → db_agent，GROUP BY + ORDER BY
        "找出消费总额前 3 名的用户姓名和消费金额",             # 轮12 → db_agent，JOIN + SUM + LIMIT
        "查询库存数量为 0 的商品名称和分类",                   # 轮13 → db_agent，JOIN products+categories，WHERE stock=0
        "查询平均评分低于 3 分且评价数超过 1 条的商品",        # 轮14 → db_agent，HAVING

        # ══════════════════════════════════════════════════════════════════
        # 【组4】纯文件操作（3 题）
        #
        # 预期：每题 1 个 file_agent
        # 注意：write 之后 read，验证内容确实写入
        # ══════════════════════════════════════════════════════════════════
        (                                                      # 轮15 → file_agent
            "在 File_Agent/demo/ 目录下创建文件 test_note.txt，"
            "内容为：今天是测试日，Hello World！"
        ),
        "读取刚才创建的 File_Agent/demo/test_note.txt 文件，告诉我内容",  # 轮16 → file_agent，应读到上面写的内容
        "列出 File_Agent/demo/ 目录下所有文件，显示文件名",    # 轮17 → file_agent，list_directory

        # ══════════════════════════════════════════════════════════════════
        # 【组5】HTTP 请求（1 题）
        #
        # 使用 api.github.com/zen，这个接口稳定、无需 token、返回一句英文格言
        # 预期：1 个 http_agent，调用 fetch_url
        # ══════════════════════════════════════════════════════════════════
        "访问 https://api.github.com/zen，返回的是什么内容？",  # 轮18 → http_agent

        # ══════════════════════════════════════════════════════════════════
        # 【组6】串行依赖链（DB → Math → File）（1 题）
        #
        # 预期分拆（3层串行）：
        #   任务0  db_agent：查询 Montreal 的活跃用户总数
        #   任务1  math_agent：用户数 × 500，depends_on=[0]
        #   任务2  file_agent：把结果写入 File_Agent/demo/montreal_report.txt，depends_on=[1]
        #
        # 关键验证点：
        #   - 层0→层1：math_agent 的 description 里应有具体数字（不是"查询结果"）
        #   - 层1→层2：file_agent 写入的是计算后的数值
        # ══════════════════════════════════════════════════════════════════
        (                                                      # 轮19 → 3层串行
            "请依次完成三步："
            "① 查询 Montreal 的活跃用户总数；"
            "② 把这个数字乘以 500；"
            "③ 把最终结果写入 File_Agent/demo/montreal_report.txt"
        ),

        # ══════════════════════════════════════════════════════════════════
        # 【组7】复杂拓扑：并行 + 扇出 + 扇入（2 题）
        #
        # 题目A（扇入型）：两个独立 DB 查询 → 合并写文件
        #   预期分拆（2层）：
        #     层0  任务0 db_agent：查询最年轻的用户（无依赖）
        #          任务1 db_agent：查询价格最高的商品（无依赖）
        #     层1  任务2 file_agent：合并写入 File_Agent/demo/extremes.txt，depends_on=[0,1]
        #   验证：层0 两个 db_agent 真并行，层1 等两个都完成后才写文件
        #
        # 题目B（扇出型）：一个 DB 查询 → 两个独立计算
        #   预期分拆（2层）：
        #     层0  任务0 db_agent：查询 Calgary 活跃用户数
        #     层1  任务1 math_agent：用户数的平方，depends_on=[0]
        #          任务2 math_agent：用户数 × 1000，depends_on=[0]
        #   验证：层1 两个 math_agent 真并行
        # ══════════════════════════════════════════════════════════════════
        (                                                      # 轮20 → 扇入，层0并行×2 + 层1×1
            "同时帮我查两件事，然后合并写文件：\n"
            "① 查询数据库中年龄最小的用户姓名和年龄；\n"
            "② 查询价格最高的商品名称和价格；\n"
            "③ 把两个结果合并写入 File_Agent/demo/extremes.txt，"
            "格式：'最年轻用户：XXX(N岁)，最贵商品：YYY(¥ZZZ)'"
        ),
        (                                                      # 轮21 → 扇出，层0×1 + 层1并行×2
            "先查询来自 Calgary 的活跃用户数量，"
            "然后同时计算：① 这个数字的平方；② 这个数字乘以 1000"
        ),

        # ══════════════════════════════════════════════════════════════════
        # 【组8】长对话记忆压力测试（3 题）
        #
        # 此时已经经过 21 轮对话（42 条消息），摘要应已更新 4 次。
        # 组1 建立的 Lily / 温哥华 / UI设计师 / Python 信息应仍在摘要中。
        #
        # 题目1：纯身份回忆（最早期信息，轮1告知，跨20轮）
        #   预期：direct × 1，答出 Lily / 温哥华 / UI设计师
        #   失败信号：回答"没有提供个人信息" → 摘要机制失效
        #
        # 题目2：身份 + 即时计算混合（验证摘要不干扰工具路由）
        #   预期：direct × 1（身份部分） + math_agent × 1（计算部分）
        #   或拆成 2 个任务：[0] math_agent，[1] direct（depends_on=[0] 可选）
        #
        # 题目3：全面身份回忆（4 项，压力最大）
        #   预期：direct × 1，答出组1所有字段
        # ══════════════════════════════════════════════════════════════════
        "我叫什么名字？我住在哪个城市？",                     # 轮22 → direct，应答 Lily / 温哥华
        (                                                      # 轮23 → math_agent + direct
            "我的职业是什么？另外帮我计算一下：我的年龄（30岁）乘以 12 等于多少？"
        ),
        (                                                      # 轮24 → direct × 1，4 项全回忆
            "现在综合告诉我：① 我叫什么名字？"
            "② 我住在哪里？③ 我的职业是什么？④ 我在学什么技能？"
        ),
    ]

    # ★ 修复Z（核心 Checkpoint 修复）：invoke 时只传当前轮的新消息
    #
    # 原方案的致命问题：
    #   graph.ainvoke({
    #       "messages":        [HumanMessage(content=q)],  # 新消息 ✅
    #       "task_plan":       [],      # ← 强制覆盖 checkpoint 里的 task_plan ❌
    #       "current_task_id": 0,       # ← 强制覆盖 ❌
    #       "next_agent":      "",      # ← 强制覆盖 ❌
    #   }, config=config)
    #
    # LangGraph 的 checkpoint 合并规则：
    #   invoke 传入的 state 会与 checkpoint 里存的 state 做 merge。
    #   对于 Annotated[list, add_messages] 字段（messages）：新消息追加，历史保留 ✅
    #   对于普通字段（task_plan / current_task_id / next_agent）：新值直接覆盖旧值 ❌
    #
    # 每次 invoke 都传 task_plan=[] 等于每次都清空任务计划，
    # 导致即使 checkpoint 里有上一轮的 task_plan，也会被强制清零。
    #
    # 正确做法：只传当前轮的 HumanMessage，其他字段让 checkpoint 自然恢复：
    #   - messages 字段：add_messages reducer 会把新消息追加到历史末尾 ✅
    #   - task_plan 等字段：从 checkpoint 恢复上一轮的值，然后被本轮的
    #     planner_node / parallel_executor_node 正常更新 ✅
    #
    # 这样 MemorySaver 才能真正发挥作用：
    #   第 N 轮 invoke → checkpoint 保存 state（含完整消息历史）
    #   第 N+1 轮 invoke → checkpoint 恢复 → messages 里有前 N 轮的完整对话
    #   planner_node 从 messages 里找到历史 AIMessage，注入上一轮结果
    #   final_answer_node 从 messages 里读完整历史，正确解析跨轮引用
    async def _run_question(q: str, thread_id: str = "cli_user_1") -> None:
        print(f"\n{'━' * 60}\n❓ {q}\n{'━' * 60}")
        print(f"📌 thread_id: {thread_id}")

        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = await graph.ainvoke(
                # ★ 只传当前消息，不传 task_plan/current_task_id/next_agent
                # 让 LangGraph 从 checkpoint 恢复其他字段，再由各 node 正常更新
                {"messages": [HumanMessage(content=q)]},
                config=config,
            )
            answer = _get_message_content(result["messages"][-1])
            print(f"\n{'═' * 60}")
            print(f"✨ 最终答案：\n{answer}")
            print(f"{'═' * 60}")

            # ── 验证 checkpoint 是否生效 ──────────────────────────────
            # 消息数量应该随轮次递增（add_messages 追加，不覆盖）
            # 第1轮：2条（Human + AI）
            # 第2轮：4条（Human + AI + Human + AI）
            # 以此类推...
            try:
                saved_state = _checkpointer.get(config)
                if saved_state:
                    channel_values = saved_state.get("channel_values", {})
                    msgs_in_cp = channel_values.get("messages", [])
                    summary_in_cp = channel_values.get("conversation_summary", "")
                    summary_display = f"，摘要：{summary_in_cp[:40]}..." if summary_in_cp else "，摘要：（尚未生成）"
                    print(f"💾 [Checkpoint] thread '{thread_id}' 已存 {len(msgs_in_cp)} 条消息{summary_display}")
                else:
                    print(f"💾 [Checkpoint] thread '{thread_id}' 暂无存档")
            except Exception as cp_err:
                print(f"💾 [Checkpoint] 读取存档时出错：{cp_err}")

        except Exception as e:
            print(f"\n❌ 执行出错：{e}")
            traceback.print_exc()

    async def _interactive() -> None:
        print("\n" + "═" * 60)
        print("🤖  MCP Multi-Agent 并行 CLI 就绪（已启用 MemorySaver Checkpoint + InMemoryStore）")
        print("    输入问题后回车执行，输入 'quit' / 'exit' / 'q' 退出")
        print("    输入 'batch' 快速跑完 QUESTIONS 列表")
        print("    输入 'new' 开始新会话（新 thread_id）")
        print("─── Memory Store 命令（实验用）───────────────────────────")
        print("    !memory list              → 列出所有全局记忆")
        print("    !memory put <key> <value> → 写入一条全局记忆")
        print("    !memory get <key>         → 读取一条全局记忆")
        print("    !memory del <key>         → 删除一条全局记忆")
        print("    !memory clear             → 清空所有全局记忆")
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

            # ── Memory Store 命令解析 ──────────────────────────────────
            if q.startswith("!memory"):
                parts = q.split(maxsplit=3)
                cmd   = parts[1].lower() if len(parts) > 1 else "list"

                if cmd == "list":
                    items = store_list()
                    if items:
                        print("🗄️  [Store] 当前全局记忆：")
                        for k, v in items.items():
                            print(f"   {k}: {json.dumps(v, ensure_ascii=False)}")
                    else:
                        print("🗄️  [Store] 暂无全局记忆（store 为空）")

                elif cmd == "put" and len(parts) >= 4:
                    key, val = parts[2], parts[3]
                    store_put(key, val)
                    print(f"✅ 已写入：{key} = {val}")

                elif cmd == "get" and len(parts) >= 3:
                    key = parts[2]
                    val = store_get(key)
                    print(f"📖 {key} = {json.dumps(val, ensure_ascii=False) if val else '（不存在）'}")

                elif cmd == "del" and len(parts) >= 3:
                    store_delete(parts[2])

                elif cmd == "clear":
                    items = store_list()
                    for k in list(items.keys()):
                        store_delete(k)
                    print(f"🗑️  已清空 {len(items)} 条全局记忆")

                else:
                    print("⚠️  用法：!memory list / put <key> <value> / get <key> / del <key> / clear")
                continue

            await _run_question(q, thread_id=session_thread_id)

    async def _batch() -> None:
        print(f"\n🚀 批量测试模式，共 {len(QUESTIONS)} 个问题")
        # ★ 批量模式：所有问题共享同一个 batch_session thread_id，
        # 模拟一个完整的多轮对话，验证：
        #   1. 轮次增加时任务数不增长（组1）
        #   2. 跨轮引用能正确解析（组4：刚才的结果）
        #   3. 身份信息更新后能以最新为准（组6：Tony→Alice）
        thread_id = f"batch_{int(time.time())}"
        print(f"📌 批量会话 thread_id: {thread_id}")
        for idx, q in enumerate(QUESTIONS, 1):
            print(f"\n{'─' * 40}")
            print(f"📝 问题 {idx}/{len(QUESTIONS)}")
            await _run_question(q, thread_id=thread_id)

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
    
      # uv run python src/langgraph_parallel_agent.py