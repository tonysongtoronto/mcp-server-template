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

  ★ 修复I（链式聚合 → dataframe_summary 数据传递问题）
    问题根因：
      group_and_aggregate 的 result 是自然语言表格文本（"dept  budget_sum\n  市场  65000\n..."），
      而不是 JSON 数组。当下游 dataframe_summary 任务通过 depends_on 获取该 result 时，
      data_agent 在 _resolved_description 里只看到文本，找不到 JSON，拒绝调用工具。

    双层修复：
      1. Planner system prompt 新增【★ 重要：聚合结果 → dataframe_summary】规则：
         当用户要求"先聚合再对结果做 dataframe_summary"时，
         Planner 在 dataframe_summary 任务的 description 里直接嵌入
         根据原始数据推算的聚合 JSON（作为工具调用的数据底座），
         同时保留 depends_on 引用以便 data_agent 用实际数据修正。

      2. data_agent system prompt 新增【从运行时参数里构造 JSON】规则：
         当 description 里无 JSON 但【运行时参数】里有聚合文本时，
         data_agent 将空白分隔的表格文本解析为 JSON 数组后调用 dataframe_summary。
         这是双重保险：Planner 嵌入推算 JSON 是主路径，
         data_agent 自解析是兜底路径（应对 Planner 漏注入的情况）。

    效果：
      - 轮20（3步串行链：POST → group_and_aggregate → dataframe_summary）
        task[2] 将正确调用 dataframe_summary 工具，不再因"无 JSON 数据"拒绝执行。

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
              {"messages": [...], "task_plan": [],
            task_plan / next_agent 是普通字段，LangGraph merge 规则
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
      summary_turn_count：已被摘要覆盖的消息条数索引，用于控制摘要更新频率。

    摘要更新时机：在 final_answer_node 末尾，每新增 4 条消息（约 2 轮）触发一次。
      - 异步 LLM 调用生成摘要，不阻塞主流程
      - 有 existing_summary 时做增量更新，无新信息时保留现有摘要
      - 只提取用户画像，不记录工具执行结果（避免摘要膨胀）

    摘要注入位置：
      1. planner_node：摘要 + 最近3轮历史 → Planner 知道用户画像 → 正确路由（问题1自然消失）
      2. _run_direct_task：摘要 + 最近20条 → 长期记忆 + 近期细节（问题2修复）
      3. final_answer_node：摘要 + 最近20条 → 同上

    效果：
      - 无论对话多长，早期信息都不会丢失（存在摘要里）
      - 近期更新的信息优先于摘要（近期消息自然覆盖摘要中的旧值）
      - 不需要任何硬规则，Planner 看到摘要就能自主正确路由

  ★ 修复T（跨进程短对话记忆丢失 —— TEST 3 公司/产品/团队人数失败的根治方案）

    问题根因（三层叠加）：
      1. 摘要阈值 >= 10（5轮）对短对话无效：TEST 3 首次运行仅 3 轮 = 6 条消息，
         永远不触发摘要，跨进程后 conversation_summary 为空字符串。
      2. Planner 只注入"最后一条 AI 回复"作为上下文：--rerun 轮1 的 AI 只复述了
         基础身份（姓名/城市/职业），未提及工作细节（公司/产品/团队）。
         Planner 看到的"上一轮"没有公司信息，无法生成包含具体内容的 description。
      3. _run_direct_task 的 LLM 在 conv_summary 为空时，虽有 recent_history，
         但因 Planner 生成的 description 过于抽象（只写"回答公司信息"），
         LLM 注意力集中在 intent 上，对 recent_history 里的早期信息关注不足。

    三项修复（方案C）：
      a. 摘要阈值 >= 10 → >= 4（每2轮触发），短对话第2轮结束即生成摘要
      b. Planner 上下文：从"最后一条 AI"→"最近 3 轮 Human+AI 交替对话（6条）"
         Planner 能看到完整的近期对话，在 description 里写入具体内容
      c. _update_summary 每条消息截断 200 → 300 字，减少 AI 确认回复被截断的概率

    副作用：
      - 每2轮多一次 LLM 调用（摘要生成），约增加 1-2s 延迟
      - Planner 的 system prompt 略长（多了最近6条消息），token 略增

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
from typing import Any, TypedDict, Annotated
from langgraph.graph.message import add_messages   # ← 新增

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END

# ★ HITL 改动1：interrupt / Command 用于人工介入断点
#
# interrupt(payload)：在节点函数内部调用，会把 payload 存进 checkpoint 并
#   冻结图的执行（依赖已有的 AsyncSqliteSaver，天然支持跨进程恢复）。
# Command(resume=value)：恢复时用它替代普通的 state dict 传给 graph.ainvoke()，
#   value 会成为对应 interrupt() 调用的返回值，从冻结点精确继续执行。
#
# 需要较新版本的 langgraph（pip install -U langgraph）才有 langgraph.types.interrupt。
from langgraph.types import interrupt, Command

# ★ CHECKPOINT 持久化升级：MemorySaver → AsyncSqliteSaver
#
# AsyncSqliteSaver 把 checkpoint 写入本地 SQLite 文件（checkpoints.db）。
# 优点：进程重启后对话历史完整保留，零额外依赖（SQLite 是 Python 内置）。
# 安装：pip install langgraph-checkpoint-sqlite
#
# 数据库文件路径：项目根目录 / checkpoints.db（可通过环境变量 CHECKPOINT_DB 覆盖）
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver   # ← 持久化升级

# ★ STORE 改动1/5：InMemoryStore 导入（CLI 模式用；webapp 模式由 lifespan 注入 AsyncSqliteStore）
#
# 两种 Store 对比：
#   InMemoryStore   → Python 字典，进程重启清空，CLI 实验专用
#   AsyncSqliteStore → 写入磁盘，进程重启后数据仍在，webapp 生产使用
#
# Store 和 Checkpointer 的区别：
#   Checkpointer  → 存每个 thread_id 的对话历史（per-user、per-session）
#   Store         → 存跨 thread_id 共享的全局记忆（系统配置、管理员预置知识库等）
from langgraph.store.sqlite.aio import AsyncSqliteStore   # ← 持久化升级

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client
from mcp import StdioServerParameters
from pathlib import Path

_dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(str(_dotenv_path), override=False)

# ★ HITL 测试专用模块（独立文件：src/hitl_test_scenarios.py）
#
# 只在这里做一次性导入 + 极小的旁路判断，所有测试场景定义、失败模拟、
# 确定性汇总文本都在独立文件里维护，不污染本文件的正常调度逻辑。
# 用 try/except 包一层：即使这个文件被误删/损坏，也只是让 "/hitl_test ..."
# 指令退化成一次普通对话请求（Planner 会把它当成一句正常用户输入去规划），
# 不会导致整个服务起不来。
try:
    import hitl_test_scenarios as _hitl_test
except ImportError:
    _hitl_test = None

# test_agent 这个 agent 名字专供 HITL 测试场景使用，调度逻辑里有两处需要
# 识别它、不要把它当成真实 MCP agent 处理（见下方 _exec_one 里的两处引用）。
_TEST_AGENT_NAME = getattr(_hitl_test, "TEST_AGENT_NAME", "test_agent")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ══════════════════════════════════════════════════════
# 1. LLM
# ══════════════════════════════════════════════════════
llm = ChatOpenAI(
    model="deepseek-v4-flash",
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

# 8001/8003: 直接 Python server → Streamable HTTP /mcp
_SERVER_MCP_URL    = f"http://127.0.0.1:{_SERVER_PORT}/mcp"
_DB_SERVER_MCP_URL = f"http://127.0.0.1:{_DB_SERVER_PORT}/mcp"
# 8002/8004: mcp-proxy 暴露固定 SSE，无法改为 Streamable HTTP
_FS_PROXY_SSE_URL   = f"http://127.0.0.1:{_FS_PROXY_PORT}/sse"
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


# ★ HITL 改动2：Task 状态机扩展
#
# 旧状态：pending / in_progress / done / failed（4个，failed 会被"硬跑"给下游）
# 新状态机（8个，每个都有明确的下一步动作，failed 不再是终态）：
#
#   pending           → 计划里，还未调度
#   pending_approval  → 高风险操作（写文件/写数据库等），执行前需人工批准
#   in_progress       → 正在执行
#   done              → 成功
#   failed            → 瞬时态：仅在一次执行尝试失败的那一刻短暂出现，
#                        本节点内部立刻会被重新分类成 pending（可自动重试）
#                        或 needs_human（不可重试/重试耗尽），不会被返回给
#                        上层持久化为终态。
#   blocked           → 依赖链上有 needs_human/pending_approval/blocked 任务，
#                        暂时无法解析 inputs，不会被调度执行
#   needs_human       → 自动重试耗尽，或错误不可重试，等待人工决策
#   skipped           → 人工决定跳过（不影响其他任务，下游拿到占位结果）
#
# 注意：不设 "aborted" 任务状态——是否终止整个计划是全局态
# （AgentState.plan_status == "aborted"），不是某个任务的属性。
class Task(TypedDict):
    task_id: int
    description: str
    agent: str
    inputs: dict[str, TaskInput]
    depends_on: list[int]
    status: str
    result: str
    _resolved_description: str
    retry_count: int        # ★ 新增：已自动重试次数
    max_retries: int        # ★ 新增：最多自动重试几次（默认见 DEFAULT_MAX_RETRIES）
    last_error: str         # ★ 新增：最近一次失败的错误信息（needs_human 时展示给用户）
    high_risk: bool         # ★ 新增：planner 可显式标记高风险任务；未标记时由启发式规则兜底判断


# ★ HITL 改动3：人工介入相关的数据结构
#
# GateItem：parallel_executor_node 产出的"待办事项"，交给 human_review_gate
#   节点打包进 interrupt() payload，前端据此渲染决策 UI。
class GateItem(TypedDict):
    task_id: int
    reason: str                  # "needs_human" | "pending_approval"
    description: str             # 任务描述，前端展示用
    error: str | None            # needs_human 时的失败原因
    risk_type: str | None        # pending_approval 时标注是哪个 agent 触发的风险
    downstream_blocked: list[int]  # 被这个任务连带 blocked 的下游任务 id 列表


# HumanDecision：前端针对某个 GateItem 提交的决策，一次性批量提交一个数组。
class HumanDecision(TypedDict):
    task_id: int
    action: str        # "retry" | "edit_and_retry" | "skip" | "approve" | "reject" | "abort_all"
    patch: dict | None  # edit_and_retry 时可带 {"description": "..."}；
                         # skip 时可选带 {"manual_result": "..."} 由人工提供替代结果


# 自动重试策略的默认参数（可按需调整，或未来改成从 Task 里读取每任务自定义值）
DEFAULT_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 2.0

# 判定"可自动重试"的错误特征（网络/超时/限流类，通常是瞬时故障）
_RETRYABLE_ERROR_HINTS = (
    "timeout", "timed out", "connection", "connect", "temporarily",
    "rate limit", "429", "503", "502", "overloaded", "socket", "eof",
)


def _classify_failure(exc: BaseException | None) -> str:
    """判定一次任务失败是 'retryable'（值得自动重试）还是 'permanent'（不值得，直接转人工）。"""
    if exc is None:
        return "permanent"
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, ConnectionResetError, ConnectionRefusedError)):
        return "retryable"
    msg = str(exc).lower()
    return "retryable" if any(h in msg for h in _RETRYABLE_ERROR_HINTS) else "permanent"


# 高风险任务判定：agent 命中集合 + description/_resolved_description 含风险关键词。
# 这是一个启发式兜底规则，可通过 Task["high_risk"]=True 让 planner 显式标注覆盖它。
_HIGH_RISK_AGENTS = {"file_agent", "db_agent"}
_HIGH_RISK_KEYWORDS = (
    "写入", "删除", "更新", "修改", "覆盖", "移动", "创建目录",
    "write", "delete", "update", "insert", "move_file", "execute_db", "create_directory",
)


def _is_high_risk_task(task: "Task") -> bool:
    if task.get("high_risk"):
        return True
    if task.get("agent") not in _HIGH_RISK_AGENTS:
        return False
    text = f"{task.get('description', '')} {task.get('_resolved_description', '')}".lower()
    return any(kw.lower() in text for kw in _HIGH_RISK_KEYWORDS)


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
    next_agent: str
    # ★ 摘要记忆字段（解决长对话记忆丢失问题）
    #
    # conversation_summary：滚动摘要，把早期对话浓缩成自然语言。
    #   格式：用户叫Tony，住多伦多，26岁，后端工程师，用Python/Go，喜欢篮球和吉他。后更名为Alice，职业改为数据科学家。
    #   每5轮（10条消息）在 final_answer_node 末尾异步更新一次。
    #   注入到 planner_node / _run_direct_task / final_answer_node，
    #   让这三个节点都能"记得"早期信息，不受窗口限制。
    #
    # summary_turn_count：已被摘要覆盖的消息总条数（复用字段，原语义为轮次数，已重定义）。
    #   用于判断"当前消息数 - summary_turn_count >= 4"时触发更新。
    #   防止每轮都重新生成摘要（摘要生成也消耗 token）。
    conversation_summary: str   # 对话摘要，初始为空字符串
    summary_turn_count: int     # 已摘要到的消息条数索引，初始为 0
    # _thread_id 已移至 config["configurable"]["_stream_request_id"]
    # 不再污染 state，LangSmith Input/Output 只显示 messages

    # ★ HITL 改动4：AgentState 新增三个字段，支撑断点续传 + 人工介入
    #
    # plan_status：全局态，描述"整个任务计划"处于什么阶段。
    #   "running"        → 正常执行中
    #   "waiting_human"  → parallel_executor 遇到 needs_human/pending_approval，
    #                       已 interrupt()，等待前端提交决策
    #   "completed"      → 本轮所有任务都到达终态（done/skipped）
    #   "aborted"        → 用户在人工审核环节选择终止整个计划
    #
    #   注意：这是"计划"的宏观状态，不是任务状态。是否终止整个流程
    #   属于全局决策，因此 aborted 只出现在这里，不出现在 Task.status 里。
    plan_status: str

    # pending_gate_items：当前这一批待人工处理的事项（GateItem 列表）。
    #   由 parallel_executor_node 产出，human_review_gate_node 消费。
    #   human_review_gate 处理完当批决策后会清空为 []。
    pending_gate_items: list[GateItem]

    # decision_log：人工决策审计轨迹（HumanDecision 历史记录），可选但建议保留，
    #   方便前端展示"这个任务之前被人工重试过几次/何时被跳过"。
    decision_log: list[HumanDecision]


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
_CHECKPOINT_DB = os.getenv(
    "CHECKPOINT_DB",
    str(Path(__file__).parent.parent / "data" / "checkpoints.db"),
)

# ★ 统一修复：store 独立文件，不再和 checkpointer 混用同一个 SQLite。
#
# 旧做法：_store_cm = AsyncSqliteStore.from_conn_string(_CHECKPOINT_DB)
#   → 对话历史和全局记忆混存在同一个文件，备份/清理/迁移不方便。
#
# 新做法：分两个文件，路径可通过环境变量覆盖。
#   CLI / api.py 默认 → data/checkpoints.db + data/memory_store.db
#   webapp.py 默认   → 同上（已删除 webapp lifespan 里的环境变量覆盖）
#   三条路径路径完全一致，数据互通，切换运行方式不丢历史。
_STORE_DB = os.getenv(
    "STORE_DB",
    str(Path(__file__).parent.parent / "data" / "memory_store.db"),
)

# ★ 修复根因：from_conn_string() 返回的是异步上下文管理器，不是实例本身。
#   必须在 async 上下文里 __aenter__() 后才能得到真正可用的 saver/store。
#   因此这里初始化为 None，由 _open_sqlite_backends() 在异步环境里完成赋值。
#
#   两种运行模式的调用时机：
#     api.py 模式     → lifespan 最开头调 _open_sqlite_backends()，
#                        MCP 初始化（_start_mcp_sessions_stdio）在其之后
#     langgraph dev   → webapp.py lifespan 里先调 _open_sqlite_backends()，
#                        再调 _start_mcp_sessions()，最后用 webapp 自己的
#                        saver/store 重建 graph（两级 saver 互不干扰）
#     __main__ CLI    → main() 开头调 _open_sqlite_backends()
_checkpointer: AsyncSqliteSaver | None = None
_store:        AsyncSqliteStore | None = None

# 保存 context manager 引用，用于在 lifespan/main 结束时正确 __aexit__
_checkpointer_cm = None
_store_cm        = None


async def _open_sqlite_backends() -> None:
    """
    在 async 上下文里打开 SQLite 连接，赋值给模块级 _checkpointer / _store。

    调用约束：
      - 必须在任何 _init_registry() / build_graph() 调用之前完成
      - 幂等：重复调用直接跳过（已打开则不重新打开）
      - webapp 模式下调用后，lifespan 会用自己的 saver/store 重建 graph，
        这里的 _checkpointer/_store 只作"过渡桥梁"，不影响最终运行时

    数据库文件（三条路径统一，默认均在 data/ 子目录）：
      checkpointer → data/checkpoints.db  （对话历史）
      store        → data/memory_store.db  （全局记忆，独立文件）
    """
    global _checkpointer, _store, _checkpointer_cm, _store_cm
    if _checkpointer is not None:
        return  # 幂等：已初始化则跳过

    # asyncio.to_thread 把同步 I/O 移到线程池，避免在 ASGI 事件循环里阻塞
    # （langgraph dev 用 blockbuster 检测阻塞调用，直接调 os.mkdir 会被拦截）
    for _p in [Path(_CHECKPOINT_DB).parent, Path(_STORE_DB).parent]:
        await asyncio.to_thread(lambda p=_p: p.mkdir(parents=True, exist_ok=True))

    _checkpointer_cm = AsyncSqliteSaver.from_conn_string(_CHECKPOINT_DB)
    _checkpointer    = await _checkpointer_cm.__aenter__()

    _store_cm = AsyncSqliteStore.from_conn_string(_STORE_DB)   # ← 独立文件
    _store    = await _store_cm.__aenter__()

    print(f"✅ [SQLite] 持久化后端已就绪：checkpoint={_CHECKPOINT_DB}  store={_STORE_DB}")

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
_stream_queues: dict[str, asyncio.Queue] = {}   # thread_id → Queue，支持并发请求


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

    for tag, url, transport in [
        ("server.py",  _SERVER_MCP_URL,    "streamable_http"),
        ("filesystem", _FS_PROXY_SSE_URL,  "sse"),
        ("db_server",  _DB_SERVER_MCP_URL, "streamable_http"),
        ("math-mcp",   _MATH_PROXY_SSE_URL, "sse"),
    ]:
        try:
            client_ctx = (
                streamable_http_client(url) if transport == "streamable_http"
                else sse_client(url)
            )
            conn = await stack.enter_async_context(client_ctx)
            r, w = conn[0], conn[1]   # streamable_http returns (r, w, get_session_id); sse returns (r, w)
            s    = await stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            tools = await load_tools(s)
            print(f"✅ [MCP] {tag} 工具：{[t.name for t in tools]}")
            all_tools.extend(tools)
        except Exception as exc:
            print(f"❌ [MCP] {tag} Streamable HTTP 连接失败（{url}）：{exc}", file=sys.stderr)
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
    """
    ★ 修复：args_schema 直接复用 MCP inputSchema（JSON Schema dict）
    ──────────────────────────────────────────────────────────────

    新实现：langchain-core >= 0.3.40 起，StructuredTool.args_schema
            支持直接传入 JSON Schema dict（不再强制要求 pydantic BaseModel）。
            这里把 MCP 返回的 inputSchema 原样（1:1）传给 args_schema，
            bind_tools() 时会被原样转换成 OpenAI/DeepSeek 的
            function-calling parameters，模型能看到与 MCP 完全一致的
            type / description / enum / items / required 等约束。
    """
    lc_tools: list[StructuredTool] = []
    for t in (await session.list_tools()).tools:
        raw_schema: dict = t.inputSchema or {}

        # 兜底：确保是一个合法的 object schema，即使 MCP 没给 inputSchema
        # 或者 inputSchema 里缺 type/properties。
        args_schema: dict = {
            "type": raw_schema.get("type", "object"),
            "properties": raw_schema.get("properties", {}),
            "required": raw_schema.get("required", []),
        }
        # 透传其他字段（$defs / additionalProperties / title 等），
        # 避免带 $ref 嵌套结构的复杂 schema 丢信息。
        for k, v in raw_schema.items():
            if k not in args_schema:
                args_schema[k] = v

        tool_name = t.name

        async def _call(_name=tool_name, _sess=session, **kwargs) -> str:
            print(f"    🔧 [MCP] {_name}({kwargs})")
            res  = await _sess.call_tool(_name, kwargs)
            text = res.content[0].text if res.content else "（无结果）"
            print(f"    ✅ {text[:200]}")
            return text

        lc_tools.append(StructuredTool.from_function(
            coroutine=_call, name=t.name,
            description=t.description or "", args_schema=args_schema,
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
    #   所以这里传 _store 只是过渡状态，不影响最终运行。
    #
    # ★ None 兜底：模块 import 阶段（langgraph dev 扫描期）_checkpointer/_store
    #   尚未打开，此时传 None 给 build_graph 完全合法（LangGraph 允许 None）。
    #   api.py / webapp.py 的 lifespan 会在真正处理请求前重建带持久化的 graph。
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
            # ★ 修复Bug4-a：原 re.sub 只能去掉英文字母组成的标识符，
            #   但 ``` 后直接跟换行（无标识符）时，candidate 以 "\n" 开头，
            #   strip() 统一处理，兼容有/无标识符两种格式。
            candidate = re.sub(r"^[a-zA-Z]*\n", "", candidate).strip()
            if candidate:
                return candidate
        # 所有代码块都是空的，fallthrough 到下面的 fallback

    # ── fallback：找第一个 [ 或 {，截取到对应的匹配括号 ─────────────────
    # ★ 修复Bug4-b：原做法 raw[m.start():] 把括号后面的废话文字也带上，
    #   导致 json.loads 失败（如 "[{...}]\n好的，任务规划如上。"）。
    #   新做法：找到 [ 后，用括号计数法找到对应的 ] 截断，只返回 JSON 本身。
    m = re.search(r"[\[{]", raw)
    if m:
        start = m.start()
        open_char  = raw[start]
        close_char = "]" if open_char == "[" else "}"
        depth, in_str, escape = 0, False, False
        for idx in range(start, len(raw)):
            c = raw[idx]
            if escape:
                escape = False
                continue
            if c == "\\" and in_str:
                escape = True
                continue
            if c == '"':
                in_str = not in_str
            if not in_str:
                if c == open_char:
                    depth += 1
                elif c == close_char:
                    depth -= 1
                    if depth == 0:
                        return raw[start:idx + 1]
        # 括号没有闭合，返回从 start 到末尾（让上层 json.loads 报错触发重试）
        return raw[start:].strip()

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


def _drop_orphan_human_messages(msgs: list) -> list:
    """
    过滤掉"孤儿 HumanMessage"——即从未收到过对应 AI 回复的用户消息。

    典型成因：graph 在处理某条 HumanMessage 的过程中被中断（例如进程被
    SIGKILL、或客户端断开导致服务端 invoke 半途夭折）。这条消息本身在
    LangGraph 的 __start__ 阶段就已经作为 checkpoint 落盘了，但对应的
    AIMessage 永远不会生成——它会以"孤儿"的形态永久留在这个 thread 的
    历史里。

    危害：如果不过滤，下游给 planner / direct_answer / final_answer 这些
    节点组装 prompt 时，会拼出"HumanMessage 后面紧跟着另一条 HumanMessage、
    中间没有任何 AI 回复"这种在正常多轮对话里不会出现的异常结构。这种
    结构容易让 LLM 误判——即使 system prompt 里已经明确写了"只回答当前
    这一个子任务"，模型仍可能被更靠前、内容更具体的孤儿消息带偏，生成
    文不对题的长篇回答，同时白白拖慢响应速度（实测中一次简单的身份确认
    问题因此被拖到 29~32 秒）。

    规则：一条 HumanMessage 被判定为"孤儿"，当且仅当消息列表里紧跟在它
    后面的下一条消息仍然是 HumanMessage（说明它从未等到 AI 回复）。
    必须传入包含"当前这一轮"在内的完整消息列表调用本函数（不要预先
    切掉最后一条），这样列表中真正的最后一条消息才不会被误判——它后面
    没有更晚的消息，天然不满足"下一条也是 HumanMessage"的孤儿判定条件。
    调用方如果只需要"历史部分"，应该在过滤完成后再自行排除最后一条。
    """
    filtered: list = []
    n = len(msgs)
    for i, m in enumerate(msgs):
        if isinstance(m, HumanMessage):
            nxt = msgs[i + 1] if i + 1 < n else None
            if isinstance(nxt, HumanMessage):
                # 孤儿消息：紧接着的下一条依然是 HumanMessage，
                # 说明它从未被任何 AI 回复覆盖过，丢弃、不进入下游 prompt。
                continue
        filtered.append(m)
    return filtered


# ══════════════════════════════════════════════════════
# 6b. Memory Store 辅助函数（实验 & 管理用）
# ══════════════════════════════════════════════════════
#
# 这三个函数包装了 _store 的 put / get / list 操作，
# 供 CLI 交互模式（!memory 命令）和 webapp 端点调用。
# 对 InMemoryStore 用同步 API，对 AsyncSqliteStore 在外层 async 里用 await。

async def store_put(key: str, value: Any, namespace: tuple = ("system",)) -> None:
    """写入一条全局记忆（async，适配 AsyncSqliteStore）。"""
    await _store.aput(namespace, key, value if isinstance(value, dict) else {"value": value})
    print(f"  💾 [Store] 写入 {namespace}/{key} = {str(value)[:60]}")


async def store_get(key: str, namespace: tuple = ("system",)) -> Any | None:
    """读取一条全局记忆（async）。不存在时返回 None。"""
    try:
        item = await _store.aget(namespace, key)
        return item.value if item else None
    except Exception:
        return None


async def store_list(namespace: tuple = ("system",)) -> dict:
    """列出命名空间下所有记忆（async），返回 {key: value} 字典。"""
    try:
        results = await _store.asearch(namespace)
        return {r.key: r.value for r in results}
    except Exception:
        return {}


async def store_delete(key: str, namespace: tuple = ("system",)) -> bool:
    """删除一条全局记忆（async）。成功返回 True。"""
    try:
        await _store.adelete(namespace, key)
        print(f"  🗑️  [Store] 删除 {namespace}/{key}")
        return True
    except Exception as e:
        print(f"  ⚠️  [Store] 删除失败：{e}")
        return False


async def store_list_by_namespace(namespace_prefix: tuple) -> dict:
    """
    按命名空间前缀列出记忆（async），保留完整 namespace 信息。

    与 store_list() 的区别：store_list() 只返回 {key: value}，如果用前缀
    （比如 ("user",)）一次查出多个不同用户的记忆，会因为 key 相同而互相覆盖。
    这个函数返回 {namespace_tuple: {key: value}}，按子命名空间（比如各个 user_id）
    分组，供管理后台展示 "user 命名空间" 时使用。
    """
    try:
        results = await _store.asearch(namespace_prefix)
        grouped: dict = {}
        for r in results:
            grouped.setdefault(r.namespace, {})[r.key] = r.value
        return grouped
    except Exception:
        return {}


# ══════════════════════════════════════════════════════
# 7. 对话摘要生成（解决长对话记忆丢失）
# ══════════════════════════════════════════════════════

def _load_summary_dict(raw: str) -> dict:
    """
    把 state 里 conversation_summary 字段（JSON 字符串）解析成 dict。

    向后兼容：如果读到的是旧版本留下的自然语言纯文本（无法 json.loads），
    不丢弃它，包装成 {"历史摘要（旧版）": 原文本} 放进新结构里，
    后续新字段会正常追加，不会因为格式升级丢失旧数据。
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"历史摘要（旧版）": raw}


def _dump_summary_dict(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _summary_dict_to_text(data: dict) -> str:
    """把结构化摘要字段渲染成自然语言，供注入 planner / direct_task / final_answer 的 prompt。"""
    if not data:
        return ""
    return "；".join(f"{k}：{v}" for k, v in data.items())


async def _update_summary(
    messages: list,
    existing_summary: str,
) -> str:
    """
    生成/更新对话摘要（结构化增量更新策略）。

    ★ 修复"摘要自噬"（信息随轮次增加反而越来越少）：
      旧方案每次都让 LLM 把"现有摘要 + 全部历史"压缩成固定的
      "一到五句话"，随着用户画像字段增多（姓名/城市/职业/项目/习惯/搭档……），
      这个句数上限会强迫模型主动丢弃信息——句数限制和信息量增长天然矛盾，
      是信息蒸发的直接原因，而不是"小模型压缩能力不足"。

      新方案：LLM 只负责"从本次新增对话里提取增量字段"，输出一个
      JSON 补丁（只包含本轮新出现/发生变化的字段），
      合并（dict.update）这一步交给代码而不是 LLM——
      代码合并是确定性的：旧字段永远保留，除非补丁里出现了同名 key 才会被覆盖，
      彻底消除"LLM 重写时顺手漏掉几条"的可能性。

    参数：
      messages        - 完整消息列表（Human + AI 交替）
      existing_summary - 上一次的摘要（JSON 字符串，可为空字符串；
                          也兼容旧版本留下的纯文本，见 _load_summary_dict）

    返回：
      更新后的摘要（JSON 字符串，dict 形式）
    """
    existing_dict = _load_summary_dict(existing_summary)

    # 把消息列表格式化成对话文本，供 LLM 读取
    # 只取 Human + AI 交替消息（过滤掉 System / Tool 等）
    convo_lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            convo_lines.append(f"用户：{_get_message_content(m)[:300]}")
        elif isinstance(m, AIMessage):
            convo_lines.append(f"AI：{_get_message_content(m)[:300]}")
    convo_text = "\n".join(convo_lines)

    existing_keys_hint = (
        "、".join(existing_dict.keys()) if existing_dict else "（目前为空，本次是首次提取）"
    )

    prompt = (
        "你是对话信息提取助手。请从下面【新增对话内容】里提取用户的个人画像信息更新，"
        "只输出一个 JSON 对象（增量补丁），不要输出完整摘要。\n\n"
        "【规则】\n"
        "1. 只输出一个 JSON 对象，不要 markdown 代码块标记，不要任何解释性文字。\n"
        "2. key 是信息类别（中文短词），value 是该类别对应的最新值（字符串）。\n"
        "3. 只输出本次对话中新出现的信息，或发生了变化需要更新的信息；"
        "没有变化、之前已经提取过的字段【不要】重复输出（代码会自动保留旧字段，"
        "重复输出不会有额外好处，反而增加你漏掉真正新信息的风险）。\n"
        "4. 如果同一类别信息在【现有字段】里已经存在对应的 key，请复用完全相同的 key 名，"
        "不要为同一个概念新造近义词 key（比如已有'城市'就不要再造'居住地'/'所在城市'）。\n"
        "5. 常见类别命名参考（可按需使用未列出的类别，但同一概念务必保持 key 名一致）：\n"
        "   姓名、年龄、城市、职业、公司、产品名称、团队规模、编程语言、爱好、宠物、\n"
        "   饮食习惯、作息习惯、搭档信息、项目进度、其他。\n"
        "6. 不要包含工具任务结果（数学计算结果、数据库查询结果、文件操作结果等），"
        "那些不是用户画像信息。\n"
        "7. 如果本次对话里没有任何新的或变化的用户画像信息，输出空对象 {}。\n\n"
        f"【现有字段（供参考，同类信息请复用这些 key 名）】\n{existing_keys_hint}\n\n"
        "【新增对话内容】\n"
        f"{convo_text}"
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = _extract_llm_content(response).strip()
        # 防御性清理：万一模型还是套了代码块标记
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        patch = json.loads(raw) if raw else {}
        if not isinstance(patch, dict):
            raise ValueError(f"提取结果不是 JSON 对象：{raw[:100]}")
    except Exception as e:
        print(f"  ⚠️ [Summary] 增量提取失败：{e}，保留现有摘要不变")
        return _dump_summary_dict(existing_dict)

    if not patch:
        print("  📝 [Summary] 本轮无新增/变更字段，摘要保持不变")
        return _dump_summary_dict(existing_dict)

    merged = dict(existing_dict)
    merged.update(patch)  # 只有补丁里出现的 key 才会被覆盖，其余旧字段原样保留
    print(f"  📝 [Summary] 摘要字段已更新：新增/变更 {list(patch.keys())}，"
          f"当前共 {len(merged)} 个字段")
    return _dump_summary_dict(merged)


# ══════════════════════════════════════════════════════
# 8. Planner
# ══════════════════════════════════════════════════════

def _planner_system() -> str:
    return f"""你是任务规划器。把用户问题拆解为有序子任务列表。

{_registry.agent_desc_block}

{_registry.tool_desc_block}

━━ 数据传递规则（用户消息含 JSON 时必读）━━

如果用户消息中包含 JSON 数据（数组 [...] 或对象 {{...}}），
必须把该数据原样嵌入到使用它的任务的 description 字段里。
严禁只写意图（如"对用户提供的数据做统计"）——执行器看不到原始消息，
description 是它获取数据的唯一来源。

【单任务 → 直接嵌入】
  用户说："对这个数据做统计：[{{"name":"Alice","score":90}},{{"name":"Bob","score":75}}]"
  → description 写："对以下数据做统计分析：[{{"name":"Alice","score":90}},{{"name":"Bob","score":75}}]"

【多组数据 → 多任务并行，每个任务嵌入对应的那一组】
  用户说："分别统计这两组：[{{"city":"Toronto","cnt":5}}] 和 [{{"city":"NYC","cnt":8}}]"
  → task_0 description："对以下数据做统计：[{{"city":"Toronto","cnt":5}}]"
  → task_1 description："对以下数据做统计：[{{"city":"NYC","cnt":8}}]"

【数据 + 后续工具 → 只有直接使用数据的第一个任务嵌入，后续任务通过 depends_on 获取结果】
  用户说："统计这个数据后把结果 POST 到 https://api.example.com/report：[{{"product":"A","sales":100}}]"
  → task_0（data_agent）description："对以下数据做统计：[{{"product":"A","sales":100}}]"
    inputs={{}}，depends_on=[]
  → task_1（http_agent）description："向 https://api.example.com/report POST 统计结果"
    inputs={{"data": {{"from_task": 0, "field": "result"}}}}，depends_on=[0]
    ← task_1 不重复嵌入 JSON，数据通过 depends_on 从 task_0 获取

【★ 重要：聚合结果 → dataframe_summary（链式数据分析）】
  当用户要求"先聚合再对聚合结果做 dataframe_summary"时，
  dataframe_summary 的 description 必须直接嵌入原始数据（而非依赖 depends_on result）。

  原因：group_and_aggregate 返回的是自然语言表格文本（"dept  budget_sum\n  市场  65000\n  研发  150000"），
  data_agent 无法从自然语言文本中调用 dataframe_summary 工具。
  正确做法：在 dataframe_summary 任务的 description 里直接写入
  group_and_aggregate 预期输出的 JSON 形式（根据原始数据推算）。

  示例 → 用户说："按 dept 分组求 budget 总和，再对求和结果做 dataframe_summary"
  原始数据：[{{"dept":"研发","budget":50000}},{{"dept":"市场","budget":30000}},{{"dept":"研发","budget":45000}}]

  → task_0（data_agent）description："按 dept 分组对 budget 求和：[{{"dept":"研发","budget":50000}},...原始数据...]"
    depends_on=[]
  → task_1（data_agent）description：
    "对以下分组求和结果做 dataframe_summary：
     [{{"dept":"研发","budget_sum":95000}},{{"dept":"市场","budget_sum":30000}}]
     注：以上 JSON 根据原始数据推算，实际数值请以 task_0 结果为准"
    inputs={{"grouped_data": {{"from_task": 0, "field": "result"}}}}，depends_on=[0]

  ← task_1 的 description 嵌入推算 JSON，data_agent 可直接调用 dataframe_summary 工具；
    同时通过 inputs/depends_on 保留对 task_0 result 的引用，data_agent 可用实际结果修正。

【数据较长时】超过 500 字的 JSON 可截断关键字段并注明"（数据已截断，完整数据见原始消息）"。

━━ 信息时效性规则（最高优先级）━━

★ 当【最近对话历史】中包含明确的变更表述（如“搬家了”、“不再是”、“改为”、“现在”、“不再”等）时，
   这些最新消息中的信息具有最高优先级，必须**完全覆盖**下方“用户画像摘要”中的旧信息。

★ 具体做法：
   - 规划任务时，首先检查【最近对话历史】（注入在系统 prompt 末尾），提取其中关于个人属性（城市、职业、姓名、宠物数量等）的最新声明。
   - 如果发现变更，则在 description 中使用最新值，绝不使用摘要中的旧值。
   - 示例：摘要写“上海，工程师”，但最近用户说“搬到深圳，游戏策划”，则 description 必须写“深圳，游戏策划”。

★ 若最近消息未提及变更，则正常使用摘要中的信息。

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
   ★ 以下三种情况 description 必须包含实际内容，而不能只写意图：
     a. 用户消息中含 JSON 数据 → 把数据嵌入 description（见上方"数据传递规则"）
     b. Memory Store 全局记忆中有答案 → 把记忆内容写入 description
     c. 用户画像摘要中有答案 → 把摘要内容写入 description（但受上方"信息时效性规则"约束）
   原因：执行器只看 description，看不到原始消息、store 或摘要，只写意图会导致执行器反问用户。
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


async def planner_node(state: AgentState, *, store=None, config: RunnableConfig = None) -> AgentState:
    # ★ STORE 改动5/5：planner_node 新增 store=None 参数
    #
    # LangGraph 的 store 注入机制：
    #   compile(store=store) 后，LangGraph 在调用节点前会检查函数签名。
    #   如果有 `*, store=None` 关键字参数，自动把 store 对象注入进来。
    #   这是 LangGraph 原生支持的特性，不需要修改图结构。
    await _ensure_registry()

    # ★ HITL 防御性检查：正常情况下，只要 api.py 按规范先查询 /state 再决定
    #   是走 /resume 还是发新消息，就不会出现"上一轮还卡在 waiting_human，
    #   这一轮又直接发了新问题"的情况（graph.ainvoke 遇到 pending interrupt
    #   时不会自动帮你续上，而是从 planner 重新起跑，等于放弃了上一轮未决的
    #   人工审核）。这里只做日志提醒，不阻断执行——真正的阻断建议放在
    #   api.py 层（见 _reject_if_awaiting_human），因为那里能在调用 graph
    #   之前就返回明确的 409，体验更好。
    if state.get("plan_status") == "waiting_human":
        print("  ⚠️ [Planner] 检测到上一轮仍处于 waiting_human 状态却收到了新规划请求，"
              "上一轮未处理完的人工审核事项将被放弃，本轮将生成全新的 task_plan。")

    msgs = state.get("messages", [])
    
    # ── 修复：messages 为空 → 生成错误消息任务，而不是直接 FINISH ──
    if not msgs:
        error_task: Task = {
            "task_id": 0,
            "description": "向用户说明未收到任何消息",
            "agent": "direct",
            "inputs": {},
            "depends_on": [],
            "status": "done",   # ← done 状态让 parallel_executor 跳过执行
            "result": "您好，我没有收到您的消息，请重新提问。",
            "_resolved_description": "",
            "retry_count": 0, "max_retries": DEFAULT_MAX_RETRIES, "last_error": "", "high_risk": False,
        }
        return {
            "next_agent": "",          # ← 不为 "FINISH"，路由到 parallel_executor
            "task_plan": [error_task],
            # ★ HITL 改动：每次重新规划都显式重置全局态，避免上一轮遗留的
            #   waiting_human/aborted 污染新的一轮（这两个字段是普通覆盖字段，不会自动清空）。
            "plan_status": "running",
            "pending_gate_items": [],
        }

    # 取最后一条 HumanMessage 作为当前问题
    last_human_msg = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage)),
        None
    )
    
    # ── 修复：没有 HumanMessage → 生成错误消息任务，而不是直接 FINISH ──
    if not last_human_msg:
        error_task: Task = {
            "task_id": 0,
            "description": "向用户说明未找到有效问题",
            "agent": "direct",
            "inputs": {},
            "depends_on": [],
            "status": "done",
            "result": "抱歉，我没有理解您的问题，请重新描述。",
            "_resolved_description": "",
            "retry_count": 0, "max_retries": DEFAULT_MAX_RETRIES, "last_error": "", "high_risk": False,
        }
        return {
            "next_agent": "",
            "task_plan": [error_task],
            "plan_status": "running",
            "pending_gate_items": [],
        }

    user_msg = _get_message_content(last_human_msg)
    print(f"\n📋 [Planner] 规划任务：{user_msg[:80]}")

    # ★ HITL 测试专用钩子（见 src/hitl_test_scenarios.py）
    #
    # 当用户消息以 "/hitl_test " 开头时，完全跳过真实 LLM 规划，直接用
    # hitl_test_scenarios.build_scenario_task_plan() 产出一份手工编排好的
    # task_plan（agent 全部标记为 "test_agent"）。后续 parallel_executor /
    # human_review_gate / final_answer 的调度逻辑不受任何影响，走的还是
    # 真实代码路径——被跳过的只是"规划"和"最终自然语言汇总"这两步会
    # 用到真实 LLM 的环节，方便在没有 LLM API Key / 无法出网的环境
    # （比如 CI）里也能对 HITL 断点续传机制做完整的自动化回归测试。
    if _hitl_test is not None and _hitl_test.is_test_command(user_msg):
        scenario_name = _hitl_test.parse_scenario_name(user_msg)
        try:
            test_task_plan = _hitl_test.build_scenario_task_plan(scenario_name)
        except KeyError as e:
            test_task_plan = [{
                "task_id": 0,
                "description": "报告未知的 HITL 测试场景名",
                "agent": "direct",
                "inputs": {}, "depends_on": [],
                "status": "done", "result": str(e),
                "_resolved_description": "",
                "retry_count": 0, "max_retries": DEFAULT_MAX_RETRIES,
                "last_error": "", "high_risk": False,
            }]
        print(f"  🧪 [Planner] 命中 HITL 测试指令，场景={scenario_name!r}，"
              f"跳过真实 LLM 规划，任务数={len(test_task_plan)}")
        return {
            "task_plan":          test_task_plan,
            "next_agent":         "",
            "plan_status":        "running",
            "pending_gate_items": [],
        }

    # ── 调试：打印 msgs 全貌，确认跨轮历史是否被平台恢复 ────────────────
    print(f"  🔍 [Planner-debug] msgs 共 {len(msgs)} 条：")
    for i, m in enumerate(msgs):
        mtype = type(m).__name__
        content = _get_message_content(m)[:60].replace("\n", " ")
        print(f"    [{i}] {mtype}: {content}")

    # ★ 修复X（升级版）：Planner 注入摘要 + 最近 3 轮 Human+AI 对话历史
    #
    # 原方案只注入"上一轮 AI 回复"，存在两个缺陷：
    #   1. 跨进程恢复后，最近一条 AI 回复可能只复述了部分信息（如"Diana，广州，产品经理"），
    #      而工作细节（公司/产品/团队）存在更早的轮次里——单条 AI 覆盖不到。
    #   2. Planner 只看一条 AI 回复，无法判断"哪些信息已经确认过"，
    #      生成的 description 过于抽象，导致 _run_direct_task 无从回答。
    #
    # 新方案：注入最近 3 轮（6条）Human+AI 交替对话，让 Planner 有足够上下文：
    #   - 能看到历史中明确提及/确认过的信息
    #   - 能在 description 里写入具体内容（而非抽象意图）
    #   - 保留摘要作为长期记忆兜底（不受轮次限制）
    last_ai_context = ""

    # 层1：摘要注入（长期记忆，任意早期信息）
    # ★ conversation_summary 现在存的是 JSON 字符串（结构化字段），渲染成可读文本再用
    conv_summary = _summary_dict_to_text(_load_summary_dict(state.get("conversation_summary", "")))
    if conv_summary:
        last_ai_context += (
            f"\n\n【用户画像摘要（从历史对话中提取，供参考）】\n"
            f"{conv_summary}\n"
            "⚠️ 以上是对用户已知信息的摘要。若用户当前问题涉及自身信息（如'我叫什么名字'、"
            "'我住在哪里'、'我的职业是什么'），直接用 direct 回答，不要查数据库。\n"
            "⚠️ 回答时必须把摘要或近期对话中的实际内容写入 description，执行器看不到摘要本身。"
        )

    # 层2：最近 3 轮 Human+AI 交替对话（近期细节，覆盖最近 3 轮完整上下文）
    # ★ 修复"孤儿消息污染"：先对完整 msgs（含当前这一轮）做孤儿过滤，
    #   再排除当前 HumanMessage、截取最近 6 条 Human+AI 消息。
    #   顺序不能颠倒——必须在孤儿过滤时让"当前这轮问题"仍然可见，
    #   这样孤儿判定（"下一条是不是还是 HumanMessage"）才能正确识别
    #   出那些被中断、从未得到 AI 回复的历史消息。
    _cleaned_msgs = _drop_orphan_human_messages(msgs)
    recent_for_planner = [
        m for m in _cleaned_msgs[:-1]
        if isinstance(m, (HumanMessage, AIMessage))
    ][-6:]  # 最近 3 轮（6 条）

    if recent_for_planner:
        history_lines = []
        for m in recent_for_planner:
            if isinstance(m, HumanMessage):
                history_lines.append(f"用户：{_get_message_content(m)[:200]}")
            elif isinstance(m, AIMessage):
                history_lines.append(f"AI：{_get_message_content(m)[:200]}")
        last_ai_context += (
            f"\n\n【最近对话历史（最近 3 轮，供参考）】\n"
            + "\n".join(history_lines)
            + "\n⚠️ 以上是最近几轮的对话。只在当前问题引用'刚才'/'上一步'/'上面'等时才使用数值结果；"
            "用户身份信息（姓名/公司/产品等）若在其中出现，用 direct 回答时必须把具体内容写入 description。"
        )

    max_retries = 3
    task_plan: list[Task] = []

    retry_feedback:  str = ""
    last_raw_output: str = ""

    # ★ STORE 改动5/5（续）：从 Memory Store 读取全局记忆
    store_context = ""
    if store:
        try:
            if hasattr(store, "asearch"):
                system_results = await store.asearch(("system",))
            else:
                system_results = store.search(("system",))

            # ★ user 命名空间按 user_id 隔离（同一用户跨会话共享），
            #   不能按 thread_id 隔离，否则同一用户换个新会话就读不到之前写的记忆。
            #   configurable 里优先取显式传入的 user_id（api.py 的 chat/chat_stream
            #   已经会传）；兼容旧调用路径（比如 LangSmith Studio 直接调图，不传
            #   user_id）时，退化为从复合 thread_id（"{user_id}__{raw_tid}"）里拆出来。
            configurable = (config or {}).get("configurable", {})
            user_id = configurable.get("user_id", "")
            if not user_id:
                raw_thread_id = configurable.get("thread_id", "")
                if "__" in raw_thread_id:
                    user_id = raw_thread_id.split("__", 1)[0]

            if user_id:
                 if hasattr(store, "asearch"):
                    user_results = await store.asearch(("user", user_id))
                 else:
                    user_results = store.search(("user", user_id))
            else:
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
                # ★ HITL 改动：新增字段的默认值（planner 的 LLM 输出不会包含这几个字段）
                t.setdefault("retry_count", 0)
                t.setdefault("max_retries", DEFAULT_MAX_RETRIES)
                t.setdefault("last_error", "")
                t.setdefault("high_risk", False)

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
            retry_feedback = (
                f"上次输出问题：{e}\n"
                f"你的原始输出（前300字）：{last_raw_output[:300]}\n"
                f"请只输出合法的 JSON 数组，不要有任何额外文字或代码块标记。"
            )
            print(f"  ⚠️ Planner 第 {attempt+1} 次失败：{e}")
            if attempt == max_retries - 1:
                print("  ❌ Planner 全部失败，将向用户报告错误")
                # ★ 修复：Planner 全部失败时生成错误消息任务
                error_task: Task = {
                    "task_id": 0,
                    "description": "向用户说明任务规划失败，请用户重新描述需求",
                    "agent": "direct",
                    "inputs": {},
                    "depends_on": [],
                    "status": "done",
                    "result": f"抱歉，任务规划失败（已重试 {max_retries} 次）。错误信息：{e}。请重新描述您的需求，或换一种方式提问。",
                    "_resolved_description": "",
                    "retry_count": 0, "max_retries": DEFAULT_MAX_RETRIES, "last_error": "", "high_risk": False,
                }
                return {
                    "task_plan":       [error_task],
                    "next_agent":      "",   # ← 路由到 parallel_executor
                    "plan_status":     "running",
                    "pending_gate_items": [],
                }

    return {
        "task_plan":       task_plan,
        "next_agent":      "",   # ← 路由到 parallel_executor
        # ★ HITL 改动：新一轮规划开始，显式重置全局态和待办清单。
        #   防止上一轮如果是被用户中途放弃的 waiting_human/aborted 状态
        #   残留到这一轮（这两个字段是普通覆盖字段，checkpoint 恢复后
        #   不会自动清空，必须每次规划时显式重置）。
        "plan_status":        "running",
        "pending_gate_items": [],
    }
# ══════════════════════════════════════════════════════
# 8. 并行调度核心
# ══════════════════════════════════════════════════════

def _topo_layers(tasks: list[Task], pre_done: set[int] | None = None) -> list[list[Task]]:
    """
    拓扑 BFS 分层。返回 [[layer0], [layer1], ...]：
    - 同层内任务互无依赖，可 asyncio.gather() 并行执行
    - 层与层之间严格串行（后层依赖前层全部完成）

    ★ 修复Bug3：pre_done 参数接收"已完成任务的 task_id 集合"。
      checkpoint 恢复场景下，只对 pending 任务调用此函数，
      但 depends_on 中引用的可能是已完成（done）的任务 ID。
      若 done_ids 初始为空集，这些依赖永远无法满足，导致强制入队乱序。
      传入 pre_done 后，done_ids 初始就包含已完成任务，依赖正常解析。
    """
    done_ids: set[int] = set(pre_done or [])
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


def _resync_blocking(task_plan: list[Task]) -> None:
    """
    ★ HITL 改动5：级联阻塞收敛（双向），就地修改 task_plan。

    【★ 修复】这是一个纯同步计算（只读写内存里的 task_plan 列表，没有任何
    I/O），不应该是 async def。之前误标成了 async def，而两处调用点都是
    普通调用（没有 await），导致这个函数实际上从未被真正执行过——Python
    只会创建一个协程对象然后静默丢弃（垃圾回收时最多打一条
    "coroutine was never awaited" 警告，不会报错、不会挂起），
    级联阻塞逻辑等于完全失效。现在改回 def，两处调用点无需改动。

    "坏状态" = needs_human / pending_approval / blocked —— 这些任务目前都
    没有可用的 result，依赖它们的下游任务不能安全解析 inputs。

    双向收敛，循环到不动点：
      pending → blocked：任务的某个依赖处于坏状态，还不能跑，标记阻塞。
      blocked → pending：任务原本被阻塞，但阻塞它的依赖已经全部脱离坏状态
                          （比如人工把上游任务 retry 成功/skip 了），解除阻塞，
                          重新变成可调度。

    ★ 设计取舍：skipped 不计入"坏状态"。
      人工选择跳过一个任务后，该任务会带着一个占位 result（见
      human_review_gate_node），下游任务可以正常拿到这个占位值继续跑——
      这与现有代码"部分任务失败不影响其他任务汇报"的设计哲学一致，
      也避免了"永久跳过 → 下游永远 blocked 没有出路"的死循环。

    ★ 是否会死循环？不会。每轮的 bad_ids 都是基于上一轮结束时状态的
      一次性快照（标准的"标签传播/松弛迭代"模式，类似 Bellman-Ford），
      即使某个任务中途短暂"抖动"（blocked→pending→blocked），也保证在
      有限轮内收敛到不动点后 changed=False 退出。唯一的风险场景是
      task_plan 里存在真正的循环依赖（planner 的 LLM 输出畸形，比如
      A 依赖 B、B 又依赖 A）——这种情况下算法仍会收敛（不会无限循环），
      但收敛到的结果可能是"永久死锁"（A、B 都卡在 blocked，且没有
      needs_human/pending_approval 触发 interrupt 提醒人工）。
      下面加了一个迭代次数上限作为保险丝：即使未来代码改动引入了真正的
      死循环风险，也会在有限步内主动跳出并打印警告，而不是静默挂起。

    调用时机：
      1. parallel_executor_node 每层执行结束后（把新产生的失败/风险任务的
         下游标记为 blocked）
      2. human_review_gate_node 应用完人工决策后（把已解决的 blocked 任务
         解除阻塞，恢复成 pending 以便下一轮调度）
    """
    # 迭代上限：正常情况下最多需要 len(task_plan) 轮就能收敛
    #（每轮至少把"坏状态"沿依赖链传播/回收一跳），+5 留一点安全余量。
    max_iterations = len(task_plan) + 5
    changed = True
    iterations = 0
    while changed:
        iterations += 1
        if iterations > max_iterations:
            print(f"  ⚠️ [_resync_blocking] 超过 {max_iterations} 轮仍未收敛，"
                  f"疑似 task_plan 存在循环依赖，强制停止收敛（可能导致部分"
                  f"任务的 blocked 状态不准确，建议检查 planner 生成的 depends_on）")
            break
        changed = False
        bad_ids = {
            t["task_id"] for t in task_plan
            if t.get("status") in ("needs_human", "pending_approval", "blocked")
        }
        for t in task_plan:
            deps = t.get("depends_on", [])
            has_bad_dep = any(d in bad_ids for d in deps)
            if t.get("status") == "pending" and has_bad_dep:
                t["status"] = "blocked"
                changed = True
            elif t.get("status") == "blocked" and not has_bad_dep:
                t["status"] = "pending"
                changed = True


def _transitive_dependents(task_plan: list[Task], root_id: int) -> list[int]:
    """找出 task_plan 中所有直接或间接依赖 root_id 的任务 id（BFS，沿 depends_on 反向遍历）。"""
    result: set[int] = set()
    frontier = {root_id}
    while frontier:
        next_frontier: set[int] = set()
        for t in task_plan:
            tid = t["task_id"]
            if tid == root_id or tid in result:
                continue
            if any(d in frontier for d in t.get("depends_on", [])):
                result.add(tid)
                next_frontier.add(tid)
        frontier = next_frontier
    return sorted(result)


def _fill_downstream_blocked(task_plan: list[Task], gate_items: list[GateItem]) -> None:
    """给每个 GateItem 补上"它连带阻塞了哪些下游任务"，前端展示影响范围用。就地修改。"""
    for item in gate_items:
        if not item.get("downstream_blocked"):
            item["downstream_blocked"] = _transitive_dependents(task_plan, item["task_id"])


async def _spawn_session_for(
    agent_name: str,
    stack: AsyncExitStack,
    use_sse: bool = False,
) -> ClientSession:
    """为单个任务独立 spawn 一个 MCP session。"""
    if agent_name in ("math_agent",):
        stdio_params = math_mcp_params
        http_url     = _MATH_PROXY_SSE_URL
        use_streamable = False
    elif agent_name in ("file_agent",):
        stdio_params = filesystem_mcp_params
        http_url     = _FS_PROXY_SSE_URL
        use_streamable = False
    elif agent_name in ("db_agent",):
        stdio_params = db_mcp_params
        http_url     = _DB_SERVER_MCP_URL
        use_streamable = True
    else:
        stdio_params = mcp_params
        http_url     = _SERVER_MCP_URL
        use_streamable = True

    if use_sse:
        client_ctx = (
            streamable_http_client(http_url) if use_streamable
            else sse_client(http_url)
        )
        conn = await stack.enter_async_context(client_ctx)
        r, w = conn[0], conn[1]   # streamable_http returns (r, w, get_session_id); sse returns (r, w)
    else:
        r, w = await stack.enter_async_context(stdio_client(stdio_params()))

    session = await stack.enter_async_context(ClientSession(r, w))
    await session.initialize()
    return session


# ★ 修复5 — 这几个 agent 在 _spawn_session_for 里没有专属分支（会落进 else，
#   只连 server.py），但它们的工具是 _match_agent 按"工具名匹配 AGENT_TOOL_PATTERNS，
#   匹配不上/明确归属 server.py 的工具"聚合出来的逻辑分组，并不对应一个独立可 spawn
#   的 MCP 服务。之前的实现里，对这三个 agent 调 _spawn_session_for 会连到 server.py、
#   拿到 server.py 暴露的全部工具（load_tools 不按 agent 过滤），而不是 registry 里
#   按 _match_agent 精确分类过的那个子集——等于 scope 泄漏：比如 data_agent 任务实际上
#   连 fetch_url/post_json/get_server_info 也会被一并 bind_tools 绑给 LLM。
#   这几个 agent 本来就没有独立进程可 spawn，与其"假装 spawn 一次再纠错"，
#   不如直接用 registry 里已经精确分类好的全局共享工具（对应的 MCP session 由
#   _start_mcp_sessions_stdio/_start_mcp_sessions 在应用启动时常驻打开，
#   进程生命周期内一直有效，不存在"用了已关闭连接"的问题）。
_SHARED_SESSION_AGENTS = {"default_agent", "data_agent", "http_agent"}


async def run_agent_isolated(
    task: Task,
    system_prompt: str,
    use_sse: bool = False,
) -> str:
    """
    单任务执行单元：独立 spawn session → load_tools → run → close session。
    返回任务结果字符串，不修改全局状态。

    ★ 修复4 — 两层 fallback（仅适用于有独立 MCP 服务、走独立 spawn 的
      math_agent / file_agent / db_agent）：
      第一层：session spawn 失败 → 优先用该 agent 在 registry 里的已注册工具重试，
              仍为空则再退一步用 default_agent 的全局工具
      第二层：spawn 成功但工具列表为空 → 同上，先试该 agent 自己的 registry 工具，
              再退一步用 default_agent
      两层都失败时 → 纯 LLM 语言推理兜底（不调工具，直接回答）

    ★ 修复5：default_agent / data_agent / http_agent 直接走 registry 共享 session，
      不再尝试独立 spawn（原因见 _SHARED_SESSION_AGENTS 上方注释）。
    """
    agent_name = task.get("agent", "default_agent")
    intent     = task.get("_resolved_description") or task.get("description", "")

    print(f"\n🤖 [{agent_name}] 任务[{task['task_id']}] 开始：{intent[:60]}")
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
            # ★ 修复：达到最大步数时，last_response 很可能是"我需要调用工具X"
            #   这样的中间状态，直接返回会产生误导性输出。
            #   追加一次无工具的 LLM 调用，强制让模型根据已有工具结果给出最终答案。
            print(f"  ⚠️ [{agent_name}] task[{task['task_id']}] 达到最大步数 {max_steps}，"
                  f"追加一次无工具调用以汇总结果")
            summary_prompt = (
                "以上是你已完成的所有工具调用和结果。"
                "请根据这些结果，给出一个简洁的最终答案，不要再调用任何工具。"
            )
            msgs.append(HumanMessage(content=summary_prompt))
            # 用不绑定工具的 llm 调用，强制纯文本输出
            final_resp = await llm.ainvoke(msgs)
            return _extract_llm_content(final_resp) if final_resp else "（无结果）"

        return _extract_llm_content(last_response) if last_response else "（无结果）"

    def _registry_fallback(name: str, known_tools: list | None = None) -> tuple[list, str]:
        """
        按 name 从 registry 取工具+配套 prompt，找不到工具就退一步用 default_agent。
        known_tools：如果调用方已经查过 tools_for(name) 且已确认为空，
        直接传进来，避免这里再重复查一次同一个 key。
        """
        tools = _registry.tools_for(name) if known_tools is None else known_tools
        if tools:
            prompt = AGENT_SYSTEM_PROMPTS.get(name, system_prompt)
            return tools, prompt
        tools = _registry.tools_for("default_agent") or []
        prompt = AGENT_SYSTEM_PROMPTS.get("default_agent", DEFAULT_AGENT_SYSTEM_PROMPT)
        return tools, prompt

    # ── default_agent / data_agent / http_agent：无独立 MCP 服务，
    #    直接用 registry 里已按 _match_agent 精确分类好的共享工具，不走 spawn ──
    if agent_name in _SHARED_SESSION_AGENTS:
        tools = _registry.tools_for(agent_name) or []
        if not tools:
            print(f"  ⚠️ [{agent_name}] registry 中无对应工具，"
                  f"fallback → default_agent 全局工具")
            tools, prompt = _registry_fallback(agent_name, known_tools=tools)  # tools 已知为空，不重复查
        else:
            prompt = system_prompt
        result = await _run_with_tools(tools, prompt)

        elapsed = time.perf_counter() - t0
        print(f"  ⏱️ [{agent_name}] task[{task['task_id']}] 耗时 {elapsed:.2f}s")
        return result

    # ── math_agent / file_agent / db_agent：有独立 MCP 服务，独立 spawn session ──
    async with AsyncExitStack() as stack:
        try:
            session = await _spawn_session_for(agent_name, stack, use_sse=use_sse)
            tools   = await load_tools(session)

            # 第二层（spawn 成功但工具为空）：先试该 agent 自己的 registry 工具，
            # 再退一步用 default_agent（与第一层的 fallback 逻辑保持一致）
            if not tools:
                print(f"  ⚠️ [{agent_name}] session 已连接但工具列表为空，"
                      f"fallback → 使用 _registry 中的工具")
                fallback_tools, fallback_prompt = _registry_fallback(agent_name)
                result = await _run_with_tools(fallback_tools, fallback_prompt)
            else:
                result = await _run_with_tools(tools, system_prompt)

        except Exception as exc:
            # 第一层：spawn 失败 → 先试该 agent 自己的 registry 工具，
            # 再退一步用 default_agent
            print(f"  ⚠️ [{agent_name}] session 启动失败：{exc}，"
                  f"fallback → 使用 _registry 中的工具", file=sys.stderr)
            fallback_tools, fallback_prompt = _registry_fallback(agent_name)
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
    # ★ conversation_summary 是 JSON 字符串，渲染成可读文本再用
    conv_summary = _summary_dict_to_text(_load_summary_dict(state.get("conversation_summary", "")))
    summary_prefix = ""
    if conv_summary:
        summary_prefix = (
            f"【对话摘要（用户画像，供参考）】\n{conv_summary}\n\n"
        )

    # ★ 修复"孤儿消息污染"：这里是实测中真正触发 bug 的地方——
    #   过滤前，msgs[:-1] 里可能混入一条从未被回答的孤儿 HumanMessage，
    #   紧跟着 HumanMessage(content=intent) 这条任务描述，形成"连续两条
    #   Human 消息中间没有 AI 回复"的异常结构，导致模型即使被 system
    #   prompt 明确要求"只回答当前子任务"，仍会被内容更具体的孤儿消息
    #   带偏，生成文不对题的长篇回答（实测：本该回答"你叫什么名字"，
    #   却生成了一整篇跟当前问题无关的技术方案，白白耗费 25~27 秒）。
    #   先对完整 msgs 做孤儿过滤，再排除当前任务、截取最近 20 条。
    _cleaned_msgs = _drop_orphan_human_messages(msgs)
    recent_history: list = [
        m for m in _cleaned_msgs[:-1]
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
            "4. 只回答子任务要求的内容，不要主动补充与子任务无关的信息。\n"
            "5. 下方历史中如果出现某条用户消息，其后没有任何 AI 回复紧跟着，"
            "视为系统尚未处理完的残留消息，完全忽略其内容，绝不能把它当作"
            "当前要回答的问题。"
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
    核心并行调度节点（★ HITL 改造版）。

    整体流程（每次调用只处理"当前能处理的部分"，处理不完就交给 human_review_gate）：
      对每一层（拓扑排序得到）：
        Step A 高风险预批准：本层里的高风险任务摘出来，标记 pending_approval，
                不参与本层执行；同层其余任务照常并行执行（不因为兄弟任务要审批而等待）。
        Step B 执行 + 分类 + 层内自动重试：gather 执行 → 按错误类型分类 →
                可自动重试的原地重跑（带退避），重试耗尽/不可重试的转 needs_human。
        Step C 级联阻塞收敛：本层结束后重新计算哪些下游任务因为上游进了
                needs_human/pending_approval/blocked 而无法继续。
        若本轮出现任何 needs_human/pending_approval → 停止推进后续层，
        整理成 pending_gate_items，交给 human_review_gate 节点。

    checkpoint 失败恢复机制（延续原设计，含义已扩展）：
      status 为 done/skipped 的任务会被跳过，不重复执行；
      blocked/needs_human/pending_approval 的任务不会被本轮调度
      （blocked 等待级联解除，needs_human/pending_approval 等待人工决策）。
    """
    task_plan: list[Task] = state.get("task_plan", [])

    if not task_plan:
        print("\n🏁 [ParallelExecutor] task_plan 为空 → 跳过")
        return {"next_agent": "FINISH", "plan_status": "completed"}

    # 兼容旧 checkpoint：老数据里的任务可能没有这几个新字段
    for t in task_plan:
        t.setdefault("retry_count", 0)
        t.setdefault("max_retries", DEFAULT_MAX_RETRIES)
        t.setdefault("last_error", "")
        t.setdefault("high_risk", False)

    # ── 已到终态（done/skipped）的任务不再需要处理 ──────────────────────
    unresolved = [t for t in task_plan if t.get("status") not in ("done", "skipped")]
    if not unresolved:
        print("\n🏁 [ParallelExecutor] 所有任务已到终态（done/skipped）→ 直接汇总")
        return {"next_agent": "FINISH", "plan_status": "completed"}

    already_resolved = len(task_plan) - len(unresolved)
    if already_resolved > 0:
        print(f"\n⏭️  [ParallelExecutor] 跳过 {already_resolved} 个已到终态的任务（checkpoint 恢复）")

    # ★ 关键：只对 status=="pending" 的任务参与本轮拓扑分层。
    #   blocked/needs_human/pending_approval 不参与——它们要么在等级联解除，
    #   要么在等人工决策，绝不能被"强制硬跑"（这正是本次改造要修的根因）。
    already_done_ids = {t["task_id"] for t in task_plan if t.get("status") in ("done", "skipped")}
    schedulable = [t for t in task_plan if t.get("status") == "pending"]
    layers = _topo_layers(schedulable, pre_done=already_done_ids)
    total  = len(task_plan)
    done_count = len(already_done_ids)

    print(f"\n🚀 [ParallelExecutor] 本轮 {len(schedulable)} 个可调度任务，分 {len(layers)} 层"
          f"（整体进度 {done_count}/{total}）")
    for i, layer in enumerate(layers):
        print(f"   层 {i}: {[t['task_id'] for t in layer]}")

    pending_gate_items: list[GateItem] = list(state.get("pending_gate_items", []))

    # ★ 修复：_exec_one 定义在 for 循环外部，避免在循环体内反复重新定义函数。
    #   task 作为参数显式传入（而非闭包捕获），消除 late-binding 风险。
    async def _exec_one(task: Task) -> tuple[int, str]:
        agent = task.get("agent", "default_agent")

        # 第一层（执行前）：检查 registry 里该 agent 是否有工具，没有就前置 fallback。
        # ★ HITL 测试钩子：test_agent 跟 direct 一样是"虚拟 agent"，不挂真实
        #   MCP 工具，registry 里天然找不到它，必须和 direct 一起排除在这个
        #   fallback 判断之外，否则测试任务会被误改写成 default_agent，
        #   跑真实（且未 mock 的）MCP 工具调用。
        if (agent not in ("direct", _TEST_AGENT_NAME)
                and _registry.agents
                and agent not in _registry.agents
        ):
            print(f"  ⚠️ [{agent}] 不在 registry 中，前置 fallback → default_agent")
            task["agent"] = "default_agent"
            agent = "default_agent"

        if agent == "direct":
            result = await _run_direct_task(task, state)
        elif agent == _TEST_AGENT_NAME and _hitl_test is not None:
            # ★ HITL 测试钩子（见 src/hitl_test_scenarios.py）：
            #   确定性地返回成功文本，或抛出异常模拟 retryable/permanent 失败，
            #   走的仍是本函数外层 asyncio.gather + 分类重试的真实调度逻辑。
            result = await _hitl_test.run_test_agent(task, state)
        else:
            system_prompt = AGENT_SYSTEM_PROMPTS.get(agent, DEFAULT_AGENT_SYSTEM_PROMPT)
            result = await run_agent_isolated(task, system_prompt, use_sse=_use_sse())
        return task["task_id"], result

    for layer_idx, layer in enumerate(layers):
        # ── Step A：高风险预批准过滤（执行前拦截，不进入本层的 gather）──────
        auto_run: list[Task] = []
        for task in layer:
            if _is_high_risk_task(task):
                task["status"] = "pending_approval"
                print(f"  🛑 [层 {layer_idx}] 任务[{task['task_id']}] 判定为高风险操作"
                      f"（agent={task.get('agent')}），暂停等待人工批准")
                pending_gate_items.append({
                    "task_id": task["task_id"],
                    "reason": "pending_approval",
                    "description": task.get("description", ""),
                    "error": None,
                    "risk_type": task.get("agent"),
                    "downstream_blocked": [],
                })
            else:
                auto_run.append(task)

        if auto_run:
            # ── 解析运行时 inputs（依赖前面层/已完成任务的结果）──────────────
            for task in auto_run:
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

                # ★ 保留 planner 注入的JSON：以 _resolved_description 为基础，追加运行时参数
                base_desc = task.get("_resolved_description") or task["description"]
                resolved_desc = base_desc
                if resolved_parts:
                    resolved_desc += "\n\n【运行时参数】\n" + "\n".join(resolved_parts)
                task["_resolved_description"] = resolved_desc
                task["status"] = "in_progress"

            # ── Step B：并行执行 + 分类 + 层内自动重试（带退避）──────────────
            print(f"\n▶ [层 {layer_idx}] 并行执行 {len(auto_run)} 个任务："
                  f"{[t['task_id'] for t in auto_run]}")
            t_layer_start = time.perf_counter()

            current_batch = auto_run
            while current_batch:
                # return_exceptions=True：防止单任务异常拖垮整批
                raw_results = await asyncio.gather(
                    *[_exec_one(t) for t in current_batch],
                    return_exceptions=True,
                )

                retry_next: list[Task] = []
                for task, r in zip(current_batch, raw_results):
                    if isinstance(r, BaseException):
                        task["last_error"] = str(r)
                        kind = _classify_failure(r)
                        if kind == "retryable" and task["retry_count"] < task["max_retries"]:
                            task["retry_count"] += 1
                            print(f"  🔁 [层 {layer_idx}] 任务[{task['task_id']}] 失败"
                                  f"（{kind}，第 {task['retry_count']}/{task['max_retries']} 次重试）："
                                  f"{task['last_error'][:120]}")
                            retry_next.append(task)
                        else:
                            task["status"] = "needs_human"
                            if task.get("result", "") == "":
                                task["result"] = "（执行异常，等待人工处理）"
                            print(f"  ⏸️ [层 {layer_idx}] 任务[{task['task_id']}] 转人工审核"
                                  f"（{kind}，已重试 {task['retry_count']} 次）："
                                  f"{task['last_error'][:120]}")
                            pending_gate_items.append({
                                "task_id": task["task_id"],
                                "reason": "needs_human",
                                "description": task.get("description", ""),
                                "error": task["last_error"],
                                "risk_type": None,
                                "downstream_blocked": [],
                            })
                    else:
                        task_id, result = r
                        task["status"] = "done"
                        task["result"] = result
                        task["last_error"] = ""
                        done_count += 1

                if retry_next:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
                    for task in retry_next:
                        task["status"] = "in_progress"
                    current_batch = retry_next
                else:
                    current_batch = []

            layer_elapsed = time.perf_counter() - t_layer_start
            print(f"◀ [层 {layer_idx}] 结束，耗时 {layer_elapsed:.2f}s，"
                  f"整体进度 {done_count}/{total}")

        # ── Step C：本层结束，重新计算级联阻塞（不管 auto_run 是否为空都要做）──
        _resync_blocking(task_plan)

        if pending_gate_items:
            # 出现了需要人工处理的事项：不再推进到下一层
            #（下一层可能依赖本层还没跑完/被拦截的任务，继续跑会读到空结果）。
            _fill_downstream_blocked(task_plan, pending_gate_items)
            print(f"\n⏸️ [ParallelExecutor] 出现 {len(pending_gate_items)} 项待人工处理事项，"
                  f"暂停后续层，路由到 human_review_gate")
            break

    # ★ 修复（第二层保险）：判断"是否完成"不能只看本轮新产生的 pending_gate_items，
    #   还要检查 task_plan 里是否残留 needs_human/pending_approval 的孤儿任务——
    #   例如 human_review_gate 收到未知 action 后把任务转回 needs_human，
    #   但由于某种疏漏没能重新纳入 pending_gate_items（本次已在 gate 节点修复，
    #   但这里的兜底能防止未来类似疏漏再次导致任务被静默吞掉、误判为 completed）。
    orphaned = [
        t for t in task_plan
        if t.get("status") in ("needs_human", "pending_approval")
    ]
    if orphaned and not pending_gate_items:
        print(f"\n⚠️ [ParallelExecutor] 发现 {len(orphaned)} 个孤儿任务"
              f"（状态为 needs_human/pending_approval 但本轮未产生对应 gate_item），"
              f"重新打包成待办事项，避免被误判为 completed：")
        for t in orphaned:
            print(f"   task_id={t['task_id']} status={t.get('status')}")
            pending_gate_items.append({
                "task_id":            t["task_id"],
                "reason":             t.get("status"),
                "description":        t.get("description", ""),
                "error":              t.get("last_error", ""),
                "risk_type":          t.get("agent") if t.get("status") == "pending_approval" else None,
                "downstream_blocked": [],
            })
        _fill_downstream_blocked(task_plan, pending_gate_items)  

    if pending_gate_items:
        plan_status = "waiting_human"
        next_agent  = ""   # 路由改由 plan_status 驱动（见 executor_route），next_agent 不再使用
    else:
        plan_status = "completed"
        next_agent  = "FINISH"
        print(f"\n🏁 [ParallelExecutor] 全部 {total} 个任务处理完毕（含 done/skipped）")

    return {
        "task_plan":           task_plan,
        "pending_gate_items":  pending_gate_items,
        "plan_status":         plan_status,
        "next_agent":          next_agent,
    }


# ══════════════════════════════════════════════════════
# 10b. human_review_gate_node —— 人工介入断点（★ HITL 核心新增节点）
# ══════════════════════════════════════════════════════

async def human_review_gate_node(state: AgentState) -> AgentState:
    """
    职责很薄，只做两件事：
      1. 把 parallel_executor 产出的 pending_gate_items 一次性打包，
         调用 interrupt() 冻结图执行，等待前端一次性提交决策数组。
      2. 恢复后，把决策写回 task_plan（不做任何执行逻辑），
         调用 _resync_blocking 解除因决策而不再受阻的下游任务。

    为什么是"批量一次性"而不是逐个 interrupt()？
      一次 interrupt() 冻结一次 checkpoint，多个待办事项打包成一个数组一起
      展示、一起提交，用户体验上是"一次决策会话"，而不是好几轮来回等待。
      新一批待办事项（比如刚才 retry 的任务又失败了）会在 parallel_executor
      下一次运行时产生，属于新一轮 pending_gate_items，走的是新一次
      interrupt() 调用，不是同一批事项被拆开来回问。
    """
    gate_items: list[GateItem] = state.get("pending_gate_items", [])
    task_plan:  list[Task]     = state.get("task_plan", [])

    if not gate_items:
        # 安全兜底：正常路由不会在没有待办事项时进入这个节点
        return {"plan_status": "running"}

    # ★ interrupt() 调用点：LangGraph 会把这个 payload 连同当前完整 state
    #   一起写入 AsyncSqliteSaver 的 checkpoint，图在此冻结。
    #   前端可随时通过 graph.aget_state(config) 读到同样的 payload（见 api.py 的
    #   GET /session/{user_id}/{thread_id}/state）。
    #   恢复时调用方用 Command(resume=decisions) 传入的 decisions 就是这里的返回值。
    decisions: list[HumanDecision] = interrupt({
        "type":  "task_review_batch",
        "items": gate_items,
    }) or []

    print(f"\n👤 [HumanReviewGate] 收到 {len(decisions)} 条人工决策")

    by_id = {t["task_id"]: t for t in task_plan}
    # ★ 修复（安全校验）：记录这一批 interrupt() 实际交出去问人的 task_id 集合。
    #   如果不做这层校验，只要 task_id 在 task_plan 里存在就会被套用决策——
    #   这在多标签页/请求重放/客户端持有过期快照（比如刷新前拿到的旧
    #   pending_gate_items）等场景下，会把一个早就已经 done/skipped 的任务
    #   错误地重新打回 pending 并再跑一次，可能有副作用（发消息、下单等）。
    #   只允许对"确实还在等这批人工决策"的任务应用决策，其余一律忽略。
    gate_task_ids = {g["task_id"] for g in gate_items}
    decision_log = list(state.get("decision_log", []))
    plan_status  = "running"
    # ★ 修复：不再无条件返回 []，而是收集"处理完这批决策后仍然需要
    #   人工再看一眼"的事项（目前只有 else 分支的未知 action 会产生）。
    unresolved_gate_items: list[GateItem] = []

    for d in decisions:
        tid    = d.get("task_id")
        action = d.get("action")

        decision_log.append({
            "task_id":   tid,
            "action":    action,
            "patch":     d.get("patch"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        # ★ abort_all 是全局动作，不依赖某个具体 task_id 存在，单独处理
        if action == "abort_all":
            plan_status = "aborted"
            print(f"  🛑 人工选择终止整个任务计划（决策携带的 task_id={tid}，仅作参考）")
            continue

        # ★ 修复：task_id 必须属于本轮 interrupt() 交出去的那批，
        #   否则视为过期/无关决策，忽略并打日志，不去动这个任务的状态。
        if tid not in gate_task_ids:
            print(f"  ⚠️ [HumanReviewGate] 决策引用的 task_id={tid} 不在本轮待办事项中"
                  f"（可能是过期快照或重复提交），忽略")
            continue

        task = by_id.get(tid)
        if task is None:
            print(f"  ⚠️ [HumanReviewGate] 决策引用了不存在的 task_id={tid}，忽略")
            continue

        if action in ("retry", "edit_and_retry", "approve"):
            patch = d.get("patch") or {}
            if action == "edit_and_retry":
                if "description" in patch:
                    task["description"] = patch["description"]
                    # 修改了任务描述，运行时 inputs 需要重新解析，清空旧的 _resolved_description
                    task["_resolved_description"] = ""
            # ★ Bug 修复：approve 之前只把 status 改回 "pending"，没有清除
            #   high_risk 标记。parallel_executor_node 每轮调度前都会用
            #   _is_high_risk_task() 重新判定——high_risk 仍为 True 时，
            #   任务会在还没真正执行前就又被 Step A 打回 "pending_approval"，
            #   形成"批准 → 重新调度 → 又被判定为高风险 → 又要审批"的死循环，
            #   人工批准实际上永远无法让任务真正跑起来。
            #   这里显式清掉 high_risk，代表"已经被人工批准过，这一次不用再拦"。
            if action == "approve":
                task["high_risk"] = False
            task["status"]      = "pending"
            task["retry_count"] = 0
            task["last_error"]  = ""
            print(f"  ▶️ 任务[{tid}] {action} → 重新排入调度")

        elif action in ("skip", "reject"):
            task["status"] = "skipped"
            manual_result = (d.get("patch") or {}).get("manual_result")
            task["result"] = manual_result or "（该任务已被人工跳过，无可用结果）"
            print(f"  ⏭️ 任务[{tid}] 已跳过"
                  f"{'（含人工提供的替代结果）' if manual_result else ''}")

        else:
            print(f"  ⚠️ 未知 action='{action}'（task_id={tid}），保守转回 needs_human")
            task["status"] = "needs_human"
            # ★ 修复：之前这里只改了状态，却没有把任务重新塞回 gate_items，
            #   导致函数末尾统一返回的 pending_gate_items 硬编码为 []，
            #   这个任务就变成了"状态是 needs_human，但没人再问"的孤儿——
            #   下一轮 parallel_executor 不会调度它（非 pending），
            #   也不会把它重新纳入 pending_gate_items（它没有"刚失败"），
            #   如果本轮恰好没有别的新待办事项，plan_status 就会被误判为
            #   completed，图直接结束，这个任务永远得不到人工再次决策。
            unresolved_gate_items.append({
                "task_id":            tid,
                "reason":             "needs_human",
                "description":        task.get("description", ""),
                "error":              task.get("last_error", "") or f"未知的人工决策 action: {action}",
                "risk_type":          None,
                "downstream_blocked": [],
            })

    if plan_status != "aborted":
        # 解除因为决策而不再受阻的下游任务；abort 时不需要，final_answer 会统一报告
        _resync_blocking(task_plan)

    # ★ 修复：plan_status 不再无条件设为 "running"（那是函数开头的初始值），
    #   而是取决于本轮结束后是否还有未解决的 gate_items——
    #   如果有（比如上面 else 分支新产生的），必须留在 waiting_human，
    #   让 gate_route 把图重新路由回 human_review_gate 再问一轮，
    #   而不是放行到 parallel_executor / final_answer。
    if plan_status != "aborted":
        plan_status = "waiting_human" if unresolved_gate_items else "running"
        
    if unresolved_gate_items:
        _fill_downstream_blocked(task_plan, unresolved_gate_items)  

    return {
        "task_plan":          task_plan,
        "pending_gate_items": unresolved_gate_items,
        "plan_status":        plan_status,
        "decision_log":       decision_log,
    }


# ══════════════════════════════════════════════════════
# 11. final_answer_node（不变）
# ══════════════════════════════════════════════════════

async def final_answer_node(state: AgentState, config: RunnableConfig) -> AgentState:
    # ★ 修复：从 config["configurable"] 读取 request_id，不再污染 state
    #   旧：state.get("_thread_id") → 导致 LangSmith Input 显示 UUID 而非问题内容
    #   新：config["configurable"].get("_stream_request_id") → state 干净，LangSmith 正常
    tid = (config or {}).get("configurable", {}).get("_stream_request_id", "")
    print(f"  🔍 [final_answer] _stream_request_id={tid} queues={list(_stream_queues.keys())}")
    task_plan: list[Task] = state.get("task_plan", [])
    plan_status: str = state.get("plan_status", "completed")

    # ★ HITL 改动：状态标签映射扩展到全部 8 个任务状态，
    #   之前只有 done/❌ 两档，failed 会被误判成"失败"永久展示（现在 failed
    #   已经是瞬时态，几乎不会出现在这里，仍保留兜底文案以防万一）。
    _STATUS_LABELS = {
        "done":              "✅已完成",
        "skipped":           "⏭️已跳过（人工决定）",
        "blocked":           "⛔未执行（依赖任务未就绪）",
        "needs_human":       "⏸️等待人工处理",
        "pending_approval":  "⏸️等待人工审批（高风险操作）",
        "pending":           "⏳未执行",
        "in_progress":       "⏳执行中",
        "failed":            "❌失败",
    }

    def _status_label(t: Task) -> str:
        return _STATUS_LABELS.get(t.get("status", ""), f"❓{t.get('status')}")

    tool_tasks   = [t for t in task_plan if t.get("agent") != "direct"]
    direct_tasks = [t for t in task_plan if t.get("agent") == "direct"]

    all_results_lines: list[str] = []
    if direct_tasks:
        all_results_lines.append("【直接回答任务】")
        for t in direct_tasks:
            all_results_lines.append(
                f"  任务[{t['task_id']}]（{t['description']}）[{_status_label(t)}]：{t['result']}"
            )

    if tool_tasks:
        all_results_lines.append("【工具执行任务（以下为工具实际返回值，必须以此为准）】")
        for t in tool_tasks:
            all_results_lines.append(
                f"  任务[{t['task_id']}]（{t['description']}）[{_status_label(t)}]：{t['result']}"
            )

    if plan_status == "aborted":
        all_results_lines.append(
            "\n【计划状态】用户在人工审核环节主动终止了本次任务计划，"
            "以上未到达'已完成/已跳过'状态的任务均未执行。"
        )

    results_text = "\n".join(all_results_lines)
    print(f"\n  📝 汇总所有任务结果（plan_status={plan_status}）：\n{results_text}")

    msgs = state.get("messages", [])

    # ── 取当前轮问题（最后一条 HumanMessage）────────────────────────
    # msgs[-1] 就是当前轮 HumanMessage（invoke 时传入的），
    # 但用 reversed 查找更健壮，防止极端情况。
    last_human = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage)),
        HumanMessage(content="")
    )

    # ★ HITL 测试专用钩子（见 src/hitl_test_scenarios.py）
    #
    # 当这一轮的用户消息以 "/hitl_test " 开头时，跳过真实 LLM 汇总调用，
    # 直接用 hitl_test_scenarios.build_final_answer() 拼一份基于任务结果的
    # 确定性文本。这一步是整条链路里唯一必须调用真实 LLM 的环节
    # （规划已经在 planner_node 里被同样的钩子跳过了），跳过它之后，
    # 整个 "/hitl_test scenario_xxx" 请求可以在完全没有 LLM API Key /
    # 无法访问外部 LLM 服务的环境下端到端跑完，方便自动化回归测试。
    if _hitl_test is not None and _hitl_test.is_test_command(_get_message_content(last_human)):
        test_answer = _hitl_test.build_final_answer(task_plan, plan_status)
        print(f"  🧪 [FinalAnswer] 命中 HITL 测试指令，跳过真实 LLM 汇总")
        # ★ Bugfix：之前这里直接 return，完全没有往 _stream_queues[tid] 推送
        #   任何内容。api.py 的 _stream_graph_run 是靠"队列里能不能收到
        #   token"来判断本轮有没有正文可流式推送的——而这个分支完全不调用
        #   真实 LLM，graph 的 invoke_task 几乎瞬间完成，于是
        #   _stream_graph_run 里 asyncio.wait(get_task, invoke_task) 总是
        #   invoke_task 先就绪，直接走"没有中断→yield [DONE]"分支，
        #   一个 token 都没推过。前端因此收到一个空答案，AI 气泡一直是空的
        #   （就是"测试对话成功但收不到返回消息"那个现象）。
        #   这里补上跟下面 `async for chunk in llm.astream(...)` 完全同一套
        #   协议：有队列就把内容推进去，最后放哨兵 None 通知流结束。
        q = _stream_queues.get(tid) if tid else None
        if q is not None:
            await q.put(test_answer)
            await q.put(None)
        return {
            "messages":             [AIMessage(content=test_answer)],
            "conversation_summary": state.get("conversation_summary", ""),
            "summary_turn_count":   state.get("summary_turn_count", 0),
        }

    # ── 构建"摘要前缀"────────────────────────────────────────────────
    # ★ conversation_summary 是 JSON 字符串：conv_summary 保留原始字符串
    #   （后面 _update_summary 合并要用），conv_summary_text 只用于渲染显示
    conv_summary = state.get("conversation_summary", "")
    conv_summary_text = _summary_dict_to_text(_load_summary_dict(conv_summary))
    summary_prefix = ""
    if conv_summary_text:
        summary_prefix = (
            f"【对话摘要（用户画像，早期对话的精华提取）】\n{conv_summary_text}\n\n"
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
    # ★ 修复"孤儿消息污染"：与 _run_direct_task 同样的原因，final_answer
    #   节点也会读取最近历史来汇总回答。虽然实测两次都靠这里的 system
    #   prompt（"只回答用户当前这一轮的问题"）把跑题内容兜底过滤掉了，
    #   但那只是运气好，不该依赖兜底——这里同样要在源头把孤儿消息摘除。
    _cleaned_msgs = _drop_orphan_human_messages(msgs)
    recent_history: list = [
        m for m in _cleaned_msgs[:-1]
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
        "5. 对话摘要提供了用户的基础画像，近期对话历史中有更新时以最新为准。\n"
        "6. 【关键】上方子任务结果是工具的实际执行输出，必须以这些结果为准汇总回答。\n"
        "   严禁用对话历史中的旧数据替换工具的实际返回值；\n"
        "   严禁因为某个子任务失败就忽略其他已成功子任务的结果——\n"
        "   每个任务的结果独立汇报，成功的如实展示，失败的说明原因，不互相影响。\n"
        "7. 【关键】如果多个子任务中只有部分失败，其他成功任务的结果仍须完整呈现，\n"
        "   不能以'第N步失败导致后续无法执行'为由跳过已经执行并有结果的任务。\n"
        "8. 若某任务标记为'等待人工处理'或'等待人工审批'，如实告知用户该任务正在"
        "等待人工确认，不要编造它已经完成或已经失败。\n"
        "9. 若【计划状态】提示用户主动终止了计划，简洁说明哪些任务已完成、"
        "哪些因为用户中止而未执行，不要过度解释原因。"
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

    q = _stream_queues.get(tid) if tid else None

    if q is not None:
        # ── 流式路径（SSE 模式）────────────────────────────────────────
        full_content = ""
        try:
            async for chunk in llm.astream(msgs_for_llm):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_content += token
                    await q.put(token)               # 推送 token 给 SSE 端点
        finally:
            await q.put(None)                        # 哨兵：通知流结束
            _stream_queues.pop(tid, None)            # 用完清理，避免内存泄漏
        new_ai_msg = AIMessage(content=full_content)
    else:
        # ── 非流式路径（CLI 模式，行为不变）────────────────────────────
        response = await llm.ainvoke(msgs_for_llm)
        new_ai_msg = AIMessage(content=_extract_llm_content(response))

    # ── 触发摘要更新（每4条新消息更新一次，约每2轮）────────────────────
    #
    # ★ 修复（方案C）：原阈值 >= 10（约5轮）对短对话完全无效。
    #   TEST 3 首次运行只有 3 轮 = 6 条消息，永远不触发摘要，
    #   跨进程后只能依赖 recent_history，而 recent_history 在某些路径下
    #   覆盖不全（如 Planner 只注入最后一条 AI 回复）。
    #
    #   新阈值 >= 4（约每2轮）：
    #     第2轮结束（4条）→ 首次生成摘要，包含第1-2轮全部用户信息
    #     第4轮结束（8条）→ 增量更新摘要
    #     以此类推，无论对话多短，只要有2轮就有摘要兜底。
    #
    #   代价：每2轮多一次 LLM 调用（摘要生成），延迟略增约 1-2s。
    #   收益：跨进程记忆完整性大幅提升，TEST 3 三项失败全部修复。
    #
    # ★ summary_turn_count 语义：已被摘要覆盖的消息总条数（非轮次数）。
    #   注意：summary_turn_count 字段名保持不变（避免改 AgentState 定义），
    #         只改语义：以前存"已摘要轮次"，现在存"已摘要到的消息总条数"。
    full_msg_count = len(msgs) + 1         # 本轮 AI 消息追加后的总条数
    last_summarized_idx = state.get("summary_turn_count", 0)  # 字段复用，存消息条数索引
    new_summary = conv_summary
    new_summary_turn_count = last_summarized_idx

    # 每新增 4 条消息（约 2 轮对话）更新一次摘要
    if full_msg_count - last_summarized_idx >= 4:
        print(f"  🔄 [Summary] 触发摘要更新（总消息数={full_msg_count}，"
              f"上次摘要位置={last_summarized_idx}）")
        # 把本轮 AI 回复也加进去，摘要覆盖最新内容
        all_msgs_for_summary = list(msgs) + [new_ai_msg]
        new_summary = await _update_summary(all_msgs_for_summary, conv_summary)
        new_summary_turn_count = full_msg_count  # 更新已摘要消息索引

    # ★ 返回新增字段。add_messages reducer 只追加 new_ai_msg，不覆盖历史。
    # 不展开 **state：messages 用 add_messages reducer 追加，其他字段
    # LangGraph 从 checkpoint 保留，只需返回本节点实际改变的字段。
    return {
        "messages":            [new_ai_msg],
        "conversation_summary": new_summary,
        "summary_turn_count":   new_summary_turn_count,
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
        "你是数据分析专家，核心任务是处理消息中直接提供的 JSON 数据。\n\n"
        "【最高优先级规则】\n"
        "1. 数据来源：消息中如果包含【数据】标记或 JSON 数组，它就是要分析的数据，直接使用。"
        "严禁调用 fetch_url、get_server_info 等工具去获取数据。\n"
        "2. 无数据时的处理（★ 升级版）：\n"
        "   a. 如果 description 里没有 JSON，但【运行时参数】里有前置任务的聚合结果文本\n"
        "      （形如 'dept  budget_sum\\n  市场  65000\\n  研发  150000'），\n"
        "      则把该文本解析为 JSON 数组后调用工具。\n"
        "      解析方法：把空白分隔的表格文本转成 [{\"列名\": 值, ...}] 格式的 JSON。\n"
        "      示例：'dept  salary_sum\\n  工程  49000\\n  设计  26000'\n"
        "      → [{\"dept\": \"工程\", \"salary_sum\": 49000}, {\"dept\": \"设计\", \"salary_sum\": 26000}]\n"
        "      解析后直接调用 dataframe_summary(records_json='[...]')。\n"
        "   b. 如果 description 里有推算好的 JSON（Planner 预填的），优先用它，\n"
        "      再检查【运行时参数】里的实际数据是否与之一致，以实际数据为准修正后调用工具。\n"
        "   c. 如果既无 JSON 也无可解析的聚合文本，才回复：\n"
        "      '❌ 请在同一条消息中提供需要分析的 JSON 数据，例如：[{\"col\":\"val\"}]'。\n"
        "3. 禁止反问：除无数据提示外，严禁其他形式的'请提供数据'反问。\n"
        "4. 错误如实上报：工具不支持的操作（如 agg_func=median）必须直接告知用户"
        "'❌ 不支持该操作，group_and_aggregate 仅支持 sum/mean/max/min/count'，"
        "严禁自行替换为其他聚合函数或绕过限制。\n\n"
        "【工具选择规则】\n"
        "- 统计摘要（统计信息/描述/概览/行数/列名）→ 必须调用 dataframe_summary\n"
        "- 分组聚合（按 X 分组/求和/平均/最大/最小/计数）→ 必须调用 group_and_aggregate\n"
        "  ★ agg_func 只允许：sum/mean/max/min/count，传入其他值工具会报错，直接返回报错内容给用户\n"
        "- 过滤行（筛选/filter）→ filter_rows；排序 → sort_dataframe；透视 → pivot_table\n\n"
        "【调用示例】\n"
        "统计摘要：dataframe_summary(records_json='[{\"order_id\":\"A001\",\"amount\":120.5}]')\n"
        "分组聚合：group_and_aggregate(records_json='...', group_by=\"product\", agg_col=\"price\", agg_func=\"sum\")\n\n"
        "【从运行时参数里构造 JSON 的示例】\n"
        "运行时参数包含：grouped_data = 'dept  budget_sum\\n  市场  65000\\n  研发  150000'\n"
        "→ 解析为：[{\"dept\": \"市场\", \"budget_sum\": 65000}, {\"dept\": \"研发\", \"budget_sum\": 150000}]\n"
        "→ 调用：dataframe_summary(records_json='[{\"dept\":\"市场\",\"budget_sum\":65000},{\"dept\":\"研发\",\"budget_sum\":150000}]')\n\n"
        "任务要求几种操作就做几种，不要自行扩展。完成后给出简洁结论。"
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
    图结构（★ HITL 改造后）：

        planner → parallel_executor ──(plan_status=="waiting_human")──→ human_review_gate
                        ↑                                                     │
                        │                    (plan_status=="running"，恢复继续跑)│
                        └─────────────────────────────────────────────────────┘
                        │
                        └──(plan_status in "completed"/"aborted")──→ final_answer → END

    human_review_gate 内部调用 interrupt()，图在此冻结并把 payload+完整 state
    落盘到 checkpointer；前端通过 Command(resume=decisions) 恢复。
    parallel_executor ↔ human_review_gate 之间可能来回多轮
    （每轮处理一批 pending_gate_items），直到不再产生新的待办事项或用户选择终止。
    """

    def planner_route(state: AgentState) -> str:
        if state.get("next_agent") == "FINISH":
            return "END"
        return "parallel_executor"

    def executor_route(state: AgentState) -> str:
        if state.get("plan_status") == "waiting_human":
            return "human_review_gate"
        return "final_answer"

    def gate_route(state: AgentState) -> str:
        ps = state.get("plan_status", "running")
        if ps == "aborted":
            return "final_answer"
        if ps == "waiting_human":
            # 安全兜底：正常不会出现（human_review_gate 一次性处理完当批事项），
            # 万一出现（比如决策数组没能覆盖所有 gate_items），留在原地再问一轮。
            return "human_review_gate"
        return "parallel_executor"

    g = StateGraph(AgentState)
    g.add_node("planner",            planner_node)
    g.add_node("parallel_executor",  parallel_executor_node)
    g.add_node("human_review_gate",  human_review_gate_node)
    g.add_node("final_answer",       final_answer_node)

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", planner_route, {
        "END":               END,
        "parallel_executor": "parallel_executor",
    })

    g.add_conditional_edges("parallel_executor", executor_route, {
        "human_review_gate": "human_review_gate",
        "final_answer":      "final_answer",
    })

    g.add_conditional_edges("human_review_gate", gate_route, {
        "parallel_executor":  "parallel_executor",
        "human_review_gate":  "human_review_gate",
        "final_answer":       "final_answer",
    })

    g.add_edge("final_answer", END)

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

    BATCH_MODE = True  # True → 自动跑完 QUESTIONS；False → 交互式 CLI

    QUESTIONS = [
    # === 阶段1：建立初始身份（轮1-2）===
    "你好，我是工程师张伟，今年28岁，目前住在上海。我养了一只叫‘年糕’的橘猫。",

    "我最喜欢的编程语言是 Rust，最喜欢的框架是 Axum。",

    # === 阶段2：工具任务密集轰炸（轮3-5）—— 这些会生成大量工具日志，把早期信息挤出窗口，迫使系统依赖摘要 ===
    "查询一下数据库中所有来自上海的用户数量，然后用这个数字乘以 15，最后把结果写入 File_Agent/demo/shanghai_count.txt",

    "帮我计算一下 1234 × 5678 等于多少，然后用这个结果减去 8888。",

    "访问 https://api.github.com/zen，然后把返回的格言和刚才计算的最终结果一起，追加写入 File_Agent/demo/shanghai_count.txt 的末尾。",

    # === 阶段3：身份更新（轮6-7）—— 测试摘要的“更新覆盖”能力（旧信息应被覆盖）===
    "对了，我最近搬家了，现在不住上海了，我搬到了深圳。而且我不再是工程师了，我现在是一名全职的游戏策划。",

    "还有，我的猫‘年糕’最近生了一窝小猫，我现在有 5 只猫了。",

    # === 阶段4：干扰项（轮8）—— 纯计算，不涉及身份 ===
    "计算一下 99 的平方根，保留两位小数。",

    # === 阶段5：终极回忆高压测试（轮9-10）—— 此时已经过去 8 轮，摘要必须足够聪明 ===
    "我现在住在哪个城市？我的职业是什么？",

    "我最初用的编程语言是什么？我最初养的那只猫叫什么名字？我现在一共有几只猫？"
]

    # ★ 修复Z（核心 Checkpoint 修复）：invoke 时只传当前轮的新消息
    #
    # 原方案的致命问题：
    #   graph.ainvoke({
    #       "messages":        [HumanMessage(content=q)],  # 新消息 ✅
    #       "task_plan":       [],      # ← 强制覆盖 checkpoint 里的 task_plan ❌
    #
    #       "next_agent":      "",      # ← 强制覆盖 ❌
    #   }, config=config)
    #
    # LangGraph 的 checkpoint 合并规则：
    #   invoke 传入的 state 会与 checkpoint 里存的 state 做 merge。
    #   对于 Annotated[list, add_messages] 字段（messages）：新消息追加，历史保留 ✅
    #   对于普通字段（task_plan / next_agent）：新值直接覆盖旧值 ❌
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
                # ★ 只传当前消息，不传 task_plan/next_agent
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
                saved_tuple = await _checkpointer.aget_tuple(config)
                # aget_tuple() 返回 CheckpointTuple | None，需用属性访问
                # CheckpointTuple.checkpoint → dict，含 channel_values 等
                # （注意：aget() 直接返回 dict，不是 CheckpointTuple，不能用属性访问）
                if saved_tuple is not None and saved_tuple.checkpoint:
                    channel_values = saved_tuple.checkpoint.get("channel_values", {})
                    msgs_in_cp    = channel_values.get("messages", [])
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
        print("🤖  MCP Multi-Agent 并行 CLI 就绪（已启用 AsyncSqliteSaver Checkpoint + AsyncSqliteStore）")
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
                    items = await store_list()
                    if items:
                        print("🗄️  [Store] 当前全局记忆：")
                        for k, v in items.items():
                            print(f"   {k}: {json.dumps(v, ensure_ascii=False)}")
                    else:
                        print("🗄️  [Store] 暂无全局记忆（store 为空）")

                elif cmd == "put" and len(parts) >= 4:
                    key, val = parts[2], parts[3]
                    await store_put(key, val)
                    print(f"✅ 已写入：{key} = {val}")

                elif cmd == "get" and len(parts) >= 3:
                    key = parts[2]
                    val = await store_get(key)
                    print(f"📖 {key} = {json.dumps(val, ensure_ascii=False) if val else '（不存在）'}")

                elif cmd == "del" and len(parts) >= 3:
                    await store_delete(parts[2])

                elif cmd == "clear":
                    items = await store_list()
                    for k in list(items.keys()):
                        await store_delete(k)
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
        # ★ 持久化升级：先打开 SQLite 连接，再初始化 MCP sessions
        await _open_sqlite_backends()
        await _start_mcp_sessions_stdio()
        try:
            if BATCH_MODE:
                await _batch()
            else:
                await _interactive()
        finally:
            await _stop_mcp_sessions()
            # 关闭 SQLite 连接（通过 context manager 引用正确退出）
            try:
                if _store_cm is not None:
                    await _store_cm.__aexit__(None, None, None)
                if _checkpointer_cm is not None:
                    await _checkpointer_cm.__aexit__(None, None, None)
                print("✅ [SQLite] 连接已关闭")
            except Exception:
                pass

    asyncio.run(main())
    
      # uv run python src/langgraph_parallel_agent.py
    #  uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1