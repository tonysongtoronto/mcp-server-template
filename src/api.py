"""
api.py  ——  LangGraph Parallel Agent 的 FastAPI 后台服务
            持久化版本：AsyncSqliteSaver（Checkpoint）+ AsyncSqliteStore（全局记忆）
            v2.2.0 新增：user_id 多用户隔离（向后兼容单用户模式）+ 会话列表接口

【升级内容对比】
  MemorySaver（旧）     → 进程内字典，重启丢数据
  AsyncSqliteSaver（新）→ 写入 checkpoints.db，重启保留所有对话历史 ✅

  InMemoryStore（旧）   → 进程内字典，重启丢数据
  AsyncSqliteStore（新）→ 同一个 checkpoints.db，重启保留全局记忆 ✅

【v2.2.0 多用户隔离方案】
  核心思路：不新增表、不改 langgraph_parallel_agent.py / webapp.py，
            只在 api.py 这一层把 user_id 编码进 thread_id。

  内部 thread_id 格式固定为：  {user_id}__{raw_thread_id}
    - user_id      缺省值 "default" → 完全等价于旧的单用户模式
    - raw_thread_id 缺省时自动生成 "user_{8位hex}"（命名沿用旧习惯，与 user_id 无关）

  对外（请求/响应）只暴露两个字段：
    - user_id  ：用户身份（缺省 "default"）
    - thread_id：原始会话 ID（不含 user_id 前缀，前端继续像以前一样用）

  这样：
    - 老客户端不传 user_id → 行为与升级前完全一致（单用户模式）
    - 新客户端传 user_id   → 自动获得隔离，且 /sessions/{user_id} 只能查到自己的会话
    - 即使两个用户碰巧起了同名的 thread_id，底层 key 也不会冲突
      （"alice__chat1" ≠ "bob__chat1"）

  注意：这是"应用层隔离"，不是身份认证。它能保证查询/状态正确隔离，
        但 user_id 本身仍由调用方传入，没有做 token/密码校验。
        如果要防止用户 A 伪造 user_id 冒充用户 B，需要在更上层加登录鉴权
        （比如网关校验 JWT 后把真实 user_id 注入请求），这一步不在本文件范围内。

【接口一览】
  POST   /chat                    → 普通对话（等待完整答案后返回）
  GET    /chat/stream              → 流式对话（SSE，逐 token 推送）
  GET    /health                   → 健康检查（服务状态 + MCP 工具数量）
  POST   /session/new              → 新建会话（返回新 thread_id）
  GET    /sessions/{user_id}       → 列出该用户的所有历史会话（新增）
  DELETE /session/{tid}            → 清除某个会话的 checkpoint 历史（兼容旧用法，等价 user_id=default）
  DELETE /session/{user_id}/{tid}  → 清除指定用户名下某个会话（新增，显式双段路径）
  GET    /memory                   → 列出全局 Store 记忆
  POST   /memory                   → 写入一条全局 Store 记忆
  DELETE /memory/{key}             → 删除一条全局 Store 记忆

【★ HITL 改动：v2.3.0 新增 —— 人工介入断点续传接口】
  GET  /session/{user_id}/{thread_id}/state   → 查询任务计划当前状态（含待人工处理事项）
  POST /session/{user_id}/{thread_id}/resume  → 提交人工决策，从中断点精确恢复执行
  POST /session/{user_id}/{thread_id}/abort   → 放弃当前中断，直接终止整个任务计划

  配套改动：
    - /chat、/chat/stream 在调用前会先检查线程是否还冻结在上一轮的人工审核上
      （409 拒绝，避免新消息悄悄覆盖掉未处理完的中断）
    - /chat 的响应新增 plan_status / is_awaiting_human / pending_gate_items 三个字段
      （均带默认值，向后兼容老客户端）
    - /chat/stream 遇到中断时会推送 `event: interrupted` 这个专门的 SSE 事件，
      而不是让客户端傻等到 120s 超时

【启动方式】
  pip install langgraph-checkpoint-sqlite
  uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

  注意：必须 workers=1（SQLite 不支持多进程并发写）。
        如需横向扩展，把两个 SQLite 后端换成 PostgresSaver / AsyncPostgresStore。

【数据库文件位置】
  默认：项目根目录 / checkpoints.db
  自定义：设置环境变量 CHECKPOINT_DB=/path/to/your.db
"""

import sys
import json

# ──────────────────────────────────────────────────────────────────────
# Windows 编码兜底：当本进程的 stdout/stderr 被重定向到文件而不是连接到
# 控制台时（例如被测试脚本 / systemd / Docker 当作子进程启动，stdout 重定向
# 进日志文件），Python 不会沿用控制台的 UTF-8 代码页，而是退回到系统 ANSI
# 代码页（cp1252 / cp936 等）。本文件里大量 print() 用了 emoji，一旦被这种
# 方式启动就会在 lifespan 里直接 UnicodeEncodeError 崩溃、应用根本起不来。
# 这里强制把标准输出/错误流统一成 UTF-8，且遇到无法编码的字符用 replace
# 而不是抛异常，保证不管以什么方式拉起本服务都不会被这个问题绊倒。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# ★ HITL 改动：Command 用于恢复被 interrupt() 冻结的图
from langgraph.types import Command

import langgraph_parallel_agent as agent_module

# ══════════════════════════════════════════════════════
# 1. Pydantic 请求 / 响应模型
# ══════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question:  str = Field(..., description="用户输入的问题或指令")
    user_id:   str = Field("default", description="用户 ID，缺省 'default'（单用户模式，向后兼容）")
    thread_id: str = Field("",  description="会话 ID，留空则自动生成。同一 user_id 下唯一即可，不同 user_id 允许重名")

class ChatResponse(BaseModel):
    answer:        str   = Field(..., description="AI 最终回答")
    user_id:       str   = Field(..., description="本次对话所属的用户 ID")
    thread_id:     str   = Field(..., description="本次对话所属的会话 ID（不含 user_id 前缀）")
    message_count: int   = Field(..., description="该会话累计消息条数")
    duration_ms:   float = Field(..., description="本次请求耗时（毫秒）")
    # ★ HITL 改动：新增三个字段，全部带默认值，老客户端可以直接忽略、向后兼容。
    plan_status:        str        = Field("completed", description="running/waiting_human/completed/aborted")
    is_awaiting_human:    bool       = Field(False, description="True 时表示图冻结在 human_review_gate，answer 只是提示文案，需调用 /resume 才能拿到真正结果")
    pending_gate_items:    list[dict] = Field(default_factory=list, description="is_awaiting_human=True 时，本批待人工处理的事项")

class SessionResponse(BaseModel):
    user_id:    str   = Field(..., description="所属用户 ID")
    thread_id:  str   = Field(..., description="新建会话的 ID")
    created_at: float = Field(..., description="创建时间戳")

class SessionInfo(BaseModel):
    thread_id:     str            = Field(..., description="会话 ID（不含 user_id 前缀）")
    last_message:  str            = Field("", description="该会话最后一条消息内容预览")
    message_count: int            = Field(0,  description="该会话累计消息条数")
    updated_at:    float | None   = Field(None, description="最后一次更新的时间戳（秒，来自 checkpoint ts）")

class SessionListResponse(BaseModel):
    user_id:  str               = Field(..., description="查询的用户 ID")
    sessions: list[SessionInfo] = Field(..., description="该用户名下的所有会话，按更新时间倒序")

class MemoryItem(BaseModel):
    key:       str                       = Field(..., description="记忆键名")
    value:     str                       = Field(..., description="记忆内容（字符串）")
    namespace: Literal["system", "user"] = Field("system", description="记忆命名空间：system=全局共享，user=按 user_id 隔离共享")
    user_id:   str | None                = Field(None, description="namespace='user' 时必填，标识这条记忆属于哪个用户（跨该用户所有会话共享）")

class MemoryListResponse(BaseModel):
    system: dict = Field(..., description="system 命名空间下的全局记忆 {key: value}")
    user:   dict = Field(..., description="user 命名空间下的记忆，按 user_id 分组：{user_id: {key: value}}")

# ══════════════════════════════════════════════════════
# 1b. ★ HITL 改动：人工介入相关的请求/响应模型
# ══════════════════════════════════════════════════════

class TaskInfo(BaseModel):
    task_id:               int
    description:           str
    agent:                 str
    status:                str
    result:                str
    depends_on:             list[int] = Field(default_factory=list)
    retry_count:            int       = 0
    max_retries:            int       = 0
    last_error:             str       = ""
    high_risk:              bool      = False


class GateItemModel(BaseModel):
    task_id:            int
    reason:              str                 = Field(..., description="'needs_human' 或 'pending_approval'")
    description:         str                 = ""
    error:                str | None          = None
    risk_type:            str | None          = None
    downstream_blocked:   list[int]           = Field(default_factory=list)


class TaskPlanStateResponse(BaseModel):
    user_id:              str
    thread_id:            str
    plan_status:           str                = Field(..., description="running / waiting_human / completed / aborted")
    is_awaiting_human:      bool               = Field(..., description="图当前是否正冻结在 human_review_gate 等待决策")
    task_plan:              list[TaskInfo]     = Field(default_factory=list)
    pending_gate_items:      list[GateItemModel] = Field(default_factory=list)
    # ★ Bugfix：之前完全没有这个字段，导致「刷新状态」（GET /state）只能
    #   看到任务计划的状态数据，看不到真正的最终回答文本。
    answer:                 str | None = Field(None, description="最新一轮的最终回答文本；仍冻结在 human_review_gate（is_awaiting_human=True）时为 None")


class HumanDecisionIn(BaseModel):
    task_id: int
    action:   str            = Field(..., description="retry / edit_and_retry / skip / approve / reject / abort_all")
    patch:    dict | None    = Field(None, description="edit_and_retry 时可带 {'description': '...'}；skip 时可带 {'manual_result': '...'}")


class ResumeRequest(BaseModel):
    decisions: list[HumanDecisionIn] = Field(..., description="一次性提交本批全部待办事项的决策（数组顺序不要求，按 task_id 匹配）")


class ResumeResponse(BaseModel):
    user_id:               str
    thread_id:              str
    plan_status:             str
    is_awaiting_human:        bool
    answer:                   str | None       = Field(None, description="plan_status 变为 completed/aborted 时的最终回答；仍为 waiting_human 时为 None")
    pending_gate_items:        list[GateItemModel] = Field(default_factory=list, description="若恢复后又产生了新一批待办事项（比如刚重试的任务又失败了），列在这里")
    duration_ms:              float


class HealthResponse(BaseModel):
    status:            str       = Field(..., description="ok / degraded / initializing")
    tool_count:        int       = Field(..., description="已加载的 MCP 工具数量")
    agents:            list[str] = Field(..., description="已注册的 Agent 列表")
    agent_desc_block:  str       = Field(..., description="Agent 列表的可读描述文本（含各 Agent 对应工具映射，末尾附 direct 选项，用于 Prompt 拼接）")
    uptime_seconds:    float     = Field(..., description="服务运行时间（秒）")
    checkpoint_db:     str       = Field(..., description="SQLite 数据库文件路径")

# ══════════════════════════════════════════════════════
# 2. 应用生命周期
# ══════════════════════════════════════════════════════

_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    启动顺序：
      1. 打开 SQLite 连接（AsyncSqliteSaver + AsyncSqliteStore）
      2. 初始化 MCP sessions（spawn 子进程）

    关闭顺序：
      1. 关闭 MCP sessions
      2. 关闭 SQLite 连接
    """
    # ── Step 1：SQLite 持久化后端 ─────────────────────────────────────
    print("🗄️  [API] 初始化 SQLite 持久化后端...")
    try:
        await agent_module._open_sqlite_backends()
    except Exception as exc:
        print(f"❌ [API] SQLite 初始化失败：{exc}，服务以降级模式运行")

    # ── Step 2：MCP sessions ──────────────────────────────────────────
    print("🚀 [API] 初始化 MCP sessions...")
    try:
        await agent_module._start_mcp_sessions_stdio()
        print(f"✅ [API] MCP 初始化完成，共 {len(agent_module._tools)} 个工具")
    except Exception as exc:
        print(f"❌ [API] MCP 初始化失败：{exc}，服务以降级模式运行")

    yield  # ← FastAPI 在此处理请求

    # ── 关闭 ──────────────────────────────────────────────────────────
    print("🛑 [API] 关闭服务...")
    await agent_module._stop_mcp_sessions()
    try:
        # 通过 context manager 引用正确退出（_checkpointer 本身是打开后的实例，
        # 必须用 _store_cm / _checkpointer_cm 来 __aexit__）
        if agent_module._store_cm is not None:
            await agent_module._store_cm.__aexit__(None, None, None)
        if agent_module._checkpointer_cm is not None:
            await agent_module._checkpointer_cm.__aexit__(None, None, None)
        print("✅ [SQLite] 连接已关闭")
    except Exception:
        pass
    print("✅ [API] 关闭完成")


# ══════════════════════════════════════════════════════
# 3. FastAPI 应用实例
# ══════════════════════════════════════════════════════

app = FastAPI(
    title="LangGraph Parallel Agent API",
    description=(
        "Multi-Agent 并行对话后台服务（持久化版本）。\n\n"
        "**持久化**：AsyncSqliteSaver（对话历史）+ AsyncSqliteStore（全局记忆），"
        "进程重启后数据完整保留。\n\n"
        "**多用户隔离**：传 `user_id`（缺省 'default'）即可隔离不同用户的会话历史，"
        "不传则完全等价于单用户模式。"
    ),
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 生产环境请改为实际域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Thread-Id", "X-User-Id"],   # ← 让浏览器能读到这两个 header
)


# ── UTF-8 编码中间件（解决 Windows 客户端中文乱码）──────────────────
# 强制所有 JSON 响应 Content-Type 带 charset=utf-8。
# FastAPI 默认 Content-Type: application/json（不含 charset），
# Windows PowerShell / 部分 HTTP 客户端会用系统默认编码（GBK）解析，导致中文乱码。
# 加了这个中间件后，客户端收到 Content-Type: application/json; charset=utf-8，
# 会正确用 UTF-8 解码，无需客户端做任何配置。
@app.middleware("http")
async def force_utf8_middleware(request: Request, call_next):
    response = await call_next(request)
    if "application/json" in response.headers.get("content-type", ""):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

# ══════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════

# ── 多用户隔离：user_id + thread_id 编解码 ──────────────────────────
#
# 内部复合 key 格式固定为：  {user_id}__{raw_thread_id}
#   - 选用 "__"（双下划线）作为分隔符，因为 uuid.hex 和常见用户名都不含此字符，
#     冲突概率极低；即使 user_id 本身含单下划线也不会误split（用 maxsplit + 标记位）。
#   - 真正做拆分时用「找第一个 __」而不是 split("_")，避免 user_id 里有下划线时出错。

_SEP = "__"


def _normalize_user_id(user_id: str) -> str:
    """空白/空字符串都归一化为 default，保证旧客户端（不传 user_id）走单用户模式。"""
    uid = (user_id or "").strip()
    return uid if uid else "default"


def _make_internal_thread_id(user_id: str, thread_id: str) -> str:
    """
    生成写入 SQLite 的复合 thread_id。
    raw thread_id 留空时自动生成（沿用旧命名习惯 user_xxxxxxxx，注意这里的 "user_" 前缀
    只是历史命名，不代表用户身份，真正的用户身份隔离在外面的 {user_id}__ 前缀）。
    """
    uid = _normalize_user_id(user_id)
    raw = thread_id.strip() if thread_id and thread_id.strip() else f"user_{uuid.uuid4().hex[:8]}"
    if uid == "default":
        return raw
    return f"{uid}{_SEP}{raw}"


def _split_internal_thread_id(internal_tid: str) -> tuple[str, str]:
    """
    把复合 thread_id 拆回 (user_id, raw_thread_id)。
    找不到分隔符时（比如升级前遗留的旧 thread_id，没有 user_id 前缀），
    整体当作 raw_thread_id，user_id 归为 "default"，保证老数据仍然可读。
    """
    if _SEP in internal_tid:
        uid, raw = internal_tid.split(_SEP, 1)
        return uid, raw
    return "default", internal_tid


async def _get_checkpoint_message_count(config: dict) -> int:
    """
    读取 checkpoint 里已保存的消息条数。

    必须用 aget_tuple()，不能用 aget()：
      aget_tuple() → CheckpointTuple | None，有 .checkpoint 属性
      aget()       → Checkpoint(dict) | None，直接是 dict，没有 .checkpoint 属性
                     在 dict 上访问 .checkpoint 会抛 AttributeError
    """
    try:
        saved = await agent_module._checkpointer.aget_tuple(config)
        # saved 是 CheckpointTuple | None，用属性访问 .checkpoint
        if saved is not None and saved.checkpoint:
            msgs = saved.checkpoint.get("channel_values", {}).get("messages", [])
            return len(msgs)
    except Exception:
        pass
    return 0


# ── 会话级并发锁：同一个 thread_id 的请求必须排队执行 ────────────────
#
# 背景（实测踩过的坑）：
#   /chat/stream 用 asyncio.create_task 后台跑 graph.ainvoke，SSE 流的
#   [DONE] 信号只代表"回答文本已经吐完"，不代表 checkpoint 已经落盘
#   （final_answer_node 在吐完文本之后还要做摘要更新，然后才 return 触发
#   LangGraph 落 checkpoint）。如果客户端一看到 [DONE] 就立刻发下一轮
#   请求，两次 graph.ainvoke() 会各自基于同一个"旧 checkpoint 快照"起步，
#   谁的写入落盘晚，谁的结果就会把另一轮的 AI 消息/摘要更新整体覆盖掉
#   ——表现为多轮对话里某几轮的回复和摘要更新无声消失。
#
#   下面这把锁 + 修复过的 [DONE] 发送时机（见 chat_stream）双管齐下：
#   - [DONE] 改到 invoke_task 真正跑完（checkpoint 已落盘）之后再发，
#     从根上避免客户端"看到 DONE 就抢跑"。
#   - 这把锁作为兜底：即使客户端没等 DONE、或未来某个调用路径又绕过了
#     这个时序（比如重试逻辑、多个前端标签页并发发同一个 thread_id），
#     也能物理上保证同一个 thread_id 不会有两个 ainvoke 同时在跑。
#
# 注意：这是进程内锁，只对单个 uvicorn worker 有效——这也是本文件顶部
# 强调"必须 workers=1（SQLite 不支持多进程并发写）"的原因，天然满足前提。
_thread_locks: dict[str, asyncio.Lock] = {}


def _get_thread_lock(internal_tid: str) -> asyncio.Lock:
    lock = _thread_locks.get(internal_tid)
    if lock is None:
        lock = asyncio.Lock()
        _thread_locks[internal_tid] = lock
    return lock


# ══════════════════════════════════════════════════════
# 3b. ★ HITL 改动：查询 / 恢复中断状态的公共辅助函数
# ══════════════════════════════════════════════════════

async def _check_awaiting_human(config: dict) -> tuple[bool, list[dict], str]:
    """
    查询指定 thread 当前是否冻结在 human_review_gate。

    直接读 graph.aget_state(config)——这是 LangGraph 自带的 checkpoint 查询接口，
    不需要我们额外维护一张"待办表"：interrupt() 调用时已经把 payload 和完整
    state 一起落盘了，这里查到的就是最新真相，即使跨进程/客户端断线重连也一样。

    返回 (是否处于等待人工态, 待办事项列表, plan_status)
    """
    snapshot = await agent_module.graph.aget_state(config)
    if not snapshot or not snapshot.values:
        return False, [], "completed"

    plan_status = snapshot.values.get("plan_status", "completed")
    is_waiting  = bool(snapshot.next) and "human_review_gate" in snapshot.next

    gate_items = snapshot.values.get("pending_gate_items", []) if is_waiting else []
    return is_waiting, gate_items, plan_status


# ══════════════════════════════════════════════════════
# ★ 新增：流式消费的公共逻辑，chat_stream 和 resume_stream 共用
# ══════════════════════════════════════════════════════
#
# 这段本来整块写死在 chat_stream.generate() 里，只认"新提问"这一种场景。
# 但它其实跟"这次 ainvoke 到底是普通提问还是 Command(resume=...)"完全无关，
# 只关心两件事：
#   1) 从 queue 里把 final_answer_node 推过来的 token 逐个吐给客户端
#   2) invoke_task 跑完之后，判断是"正常结束"还是"又被 interrupt 冻结"，
#      发对应的 SSE 事件
# 所以直接抽出来，两个端点各自负责"怎么发起这次 ainvoke"，
# 发起之后的消费逻辑完全复用，不用维护两份几乎一样的代码。
async def _stream_graph_run(
    invoke_task: "asyncio.Task",
    q: "asyncio.Queue",
    raw_tid: str,
) -> AsyncGenerator[str, None]:
    while True:
        get_task = asyncio.ensure_future(q.get())
        done, _pending = await asyncio.wait(
            {get_task, invoke_task},
            timeout=120.0,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            get_task.cancel()
            yield "data: [ERROR] 等待超时（120s）\n\n"
            invoke_task.cancel()
            return

        if get_task in done:
            token = get_task.result()
            if token is None:
                # final_answer_node 的 llm.astream() 吐完文本了（正常收尾路径）
                break
            # ★ Bugfix v2（推翻上一版"多行 data: 帧"的方案）：
            #   上一版为了不丢换行，把一个 token 拆成多行 data: + 一个空行，
            #   一个 token 从 1 次 yield 变成 2 次 yield，客户端也从"读到
            #   一行就立刻派发"改成"必须等到空行才能确认消息结束"。这在
            #   真实 LLM 逐 token 流场景下，给本该逐字实时显示的内容凭空
            #   多引入了一个 await send() 调度点/一次"等下一行"的等待，
            #   牺牲了原本"读到即发"的实时性——不该为了修多行文本的格式
            #   丢失问题，动了单行 token 的高频热路径。
            #   改用换行转义：\u2028（Unicode 行分隔符，不会出现在正常
            #   文本里，也不会跟下面 [DONE]/[ERROR]/[WAITING_HUMAN] 的
            #   前缀判断冲突）替换 token 里的真实 \n。仍然是一个 token
            #   一行 data:、一次 yield，读到即发，完全恢复原有实时性；
            #   客户端拿到后再把 \u2028 还原成真正的换行用于渲染。
            safe_token = str(token).replace("\n", "\u2028")
            yield f"data: {safe_token}\n\n"
            continue

        # invoke_task 先完成 —— 说明图没有走到 final_answer 就结束了，
        # 最典型的原因是被 human_review_gate 的 interrupt() 冻结，
        # 根本没有 token 会被推进 queue。取消还没用上的 get_task。
        get_task.cancel()
        break

    try:
        result = await invoke_task
    except Exception as exc:
        yield f"data: [ERROR] {exc}\n\n"
        return

    # 区分"正常跑完"和"被 interrupt 冻结"两种收尾
    # （resume_stream 场景下，"又被冻结"意味着这一批决策提交后又产生了新的
    #   pending_gate_items，客户端要再调一次 resume/stream 处理下一批）
    interrupted = bool(result.get("__interrupt__")) or result.get("plan_status") == "waiting_human"
    if interrupted:
        gate_items = result.get("pending_gate_items", [])
        payload = json.dumps({
            "plan_status":        result.get("plan_status", "waiting_human"),
            "pending_gate_items": gate_items,
        }, ensure_ascii=False)
        yield f"event: interrupted\ndata: {payload}\n\n"
        yield f"data: [WAITING_HUMAN:{raw_tid}]\n\n"
    else:
        yield f"data: [DONE:{raw_tid}]\n\n"


def _snapshot_to_state_response(user_id: str, raw_tid: str, snapshot) -> "TaskPlanStateResponse":
    values      = snapshot.values or {} if snapshot else {}
    plan_status = values.get("plan_status", "completed")
    is_waiting  = bool(snapshot and snapshot.next and "human_review_gate" in snapshot.next)

    # ★ Bugfix：之前这里完全不读 messages，answer 永远拿不到，前端「刷新
    #   状态」只能瞎拼占位文案。跟 /chat、/resume 等接口一样，直接从
    #   checkpoint 里取最后一条 AIMessage 作为 answer。
    #   is_waiting=True 时刻意留空：这时最后一条消息大概率还是用户当轮
    #   提问（图还没跑到 final_answer_node），把上一轮的旧回答当成"本轮
    #   结果"展示出来反而会误导用户。
    answer = None
    if not is_waiting:
        msgs = values.get("messages", [])
        last_ai = next(
            (m for m in reversed(msgs) if isinstance(m, agent_module.AIMessage)),
            None,
        )
        if last_ai is not None:
            answer = agent_module._get_message_content(last_ai)

    return TaskPlanStateResponse(
        user_id            = user_id,
        thread_id          = raw_tid,
        plan_status        = plan_status,
        is_awaiting_human  = is_waiting,
        task_plan          = values.get("task_plan", []),
        pending_gate_items = values.get("pending_gate_items", []) if is_waiting else [],
        answer             = answer,
    )


# ══════════════════════════════════════════════════════
# 4. 核心接口：POST /chat（普通对话）
# ══════════════════════════════════════════════════════

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="发送消息（非流式）",
    description=(
        "发送一条消息，等待 Agent 完成所有子任务后返回最终答案。\n\n"
        "**多轮记忆**：对话历史持久化在 SQLite，进程重启后仍可继续上次对话。\n\n"
        "**多用户隔离**：不传 `user_id` 时默认 `'default'`（单用户模式，行为与升级前一致）；"
        "传入真实 `user_id` 后自动隔离，不同用户即使 `thread_id` 重名也不会冲突。"
    ),
)
async def chat(req: ChatRequest) -> ChatResponse:
    if not agent_module._registry.agents:
        raise HTTPException(
            status_code=503,
            detail="MCP 服务尚未就绪，请稍后重试（检查 /health 接口）",
        )

    user_id      = _normalize_user_id(req.user_id)
    internal_tid = _make_internal_thread_id(user_id, req.thread_id)
    _, raw_tid   = _split_internal_thread_id(internal_tid)
    # ★ 显式传 user_id，供 planner 节点读取 Memory Store 的 user 命名空间
    #   （("user", user_id)，跨该用户所有会话共享）；不依赖从 thread_id 反解析。
    config       = {"configurable": {"thread_id": internal_tid, "user_id": user_id}}

    start_ms = time.time()
    # ★ 同一个 thread_id 排队执行，避免和另一个并发请求互相踩 checkpoint
    async with _get_thread_lock(internal_tid):
        # ★ HITL 改动：如果上一轮还冻结在 human_review_gate 等人工决策，
        #   这里直接拒绝新消息（否则 planner 会覆盖生成一份全新 task_plan，
        #   等于悄悄丢弃了上一轮未处理完的人工审核事项）。
        awaiting, gate_items, plan_status = await _check_awaiting_human(config)
        if awaiting:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "当前会话存在待人工处理的任务，请先调用 "
                               "POST /session/{user_id}/{thread_id}/resume 处理完，"
                               "或调用 DELETE /session/{user_id}/{thread_id}/abort 放弃后再发新消息。",
                    "plan_status":        plan_status,
                    "pending_gate_items": gate_items,
                },
            )

        try:
            result = await agent_module.graph.ainvoke(
                {"messages": [HumanMessage(content=req.question)]},
                config=config,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent 执行失败：{exc}") from exc

        # ★ HITL 改动：本轮执行途中被 interrupt() 冻结了（出现了 needs_human/
        #   pending_approval），result 里不会有新的 AI 回复消息——图根本还没
        #   跑到 final_answer。这时不能像原来那样直接取 messages[-1]。
        plan_status = result.get("plan_status", "completed")
        gate_items  = result.get("pending_gate_items", [])
        interrupted = bool(result.get("__interrupt__")) or plan_status == "waiting_human"

        if interrupted:
            answer = (
                f"本次请求中有 {len(gate_items)} 个任务需要人工确认后才能继续"
                "（自动重试已耗尽，或涉及高风险操作），请查看 pending_gate_items 并调用 "
                "/resume 提交决策。"
            )
        else:
            answer = agent_module._get_message_content(result["messages"][-1])

        duration  = (time.time() - start_ms) * 1000
        msg_count = await _get_checkpoint_message_count(config)

    return ChatResponse(
        answer             = answer,
        user_id            = user_id,
        thread_id          = raw_tid,
        message_count      = msg_count,
        duration_ms        = round(duration, 1),
        plan_status        = plan_status,
        is_awaiting_human  = interrupted,
        pending_gate_items = gate_items,
    )


# ══════════════════════════════════════════════════════
# 5. 流式接口：GET /chat/stream（SSE 逐 token 推送）
# ══════════════════════════════════════════════════════

@app.get(
    "/chat/stream",
    summary="发送消息（流式 SSE）",
    description=(
        "通过 Server-Sent Events（SSE）流式推送 Agent 回答。\n\n"
        "**多用户隔离**：不传 `user_id` 时默认 `'default'`；传入真实 `user_id` 后自动隔离。\n\n"
        "**前端接入示例**：\n"
        "```javascript\n"
        "const es = new EventSource('/chat/stream?question=你好&user_id=alice&thread_id=abc');\n"
        "es.onmessage = (e) => {\n"
        "  if (e.data === '[DONE]') { es.close(); return; }\n"
        "  document.body.innerText += e.data;\n"
        "};\n"
        "```\n\n"
        "事件格式：\n"
        "- `data: <token>` — 普通 token\n"
        "- `data: [DONE]`  — 回答结束\n"
        "- `data: [ERROR] ...` — 执行出错"
    ),
)
async def chat_stream(
    question:  str = Query(..., description="用户输入的问题"),
    user_id:   str = Query("default", description="用户 ID，缺省 'default'（单用户模式）"),
    thread_id: str = Query("",  description="会话 ID，留空自动生成"),
) -> StreamingResponse:

    if not agent_module._registry.agents:
        async def _err():
            yield "data: [ERROR] MCP 服务尚未就绪，请稍后重试\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    resolved_uid  = _normalize_user_id(user_id)
    internal_tid  = _make_internal_thread_id(resolved_uid, thread_id)
    _, raw_tid    = _split_internal_thread_id(internal_tid)
    # ★ 修复1：request_id 在 config 之前生成，才能放进 configurable
    request_id    = f"{internal_tid}_{uuid.uuid4().hex[:8]}"  # 每次请求唯一，避免并发竞争
    # ★ 修复2：_stream_request_id 走 configurable（侧信道），不污染 state
    #   final_answer_node 从 config["configurable"]["_stream_request_id"] 读 queue key
    #   state 只含 messages → LangSmith Input/Output 显示正常问题内容
    # ★ 同上：补传 user_id，供 planner 节点读取 Memory Store 的 user 命名空间
    config        = {"configurable": {"thread_id": internal_tid, "user_id": resolved_uid, "_stream_request_id": request_id}}

    async def generate() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue()
        agent_module._stream_queues[request_id] = q

        # ★ 同一个 thread_id 排队执行，避免和另一个并发请求互相踩 checkpoint
        lock = _get_thread_lock(internal_tid)
        lock_acquired = False

        try:
            await lock.acquire()
            lock_acquired = True

            # ★ HITL 改动：同 /chat，先检查是否还冻结在上一轮的人工审核上
            awaiting, gate_items, plan_status = await _check_awaiting_human(config)
            if awaiting:
                payload = json.dumps({
                    "message": "当前会话存在待人工处理的任务，请先调用 /resume 处理完再发新消息。",
                    "plan_status": plan_status,
                    "pending_gate_items": gate_items,
                }, ensure_ascii=False)
                yield f"event: rejected\ndata: {payload}\n\n"
                return

            invoke_task = asyncio.create_task(
                agent_module.graph.ainvoke(
                    # ★ 修复3：state 只传 messages，去掉 _thread_id
                    {"messages": [HumanMessage(content=question)]},
                    config=config,
                )
            )

            # ★ 改动：消费逻辑抽成了 _stream_graph_run，resume/stream 端点
            #   跑的是同一份逻辑，只是发起 ainvoke 的方式不同（见该函数）。
            async for chunk in _stream_graph_run(invoke_task, q, raw_tid):
                yield chunk
        finally:
            agent_module._stream_queues.pop(request_id, None)  # 兜底清理
            if lock_acquired:
                lock.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "X-User-Id":         resolved_uid,
            "X-Thread-Id":       raw_tid,
        },
    )


# ══════════════════════════════════════════════════════
# 5b. ★ HITL 改动：任务状态查询 + 人工决策恢复接口
# ══════════════════════════════════════════════════════

@app.get(
    "/session/{user_id}/{thread_id}/state",
    response_model=TaskPlanStateResponse,
    summary="查询任务计划当前状态（含待人工处理事项）",
    description=(
        "返回当前 task_plan 的完整状态快照。前端可以据此渲染任务时间线/进度图，"
        "以及在 is_awaiting_human=True 时展示 pending_gate_items 供用户决策。\n\n"
        "直接读取 LangGraph checkpoint（graph.aget_state），不需要额外的存储层——"
        "interrupt() 冻结时已经把完整 state 落盘了，这里查到的就是最新真相，"
        "哪怕客户端断线重连、换个进程查询都一样准确。"
    ),
)
async def get_task_plan_state(user_id: str, thread_id: str) -> TaskPlanStateResponse:
    resolved_uid = _normalize_user_id(user_id)
    internal_tid = _make_internal_thread_id(resolved_uid, thread_id)
    _, raw_tid   = _split_internal_thread_id(internal_tid)
    config       = {"configurable": {"thread_id": internal_tid, "user_id": resolved_uid}}

    snapshot = await agent_module.graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="会话不存在或尚未产生任何任务计划")

    return _snapshot_to_state_response(resolved_uid, raw_tid, snapshot)


@app.post(
    "/session/{user_id}/{thread_id}/resume",
    response_model=ResumeResponse,
    summary="提交人工决策，恢复被中断的任务计划",
    description=(
        "会话冻结在 human_review_gate（is_awaiting_human=True）时调用此接口。\n\n"
        "**一次性批量提交**：decisions 数组里包含针对本批 pending_gate_items 的"
        "全部决策，一次提交即可（不需要逐个任务分别调用）。\n\n"
        "action 取值：\n"
        "- `retry` / `approve`：原样重新执行该任务\n"
        "- `edit_and_retry`：patch.description 修改任务描述后重新执行\n"
        "- `skip` / `reject`：跳过该任务，可选 patch.manual_result 提供替代结果\n"
        "- `abort_all`：终止整个任务计划（task_id 随便填一个即可，此动作是全局的）\n\n"
        "恢复后如果又出现了新一批待处理事项（比如刚重试的任务又失败了），"
        "会在响应的 pending_gate_items 里返回，此时 is_awaiting_human 仍为 True，"
        "需要再调用一次本接口处理新一批。"
    ),
)
async def resume_task_plan(user_id: str, thread_id: str, req: ResumeRequest) -> ResumeResponse:
    resolved_uid = _normalize_user_id(user_id)
    internal_tid = _make_internal_thread_id(resolved_uid, thread_id)
    _, raw_tid   = _split_internal_thread_id(internal_tid)
    config       = {"configurable": {"thread_id": internal_tid, "user_id": resolved_uid}}

    start_ms = time.time()
    async with _get_thread_lock(internal_tid):
        awaiting, _gate_items, plan_status = await _check_awaiting_human(config)
        if not awaiting:
            raise HTTPException(
                status_code=409,
                detail=f"当前会话不处于等待人工处理状态（plan_status={plan_status}），无需调用 /resume。",
            )

        decisions_payload = [d.model_dump() for d in req.decisions]
        try:
            # ★ HITL 核心：Command(resume=decisions_payload) 会被喂给冻结在
            #   human_review_gate 里那次 interrupt() 调用，作为它的返回值，
            #   图从冻结点精确继续执行——不会重新跑 planner，也不会重复执行
            #   已经 done 的任务。
            result = await agent_module.graph.ainvoke(
                Command(resume=decisions_payload),
                config=config,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"恢复执行失败：{exc}") from exc

        new_plan_status = result.get("plan_status", "completed")
        gate_items       = result.get("pending_gate_items", [])
        still_waiting    = bool(result.get("__interrupt__")) or new_plan_status == "waiting_human"

        answer = None
        if not still_waiting:
            # plan_status 变成了 completed 或 aborted，图已经跑到 final_answer 了
            answer = agent_module._get_message_content(result["messages"][-1])

        duration = (time.time() - start_ms) * 1000

    return ResumeResponse(
        user_id            = resolved_uid,
        thread_id          = raw_tid,
        plan_status        = new_plan_status,
        is_awaiting_human  = still_waiting,
        answer             = answer,
        pending_gate_items = gate_items,
        duration_ms        = round(duration, 1),
    )


@app.post(
    "/session/{user_id}/{thread_id}/resume/stream",
    summary="提交人工决策，以流式（SSE）方式恢复执行",
    description=(
        "语义跟 /resume 完全一样（提交 decisions，从 human_review_gate 冻结点精确恢复），"
        "唯一区别：如果恢复后图一路跑到了 final_answer_node，会用 SSE 逐 token 推送"
        "最终回答，而不是等生成完毕再一次性返回整段 answer。\n\n"
        "如果这批决策提交后又产生了新的 pending_gate_items（比如刚重试的任务又失败了），"
        "会收到 `event: interrupted`，跟 /chat/stream 遇到中断时的事件格式完全一致，"
        "此时需要再调一次本接口处理下一批，直到收到 `[DONE:...]`。\n\n"
        "⚠️ 前端注意：这是 POST 接口，浏览器原生 `EventSource` 不支持带 body 的 POST，"
        "请用 `fetch()` + 手动读 `ReadableStream` 解析 SSE，不要用 `new EventSource(...)`。\n\n"
        "事件格式（跟 /chat/stream 一致）：\n"
        "- `data: <token>` — 回答文本 token\n"
        "- `event: interrupted` + `data: {plan_status, pending_gate_items}` — 又冻结了\n"
        "- `data: [WAITING_HUMAN:<thread_id>]` — 冻结收尾标记\n"
        "- `data: [DONE:<thread_id>]` — 本轮真正跑完\n"
        "- `event: rejected` — 该会话当前根本不处于等待人工态，无需调用本接口\n"
        "- `data: [ERROR] ...` — 执行出错或等待超时"
    ),
)
async def resume_task_plan_stream(
    user_id: str,
    thread_id: str,
    req: ResumeRequest,
) -> StreamingResponse:
    resolved_uid = _normalize_user_id(user_id)
    internal_tid = _make_internal_thread_id(resolved_uid, thread_id)
    _, raw_tid   = _split_internal_thread_id(internal_tid)
    request_id   = f"{internal_tid}_{uuid.uuid4().hex[:8]}"
    # ★ 跟 chat_stream 一样：config 里塞 _stream_request_id，
    #   final_answer_node 才会走 llm.astream() 那条流式分支
    #   （否则就是我们上面确认过的、tid="" 触发的非流式兜底分支）
    config = {
        "configurable": {
            "thread_id":          internal_tid,
            "user_id":            resolved_uid,
            "_stream_request_id": request_id,
        }
    }
    decisions_payload = [d.model_dump() for d in req.decisions]

    async def generate() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue()
        agent_module._stream_queues[request_id] = q

        # 跟 /resume 用同一把锁：同一个 thread_id 排队，
        # 避免和另一个并发请求互相踩同一份 checkpoint
        lock = _get_thread_lock(internal_tid)
        lock_acquired = False

        try:
            await lock.acquire()
            lock_acquired = True

            awaiting, _gate_items, plan_status = await _check_awaiting_human(config)
            if not awaiting:
                payload = json.dumps({
                    "message": (
                        f"当前会话不处于等待人工处理状态（plan_status={plan_status}），"
                        "无需调用 /resume/stream。"
                    ),
                    "plan_status": plan_status,
                }, ensure_ascii=False)
                yield f"event: rejected\ndata: {payload}\n\n"
                return

            # ★ HITL 核心：跟非流式版 /resume 完全一样的 Command(resume=...)，
            #   只是这里用 create_task 让它在后台跑，好让下面能同时消费 queue
            invoke_task = asyncio.create_task(
                agent_module.graph.ainvoke(
                    Command(resume=decisions_payload),
                    config=config,
                )
            )

            async for chunk in _stream_graph_run(invoke_task, q, raw_tid):
                yield chunk
        finally:
            agent_module._stream_queues.pop(request_id, None)  # 兜底清理
            if lock_acquired:
                lock.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "X-User-Id":         resolved_uid,
            "X-Thread-Id":       raw_tid,
        },
    )


@app.post(
    "/session/{user_id}/{thread_id}/abort",
    response_model=ResumeResponse,
    summary="放弃当前中断，直接终止整个任务计划",
    description="等价于对 /resume 提交一个 action=abort_all 的决策，不需要知道具体待办事项的 task_id。",
)
async def abort_task_plan(user_id: str, thread_id: str) -> ResumeResponse:
    resolved_uid = _normalize_user_id(user_id)
    internal_tid = _make_internal_thread_id(resolved_uid, thread_id)
    _, raw_tid   = _split_internal_thread_id(internal_tid)
    config       = {"configurable": {"thread_id": internal_tid, "user_id": resolved_uid}}

    start_ms = time.time()
    async with _get_thread_lock(internal_tid):
        awaiting, gate_items, plan_status = await _check_awaiting_human(config)
        if not awaiting:
            raise HTTPException(
                status_code=409,
                detail=f"当前会话不处于等待人工处理状态（plan_status={plan_status}），无需终止。",
            )

        # abort_all 是全局动作，task_id 随便取第一个待办事项的即可（node 内部不校验存在性）
        placeholder_tid = gate_items[0]["task_id"] if gate_items else 0
        try:
            result = await agent_module.graph.ainvoke(
                Command(resume=[{"task_id": placeholder_tid, "action": "abort_all"}]),
                config=config,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"终止失败：{exc}") from exc

        answer   = agent_module._get_message_content(result["messages"][-1])
        duration = (time.time() - start_ms) * 1000

    return ResumeResponse(
        user_id            = resolved_uid,
        thread_id          = raw_tid,
        plan_status        = result.get("plan_status", "aborted"),
        is_awaiting_human  = False,
        answer             = answer,
        pending_gate_items = [],
        duration_ms        = round(duration, 1),
    )


@app.post(
    "/session/{user_id}/{thread_id}/abort/stream",
    summary="放弃当前中断，以流式（SSE）方式终止整个任务计划",
    description=(
        "语义跟 /abort 完全一样（等价于提交一个 action=abort_all 的决策，不需要知道"
        "具体待办事项的 task_id），唯一区别：abort_all 会让图直接路由到 "
        "final_answer_node（gate_route 里 plan_status==\"aborted\" 时不会再回 "
        "human_review_gate），这里改用 SSE 逐 token 推送最终的收尾回答，而不是"
        "等生成完毕再一次性返回整段 answer。\n\n"
        "⚠️ 前端注意：这是 POST 接口，浏览器原生 `EventSource` 不支持带 body 的 POST，"
        "请用 `fetch()` + 手动读 `ReadableStream` 解析 SSE，不要用 `new EventSource(...)`。\n\n"
        "事件格式（跟 /chat/stream、/resume/stream 一致）：\n"
        "- `data: <token>` — 收尾回答文本 token\n"
        "- `data: [DONE:<thread_id>]` — 终止流程真正跑完\n"
        "- `event: rejected` — 该会话当前根本不处于等待人工态，无需调用本接口\n"
        "- `data: [ERROR] ...` — 执行出错或等待超时\n\n"
        "（理论上不会收到 event: interrupted——abort_all 会让 plan_status 直接"
        "变成 aborted，gate_route 只会路由到 final_answer，不会再产生新一批 "
        "pending_gate_items；这里复用 _stream_graph_run 只是为了跟 /resume/stream "
        "共用同一套消费逻辑，多一层保险。）"
    ),
)
async def abort_task_plan_stream(user_id: str, thread_id: str) -> StreamingResponse:
    resolved_uid = _normalize_user_id(user_id)
    internal_tid = _make_internal_thread_id(resolved_uid, thread_id)
    _, raw_tid   = _split_internal_thread_id(internal_tid)
    request_id   = f"{internal_tid}_{uuid.uuid4().hex[:8]}"
    # 跟 chat_stream / resume_stream 一样：config 里塞 _stream_request_id，
    # final_answer_node 才会走 llm.astream() 那条流式分支
    config = {
        "configurable": {
            "thread_id":          internal_tid,
            "user_id":            resolved_uid,
            "_stream_request_id": request_id,
        }
    }

    async def generate() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue()
        agent_module._stream_queues[request_id] = q

        # 跟 /abort 用同一把锁：同一个 thread_id 排队，
        # 避免和另一个并发请求（比如同时在提交决策）互相踩同一份 checkpoint
        lock = _get_thread_lock(internal_tid)
        lock_acquired = False

        try:
            await lock.acquire()
            lock_acquired = True

            awaiting, gate_items, plan_status = await _check_awaiting_human(config)
            if not awaiting:
                payload = json.dumps({
                    "message": (
                        f"当前会话不处于等待人工处理状态（plan_status={plan_status}），"
                        "无需终止。"
                    ),
                    "plan_status": plan_status,
                }, ensure_ascii=False)
                yield f"event: rejected\ndata: {payload}\n\n"
                return

            # abort_all 是全局动作，task_id 随便取第一个待办事项的即可（node 内部不校验存在性）
            placeholder_tid = gate_items[0]["task_id"] if gate_items else 0
            invoke_task = asyncio.create_task(
                agent_module.graph.ainvoke(
                    Command(resume=[{"task_id": placeholder_tid, "action": "abort_all"}]),
                    config=config,
                )
            )

            async for chunk in _stream_graph_run(invoke_task, q, raw_tid):
                yield chunk
        finally:
            agent_module._stream_queues.pop(request_id, None)  # 兜底清理
            if lock_acquired:
                lock.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "X-User-Id":         resolved_uid,
            "X-Thread-Id":       raw_tid,
        },
    )


# ══════════════════════════════════════════════════════
# 6. 健康检查：GET /health
# ══════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    agents     = agent_module._registry.agents
    agent_desc_block     = agent_module._registry.agent_desc_block
    tool_count = len(agent_module._tools)
    status     = "ok" if tool_count >= 4 else ("initializing" if tool_count == 0 else "degraded")

    return HealthResponse(
        status         = status,
        tool_count     = tool_count,
        agents         = agents,
        agent_desc_block =agent_desc_block ,
        uptime_seconds = round(time.time() - _start_time, 1),
        checkpoint_db  = agent_module._CHECKPOINT_DB,
    )


# ══════════════════════════════════════════════════════
# 7. 会话管理
# ══════════════════════════════════════════════════════

@app.post("/session/new", response_model=SessionResponse, summary="新建会话")
async def new_session(
    user_id: str = Query("default", description="用户 ID，缺省 'default'（单用户模式）"),
) -> SessionResponse:
    uid = _normalize_user_id(user_id)
    tid = f"user_{uuid.uuid4().hex[:12]}"  # 这里的 "user_" 只是历史命名习惯，不代表用户身份
    return SessionResponse(user_id=uid, thread_id=tid, created_at=time.time())


@app.get(
    "/sessions/{user_id}",
    response_model=SessionListResponse,
    summary="列出某用户的所有历史会话",
    description=(
        "几天后用户回来时，前端用这个接口拿到该用户名下所有 thread_id，"
        "展示成一个会话列表供用户选择继续。\n\n"
        "单用户模式下传 `user_id=default` 即可拿到旧版所有会话（升级前数据也会被识别为 default）。\n\n"
        "直接查询 AsyncSqliteSaver 底层 SQLite 的 checkpoints 表，按 user_id 前缀过滤、按更新时间倒序。"
    ),
)
async def list_sessions(user_id: str) -> SessionListResponse:
    uid = _normalize_user_id(user_id)
    cp  = agent_module._checkpointer

    if cp is None or not hasattr(cp, "conn"):
        raise HTTPException(status_code=503, detail="SQLite checkpointer 尚未就绪")

    try:
        # checkpoints 表里同一个 thread_id 有多行（每一步一条 checkpoint），
        # 这里按 thread_id 分组，取 ts 最大的一条即为"最新状态"。
        #
        # default 用户：thread_id 没有前缀（格式 "user_xxxxxxxx"），
        #   反向过滤——查所有不含 __ 的行，这样同时兼容升级前的旧数据。
        # 非-default 用户：thread_id 带 "{uid}__" 前缀，正向 LIKE 匹配；
        #   为防止 user_id 里含 % 或 _ 这类 SQL LIKE 通配符引发误匹配，转义后再拼。
        if uid == "default":
            cursor = await cp.conn.execute(
                "SELECT thread_id, MAX(rowid) as last_rowid "
                "FROM checkpoints "
                "WHERE thread_id NOT LIKE '%__%' ESCAPE '\\' "
                "GROUP BY thread_id "
                "ORDER BY last_rowid DESC",
            )
        else:
            escaped_sep = _SEP.replace("_", r"\_")  # 结果为 "\\_\\_"（字符串字面量）

            like_prefix = uid.replace("%", r"\%").replace("_", r"\_") + escaped_sep

            cursor = await cp.conn.execute(
                "SELECT thread_id, MAX(rowid) as last_rowid "
                "FROM checkpoints "
                "WHERE thread_id LIKE ? ESCAPE '\\' "
                "GROUP BY thread_id "
                "ORDER BY last_rowid DESC",
                (f"{like_prefix}%",),
              )
        rows = await cursor.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询会话列表失败：{exc}") from exc

    sessions: list[SessionInfo] = []
    for row in rows:
        internal_tid = row[0]
        last_ts_raw  = row[1]
        _, raw_tid   = _split_internal_thread_id(internal_tid)

        config    = {"configurable": {"thread_id": internal_tid}}
        msg_count = await _get_checkpoint_message_count(config)

        last_message = ""
        try:
            saved = await cp.aget_tuple(config)
            if saved is not None and saved.checkpoint:
                msgs = saved.checkpoint.get("channel_values", {}).get("messages", [])
                if msgs:
                    last_message = agent_module._get_message_content(msgs[-1])
        except Exception:
            pass

        # rowid 是插入顺序，不是时间戳，无法转成 updated_at；留 None，前端按需处理。
        updated_at: float | None = None

        sessions.append(SessionInfo(
            thread_id     = raw_tid,
            last_message  = (last_message[:200] if last_message else ""),  # 截断预览，避免超长摘要占满列表
            message_count = msg_count,
            updated_at    = updated_at,
        ))

    return SessionListResponse(user_id=uid, sessions=sessions)


async def _delete_session_internal(internal_tid: str) -> dict:
    """实际执行删除的共用逻辑，被两个 DELETE 路由复用。"""
    try:
        cp = agent_module._checkpointer

        # langgraph-checkpoint-sqlite >= 2.0 提供 adelete_thread
        if hasattr(cp, "adelete_thread"):
            await cp.adelete_thread(internal_tid)
            return {"success": True, "method": "adelete_thread"}

        # 兜底：直接执行 SQL
        if hasattr(cp, "conn"):
            await cp.conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (internal_tid,)
            )
            await cp.conn.execute(
                "DELETE FROM writes WHERE thread_id = ?", (internal_tid,)
            )
            await cp.conn.commit()
            return {"success": True, "method": "sql_delete"}

        return {"success": False, "warning": "不支持的 checkpointer 类型"}

    except Exception as exc:
        return {"success": False, "error": str(exc)}


@app.delete(
    "/session/{thread_id}",
    summary="清除会话历史（单用户兼容写法）",
    description=(
        "删除指定 thread_id 的 checkpoint。\n\n"
        "**兼容说明**：不带 user_id，等价于操作 user_id='default' 名下的会话。\n"
        "如果你已经在用多用户模式，请改用 `DELETE /session/{user_id}/{thread_id}`，"
        "否则可能删错（比如真的有用户名叫 default 的情况）。"
    ),
)
async def clear_session(thread_id: str) -> dict:
    internal_tid = _make_internal_thread_id("default", thread_id)
    result = await _delete_session_internal(internal_tid)
    return {**result, "user_id": "default", "thread_id": thread_id}


@app.delete(
    "/session/{user_id}/{thread_id}",
    summary="清除指定用户名下的会话历史",
    description="多用户模式下推荐用法：显式指定 user_id，避免误删其他用户的同名 thread_id 会话。",
)
async def clear_session_for_user(user_id: str, thread_id: str) -> dict:
    uid = _normalize_user_id(user_id)
    internal_tid = _make_internal_thread_id(uid, thread_id)
    result = await _delete_session_internal(internal_tid)
    return {**result, "user_id": uid, "thread_id": thread_id}


# ══════════════════════════════════════════════════════
# 8. 全局 Store 记忆管理（AsyncSqliteStore 版本）
# ══════════════════════════════════════════════════════

@app.get(
    "/memory",
    response_model=MemoryListResponse,
    summary="列出全局记忆",
    description=(
        "返回 AsyncSqliteStore 里所有记忆，按命名空间拆成两块：\n\n"
        "- `system`：全局共享，所有用户所有会话都能读到。\n"
        "- `user`：按 user_id 分组，`{user_id: {key: value}}`，同一用户的所有会话共享，"
        "不同用户互相隔离。\n\n"
        "数据持久化在 SQLite，重启后仍然保留。"
    ),
)
async def list_memory() -> MemoryListResponse:
    system_items = await agent_module.store_list(("system",))
    user_grouped_raw = await agent_module.store_list_by_namespace(("user",))
    # user_grouped_raw 的 key 是完整 namespace 元组 ("user", "<user_id>")，
    # 拍平成 {user_id: {key: value}} 给前端用
    user_items = {ns[1]: kv for ns, kv in user_grouped_raw.items() if len(ns) >= 2}
    return MemoryListResponse(system=system_items, user=user_items)


@app.post("/memory", summary="写入全局记忆")
async def put_memory(item: MemoryItem) -> dict:
    if item.namespace == "user":
        uid = _normalize_user_id(item.user_id or "")
        if not item.user_id or not item.user_id.strip():
            raise HTTPException(status_code=400, detail="写入 user 命名空间必须提供 user_id")
        namespace = ("user", uid)
    else:
        uid = None
        namespace = ("system",)

    await agent_module.store_put(item.key, item.value, namespace=namespace)
    return {"success": True, "key": item.key, "value": item.value, "namespace": item.namespace, "user_id": uid}


@app.delete("/memory/{key}", summary="删除全局记忆")
async def delete_memory(
    key:       str,
    namespace: Literal["system", "user"] = Query("system", description="要删除的命名空间"),
    user_id:   str | None                = Query(None, description="namespace='user' 时必填"),
) -> dict:
    if namespace == "user":
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="删除 user 命名空间记忆必须提供 user_id")
        ns = ("user", _normalize_user_id(user_id))
    else:
        ns = ("system",)

    success = await agent_module.store_delete(key, namespace=ns)
    if not success:
        raise HTTPException(status_code=404, detail=f"记忆 '{key}' 不存在或删除失败（namespace={namespace}）")
    return {"success": True, "key": key, "namespace": namespace, "user_id": user_id}


# ══════════════════════════════════════════════════════
# 9. 数据库清理：DELETE /db/cleanup
# ══════════════════════════════════════════════════════
#
# 保留最近 keep_threads 个会话（按 MAX(rowid) 倒序），其余全部删除。
#
# 为什么用 rowid 而不是时间戳：
#   langgraph-checkpoint-sqlite 的 checkpoints 表没有独立时间列，
#   checkpoint_id 虽然是类 UUIDv7 格式，但实测前 48 bit 不是标准
#   unix_ms，无法可靠解析。rowid 是 SQLite 自增主键，天然单调递增，
#   MAX(rowid) 最大的 thread_id 就是最近有活动的会话，排序语义正确。
#
# 实际表名（inspect 确认）：
#   checkpoints  — 主表
#   writes       — 对应旧文档里的 checkpoint_writes
#
# 参数：
#   keep_threads  int   保留最近几个会话（默认 150）
#   dry_run       bool  True=只统计不删除，False=真正执行删除（默认 False）
#
# 响应字段：
#   keep_threads       实际使用的保留数量
#   total_threads      当前数据库里的会话总数
#   dry_run            是否为预演模式
#   threads_to_delete  将被/已被删除的 thread_id 列表（raw，不含 user_id 前缀）
#   threads_kept       保留的会话数量
#   rows_deleted       checkpoints 表删除行数（dry_run 时为 0）
#   writes_deleted     writes 表删除行数（dry_run 时为 0）

class CleanupResponse(BaseModel):
    keep_threads:       int       = Field(..., description="保留最近几个会话")
    total_threads:      int       = Field(..., description="清理前数据库里的会话总数")
    dry_run:            bool      = Field(..., description="True=预演（只统计不删），False=已真正删除")
    threads_to_delete:  list[str] = Field(..., description="将被/已被删除的 thread_id 列表（raw，不含 user_id 前缀）")
    threads_kept:       int       = Field(..., description="保留的会话数量")
    rows_deleted:       int       = Field(..., description="checkpoints 表删除行数（dry_run 时为 0）")
    writes_deleted:     int       = Field(..., description="writes 表删除行数（dry_run 时为 0）")


@app.delete(
    "/db/cleanup",
    response_model=CleanupResponse,
    summary="清理旧 Checkpoint 数据",
    description=(
        "按活动顺序保留最近 `keep_threads` 个会话，其余全部删除。\n\n"
        "**排序依据**：`checkpoints` 表的 `rowid`（SQLite 自增，越大越新）。"
        "取每个 `thread_id` 的 `MAX(rowid)` 代表该会话最后一次活动的相对顺序。\n\n"
        "**建议流程**：先用 `dry_run=true` 预览将被删除的会话列表，确认无误后再用 "
        "`dry_run=false` 执行真正删除。\n\n"
        "⚠️ **不可逆操作**：`dry_run=false` 执行后数据无法恢复，请确认备份或先预演。"
    ),
)
async def cleanup_checkpoints(
    keep_threads: int  = Query(150,   ge=0, description="保留最近几个会话（0 = 删除所有会话）"),
    dry_run:      bool = Query(False,       description="True=预演模式（只统计，不删除）"),
) -> CleanupResponse:
    cp = agent_module._checkpointer

    if cp is None or not hasattr(cp, "conn"):
        raise HTTPException(status_code=503, detail="SQLite checkpointer 尚未就绪")

    try:
        # ── Step 1：按 MAX(rowid) 倒序列出所有 thread_id ──────────────
        # rowid 越大 = 越晚插入 = 越近的活动，前 keep_threads 个是要保留的
        cursor = await cp.conn.execute(
            "SELECT thread_id FROM checkpoints "
            "GROUP BY thread_id "
            "ORDER BY MAX(rowid) DESC"
        )
        all_threads: list[str] = [row[0] for row in await cursor.fetchall()]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询 checkpoints 失败：{exc}") from exc

    total        = len(all_threads)
    keep_set     = set(all_threads[:keep_threads])        # 保留：rowid 最大的前 N 个
    to_delete    = [t for t in all_threads[keep_threads:]]  # 删除：其余
    kept_count   = len(keep_set)

    # 对外展示 raw_thread_id（去掉 user_id 前缀），方便人工核对
    raw_to_delete = [_split_internal_thread_id(tid)[1] for tid in to_delete]

    # ── Step 2：dry_run 分支 ───────────────────────────────────────
    if dry_run or not to_delete:
        return CleanupResponse(
            keep_threads      = keep_threads,
            total_threads     = total,
            dry_run           = dry_run,
            threads_to_delete = raw_to_delete,
            threads_kept      = kept_count,
            rows_deleted      = 0,
            writes_deleted    = 0,
        )

    # ── Step 3：真正删除 ───────────────────────────────────────────
    try:
        placeholders = ",".join("?" * len(to_delete))

        cur1 = await cp.conn.execute(
            f"DELETE FROM checkpoints WHERE thread_id IN ({placeholders})",
            to_delete,
        )
        rows_deleted = cur1.rowcount

        cur2 = await cp.conn.execute(
            f"DELETE FROM writes WHERE thread_id IN ({placeholders})",
            to_delete,
        )
        writes_deleted = cur2.rowcount

        await cp.conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除操作失败：{exc}") from exc

    return CleanupResponse(
        keep_threads      = keep_threads,
        total_threads     = total,
        dry_run           = False,
        threads_to_delete = raw_to_delete,
        threads_kept      = kept_count,
        rows_deleted      = rows_deleted,
        writes_deleted    = writes_deleted,
    )


# ══════════════════════════════════════════════════════
# 10. 根路径
# ══════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service":       "LangGraph Parallel Agent API",
        "version":       "2.2.0",
        "persistence":   "AsyncSqliteSaver + AsyncSqliteStore",
        "checkpoint_db": agent_module._CHECKPOINT_DB,
        "multi_user":    "user_id 缺省 'default'，传入真实 user_id 即可隔离（详见 /sessions/{user_id}）",
        "docs":          "/docs",
        "health":        "/health",
    }
    
    # uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1