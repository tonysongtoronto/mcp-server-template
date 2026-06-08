"""
api.py  ——  LangGraph Parallel Agent 的 FastAPI 后台服务
            持久化版本：AsyncSqliteSaver（Checkpoint）+ AsyncSqliteStore（全局记忆）
            v2.1.0 修复：UTF-8 强制编码（解决 Windows 乱码）+ CheckpointTuple 正确读取

【升级内容对比】
  MemorySaver（旧）     → 进程内字典，重启丢数据
  AsyncSqliteSaver（新）→ 写入 checkpoints.db，重启保留所有对话历史 ✅

  InMemoryStore（旧）   → 进程内字典，重启丢数据
  AsyncSqliteStore（新）→ 同一个 checkpoints.db，重启保留全局记忆 ✅

【接口一览】
  POST   /chat              → 普通对话（等待完整答案后返回）
  GET    /chat/stream       → 流式对话（SSE，逐 token 推送）
  GET    /health            → 健康检查（服务状态 + MCP 工具数量）
  POST   /session/new       → 新建会话（返回新 thread_id）
  DELETE /session/{tid}     → 清除某个会话的 checkpoint 历史
  GET    /memory            → 列出全局 Store 记忆
  POST   /memory            → 写入一条全局 Store 记忆
  DELETE /memory/{key}      → 删除一条全局 Store 记忆

【启动方式】
  pip install langgraph-checkpoint-sqlite
  uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

  注意：必须 workers=1（SQLite 不支持多进程并发写）。
        如需横向扩展，把两个 SQLite 后端换成 PostgresSaver / AsyncPostgresStore。

【数据库文件位置】
  默认：项目根目录 / checkpoints.db
  自定义：设置环境变量 CHECKPOINT_DB=/path/to/your.db
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import langgraph_parallel_agent as agent_module

# ══════════════════════════════════════════════════════
# 1. Pydantic 请求 / 响应模型
# ══════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question:  str = Field(..., description="用户输入的问题或指令")
    thread_id: str = Field("",  description="会话 ID，留空则自动生成（实现多用户隔离）")

class ChatResponse(BaseModel):
    answer:        str   = Field(..., description="AI 最终回答")
    thread_id:     str   = Field(..., description="本次对话所属的会话 ID")
    message_count: int   = Field(..., description="该会话累计消息条数")
    duration_ms:   float = Field(..., description="本次请求耗时（毫秒）")

class SessionResponse(BaseModel):
    thread_id:  str   = Field(..., description="新建会话的 ID")
    created_at: float = Field(..., description="创建时间戳")

class MemoryItem(BaseModel):
    key:   str = Field(..., description="记忆键名")
    value: str = Field(..., description="记忆内容（字符串）")

class MemoryListResponse(BaseModel):
    items: dict = Field(..., description="当前所有全局记忆 {key: value}")

class HealthResponse(BaseModel):
    status:         str       = Field(..., description="ok / degraded / initializing")
    tool_count:     int       = Field(..., description="已加载的 MCP 工具数量")
    agents:         list[str] = Field(..., description="已注册的 Agent 列表")
    uptime_seconds: float     = Field(..., description="服务运行时间（秒）")
    checkpoint_db:  str       = Field(..., description="SQLite 数据库文件路径")

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
        "**多用户隔离**：不同用户使用不同 `thread_id`，历史互不干扰。"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 生产环境请改为实际域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Thread-Id"],   # ← 让浏览器能读到 X-Thread-Id header
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

def _resolve_thread_id(thread_id: str) -> str:
    return thread_id.strip() if thread_id.strip() else f"user_{uuid.uuid4().hex[:8]}"


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
        "**多用户隔离**：不同用户使用不同 `thread_id`，历史互不干扰。"
    ),
)
async def chat(req: ChatRequest) -> ChatResponse:
    if not agent_module._registry.agents:
        raise HTTPException(
            status_code=503,
            detail="MCP 服务尚未就绪，请稍后重试（检查 /health 接口）",
        )

    thread_id = _resolve_thread_id(req.thread_id)
    config    = {"configurable": {"thread_id": thread_id}}

    start_ms = time.time()
    try:
        result = await agent_module.graph.ainvoke(
            {"messages": [HumanMessage(content=req.question)]},
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{exc}") from exc

    answer    = agent_module._get_message_content(result["messages"][-1])
    duration  = (time.time() - start_ms) * 1000
    msg_count = await _get_checkpoint_message_count(config)

    return ChatResponse(
        answer        = answer,
        thread_id     = thread_id,
        message_count = msg_count,
        duration_ms   = round(duration, 1),
    )


# ══════════════════════════════════════════════════════
# 5. 流式接口：GET /chat/stream（SSE 逐 token 推送）
# ══════════════════════════════════════════════════════

@app.get(
    "/chat/stream",
    summary="发送消息（流式 SSE）",
    description=(
        "通过 Server-Sent Events（SSE）流式推送 Agent 回答。\n\n"
        "**前端接入示例**：\n"
        "```javascript\n"
        "const es = new EventSource('/chat/stream?question=你好&thread_id=abc');\n"
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
    thread_id: str = Query("",  description="会话 ID，留空自动生成"),
) -> StreamingResponse:

    if not agent_module._registry.agents:
        async def _err():
            yield "data: [ERROR] MCP 服务尚未就绪，请稍后重试\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    resolved_tid = _resolve_thread_id(thread_id)
    # ★ 修复1：request_id 在 config 之前生成，才能放进 configurable
    request_id   = f"{resolved_tid}_{uuid.uuid4().hex[:8]}"  # 每次请求唯一，避免并发竞争
    # ★ 修复2：_stream_request_id 走 configurable（侧信道），不污染 state
    #   final_answer_node 从 config["configurable"]["_stream_request_id"] 读 queue key
    #   state 只含 messages → LangSmith Input/Output 显示正常问题内容
    config       = {"configurable": {"thread_id": resolved_tid, "_stream_request_id": request_id}}

    async def generate() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue()
        agent_module._stream_queues[request_id] = q

        invoke_task = asyncio.create_task(
            agent_module.graph.ainvoke(
                # ★ 修复3：state 只传 messages，去掉 _thread_id
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
        )

        try:
            while True:
                try:
                    token = await asyncio.wait_for(q.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    yield "data: [ERROR] 等待超时（120s）\n\n"
                    invoke_task.cancel()
                    break

                if token is None:
                    yield f"data: [DONE:{resolved_tid}]\n\n"
                    break

                safe_token = str(token).replace("\n", " ")
                yield f"data: {safe_token}\n\n"

            try:
                await invoke_task
            except Exception as exc:
                yield f"data: [ERROR] {exc}\n\n"
        finally:
            agent_module._stream_queues.pop(request_id, None)  # 兜底清理

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id":       resolved_tid,
        },
    )


# ══════════════════════════════════════════════════════
# 6. 健康检查：GET /health
# ══════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    agents     = agent_module._registry.agents
    tool_count = len(agent_module._tools)
    status     = "ok" if tool_count >= 4 else ("initializing" if tool_count == 0 else "degraded")

    return HealthResponse(
        status         = status,
        tool_count     = tool_count,
        agents         = agents,
        uptime_seconds = round(time.time() - _start_time, 1),
        checkpoint_db  = agent_module._CHECKPOINT_DB,
    )


# ══════════════════════════════════════════════════════
# 7. 会话管理
# ══════════════════════════════════════════════════════

@app.post("/session/new", response_model=SessionResponse, summary="新建会话")
async def new_session() -> SessionResponse:
    tid = f"user_{uuid.uuid4().hex[:12]}"
    return SessionResponse(thread_id=tid, created_at=time.time())


@app.delete(
    "/session/{thread_id}",
    summary="清除会话历史",
    description="删除指定 thread_id 的 checkpoint（清空该用户的聊天记录）。数据从 SQLite 永久删除。",
)
async def clear_session(thread_id: str) -> dict:
    try:
        cp = agent_module._checkpointer

        # langgraph-checkpoint-sqlite >= 2.0 提供 adelete_thread
        if hasattr(cp, "adelete_thread"):
            await cp.adelete_thread(thread_id)
            return {"success": True, "thread_id": thread_id, "method": "adelete_thread"}

        # 兜底：直接执行 SQL
        if hasattr(cp, "conn"):
            await cp.conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
            )
            await cp.conn.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,)
            )
            await cp.conn.commit()
            return {"success": True, "thread_id": thread_id, "method": "sql_delete"}

        return {"success": False, "warning": "不支持的 checkpointer 类型"}

    except Exception as exc:
        return {"success": False, "thread_id": thread_id, "error": str(exc)}


# ══════════════════════════════════════════════════════
# 8. 全局 Store 记忆管理（AsyncSqliteStore 版本）
# ══════════════════════════════════════════════════════

@app.get(
    "/memory",
    response_model=MemoryListResponse,
    summary="列出全局记忆",
    description="返回 AsyncSqliteStore 里所有记忆。数据持久化在 SQLite，重启后仍然保留。",
)
async def list_memory() -> MemoryListResponse:
    items = await agent_module.store_list()
    return MemoryListResponse(items=items)


@app.post("/memory", summary="写入全局记忆")
async def put_memory(item: MemoryItem) -> dict:
    await agent_module.store_put(item.key, item.value)
    return {"success": True, "key": item.key, "value": item.value}


@app.delete("/memory/{key}", summary="删除全局记忆")
async def delete_memory(key: str) -> dict:
    success = await agent_module.store_delete(key)
    if not success:
        raise HTTPException(status_code=404, detail=f"记忆 '{key}' 不存在或删除失败")
    return {"success": True, "key": key}


# ══════════════════════════════════════════════════════
# 9. 根路径
# ══════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service":       "LangGraph Parallel Agent API",
        "version":       "2.1.0",
        "persistence":   "AsyncSqliteSaver + AsyncSqliteStore",
        "checkpoint_db": agent_module._CHECKPOINT_DB,
        "docs":          "/docs",
        "health":        "/health",
    }
    
    # uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1