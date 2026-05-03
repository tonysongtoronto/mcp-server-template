# src/webapp.py
#
# langgraph dev 的 lifespan 入口。
# langgraph.json 配置：
#   "http": { "app": "./src/webapp.py:app" }
#
# ★ 子进程清单（lifespan 负责拉起 + 关闭）：
#   1. mcp_server_template/server.py  → SSE @ 8001  data / http 工具
#   2. mcp-proxy (filesystem)         → SSE @ 8002  文件系统工具
#   3. mcp_db_server/server.py        → SSE @ 8003  数据库工具
#   4. mcp-proxy (math-mcp)           → SSE @ 8004  数学工具
#
# ★ 阶段二新增：
#   - AsyncSqliteSaver 替换 MemorySaver，对话历史持久化到 SQLite
#   - POST /chat/stream  SSE 流式端点，逐 token 推送 AI 回复
#   - thread_id 从请求头 X-Thread-Id 读取，缺省自动生成 UUID

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
_SERVER_PORT     = int(os.getenv("MCP_SERVER_PORT",     "8001"))
_FS_PROXY_PORT   = int(os.getenv("MCP_FS_PROXY_PORT",   "8002"))
_DB_SERVER_PORT  = int(os.getenv("MCP_DB_SERVER_PORT",  "8003"))
_MATH_PROXY_PORT = int(os.getenv("MCP_MATH_PROXY_PORT", "8004"))

_MCP_FS_ENV = os.getenv("MCP_FS_BASE_DIR", "")
if _MCP_FS_ENV:
    _FS_BASE_DIR = Path(_MCP_FS_ENV)
else:
    _FS_BASE_DIR = Path(__file__).parent.parent / "File_Agent"

_SERVER_PY    = Path(__file__).parent / "mcp_server_template" / "server.py"
_DB_SERVER_PY = Path(__file__).parent / "mcp_db_server" / "server.py"
_MATH_MCP_JS  = Path(__file__).parent / "math-mcp" / "build" / "index.js"

_NPX  = "npx.cmd" if sys.platform == "win32" else "npx"
_NODE = "node.exe" if sys.platform == "win32" else "node"


# ──────────────────────────────────────────
# 工具函数（全部 async）
# ──────────────────────────────────────────

def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


async def _kill_port(port: int) -> None:
    print(f"  ⚠️  端口 {port} 被占用，尝试强制释放...", file=sys.stderr)
    if sys.platform == "win32":
        await asyncio.to_thread(
            os.system,
            f'FOR /F "tokens=5" %P IN '
            f'(\'netstat -ano ^| findstr ":{port} "\') DO taskkill /F /PID %P >nul 2>&1',
        )
    else:
        ret = await asyncio.to_thread(os.system, f"fuser -k {port}/tcp 2>/dev/null")
        if ret != 0:
            await asyncio.to_thread(
                os.system,
                f"lsof -ti tcp:{port} 2>/dev/null | xargs kill -9 2>/dev/null || true",
            )
    await asyncio.sleep(0.8)


async def _wait_for_sse(url: str, timeout: float = 30.0, interval: float = 1.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code in (200, 405):
                        return True
        except Exception:
            pass
        await asyncio.sleep(interval)
    return False


async def _launch_subprocess(
    tag: str,
    cmd: list,
    env: dict | None = None,
    port: int | None = None,
) -> subprocess.Popen | None:
    if port and _port_in_use(port):
        await _kill_port(port)
        if _port_in_use(port):
            print(f"  ❌ [{tag}] 端口 {port} 无法释放，跳过启动", file=sys.stderr)
            return None

    merged_env = {**os.environ, **(env or {})}
    print(f"  🚀 [{tag}] 启动：{' '.join(str(c) for c in cmd)}", file=sys.stderr)

    try:
        kwargs: dict = dict(
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=merged_env,
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = await asyncio.to_thread(subprocess.Popen, cmd, **kwargs)
        print(f"  ✅ [{tag}] 进程已启动 PID={proc.pid}", file=sys.stderr)
        return proc

    except FileNotFoundError as e:
        print(f"  ❌ [{tag}] 命令未找到：{e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ❌ [{tag}] 启动失败：{type(e).__name__}: {e}", file=sys.stderr)
        return None


async def _terminate_subprocess(tag: str, proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print(f"  🛑 [{tag}] 终止子进程 PID={proc.pid}", file=sys.stderr)
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                await asyncio.to_thread(proc.wait, timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception as e:
        print(f"  ⚠️  [{tag}] 终止时异常：{e}", file=sys.stderr)
        try:
            proc.kill()
        except Exception:
            pass


_subprocesses: list = []


# ──────────────────────────────────────────
# lifespan
# ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["MCP_USE_SSE"] = "1"
    print("\n🟢 [lifespan] 开始启动 MCP 子进程...", file=sys.stderr)

    # ── 持久化：AsyncSqliteSaver（进程重启后对话历史仍保留）────────────────
    #
    # 为什么不用平台自带的持久化？
    #   平台持久化只对 /threads/{id}/runs 端点有效。
    #   我们的 /chat/stream 直接调 graph.ainvoke()，绕过了平台的 thread 管理层。
    #
    # 为什么不用 MemorySaver？
    #   MemorySaver 进程重启后历史清空，生产环境不可用。
    #   AsyncSqliteSaver 把 checkpoint 写入磁盘，重启后历史完整保留。
    #
    # 关键：lifespan 在 _start_mcp_sessions() 之后再用 saver 重建 graph，
    #   覆盖掉 _init_registry 里用 _checkpointer（MemorySaver）建的那个版本，
    #   确保运行时的 graph 用的是 AsyncSqliteSaver。
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import src.langgraph_parallel_agent as agent_module

    _CHECKPOINT_DB = Path(__file__).parent.parent / "data" / "checkpoints.db"
    _CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    print(f"  💾 [Checkpoint] SQLite 路径：{_CHECKPOINT_DB}", file=sys.stderr)

    async with AsyncSqliteSaver.from_conn_string(str(_CHECKPOINT_DB)) as saver:
        app.state.checkpointer = saver
        print("  ✅ [Checkpoint] AsyncSqliteSaver 就绪", file=sys.stderr)

        # ── 1. 启动 server.py（SSE @ 8001）──────────────────────────────
        server_proc = await _launch_subprocess(
            tag="server.py",
            cmd=[sys.executable, "-u", str(_SERVER_PY), "--sse"],
            env={
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PORT": str(_SERVER_PORT),
                "MCP_FS_BASE_DIR": str(_FS_BASE_DIR),
            },
            port=_SERVER_PORT,
        )
        if server_proc:
            _subprocesses.append(server_proc)
            ok = await _wait_for_sse(f"http://127.0.0.1:{_SERVER_PORT}/sse")
            print(f"  {'✅' if ok else '❌'} [server.py] SSE {'就绪' if ok else '超时'}",
                  file=sys.stderr)

        # ── 2. 启动 mcp-proxy（SSE @ 8002：filesystem）──────────────────
        _fs_sub_cmd = f"npx -y @modelcontextprotocol/server-filesystem {_FS_BASE_DIR}"
        fs_proc = await _launch_subprocess(
            tag="mcp-proxy(fs)",
            cmd=[_NPX, "mcp-proxy", "--port", str(_FS_PROXY_PORT),
                 "--server", "sse", "--shell", "--", _fs_sub_cmd],
            port=_FS_PROXY_PORT,
        )
        if fs_proc:
            _subprocesses.append(fs_proc)
            ok = await _wait_for_sse(f"http://127.0.0.1:{_FS_PROXY_PORT}/sse")
            print(f"  {'✅' if ok else '❌'} [mcp-proxy(fs)] SSE {'就绪' if ok else '超时'}",
                  file=sys.stderr)

        # ── 3. 启动 db_server.py（SSE @ 8003）───────────────────────────
        db_proc = await _launch_subprocess(
            tag="db_server.py",
            cmd=[sys.executable, "-u", str(_DB_SERVER_PY), "--sse"],
            env={
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PORT": str(_DB_SERVER_PORT),
            },
            port=_DB_SERVER_PORT,
        )
        if db_proc:
            _subprocesses.append(db_proc)
            ok = await _wait_for_sse(f"http://127.0.0.1:{_DB_SERVER_PORT}/sse")
            print(f"  {'✅' if ok else '❌'} [db_server.py] SSE {'就绪' if ok else '超时'}",
                  file=sys.stderr)

        # ── 4. 启动 mcp-proxy（SSE @ 8004：math-mcp）────────────────────
        _math_sub_cmd = f"{_NODE} {_MATH_MCP_JS}"
        math_proc = await _launch_subprocess(
            tag="mcp-proxy(math)",
            cmd=[_NPX, "mcp-proxy", "--port", str(_MATH_PROXY_PORT),
                 "--server", "sse", "--shell", "--", _math_sub_cmd],
            port=_MATH_PROXY_PORT,
        )
        if math_proc:
            _subprocesses.append(math_proc)
            ok = await _wait_for_sse(f"http://127.0.0.1:{_MATH_PROXY_PORT}/sse")
            print(f"  {'✅' if ok else '❌'} [mcp-proxy(math)] SSE {'就绪' if ok else '超时'}",
                  file=sys.stderr)

        # ── 5. 初始化 MCP sessions ───────────────────────────────────────
        # 注意：_start_mcp_sessions() 内部会调用 _init_registry()，
        # _init_registry 在 webapp 模式下会调用 build_graph()（不带 checkpointer）。
        # 因此 graph 必须在这一步之后重建，才不会被覆盖。
        await agent_module._start_mcp_sessions()

        # ── 6. 用 AsyncSqliteSaver 重建 graph（必须在 _start_mcp_sessions 之后）──
        #
        # _start_mcp_sessions() → _init_registry() 会用模块级 _checkpointer（MemorySaver）
        # 建一个 graph。这里再用 AsyncSqliteSaver 覆盖，确保运行时持久化到 SQLite。
        # 平台在模块加载时只扫描模块底部那行 graph = build_graph()（无 checkpointer），
        # lifespan 启动后平台不再扫描，所以这里覆盖是安全的。
        agent_module.graph = agent_module.build_graph(checkpointer=saver)
        print("  ✅ [Checkpoint] graph 已用 AsyncSqliteSaver 重新编译（持久化到 SQLite）",
              file=sys.stderr)
        print("🟢 [lifespan] 全部就绪，开始服务\n", file=sys.stderr)

        try:
            yield

        finally:
            await agent_module._stop_mcp_sessions()
            print("\n🛑 [lifespan] 关闭子进程...", file=sys.stderr)
            for proc in reversed(_subprocesses):
                await _terminate_subprocess("subprocess", proc)
            _subprocesses.clear()
            print("🛑 [lifespan] 全部已关闭\n", file=sys.stderr)

    # async with AsyncSqliteSaver 在这里自动关闭数据库连接，checkpoints.db 安全落盘


# ──────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────
app = FastAPI(lifespan=lifespan)


# ──────────────────────────────────────────
# POST /chat/stream   ← 阶段二核心端点
# ──────────────────────────────────────────
#
# 请求格式：
#   POST /chat/stream
#   Header:  X-Thread-Id: <会话ID>   （省略则自动生成新 UUID）
#   Body:    {"message": "你好，帮我查一下订单数量"}
#
# 响应格式（text/event-stream，SSE）：
#   data: 好的\n\n
#   data: ，数据库\n\n
#   data: 中共有\n\n
#   ...
#   data: [DONE]\n\n
#   data: {"thread_id": "abc-123"}\n\n
#
# 前端如何使用 thread_id：
#   第一次请求：不带 X-Thread-Id，从最后一帧拿到 thread_id 保存起来
#   后续请求：每次请求都带上这个 thread_id → 多轮对话、历史持久化

@app.post("/chat/stream")
async def chat_stream(request: Request) -> StreamingResponse:
    import src.langgraph_parallel_agent as agent_module
    from langchain_core.messages import HumanMessage

    # ── 解析请求体 ────────────────────────────────────────────────────────
    body = await request.json()
    message: str = body.get("message", "").strip()
    if not message:
        async def _err():
            yield "data: [ERROR] message 字段不能为空\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    # ── 读取或生成 thread_id ──────────────────────────────────────────────
    thread_id: str = request.headers.get("X-Thread-Id", "").strip()
    if not thread_id:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    # ── 创建流式队列，注入到 agent 模块 ──────────────────────────────────
    # final_answer_node 检测到 _stream_queue 不为 None，就用 astream() 逐 token 推送
    queue: asyncio.Queue = asyncio.Queue()
    agent_module._stream_queue = queue

    async def _generate():
        # 在后台启动 graph.ainvoke()，不阻塞 _generate()
        # graph 处理到 final_answer_node 时，会往 queue 里放 token
        invoke_task = asyncio.create_task(
            agent_module.graph.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )
        )

        try:
            # 从队列里读 token，立即推送给浏览器
            while True:
                token = await queue.get()

                if token is None:
                    # 哨兵：final_answer_node 生成完毕
                    break

                # SSE 格式要求：每条消息以 \n\n 结尾
                # token 里的换行符要转义，否则会破坏 SSE 帧格式
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"

            # 等待 invoke 完全结束（含摘要更新等后续操作）
            await invoke_task

        except Exception as e:
            print(f"❌ [/chat/stream] 出错：{e}", file=sys.stderr)
            yield f"data: [ERROR] {str(e)}\n\n"
            if not invoke_task.done():
                invoke_task.cancel()

        finally:
            # 清除队列引用（非常重要！否则下一个请求会用到旧队列）
            agent_module._stream_queue = None

            # 结束标志 + thread_id（前端收到 [DONE] 后关闭连接）
            yield "data: [DONE]\n\n"
            yield f"data: {json.dumps({'thread_id': thread_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        },
    )