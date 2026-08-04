# src/webapp.py
#
# langgraph dev 的 lifespan 入口。
# langgraph.json 配置：
#   "http": { "app": "./src/webapp.py:app" }
#
# ★ 子进程清单（lifespan 负责拉起 + 关闭）：
#   1. mcp_server_template/server.py  → Streamable HTTP @ 8001  data / http 工具
#   2. mcp-proxy (filesystem)         → SSE              @ 8002  文件系统工具
#   3. mcp_db_server/server.py        → Streamable HTTP  @ 8003  数据库工具
#   4. mcp-proxy (math-mcp)           → SSE              @ 8004  数学工具
#
# ★ 阶段二新增：
#   - AsyncSqliteSaver 替换 MemorySaver，对话历史持久化到 SQLite
#   - POST /chat/stream  SSE 流式端点，逐 token 推送 AI 回复
#   - thread_id 从请求头 X-Thread-Id 读取，缺省自动生成 UUID

import asyncio
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
from fastapi.responses import StreamingResponse, JSONResponse

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


async def _wait_for_http(url: str, timeout: float = 40.0, interval: float = 1.0) -> bool:
    """等待 HTTP 端点可用（GET 或 POST 均接受，适用于 Streamable HTTP /mcp）"""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                # 用 stream() 而不是 get()：只等响应头到达就返回，
                # 不会卡在读取 SSE 永不结束的响应体上。
                async with client.stream("GET", url) as resp:
                    if resp.status_code in (200, 405, 406):
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

    import src.langgraph_parallel_agent as agent_module

    # ── 持久化：复用 agent_module 自己的 SQLite 连接，不再重复开一组 ──────
    #
    # ★ 重构说明（原方案的问题）：
    #   旧代码在这里自己 `async with AsyncSqliteSaver/AsyncSqliteStore` 开了
    #   webapp 专属的一组连接（saver/store），跟 agent_module._open_sqlite_backends()
    #   打开的模块级 _checkpointer/_store 是两条独立的 connection，只是连到
    #   同一份 .db 文件。这带来两个问题：
    #     1. 同一份文件被两条独立 connection 同时打开，本身就多一份心智负担。
    #     2. 更麻烦的是：_ensure_registry() 懒初始化在请求期间可能重新调用
    #        _init_registry() → build_graph(_checkpointer, _store)，会把
    #        graph 悄悄换回 agent_module 那组连接，且没有任何日志提示，
    #        属于运行时的隐性风险。
    #
    # ★ 新方案：不在 webapp.py 里另开一组连接，直接把
    #   agent_module._open_sqlite_backends() 打开的那一组，原样挂到
    #   app.state 上。整个进程从头到尾只有一份 checkpointer、一份 store，
    #   不管是 _init_registry() 内部触发的重建，还是 webapp 这边的引用，
    #   拿到的永远是同一个对象——不存在"悄悄换成另一条连接"这种可能性，
    #   因为压根不存在"另一条连接"。
    #
    # 为什么不用 MemorySaver / InMemoryStore？
    #   进程重启后历史会清空，生产环境不可用；SQLite 落盘才能保证重启后
    #   对话历史和全局记忆都还在。
    #
    # 注意：agent_module.py 本身完全没有改动，_open_sqlite_backends() /
    #   _checkpointer_cm / _store_cm 这些接口原样保留，CLI / api.py 模式
    #   依然可以独立运行，不依赖 webapp.py。
    await agent_module._open_sqlite_backends()
    app.state.checkpointer = agent_module._checkpointer
    app.state.store = agent_module._store
    print(f"  💾 [Checkpoint] SQLite 路径：{agent_module._CHECKPOINT_DB}", file=sys.stderr)
    print(f"  🗄️  [MemoryStore] SQLite 路径：{agent_module._STORE_DB}", file=sys.stderr)
    print("  ✅ [Persistence] AsyncSqliteSaver + AsyncSqliteStore 就绪（唯一一组连接）",
          file=sys.stderr)

    # ── 1. 启动 server.py（Streamable HTTP @ 8001）──────────────────────────────
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
        ok = await _wait_for_http(f"http://127.0.0.1:{_SERVER_PORT}/mcp")
        print(f"  {'✅' if ok else '❌'} [server.py] HTTP {'就绪' if ok else '超时'}",
              file=sys.stderr)

    # ── 2. 启动 mcp-proxy（Streamable HTTP @ 8002：filesystem）──────────────────
    # ★ 超时修复：npx 首次运行需要从 npm 下载包，网络慢时耗时长。
    # pre-warm：先让 npx 把包下载到本地缓存，再启动 mcp-proxy 时就走缓存秒起。
    print("  ⏳ [mcp-proxy(fs)] 预热 npm 包（首次约 30-60s）...", file=sys.stderr)
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["npx", "--yes", "@modelcontextprotocol/server-filesystem", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
        )
    except Exception:
        pass  # pre-warm 失败不阻断，继续尝试启动

    fs_proc = await _launch_subprocess(
        tag="mcp-proxy(fs)",
        cmd=["mcp-proxy", "--port", str(_FS_PROXY_PORT), "--",
             "npx", "--yes", "@modelcontextprotocol/server-filesystem", str(_FS_BASE_DIR)],
        port=_FS_PROXY_PORT,
    )
    if fs_proc:
        _subprocesses.append(fs_proc)
        # timeout 提高到 90s，覆盖网络慢的场景
        ok = await _wait_for_http(f"http://127.0.0.1:{_FS_PROXY_PORT}/sse",
                                  timeout=90.0)
        print(f"  {'✅' if ok else '❌'} [mcp-proxy(fs)] SSE {'就绪' if ok else '超时'}",
              file=sys.stderr)

    # ── 3. 启动 db_server.py（Streamable HTTP @ 8003）───────────────────────────
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
        ok = await _wait_for_http(f"http://127.0.0.1:{_DB_SERVER_PORT}/mcp")
        print(f"  {'✅' if ok else '❌'} [db_server.py] HTTP {'就绪' if ok else '超时'}",
              file=sys.stderr)

    # ── 4. 启动 mcp-proxy（Streamable HTTP @ 8004：math-mcp）────────────────────
    math_proc = await _launch_subprocess(
        tag="mcp-proxy(math)",
        cmd=["mcp-proxy", "--port", str(_MATH_PROXY_PORT), "--",
             _NODE, str(_MATH_MCP_JS)],
        port=_MATH_PROXY_PORT,
    )
    if math_proc:
        _subprocesses.append(math_proc)
        ok = await _wait_for_http(f"http://127.0.0.1:{_MATH_PROXY_PORT}/sse",
                                  timeout=90.0)
        print(f"  {'✅' if ok else '❌'} [mcp-proxy(math)] SSE {'就绪' if ok else '超时'}",
              file=sys.stderr)

    # ── 5. 初始化 MCP sessions ───────────────────────────────────────
    #
    # _start_mcp_sessions() 内部会调 _init_registry() → build_graph(_checkpointer, _store)。
    # 因为 _open_sqlite_backends() 已经在最前面调用过，_checkpointer/_store
    # 此刻就是"唯一的那组"持久化连接——这一次 build_graph() 编译出来的
    # graph 已经是最终版本，不再需要像旧方案那样等这一步完成后再用
    # 另一组连接重建一次。
    await agent_module._start_mcp_sessions()
    print("  ✅ [Graph] 已用 AsyncSqliteSaver + AsyncSqliteStore 编译就绪", file=sys.stderr)
    print("🟢 [lifespan] 全部就绪，开始服务\n", file=sys.stderr)

    try:
        yield

    finally:
        await agent_module._stop_mcp_sessions()
        print("\n🛑 [lifespan] 关闭子进程...", file=sys.stderr)
        for proc in reversed(_subprocesses):
            await _terminate_subprocess("subprocess", proc)
        _subprocesses.clear()

        # ── 关闭持久化连接（唯一的一组，agent_module._checkpointer_cm/_store_cm）──
        #
        # 因为整个进程只开过这一组连接，这里就是全部需要关闭的资源，
        # 不再有"webapp 自己那组"和"agent_module 那组"的区分。
        # 关闭顺序与打开顺序相反：先 store 后 checkpointer。
        # 用 getattr 兜底：如果启动阶段提前异常退出，这两个属性可能还是 None。
        store_cm = getattr(agent_module, "_store_cm", None)
        checkpointer_cm = getattr(agent_module, "_checkpointer_cm", None)
        try:
            if store_cm is not None:
                await store_cm.__aexit__(None, None, None)
            if checkpointer_cm is not None:
                await checkpointer_cm.__aexit__(None, None, None)
            print("🛑 [lifespan] SQLite 连接已关闭（checkpoints.db + memory_store.db）",
                  file=sys.stderr)
        except Exception as e:
            # 关闭失败不应该阻塞其余的优雅退出流程，记录日志即可
            print(f"⚠️ [lifespan] SQLite 关闭异常（已忽略）：{e}", file=sys.stderr)

        print("🛑 [lifespan] 全部已关闭\n", file=sys.stderr)


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

    # ── 创建流式队列，注入到 agent 模块 ──────────────────────────────────
    # ★ 修复1：用 request_id 作 key 存入 dict（_stream_queues），而非单一变量
    #   final_answer_node 通过 config["configurable"]["_stream_request_id"] 查找对应 queue；
    #   之前用 _stream_queue（单数）存，节点查 _stream_queues[key] 永远找不到，
    #   导致 queues=[] → 不写哨兵 → _generate() 永久阻塞。
    request_id = f"{thread_id}_{uuid.uuid4().hex[:8]}"

    config = {"configurable": {"thread_id": thread_id, "_stream_request_id": request_id}}
    queue: asyncio.Queue = asyncio.Queue()
    agent_module._stream_queues[request_id] = queue

    async def _generate():
        # ★ 修复2：_stream_request_id 走 config["configurable"]，节点才能找到自己的 queue
        invoke_task = asyncio.create_task(
            agent_module.graph.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )
        )

        try:
            while True:
                # ★ 修复3：加 120s 超时兜底，防止节点意外不写哨兵时永久卡死
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    yield "data: [ERROR] 等待超时（120s）\n\n"
                    invoke_task.cancel()
                    break

                if token is None:
                    # 哨兵：final_answer_node 生成完毕
                    break

                # SSE 帧内不能有裸换行，转义后前端再还原
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"

            # 等待 invoke 完全结束（含摘要更新等后续操作）
            try:
                await invoke_task
            except asyncio.CancelledError:
                # ★ 修复5：CancelledError 继承自 BaseException，不是 Exception。
                # 这里的取消是我们自己在上面 120s 超时分支主动 invoke_task.cancel()
                # 触发的，属于预期内的收尾动作。注意：那个分支在调用 cancel() 之前
                # 已经 yield 过一条 "等待超时" 的 ERROR 消息给前端了，所以这里不用
                # 再重复 yield 一次，直接 pass 即可，避免前端收到两条重复的 ERROR。
                #
                # 但为了防止未来有人在别处新增 invoke_task.cancel() 调用、却忘了
                # 提前 yield 消息告知前端（导致前端静默卡住、排查困难），这里保留一条
                # 服务端日志作为兜底：不发给前端，但至少能在服务端日志里看到痕迹。
                print(f"ℹ️  [/chat/stream] invoke_task 已按预期被取消（request_id={request_id}）",
                      file=sys.stderr)
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"

        except asyncio.CancelledError:
            # ★ 修复5（外层，注意和内层 416 行语义不同）：
            # 这里捕获到的 CancelledError，大概率意味着 _generate() 这个
            # 协程本身被取消了（例如客户端中途断开连接，Starlette/uvicorn
            # 会取消驱动这个异步生成器的 Task），而不是我们主动取消
            # invoke_task 那种"预期内"的情况。
            #
            # 按 asyncio 的协作取消约定：捕获到"自己被取消"的 CancelledError
            # 后不应该吞掉，而是做完必要清理后重新 raise，把取消信号正确
            # 传递回调用方（driving 这个生成器的 Task）。直接 pass 掉会让
            # 上层误以为这个 Task 正常结束、并没有真的响应取消请求。
            print("⚠️  [/chat/stream] 连接被取消（客户端断开或上游取消），清理后重新抛出",
                  file=sys.stderr)
            if not invoke_task.done():
                invoke_task.cancel()
            raise

        except Exception as e:
            print(f"❌ [/chat/stream] 出错：{e}", file=sys.stderr)
            yield f"data: [ERROR] {str(e)}\n\n"
            if not invoke_task.done():
                invoke_task.cancel()

        finally:
            # ★ 修复4：从 dict 里移除，不影响其他并发请求
            agent_module._stream_queues.pop(request_id, None)

            # [DONE:thread_id] 格式与 api.py / client.js 保持一致
            # 前端收到后直接解析 threadId，无需再读额外 JSON 帧
            yield f"data: [DONE:{thread_id}]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────
# Memory Store 管理端点（实验 & 运维用）
# ──────────────────────────────────────────
#
# 这三个端点让你不用重启服务就能动态管理全局记忆。
# 典型用途：
#   预置业务规则   → PUT /memory/put  {"key":"discount","value":"所有用户享受10%折扣"}
#   查看当前记忆   → GET /memory/list
#   删除过期记忆   → DELETE /memory/delete?key=discount
#
# 默认使用 ("system",) 命名空间（全体用户可见）。
# 如需用户级命名空间，传 namespace 参数：?namespace=user:u001

def _parse_namespace(ns_str: str | None) -> tuple:
    """把 'system' 或 'user:uid123' 这样的字符串转成 tuple('system',) 或 ('user','uid123')"""
    if not ns_str or ns_str == "system":
        return ("system",)
    parts = ns_str.split(":", 1)
    return tuple(parts)


@app.post("/memory/put")
async def memory_put(request: Request) -> JSONResponse:
    """
    写入一条全局记忆。
    Body: {"key": "city", "value": "Toronto", "namespace": "system"}
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        return JSONResponse({"error": "Memory Store 未初始化"}, status_code=503)

    body = await request.json()
    key       = body.get("key", "").strip()
    value     = body.get("value")
    namespace = _parse_namespace(body.get("namespace", "system"))

    if not key:
        return JSONResponse({"error": "key 不能为空"}, status_code=400)
    if value is None:
        return JSONResponse({"error": "value 不能为 null"}, status_code=400)

    # value 若是字符串，包装成 dict 存储（Store 要求 value 为 dict）
    stored_val = value if isinstance(value, dict) else {"value": value}

    try:
        await store.aput(namespace, key, stored_val)
        print(f"  💾 [/memory/put] {namespace}/{key} = {str(value)[:60]}", file=sys.stderr)
        return JSONResponse({"ok": True, "namespace": list(namespace), "key": key})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/memory/list")
async def memory_list(request: Request, namespace: str = "system") -> JSONResponse:
    """
    列出命名空间下所有全局记忆。
    Query: ?namespace=system（默认）
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        return JSONResponse({"error": "Memory Store 未初始化"}, status_code=503)

    ns = _parse_namespace(namespace)
    try:
        results = await store.asearch(ns)
        items   = {r.key: r.value for r in results}
        return JSONResponse({"namespace": list(ns), "count": len(items), "items": items})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/memory/delete")
async def memory_delete(request: Request, key: str, namespace: str = "system") -> JSONResponse:
    """
    删除一条全局记忆。
    Query: ?key=city&namespace=system
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        return JSONResponse({"error": "Memory Store 未初始化"}, status_code=503)

    ns = _parse_namespace(namespace)
    try:
        await store.adelete(ns, key)
        print(f"  🗑️  [/memory/delete] {ns}/{key}", file=sys.stderr)
        return JSONResponse({"ok": True, "namespace": list(ns), "key": key})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
    # npx @modelcontextprotocol/inspector npx -y @modelcontextprotocol/server-filesystem "C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/File_Agent"
    
    # C:\Users\tonysong\Desktop\AI_Python\mcp-server-template\File_Agent\memory.txt
    
    # npx -y supergateway --port 8002 --stdio 'npx -y @modelcontextprotocol/server-filesystem "C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/File_Agent"'
    
    # npx @modelcontextprotocol/inspector
    
    # http://localhost:8002/sse
    
    # npx -y supergateway --port 8004 --stdio 'node "C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/src/math-mcp/build/index.js"'
    # npx @modelcontextprotocol/inspector
    
    # http://localhost:8004/sse
    
    # --stdio 
    
    # npx @modelcontextprotocol/inspector node "C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/src/math-mcp/build/index.js"
    
    #     方案A：最简单，单独测 filesystem，不启动你的 webapp

    # npx @modelcontextprotocol/inspector npx -y @modelcontextprotocol/server-filesystem "C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/File_Agent"

    # 它会自动打开浏览器 http://127.0.0.1:6274，左边不要改成SSE，要保持：
    # Transport Type: STDIO
    # Command: npx
    # Args: -y @modelcontextprotocol/server-filesystem C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/File_Agent
    # 点最下面的 Connect，连上后右上角会变绿 Connected。

    # 测试：

    # path* C:\Users\tonysong\Desktop\AI_Python\mcp-server-template\File_Agent\memory_tony.txt

    # content*  inputinfo

    # Read Text File：

    # path*
    # C:\Users\tonysong\Desktop\AI_Python\mcp-server-template\File_Agent\memory_tony.txt

    # 方案 B sse

    #  npx -y supergateway --port 8002 --stdio 'npx -y @modelcontextprotocol/server-filesystem "C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/File_Agent"'
    
    # 另一个终端 运行 
    #     # npx @modelcontextprotocol/inspector
    #     UI Transport Type sse
    # 	URL:
    #     # http://localhost:8002/sse
        
    # 	connect
    
    #     math-mcp  mcp inspector 测试：

    # --stdio 
        
    # npx @modelcontextprotocol/inspector node "C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/src/math-mcp/build/index.js"

    # --sse 

    # npx -y supergateway --port 8004 --stdio 'node "C:/Users/tonysong/Desktop/AI_Python/mcp-server-template/src/math-mcp/build/index.js"'

    # 另一个窗口

    # npx @modelcontextprotocol/inspector

    # Transport Type sse

    # URL  http://localhost:8004/sse

    # Connection Type Via Proxy
    
    # uv run langgraph dev