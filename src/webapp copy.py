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
#   4. mcp-proxy (math-mcp)           → SSE @ 8004  数学工具（★ 新增）
#      └─ 底层：node src/math-mcp/build/index.js（stdio）
#              由 mcp-proxy 包成 SSE 对外暴露

import os
import signal
import socket
import subprocess
import sys
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
_SERVER_PORT     = int(os.getenv("MCP_SERVER_PORT",     "8001"))
_FS_PROXY_PORT   = int(os.getenv("MCP_FS_PROXY_PORT",   "8002"))
_DB_SERVER_PORT  = int(os.getenv("MCP_DB_SERVER_PORT",  "8003"))
_MATH_PROXY_PORT = int(os.getenv("MCP_MATH_PROXY_PORT", "8004"))   # ★ 新增

_MCP_FS_ENV = os.getenv("MCP_FS_BASE_DIR", "")
if _MCP_FS_ENV:
    _FS_BASE_DIR = Path(_MCP_FS_ENV)
else:
    _FS_BASE_DIR = Path(__file__).parent.parent / "File_Agent"

_SERVER_PY    = Path(__file__).parent / "mcp_server_template" / "server.py"
_DB_SERVER_PY = Path(__file__).parent / "mcp_db_server" / "server.py"

# ★ 新增：math-mcp Node.js 入口（src/math-mcp/build/index.js）
_MATH_MCP_JS  = Path(__file__).parent / "math-mcp" / "build" / "index.js"

_NPX = "npx.cmd" if sys.platform == "win32" else "npx"
_NODE = "node.exe" if sys.platform == "win32" else "node"


# ──────────────────────────────────────────
# 工具函数（与原版完全一致，不改动）
# ──────────────────────────────────────────

def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_port(port: int) -> None:
    print(f"  ⚠️  端口 {port} 被占用，尝试强制释放...", file=sys.stderr)
    if sys.platform == "win32":
        os.system(
            f'FOR /F "tokens=5" %P IN '
            f'(\'netstat -ano ^| findstr ":{port} "\') DO taskkill /F /PID %P >nul 2>&1'
        )
    else:
        ret = os.system(f"fuser -k {port}/tcp 2>/dev/null")
        if ret != 0:
            os.system(
                f"lsof -ti tcp:{port} 2>/dev/null | xargs kill -9 2>/dev/null || true"
            )
    time.sleep(0.8)


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


def _launch_subprocess(
    tag: str,
    cmd: list[str],
    env: dict | None = None,
    port: int | None = None,
) -> subprocess.Popen | None:
    if port and _port_in_use(port):
        _kill_port(port)
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

        proc = subprocess.Popen(cmd, **kwargs)
        print(f"  ✅ [{tag}] 进程已启动 PID={proc.pid}", file=sys.stderr)
        return proc

    except FileNotFoundError as e:
        print(f"  ❌ [{tag}] 命令未找到：{e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ❌ [{tag}] 启动失败：{type(e).__name__}: {e}", file=sys.stderr)
        return None


def _terminate_subprocess(tag: str, proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print(f"  🛑 [{tag}] 终止子进程 PID={proc.pid}", file=sys.stderr)
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception as e:
        print(f"  ⚠️  [{tag}] 终止时异常：{e}", file=sys.stderr)
        try:
            proc.kill()
        except Exception:
            pass


# ──────────────────────────────────────────
# 共享子进程列表
# ──────────────────────────────────────────
_subprocesses: list[subprocess.Popen] = []


# ──────────────────────────────────────────
# lifespan
# ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🟢 [lifespan] 开始启动 MCP 子进程...", file=sys.stderr)

    # ── 1. 启动 server.py（SSE @ 8001：data / http）──────────────────
    server_proc = _launch_subprocess(
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
        print(f"  ⏳ [server.py] 等待 SSE 就绪 http://127.0.0.1:{_SERVER_PORT}/sse ...",
              file=sys.stderr)
        ok = await _wait_for_sse(f"http://127.0.0.1:{_SERVER_PORT}/sse")
        print(f"  {'✅' if ok else '❌'} [server.py] SSE {'就绪' if ok else '超时（30s）'}",
              file=sys.stderr)

    # ── 2. 启动 mcp-proxy（SSE @ 8002：mcp-server-filesystem）───────
    _fs_sub_cmd = f"npx -y @modelcontextprotocol/server-filesystem {_FS_BASE_DIR}"
    fs_proc = _launch_subprocess(
        tag="mcp-proxy(fs)",
        cmd=[_NPX, "mcp-proxy",
             "--port", str(_FS_PROXY_PORT),
             "--server", "sse",
             "--shell",
             "--", _fs_sub_cmd],
        port=_FS_PROXY_PORT,
    )
    if fs_proc:
        _subprocesses.append(fs_proc)
        print(f"  ⏳ [mcp-proxy] 等待 SSE 就绪 http://127.0.0.1:{_FS_PROXY_PORT}/sse ...",
              file=sys.stderr)
        ok = await _wait_for_sse(f"http://127.0.0.1:{_FS_PROXY_PORT}/sse")
        print(f"  {'✅' if ok else '❌'} [mcp-proxy] SSE {'就绪' if ok else '超时（30s）'}",
              file=sys.stderr)

    # ── 3. 启动 db_server.py（SSE @ 8003：数据库工具）───────────────
    db_proc = _launch_subprocess(
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
        print(f"  ⏳ [db_server.py] 等待 SSE 就绪 http://127.0.0.1:{_DB_SERVER_PORT}/sse ...",
              file=sys.stderr)
        ok = await _wait_for_sse(f"http://127.0.0.1:{_DB_SERVER_PORT}/sse")
        print(f"  {'✅' if ok else '❌'} [db_server.py] SSE {'就绪' if ok else '超时（30s）'}",
              file=sys.stderr)

    # ── 4. ★ 新增：启动 mcp-proxy（SSE @ 8004：math-mcp Node.js）────
    #   底层命令：node src/math-mcp/build/index.js（stdio 模式）
    #   由 mcp-proxy 包成 SSE 对外暴露，保持与其他 MCP 服务一致的接入方式。
    _math_sub_cmd = f"{_NODE} {_MATH_MCP_JS}"
    math_proc = _launch_subprocess(
        tag="mcp-proxy(math)",
        cmd=[_NPX, "mcp-proxy",
             "--port", str(_MATH_PROXY_PORT),
             "--server", "sse",
             "--shell",
             "--", _math_sub_cmd],
        port=_MATH_PROXY_PORT,
    )
    if math_proc:
        _subprocesses.append(math_proc)
        print(f"  ⏳ [mcp-proxy(math)] 等待 SSE 就绪 http://127.0.0.1:{_MATH_PROXY_PORT}/sse ...",
              file=sys.stderr)
        ok = await _wait_for_sse(f"http://127.0.0.1:{_MATH_PROXY_PORT}/sse")
        print(f"  {'✅' if ok else '❌'} [mcp-proxy(math)] SSE {'就绪' if ok else '超时（30s）'}",
              file=sys.stderr)

    # ── 5. 初始化 MCP sessions（SSE 连接）────────────────────────────
    from src.langgraph_stdio_agent import _start_mcp_sessions
    await _start_mcp_sessions()
    print("🟢 [lifespan] 全部就绪，开始服务\n", file=sys.stderr)

    try:
        yield
    finally:
        # ── 6. 关闭 MCP sessions ─────────────────────────
        from src.langgraph_stdio_agent import _stop_mcp_sessions
        await _stop_mcp_sessions()

        # ── 7. 终止所有子进程 ────────────────────────────
        print("\n🛑 [lifespan] 关闭子进程...", file=sys.stderr)
        for proc in reversed(_subprocesses):
            _terminate_subprocess("subprocess", proc)
        _subprocesses.clear()
        print("🛑 [lifespan] 全部已关闭\n", file=sys.stderr)


# ──────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────
app = FastAPI(lifespan=lifespan)