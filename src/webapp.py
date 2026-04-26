# src/webapp.py
#
# langgraph dev 的 lifespan 入口。
# langgraph.json 配置：
#   "http": { "app": "./src/webapp.py:app" }
#
# ★ SSE 版本改造说明：
#   lifespan 启动时自动拉起两个独立 MCP 进程：
#     1. server.py          → SSE 模式，监听 localhost:8001
#     2. mcp-proxy          → 把 filesystem stdio 包装成 SSE，监听 localhost:8002
#   langgraph dev 关闭时自动终止这两个子进程。
#   前端测试只需一条命令：uv run langgraph dev
#
# ★ 端口冲突处理：
#   启动前检测 8001/8002 是否已被占用，自动 kill 占用进程后再启动，
#   防止上次测试异常退出后端口残留导致新进程启动失败。
#
# ★ Windows 兼容说明：
#   asyncio.create_subprocess_exec 在 Windows SelectorEventLoop（uvicorn 默认）下
#   会抛出 NotImplementedError，因此改用同步的 subprocess.Popen 启动子进程，
#   完全绕开 event loop 限制。等待 SSE 就绪仍用 asyncio + httpx。

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
_SERVER_PORT   = int(os.getenv("MCP_SERVER_PORT",   "8001"))
_FS_PROXY_PORT = int(os.getenv("MCP_FS_PROXY_PORT", "8002"))

# ★ 同样避免相对路径 + .resolve()，改用 __file__ 推导绝对路径
_MCP_FS_ENV = os.getenv("MCP_FS_BASE_DIR", "")
if _MCP_FS_ENV:
    _FS_BASE_DIR = Path(_MCP_FS_ENV)
else:
    _FS_BASE_DIR = Path(__file__).parent.parent / "File_Agent"

# server.py 的绝对路径（相对于本文件推导）
_SERVER_PY = Path(__file__).parent / "mcp_server_template" / "server.py"

# npx 在 Windows 下的命令名
_NPX = "npx.cmd" if sys.platform == "win32" else "npx"


# ──────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────

def _port_in_use(port: int) -> bool:
    """检查端口是否已被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_port(port: int) -> None:
    """
    强制释放被占用的端口（跨平台）。
    Windows 用 netstat + taskkill，Linux/macOS 用 fuser 或 lsof。
    """
    print(f"  ⚠️  端口 {port} 被占用，尝试强制释放...", file=sys.stderr)
    if sys.platform == "win32":
        os.system(
            f'FOR /F "tokens=5" %P IN '
            f'(\'netstat -ano ^| findstr ":{port} "\') DO taskkill /F /PID %P >nul 2>&1'
        )
    else:
        # fuser 优先，没有则 lsof
        ret = os.system(f"fuser -k {port}/tcp 2>/dev/null")
        if ret != 0:
            os.system(
                f"lsof -ti tcp:{port} 2>/dev/null | xargs kill -9 2>/dev/null || true"
            )
    time.sleep(0.8)  # 等端口释放


async def _wait_for_sse(url: str, timeout: float = 30.0, interval: float = 1.0) -> bool:
    """
    轮询 SSE 端点直到可连接（或超时）。
    SSE 是长连接，不能用普通 GET（会永远挂起）。
    改用 stream() + 只读第一个字节来验证端口已就绪。
    """
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
    """
    ★ 用同步 subprocess.Popen 启动子进程。
    Windows SelectorEventLoop（uvicorn 默认）下 asyncio.create_subprocess_exec
    会抛出 NotImplementedError，Popen 完全绕开此限制。
    返回 Popen 对象，失败返回 None。
    """
    # 端口冲突预处理
    if port and _port_in_use(port):
        _kill_port(port)
        if _port_in_use(port):
            print(f"  ❌ [{tag}] 端口 {port} 无法释放，跳过启动", file=sys.stderr)
            return None

    merged_env = {**os.environ, **(env or {})}
    print(f"  🚀 [{tag}] 启动：{' '.join(str(c) for c in cmd)}", file=sys.stderr)

    try:
        # Windows 下用 CREATE_NEW_PROCESS_GROUP 确保子进程可以独立终止
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
        print(f"       请确认命令存在且在 PATH 中：{cmd[0]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ❌ [{tag}] 启动失败：{type(e).__name__}: {e}", file=sys.stderr)
        return None


def _terminate_subprocess(tag: str, proc: subprocess.Popen) -> None:
    """同步终止子进程（兼容 Windows）"""
    if proc.poll() is not None:
        return  # 已经退出
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
    """
    FastAPI lifespan：
      启动时 → 用 Popen 拉起两个子进程（绕开 Windows SelectorEventLoop 限制）
             1. server.py          @ 8001  →  math / data / http 工具
             2. mcp-proxy          @ 8002  →  mcp-server-filesystem 文件系统工具
             → 用 httpx 异步等待两个 SSE 端点就绪
             → 初始化 MCP sessions（SSE 连接）
      关闭时 → 关闭 MCP sessions → 终止所有子进程

    ★ Windows mcp-proxy 命令说明：
      mcp-proxy 在 Windows 下直接 spawn npx 会 ENOENT，
      必须加 --shell 并把子命令包成单个字符串才能找到 npx.cmd。
      已验证命令：
        npx.cmd mcp-proxy --port 8002 --server sse --shell --
          "npx -y @modelcontextprotocol/server-filesystem <path>"
    """
    print("\n🟢 [lifespan] 开始启动 MCP 子进程...", file=sys.stderr)

    # ── 1. 启动 server.py（SSE @ 8001：math / data / http）──────────
    server_proc = _launch_subprocess(
        tag="server.py",
        cmd=[sys.executable, "-u", str(_SERVER_PY), "--sse"],
        env={"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
             "PORT": str(_SERVER_PORT),
             "MCP_FS_BASE_DIR": str(_FS_BASE_DIR)},
        port=_SERVER_PORT,
    )
    if server_proc:
        _subprocesses.append(server_proc)
        print(f"  ⏳ [server.py] 等待 SSE 就绪 http://127.0.0.1:{_SERVER_PORT}/sse ...",
              file=sys.stderr)
        ok = await _wait_for_sse(f"http://127.0.0.1:{_SERVER_PORT}/sse")
        print(f"  {'✅' if ok else '❌'} [server.py] SSE {'就绪' if ok else '超时（30s）'}",
              file=sys.stderr)

    # ── 2. 启动 mcp-proxy（SSE @ 8002：mcp-server-filesystem）────────
    # ★ Windows 关键：--shell + 子命令整体加引号，让 shell 去解析 npx.cmd。
    #   子命令必须是单个字符串，不能拆开，否则 mcp-proxy 会把路径当 npm 包名。
    _fs_sub_cmd = (
        f"npx -y @modelcontextprotocol/server-filesystem {_FS_BASE_DIR}"
    )
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

    # ── 3. 初始化 MCP sessions（SSE 连接）────────────────────────────
    from src.langgraph_stdio_agent import _start_mcp_sessions
    await _start_mcp_sessions()
    print("🟢 [lifespan] 全部就绪，开始服务\n", file=sys.stderr)

    try:
        yield
    finally:
        # ── 4. 关闭 MCP sessions ─────────────────────────
        from src.langgraph_stdio_agent import _stop_mcp_sessions
        await _stop_mcp_sessions()

        # ── 5. 终止所有子进程 ────────────────────────────
        print("\n🛑 [lifespan] 关闭子进程...", file=sys.stderr)
        for proc in reversed(_subprocesses):
            _terminate_subprocess("subprocess", proc)
        _subprocesses.clear()
        print("🛑 [lifespan] 全部已关闭\n", file=sys.stderr)


# ──────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────
app = FastAPI(lifespan=lifespan)