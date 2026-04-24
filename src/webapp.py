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

import asyncio
import os
import signal
import socket
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
_SERVER_PORT   = int(os.getenv("MCP_SERVER_PORT",     "8001"))
_FS_PROXY_PORT = int(os.getenv("MCP_FS_PROXY_PORT",   "8002"))
_FS_BASE_DIR   = Path(os.getenv("MCP_FS_BASE_DIR", "./File Agent")).resolve()

# server.py 的绝对路径（相对于本文件推导）
_SERVER_PY = Path(__file__).parent / "mcp_server_template" / "server.py"

# npx 在 Windows 下的命令名
_NPX = "npx.cmd" if sys.platform == "win32" else "npx"

# 子进程持有列表（lifespan 用）
_subprocesses: list[asyncio.subprocess.Process] = []


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


async def _wait_for_sse(url: str, timeout: float = 30.0, interval: float = 0.8) -> bool:
    """
    轮询 SSE 端点直到可连接（或超时）。
    用 httpx 发送 GET，不要求完整 SSE 握手，能建立连接即视为就绪。
    """
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=3.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url)
                # SSE 端点返回 200 或 405（Method Not Allowed）都算"活着"
                if resp.status_code in (200, 405):
                    return True
            except Exception:
                pass
            await asyncio.sleep(interval)
    return False


async def _launch_subprocess(
    tag: str,
    cmd: list[str],
    env: dict | None = None,
    port: int | None = None,
    sse_check_url: str | None = None,
) -> asyncio.subprocess.Process | None:
    """
    启动一个子进程，可选等待 SSE 端口就绪。
    返回 Process 对象，失败返回 None。
    """
    # 端口冲突预处理
    if port and _port_in_use(port):
        _kill_port(port)
        if _port_in_use(port):
            print(f"  ❌ [{tag}] 端口 {port} 无法释放，跳过启动", file=sys.stderr)
            return None

    merged_env = {**os.environ, **(env or {})}
    print(f"  🚀 [{tag}] 启动：{' '.join(cmd)}", file=sys.stderr)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
    except FileNotFoundError as e:
        print(f"  ❌ [{tag}] 命令未找到：{e}，请确认已安装", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ❌ [{tag}] 启动失败：{e}", file=sys.stderr)
        return None

    # 等待 SSE 端点就绪
    if sse_check_url:
        print(f"  ⏳ [{tag}] 等待 SSE 就绪 {sse_check_url} ...", file=sys.stderr)
        ok = await _wait_for_sse(sse_check_url)
        if ok:
            print(f"  ✅ [{tag}] SSE 就绪", file=sys.stderr)
        else:
            print(f"  ❌ [{tag}] SSE 超时（30s），可能启动失败", file=sys.stderr)
            proc.kill()
            return None

    return proc


async def _terminate_subprocess(tag: str, proc: asyncio.subprocess.Process) -> None:
    """优雅终止子进程"""
    if proc.returncode is not None:
        return  # 已经退出
    print(f"  🛑 [{tag}] 终止子进程 PID={proc.pid}", file=sys.stderr)
    try:
        if sys.platform == "win32":
            proc.kill()
        else:
            proc.send_signal(signal.SIGTERM)
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        proc.kill()
    except Exception as e:
        print(f"  ⚠️  [{tag}] 终止时异常：{e}", file=sys.stderr)


# ──────────────────────────────────────────
# lifespan
# ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan：
      启动时 → 拉起 server.py SSE + mcp-proxy SSE → 初始化 MCP sessions
      关闭时 → 终止子进程 → 关闭 MCP sessions
    """
    print("\n🟢 [lifespan] 开始启动 MCP 子进程...", file=sys.stderr)

    # ── 1. 启动 server.py（SSE 模式）────────────────────
    server_proc = await _launch_subprocess(
        tag="server.py",
        cmd=[sys.executable, "-u", str(_SERVER_PY), "--sse"],
        env={"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
             "PORT": str(_SERVER_PORT)},
        port=_SERVER_PORT,
        sse_check_url=f"http://127.0.0.1:{_SERVER_PORT}/sse",
    )
    if server_proc:
        _subprocesses.append(server_proc)

    # ── 2. 启动 mcp-proxy（filesystem stdio → SSE）───────
    if not _FS_BASE_DIR.exists():
        print(f"  ⚠️  [mcp-proxy] BASE_DIR 不存在：{_FS_BASE_DIR}，跳过 filesystem MCP",
              file=sys.stderr)
        fs_proc = None
    else:
        fs_proc = await _launch_subprocess(
            tag="mcp-proxy",
            cmd=[
                _NPX, "-y", "mcp-proxy",
                "--port", str(_FS_PROXY_PORT),
                "--",
                _NPX, "-y", "@modelcontextprotocol/server-filesystem",
                str(_FS_BASE_DIR),
            ],
            port=_FS_PROXY_PORT,
            sse_check_url=f"http://127.0.0.1:{_FS_PROXY_PORT}/sse",
        )
        if fs_proc:
            _subprocesses.append(fs_proc)

    # ── 3. 初始化 LangGraph agent 的 MCP sessions（SSE 连接）
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
            await _terminate_subprocess("subprocess", proc)
        _subprocesses.clear()
        print("🛑 [lifespan] 全部已关闭\n", file=sys.stderr)


# ──────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────
app = FastAPI(lifespan=lifespan)