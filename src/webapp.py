# src/webapp.py
#
# ★ 这是 langgraph dev 的唯一 lifespan 入口。
#   官方文档要求：lifespan 必须通过 FastAPI app 对象触发，
#   langgraph.json 里用 "http": {"app": "src/webapp.py:app"} 来注册。
#   直接在 langgraph.json 里写 "lifespan": "..." 是无效字段，langgraph 不认识。
#
# 运行方式：
#   uv run langgraph dev          → 本文件的 lifespan 被调用，MCP 初始化
#   uv run python src/langgraph_stdio_agent.py  → 走 __main__ 路径，本文件不参与

from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.langgraph_stdio_agent import (
    _mcp_manager,
    load_tools,
    _tools,
    _init_registry,
    _registry,
)
import sys


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔔 [webapp/lifespan] 启动中...")

    # 启动后台 MCP 连接（单个 server 失败只警告，不抛出）
    await _mcp_manager.start()

    all_tools = []

    # server.py 工具
    if "server" not in _mcp_manager._failed_servers:
        try:
            server_tools = await load_tools(session_name="server", use_manager=True)
            print(f"✅ [webapp/lifespan] server.py 工具：{[t.name for t in server_tools]}")
            all_tools.extend(server_tools)
        except Exception as exc:
            print(f"⚠️ [webapp/lifespan] server.py 工具加载失败：{exc}", file=sys.stderr)
    else:
        print("⚠️ [webapp/lifespan] server.py 未就绪，跳过", file=sys.stderr)

    # filesystem 工具（可选，失败降级）
    if "filesystem" not in _mcp_manager._failed_servers:
        try:
            fs_tools = await load_tools(session_name="filesystem", use_manager=True)
            print(f"✅ [webapp/lifespan] filesystem 工具：{[t.name for t in fs_tools]}")
            all_tools.extend(fs_tools)
        except Exception as exc:
            print(f"⚠️ [webapp/lifespan] filesystem 工具加载失败（降级运行）：{exc}", file=sys.stderr)
    else:
        print("⚠️ [webapp/lifespan] filesystem MCP 未就绪，跳过（降级运行）", file=sys.stderr)

    # ★ 无论哪些 server 失败，都必须调用 _init_registry
    #   否则 run_agent 里的 _get_registry_ready_event().wait() 永远不会被 set
    _tools.extend(all_tools)
    _init_registry(all_tools)
    print(f"🚀 [webapp/lifespan] 就绪，共 {len(all_tools)} 个工具，agents: {_registry.agents}")

    yield

    await _mcp_manager.stop()
    _tools.clear()
    print("🛑 [webapp/lifespan] 已关闭")


app = FastAPI(lifespan=lifespan)