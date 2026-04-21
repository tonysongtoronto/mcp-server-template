# src/webapp.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
import sys
import os
from pathlib import Path

# 引用主模块的初始化函数
from src.langgraph_stdio_agent import (
    load_tools, _init_registry, _rebuild_graph,
    _tools, mcp_params
)
from mcp import ClientSession
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).parent / "mcp_server_template" / "server.py"

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not SERVER_PATH.exists():
        raise FileNotFoundError(f"找不到 MCP server：{SERVER_PATH}")
    async with stdio_client(mcp_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await load_tools(session)
            _tools.extend(loaded)
            _init_registry(loaded)
            _rebuild_graph()
            print("🚀 [lifespan] MCP + ToolRegistry 就绪")
            yield
    _tools.clear()
    print("🛑 [lifespan] MCP 已关闭")

app = FastAPI(lifespan=lifespan)