"""
src/mcp_db_server/server.py

DB MCP Server — 电商数据库查询服务
支持两种传输模式：
  --sse / --dev   → SSE 模式，监听 http://0.0.0.0:8003
  (默认)          → STDIO 模式（Claude Desktop / MCP 客户端 / __main__ 测试）

暴露的 MCP 工具：
  ping()                        → 健康检查
  get_schema()                  → 返回数据库表结构
  ask_db(question)              → 自然语言 → SQL → 执行 → 结果
  query_db(sql)                 → 直接执行 SELECT（带安全检查）
  execute_db(sql)               → 直接执行 INSERT/UPDATE（带安全检查）
"""

import sys
import json
import os
import asyncio
from pathlib import Path

# ── 路径修复：确保 src/ 在 sys.path 里 ───────────────────────────────
_SRC_DIR = Path(__file__).parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
sys.stderr.reconfigure(encoding="utf-8", errors="ignore")


def _log(msg: str):
    print(f"[db-mcp] {msg}", file=sys.stderr, flush=True)


_log("server.py starting...")
_log(f"Python: {sys.version}")
_log(f"src dir: {_SRC_DIR}")

# ── 确保数据库已初始化 ────────────────────────────────────────────────
try:
    from DB.init_db import init_db, DB_PATH
    if not DB_PATH.exists():
        _log("DB not found, initializing...")
        init_db()
    else:
        _log(f"DB found at: {DB_PATH}")
except Exception as e:
    _log(f"DB init error: {e}")

# ── FastMCP 实例 ──────────────────────────────────────────────────────
mcp = FastMCP("db-agent")
_log("FastMCP initialized ✅")


# ════════════════════════════════════════════════════════
# MCP Tools
# ════════════════════════════════════════════════════════

@mcp.tool()
def ping() -> str:
    """健康检查，返回 pong"""
    _log("ping() called")
    return "pong"


@mcp.tool()
def get_schema() -> str:
    """
    返回数据库完整表结构（供 LangGraph Agent 了解数据库）。
    返回人类可读的文本格式，包含所有表名、列名、类型、主键、外键和行数。
    """
    _log("get_schema() called")
    try:
        from DB.schema import get_schema_text
        return get_schema_text()
    except Exception as e:
        _log(f"get_schema error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def ask_db(question: str) -> str:
    """
    自然语言查询数据库。
    输入：用中文或英文描述你的查询需求，例如 "查询所有来自 Toronto 的用户"
    输出：JSON 字符串，包含 sql、action、result（字典列表）、error 字段
    适合：不确定 SQL 语法时使用，由 AI 自动生成并执行 SQL
    """
    _log(f"ask_db() question: {question}")
    try:
        from DBAgent.agent import run
        result = run(question)
        _log(f"ask_db OK, sql: {result.get('sql', '?')[:80]}")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        _log(f"ask_db error: {e}")
        return json.dumps({"error": str(e), "sql": None, "result": None}, ensure_ascii=False)


@mcp.tool()
def query_db(sql: str) -> str:
    """
    直接执行 SELECT 查询（带安全检查和自动 LIMIT）。
    输入：合法的 SQLite SELECT 语句
    输出：JSON 字符串，包含 sql、result（字典列表）、error 字段
    适合：已知 SQL 语句时直接执行，比 ask_db 更快（无需 LLM 生成 SQL）
    注意：自动添加 LIMIT 100 防止大结果集；禁止 DROP/TRUNCATE/ALTER 等危险语句
    """
    _log(f"query_db() sql: {sql[:80]}")
    try:
        from DBAgent.optimizer import SQLOptimizer
        from DBAgent.tools import query_db as _query
        optimizer = SQLOptimizer()
        optimized = optimizer.optimize(sql)
        result = _query(optimized.sql)
        return json.dumps({
            "sql": optimized.sql,
            "result": result,
            "error": None
        }, ensure_ascii=False)
    except ValueError as e:
        _log(f"query_db blocked: {e}")
        return json.dumps({"sql": sql, "result": None, "error": str(e)}, ensure_ascii=False)
    except Exception as e:
        _log(f"query_db error: {e}")
        return json.dumps({"sql": sql, "result": None, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
def execute_db(sql: str) -> str:
    """
    直接执行 INSERT 或 UPDATE 语句（带安全检查）。
    输入：合法的 SQLite INSERT 或 UPDATE 语句
    输出：JSON 字符串，包含 sql、rows_affected、error 字段
    注意：UPDATE 必须带 WHERE 子句；禁止 DROP/TRUNCATE/ALTER 等危险语句
    """
    _log(f"execute_db() sql: {sql[:80]}")
    try:
        from DBAgent.optimizer import SQLOptimizer
        from DBAgent.tools import execute_db as _execute
        optimizer = SQLOptimizer()
        optimized = optimizer.optimize(sql)
        result = _execute(optimized.sql)
        return json.dumps({
            "sql": optimized.sql,
            "rows_affected": result.get("rows_affected", 0),
            "error": None
        }, ensure_ascii=False)
    except ValueError as e:
        _log(f"execute_db blocked: {e}")
        return json.dumps({"sql": sql, "rows_affected": 0, "error": str(e)}, ensure_ascii=False)
    except Exception as e:
        _log(f"execute_db error: {e}")
        return json.dumps({"sql": sql, "rows_affected": 0, "error": str(e)}, ensure_ascii=False)


# ════════════════════════════════════════════════════════
# 启动入口
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    if "--sse" in sys.argv or "--dev" in sys.argv:
        port = int(os.environ.get("PORT", "8003"))
        _log(f"SSE mode, listening on http://0.0.0.0:{port}")
        _log(f"SSE endpoint: http://localhost:{port}/sse")

        from starlette.middleware.cors import CORSMiddleware
        import uvicorn

        app = mcp.sse_app()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        uvicorn.run(app, host="0.0.0.0", port=port)

    else:
        _log("STDIO mode (Claude Desktop / MCP client)")
        mcp.run(transport="stdio")