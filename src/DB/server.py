import sys
import json
import os
import asyncio
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from DBAgent.agent import run

sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
sys.stderr.reconfigure(encoding="utf-8", errors="ignore")

sys.path.append(str(Path(__file__).parent.parent / "src"))

def log(msg: str):
    print(f"[sql-agent] {msg}", file=sys.stderr, flush=True)

log("server.py 启动中...")
log(f"Python: {sys.version}")
log(f"工作目录: {Path(__file__).parent}")

mcp = FastMCP("sql-agent")
log("FastMCP 初始化完成 ✅")


@mcp.tool()
def ping() -> str:
    """测试服务是否在线"""
    log("ping() 被调用")
    return "pong"


@mcp.tool()
def ask_db(question: str) -> str:
    """用自然语言查询数据库，返回 SQL 和查询结果"""
    log(f"ask_db() 收到问题: {question}")
    try:
        log("正在调用 run()...")
        result = run(question)
        log(f"run() 返回成功, SQL: {result.get('sql', '?')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"ask_db() 出错: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    if "--sse" in sys.argv or "--dev" in sys.argv:
        port = int(os.environ.get("PORT", "8001"))
        log(f"SSE 模式启动，监听 http://0.0.0.0:{port}")
        log(f"Inspector 连接地址: http://localhost:{port}/sse")

        # ✅ 手动加 CORS 中间件，解决 Inspector OPTIONS 405 问题
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
        log("STDIO 模式启动（Claude Desktop / MCP 客户端）")
        mcp.run(transport="stdio")