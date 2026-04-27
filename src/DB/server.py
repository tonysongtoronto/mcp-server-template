import sys
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from DBAgent.agent import run

sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
sys.stderr.reconfigure(encoding="utf-8", errors="ignore")

# ✅ 修路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

# =========================
# 🔧 调试专用 logger（写 stderr，不污染 MCP STDIO 协议）
# =========================
def log(msg: str):
    print(f"[sql-agent] {msg}", file=sys.stderr, flush=True)


# =========================
# 🚀 启动
# =========================
log("server.py 启动中...")
log(f"Python: {sys.version}")
log(f"工作目录: {Path(__file__).parent}")

mcp = FastMCP("sql-agent")
log("FastMCP 初始化完成 ✅")


# =========================
# 🛠 Tools
# =========================
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
        # ✅ FastMCP 必须返回字符串，dict 会导致卡住
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        log(f"ask_db() 出错: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# =========================
# 🏁 运行
# =========================
if __name__ == "__main__":
    log("mcp.run() 开始，等待连接...")
    mcp.run(transport="sse") 